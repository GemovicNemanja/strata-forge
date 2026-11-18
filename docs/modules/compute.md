# `strata_forge.compute` — task shapes, backends, batch inference, serving adapters

`strata_forge.compute` is the remote-compute orchestration layer. It
ships typed task / job / status Pydantic shapes, a
:class:`Backend` Protocol, three concrete backends
(:class:`LocalBackend`, :class:`SSHBackend`,
:class:`SkyPilotBackend`), a concurrency-bounded
:class:`BatchInferenceRunner` for fan-out over a shared
:class:`LLMClient`, and small task-builder functions for
self-hosted inference servers (vLLM, TGI, SGLang). See
[ADR 0013](../architecture/adr/0013-compute-task-and-backend-shapes.md)
for the task-as-data + Protocol design rationale, and
[ADR 0019](../architecture/adr/0019-secrets-travel-beside-the-task.md)
for how a job's credentials reach it.

Integration points:

- **Tasks:** :class:`Task`, :class:`ResourceSpec`. Pure data
  shapes that serialize to / from YAML (SkyPilot-style).
- **Jobs:** :class:`Job`, :class:`JobStatus`, :data:`JobState`.
  Five canonical states: ``pending`` / ``running`` /
  ``succeeded`` / ``failed`` / ``cancelled``.
- **Protocol:** :class:`Backend` — async ``submit`` / ``status``
  / ``logs`` / ``cancel`` / ``cleanup``.
- **Backends:** :class:`LocalBackend` (in-process subprocess),
  :class:`SSHBackend` (asyncssh, lazy ``[compute]`` extra),
  :class:`SkyPilotBackend` (sky.api.sdk, lazy ``[compute]``
  extra).
- **Batch inference:** :class:`BatchInferenceRunner`,
  :class:`BatchInferenceResult` — concurrency-capped async
  fan-out over :class:`LLMClient`.
- **Serving adapters:** :func:`build_vllm_task`,
  :func:`build_tgi_task`, :func:`build_sglang_task`,
  :func:`serving_endpoint`, :func:`wait_for_endpoint`,
  :class:`ServingEndpoint`.

Module rules: [`src/strata_forge/compute/CLAUDE.md`](../../src/strata_forge/compute/CLAUDE.md).
Source: [`src/strata_forge/compute/`](../../src/strata_forge/compute/).

---

## Contents

- [Quickstart](#quickstart)
- [Task and ResourceSpec](#task-and-resourcespec)
  - [Secrets](#secrets)
- [Job lifecycle](#job-lifecycle)
  - [Evaluation runner](#evaluation-runner)
  - [The runner contract](#the-runner-contract)
- [Backend protocol](#backend-protocol)
  - [Cleanup is verified](#cleanup-is-verified)
- [LocalBackend](#localbackend)
- [SSHBackend](#sshbackend)
- [SkyPilotBackend](#skypilotbackend)
- [Batch inference](#batch-inference)
- [Serving adapters](#serving-adapters)
- [Lazy-import contract](#lazy-import-contract)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```python
from strata_forge.compute import LocalBackend, Task

backend = LocalBackend()
task = Task(name="hello", run="echo 'hi from compute'")
job = await backend.submit(task)

status = await backend.status(job)
print(status.state)  # → "running" or "succeeded"

logs = await backend.logs(job)
await backend.cleanup(job)
```

For batch inference:

```python
from strata_forge.compute import BatchInferenceRunner
from strata_forge.llm import LLMClient, Message

client = LLMClient(model="claude-haiku-4-5")
runner = BatchInferenceRunner(client, concurrency=10, on_error="collect")
results = await runner.run([
    [Message.user(prompt)] for prompt in big_prompt_list
])
for r in results:
    if r.succeeded:
        print(r.response.text)
    else:
        print(f"failed: {r.error}")
```

For a self-hosted vLLM server:

```python
from strata_forge.compute import LocalBackend
from strata_forge.compute.serving import build_vllm_task, serving_endpoint
from strata_forge.llm import LLMClient
from strata_forge.llm.providers.config import OpenAICompatConfig

task = build_vllm_task("meta-llama/Llama-3.1-8B-Instruct", port=8000)
async with serving_endpoint(
    LocalBackend(), task, base_url="http://localhost:8000/v1"
) as endpoint:
    client = LLMClient(
        model="meta-llama/Llama-3.1-8B-Instruct",
        provider="openai_compat",
        provider_config=OpenAICompatConfig(base_url=endpoint.base_url),
    )
    response = await client.complete(messages=[Message.user("Hello!")])
```

---

## Task and ResourceSpec

A :class:`Task` is a frozen Pydantic instance describing what to
run and on what kind of hardware. It maps directly to the
SkyPilot YAML task shape so users with existing YAML pipelines
can adopt it incrementally.

```python
from strata_forge.compute import Task, ResourceSpec

task = Task(
    name="train-llama",
    setup="pip install torch transformers trl",
    run="python train.py --epochs 3",
    env={"WANDB_PROJECT": "forge-experiments"},
    resources=ResourceSpec(accelerators="A100:8", cpus=64, memory_gb=256),
)

# YAML round-trips.
yaml_text = task.to_yaml()
roundtripped = Task.from_yaml_str(yaml_text)
```

`ResourceSpec.accelerators` follows the SkyPilot convention:
``"A100:8"``, ``"H100:4"``, etc. The local backend ignores
``resources`` entirely; SSH inherits whatever the host has;
SkyPilot forwards it verbatim.

### Secrets

A credential the job needs goes in ``secrets``, never in ``env``:

```python
from pydantic import SecretStr

task = Task(
    name="push-results",
    run="python -m my_pipeline",
    env={"WANDB_PROJECT": "forge-experiments"},
    secrets={"HF_TOKEN": SecretStr(token)},
)
```

``Task.secrets`` travels beside the task rather than in it:

- It is ``dict[str, SecretStr]``, excluded from ``model_dump`` /
  ``model_dump_json`` / ``to_yaml`` and from ``repr``, and never
  read from YAML. The values are masked before any validator runs,
  so no rendering of a validation error (``str``, ``errors()``,
  ``json()``) carries one. A pickled ``Task`` does carry them: never
  pickle a task with secrets.
- Keys look like environment variable names
  (``^[A-Z][A-Z0-9_]{0,63}$``); at most 8; values non-empty; a key
  may not also appear in ``env``. ``env`` may not set
  ``FORGE_SECRETS_FILE``. Set secrets through ``Task(...)`` or
  ``Task.model_validate``: ``model_copy(update=...)`` skips the
  validators, so every backend rebuilds the task through them
  (``Task.revalidated()``) before it delivers anything.
- The backend writes them as one JSON object to a 0600
  ``.secrets.json`` in a 0700 directory and exports
  ``FORGE_SECRETS_FILE`` (``strata_forge.compute.SECRETS_FILE_ENV``),
  its absolute path, to the task's ``run`` step only; ``setup`` does
  not see it. That path is the only secret-related thing in the
  job's environment. An outer shell removes the file on exit, setup
  failure and SIGTERM included, while ``setup`` and ``run`` execute
  in a subshell below it, so a ``trap ... EXIT`` the task sets
  cannot displace that removal.
- A backend with no private channel refuses a task with secrets
  rather than falling back to the environment
  (:class:`SkyPilotBackend` raises ``ValueError``).

A job reads the file once and deletes it; the pipeline runners do
that before anything else (see [the runner contract](#the-runner-contract)).

## Job lifecycle

Backends return :class:`Job` handles from ``submit``. A
:class:`JobStatus` carries a canonical :data:`JobState`:

| State | Meaning |
|---|---|
| ``pending`` | Queued / setting up; not yet running. |
| ``running`` | Actively executing. |
| ``succeeded`` | Exit code 0. |
| ``failed`` | Non-zero exit, infrastructure error, unparseable status, or a process that vanished without recording an exit code (an OOM kill, a reboot). |
| ``cancelled`` | Killed **by an explicit** :meth:`cancel`. |

The five-state set is intentionally narrow — backend-specific
nuance (SkyPilot's ``SETTING_UP``, SSH's process-gone-no-exit-file)
lands in :attr:`JobStatus.message` rather than expanding the state
machine.

``cancelled`` is reserved for a death somebody **asked for**. A cancelled
process and one the machine killed look identical afterwards — pid gone,
no exit code — so ``SSHBackend.cancel`` records a marker in the job's
workdir *before* it signals, and only that marker earns ``cancelled``.
Anything else that vanished is ``failed``: it is a real failure, and
because orchestrators capture diagnostics for failures and not for
cancellations, mislabelling it also threw away the logs of the one kind
of death nobody chose. (A ``status`` call after ``cleanup`` has removed
the workdir has no evidence left to read and reports ``failed``; the
lifecycle does not define ``status`` after teardown.)

### Cancel stops the whole tree

A job is not one process. The pid a launcher records belongs to a
bookkeeping shell; the work — the wrapper, the runner, an inference
engine and its workers — lives below it, and that is what holds the
GPU. Signalling only the recorded pid therefore reparents the run onto
init, where it keeps the device busy for whatever runs next, while the
job reports ``cancelled``.

Both launchers give a job a process group of its own and both cancels
signal that **group**:

- ``SSHBackend`` asserts job control (``set -m``) inside an explicit
  ``bash -c``, because sshd hands the command to the user's login shell
  and that lottery decides whether the job gets its own group at all —
  ``dash``/``sh`` accept the option but still leave background jobs in
  the session's group, and ``zsh`` rejects it outright. Job control is
  then **verified, not assumed**: the launcher reports ``$-`` and the
  pgid ``ps`` measured, and only a job whose group is confirmed is
  group-signalled. Jobs submitted before this carry no such marker and
  keep the single-pid path, since their pid names no group.
- ``LocalBackend`` spawns with ``start_new_session``, without which the
  child would share the orchestrator's group and the signal would come
  back at the caller.

Liveness is asked of the group too: the bookkeeper can die while the
runner it launched keeps running, and a pid-only probe calls that job
finished.

Cancellation is ``SIGTERM``, a grace period, then ``SIGKILL``. The grace
is load-bearing rather than polite: a runner that started a model server
through a backend put that server in a session of **its own**, so the
group signal never reaches it, and the only thing that stops it is the
runner's own teardown. Every ``strata_forge.pipelines`` runner therefore
turns ``SIGTERM`` into an ordinary cancellation so its ``finally`` blocks
unwind — Python's default handling would terminate the interpreter where
it stands and strand exactly the process the cancel existed to stop. That
handling lives once, in ``strata_forge.pipelines._common``, alongside the
rest of the plumbing every VM-side runner shares (token scrubbing, repo-id
re-validation, the elapsed-stamping phase ticker, and the entry point that
reports an outcome exactly once).

Scrubbing is :class:`strata_forge.core.redact.Redactor`, built once per run
from the write token (``run_redactor``) and applied to every phase caption,
every error event, the failure reason printed to stderr, and the per-row
``error`` column the inference runner writes into its results and pushes to
the Hub. A token too short to redact safely fails the run before it starts.
See [Redacting the console](#redacting-the-console) for the half a relay owns.

The package ships two runners over that plumbing:
``inference_runner`` (serve a model with vLLM, run a batch over a dataset)
and ``finetune_runner`` (train with :mod:`strata_forge.training`, push the
adapter or merged model to the Hub). Both read an inert JSON spec from
``STRATA_RUN_CONFIG``, take the Hugging Face token only from the private
secrets file the backend wrote beside the job, and append the same ``ProgressEvent`` stream to
``FORGE_PROGRESS_PATH`` — so an orchestrator reads one protocol regardless
of which is running.

### Batch-inference pins, sampling and manifest

``RunSpec.model_revision`` pins the model to a full 40-character lowercase
Hub commit id, and the runner downloads only that revision. A branch, tag or
abbreviated id is refused at spec load, before anything is downloaded,
because the revision it names can move. Left unset, the run serves the
repo's default branch as it is at download time.

``Hyperparams.seed`` and ``Hyperparams.top_k`` are sent to the model server
with every request, next to ``temperature`` / ``top_p`` / ``max_tokens``:
``seed`` as an OpenAI-compatible request field, ``top_k`` inside
``extra_body`` because chat completions has no parameter for it. Unknown
hyperparameter keys are refused like any other unknown spec key.

### Reasoning models (gpt-oss)

A reasoning model's reasoning is kept out of its answer. ``Hyperparams.reasoning_parser``
(``openai_gptoss`` or ``qwen3``, an allow-list because it becomes the model server's
``--reasoning-parser`` flag) makes vLLM split the reasoning from the final text and return it in
the response's ``reasoning`` field; the runner stores it in the ``reasoning`` column and the final
text alone in ``output``. The parser is settled against the checkpoint rather than the model id:
before the dataset or the weights are downloaded, the runner reads only the model's
``config.json`` (at the pinned revision; an unpinned run then downloads the weights at the commit
that file was read at) and its ``model_type`` decides. A gpt-oss checkpoint (``gpt_oss``) with no
parser is served with ``openai_gptoss``, whose absence would put the harmony analysis channel
inside every ``output``, and any other parser on it is refused; ``openai_gptoss`` on another
model, and ``qwen3`` on anything but ``qwen3`` / ``qwen3_moe``, are refused. The parser served
with is recorded as ``serving.reasoning_parser``.

``Hyperparams.reasoning_effort`` (``low`` / ``medium`` / ``high``) is gpt-oss's harmony effort and
is sent in ``extra_body`` (a top-level field of the request body) with every request. It is
refused at spec load with any parser but ``openai_gptoss``, and at the config check on a model
that is not gpt-oss; the same holds for ``harmony_date``.

The harmony system message states a "Current date", which vLLM takes from
``VLLM_SYSTEM_START_DATE`` and otherwise from its own clock on every request, so unpinned
prompts differ from one day to the next and a run can cross midnight. A gpt-oss run therefore
always pins it: ``Hyperparams.harmony_date`` (a calendar date written ``YYYY-MM-DD``) when given, else the serving machine's local
date taken once at start-up. The manifest records the date used as ``serving.harmony_date``
(``None`` for a model without harmony).

vLLM's defaults run gpt-oss-120b out of memory on a single 80 GB GPU. Its recipe raises
``--gpu-memory-utilization`` (e.g. ``0.95``) or lowers ``--max-num-batched-tokens`` (e.g.
``1024``), which ``Hyperparams.gpu_memory_utilization`` and ``Hyperparams.max_num_batched_tokens``
set; ``max_model_len`` bounds the context the KV cache must hold. The ``[serving]`` extra's
``vllm>=0.24,<1`` range serves gpt-oss's MXFP4 checkpoints with the harmony parser on every
release in it.

Each results row is ``{custom_id, output, reasoning, error, error_kind,
finish_reason, input_tokens, output_tokens, latency_ms}``; ``reasoning`` is
``None`` when the server returned none, and a failed row carries ``None`` in
it, ``finish_reason``, the token counts and ``latency_ms``. ``latency_ms`` is
the client's own measure of the row's request, in whole milliseconds (a
guided-JSON retry counts both attempts). ``error_kind`` is ``None`` on a
row that succeeded and, on a failed one, a stable class read off the
exception's type, never its text: ``timeout``, ``rate_limited``,
``model_error`` (any other provider error), ``invalid_output`` (a
structured-output or validation error) or ``other``; a fallback chain is
classed by its last attempt.

A row is named by ``custom_id``: the value of the spec's ``id_column`` when
one is declared (it must be a column of the split, and its values present,
1 to 512 printable characters, not blank (``valid_row_id``, the rule the
control plane applies to a retry's ids) and unique across the rows the run
reads, else the run is refused naming the row position, never the value),
else its position
``row-<i>``. A position names a different row as soon as anything before it
changes, so a declared id is what lets a later run name the same rows.
``row_ids`` (1-100,000 distinct ids, as ``custom_id`` names them) runs only
those rows, selected in split order from the rows the run reads (the first
``row_limit`` rows, or the whole split without one): a retry that carries
its source run's ``row_limit`` reads the same rows and checks their ids for
uniqueness over the same window. A run whose requested ids are not all
among those rows is refused with the count it could not find. Both are
chat-only: a scoring engine's rows carry their own declared ``item_id``,
and a spec setting either with another engine is refused. ``finish_reason`` is the client's normalized value (``stop``,
``length``, ``tool_use``, ``content_filter``, ``error``), and a reason the
client does not recognise is reported as ``stop``; ``length`` (the output
hit ``max_tokens``) is always preserved. Beside ``results.parquet`` the runner writes ``manifest.json``
(schema ``strata.run.manifest/v1``, ``kind`` ``inference``; every runner's
manifest shares the schema), pushed with the results or kept with them on
the machine. It records:

- the model id, the requested revision and the commit actually served
  (the requested pin, or the commit the Hub cache resolved the default
  branch to);
- the dataset id, config, split and requested revision;
- the engine version, the Python version and the installed versions of
  ``vllm``, ``transformers``, ``torch``, ``datasets`` and
  ``huggingface-hub``, which the machine installs fresh per run within a
  range;
- what the model server actually ran with, which can differ from what was
  requested: the ``max_model_len`` its ``/models`` route reports, and the
  dtype requested, declared by the checkpoint and resolved by the server
  (from its start-up configuration line), each ``None`` when unknown, and
  the server's command-line ``flags`` from ``--host`` on (the interpreter
  and the local snapshot path are left out), and the ``harmony_date`` a
  gpt-oss system message stated;
- a SHA-256 of the template (never its text), the column mapping and the
  hyperparameters;
- the ``id_column`` the rows were named by (``None`` for positions) and,
  for a run given ``row_ids``, their ``count`` and a ``sha256`` of the ids
  sorted and joined by newlines (never the ids themselves), else ``None``;
- row, success and failure counts, a ``finish_reason`` histogram and token
  totals.

It holds no row content and no credential.

The runner also writes ``receipt.json`` (``RECEIPT_FILENAME``, schema
``strata.run.receipt/v1``) into its **working directory**, the job
workdir an orchestrator reads before it removes it; the manifest sits
beside the results, outside that directory. Written once the results and
manifest are written, whether the push succeeded or failed, it carries the
SHA-256 of the ``manifest.json`` bytes as written and the facts an
orchestrator compares with what it admitted: ``model`` (``id``,
``requested_revision``, ``commit``), ``dataset`` (``id``, ``config``,
``split``, ``requested_revision`` and the ``commit`` the split was read at:
an unpinned run resolves the head to a commit before the load and loads at
it, so this names the rows read), ``template_sha256`` (null on a run that
rendered a pinned prompt build), ``prompt`` and ``system_prompt`` (each
``{build_id, build_hash}`` from the manifest's prompt blocks, or ``None``),
``engine_version``, ``id_column``, ``row_ids`` (``{count, sha256}`` or
``None``), ``rows`` (``total``, ``succeeded``, ``failed``) and ``output``
(``repo_id`` and the ``commit_sha`` of the Hub commit the push made, or
``None`` when nothing was pushed). No row text, no prompt text and no
credential. Its ``kind`` is ``chat``.

Beside it the chat runner writes ``failed_rows.json``
(``FAILED_ROWS_FILENAME``, schema ``strata.run.failed_rows/v1``): the
``id_column``, the ``count`` of distinct failed rows and, when rows are
named by ``id_column``, their ``row_ids`` in split order, bounded by
``MAX_FAILED_ROW_IDS`` (10,000) and ``MAX_FAILED_ROW_ID_BYTES`` (64 KiB),
with ``complete`` saying whether every failed row is listed. Positional
names are counted and never listed. It is what a retry of the failed rows
may name; the ids are the user's data and appear nowhere else.

A run that pushed ends with ``repo@commit`` as its end event's message
(``_common.pushed_at``; the repo alone when the Hub named no commit), so an
orchestrator records the commit the output landed in. Chat, scoring,
fine-tuning and evaluation runs all end this way.

Every other pushing runner writes one too, through
``_common.build_receipt(kind, manifest_path, output, rows=...)``: a scoring
run (``kind`` ``scoring``: ``hf_classify``, ``hf_label_logits``, verifier
artifacts), a fine-tuning run including ``seq_cls`` (``training``) and an
evaluation run (``eval``). It carries the manifest digest, the ``model``,
``dataset`` and ``engine_version`` the manifest records (null where that
manifest has none), ``rows`` where the run has rows, and ``output`` with the
commit the push made. That commit is what an orchestrator links runs by: a
scoring run that reads a trained model at that commit, and an evaluation
that reads a scoring run's predictions at theirs, link exactly, where a
repo's head could have moved. A model registry pinning a training output
reads the same commit.

A pushed run goes only to a **private** output dataset repo: the runner
creates it private when it is missing and refuses a public one, first
before any download or GPU time and again at push time, because a run's
outputs and manifest name the model and dataset it read. The results and
the manifest go up in one Hub commit of exactly those two files, so a repo
never holds one run's results beside another run's manifest. The check
creates a missing repo private, so a run that fails afterwards can leave an
empty private repo behind.

### Samples per row (`n`) and chat-template kwargs

``Hyperparams.n`` (1-16, default 1) draws ``n`` samples per row, each its
own request. A seeded run gives sample ``k`` the seed ``seed + k``, which
is exactly the seed vLLM gives the ``k``-th child of an ``n``-sample
request (``vllm/v1/engine/parallel_sampling.py``,
``_get_child_sampling_params``), so each sample is one reproducible request
with its own token counts and finish reason, which a shared ``n``-choice
response does not report. Each sample's requests for a chunk run as one
batch, and the model server's prefix cache serves the repeated prompt.
``n > 1`` below ``temperature`` 1e-5 is refused (vLLM decodes greedily
below that, ``_SAMPLING_EPS``, so it would draw the same reply ``n`` times), as is a ``seed`` for which ``seed + n - 1`` leaves the
server's int64 range. Scoring engines refuse ``n``.

Whenever a run declares ``n`` (``1`` included), every results row is keyed
by ``(custom_id, sample_index)`` and carries the ``seed`` its request sent
(``None`` unseeded), rows ordered by row, then sample. Progress counts
sample rows (``total_steps`` is rows x ``n``), and a chunk holds
``progress_chunk`` requests (``progress_chunk // n`` rows), so the liveness
check and the progress event between chunks keep their spacing; with guided JSON each
sample gets its own repair turn under its own seed. The manifest adds
``sampling: {n, seed, sample_seed: "seed + sample_index"}`` and
``counts.prompts`` (distinct rows); a run that does not declare ``n`` has
``sampling: null`` and its rows are unchanged.

``Hyperparams.chat_template_kwargs`` also applies to chat runs, with the
rules of the label-logits engine below, except that values are FIXED for
the run (a ``{name}`` placeholder in them is refused; per-row values are
rendered only by ``hf_label_logits``). They ride every request as vLLM's
``chat_template_kwargs`` in canonical key order; ``enable_thinking`` is
passed only this way and must be a boolean. Before the model server
starts, the runner reads the snapshot's chat template
in vLLM's order of preference: the processor's ``chat_template.json``,
then ``chat_template.jinja``, then ``tokenizer_config.json``'s
``chat_template`` (the ``default`` entry of a named list) and refuses a
kwarg it never reads, since vLLM keeps only the kwargs its template reads
(``vllm/renderers/hf.py``, ``resolve_chat_template_kwargs``); a model with
no template; and any kwarg on a gpt-oss model, which vLLM renders with
openai-harmony rather than the template. The manifest records
``chat_template: {sha256, kwargs_sha256}``.

### Guided JSON (`guided_json`)

``Hyperparams.guided_json`` is ONE JSON Schema for every reply of a chat
run. It is sent with every request as vLLM's structured-output constraint,
``extra_body: {"structured_outputs": {"json": <schema>}}`` (the request
field vLLM has taken since it removed ``guided_json`` in 0.12; the locked
0.27.1's ``ChatCompletionRequest.structured_outputs`` /
``StructuredOutputsParams.json``), so decoding can only produce text the
schema's grammar admits. Every reply is still validated, because the
constraint is not the whole schema: a reply cut off at ``max_tokens`` is
not JSON at all, and a grammar backend does not enforce every keyword.

At spec load the schema must be a non-empty JSON object nested at most 32
levels deep (found iteratively, before anything recursive walks it, so a
deeper schema is a spec error rather than a ``RecursionError``), of at most
16 KiB (canonical UTF-8 JSON), a valid schema for the draft its ``$schema`` names
(2020-12 when it names none), and every ``$ref`` / ``$dynamicRef`` /
``$recursiveRef`` in it must point inside it (``#…``). ``jsonschema`` still
retrieves a remote reference by default, so the validator is also built
with an empty ``referencing.Registry`` and fetches nothing either way.

A reply that fails validation (not JSON, ``NaN``/``Infinity``, or a schema
violation) gets exactly ONE repair turn: the original conversation, the
invalid reply as the assistant turn, and the fixed message
``Your previous reply does not satisfy the required JSON schema: {error}``
``Reply again with only the corrected JSON.``, under the same constraint
and sampling settings. The repaired reply becomes ``output``; tokens and
latency are the sum of both attempts; a repair request that fails keeps
the first reply. A row whose request failed is not repaired. A reply nested more than 64 levels deep is
recorded invalid without being validated, and anything the validator raises
on one reply marks only that row invalid: one reply never ends the run.

Guided rows add ``json_valid`` (``None`` for a row with no reply),
``json_retried`` and ``json_error`` (the final validation error, at most
300 characters plus its JSON pointer, scrubbed like ``error``). Progress
and ``end`` metrics add ``json_invalid`` and ``json_retried``. The manifest
gains a ``guided`` block (``kind: json``, ``schema_sha256`` over canonical
JSON, ``constraint: structured_outputs.json``, the ``validator`` draft
class, ``max_repairs: 1``, ``repair_prompt_sha256``) and ``counts`` gains
``json_valid``, ``json_invalid``, ``json_retried`` and ``json_repaired``.
An unguided run's manifest has ``guided: null`` and its rows have none of
these columns.

### Classifier scoring (`hf_classify`)

``Hyperparams.engine`` chooses what an inference run does with each row:
``vllm_chat`` (the default) generates a reply through the model server;
``hf_classify`` reads a sequence-classification head's logits and starts
no model server. The scoring knobs live in ``Hyperparams``, so a control
plane that validates hyperparameters with this model accepts them with no
schema of its own:

| Field | Meaning |
|---|---|
| ``pair_template`` | Required. The second text of each pair (the hypothesis); ``template`` renders the first (the premises). Its placeholders need ``column_mapping`` entries like the template's. |
| ``list_separator`` | Joins a list-valued column (a row's premises) when it fills a placeholder. Default ``"\n"``. The list's elements must be strings or numbers; a ``list<struct>``, nested-list, boolean or null element is refused, naming the column (stage a ``list<string>`` column instead). |
| ``label_alias`` | Required. Names every ``id2label`` class exactly, no missing and no extra key: each maps to its output label, or to ``null`` to EXCLUDE it (no columns; the softmax and argmax run over the included classes). Two classes mapping to one label are refused (no combine rule exists); at least two classes stay included. |
| ``id_column`` | Optional. A column whose value becomes each row's ``item_id``: a string or an integer, unique, at most 256 characters. |
| ``slice_column`` | Optional. A column (at most 256 distinct values) the counts are kept per value of, for per-tier receipts. A missing value counts under ``(null)``; a value spelled ``(null)``, one over 64 characters, or one spelling held by two types (``1`` and ``"1"``) is refused rather than merged. |
| ``scoring_batch_size`` | Rows per forward pass, 1-1024. Default 16. |
| ``max_model_len`` | Optional token limit. Unset, the model's own: the smaller of the position table (less ``pad_token_id + 1`` slots on padding-offset architectures such as RoBERTa, whose position ids start after the padding index, so 514 slots fit 512 tokens) and the tokenizer's ``model_max_length``. A larger one is refused. |
| ``dtype`` | Required, no default: ``float32`` or ``bfloat16``. In ``float32`` the TF32 matmul and cuDNN paths are switched off (``torch.backends.cuda.matmul.allow_tf32`` / ``cudnn.allow_tf32``) before the model loads. In ``bfloat16`` a row's logits can depend on which rows share its batch; in ``float32`` the difference is at rounding level. |
| ``row_limit`` | As for chat. |

Every other knob generates text, so it is refused with this engine, and the
scoring knobs are refused with ``vllm_chat``.

The model comes from the same filtered snapshot as a chat run (safetensors
only, no repo code) and loads with ``AutoModelForSequenceClassification``,
``trust_remote_code=False``, from the local snapshot alone. A load that
left any weight uninitialised, reported a shape mismatch or an error, or
ignored a checkpoint tensor under ``classifier.`` / ``score.`` / ``head.``
is refused (``training.loading.check_loading_info``); other ignored
checkpoint keys (a pooler, a ``position_ids`` buffer) are recorded.
Labels are read from ``config.id2label`` in class-id order, never in
dictionary order.

**Nothing is truncated.** Each pair is measured with its special tokens; a
pair over the limit is dropped before any forward pass, counted, and still
written as a row with ``overflow`` true and no score. The rest run in
length-sorted batches, longest first, right-padded. A run where every row
overflowed fails after writing its results.

Each results row is ``{custom_id, item_id?, slice?, input_tokens, overflow,
pred, logit_<label>..., p_<label>...}``: the raw head logits, their plain
softmax over the included classes (temperature 1; no calibration is applied)
and the argmax label over them (the lowest class id on a tie). The ``end`` event's metrics are ``succeeded``
(scored rows), ``failed`` (0) and ``overflow``.

The manifest beside the results has ``kind`` ``scoring``. It records the
model's requested revision and loaded commit, and the dataset's requested
revision and the commit actually read: an unpinned dataset is resolved to
its default branch's head once, before the load, and loaded at that commit.
Beside the engine versions (``transformers``, ``torch``, ``tokenizers``,
``safetensors``, ``datasets``, ``huggingface-hub``), the template and pair
template digests (the hyperparameters are recorded without the pair
template's text), and the column mapping, its ``scoring`` block records the
model's ``id2label``, the ``label_alias`` used, the included labels and the
excluded classes, the length limit and where it came from, the overflow
policy (``drop_and_count``), ``calibration: null``, the padding side and pad
token, the loaded dtype, the TF32 switches as they stood, the device, the
batch plan (``order`` ``token_length_desc_then_row_index``, ``batch_size``,
the number of batches and a SHA-256 of the plan), and the ignored checkpoint
keys; ``counts`` holds rows, scored and overflow, overall and ``by_slice``.
An excluded class leaves no column and never shifts another class's: each
``logit_<label>`` is that class id's logit.

#### Scoring a trained verifier (``artifact: strata_verifier``)

``hyperparams.artifact = "strata_verifier"`` scores a verifier trained by
``seq_cls`` (encoder or decoder) exactly as it was trained. Everything about
its input comes from the artifact's ``strata_verifier.json``, never from run
parameters: the premise lines (``"{pid}: {text}"`` or the text alone, joined
by ``"\n"``), the encoding (an encoder's ``(premises, hypothesis)`` pair, or a
decoder's ``decoder_template`` filled in one pass with the tokenizer's own
special tokens, answer read at the final token), the pooling, the label order
(the model's ``id2label`` must name the verifier's labels in order) and the
auxiliary heads. The trainer and the scorer render through one module,
``strata_forge.training.verifier_render``, and read the auxiliary heads
through one function, ``classifier_heads.aux_outputs``. The spec's templates
only name the two roles, so they are fixed: ``template`` ``{premises}``,
``pair_template`` ``{hypothesis}``, each mapped to its column; ``premises`` is
a list of ``{pid, text}`` with pids ``p1..pN``, as in training, and a malformed
cell refuses the run naming its row. ``label_alias`` is optional (each label
as itself), ``list_separator`` is refused, and ``dtype`` stays required. The
length limit is the artifact's training ``max_length`` unless
``max_model_len`` asks for a smaller one (a larger one is refused: a verifier
is scored only on rows it could have been trained on; ``max_length_source``
``artifact``); overflow is dropped and counted as in pair mode. The record is
read back under the trainer's rules: a decoder template must hold
``{premises}`` and ``{hypothesis}`` once each and match its recorded
``decoder_template_sha256``. A decoder row whose last token is the pad id is
refused, as training refuses it: the stock head reads the rightmost non-pad
token and the auxiliary heads the last one the attention mask keeps, and only
on such a row would they differ.

Results rows add, when the artifact has those heads, ``logit_inconsistent``
and ``p_inconsistent`` (its sigmoid) and ``support_scores``: a list of
``{pid, logit, p}``, one per premise in premise order (``null`` on an
unscored row). ``apply_calibration`` (artifact mode only) is ``off`` by
default; ``auto`` divides the logits by the artifact's ``calibration.json``
temperature before the ``p_*`` softmax. The ``logit_*`` columns are always the
raw logits. The manifest's ``scoring`` block adds ``input``
(``strata_verifier`` or ``pair``), ``artifact`` (the verifier record's SHA-256,
its rendering, pooling and heads) and ``calibration`` (the mode, whether it
was applied, the temperature, the record's SHA-256 and fit, and
``raw_logits_kept``).

A scoring run pushes only to an output repo of its own: one holding any
file but ``.gitattributes`` is refused before the dataset loads and again at
push time, so a repo never mixes two runs' rows.

### Label-token logits after a chat template (`hf_label_logits`)

``engine="hf_label_logits"`` reads a generative model's next-token logits
for a few declared label tokens instead of sampling an answer, with no
model server. Each row's ``template`` renders into one user message; the
model's own chat template turns it into a prompt with the assistant's
generation header (``add_generation_prompt=True``) and the run's
``chat_template_kwargs``; ``assistant_prefill`` is appended after the
header; one forward pass reads the logits at the prompt's last position.
It shares the scoring frame above (the snapshot, the private fresh output
repo, ``id_column`` / ``slice_column`` / ``list_separator`` /
``scoring_batch_size`` / ``max_model_len`` / ``dtype`` / ``row_limit``,
drop-and-count overflow and the one-commit push) and adds:

| Field | Meaning |
|---|---|
| ``label_tokens`` | Required. 2-64 ``{label, token}`` entries in order: the output label (the classifier's label grammar) and the token's TEXT, leading space included where the vocabulary has one (``" True"``). Labels and tokens are distinct. |
| ``assistant_prefill`` | Optional, at most 1,024 characters. Text appended after the assistant header and read past (a judge's ``"<score>"``). Literal: no placeholder in it is substituted. |
| ``chat_template_kwargs`` | Optional. Variables the chat template reads (``enable_thinking``, ``documents``, ``guardian_config`` …), at most 16 identifier keys and 16 KiB of JSON, nested at most 6 deep. Any string inside a value may hold ``{name}`` placeholders, rendered per row like the template (a list column is joined), and covered by the same mapping check. |

``pair_template`` and ``label_alias`` belong to ``hf_classify`` and are
refused here, as every generation knob is.

**What is refused, and when.** At spec load: a key that names one of
``apply_chat_template``'s own parameters (``chat_template``, ``tokenize``,
``add_generation_prompt``, ``tools`` …), the conversation (``messages``)
or a special token (``bos_token`` …), because those change how the
template is chosen or rendered rather than what it says; ``documents`` is
allowed, since templates read it as data. At model load, before any GPU
work: a model without a chat template; a declared kwarg the template never
reads (both transformers and vLLM drop such a kwarg silently, so it would
record a setting the model never saw; the template's variables are found
by parsing it in transformers' own Jinja environment, the way vLLM's
``_resolve_chat_template_kwargs`` does); a label token that is not exactly
one token on its own, or two labels with one token id; an id outside the
model's output layer; a model whose ``forward`` cannot project selected
positions (``logits_to_keep``); and any load that left a weight
uninitialised. While measuring: a prefill that merges with the end of the
rendered prompt when tokenized (the prompt's own tokens must be a prefix of
prompt + prefill), or that tokenizes differently on different rows; and a
label token that is not the single next token after the first row's prompt
and prefill. A chat template that raises (``raise_exception``) stops the
run.

The prompt and the prefill are tokenized together, with
``add_special_tokens=False`` as ``apply_chat_template`` itself does, since
the template places its own special tokens. Rows run right-padded in
length-sorted batches; only each row's last real position is projected
onto the vocabulary (``logits_to_keep``), and padding after a row never
changes it. The model loads with ``AutoModelForCausalLM``,
``trust_remote_code=False``, safetensors only, in the run's required
``dtype`` (``float32`` or ``bfloat16``, as for ``hf_classify``; ``float32``
switches the TF32 matmul and cuDNN paths off first).

Each results row is ``{custom_id, item_id?, slice?, input_tokens, overflow,
pred, logit_<label>..., p_<label>..., mass_outside}``: the RAW logit of
each declared token, their softmax over the declared tokens only
(temperature 1, no calibration), the argmax label (the first declared on a
tie), and ``mass_outside``, the probability the full output distribution
puts on every other token, computed in float64 as
``-expm1(logsumexp(declared) - logsumexp(all))``. An overflowed row has
``None`` in every score column.

The ``kind`` ``scoring`` manifest's ``scoring`` block records
``head: causal_lm_next_token``, the labels, each label's token text,
resolved ``token_id`` and vocabulary ``piece``, ``add_generation_prompt``,
``chat_template_sha256`` (the template text the tokenizer used),
``chat_template_kwargs_sha256`` (over canonical JSON, keys sorted at every
depth, which is also the order they are rendered in, so a stored and
re-read spec renders the same prompt), ``assistant_prefill_sha256`` and
``assistant_prefill_token_ids``, the output layer's ``vocab_size``, the
length limit and its source, the overflow policy, the padding side, dtype,
the TF32 switches, the batch plan, device and the ignored checkpoint keys. Like the pair
template, the prefill and the kwargs are left out of the manifest's
``hyperparams`` dump: only their digests are pushed.
### Evaluation runner

``strata_forge.pipelines.eval_runner`` scores pinned prediction files against a
pinned gold split; it loads no model. Its spec (``EvalSpec``) names the gold
dataset, ``dataset_commit_sha`` (required), config and split, plus an
``evaluation`` block (``EvalSettings``, the model a control plane validates a
submission with):

- ``labels``: the ordered label space; ``gold_column`` and ``id_column`` on the
  gold split; ``gold_aliases`` mapping gold values onto labels.
- ``predictions``: the candidate system and ``compare_to`` up to eight
  baselines. Each is a ``PredictionSource``: a ``name``, a dataset ``repo_id``,
  a full ``commit_sha`` (required), a ``.parquet`` ``path`` (default
  ``results.parquet``), its ``id_column`` (default ``item_id``) and
  ``prediction_column`` (default ``pred``), an optional boolean
  ``overflow_column`` and ``label_aliases``.
- ``cluster_column`` (each item is its own cluster when unset),
  ``slice_columns`` (up to eight; a list value puts a row in each listed slice;
  at most 64 values per column), an optional boolean ``exclude_column``,
  ``seed``, ``resamples`` (default 10,000) and ``min_slice_rows`` (default 500).

Every read is pinned, so a run reads only data it was explicitly pointed at,
and only the declared gold columns are materialised. Every row has a status per
system (``status__<name>`` in ``items.parquet``):

- ``tier_refused``: over the ``tier_rule`` limit for its tier
  (``limits[<tier_column value>]``, else ``default``; ``max_premises`` counts the
  ``premises_column`` list, ``max_tokens`` reads the gold ``token_column``), and
  ``excluded``: flagged in ``exclude_column``. Both are common to every system,
  left out of every population and counted in ``population``.
- ``overflow``: the system's ``overflow_column`` is true.
- ``missing``: no prediction, a null one, or a value outside the label space. A
  missing row is always invalid (a false negative for its gold class, no false
  positive) and is never excluded.
- ``ok`` otherwise. A prediction id with no gold item is counted and ignored; a
  repeated id refuses.

The primary read is over the **shared** population, the rows no system
overflowed, so every system and comparison is read on the same rows; each slice
reports ``n`` (shared), ``n_tier``, ``overflow_excluded`` and ``overflow_share``.
Each system also gets a ``full_tier`` secondary read over every non-refused row,
its own overflow counted as invalid. A slice lists ``absent_classes`` (declared
labels with no gold row and no prediction in it).

``tier_rule.token_method_column`` names how ``token_column`` was counted (for
example ``tokenizer:<repo>@<commit>``): every row needs one, a tier that mixes
methods refuses, and each tier's method is echoed in ``population.token_methods``.
A declared ``premise_count_column`` must equal the platform's count of the
``premises_column`` list on every row, else the run refuses naming the row.
Premises are ``{pid, text}`` objects; a plain-string premise refuses.

``label_space_column`` with ``label_spaces`` (``{<value>: [labels]}``, each two or
more of the declared ``labels``) gives each row its declared label space (a 2-way
tier declares its own two labels; a row with no entry declares ``labels``). A
population declares the union of its rows' spaces (``labels`` on each slice), so
a 2-way slice is never missing the 3-way classes; a gold label outside its row's
space refuses, and a prediction outside the population's labels is invalid.

**Absent classes are the gold's.** A class is absent from a population when its
GOLD has no row of it, whatever any system predicted. Then every system's
macro-F1 and every paired macro-F1 delta have NO value: ``point``, ``lower`` and
``upper`` are null with ``reason: "absent_gold_class:<label>[,<label>…]"`` and
``absent_classes`` (null by rule, not unstable; never a mean with a zero in it).
The slice's ``absent_classes`` and every system's are the same list, both read
from the shared rows' gold. The ``end`` event carries ``macro_f1_undefined_share`` always, and ``macro_f1_undefined: 1``
(and ``delta_macro_f1_undefined__<baseline>``) in place of the value. Beside it,
``macro_f1_present_classes`` averages the classes the gold has (listed in
``classes``, the same set for every system); it is a diagnostic with
``gate_read: false`` and decides nothing. ``absent_class_predictions`` counts,
per system, its predictions of each gold-absent class (a diagnostic, never
gated). Within the bootstrap a resample whose own gold lacks a declared class is
undefined and counts toward the ``unstable`` threshold. Label spaces are per row
(below), so a 2-way tier is never missing the 3-way classes. Each slice also
reports its ``tier_refused`` rows. ``tier_rule.token_column`` and
``token_method_column`` are declared together, and a ``label_space_column``
value with no entry in ``label_spaces`` refuses, naming the row.

``population.fallback_keyed`` and each slice's ``fallback_keyed`` count the rows
clustered on the premise-hash fallback (0 on a suite tier, whose rows all carry a
cluster value).

Every interval carries ``undefined_share`` (its undefined resamples over
``resamples``) and ``unstable``: true when that share exceeds
``unstable_threshold`` (default 0.01, recorded in ``population``) or no resample
was defined. An unstable interval never decides a gate: a delta's
``excludes_zero`` is then ``"unresolved"``, and the ``end`` event leaves its
bounds out.

Id, cluster and slice columns must be short identifiers, never text: their
values are written to ``items.parquet``, so a value over 256 characters refuses,
as does a column that mixes value types (``1`` and ``"1"``, ``True`` and
``"True"``). A list slice value names each slice once per row. An
``overflow_column`` must be boolean, like ``exclude_column``. A pushed run's
``end`` message is ``<repo>@<commit>``, the commit it made, so the report is read
back at that commit rather than at a head another run may have moved.

The cluster is ``cluster_column``. A row without one (or every row, when only
``premises_column`` is declared) clusters on ``premise_cluster_key`` of its
premises: each premise text (a string, or the ``text`` of a ``{pid, text}``
object) NFC-normalised with whitespace runs collapsed and ends stripped, the
texts sorted, encoded as a compact UTF-8 JSON array, and hashed with SHA-256
(``premises:sha256:<hex>``); case is kept. Without either, each item is its own
cluster. A slice resamples only the clusters present in it.

The overall population and each slice get accuracy, macro-F1 and invalid rate
per system with the [cluster percentile
bootstrap](metrics.md#cluster-percentile-bootstrap), and a paired
candidate-minus-baseline delta per baseline with its paired discordance rate
(``discordance``: ``baseline_only_correct``, ``candidate_only_correct``,
``p_d`` = their sum over ``n``). A slice whose shared population is under
``min_slice_rows`` rows is reported with ``diagnostic: true``. An interval
records ``undefined_resamples`` (see the metrics reference); a one-sided bound is
the matching end of the two-sided interval. The outputs are ``items.parquet``
(per gold item: id, gold, cluster, ``excluded``, ``slice__<column>``, and per
system ``pred__<name>`` and ``correct__<name>``), ``metrics.json`` (schema
``strata.eval.report/v1``: the systems with their join counts, the population,
and one entry per slice with ``n``, ``clusters``, ``diagnostic``, per-system
intervals, the overall per-class scores and confusion, and the comparisons),
``manifest.json`` (``kind`` ``evaluation``: the pinned inputs, settings,
engine, counts, and the overall read's ``bootstrap`` contract id, NumPy version,
seed, resamples and cluster count) and a dataset card whose ``configs:`` points the Hub viewer
at ``items.parquet``. They are pushed in one commit to a private output dataset
repo, refused when public, or kept on the machine under
``~/strata-eval-results/<run_id>``. The ``end`` event carries the candidate's
overall figures and bounds and each baseline's accuracy and macro-F1 deltas.
The machine needs ``strata-forge[scoring,storage,hf]``.

``EvalSettings.calibration`` (``CalibrationSettings``) adds probability reads on the
shared rows. Every file must carry ``<logit_prefix><label>`` (default ``logit_``)
for each declared label; an overflowed, refused or non-finite row has none. Three
reads are reported apart, per system and slice, each with ``n``, ``coverage``,
sum-form Brier (Σ_k (p_k − y_k)² on [0, 2]) with a cluster interval, NLL and
top-label ECE (``ece_bins``, default 15, equal mass, ties by item id) and that
ECE's own bins, ``reliability_bins``: one object per bin in confidence order,
empty ones included, each ``{lo, hi, n, mean_confidence, accuracy}`` (``lo`` /
``hi`` the smallest and largest confidence in the bin; an empty bin has ``n`` 0
and null for the rest). Their ``n``-weighted mean of
``|accuracy - mean_confidence|`` is the block's ``ece``:

- ``raw``: the logits' softmax.
- ``cross_fit``: ``cross_fit_temperature`` over the fit population (the scored
  rows of this split, or of its named slice ``fit_column`` = ``fit_value``), with
  ``folds`` (default 2) folds by cluster: the seeded permutation of the sorted
  cluster keys, the first ``ceil(C/2)`` in fold 0. Null outside the fit population.
- ``applied``: the system's own temperature, from the ``calibration.json`` its
  ``PredictionSource.calibration_ref`` pins (a model or dataset repo, a commit and
  a path), when ``apply_calibration`` is ``auto``. A ``calibration_ref`` without
  ``calibration`` settings refuses. Its SHA-256 and where its labels came from
  (``labels_from``) are recorded.

**The calibration file contract** (one schema, shared by the seq-cls trainer that
writes it and the evaluation runner that reads it):

- ``calibration.json`` is a JSON object (at most 64 KiB) with
  ``"schema": "strata.calibration/v1"`` and ``fit``, the trainer's
  ``TemperatureFit``: ``fit.temperature`` (positive, finite) is the temperature
  applied, with ``nll_before`` / ``nll_after`` / ``at_bound`` beside it. The
  trainer also records ``split``, ``rows``, ``excluded_rows`` and ``rows_sha256``
  (the ID-dev rows fitted on).
- Labels: the file's ``labels`` (the class order of the logits the temperature
  scales) when present, else the ``labels`` of ``strata_verifier.json``
  (``strata.verifier/v1``) in the same directory of the same repo at the same
  commit. Either way they must equal the evaluation's declared ``labels`` in
  order, or the run refuses.

Each probability block reports ``n`` (rows with probabilities),
``n_population`` and ``coverage`` (their ratio), and is computed on those rows
only, with no "no mass" term for the rest. Each comparison gains ``brier_delta``:
the paired candidate-minus-baseline raw Brier on the rows BOTH systems have
probabilities for, with that ``n`` and ``coverage`` and ``sd``, the SD of the
per-item Brier differences and its interval; with calibration the comparison's
``discordance`` (p_d) is read on those same rows, with its ``n`` and
``coverage``. **Gate rule:** a probability read with ``coverage`` < 1 is
unresolved (``gate: "unresolved"`` on a block or p_d, ``excludes_zero:
"unresolved"`` on the Brier delta), and unresolved counts as NOT met. The
``end`` event carries ``brier_coverage`` and leaves the Brier bounds out below
full coverage. The report's ``calibration`` block records the fit
population, the fold rule, each system's per-fold temperatures and its own
temperature. No temperature selects a system, a reference or a threshold. The
``end`` event adds the candidate's raw ``brier`` and its bounds.

### LLM grader (graded evaluation)

``python -m strata_forge.pipelines.grade_runner`` grades each system's OUTPUTS with
a pinned judge model against a rubric. Its spec (``GradeSpec``) pins the gold split
(``dataset_id``, a 40-hex ``dataset_commit_sha``, ``dataset_config``, ``split``) and
carries ``grading`` (``GradeSettings``):

- ``rubric`` (``strata_forge.pipelines._grading.Rubric``): a ``name``, one prompt
  ``template`` and 1-8 ``criteria``. A criterion is ``pass_fail`` or ``ordinal``
  (integer ``min``/``max`` within [0, 10], optional ``pass_at``), with an optional
  ``gold_column`` naming a human score of the same output in the output file. The
  template must contain ``{output}`` and ``{criteria}`` (the numbered criteria
  list); every other ``{name}`` is a gold column, mapped by ``column_mapping``,
  which must map exactly those placeholders.
- ``judge_model_id`` and ``judge_revision`` (a full commit), and
  ``judge_hyperparams``: the judge's chat-run settings (``max_tokens``,
  ``temperature``, serving knobs). The grader owns ``engine`` (``vllm_chat``),
  ``guided_json`` (the rubric's answer schema), ``n`` (one answer), ``row_limit``
  and every scoring knob; temperature is 0 unless set.
- ``outputs``: the candidate first, then up to three baselines, each a parquet at a
  pinned commit with its ``output_column`` (default ``output``) and
  ``error_column`` (default ``error``). With a gold ``id_column`` every output
  names its own ``id_column``; without one every output joins by position
  (``custom_id`` = ``row-<i>``, a batch run's results), which is admitted only when
  the output file's ``manifest.json`` names this dataset, config, split and commit.
  A file with several samples per row is refused, and repeated ids are refused.
  Ids are joined type-exactly (the integer 1 never joins the string "1") and
  copied into the items file as declared.
- ``cluster_column`` (each case its own cluster when null), up to eight
  ``slice_columns``, ``seed``, ``resamples`` (default 10,000) and
  ``min_slice_rows`` (diagnostic below it). A slice column is a declared label:
  at most 64 values of at most 64 characters, else the run refuses it, because
  its values are copied into the report and the items file.

The judge is the batch-inference runner's ``vllm_chat`` path, run in-process with
prepared rows: one request per (case, system with an output), the rubric rendered
by the same non-executing substitution, under the rubric's guided-JSON schema
(``rationale`` first, then exactly one bounded score per criterion) with the one
repair turn. Each (case, system) is ``graded`` or unscored with a reason:
``output_missing`` (no row, an empty output or a generation error),
``judge_error`` or ``judge_invalid`` (not a complete answer after the repair,
re-read by ``parse_grade``). Unscored cases stay visible, are counted per reason
and in coverage, and never pass. Tier rules are not applied: a grader touches no
research gate tier.

``metrics.json`` (``strata.eval.grades/v1``) records the rubric (with its SHA-256
and the answer schema's), the judge (requested revision, the commit served, its
settings and serving facts), each output's pins, join and counts, and per slice
(overall first): per system ``n``, ``graded``, ``coverage``, ``gate`` (``resolved``
only at full coverage), unscored counts, and per criterion the cluster-bootstrap
``mean`` (the pass rate on a pass/fail scale) and, with ``pass_at``, ``pass_rate``
(both null with ``reason: "no_graded_cases"`` when nothing was graded);
per baseline the paired candidate-minus-baseline ``delta`` on the cases both had
graded (``excludes_zero``, ``unresolved`` without bounds; null with
``reason: "fewer_than_two_shared_cases"``), with ``coverage``, ``gate`` and pass/fail
``transitions``. ``agreement`` holds, per system and
criterion with a ``gold_column``, ``judge_agreement`` over the cases with both a
gold and a judge score, plus ``n_gold``, ``coverage`` and cluster-bootstrap
intervals of exact agreement and (ordinal) absolute error. ``items.parquet``
holds one row per case: ``item_id``, ``cluster`` (null without a cluster
column), ``slice__<column>`` and per system ``status__<s>``,
``score__<criterion>__<s>``, ``gold__<criterion>__<s>`` and ``judge_retried__<s>``;
it holds no input, output or rationale text. ``rationales.parquet`` holds
``item_id``, ``system`` and the judge's ``rationale`` per graded case, capped at
2,000 characters (a rubric may allow at most that much). **A rationale may quote
the output it grades**, which is why it is kept apart: it is read by project
members, and never reaches an agent beyond a capped excerpt. With the manifest
and a card they are pushed in one commit to a private output repo; the ``end`` event's message is ``<repo>@<commit>`` and its metrics
(at most 64) are ``cases``, ``graded``, ``coverage``, each criterion's candidate
mean (bounds only at full coverage; ``<criterion>_unresolved: 1`` below it),
``delta_<criterion>__<baseline>`` (with ``delta_<criterion>_unresolved__<baseline>: 1``
below full coverage) and ``agreement_<criterion>``. The machine needs
``strata-forge[serving,scoring,storage,hf]`` and a GPU for the judge.

### Fine-tuning pins, checks and manifest

``FinetuneSpec.model_revision`` pins the base model to a full 40-character
lowercase commit id; the model, its tokenizer and the merge step read only
that commit, and a run whose base loaded at another commit is refused. A
branch, tag or abbreviated id is refused at spec load, like an unpinned
``dataset_commit_sha`` of the wrong shape. Left unset (allowed for the
causal-LM methods), the run reads the default branch and records the commit
it got. ``seq_cls`` requires both pins: it refuses a spec missing either.

``check_spec(spec)`` runs every check that needs neither the machine nor the
network, including the method's own config over ``hyperparams``, so a control
plane can refuse at submit, against the engine it bundles, exactly what the
runner would refuse on the machine.

A ``seq_cls`` run downloads the pinned snapshot (safetensors, configs and
tokenizer files only) and refuses one whose commit is not the pin; reads each
split as Arrow, projected onto the ``nli_pair`` roles (``row_limit`` takes a
prefix), never as a Python list; and runs the classifier's checks before step 0
(length, load guard, decoder step-0, memory; see
[training](training.md#checks-before-step-0)). Each passed check is a
``phase`` event at stage ``load_model`` whose ``message`` is a sentence and
whose ``metrics`` carry the numbers (``check_rows``,
``check_longest_tokens``, ``check_template_token_overhead``,
``check_rows_over_max_length``, ``check_length_min`` / ``_p50`` / ``_p95`` /
``_max``, ``step0_max_abs_logit_diff``, ``step0_tolerance``,
``step0_argmax_agree``, ``preflight_peak_bytes``,
``preflight_eval_peak_bytes``, ``preflight_total_bytes``,
``preflight_gradient_checkpointing``, ...). A refused check fails the run
with its measurement in the message (the memory refusal also names the
flag and the base commit) and leaves a ``manifest.json`` with ``outcome``
``refused`` in the run directory on the machine.

Every fine-tuning run writes ``manifest.json`` (schema
``strata.run.manifest/v1``, ``kind`` ``finetune``) into the artifact
directory, so the push carries the weights and their record in one Hub
commit and the copy on the machine carries it too. It records the outcome;
the method and its ``task_type``; the model id, requested revision and
commit; the dataset id, config, splits, requested revision, format, column
mapping and row limit; the adapter; the validated hyperparameters (without
the runner-owned fields); train and eval row counts; the trainer's steps and
metrics; the engine and the installed versions of ``torch``,
``transformers``, ``trl``, ``peft``, ``accelerate``, ``datasets``,
``tokenizers`` and ``safetensors``; for ``seq_cls``, every check's receipt;
and ``compute``: the device, the GPU count, ``wall_seconds``, per-phase
``timing`` and ``gpu_seconds``. A ``seq_cls`` run's ``checks`` also hold the loss form and
weights, the reload-parity receipt, with an eval split the in-training ``monitoring`` record (every eval, labelled monitoring only; see [training](training.md#in-training-monitoring)) and, with ``calibration_split`` declared, the calibration
record also written to ``calibration.json``. ``gpu_seconds`` is the runner's own measured
wall-clock time, from the spec check to the manifest, times the GPUs it used
(one for ``seq_cls``); provisioning and the engine install happen before the
runner exists and are the control plane's to measure. The terminal ``end``
event carries ``gpu_seconds`` and ``gpu_count`` in its metrics too, so a control plane can book
its own total GPU-held time against the same count; so does the terminal ``error`` event of a
run that ends any other way (a refused check, a failed push, a crash). The manifest holds no row
content and no credential.

#### ``calibration.json``

One contract, written by a ``seq_cls`` run with ``calibration_split`` and read by every
evaluator and scorer that applies a temperature; a reader needs this file alone, never
``strata_verifier.json``:

- ``schema``: ``"strata.calibration/v1"``.
- ``labels``: the ordered output labels the temperature applies to, one per logit, in logit
  order.
- ``fit``: the fit, with ``fit.temperature`` (positive; calibrated probabilities are
  ``softmax(logits / temperature)`` over ``labels``), ``fit_version``, ``n``, ``nll_before``,
  ``nll_after`` and ``at_bound``.
- ``split``: the dataset id, config, split and dataset commit the temperature was fitted on.
- ``rows`` and ``excluded_rows``: the fitted single-label three-way rows, and the rest.
- ``rows_sha256``: SHA-256 over one JSON line ``[input_ids, label]`` per fitted row, in split
  order.
- ``logits_from``: which logits were fitted (the saved artifact, float32, TF32 off).

A reader applies ``fit.temperature`` to the logits of exactly ``labels`` and keeps the raw logits.

A pushed fine-tune goes only to a **private** model repo, checked before any
download or GPU time and again at push time, as for batch inference. The push
never uploads ``checkpoint-*``: a ``seq_cls`` run writes its intermediate
checkpoints beside the artifact, and the causal-LM methods' are excluded from
the upload by name.

### Latency benchmark (`benchmark_runner`)

``python -m strata_forge.pipelines.benchmark_runner`` measures one pinned
model on one pinned workload split. It is a bounded measurement job, not a
serving deployment: nothing it reports is an online-serving latency or a
production target. The spec (``BenchmarkSpec``) has the batch spec's shape
with both pins required (``model_revision`` and ``dataset_commit_sha``, full
40-character commits) and ``hyperparams`` a ``BenchmarkHyperparams``:

| Field | Meaning |
|---|---|
| ``engine_settings`` | Required. A batch run's own ``Hyperparams``, whose ``engine`` picks the engine: ``hf_classify`` or ``vllm_chat`` (``max_tokens`` required: a generation's length decides its latency). ``hf_classify`` with ``artifact: strata_verifier`` times a verifier trained by ``seq_cls`` through its own rendering and heads (the **headline path**, its templates fixed to ``{premises}`` / ``{hypothesis}``; ``apply_calibration`` applies); without ``artifact`` it is the plain AutoModel path, recorded as a **third-party reference model**. An allow-list per engine: a knob the benchmark does not apply (``row_limit``, ``scoring_batch_size``, ``slice_column``, ``n``, ...) is refused. |
| ``boundaries`` | ``end_to_end`` (required: ``total_ms``, the headline) and optionally ``model_only`` (``forward_ms``, a diagnostic). ``vllm_chat`` measures ``end_to_end`` only. |
| ``batch_sizes`` / ``concurrency`` | The grid; their product is the configurations, at most 32. ``hf_classify`` runs one request stream (``concurrency`` ``[1]``); ``vllm_chat`` sends one request per iteration (``batch_sizes`` ``[1]``). |
| ``warmup`` | Requests (full batches above batch size one) run untimed before EVERY timed pass, on the same workload, with no restart and no cache clear, drawn with ``random.Random(warmup_seed + pass_index)``. Default 50. |
| ``passes`` | Timed passes per configuration, each over the whole sample in the order ``random.Random(order_seed + pass_index)`` shuffles it, never length-sorted. Default 5. |
| ``item_cap`` | Every row of the pinned split is read; a split with more rows than this is refused (read up to one past it), never truncated. Default 100,000. |
| ``sample_size`` / ``sample_seed`` | Optional ``random.Random(sample_seed).sample`` of the eligible rows, in row order; unset, every eligible row. |
| ``order_seed`` / ``warmup_seed`` | Defaults 1000 and 2000; at least ``passes`` apart, so a warmup never replays another pass's order. All three seeds are in the manifest. |
| ``wall_clock_cap_s`` | A hard stop on measuring, checked before each iteration starts. Reaching it marks the results incomplete and every unstarted configuration ``not_measured``. Default 1800, at most 6 hours. |
| ``max_input_tokens`` / ``item_count_column`` / ``max_items`` | Exclusions (the protocol's are 8,192 tokens and 20 premises), counted by reason (``unmeasurable``, ``over_model_limit``, ``over_max_input_tokens``, ``over_max_items``) and never timed. Lengths are measured at run time with this model's own tokenizer and template: the verifier's own rendering, the pair with special tokens for the reference path, the prompt after the chat template for ``vllm_chat``. |
| ``length_buckets`` / ``item_buckets`` | Inclusive ascending upper edges (``[512, 2048, 4096, 8192]`` gives ``0-512``, ``513-2048``, ...). At a batch size of one each bucket (L by tokens, P by items) gets its own percentiles. |
| ``bootstrap_resamples`` / ``bootstrap_seed`` | Optional, off by default, diagnostic only: percentile-bootstrap intervals for the pooled percentiles (``metrics.latency_bootstrap``). Never the headline and never a pass/fail input. |
| ``prefix_caching`` | ``vllm_chat`` only, default ``false``: the model server is launched with ``--no-enable-prefix-caching``, since every pass replays the same prompts and a cached prefill would understate the time. Recorded in the engine facts. |
| ``require_gpu`` | Optional. Refuse to measure unless every visible GPU's reported name contains it (``H100``). |

**Timing boundaries.** ``total_ms`` (the headline) is read from
``time.perf_counter_ns``. On the verifier path it runs from the raw
``(premises, hypothesis)`` cells in memory through the verifier's own
rendering, tokenization, the host-to-device copy, the forward pass and the
auxiliary heads, the copies back to the host, the 3-way softmax, the flag
head's sigmoid (``p_inconsistent``) and the support head's per-premise
sigmoids (``support_scores``), and stops after ONE
``torch.cuda.synchronize()``. On the reference path it runs from the text
pair through tokenization, the forward pass and a softmax to the
probabilities on the host, and stops after the same single synchronization.
``forward_ms`` (diagnostic) is two CUDA events recorded around the forward
pass in the same request and read after that synchronization; no
synchronization is added to take it (on a CPU, ``perf_counter_ns`` around
the forward pass). The dtype (bf16 accepted) and the attention
implementation are recorded. A batch above one is consecutive rows of the
order, padded to its longest pair; the rows that do not fill a last batch are
counted, not timed. With ``vllm_chat`` the model is served on loopback, and
end to end is one chat request: the HTTP request, templating, tokenization,
queueing, prefill and decode, and the response; the client has no cache and
makes one attempt. An iteration that raises stops its configuration, with its
exception text redacted.

**The GPU.** Before anything is loaded, ``nvidia-smi`` is asked for compute
processes on the GPUs ``CUDA_VISIBLE_DEVICES`` exposes, other than this
runner and its model server: one refuses the run. The same check runs before
every pass's warmup; a process found then stops the run as incomplete,
``gpu_busy``. Clocks are never locked: the SM and memory clocks, power limit
and draw, temperature and persistence mode are recorded at the start and end
of every pass (``gpu_telemetry``). The probes run between passes, never
while an iteration is timed. Without ``nvidia-smi`` the check is
``unavailable`` and the headline is not readable.

**Statistics.** Only completed passes are summarized: a pass cut short (a
failure, the cap, a busy GPU) stays in ``timings.parquet`` with
``pass_completed`` false and never enters a percentile, the across-pass
median, the throughput or a bucket. Per configuration and boundary: per-pass
p50/p95/p99 (Hyndman-Fan type 7, equal to ``numpy.percentile``), their median
across passes with min and max, and the pooled percentiles; throughput is
items (and tokens) per second over the union of the measured intervals.
Latency is per request at a batch size of one and per batch above it. The
**headline** of each configuration is the median of the per-pass end-to-end
p95 with their minimum and maximum (``headline``). ``headline_readable`` is
true only for a complete run whose GPU was checked idle throughout: an
incomplete run has no T13 read, and ``headline_unreadable_reason`` says why.
Every summary number is finite: NaN and infinity are refused.

**Outputs.** ``timings.parquet`` (one row per measured iteration: the
configuration, pass and ``pass_completed``, the iteration, the row indices,
token and item counts, buckets, ``total_ms``, ``forward_ms`` and output
tokens; never row text), ``summary.json`` (``BenchmarkSummary``, ``format``
``strata.benchmark.summary/v1``: completeness, its reason and
``incomplete_codes`` (``gpu_busy``, ``wall_clock_cap``,
``stopped_on_failure``, ``server_died``, ``not_measured``), the headline
readability, the model path, the GPU idle check and telemetry, the sample and
exclusion counts, the cold start, a short GPU receipt and every
configuration's result) and ``manifest.json`` (``kind`` ``benchmark``: both
pins and commits, the protocol in words with its seeds, the engine facts, the
full environment receipt, the counts and ``compute``). They are written
outside the workdir under ``~/strata-benchmark-results/<run_id>`` and pushed
in ONE commit to a private output repo that holds nothing else; with no write
token they stay on the machine. A copy of the summary is left in the workdir
as ``benchmark-summary.json`` for the orchestrator to read. If the model
server dies mid-run, the measured configurations are still written and pushed
(the rest ``not_measured``) and the run then fails as incomplete.

**The environment receipt** is what the machine reports, never what the run
asked for: per visible GPU from ``nvidia-smi`` (index, name, total memory,
driver, persistence mode, current and maximum SM clock, power limit, PCI bus
id) and how many compute processes held one before the model loaded; from
torch, its version, CUDA runtime, cuDNN, and the device; the installed
versions of torch, transformers, tokenizers, safetensors, vLLM, NumPy,
datasets and huggingface-hub; and ``CUDA_VISIBLE_DEVICES``. A field the
machine could not report is ``null``. The cold start is recorded separately
from every percentile.

The ``end`` event's metrics are ``iterations_measured``,
``configurations_measured``, ``configurations``, ``complete`` and
``headline_readable`` (1 or 0), ``gpu_count`` and ``gpu_seconds`` (this
process's wall time times the visible GPUs). A run that measured no
iteration at all fails after writing its files.

### The runner contract

An orchestrator that launches a ``strata_forge.pipelines`` runner on a
machine it does not own holds up its side of the contract in five places:

- **The spec is inert data** in ``STRATA_RUN_CONFIG``: JSON validated into
  a Pydantic model with ``extra="forbid"``, so a key the installed engine
  does not know is a named failure, never an ignored instruction. Nothing
  in a spec names code to run. A fine-tuning spec's
  ``hyperparams.extra_trainer_args`` (TRL's own argument surface, passed
  through verbatim) is refused by name when it sets a knob that would act
  under terms other than the run's: ``chat_template_path``,
  ``trust_remote_code``, ``model_init_kwargs``, ``push_to_hub`` and every
  ``hub_*`` / ``trackio_*`` setting (Hub reads and pushes with the
  machine's own credential, or a repo's code), ``resume_from_checkpoint``,
  ``deepspeed``, ``fsdp_config``, ``accelerator_config`` (files on the
  machine), ``logging_dir`` and ``report_to``. A caller driving
  :mod:`strata_forge.training` from Python keeps all of them.
- **The Hugging Face token travels apart from the spec**, as
  ``Task.secrets={"HF_TOKEN": SecretStr(token)}``, which the backend writes
  to the private file named by ``FORGE_SECRETS_FILE`` (see
  [Secrets](#secrets)). ``runner_main`` calls
  ``strata_forge.pipelines._common.load_secrets()`` before anything else:
  it refuses a path that is not an absolute ``.../.secrets.json`` without
  touching it, opens the file without following a symlink, requires a
  regular file owned by the runner's user with no group or other bits,
  reads at most 64 KiB, and unlinks it whatever the outcome, so the file is
  gone before the runner makes a network call or starts a subprocess. The
  file is a JSON object; ``HF_TOKEN`` is the only key a runner reads, and
  any other key is refused (counted, never named). The key is exported
  as ``strata_forge.pipelines.HF_TOKEN_SECRET`` for orchestrators. The
  open uses ``O_NONBLOCK``, so a FIFO at the path is refused rather than
  waited on. A missing, unreadable, misowned or
  malformed file fails the run with a named error that never carries the
  file's contents. The values reach the runner as ``SecretStr`` in a
  ``RunSecrets`` (``hf_token``), the callable ``runner_main`` drives takes
  ``(writer, secrets)``, and the runner scrubs the token (and anything
  token-shaped) from every message it emits. The model server the
  inference runner starts gets an allow-listed environment
  (``model_server_environ()``: ``PATH``, ``HOME``, ``USER``, ``LOGNAME``,
  ``LANG``, ``LC_ALL``, ``LC_CTYPE``, ``TMPDIR``, ``LD_LIBRARY_PATH``,
  ``XDG_CACHE_HOME``, ``HF_HOME``, ``HF_HUB_CACHE``,
  ``TRANSFORMERS_CACHE``, ``PYTHONUNBUFFERED`` and the ``CUDA_*`` /
  ``NVIDIA_*`` / ``NCCL_*`` / ``VLLM_*`` families, minus any name that
  says it holds a credential) and does not inherit the runner's own. That
  also keeps an ambient ``HF_TOKEN``, ``HF_ENDPOINT`` or ``HF_HUB_*``
  setting from reaching the server.
  Transition: with no ``FORGE_SECRETS_FILE`` set, a 0.4 runner still reads
  the token from ``HF_WRITE_TOKEN``, removes it from its own environment,
  and records a ``phase`` event (and a stderr ``warning:``) saying that
  delivery is deprecated; with a file set, the variable is never read.
  The fallback is removed in 0.5.0.
- **Hub reads use the delivered token, or none.** The same ``HF_TOKEN`` is
  the run's credential for every Hub read as well as the push: the dataset
  split, the training base model and tokenizer (and, for DPO / KTO without
  an adapter, the reference model), the merge step's base, and the
  inference runner's model download. Each read passes
  ``RunSecrets.hub_credential()`` explicitly: the token, or ``False`` when
  none was delivered, which sends no credential. Never ``None``, which every
  Hugging Face library resolves to whatever the machine holds (an
  ``HF_TOKEN`` variable, a cached login, forge's own settings), so a run
  authenticates with exactly the credential the orchestrator decided on.
  ``False`` does not reach the machine's Hub cache: ``huggingface_hub``
  answers a refused (401 / gated) request from a snapshot it already holds,
  and ``datasets`` from its own cache, so a gated or private repo that an
  earlier run on the same account cached still loads without a token. That
  is an accepted residual: the cache belongs to the account the runner
  runs as, which can read it directly anyway. An
  orchestrator therefore delivers the token to any run that reads a gated
  or private model or dataset, not only to one that pushes; the push stays
  gated on ``output_repo_id`` alone. Every model load sets
  ``trust_remote_code=False`` and ``use_safetensors=True``, and every
  tokenizer load ``trust_remote_code=False`` (see
  [training: Loading from the Hub](training.md#loading-from-the-hub)).
  The inference runner downloads the model **itself, before** the model
  server starts: ``HFHubClient(token=...).download_snapshot(model_id,
  allow_patterns=SNAPSHOT_PATTERNS, ignore_patterns=SNAPSHOT_IGNORED)``
  fetches safetensors weights, JSON configs and the tokenizer and
  chat-template files, and never a pickle checkpoint (``*.bin``, ``*.pt``,
  ``*.pth``, ``*.pkl``, ``*.ckpt``, or an array format that can carry one),
  a ``*.py`` file, compiled module or shell script, or an ``original/``
  directory. A snapshot with no ``*.safetensors`` file at its root (where
  the server reads weights) fails the run by name before the server
  starts. The server is then started on the local snapshot
  (``--model <snapshot> --served-model-name <model_id> --load-format
  safetensors``) with ``HF_HUB_OFFLINE=1``, so it never holds the token
  and fetches nothing of its own. The snapshot lands in the Hub cache
  (``HF_HOME``), so a warm machine reuses it; the directory is the cache's
  shared folder for that commit, which can also hold files an earlier,
  unfiltered download left there, and ``--load-format safetensors`` is what
  keeps the server off a pickle checkpoint among them (vLLM's default
  falls back to one). ``LocalBackend`` starts the server through a login
  shell (``bash -lc``), which re-sources the machine's profile after the
  allow-listed environment is applied: a profile that exports
  ``HF_TOKEN`` or ``HF_HUB_OFFLINE`` puts it back into the server's
  environment. The delivered token is never affected; an operator
  verifying the server's environment checks the profile as well.
- **Progress is one protocol.** Both runners append the same
  ``ProgressEvent`` stream to ``FORGE_PROGRESS_PATH``, and the exit code is
  the run's verdict.
- **The spec names the engine that validated it.** Every runner spec
  carries ``engine_version: str | None``, which the orchestrator sets to
  ``strata_forge.pipelines.SPEC_VERSION`` (the package version) — or to
  ``f"{SPEC_VERSION}+{commit}"`` when it installs the engine from a git ref
  rather than a release, ``commit`` being the full id of the commit its own
  bundled engine was built from. ``load_config`` compares the version half
  with the installed ``strata_forge.__version__`` and the commit half with
  the installed distribution's PEP 610 ``direct_url.json`` commit, and
  refuses either disagreement with ``engine version mismatch``. The check
  runs on the parsed JSON before the model validates anything else, so a
  spec that carries both a field this engine does not know and a version
  it does not match reports the mismatch, not the unknown field: the run
  record blames the stale machine, and the run exits 1 with that as its
  reason. The version half is also compared with the version the
  installed distribution's metadata records (what the pin resolved
  against), so a shadowed import or a version-string drift between
  ``pyproject.toml`` and ``__init__.py`` is a mismatch too. A ``+`` with
  anything but a full lowercase commit id after it is a malformed claim
  and is refused, never read as version-only. An engine with no recorded
  commit (a release from PyPI, an editable checkout) cannot satisfy a
  commit claim. The compare is string equality, never a PEP 440
  normalisation: ``0.3``, ``v0.3.0`` and ``0.3.0.post0`` are not ``0.3.0``.
  ``None`` makes no claim and is accepted, as the transition for an
  orchestrator from before the handshake, but never silently: the run's
  first ``phase`` event (and a stderr ``warning:`` line) says the spec
  carried no ``engine_version`` and the installed engine was not checked,
  so the record of an unchecked launch says so. An orchestrator that stamps
  every spec ends the transition on its own machines by setting
  ``FORGE_REQUIRE_ENGINE_VERSION=1`` in the runner's environment
  (``strata_forge.pipelines._common.REQUIRE_ENGINE_VERSION_ENV``): a
  missing claim is then refused like any other mismatch, which is what
  stops a rolled-back control plane that sends ``null`` from running a
  newer engine unchecked. Every runner spec declares the field:
  ``load_config`` raises ``TypeError`` for a spec class that does not, so a
  runner cannot opt out by omission.

The handshake exists for a warm machine that still runs an OLDER engine
than the one that validated the spec. ``extra="forbid"`` catches that only
when the newer spec carries a field the old engine does not know; a
behaviour change on the same spec shape (where the write token travels,
what the scrubber removes, a default) reaches the old engine with no spec
error at all, and the run fails, or silently does the old thing, long after
launch. The mirror case, a machine whose engine is NEWER than the one that
validated the spec (a rolled-back control plane, a spec replayed after an
upgrade), is the same refusal: the spec names one engine and runs under no
other. And every commit of a development branch shares one
``__version__`` until a release bump, so the commit half is what lets a
deployment that installs from a branch see that the machine runs an older
commit than the one that validated the spec. The string is the
orchestrator's, never a user's: the same value decides what the setup step
installs, so a spec cannot pick an engine the orchestrator did not validate
against. What the orchestrator must do with it: pin the install to exactly
that version (an exact ``==`` also upgrades a warm machine, because the
older copy no longer satisfies the requirement) or, for a git ref, to
exactly that commit, and force a reinstall only when the machine's recorded
commit differs.

## Backend protocol

```python
class Backend(Protocol):
    name: str
    async def submit(self, task: Task) -> Job: ...      # task.secrets -> a 0600 file, or raise
    # A submit that fails and cannot remove what it created raises SubmitCleanupError,
    # whose .job only cleanup() accepts.
    async def status(self, job: Job) -> JobStatus: ...
    async def logs(self, job: Job, *, tail: int | None = None) -> str: ...
    async def read_file(self, job: Job, path: str, *, tail: int | None = None) -> str: ...
    async def console(
        self,
        job: Job,
        *,
        stdout_offset: int = 0,
        stderr_offset: int = 0,
        max_bytes: int = MAX_CONSOLE_CHUNK_BYTES,
    ) -> ConsoleChunk: ...
    async def cancel(self, job: Job) -> None: ...
    # Returns once the job's state is gone; raises CleanupError if it survived.
    async def cleanup(self, job: Job) -> None: ...
```

### Cleanup is verified

``cleanup`` returning means the job's backend-side state is gone, not
that a removal was attempted. That state may still hold the job's
secrets file (a SIGKILL skips the wrapper's trap, and a failed submit
may never have launched the wrapper), so an orchestrator retries
cleanup until it succeeds, and it can only do that if a failed one
says so. The contract every backend keeps:

- **State that survived raises** :class:`CleanupError` (a
  :class:`ForgeError`, and an ``OSError`` so that a caller already
  treating an ``OSError`` from ``cleanup`` as "not cleaned" catches
  it). Its message names the job and how the removal failed (an exit
  status, an exception type), never what the remote printed or what
  the state contains, so it is safe to store and show.
- **State that is already absent is a success**, so the method is
  idempotent: a retry after a removal that did, in the end, happen
  returns normally.
- **A transport failure propagates as itself** (the host is
  unreachable, the command timed out). It says nothing about whether
  the state survived, so it is not reported as a cleanup that ran.

```python
from strata_forge.compute import CleanupError

try:
    await backend.cleanup(job)
except CleanupError:
    ...  # still there: keep the handle and try again later
```

### Reading the console

``logs`` answers "what has this job printed?" — the right question after a job
ends. A watcher following a *live* job asks "what has it printed since last
time?", and a tail cannot answer that: consecutive windows overlap by an unknown
amount, and de-duplicating by matching the last line seen fails on precisely the
output that makes a console worth watching, because a progress bar rewriting
itself emits the same line over and over.

``console`` answers it with byte offsets. Pass back the offsets from the previous
:class:`ConsoleChunk` (zero the first time) and receive exactly what was appended
since:

```python
chunk = ConsoleChunk()  # offsets 0, so the first read is rendered like every other
while not done:
    chunk = await backend.console(
        job, stdout_offset=chunk.stdout_offset, stderr_offset=chunk.stderr_offset
    )
    render(chunk.stdout, chunk.stderr)
    if chunk.dropped_bytes:
        render(f"[{chunk.dropped_bytes} bytes not shown]")
```

``max_bytes`` is the TOTAL for the call, split evenly between the two streams and
clamped to ``MAX_CONSOLE_CHUNK_BYTES``. When more accumulated between calls the
NEWEST bytes are kept and the shortfall is reported as ``dropped_bytes`` — a
watcher wants where the run is now, and a silent jump-cut would read as a whole
transcript. Offsets are byte offsets into the underlying stream, so a slice may
cut a multi-byte character; the boundary decodes to a replacement character
rather than raising.

The SSH backend does this in ONE round trip. It measures both files and computes
the slice lengths in the same remote shell, anchors each slice to its START
(``tail -c N`` counts back from the end of a file the job is still writing, so an
end-anchored slice would not match the header it arrived with), and base64-frames
the payloads so byte counts survive the transport. The reply is read under a
bound of the caller's own — the remote-side cap lives in a shell on a machine its
owner controls — and every field of it is validated before use: a header is a
claim, not a measurement. Anything that fails those checks costs one poll and
leaves the offsets untouched, because advancing past bytes that never arrived
loses them for good.

SkyPilot raises ``NotImplementedError``: ``sky logs`` has no way to ask for a
suffix, and re-fetching the whole log per poll is linear in memory as well as in
time inside a process shared by every account.

### Redacting the console

``console``, ``logs`` and ``read_file`` return what the job wrote, unredacted:
a backend cannot know which secrets its caller injected. Whoever relays that
text redacts it, with one :class:`~strata_forge.core.redact.Redactor` built
from every secret the job was given plus the default credential shapes, and
with the stream API rather than per-chunk calls, because a secret can straddle
two incremental reads:

```python
from strata_forge.compute import ConsoleChunk
from strata_forge.core import Redactor

# Drop absent optional secrets; a ValidationError (a value too short to match
# safely) means do not relay this console at all, never relay it raw.
redactor = Redactor(v for v in (hf_token, private_key, known_hosts) if v)
out, err = redactor.stream(), redactor.stream()  # one per stream, kept across polls
chunk = ConsoleChunk()  # offsets 0: the first read goes through the streams too
while not done:
    chunk = await backend.console(
        job, stdout_offset=chunk.stdout_offset, stderr_offset=chunk.stderr_offset
    )
    if chunk.dropped_bytes:
        render(out.gap(), err.gap())  # a hole: neither side of a cut secret survives
    render(out.feed(chunk.stdout), err.feed(chunk.stderr))
render(out.flush(), err.flush())  # once the job is terminal, and only then
```

The contract a relay relies on:

- **A stream holds back ``max_len - 1`` characters** (a few hundred), plus up
  to 8 characters of whitespace right after a redacted run, and releases them
  on ``flush``. Its concatenated output equals redacting the whole transcript
  at once, so piece boundaries never decide what is released.
- **One stream per console stream for the life of the job, in memory.** A
  stream is process state: it cannot be persisted, pickled or copied (it refuses,
  because what it holds back is raw text). A relay whose polls are otherwise
  stateless (the read offsets stored on the run's row, any worker taking the
  next poll) keeps the streams in a per-run registry in the process that polls.
  A fresh stream per poll is a per-chunk redactor again: the tail of one poll
  can end inside a token, and a fresh stream cannot know it.
- **A fresh stream in the middle of a transcript calls ``gap()`` before its
  first ``feed``.** That covers a relay restart, a lease taken over by another
  worker, and any resume from stored offsets. The new stream cannot see what
  the old one held back or which match was still open, so it treats the resume
  point as a hole and masks the first ``max_len - 1`` characters after it. A
  private-key block opened before the resume point is unknown to it; a key the
  relay itself injected is still matched by its own lines, because every value
  is matched line by line.
- **``flush`` only when the job is terminal.** Never per poll and never on an
  idle timeout: a flush releases the held tail as final, so a secret that
  continues in the next read is released half raw.
- **The held tail of a stream that dies is lost, not leaked.** A relay that
  stores the read offsets has already advanced past text its stream never
  released, so those characters (at most the hold-back) are never shown. This
  is the safe direction; a relay that wants them has to persist and re-read
  from an earlier offset, and still call ``gap()`` on its fresh stream.
- **A hole needs ``gap()``.** ``dropped_bytes`` counts bytes from either stream,
  so call it on both. It masks the held tail and the first ``max_len - 1``
  characters after the hole, and keeps a private-key block that was open across
  it open.
- **Redact before clipping**, and before parsing a structured line: a clip can
  cut a token below the length a pattern recognises, and a JSON string escapes a
  multi-line secret (the redactor matches the escaped form too).
- **Decoding.** A slice boundary can split a multi-byte character, which decodes
  to a replacement character on each side. Every credential shape the redactor
  knows is ASCII, so this cannot split one of those; a non-ASCII secret value
  split that way is not matched.
- **The relay is the only redaction of everything else the job prints.** A
  runner redacts what it emits itself (phase captions, error events, its
  failure line on stderr, the ``error`` column), but library logging, warnings
  and tracebacks from other threads reach the console file as they were
  written. The console relay is the layer that catches those.

Methods that don't apply to a particular backend raise
:class:`NotImplementedError` rather than silently passing — that
way callers can ``try/except`` if needed instead of relying on
backend-specific knowledge of which methods are no-ops.

## LocalBackend

```python
from strata_forge.compute import LocalBackend, Task

backend = LocalBackend()
job = await backend.submit(Task(name="t", run="python -m my_script"))
```

The local backend spawns each task as an async subprocess
(``bash -lc``) and tracks them by job id. It runs in
``Task.workdir`` when one is set, and touches no filesystem of
its own by default.

Both pipes are drained continuously rather than read to EOF, so
``logs`` returns partial output **while the job is still
running** — the case worth diagnosing is a process that came up
wrong and then hung, and its output exists long before it exits.
The in-memory buffer keeps the last 1 MiB per stream.

```python
backend = LocalBackend(log_dir="./job-logs")
```

``log_dir`` additionally tees both streams to
``serve.stdout.log`` / ``serve.stderr.log`` under that directory,
written as the bytes arrive. Those files are deliberately left in
place by ``cleanup``: once the process and its buffers are gone,
the file is the only remaining evidence. It is opt-in, so a
caller who never asks for it never finds log files appearing. The
names are fixed (an orchestrator finds them without knowing the
job id), so give concurrent jobs their own directories.

A task's ``secrets`` are written to ``.secrets.json`` (0600, created
``O_EXCL | O_NOFOLLOW``) in a fresh 0700 temporary directory, and the
task's ``run`` step finds it through ``FORGE_SECRETS_FILE`` (``setup``
does not). Modes are set explicitly, so a restrictive umask cannot
lock the child out. The child's outer shell removes the file on exit,
below the subshell that runs the task; the backend removes the
directory once the child is gone, and again in ``cleanup``, which
forgets the job only once ``lstat`` reports nothing at the directory's
path: a directory that survives, or one that cannot be looked at (its
parent is not searchable), raises ``CleanupError`` and the job stays
known, so a retry still has something that names it. With
``env_inherit=True`` an inherited ``FORGE_SECRETS_FILE`` is dropped:
it names the parent's file, which the child must not read (and, by
reading, delete).

A job ends when the child is reaped, not when its pipes close: a
process the child backgrounded inherits those descriptors and can
hold them open indefinitely, and gating the lifecycle on EOF
would leave such a job stuck at ``running``. The drain gets a few
seconds after the exit to finish reading.

## SSHBackend

```python
from strata_forge.compute import SSHBackend, Task

backend = SSHBackend(host="gpu-host.example.com", username="ml-team")
job = await backend.submit(Task(name="t", run="python train.py"))
```

Submission creates a private workdir on the remote host
(``~/.forge-compute/<job_id>/``), writes a wrapper into it and
launches it under ``nohup``, capturing the PID and exit code in
files. ``status`` probes ``kill -0`` for liveness, then falls back
to the exit-code file. ``cancel`` sends SIGTERM to the job's process
group, gives it up to 10 s, then SIGKILL.

The workdir is made by one ``bash -c`` step under ``umask 077``: the
remote root must be a directory owned by the login user and is
narrowed to 0700, and the job's directory is created with
``mkdir -m 700`` and no ``-p``, so a directory or symlink already at
that path fails the submit. File contents never travel in a command
string, which every user on the host can read in
``/proc/<pid>/cmdline``: the wrapper and then, when the task has
secrets, ``.secrets.json`` are written with their bytes on the
channel's stdin. Each write runs under ``umask 077``, ``cd -P`` into
the workdir and checks ``test -O .`` (so the path is resolved once and
a swapped component cannot redirect it), refuses anything already at
the name, and creates the file under ``set -C``. The explicit refusal
matters: bash's noclobber still opens an existing FIFO or device for
writing. The secrets file is written last, immediately before the
launch.

With secrets, the wrapper holds the file's absolute path in an
unexported variable, installs ``trap 'rm -f "$_forge_secrets_file"'
EXIT`` in its own shell, and runs ``set -e``, the task's ``env``,
``setup`` and ``run`` in a subshell below that trap, exporting
``FORGE_SECRETS_FILE`` (the path, never the contents) only between
``setup`` and ``run``. The subshell keeps a ``trap ... EXIT`` that the
task sets (an orchestrator's bootstrap sets one) from replacing the
removal, so the file goes when setup fails, when the job ends and when
``cancel``'s SIGTERM arrives. Only a SIGKILL skips the trap; ``cancel``
removes the file itself after its final SIGKILL, and ``cleanup``
removes the whole workdir.

``cleanup`` enters ``remote_root`` with ``cd -P`` and runs ``rm -rf``
on the job's directory and ``test ! -e`` / ``test ! -L`` on its name
from there, so its exit status means "nothing is there now" rather
than "``rm`` did not complain". The checks run inside the root because
``test`` reads a lookup it is not allowed to make as "absent": run
from outside a root at mode 000, BSD ``rm -rf`` exits 0 and both tests
pass while the workdir and its secrets file stay. A root that cannot
be entered counts as clean only when the root itself is provably
absent, proved the same way one directory up; ``CDPATH`` is unset so a
relative root cannot resolve to a directory of the same name
elsewhere. A nonzero status, or a channel that closed with none,
raises ``CleanupError`` ("not confirmed removed") without echoing the
remote's stderr (which would name the files ``rm`` could not remove).
An absent workdir is a success. Every method that composes a remote
path from a job's ``remote_workdir`` first checks it is exactly one
directory directly in ``remote_root``, which is all ``submit`` creates:
an absolute path elsewhere, a ``..`` component, a path that names the
root itself (``<root>/``, ``<root>/.``, ``<root>//``), or a nested one
(``<root>/a/b``) is a ``ValueError``. ``rm -rf`` of the root would
remove every job's workdir, and a check run in the root proves nothing
about a directory deeper down.

A submit that fails after creating the workdir removes it, within a
10 s bound. If that removal fails too (the connection is gone), an
ordinary failure is re-raised as ``SubmitCleanupError`` (a
``RuntimeError``, from the original failure) whose ``job`` is a handle
``cleanup`` accepts; a cancellation stays a cancellation. As a backstop
that needs no handle, every submit's workdir step removes
``.secrets.json`` files older than 10 minutes from job directories
that have no pid file, i.e. submits that never launched. That sweep and
the ``chmod 700`` mean ``remote_root`` must be a directory dedicated to
Forge. A remote command that ends without an exit status counts as a
failure.

Pass a pre-built ``asyncssh.SSHClientConnection`` via
``connection=`` to share a connection across multiple submits.

## SkyPilotBackend

```python
from strata_forge.compute import SkyPilotBackend, Task, ResourceSpec

backend = SkyPilotBackend()
task = Task(
    name="train",
    run="python train.py",
    resources=ResourceSpec(accelerators="A100:8"),
)
job = await backend.submit(task)
```

The backend wraps SkyPilot's sync SDK in ``asyncio.to_thread`` so
the public surface stays async-uniform. It maps SkyPilot's twelve
job states to the canonical five; unknown states fall back to
``running`` so callers don't crash on new SkyPilot versions.

``cleanup`` calls ``sky.down`` on the cluster — be aware that
this tears down the entire cluster, not just the job. The
request-based SDK's ``down`` returns a request id and reports the
outcome only when that request is resolved, so ``cleanup`` resolves it
with ``get`` before returning. A failed teardown raises
``CleanupError`` (chained to the SDK's error, naming only its type);
SkyPilot's ``ClusterDoesNotExist`` counts as success.

``submit`` raises ``ValueError`` for a task with ``secrets``, before
any SkyPilot call: the SDK's only channel for a value is ``envs``,
which is the job's environment, so there is no delivery that keeps
the credential out of it.

## Batch inference

:class:`BatchInferenceRunner` runs many prompts through one
shared :class:`LLMClient` with a concurrency cap:

```python
runner = BatchInferenceRunner(client, concurrency=20, on_error="collect")
results = await runner.run(prompts, temperature=0.7, max_tokens=500)
```

Results are positionally aligned with the input prompts.
``on_error="raise"`` (default) cancels the batch on first
failure; ``on_error="collect"`` keeps going and stores the
exception in the corresponding result slot.

## Serving adapters

The serving helpers build a :class:`Task` whose ``run`` launches
an OpenAI-compatible inference server. Combined with the
``openai_compat`` provider in :mod:`strata_forge.llm`, you can talk to
self-hosted models with the same :class:`LLMClient` API you use
for SaaS providers.

```python
from strata_forge.compute.serving import build_vllm_task, build_tgi_task, build_sglang_task

vllm_task = build_vllm_task(
    "meta-llama/Llama-3.1-8B-Instruct",
    port=8000,
    tensor_parallel_size=1,
    max_model_len=8192,
)
tgi_task = build_tgi_task("mistralai/Mistral-7B-v0.1", port=8080)
sglang_task = build_sglang_task("Qwen/Qwen2-7B-Instruct", port=30000, tp_size=4)
```

``build_vllm_task`` also takes ``served_model_name``
(``--served-model-name``). Set it when ``model`` is a local snapshot
directory, so clients keep addressing the model by its Hub id rather
than by a path on the server; the batch-inference runner serves its
pre-fetched snapshot this way (see [the runner contract](#the-runner-contract)).

:func:`serving_endpoint` is an async context manager that
submits the task, waits for the HTTP endpoint to respond,
yields a :class:`ServingEndpoint`, and on exit cancels the job
and (optionally) calls ``backend.cleanup``.

```python
from strata_forge.compute import LocalBackend
from strata_forge.compute.serving import serving_endpoint

async with serving_endpoint(
    LocalBackend(), vllm_task, base_url="http://localhost:8000/v1"
) as endpoint:
    # endpoint.job is the live Job; endpoint.base_url is what
    # OpenAICompatConfig needs.
    ...
```

Every long uncountable phase re-stamps itself with its elapsed time every
10 seconds (``_PHASE_INTERVAL_SECONDS``), rendered by ``format_elapsed``
as ``45s`` -> ``1m 30s`` -> ``1h 01m`` — rolling into a larger unit while
keeping the smaller one, since both matter, and zero-padded so the caption
does not jitter as it is re-rendered in place. A one-shot phase says a step
BEGAN and never that it is still going, which is indistinguishable from a
run that has hung.

Readiness is verified once, before the endpoint is yielded, and then
nothing watches it again — but a model server can die at any point
AFTER it came up (an OOM on a long prompt, a CUDA fault). Every request
from then on fails against a socket nobody is listening on, and a client
that retries turns a dead server into a long, expensive silence rather
than an error. ``endpoint.is_alive()`` is the probe for that: a
long-running consumer should call it at a natural checkpoint (between
batches, not between requests — it costs a backend status probe) and
stop when it reports ``False``. A status the backend cannot report counts
as alive, so a flaky probe can never kill a healthy run.

The readiness probe hits ``{base_url}/models`` until it returns
HTTP 200 or the ``wait_timeout_s`` expires.

That wait is the longest thing the caller awaits, and by default
it is silent. Pass ``on_phase`` — a sink taking one short string
— to hear what is happening:

```python
async with serving_endpoint(
    LocalBackend(), vllm_task,
    base_url="http://localhost:8000/v1",
    on_phase=print,          # "Starting the model server"
    phase_interval_s=30.0,   # "Loading the model onto the GPU (90s)"
) as endpoint:               # "Model server ready"
    ...
```

``phase_interval_s`` throttles the readiness heartbeat, which is
otherwise emitted once per probe: an orchestrator persisting
these phrases has a finite budget per run, and a 2 s poll over a
30-minute wait would burn ~900 of them. The sink must not block,
and any exception it raises is swallowed — a broken sink must not
take down a live serving job.

The sink takes a plain ``str`` rather than a progress event
because :mod:`strata_forge.compute` must not import
:mod:`strata_forge.training`; the caller (typically a
:mod:`strata_forge.pipelines` runner) wraps the phrase into a
``ProgressEvent(kind="phase", ...)``. Phrases never interpolate
``base_url``, which is caller-supplied and may carry credentials.

## Lazy-import contract

- ``asyncssh`` and ``sky.api.sdk`` are behind the ``[compute]``
  extra. Both :class:`SSHBackend` and :class:`SkyPilotBackend`
  lazy-import them inside the constructor / first-use path and
  raise :class:`ImportError` with an install hint when missing.
- ``httpx`` is a core dep (via LiteLLM); :func:`wait_for_endpoint`
  imports it eagerly inside the function.
- :class:`LocalBackend`, :class:`BatchInferenceRunner`, and the
  serving task builders have no extra requirements.

## Troubleshooting

- **`SSHBackend` hangs on first submit:** typically the SSH
  connection wasn't established because of host-key validation.
  Either set ``known_hosts=path/to/known_hosts`` or pass an
  already-built ``asyncssh.SSHClientConnection`` via
  ``connection=``.
- **`SkyPilotBackend.status` returns "failed: not found":** the
  job ID is no longer in SkyPilot's queue. SkyPilot prunes old
  jobs aggressively — query status before too much time passes,
  or rely on stored log output via ``backend.logs``.
- **`serving_endpoint` times out:** the server is slow to come
  up. Increase ``wait_timeout_s`` (default 10 min); vLLM in
  particular can take several minutes for large models.
- **Server starts but `LLMClient` can't connect:** make sure
  ``OpenAICompatConfig.base_url`` includes the ``/v1`` suffix
  the provider expects.
