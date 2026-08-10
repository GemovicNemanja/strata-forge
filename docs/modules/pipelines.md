# `strata_forge.pipelines` — runnable entrypoints for orchestrated compute targets

`strata_forge.pipelines` is the one module in the library you do not import. Every other
`strata_forge.*` module is a set of primitives you call from your own process; a pipeline is a
**program** that an orchestrator starts on a machine it has provisioned. It takes no arguments,
reads an inert run spec from an environment variable, composes the library's serving, batch and
storage primitives into one long-running job, and appends structured progress records to a file
that the orchestrator reads back over its own channel.

That inversion is the whole point. `strata_forge.compute` can submit a shell command to a remote
host and poll it; what it cannot do is give that command a typed contract. A pipeline module is
that contract: a fixed entrypoint, a validated spec shape, a defined progress format, and defined
exit semantics — so the thing driving the job never has to parse log output to know what happened.

Integration points:

- **Entrypoint:** `python -m strata_forge.pipelines.inference_runner` — batch inference over a
  Hugging Face dataset split against a model served locally by vLLM.
- **Run spec:** `RunSpec` and `Hyperparams` — frozen-shaped Pydantic models with
  `extra="forbid"`, parsed from the `STRATA_RUN_CONFIG` environment variable as data.
- **Template rendering:** `render_template` — a bounded, non-executing `{name}` substitution.
- **Progress:** `ProgressEvent` and `JsonlProgressWriter`, reused verbatim from
  [`strata_forge.training`](training.md) so training and inference emit the same event stream.
- **Composed primitives:** `LocalBackend`, `build_vllm_task`, `serving_endpoint` and
  `BatchInferenceRunner` from [`strata_forge.compute`](compute.md); `HFHubClient` from
  [`strata_forge.storage`](storage.md); `LLMClient` on the `openai_compat` route from
  [`strata_forge.llm`](llm.md).

Module rules:
[`src/strata_forge/pipelines/CLAUDE.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/pipelines/CLAUDE.md).
Source:
[`src/strata_forge/pipelines/`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/pipelines/).

---

## Contents

- [Execution model](#execution-model)
- [Launching a run](#launching-a-run)
- [Environment variables](#environment-variables)
- [The run spec](#the-run-spec)
- [Template rendering](#template-rendering)
- [The progress protocol](#the-progress-protocol)
- [Results and where they land](#results-and-where-they-land)
- [Exit codes and failure semantics](#exit-codes-and-failure-semantics)
- [Relationship to compute and storage](#relationship-to-compute-and-storage)
- [Extras and the lazy-import contract](#extras-and-the-lazy-import-contract)
- [Security boundary](#security-boundary)
- [Troubleshooting](#troubleshooting)

---

## Execution model

A pipeline run has three participants:

1. **An orchestrator** — your code, a scheduler, a CI job, or anything else that provisions a
   machine and starts a process on it. The library does not supply one; `strata_forge.compute`
   supplies the parts you would build one from.
2. **The compute target** — the machine the process runs on. It needs the model weights it will
   serve, a GPU, vLLM on its `PATH`, and the `strata-forge` package with the relevant extras.
3. **The pipeline module** — this module, running as `python -m ...` on that machine.

The orchestrator hands the pipeline everything it needs through the process environment and then
loses direct contact with it. It learns what is happening by reading the progress file back off
the machine, and it learns the final outcome from the process exit code. There is no inbound
connection to the pipeline, no RPC surface, and no callback URL.

For `inference_runner`, the sequence on the target is:

```text
read STRATA_RUN_CONFIG  ->  validate into RunSpec  ->  load the HF dataset split
  ->  render one prompt per row      ->  emit `start`
  ->  launch vLLM on 127.0.0.1:8000  ->  wait for /v1/models to answer
  ->  fan prompts out through BatchInferenceRunner, emitting `step` per chunk
  ->  tear the server down
  ->  write results.parquet          ->  push to the Hub, or keep on the machine
  ->  emit `end` naming the destination
```

## Launching a run

The spec travels as one JSON string. The runner takes no command-line arguments at all:

```bash
export STRATA_RUN_CONFIG='{
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "dataset_id": "owner/prompts",
  "split": "train",
  "template": "Summarize the following:\n\n{doc}",
  "column_mapping": {"doc": "text"},
  "output_repo_id": "owner/summaries",
  "run_id": "run-2f9c1a",
  "hyperparams": {"max_tokens": 512, "concurrency": 16, "row_limit": 5000}
}'
export HF_WRITE_TOKEN='hf_...'          # optional; enables the push to the Hub
export FORGE_PROGRESS_PATH='progress.jsonl'

python -m strata_forge.pipelines.inference_runner
```

The two exported public helpers are importable in-process, which is how you would validate a spec
before shipping it to a machine, or unit-test your own prompt templates:

```python
from strata_forge.pipelines.inference_runner import RunSpec, render_template

spec = RunSpec(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    dataset_id="owner/prompts",
    split="train",
    template="Summarize the following:\n\n{doc}",
    column_mapping={"doc": "text"},
)

# Exactly the rendering the runner will do on the target machine, for one row.
print(render_template(spec.template, {"text": "hello world"}, spec.column_mapping))
# -> "Summarize the following:\n\nhello world"

# Ship this string as STRATA_RUN_CONFIG.
payload = spec.model_dump_json()
```

Validating the spec locally is worth doing: `RunSpec` sets `extra="forbid"`, so a typo in a field
name is an error rather than a silently ignored key, and catching it before a GPU boots is
considerably cheaper than catching it after.

## Environment variables

Everything the runner needs arrives through the environment. Nothing is read from a config file,
and `strata_forge.config` is not consulted by the runner itself.

| Variable | Required | Read by | Meaning |
|---|---|---|---|
| `STRATA_RUN_CONFIG` | yes | `load_spec` | The run spec as a JSON object. Parsed as data and validated into `RunSpec`. |
| `HF_WRITE_TOKEN` | no | `main` | Hugging Face token. Used as the read token for the dataset load and as the explicit write token for the results push. |
| `FORGE_PROGRESS_PATH` | no | `_progress_path` | Where to append progress events, when the spec's own `progress_path` is unset. |
| `HF_ENDPOINT` | no | `HFHubClient` | Custom Hub endpoint, resolved through `strata_forge.config` when results are pushed. |

Three details are easy to get wrong:

- **The write token is deliberately in its own variable, never in the spec.** The spec is treated
  as low-trust data that may have travelled a long way; the credential is handed over separately
  and is passed explicitly to the Hub client rather than being picked up ambiently.
- **`HF_WRITE_TOKEN` doubles as the dataset read token.** When it is unset, the dataset load falls
  back to whatever ambient Hugging Face credentials the machine already has, which is what you
  want for a public dataset and not enough for a gated one.
- **The spec's `progress_path` wins over `FORGE_PROGRESS_PATH`.** The env var is the fallback, and
  it is read even when the spec is malformed — so a spec that fails validation can still report
  its own failure.

## The run spec

`RunSpec` is inert data. It names things to fetch and knobs to set; it carries no code, no
credentials, and no filesystem paths outside the two the runner controls.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `model_id` | `str` | required | Hugging Face model id to serve. Validated as `owner/name`. |
| `dataset_id` | `str` | required | Hugging Face dataset id to read. Validated as `owner/name`. |
| `dataset_commit_sha` | `str \| None` | `None` | Pin the dataset to a revision. |
| `dataset_config` | `str \| None` | `None` | Dataset config name, for multi-config datasets. |
| `split` | `str` | required | Split to read, e.g. `"train"`. |
| `column_mapping` | `dict[str, str]` | `{}` | Maps placeholder names in `template` to column names in the split. |
| `template` | `str` | required | The prompt template. See [Template rendering](#template-rendering). |
| `hyperparams` | `Hyperparams` | all defaults | Sampling, serving and bounding knobs. |
| `output_repo_id` | `str \| None` | `None` | Hub dataset repo to push results to. Validated as `owner/name`. |
| `progress_path` | `str \| None` | `None` | Progress file path; overrides `FORGE_PROGRESS_PATH`. |
| `run_id` | `str \| None` | `None` | Names the on-machine results directory when results are not pushed. |

`Hyperparams` is likewise `extra="forbid"`, with bounds enforced by Pydantic:

| Field | Type | Default | Bounds | Meaning |
|---|---|---|---|---|
| `temperature` | `float \| None` | `None` | — | Sampling temperature, forwarded per request. |
| `max_tokens` | `int \| None` | `None` | `>= 1` | Output cap, forwarded per request. |
| `top_p` | `float \| None` | `None` | — | Nucleus sampling, forwarded per request. |
| `concurrency` | `int` | `8` | `1..256` | Maximum in-flight requests against the local server. |
| `progress_chunk` | `int` | `200` | `>= 1` | Rows completed between `step` events. |
| `row_limit` | `int \| None` | `None` | `>= 1` | Stop reading the split after this many rows. |
| `tensor_parallel_size` | `int` | `1` | `>= 1` | vLLM `--tensor-parallel-size`. |
| `max_model_len` | `int \| None` | `None` | `>= 1` | vLLM `--max-model-len`. |
| `dtype` | `str \| None` | `None` | — | vLLM `--dtype`. |
| `wait_timeout_s` | `float` | `1800.0` | `> 0` | How long to wait for the server to answer before failing. |

`row_limit` bounds materialization rather than trimming afterwards: the split is sliced while it
is being iterated, so a split far larger than the machine's memory stops being read at the cap
instead of being loaded and then truncated. It is the safety valve for pointing a run at a dataset
whose size you do not control.

`wait_timeout_s` defaults to thirty minutes because that is a realistic cold start for a large
model on a fresh machine — weights still have to be fetched and loaded before vLLM answers.

## Template rendering

`render_template` replaces each `{name}` placeholder whose `column_mapping[name]` names a column
present in the row, substituting `str(row[column])`. Everything else is left exactly as written.

```python
from strata_forge.pipelines.inference_runner import render_template

render_template("Q: {q} / {other}", {"question": "hi"}, {"q": "question"})
# -> "Q: hi / {other}"
```

The mechanism is a regular-expression substitution over `[A-Za-z0-9_]+` placeholder names. It is
deliberately **not** `str.format` and **not** a template engine, and the difference is the point:

- `str.format` reaches attributes and indices (`{q.__class__}`, `{q[0]}`) and carries a format
  mini-language (`{q:>9999999}`) that can be used to read internals or exhaust memory. None of
  those forms is a bare `{name}` token, so none of them matches, and all of them stay literal.
- A template engine executes. This does not, so a hostile template is inert text.

Two consequences worth knowing before you write templates:

- **There is no `{{` escaping.** `"{{q}}"` renders as `"{hi}"`, because the inner `{q}` is a valid
  placeholder and the braces around it are ordinary characters.
- **Rendered prompts are capped at 200,000 characters.** A pathological row is truncated rather
  than allowed to blow up the request.

An unmapped or missing placeholder staying literal is a design choice, not an oversight: a run
over a heterogeneous split should not abort on the rows where an optional column is absent.

## The progress protocol

The runner appends [`ProgressEvent`](training.md) records — the same shape
`strata_forge.training` uses — to the progress file, one JSON object per line. `JsonlProgressWriter`
opens the file in append mode, creates parent directories, and flushes after every line, so a
reader tailing the file sees each event the moment it is written and never observes a partial
rewrite.

```json
{"kind":"phase","step":null,"total_steps":null,"epoch":null,"loss":null,"learning_rate":null,"metrics":{},"message":"Loading the dataset","ts":"2026-08-09T17:48:31.204Z"}
{"kind":"start","step":null,"total_steps":1200,"epoch":null,"loss":null,"learning_rate":null,"metrics":{},"message":"","ts":"2026-08-09T17:49:09.860582Z"}
{"kind":"step","step":400,"total_steps":1200,"epoch":null,"loss":null,"learning_rate":null,"metrics":{"succeeded":398.0,"failed":2.0},"message":"","ts":"2026-08-09T17:49:12.114Z"}
{"kind":"end","step":1200,"total_steps":1200,"epoch":null,"loss":null,"learning_rate":null,"metrics":{"succeeded":1196.0,"failed":4.0},"message":"owner/summaries","ts":"2026-08-09T18:02:41.907Z"}
```

The event sequence for a successful run is a `phase` for each uncountable provisioning step,
`start` once the work can be counted, one `step` per `progress_chunk` rows completed, and `end`:

| Kind | When | What it carries |
|---|---|---|
| `phase` | Entering a step that has nothing to count: loading the split, starting the model server, generating, writing and uploading results. | `message` only — a short phrase such as `Loading the dataset`. Never a step count. |
| `start` | After the split is loaded and prompts are built, before the server launches. | `total_steps` = number of prompts. |
| `step` | After each chunk of `progress_chunk` prompts completes. | `step` = cumulative rows completed; `metrics` = running `succeeded` / `failed` counts. |
| `end` | After results are written and, if applicable, pushed. | Final counts, and `message` = where the results landed. |
| `error` | On any unhandled failure. | `message` = the failure, with credentials scrubbed. |

`phase` is the only kind that can precede `start`, and it exists because the stretches between
countable milestones are where a run spends most of its wall clock. Without them the interval
between `start` and the first `step` is one indeterminate wait — which is exactly where a model
server that never comes up burns its entire timeout, invisibly. Phase messages pass through the
same credential scrub as `error` and are capped at 200 characters, because `serving_endpoint`'s
`on_phase` hook is public API and a caller's phrase must not be able to grow the file the
orchestrator is tailing.

`ProgressEvent` is shared with the training runners, so it has fields this pipeline never
populates: `epoch`, `loss` and `learning_rate` are always `null`, and the `eval` and `checkpoint`
kinds are never emitted. Read `metrics` for inference counters. An `error` event can appear with no
preceding `start` — a spec that fails validation never gets far enough to count anything.

### Why a file and not a socket

The progress channel is a file the orchestrator pulls, rather than a connection the pipeline
pushes, and that is a deliberate decision recorded in
[ADR 0016](../architecture/adr/0016-backend-read-file.md):

- **The `Backend` protocol has no push channel.** `submit`, `status`, `logs`, `cancel`, `cleanup`
  and `read_file` are all pull operations. A pipeline that pushed would need network reachability
  back to the orchestrator, which a machine behind NAT on a rented GPU host generally does not
  have.
- **`logs` is the wrong channel.** It carries process stdout and stderr, which on a vLLM host is a
  torrent of unrelated server output. Structured metrics muxed into that stream would have to be
  parsed back out of it.
- **A file is replayable.** The full history is on disk, so an orchestrator that restarts, or
  polls slowly, or polls late, still gets every event. A socket would have dropped whatever was
  emitted while nobody was listening.
- **It costs nothing.** JSONL needs no runtime dependency, degrades to `tail -F` over SSH when you
  are debugging by hand, and survives the process that wrote it.

`Backend.read_file(job, "progress.jsonl", tail=N)` is the matching read side. It is confined to the
job's working directory — absolute paths and `..` are rejected — and returns `""` for a file that
does not exist yet, since an early poll arriving before the first write is normal rather than an
error. `SSHBackend` and `LocalBackend` implement it; `SkyPilotBackend` currently raises
`NotImplementedError`.

## Results and where they land

Results are one row per input row, reconciled positionally against the split:

| Column | Type | Meaning |
|---|---|---|
| `custom_id` | `str` | `"row-{i}"`, where `i` is the row's index in the (capped) split iteration. |
| `output` | `str \| None` | The model's text, or `None` when that row failed. |
| `error` | `str \| None` | `repr()` of the exception for that row, or `None` on success. |

They are written as `results.parquet` — parquet because it renders directly in the Hugging Face
dataset viewer — and land in one of two places:

- **Pushed to the Hub** when `HF_WRITE_TOKEN` *and* `output_repo_id` are both present. The runner
  creates the repo as a **private dataset** repo (`exist_ok=True`), uploads `results.parquet`, and
  sets the `end` event's `message` to the repo id.
- **Kept on the machine** otherwise, under `~/strata-inference-results/<run_id>/results.parquet`,
  with `message` set to that path. The location is outside the job's working directory on purpose:
  an orchestrator that calls `Backend.cleanup` deletes the workdir, and results that only exist
  there would go with it. `run_id` must match `[A-Za-z0-9_-]+` to be used as a directory name; a
  missing or unsafe value falls back to the current directory, which is a degraded outcome — the
  results may be cleaned up — but never one that escapes to an attacker-chosen path.

Requiring both the token and the repo id before pushing means the machine never guesses where
results belong. A token without a destination, or a destination without a token, keeps the data
local rather than writing it somewhere unintended.

## Exit codes and failure semantics

`main()` returns a process exit code, and `python -m strata_forge.pipelines.inference_runner`
exits with it:

| Code | Meaning |
|---|---|
| `0` | The run completed. Results were written to their destination. |
| `1` | The run failed. An `error` event was appended to the progress file, if one was configured. |

The distinction that matters most: **a failed row is not a failed run.** The batch runs with
`on_error="collect"`, so an individual request that fails is recorded in that row's `error` column,
counted in the `failed` metric, and the batch continues. A run in which every row failed still
exits `0` and still writes a results file. Treat the `succeeded` / `failed` counts on the `end`
event as the real outcome signal, not the exit code alone.

Exit code `1` is reserved for failures that stop the run as a whole: a missing or invalid
`STRATA_RUN_CONFIG`, an id that fails validation, a `column_mapping` naming columns the split does
not have, a dataset that cannot be loaded, or a server that does not answer within
`wait_timeout_s`. Every one of these is reported through an `error` event whose message has been
scrubbed of anything token-shaped before it is written.

If no progress path is configured at all, failures are still reflected in the exit code — there is
simply nowhere to write the detail.

## Relationship to compute and storage

`strata_forge.pipelines` sits at the top of the dependency graph and is the only module that draws
on four siblings at once. It adds no new capability of its own; it is composition.

- **[`strata_forge.compute`](compute.md)** is on both sides of it. Outside, it is what an
  orchestrator uses to *start* a pipeline on a machine and to read its progress file back
  (`submit`, `status`, `read_file`, `cleanup`). Inside, the pipeline uses `LocalBackend`,
  `build_vllm_task` and `serving_endpoint` to run the model server as a child process on that same
  machine, and `BatchInferenceRunner` to bound concurrency against it. The inner backend is always
  local — the pipeline is already where the work is.
- **[`strata_forge.storage`](storage.md)** is the exit. `HFHubClient` performs the repo creation
  and upload, and is constructed with an explicit token so it never picks up the machine's ambient
  Hugging Face credentials for a write.
- **[`strata_forge.llm`](llm.md)** provides the client. The model server is reached over the
  `openai_compat` route, with the provider pinned to the loopback endpoint that
  `serving_endpoint` just verified.
- **[`strata_forge.training`](training.md)** provides the progress vocabulary, and nothing else.

The server is torn down before results are written. `serving_endpoint` is an async context manager
that cancels and cleans up the job on exit, and the write and push happen after that block closes,
so the GPU is released before the upload starts.

Note that this module does **not** use `strata_forge.datasets`. It reads the Hugging Face
`datasets` library directly, because it is consuming an arbitrary upstream split rather than a
Forge-owned `Dataset`. The two are unrelated despite the similar name.

## Extras and the lazy-import contract

Importing the module is always safe. `import strata_forge.pipelines.inference_runner` succeeds on
a bare `pip install strata-forge` with no extras at all — the heavy dependencies are imported
inside the functions that use them, so a spec can be built and validated anywhere.

Actually *running* a batch-inference job needs more:

| Requirement | Extra | Used for |
|---|---|---|
| `datasets` | `strata-forge[hf]` | Loading the input split and writing `results.parquet`. |
| `huggingface_hub` | `strata-forge[storage]` | Pushing results, when a token and output repo are set. |
| vLLM on the machine's `PATH` | `strata-forge[serving]` | Serving the model. |

vLLM is the odd one out. The runner never imports it — it builds a shell command, and
`build_vllm_task` is called with `setup=""`, so the runner does **not** install vLLM itself. It has
to already be on the target machine, which is the provisioning step's job. Installing the
`[serving]` extra as part of preparing the machine is the straightforward way to get it there.

Only the `datasets` import raises a friendly error when it is missing
(`the [hf] extra is required to load datasets: pip install 'strata-forge[hf]'`). The others surface
as an `ImportError` from `strata_forge.storage`, or as a failed server launch.

## Security boundary

The compute target is where a spec that may have travelled through several systems meets real
credentials and a real network, so the runner treats its input as untrusted even when the caller
is trusted:

- **The spec is parsed as data.** `json` plus Pydantic with `extra="forbid"` — never `eval`,
  `pickle`, or `yaml.unsafe_load`. `template`, `column_mapping` and `hyperparams` are values, not
  code.
- **Ids are re-validated on the machine.** `model_id`, `dataset_id` and `output_repo_id` must match
  `owner/name` with no traversal, scheme, whitespace or trailing newline, even if something
  upstream already checked them. The machine is the boundary that actually fetches and pushes, so
  it does its own checking.
- **Template rendering executes nothing**, and rendered output is length-capped.
- **The write token stays out of the spec**, is passed explicitly to the Hub client rather than
  read ambiently, and is scrubbed — along with anything else token-shaped, such as an `hf_...`
  string or a `Bearer ...` header fragment — from any message that reaches the progress file.
- **The model server binds to `127.0.0.1`.** vLLM's own default is `0.0.0.0`; the runner overrides
  it, because a network-reachable machine must not expose an unauthenticated model server.
- **Emitted events carry no secrets.** Events hold step counts, float metrics and a destination
  identifier; result rows hold model text only.

## Troubleshooting

- **Every row fails with `RegistryError: Unknown model`.** This is the failure mode to check
  first, because it is silent: the run exits `0`, writes a complete `results.parquet`, and emits a
  clean `end` event, but every row has `output: null` and an `error` naming the model. The cause is
  that `LLMClient` computes a USD cost for every completion by looking the model up in
  `strata_forge.llm`'s curated registry, and a self-hosted model id such as
  `Qwen/Qwen2.5-7B-Instruct` is not in it. The failure happens after the server has answered, so
  the server, the prompts and the plumbing are all fine. Read the `succeeded` count on the `end`
  event rather than trusting the exit code.
- **The run sits between `start` and the first `step`.** That interval is vLLM loading weights,
  which for a large model on a cold machine is minutes, not seconds. The `phase` events emitted
  around it say which step is actually in progress, so read those before assuming a deadlock. If it
  exceeds `wait_timeout_s` the run fails with a readiness error; raise the timeout rather than
  treating it as a hang.
- **`column_mapping references columns not in the split`.** The names on the right-hand side of
  `column_mapping` are *dataset column* names, and the names on the left are *template
  placeholder* names. Getting them the wrong way round produces exactly this error.
- **Placeholders appear literally in the prompts.** A `{name}` with no `column_mapping` entry, or
  one mapping to a column absent from the row, is left as-is by design. Check the mapping first;
  a run with an unmapped placeholder produces well-formed but wrong prompts and will not complain.
- **`STRATA_RUN_CONFIG is not set`.** The variable has to be present in the environment of the
  process itself. Exporting it in a shell that then submits work through a backend does not
  necessarily propagate it; put it in the task's `env` instead.
- **`invalid STRATA_RUN_CONFIG`.** `extra="forbid"` rejects unknown keys, so this is usually a
  misspelled field. Validate with `RunSpec.model_validate_json(...)` locally before shipping.
- **Results are not in the Hub repo.** Pushing requires *both* `HF_WRITE_TOKEN` and
  `output_repo_id`. With only one of them, results are kept on the machine instead — the `end`
  event's `message` says which happened.
- **Results vanished after the job finished.** Results only survive `Backend.cleanup` when
  `run_id` is set and matches `[A-Za-z0-9_-]+`. Without it they are written to the working
  directory, which cleanup deletes.
