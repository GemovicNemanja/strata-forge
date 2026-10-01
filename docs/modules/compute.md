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
``STRATA_RUN_CONFIG``, take the HF write token only from the private
secrets file the backend wrote beside the job, and append the same ``ProgressEvent`` stream to
``FORGE_PROGRESS_PATH`` — so an orchestrator reads one protocol regardless
of which is running.

### The runner contract

An orchestrator that launches a ``strata_forge.pipelines`` runner on a
machine it does not own holds up its side of the contract in four places:

- **The spec is inert data** in ``STRATA_RUN_CONFIG``: JSON validated into
  a Pydantic model with ``extra="forbid"``, so a key the installed engine
  does not know is a named failure, never an ignored instruction. Nothing
  in a spec names code to run.
- **The write token travels apart from the spec**, as
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
forgets the job only once nothing is left at the directory's path: a
directory that survives raises ``CleanupError`` and the job stays
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

``cleanup`` runs ``rm -rf <workdir> && test ! -e <workdir> && test ! -L
<workdir>``, so its exit status means "nothing is there now" rather
than "``rm`` did not complain"; a nonzero status, or a channel that
closed with none, raises ``CleanupError`` without echoing the remote's
stderr (which would name the files ``rm`` could not remove). An absent
workdir is a success. Every method that composes a remote path from a
job's ``remote_workdir`` first checks it is a directory *below*
``remote_root``: an absolute path, a ``..`` component, or a path that
names the root itself (``<root>/``, ``<root>/.``, ``<root>//``) is a
``ValueError``, since ``rm -rf`` of the root would remove every job's
workdir.

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
this tears down the entire cluster, not just the job. A failed
``down`` raises ``CleanupError`` (chained to the SDK's error, naming
only its type); SkyPilot's ``ClusterDoesNotExist`` counts as success.

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
