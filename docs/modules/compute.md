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
for the task-as-data + Protocol design rationale.

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
- [Job lifecycle](#job-lifecycle)
- [Backend protocol](#backend-protocol)
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

The package ships two runners over that plumbing:
``inference_runner`` (serve a model with vLLM, run a batch over a dataset)
and ``finetune_runner`` (train with :mod:`strata_forge.training`, push the
adapter or merged model to the Hub). Both read an inert JSON spec from
``STRATA_RUN_CONFIG``, take the HF write token only from its own
``HF_WRITE_TOKEN`` env var, and append the same ``ProgressEvent`` stream to
``FORGE_PROGRESS_PATH`` — so an orchestrator reads one protocol regardless
of which is running.

## Backend protocol

```python
class Backend(Protocol):
    name: str
    async def submit(self, task: Task) -> Job: ...
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
    async def cleanup(self, job: Job) -> None: ...
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
chunk = await backend.console(job)
while not done:
    chunk = await backend.console(
        job, stdout_offset=chunk.stdout_offset, stderr_offset=chunk.stderr_offset
    )
    render(chunk.stdout, chunk.stderr)
    if chunk.dropped_bytes:
        render(f"[{chunk.dropped_bytes} bytes not shown]")
```

When more than ``max_bytes`` accumulated between calls the NEWEST bytes are kept
and the shortfall is reported as ``dropped_bytes`` — a watcher wants where the
run is now, and a silent jump-cut would read as a whole transcript. Offsets are
byte offsets into the underlying stream, so a slice may cut a multi-byte
character; the boundary decodes to a replacement character rather than raising.

The SSH backend does this in ONE round trip: it measures both files and computes
the slice lengths in the same remote shell (measuring locally and slicing
remotely would race a job that is still writing) and base64-frames the payloads
so byte counts survive the transport. SkyPilot has no way to ask for a suffix of
``sky logs``, so it re-reads the whole log and slices locally — correct, but
linear in the log's size per poll.

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

Submission scripts a wrapper on the remote host (under
``~/.forge-compute/<job_id>/``) and launches it under ``nohup``,
capturing the PID and exit code in files. ``status`` probes
``kill -0`` for liveness, then falls back to the exit-code file.
``cancel`` sends SIGTERM, waits 2 s, then SIGKILL.

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
this tears down the entire cluster, not just the job.

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
