# `strata_forge.compute` — task shapes, backends, batch inference, serving adapters

`strata_forge.compute` is the remote-compute orchestration layer. It
ships typed task / job / status Pydantic shapes, a
`Backend` Protocol, three concrete backends
(`LocalBackend`, `SSHBackend`, `SkyPilotBackend`), a
concurrency-bounded `BatchInferenceRunner` for fan-out over a shared
`LLMClient`, and small task-builder functions for
self-hosted inference servers (vLLM, TGI, SGLang). See
[ADR 0013](../architecture/adr/0013-compute-task-and-backend-shapes.md)
for the task-as-data + Protocol design rationale.

Integration points:

- **Tasks:** `Task`, `ResourceSpec`. Pure data
  shapes that serialize to / from YAML (SkyPilot-style).
- **Jobs:** `Job`, `JobStatus`, `JobState`.
  Five canonical states: `pending` / `running` /
  `succeeded` / `failed` / `cancelled`.
- **Protocol:** `Backend` — async `submit` / `status` / `logs` /
  `read_file` / `cancel` / `cleanup`, plus a `name` property.
- **Backends:** `LocalBackend` (in-process subprocess),
  `SSHBackend` (asyncssh, lazy `[compute]` extra),
  `SkyPilotBackend` (sky.api.sdk, lazy `[compute]` extra).
- **Path guard:** `safe_workdir_relpath` — the workdir-confinement
  check every backend's `read_file` runs on its `path` argument.
- **Batch inference:** `BatchInferenceRunner`,
  `BatchInferenceResult` — concurrency-capped async
  fan-out over `LLMClient`.
- **Serving adapters:** `build_vllm_task`, `build_tgi_task`,
  `build_sglang_task`, `serving_endpoint`, `wait_for_endpoint`,
  `ServingEndpoint`.

Module rules: [`src/strata_forge/compute/CLAUDE.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/compute/CLAUDE.md).
Source: [`src/strata_forge/compute/`](https://github.com/GemovicNemanja/strata-forge/tree/main/src/strata_forge/compute/).

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
import asyncio

from strata_forge.compute import LocalBackend, Task


async def main() -> None:
    backend = LocalBackend()
    job = await backend.submit(Task(name="hello", run="echo 'hi from compute'"))

    while not (status := await backend.status(job)).is_terminal:
        await asyncio.sleep(0.05)

    print(status.state, (await backend.logs(job)).strip())   # → succeeded hi from compute
    await backend.cleanup(job)


asyncio.run(main())
```

`JobStatus.is_terminal` is the poll predicate — `True` for `succeeded`,
`failed`, and `cancelled`.

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
from strata_forge.compute import LocalBackend, build_vllm_task, serving_endpoint
from strata_forge.llm import LLMClient, Message, OpenAICompatConfig, OpenAICompatProvider

task = build_vllm_task("meta-llama/Llama-3.1-8B-Instruct", port=8000)
async with serving_endpoint(
    LocalBackend(), task, base_url="http://localhost:8000/v1"
) as endpoint:
    client = LLMClient(
        model="meta-llama/Llama-3.1-8B-Instruct",
        provider="openai_compat",
        # There is no `provider_config=` kwarg. Pin the endpoint by
        # supplying the whole provider client for that route.
        provider_clients={
            "openai_compat": OpenAICompatProvider(
                OpenAICompatConfig(base_url=endpoint.base_url)
            )
        },
    )
    response = await client.complete([Message.user("Hello!")])
```

> The model id must also exist in the LLM module's registry, because cost
> accounting looks it up on every response. See
> [the `openai_compat` limitation](llm.md#capability-gate).

---

## Task and ResourceSpec

A `Task` is a frozen Pydantic instance describing what to
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
`"A100:8"`, `"H100:4"`, etc. The local backend ignores
`resources` entirely; SSH inherits whatever the host has;
SkyPilot forwards it verbatim.

## Job lifecycle

Backends return `Job` handles from `submit`. A
`JobStatus` carries a canonical `JobState`:

| State | Meaning |
|---|---|
| `pending` | Queued / setting up; not yet running. |
| `running` | Actively executing. |
| `succeeded` | Exit code 0. |
| `failed` | Non-zero exit, infrastructure error, or unparseable status. |
| `cancelled` | Process killed before completion. |

The five-state set is intentionally narrow — backend-specific
nuance (SkyPilot's `SETTING_UP`, SSH's process-gone-no-exit-file)
lands in `JobStatus.message` rather than expanding the state
machine.

## Backend protocol

```python
@runtime_checkable
class Backend(Protocol):
    @property
    def name(self) -> str: ...
    async def submit(self, task: Task) -> Job: ...
    async def status(self, job: Job) -> JobStatus: ...
    async def logs(self, job: Job, *, tail: int | None = None) -> str: ...
    async def read_file(self, job: Job, path: str, *, tail: int | None = None) -> str: ...
    async def cancel(self, job: Job) -> None: ...
    async def cleanup(self, job: Job) -> None: ...
```

Methods that don't apply to a particular backend raise
`NotImplementedError` rather than silently passing — that
way callers can `try/except` if needed instead of relying on
backend-specific knowledge of which methods are no-ops.
`SkyPilotBackend.read_file` is the one method currently in that
position; `LocalBackend` and `SSHBackend` both implement it.

### `read_file` and the workdir guard

`logs` returns the job's stdout and stderr. `read_file` is the
side-channel: a structured file the job wrote inside its own working
directory — a `progress.jsonl` of metric events, a small result manifest —
that you want to tail without parsing it out of interleaved log text.
[ADR 0016](../architecture/adr/0016-backend-read-file.md) records why it
is on the Protocol rather than in each caller.

```python
events = await backend.read_file(job, "progress.jsonl", tail=20)
```

`path` is **relative to the job's workdir** and must stay inside it.
`safe_workdir_relpath` (exported from `strata_forge.compute`) enforces
that: absolute paths, `..` escapes, backslashes, and NUL bytes all raise
`ValueError`. `LocalBackend` additionally resolves the realpath and
re-confines it. Reads are capped at 8 MiB. A file that does not exist yet
returns `""` rather than raising — a progress file the job hasn't written
is not an error.

## LocalBackend

```python
from strata_forge.compute import LocalBackend, Task

backend = LocalBackend()                       # or LocalBackend(name="l2")
backend = LocalBackend(env_inherit=False)      # child sees only Task.env
job = await backend.submit(Task(name="t", run="python -m my_script"))
```

The local backend spawns each task as an async subprocess (`bash -lc`) and tracks
it by job id **in memory**. The child runs in `task.workdir` when the task sets
one and in the parent's current directory otherwise, and the backend touches no
filesystem of its own by default. A `LocalBackend` handle does not survive the
process that created it. It runs single-node tasks only — `num_nodes != 1` raises.

`env_inherit=True` (the default) gives the child the parent's `os.environ` plus
`Task.env`; `env_inherit=False` gives it `Task.env` alone, which is the safer
choice when a task should not see your local provider keys.

Both pipes are drained continuously rather than read to EOF, so `logs` returns
partial output **while the job is still running** — the case worth diagnosing is
a process that came up wrong and then hung, and its output exists long before it
exits. The in-memory buffer keeps the last 1 MiB per stream.

```python
backend = LocalBackend(log_dir="./job-logs")
```

`log_dir` additionally tees both streams to `serve.stdout.log` /
`serve.stderr.log` under that directory, written as the bytes arrive. Those files
are deliberately left in place by `cleanup`: once the process and its buffers are
gone, the file is the only remaining evidence. It is opt-in, so a caller who never
asks for it never finds log files appearing. The names are fixed (an orchestrator
finds them without knowing the job id), so give concurrent jobs their own
directories.

`cleanup` otherwise drops the in-memory record and terminates a lingering process,
escalating to a kill if it does not exit within five seconds.

A job ends when the child is reaped, not when its pipes close: a process the child
backgrounded inherits those descriptors and can hold them open indefinitely, and
gating the lifecycle on EOF would leave such a job stuck at `running`. The drain
gets a few seconds after the exit to finish reading.

## SSHBackend

```python
from strata_forge.compute import SSHBackend, Task

backend = SSHBackend(host="gpu-host.example.com", username="ml-team")
job = await backend.submit(Task(name="t", run="python train.py"))
```

Submission scripts a wrapper on the remote host (under
`~/.forge-compute/<job_id>/`) and launches it under `nohup`,
capturing the PID and exit code in files. `status` probes
`kill -0` for liveness, then falls back to the exit-code file.
`cancel` sends SIGTERM, waits 2 s, then SIGKILL.

Pass a pre-built `asyncssh.SSHClientConnection` via
`connection=` to share a connection across multiple submits. When the
backend opened the connection itself, `await backend.close()` shuts it
down; with an injected `connection=` it is a no-op and the connection
stays yours to manage.

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

The backend wraps SkyPilot's sync SDK in `asyncio.to_thread` so
the public surface stays async-uniform. It maps SkyPilot's twelve
job states to the canonical five; unknown states fall back to
`running` so callers don't crash on new SkyPilot versions.

`cleanup` calls `sky.down` on the cluster — be aware that
this tears down the entire cluster, not just the job.

## Batch inference

`BatchInferenceRunner` runs many prompts through one
shared `LLMClient` with a concurrency cap:

```python
runner = BatchInferenceRunner(client, concurrency=20, on_error="collect")
results = await runner.run(prompts, temperature=0.7, max_tokens=500)
```

`on_error="collect"` returns a tuple positionally aligned with the input
prompts: every slot holds either a `response` or an `error`, and the run
always completes.

`on_error="raise"` (the default) is not a cancellation. The runner fans
out with a bare `asyncio.gather`, so the first exception propagates out of
`run()` — which therefore returns nothing at all — while the sibling calls
that were already in flight **keep running to completion and keep costing
money**. Nothing is cancelled and no partial results are handed back. If
you want a batch that stops early, or one whose partial work you can
inspect, use `on_error="collect"` and decide what to do with the error
slots yourself.

## Serving adapters

The serving helpers build a `Task` whose `run` launches
an OpenAI-compatible inference server. Combined with the
`openai_compat` provider in `strata_forge.llm`, you can talk to
self-hosted models with the same `LLMClient` API you use
for SaaS providers.

```python
from strata_forge.compute import build_vllm_task, build_tgi_task, build_sglang_task

vllm_task = build_vllm_task(
    "meta-llama/Llama-3.1-8B-Instruct",
    port=8000,
    tensor_parallel_size=1,
    max_model_len=8192,
)
tgi_task = build_tgi_task("mistralai/Mistral-7B-v0.1", port=8080)
sglang_task = build_sglang_task("Qwen/Qwen2-7B-Instruct", port=30000, tp_size=4)
```

`serving_endpoint` is an async context manager that
submits the task, waits for the HTTP endpoint to respond,
yields a `ServingEndpoint`, and on exit cancels the job
and (optionally) calls `backend.cleanup`.

```python
from strata_forge.compute import LocalBackend, serving_endpoint

async with serving_endpoint(
    LocalBackend(), vllm_task, base_url="http://localhost:8000/v1"
) as endpoint:
    # endpoint.job is the live Job; endpoint.base_url is what
    # OpenAICompatConfig needs.
    ...
```

The readiness probe hits `{base_url}/models` until it returns
HTTP 200 or the `wait_timeout_s` expires.

The task builders only compose the shell command — `vllm serve ...` and
friends. The server binary must already be on the machine the backend
targets, either because you installed it there or because you passed a
`setup=` that does. The `[serving]` extra pins `vllm` for the case where
the machine running strata-forge is also the machine serving the model;
nothing in this module imports it.

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

- `asyncssh` and `sky.api.sdk` are behind the `[compute]` extra. Both
  `SSHBackend` and `SkyPilotBackend` lazy-import them on **first use**,
  not in the constructor: constructing either backend succeeds with the
  extra absent, and the `ImportError` (with an install hint) surfaces from
  the first `submit` / `status` / `logs` call. A capability probe that
  only constructs the backend will not detect the missing extra.
- `httpx` is a core dep (via LiteLLM); `wait_for_endpoint`
  imports it inside the function.
- `LocalBackend`, `BatchInferenceRunner`, and the
  serving task builders have no extra requirements.

## Troubleshooting

- **`SSHBackend` hangs on first submit:** typically the SSH
  connection wasn't established because of host-key validation.
  Either set `known_hosts=path/to/known_hosts` or pass an
  already-built `asyncssh.SSHClientConnection` via `connection=`.
- **`ImportError` from `submit`, not from the constructor:** expected —
  `asyncssh` and `sky` are imported on first use. Install the
  `[compute]` extra (`pip install 'strata-forge[compute]'`).
- **`SkyPilotBackend.status` returns "failed: not found":** the
  job ID is no longer in SkyPilot's queue. SkyPilot prunes old
  jobs aggressively — query status before too much time passes,
  or rely on stored log output via `backend.logs`.
- **`serving_endpoint` times out:** the server is slow to come
  up. Increase `wait_timeout_s` (default 10 min); vLLM in
  particular can take several minutes for large models.
- **Server starts but `LLMClient` can't connect:** make sure
  `OpenAICompatConfig.base_url` includes the `/v1` suffix
  the provider expects, and that you passed the provider through
  `provider_clients={"openai_compat": OpenAICompatProvider(...)}` —
  `LLMClient` has no `provider_config=` parameter.
- **`NotImplementedError` from `SkyPilotBackend.read_file`:** the SkyPilot
  backend does not implement the side-channel read. Use `logs`, or run
  the job through `SSHBackend`, which does.
- **`ValueError` from `read_file` about the workdir:** the path escaped
  the job's working directory. `read_file` takes a workdir-relative path
  only — no leading `/`, no `..`.
- **`BatchInferenceRunner.run` raised and I got nothing back:** that's
  `on_error="raise"`. Switch to `on_error="collect"` to receive an
  aligned tuple with the failures in place.

---

## See also

- [`strata_forge.llm`](llm.md) — the `LLMClient` batch inference fans out
  over, and the `openai_compat` provider a self-hosted server is reached
  through.
- [`strata_forge.pipelines`](pipelines.md) — a ready-made entrypoint that
  composes serving, batch, and storage into one launchable run.
- [`strata_forge.training`](training.md) — what usually goes inside a
  `Task.run` on a GPU host.
- [`strata_forge.storage`](storage.md) — moving inputs and artifacts to and
  from that host.
- [`strata_forge.cli`](cli.md) — `strata-forge compute` and
  `strata-forge serve`.
- [ADR 0013](../architecture/adr/0013-compute-task-and-backend-shapes.md)
  — task-as-data and the `Backend` Protocol.
- [ADR 0016](../architecture/adr/0016-backend-read-file.md) — why
  `read_file` is on the Protocol.
