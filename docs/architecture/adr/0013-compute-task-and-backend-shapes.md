# ADR 0013 — `strata_forge.compute` ships task-as-data + a Backend Protocol

**Status:** Accepted
**Date:** Initial scaffolding for `strata_forge.compute`
**Supersedes:** —
**Superseded by:** —

## Context

`strata_forge.compute` is the module that lets Forge code submit remote
work — training jobs, batch inference, eval runs — to either a
managed orchestrator (SkyPilot reaching AWS / GCP / Azure / RunPod
/ Lambda / Kubernetes) or directly to a single SSH-accessible host
(for one-off GPU boxes the user already has).

The phase needs to decide three things up front so the rest of the
module can hang off a stable seam:

1. **What shape is a "task"?** A function decorator? A class
   instance? A YAML file? Something serializable?
2. **How does Forge talk to a backend?** A class hierarchy that
   each backend subclasses? A Protocol every backend satisfies?
   A registry of submit functions?
3. **What lifecycle does a job have?** Just "submit and return,"
   or "submit, observe state, fetch logs, cancel, clean up?"

Picking wrong here means later sub-phases (SkyPilot, SSH, batch
inference, serving) either fork their interfaces or grow a
parallel runtime — both defeat the layered design.

## Decision

### 1. Tasks are declarative Pydantic data

```python
class Task(BaseModel):
    name: str
    run: str
    setup: str = ""
    workdir: str | None = None
    env: dict[str, str] = {}
    file_mounts: dict[str, str] = {}     # remote_path -> local_path
    resources: ResourceSpec | None = None
    num_nodes: int = 1

class ResourceSpec(BaseModel):
    cpus: int | None = None
    memory_gb: int | None = None
    accelerators: str | None = None       # "A100:1" / "T4:4" / ...
    cloud: str | None = None              # "aws" / "gcp" / "azure" / ...
    region: str | None = None
    disk_gb: int | None = None
```

A `Task` is **pure data**, frozen Pydantic, serializable to JSON
or YAML. Two callers describing "the same task" produce identical
bytes — making "have we already run this?" decidable, and making
tasks portable across processes (the dispatching script and the
worker that runs them are decoupled).

A YAML loader (`Task.from_yaml(path)` / `from_yaml_str(...)`)
parses the SkyPilot-compatible task subset so users can drop in
existing YAML templates without rewriting them. We don't aim for
full SkyPilot YAML compatibility — only the fields the
:class:`Task` model carries; unknown fields raise.

### 2. Backends are an async Protocol with the full lifecycle

```python
@runtime_checkable
class Backend(Protocol):
    @property
    def name(self) -> str: ...
    async def submit(self, task: Task) -> Job: ...
    async def status(self, job: Job) -> JobStatus: ...
    async def logs(self, job: Job, *, tail: int | None = None) -> str: ...
    async def cancel(self, job: Job) -> None: ...
    async def cleanup(self, job: Job) -> None: ...
```

Every backend that wants to satisfy the Protocol implements all
five methods. The full lifecycle is mandatory because the absence
of (say) `logs` or `cleanup` would force higher-level code to
special-case backends — defeating the point of the abstraction.

Backends ship in :mod:`strata_forge.compute.backends`:

- **Phase 5.1:** :class:`LocalBackend` — runs the task in a local
  `asyncio.create_subprocess_exec`. Dep-free, primarily for tests
  and ad-hoc local runs.
- **Phase 5.2:** :class:`SSHBackend` (`asyncssh`, behind
  `[compute]`) and :class:`SkyPilotBackend` (`sky.api.sdk`,
  behind `[compute]`).

### 3. Jobs are opaque handles, JobStatus is a typed state machine

```python
class Job(BaseModel):
    id: str
    backend: str
    task_name: str
    metadata: dict[str, Any] = {}

class JobStatus(BaseModel):
    state: Literal[
        "pending", "running", "succeeded", "failed", "cancelled"
    ]
    exit_code: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str = ""
```

`Job` is a serialisable handle: an `id`, the backend name, and
optional backend-specific metadata in a free-form dict. Callers
that store jobs and re-attach later (a CI script that submits in
one job, polls in another) deserialize the same `Job` and call
the backend's methods on it.

`JobStatus` collapses every backend's internal lifecycle into one
canonical five-state machine. Backend-specific states (SkyPilot's
`STARTING` vs `RUNNING` vs `SETTING_UP`, SSH's "process exists"
vs "stdout writer closed") map down to `pending` / `running`;
the difference is recovered through `message` when the user
needs it.

### 4. Backends own their own scheduling primitives

The Protocol describes what backends do, not how. SkyPilot uses
its own queue + cluster manager; SSH uses `nohup` + a PID file;
the local backend uses an in-process `dict` of running
subprocesses. Forge doesn't try to unify those — it would
re-implement orchestration we'd rather delegate.

## Consequences

**Positive**

- **One Protocol covers every backend.** Higher-level modules
  (the batch inference runner in 5.2, the training runner in 5.3)
  bind to `Backend` and accept any concrete implementation. New
  backends (a future Kubernetes-direct backend, a Modal adapter,
  a Slurm cluster) drop in by satisfying the Protocol.
- **Tasks portable across processes.** A task serialized to JSON
  / YAML at one site, reloaded at another, produces the same
  `Task`. Backends can hash the canonical form for "already ran
  this" lookups.
- **Lifecycle is uniform.** A CI script doesn't need to know
  whether the backend is SkyPilot or SSH to fetch logs or cancel
  the job.
- **No runtime dep cost up front.** :class:`LocalBackend` is
  dep-free; the heavier backends lazy-import their SDKs inside
  the constructor.

**Negative**

- **Five-method Protocol is a lot of surface area.** A backend
  that "just submits and forgets" still has to implement
  `status` / `logs` / `cancel` / `cleanup` — even if some are
  no-ops. Documentation calls this out so authors don't
  accidentally ship half-implemented backends.
- **The status state machine loses backend-specific nuance.**
  Mapping SkyPilot's seven states down to five means edge cases
  (a cluster in the middle of `SETTING_UP` reported as
  `pending`) need the `message` field for full fidelity.

**Mitigations**

- A non-applicable lifecycle method (e.g. `cleanup` on a
  serverless backend with no cleanup work) raises a clear
  `NotImplementedError` rather than silently passing. Callers
  that depend on the no-op can catch it explicitly.
- We document the five-state mapping in `docs/modules/compute.md`
  and recommend the `message` field for "what was happening
  underneath?" surface.

## Alternatives considered

1. **Tasks as Python functions / decorators.** Ergonomic locally
   but not serializable, and the function's closure-captured
   variables are a constant source of "why didn't my remote job
   see my local variable?" bugs. Rejected — declarative data
   plus a YAML loader covers the use cases without the
   serialization headaches.

2. **Backend as an ABC with abstract methods.** Forces subclassing
   on every implementation. Protocol-based composition matches
   the rest of `strata_forge.*` (graders, retrievers, vector stores) and
   lets user-supplied backends drop in without inheriting from a
   Forge class. Rejected.

3. **Submit-and-forget Protocol (no status / logs / cancel).**
   Simpler interface but pushes the lifecycle into each caller.
   The CI eval gate, the training runner, and the batch inference
   tracker all need the same plumbing — better to put it in the
   backend Protocol once. Rejected.

4. **Wrap SkyPilot directly without a layer.** Faster to ship,
   but harder to add a non-SkyPilot backend later (Modal,
   Anyscale, a custom Slurm cluster). Rejected — the Protocol
   pays for itself the moment the second backend lands.
