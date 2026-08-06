# ADR 0016 — `Backend.read_file` for workdir-confined side-channel reads

**Status:** Accepted
**Date:** Pipeline control-plane groundwork
**Supersedes:** —
**Superseded by:** —

## Context

A job's structured progress — training loss curves, eval scores, batch-inference
throughput — is emitted by the runner as a stream of `ProgressEvent`s written to a
file (a `progress.jsonl`), separate from the process stdout/stderr. A control plane
that orchestrates `strata_forge.compute` jobs (submit → poll → relay live metrics → tear
down) needs to read that file off the remote box on each poll to surface live
metrics.

The [ADR 0013](0013-compute-task-and-backend-shapes.md) `Backend` Protocol exposes
`submit` / `status` / `logs` / `cancel` / `cleanup`. None of these can read an
arbitrary file: `SSHBackend.logs` cats the hardcoded `stdout.log`/`stderr.log`, and
`SkyPilotBackend.logs` calls `client.tail_logs`. So the progress channel either has
to be muxed into stdout (coupling log noise with structured metrics) or the protocol
needs a dedicated file read.

## Decision

Add one method to the `Backend` Protocol:

```python
async def read_file(self, job: Job, path: str, *, tail: int | None = None) -> str
```

- `path` is **relative to the job's working directory and must stay within it** —
  absolute paths and `..` traversal are rejected. This is a security boundary: on a
  control plane, `path` ultimately derives from caller input, and `read_file` must
  never reach outside a job's own workdir. The shared guard `safe_workdir_relpath`
  (in `strata_forge.compute.backends.base`, re-exported from `strata_forge.compute`) normalizes and
  validates the path; every backend calls it first.
- `tail` returns only the last N lines (mirrors `logs`).
- A **missing file returns `""`**, not an error — a progress file that hasn't been
  written yet is the normal early-poll case.

It complements `logs` (process output) by reading a **side-channel file** such as a
runner's `progress.jsonl`.

### Per-backend status

- **`SSHBackend`** — full implementation: `cat`/`tail` of the validated, `shlex`-quoted
  `<workdir>/<path>` over the existing SSH connection.
- **`LocalBackend`** — full implementation: reads `<job workdir>/<path>` from the local
  filesystem (the workdir the subprocess ran in), confined by the same guard.
- **`SkyPilotBackend`** — **deferred**: it validates the path (so the confinement
  contract is uniform) then raises `NotImplementedError`. Reading an arbitrary file off
  a SkyPilot cluster needs an exec+capture round-trip over the SDK that is built and
  verified when the SkyPilot compute path lands. The Protocol is still satisfied
  structurally (the method exists with the right signature); the SSH backend is the
  current pipeline target and implements it fully.

## Consequences

- The progress channel is a dedicated file, decoupled from stdout/stderr — the poller
  parses `ProgressEvent` lines from `read_file(job, "progress.jsonl")` without log noise.
- `read_file` is a deliberate filesystem read **scoped to a job's workdir**; it is not a
  general remote-file API. The `safe_workdir_relpath` guard is the enforced boundary and
  is unit-tested (absolute paths and `..` traversal are rejected before any I/O).
- **Result is byte-capped** at `MAX_READ_FILE_BYTES` (8 MiB) on every backend. A control
  plane polls `read_file` on an interval from a *shared* process, so an unbounded read of
  a runaway file could exhaust that process's memory and degrade service for other users;
  the cap is a hard ceiling (SSH `head -c`, Local bounded `read`).
- **The guard is lexical and does not resolve symlinks.** A symlink *inside* the workdir
  pointing outside would otherwise be followed. `LocalBackend` additionally resolves the
  realpath and rejects a target outside the workdir; the SSH backend reads on the user's
  own host with a fixed caller-supplied filename (self-to-self exposure), so it relies on
  the lexical guard. If `read_file`'s `path` ever becomes agent/end-user-controlled rather
  than a fixed runner filename, add realpath confinement to the SSH path too.
- `SkyPilotBackend.read_file` is a known gap until the SkyPilot compute path; callers on
  the SSH backend are unaffected.
