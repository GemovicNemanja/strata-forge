# Agent rules — forge.compute

`forge.compute` is the remote-compute orchestration layer. It owns
typed task / job / status shapes, a YAML task loader, and a
:class:`Backend` Protocol with concrete in-process, SSH, and
SkyPilot implementations. See
[ADR 0013](../../../docs/architecture/adr/0013-compute-task-and-backend-shapes.md)
for the task-as-data + Protocol design rationale.

## Purpose

- :class:`Task` + :class:`ResourceSpec` — declarative Pydantic
  shapes describing what to run and what hardware it wants.
- YAML loader (``Task.from_yaml``, ``Task.from_yaml_str``,
  ``Task.to_yaml``) for the SkyPilot-task-YAML subset.
- :class:`Job` + :class:`JobStatus` — opaque handle plus a
  five-state lifecycle (``pending``, ``running``, ``succeeded``,
  ``failed``, ``cancelled``).
- :class:`Backend` Protocol — async ``submit`` / ``status`` /
  ``logs`` / ``cancel`` / ``cleanup``.
- :class:`LocalBackend` — in-process subprocess implementation
  (5.1). SSH + SkyPilot backends arrive in Phase 5.2; a batch
  inference runner that consumes them lands alongside.

## Boundaries

- **Owns:** ``task.py``, ``job.py``, ``backends/`` (local now;
  ssh / skypilot in 5.2), ``batch.py`` (5.2), helper YAML /
  lifecycle utilities.
- **Imports from inside ``forge``:** :mod:`forge.core` (errors,
  ids), :mod:`forge.config` (settings — backends pick up host /
  credentials from there). May import :mod:`forge.llm` for the
  batch inference runner (5.2).
- **Does NOT import** :mod:`forge.tracing`, :mod:`forge.agents`,
  :mod:`forge.evals`, :mod:`forge.rag`, :mod:`forge.datasets`,
  :mod:`forge.training`. The dependency arrow points downward —
  higher-level modules consume :mod:`forge.compute`, not the
  other way around.
- **External deps:** Pydantic + PyYAML in the core install (both
  already in core). ``asyncssh`` and ``skypilot`` are behind the
  ``[compute]`` extra; both are lazy-imported inside the backend
  constructors so ``import forge.compute`` works without them.

## Public API

The module's ``__init__.py`` re-exports:

- Data shapes: :class:`Task`, :class:`ResourceSpec`,
  :class:`Job`, :class:`JobStatus`, :data:`JobState`.
- Protocol: :class:`Backend`.
- Backend implementations: :class:`LocalBackend` (5.1);
  :class:`SSHBackend`, :class:`SkyPilotBackend` (5.2).

Errors raised from this module are :class:`ForgeError` subclasses
or :class:`ValueError` for input validation.

## Internal patterns

- **Tasks are declarative data.** No function decorators, no
  callable-as-task; the task is a frozen Pydantic instance that
  serializes to YAML/JSON. Backends produce :class:`Job` handles
  from :class:`Task` instances; everything else flows through
  those two shapes.
- **Backends own their own scheduling.** :class:`Backend` is a
  thin Protocol; SkyPilot's queue, SSH's PID-file-on-host, and
  the local subprocess dict aren't unified — each backend uses
  whatever's native to its substrate.
- **Lifecycle is mandatory.** Every backend implements every
  method on the Protocol. Methods that don't make sense for a
  backend (e.g. ``cleanup`` on a serverless backend with no
  cleanup work) raise :class:`NotImplementedError` rather than
  silently passing — callers can catch it explicitly.
- **Five canonical states.** ``pending`` / ``running`` /
  ``succeeded`` / ``failed`` / ``cancelled``. Backend-specific
  nuance (SkyPilot's ``SETTING_UP``, SSH's
  process-exists-vs-writers-closed) lands in
  :attr:`JobStatus.message` rather than expanding the state set.
- **Frozen tuple-typed collections** as elsewhere in
  :mod:`forge.*` (datasets, evals, agents, rag). Pydantic
  ``frozen=True`` plus ``extra="forbid"``.

## Test expectations

- Unit tests under ``tests/unit/compute/``, one file per source
  module.
- Coverage target: ≥ 90 % line.
- :class:`LocalBackend` is exercised with real subprocesses
  (cheap, deterministic).
- Phase 5.2 backends use ``sys.modules``-injected fake
  ``asyncssh`` / ``sky.api.sdk`` for unit tests; one
  ``@pytest.mark.integration`` test per backend exercises a live
  target.

## Gotchas

- **Don't let backends import each other.** Each
  ``backends/*.py`` is self-contained — adding a feature to SSH
  shouldn't require touching SkyPilot. Shared utility code goes
  in a private ``_helpers.py`` only if at least two backends
  need it.
- **Don't put heavy lifting in the constructor.** Backends are
  often constructed in dependency-injection paths where construction
  must be cheap; defer SDK imports and network calls until first
  use (``_get_client`` pattern).
- **Don't expose the underlying SDK in the public surface.**
  Backends accept SDK clients via an explicit ``client=`` kwarg
  for tests and for users who want to share connections, but the
  return type is always Forge-owned shapes (:class:`Job`,
  :class:`JobStatus`) — never an SDK object.
- **Never block the event loop in async methods.** Subprocess
  spawning goes through ``asyncio.create_subprocess_exec``;
  ``asyncssh`` is async-native; SkyPilot's sync calls (when
  needed) wrap in ``asyncio.to_thread``.

## When to update this file

- Adding a new backend.
- Adding a new optional dep to ``[compute]``.
- Changing the lifecycle state machine.
- Adding a new top-level public class/function to ``__init__.py``.
