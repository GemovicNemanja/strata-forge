# Agent rules — forge.tracing

`forge.tracing` is the Langfuse observability layer. It wraps every
other module *from above* — no other `forge.*` module imports
`forge.tracing`. See
[ADR 0008](../../../docs/architecture/adr/0008-tracing-as-cross-cutting.md)
for the rationale.

## Purpose

Langfuse-based observability:

- **Auto-tracing for LLM calls** via LiteLLM's Langfuse callback,
  registered once at process startup by
  :func:`install_litellm_callback`. Every subsequent LiteLLM call —
  including those from third-party libraries that don't know about
  Forge — appears in Langfuse without `forge.llm` knowing.
- **`@traced` decorator** for sync + async functions outside the LLM
  call path (eval graders, agent loops, RAG pipelines).
- **`traced_span()` async context manager** for nested observations
  inside a trace.
- **Score + metric helpers** for attaching grader feedback and
  arbitrary application metrics to traces or observations.
- **`correlation_id` integration**: structlog records carry the active
  Langfuse trace ID via the existing
  :data:`forge.core.ids.correlation_id_var` ContextVar.

## Boundaries

- **Owns:** `client.py`, `litellm_callback.py`, `decorator.py`,
  `span.py`, `score.py`, `metrics.py`.
- **Imports from inside `forge`:** `forge.core` (logger,
  correlation_id), `forge.config` (LangfuseConfig), `forge.llm`
  (message/response types — for serializing trace payloads),
  `forge.prompts` (later — for prompt metadata on traces).
- **Imports of `langfuse`** happen lazily inside the functions that
  need them. The package is behind the `[langfuse]` extra; importing
  `forge.tracing` without the extra MUST work.
- **Reverse imports are banned.** `forge.llm`, `forge.prompts`, and
  every downstream module MUST NOT import from `forge.tracing`. This
  is the cross-cutting design from ADR 0008.

## Public API

The module's `__init__.py` re-exports the supported surface:

- `get_client() -> Langfuse | None` — the cached singleton.
- `install_litellm_callback() -> None` — wire LiteLLM ↔ Langfuse.
- `@traced(name=None, tags=...)` — decorator for sync + async fns.
- `traced_span(name, **fields)` — async context manager.
- `score_trace`, `score_observation` — attach feedback to a trace.
- `record_numeric_metric`, `record_categorical_metric` — attach
  arbitrary application metrics.

Errors raised from this module are `ForgeError` subclasses; the
no-Langfuse-configured path is a silent no-op (never a crash).

## Internal patterns

- **Lazy Langfuse import.** Every function that touches the Langfuse
  SDK imports it inside the function body. `forge.tracing` imports
  cleanly without the extra installed.
- **`get_client()` is cached.** Once built, the singleton sticks for
  the lifetime of the process. Tests use `reset_client()` in conftest
  to clear it between cases.
- **No-op when unconfigured.** `LangfuseConfig.enabled` is False when
  either credential is missing. Every public function checks
  `get_client()` and short-circuits cleanly — no warnings, no
  crashes, no fallback prints. The diagnostic (`forge doctor`) is
  where missing configuration surfaces.
- **Correlation IDs propagate.** When a trace starts, the trace ID is
  written into `forge.core.ids.correlation_id_var` so subsequent log
  records carry it. The contextvar is restored on trace exit.
- **`install_litellm_callback()` is idempotent.** Calling it multiple
  times doesn't duplicate the callback. The check is by string
  identifier in `litellm.success_callback` and
  `litellm.failure_callback`.

## Test expectations

- Unit tests under `tests/unit/tracing/`, one file per source module.
- Coverage target: ≥ 90 % line.
- Mocked Langfuse client for every unit test — no live network.
- One `@pytest.mark.integration` test runs against the
  docker-compose Langfuse stack: starts a trace, runs an LLM call,
  retrieves the trace via the Langfuse API, verifies shape. Skipped
  unless `make stack-up` has the stack running and `LANGFUSE_*` env
  vars point at it.

## Gotchas

- **Forgetting to call `install_litellm_callback()`.** No traces
  surface. This is the most common reason a Forge user reports
  "tracing doesn't work." `forge doctor` reports whether the
  callback is installed.
- **Eager import of `langfuse` at module top.** Breaks the
  no-extras-installed import contract. Every Langfuse-touching
  function lazy-imports inside its body.
- **Not no-op-ing when Langfuse isn't configured.** Production calls
  should never crash because Langfuse keys are missing. Every public
  function checks `get_client()` and exits early when it returns
  `None`.
- **Reverse imports.** A module under `forge.llm` reaching for
  `forge.tracing` violates ADR 0008. The dependency arrow points one
  way.

## When to update this file

- Adding a new public function to `__init__.py`.
- Adding a new file under `src/forge/tracing/`.
- Changing the lazy-import pattern.
- Changing what counts as "configured" (currently `enabled` on
  `LangfuseConfig`).
- Adding a new dependency to the `[langfuse]` extra.
