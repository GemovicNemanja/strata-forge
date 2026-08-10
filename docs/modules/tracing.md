# `strata_forge.tracing` — Langfuse observability layered above every other module

`strata_forge.tracing` is the cross-cutting Langfuse layer. It wraps every
other strata-forge module *from above* — no other `strata_forge.*` module
imports `strata_forge.tracing`, which keeps the dependency arrow clean, the
`[langfuse]` extra truly optional, and the unit-test surface free of
tracing concerns. See
[ADR 0008](../architecture/adr/0008-tracing-as-cross-cutting.md) for
the design rationale.

Five integration points make up the public surface:

- `install_litellm_callback()` — wire LiteLLM's built-in Langfuse
  callback so every LLM call through `strata_forge.llm` (or any other
  library that uses LiteLLM) auto-traces.
- `@traced` — wrap a sync or async function in a Langfuse trace.
- `traced_span()` — async context manager for nested observations.
- `score_trace` / `score_observation` — attach grader feedback to a
  trace by ID.
- `record_numeric_metric` / `record_categorical_metric` — attach
  application metrics (token counts, latencies, model selections) to
  a trace by ID.

When Langfuse isn't configured, every public function is a silent
no-op. Tracing failures are swallowed so they never break a
production call path.

Module rules: [`src/strata_forge/tracing/CLAUDE.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/tracing/CLAUDE.md).
Source: [`src/strata_forge/tracing/`](https://github.com/GemovicNemanja/strata-forge/tree/main/src/strata_forge/tracing/).

---

## Contents

- [Quickstart](#quickstart)
- [Configuration + the lazy import contract](#configuration--the-lazy-import-contract)
- [LiteLLM auto-tracing](#litellm-auto-tracing)
- [`@traced`](#traced)
- [`traced_span()`](#traced_span)
- [Scores](#scores)
- [Metrics](#metrics)
- [Correlation IDs](#correlation-ids)
- [Resilience and the no-op contract](#resilience-and-the-no-op-contract)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```python
from strata_forge.tracing import (
    install_litellm_callback,  # auto-trace LLM calls
    traced,                    # decorate non-LLM functions
    traced_span,               # async context manager for sub-spans
    score_trace,               # attach grader feedback
    record_numeric_metric,     # attach measurements
)

# Once at startup — auto-traces every subsequent LiteLLM call.
install_litellm_callback()

@traced
async def my_workflow(query: str) -> str:
    async with traced_span("preprocess"):
        normalized = query.lower()
    # ... LLM call here, auto-traced via the callback ...
    return f"answer for {normalized}"
```

For end-to-end runnable demos:

- [`examples/15_tracing_basic.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/15_tracing_basic.py)
  — `@traced` sync + async.
- [`examples/16_tracing_spans_and_scores.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/16_tracing_spans_and_scores.py)
  — `traced` + `traced_span` + score + metric.
- [`examples/17_tracing_litellm.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/17_tracing_litellm.py)
  — full path through `LLMClient` with the callback installed.

---

## Configuration + the lazy import contract

Tracing reads its credentials from
[`strata_forge.config.LangfuseConfig`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/config/settings.py):

| Env var | Field | Default |
|---|---|---|
| `LANGFUSE_HOST` | `host` | `http://localhost:3000` |
| `LANGFUSE_PUBLIC_KEY` | `public_key` | unset |
| `LANGFUSE_SECRET_KEY` | `secret_key` | unset |
| `LANGFUSE_TRACING_ENVIRONMENT` | `tracing_environment` | unset |

`LangfuseConfig.enabled` is `True` only when **both** keys are set.
`tracing_environment` is passed to the Langfuse constructor as
`environment=`, so traces from different deployments (production,
staging, a laptop) stay separable in the Langfuse UI. Leaving it unset
keeps the SDK on its built-in `default` environment.

The `langfuse` Python SDK is behind the `[langfuse]` extra. The
**lazy-import contract** is structural: `import strata_forge.tracing` works
without the extra installed, and `get_client()` returns `None` cleanly
in three cases — keys not configured, the SDK not installed, or a
defensive guard. Every public function in this module checks
`get_client()` and short-circuits when it returns `None`, so production
code paths are never broken by missing Langfuse setup.

`get_client()` caches the constructed client via
`functools.lru_cache(maxsize=1)`. `reset_client()` clears the cache —
used by the autouse fixture in `tests/unit/tracing/conftest.py` between
test cases.

---

## LiteLLM auto-tracing

`install_litellm_callback()` is the integration point that makes every
LiteLLM call auto-trace. Idempotent — calling it multiple times leaves
the callback registered exactly once:

```python
from strata_forge.tracing import install_litellm_callback

ok = install_litellm_callback()
# ok is True when Langfuse is configured and the callback is now
# registered (whether by this call or already); False when Langfuse
# isn't configured and the call was a no-op.
```

Internally this appends `"langfuse"` to `litellm.success_callback` and
`litellm.failure_callback`. LiteLLM's built-in callback then reads
`LANGFUSE_*` env vars on each call and ships traces.

`is_litellm_callback_installed()` is the diagnostic counterpart — call it
yourself to answer the "I configured Langfuse but I'm not seeing traces"
question. It reports `True` only when `"langfuse"` appears in **both**
lists; the half-installed asymmetric state reports `False` so the
diagnostic isn't misleading. `strata-forge doctor` does not check the
callback: it probes Langfuse host reachability and whether the keys are
set, nothing more.

The constant `LITELLM_CALLBACK_NAME` is the string LiteLLM matches
against. It's exposed so tests can pin the value; if LiteLLM ever
renames its built-in Langfuse callback, the edit is one line.

---

## `@traced`

Decorator for sync and async functions outside the LLM call path —
eval graders, agent loops, RAG pipelines, anywhere you want a named
trace + structlog correlation.

```python
@traced
async def my_workflow(x: int) -> int:
    ...

@traced(name="custom", tags=["v1", "experiment"])
def my_pipeline(x: int) -> int:
    ...
```

Two forms: bare `@traced` and parameterized `@traced(name=..., tags=...)`.
PEP 695 typevars + `@overload` preserve the wrapped function's
signature end-to-end. Dispatches on `inspect.iscoroutinefunction(func)`
to pick the right wrapper.

Lifecycle:

- On entry: opens a Langfuse trace with the given name + tags, sets
  `strata_forge.core.ids.correlation_id_var` to the trace ID.
- On normal return: `trace.update(output={"status": "ok"})`.
- On exception: `trace.update(output={"error": repr(exc)}, level="ERROR")`,
  then re-raises the original exception.
- On exit: restores the prior `correlation_id_var` value.

Tracing failures are swallowed: trace creation falls through to a
bare function call; trace.update errors don't break the return; an
update failure on the error path doesn't mask the real exception.

---

## `traced_span()`

Async context manager for nested observations under the active trace:

```python
async with traced_span("preprocessing") as span:
    # ... work here ...
    # `span` is the Langfuse span object, or None if Langfuse isn't
    # configured. Callers can call `span.update(metadata={...})` if
    # span is not None.
```

`traced_span` reads `correlation_id_var` to discover the active trace ID
and opens the observation with
`client.start_observation(name=..., as_type="span", trace_context={"trace_id": ...})`
— the Langfuse SDK v4 shape, which replaced the older `client.span(...)`
call and its top-level `trace_id` kwarg. When no trace is active, the
span is created without `trace_context` (Langfuse treats it as
standalone). The block runs to completion regardless of whether Langfuse
is configured.

On exit, the span records its elapsed wall-clock time in metadata. On
normal completion: `span.end(output={"status": "ok"}, metadata={"duration_ms": ...})`.
On exception: `span.end(output={"error": ...}, level="ERROR", ...)` then
re-raises. Span-end failures are swallowed.

Span nesting is intentionally flat: sequential `traced_span` calls inside
one `@traced` function appear as siblings under the trace, not nested
within each other. A span links to the trace, never to an enclosing
span.

---

## Scores

Scores are how grader output gets persisted to Langfuse. `strata_forge.evals`
deliberately does not call these helpers — it never imports
`strata_forge.tracing` (see
[ADR 0008](../architecture/adr/0008-tracing-as-cross-cutting.md)), so
pushing an eval verdict to Langfuse is an explicit call you make with a
`GraderResult` in hand. `score_trace` attaches to a trace;
`score_observation` attaches to a specific observation (span):

```python
await score_trace(trace_id, "helpfulness", 4.2)
await score_trace(trace_id, "verdict", "GOOD", comment="strong match")
await score_observation(observation_id, "step_accuracy", True)
```

Both helpers are async and accept any `ScoreValue` — `float`, `int`,
`str`, `bool`. Langfuse infers the wire-format `data_type` from the
Python type unless you supply `data_type="NUMERIC"` /
`"CATEGORICAL"` / `"BOOLEAN"` explicitly. Optional `comment` lets the
grader attach reasoning alongside the score.

Both are silent no-ops when Langfuse isn't configured and swallow
client errors via `contextlib.suppress`.

---

## Metrics

Adjacent to scores but with different intent: metrics capture *facts*
about a call (token count, latency, cost, which model was selected)
rather than grader judgments about quality. The underlying Langfuse
mechanism is the same `client.create_score` API that scores use, with
the wire-format `data_type` set explicitly, so downstream queries can
filter "what happened" from "how well it went":

```python
await record_numeric_metric("input_tokens", 512, trace_id=trace_id)
await record_numeric_metric("latency_ms", 234.5, observation_id=span_id)
await record_categorical_metric("model", "claude-opus-4-7", trace_id=trace_id)
```

At least one of `trace_id` / `observation_id` is required —
**`ValueError` is raised unconditionally** when both are missing, even
when Langfuse isn't configured. The bug surfaces in dev (no Langfuse)
instead of silently dropping data only in production.

---

## Correlation IDs

The active Langfuse trace ID is published on
[`strata_forge.core.ids.correlation_id_var`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/core/ids.py)
while a `@traced` function is running. The structlog logger
(configured by `strata_forge.core.logging.configure_logging`) has a
processor that injects this value into every log record under
`correlation_id`. The net effect: every log line emitted inside a
`@traced` function carries the trace ID without any manual plumbing.

`@traced` saves and restores the outer `correlation_id_var` value so
nested invocations don't leak. `traced_span` reads the var to link
its span but doesn't mutate it.

---

## Resilience and the no-op contract

Every public function in `strata_forge.tracing` follows the same two
invariants:

- **No-op when unconfigured.** If `get_client()` returns `None` (no
  keys, missing extra, or Langfuse construction failed), the
  function returns without touching anything. The caller's code path
  is unaffected.
- **Failures are swallowed.** Once a client exists, any error raised
  by the Langfuse SDK (network blip, schema mismatch, server down)
  is suppressed via `contextlib.suppress(Exception)`. Tracing is
  best-effort observability — it never breaks the wrapped logic.

The one deliberate exception: `record_*_metric` raises `ValueError`
when neither `trace_id` nor `observation_id` is supplied. That's a
programmer error, not a runtime tracing failure.

---

## Troubleshooting

**"I configured Langfuse but I'm not seeing traces."**
Most often the LiteLLM callback wasn't installed. Check with
`from strata_forge.tracing import is_litellm_callback_installed;
print(is_litellm_callback_installed())`. If it prints `False`, add
`install_litellm_callback()` once at process startup. (`strata-forge
doctor` will not tell you this — it only reports whether the keys are
set and whether the host answers.)

**`ImportError: ... requires the [langfuse] extra`.**
Install with `pip install strata-forge[langfuse]` (or
`uv sync --extra langfuse`). `strata_forge.tracing` itself imports without
the extra; only constructing or using the client requires it.

**Tracing crashes my production code.**
It shouldn't. Every public function no-ops cleanly when unconfigured
and swallows client failures. If you're seeing crashes, the most
likely cause is a `record_*_metric` call missing both `trace_id` and
`observation_id` — that one raises `ValueError` deliberately because
it's a programmer error. Catch the `ValueError` or supply a target.

**Traces appear under the wrong name.**
`@traced` defaults to the wrapped function's `__name__`. Use
`@traced(name="...")` to override. Anonymous (`lambda`) callables
land under `<lambda>` — wrap them or use `@traced(name="...")`.

**Spans aren't nested the way I expected.**
Sequential `traced_span` calls inside one `@traced` function appear
as siblings under the trace, not nested within each other. Every span
links to the trace, never to an enclosing span. Organize sub-work as
sibling spans with distinct names.

**Correlation ID shows up as `None` in my logs.**
The `correlation_id_var` is only set inside a `@traced` function (or
when you explicitly call `strata_forge.core.ids.set_correlation_id`).
Outside those, log records emit with `correlation_id=null`. If you
want a request-scoped correlation ID independent of Langfuse, set
one manually at the start of your handler.

---

## See also

- [`strata_forge.llm`](llm.md) — the client whose calls auto-trace once
  `install_litellm_callback()` has run.
- [`strata_forge.core`](core.md) — `correlation_id_var`, `set_correlation_id`,
  and the structlog processor that stamps the ID onto every log record.
- [`strata_forge.evals`](evals.md) — where grader output comes from when you
  want to push it through `score_trace`.
- [`strata_forge.config`](config.md) — how `LANGFUSE_*` env vars reach
  `LangfuseConfig`.
- [ADR 0008](../architecture/adr/0008-tracing-as-cross-cutting.md) — why
  tracing wraps the other modules from above instead of being imported by
  them.
