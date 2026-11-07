# `strata_forge.tracing` — Langfuse observability layered above every other module

`strata_forge.tracing` is the cross-cutting Langfuse layer. It wraps every
other Forge module *from above* — no other `strata_forge.*` module imports
`strata_forge.tracing`, which keeps the dependency arrow clean, the
`[langfuse]` extra truly optional, and the unit-test surface free of
tracing concerns. See
[ADR 0008](../architecture/adr/0008-tracing-as-cross-cutting.md) for
the design rationale.

Five integration points ship in Phase 2.2:

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

Module rules: [`src/strata_forge/tracing/CLAUDE.md`](../../src/strata_forge/tracing/CLAUDE.md).
Source: [`src/strata_forge/tracing/`](../../src/strata_forge/tracing/).

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

- [`examples/14_tracing_basic.py`](../../examples/14_tracing_basic.py)
  — `@traced` sync + async.
- [`examples/15_tracing_spans_and_scores.py`](../../examples/15_tracing_spans_and_scores.py)
  — `traced` + `traced_span` + score + metric.
- [`examples/16_tracing_litellm.py`](../../examples/16_tracing_litellm.py)
  — full path through `LLMClient` with the callback installed.

---

## Configuration + the lazy import contract

Tracing reads its credentials from
[`strata_forge.config.LangfuseConfig`](../../src/strata_forge/config/settings.py):

| Env var | Field | Default |
|---|---|---|
| `LANGFUSE_HOST` | `host` | `http://localhost:3000` |
| `LANGFUSE_PUBLIC_KEY` | `public_key` | unset |
| `LANGFUSE_SECRET_KEY` | `secret_key` | unset |

`LangfuseConfig.enabled` is `True` only when **both** keys are set.

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

`is_litellm_callback_installed()` is the diagnostic counterpart;
`strata-forge doctor` uses it to surface the "I configured Langfuse but I'm
not seeing traces" failure mode. It reports `True` only when
`"langfuse"` appears in **both** lists — the half-installed asymmetric
state reports `False` so the diagnostic isn't misleading.

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

`traced_span` reads `correlation_id_var` to discover the active trace
ID and passes it as `trace_id` to `client.span(...)`. When no trace is
active, the span is created without a `trace_id` (Langfuse treats it
as standalone). The block runs to completion regardless of whether
Langfuse is configured.

On exit, the span records its elapsed wall-clock time in metadata. On
normal completion: `span.end(output={"status": "ok"}, metadata={"duration_ms": ...})`.
On exception: `span.end(output={"error": ...}, level="ERROR", ...)` then
re-raises. Span-end failures are swallowed.

Span nesting is intentionally flat in Phase 2 — sequential
`traced_span` calls inside one `@traced` function appear as siblings
under the trace, not nested within each other. Nested-spans-within-
spans is a later refinement.

---

## Scores

Scores are the way the eval module (Phase 2.4) will persist grader
output to Langfuse. `score_trace` attaches to a trace; `score_observation`
attaches to a specific observation (span):

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
mechanism is the same `client.score` API with the wire-format
`data_type` set explicitly, so downstream queries can filter "what
happened" from "how well it went":

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
[`strata_forge.core.ids.correlation_id_var`](../../src/strata_forge/core/ids.py)
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
Most often the LiteLLM callback wasn't installed. Run `strata-forge doctor`
to check; it reports whether `is_litellm_callback_installed()` is
True. If not, add `install_litellm_callback()` once at process
startup.

**`ImportError: ... requires the [langfuse] extra`.**
Install with `pip install ai-forge[langfuse]` (or
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
as siblings under the trace, not nested within each other. This is
intentional for Phase 2; nesting spans within spans is a later
refinement. For now, organize sub-work as sibling spans with
distinct names.

**Correlation ID shows up as `None` in my logs.**
The `correlation_id_var` is only set inside a `@traced` function (or
when you explicitly call `strata_forge.core.ids.set_correlation_id`).
Outside those, log records emit with `correlation_id=null`. If you
want a request-scoped correlation ID independent of Langfuse, set
one manually at the start of your handler.

## Explicit local prompt-attempt evidence

`strata_forge.tracing.prompt_attempts` is an opt-in, local in-memory observer above the
accepted prompt consumer. It never calls Langfuse/get_client, resolves credentials,
installs callbacks, exports records or supplies a provider/HTTP transport. Existing
Langfuse configuration and disabled no-op behavior remain unchanged.

`await prepare_invocation(consumed, values)` revalidates supplied local bytes and uses
the existing typed renderer. `await capture_once(invocation, body, transport,
parents=AttemptParents(...), declared_baseline=pin_or_none)` invokes one explicitly
supplied application byte behavior. It does not serialize request settings, route,
retry, fall back, execute tools, or claim provider wire/model use. The first profile is
qualified `caller_bytes` and `sdk_local_delegate_bytes/v1`; provider, model, revision,
usage, cost and render projection are explicitly unknown. A caller baseline supports
same/changed-byte comparison, not compiler or source attestation.

Working invocation/response/exchange objects have body-safe repr and opaque private
storage. Bodies are available through explicit accessors; only metadata snapshots
serialize. The sole retention mode is metadata_only: no bindings, headers, raw bodies
or raw exceptions enter the ledger. Original request/response byte hashes are distinct
from the uncomputed redacted hash. A hash is not anonymization or a remote grant.

Python creates exactly one owned child for each byte call, shields it, cancels once
and drains before releasing its active slot under repeated parent cancellation. Late
returns after retirement are discarded without body inspection or success. A supplied
baseline with no handoff uses the explicit not_observed relation. Close and terminal
snapshots refuse with capture_in_progress while the owned call is still active; this
avoids fabricating an incomplete terminal ledger. Snapshot reads and repeated close
are otherwise stable. Cancellation is not proof of provider/native process death.

The declared exchange callable is resolved once before reserving an attempt; missing,
noncallable or failed lookup refuses without a ticket. No other transport configuration
is inspected. The child checks its exact active generation and owner cancellation before
handoff, including cancellation queued before the child first runs. Owner cancellation
is checked again before a returned response can become success.

Limits are 64 attempts, 16 MiB per body, 4 KiB per attempt metadata and 256 KiB per
invocation metadata, with reservation before dispatch. Oversized request refuses before
calling; oversized response has an explicit omitted-coverage capture error. These are
local prototype bounds, not scientific or production quotas. The metadata JSON fits
the existing portable profile; arbitrary request JSON/fractional settings are opaque
bytes and are never hashed by parsing/reserializing that integer-only profile.

The Python local recorder provides no actual ProviderClient observation, streaming, trace
export/ingestion or service authorization. Provider/remote trace provenance is unavailable
through this API.

The private Node package exposes the equivalent explicit local wrapper, without importing
Python tracing or configuring an exporter. It owns one native delegate promise/controller
until settlement after abort, throws a fixed standard-name AbortError, and discards late
returns without inspection. Its pure DTO validators cannot create live handles. Working
objects refuse JSON; structured cloning yields an unusable empty object, not copied private
state. Under-limit byte bodies are copied. Oversize responses record known size with omitted
coverage without copying/hashing the oversize body. See the
[private package reference](../../packages/prompt-sdk/README.md) for the exact native-carrier,
promise and AbortSignal boundary.
