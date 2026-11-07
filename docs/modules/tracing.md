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
The exact state, closed/active status and Python owner are also rechecked after callable
resolution, so a declared getter cannot close or reenter the invocation and then dispatch
through stale preflight state.

The owner waits through a private completion-notification Future rather than forwarding
the child's exception to an abandoned shield Future. Its callback only signals completion
and is always removed. The owner alone consumes the actual settled result, including late
failures during cancellation drain; the event loop receives no raw-error diagnostic from
an abandoned observer Future.

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

Node deliberately requires a genuine non-Proxy declared async exchange, excluding async
generators and ordinary promise factories, before ticket creation or invocation. Its intrinsic
outer promise is observed immediately; no caller constructor/species/then getter is executed
to rescue a rejected pending shape. Refused factories create no owned work and are not claimed
as drained. Delegate-internal detached/adopted work remains the application's responsibility.
This narrows the historical proposal's general Promise-return behavior to a safe closed local
boundary; it does not add a provider adapter or alter the portable rendering profile.

Reported metadata enforces coherent exact gap sets: request_not_dispatched iff the request
is absent, response_not_observed iff response is absent, and response_oversize iff the known
response has omitted oversize coverage. Transport/capture failures require a real request
handoff. Cancel/interruption may precede handoff; optional late_return_discarded is a
separately observed retired return, not inferred from the other DTO fields. The shared
[synthetic conformance fixture](../../tests/fixtures/prompts/local-attempts-v1.json) compares
actual local Python/Node calls, with only observer ID/time normalization for golden bytes.

## Structured ProviderClient projection values

`tracing.prompt_provider_values` and `prompt_provider_types` define a separate local
`forge-provider-client-input/v1` profile. They do not invoke a provider or install an observer.
Fixed Chat/Responses field tables preserve present-versus-absent values and omit/count
unselected top-level fields without reading their values. This is an allowlisted structured
projection, never a complete kwargs/HTTP-body hash or model/credential identity attestation.

Exact builtins become tagged values: null, bool, string, canonical decimal integer,
big-endian IEEE754 binary64 hex, ordered arrays and insertion-ordered object entry pairs.
Finite float1.0 differs from integer1 and negative float zero remains distinct. Nonfinite,
custom, cyclic and non-scalar Unicode values refuse projection without serializing them.
The modular portable encoder and its integer-only profile remain unchanged.

Effective ceilings combine native depth32, weighted tagged depth56 with8 levels reserved
for fixed framing,65,536 native value/key nodes,4,096 map fields,1,024 integer digits and
16MiB complete encoded projection. Native bounds do not promise every native shape fits
its tagged expansion. The existing portable JSON depth64/64MiB/scalar-string/ASCII-object-key
limits still apply; user keys become string values in entry pairs. Portable limit refusals
remain projection_limit, not unsupported content or a truncated/prefix hash. Returned opaque
objects stay opaque; no model_dump, credential configuration or error detail is read.

The closed metadata types qualify unknown served model, immutable revision, usage/cost,
render placement, retry relationships and vendor acceptance. Count/hash/size fields remain
null when unobserved, and settled admission holes require matching declared metadata loss.
These parsers issue no live capture handle. Only the explicitly injected facade below
observes ProviderClient; pure projection/metadata helpers alone do not instrument LLMClient.

Unknown top-level names are only checked to be exact built-in strings within the bounded
map. Their content/length/Unicode is intentionally not scanned or validated as projected
content, and unknown values are never read. Selected nested object keys retain charged
scalar validation. Existing portable depth/byte/safe-integer capacity refusals map to
projection_limit; malformed structure/Unicode/key/cycle refusals map to unsupported.

## Explicit ProviderClient facade

`await prepare_provider_observation(prepared_invocation)` accepts only the exact issued
PreparedInvocation public handle; parsed metadata cannot substitute. `wrap_provider(original,
observation=observation, provider_slot=...)` returns a fixed facade for explicit injection
into a legacy `LLMClient(provider_clients=...)`. The slot is declared, not verified vendor
identity. The facade copies no secret configuration, creates no default client and installs
no callback/exporter. It observes only nonstreaming acompletion/aresponses; source native
access and rendered-to-request placement remain unknown.

The same caller task invokes the same original method once with original argument values
and nested identities. The same raw response returns, and original exceptions/controls
propagate unchanged. Observation never retries, serializes parallel calls, changes budget/
cache/fallback policy, normalizes a response or vetoes a model return. Unsupported input,
projection limits, observer construction/finalization failures and capacity exhaustion
become bounded partial metadata/loss; they cannot replace the model call's outcome.

Auth/Responses auth, route formatting, model description and both stream iterators forward
unchanged without capture. Only those named methods are a supported facade: config/type
identity and arbitrary provider extensions are not cloned. `generate_one` keeps its existing
exact OpenAICompatProvider admission; use a separate original unwrapped client for generation.
Streaming, provider retry/tool-loop causal links, served model/revision and usage/cost are
not observed by this profile. A cache hit can return before the facade is entered.

The observation admits at most64 entries and reserves4KiB per active record plus a4KiB
header within256KiB; concurrent worst-case reservations may reach budget before count64.
Completed records release unused reservation and retain entry order despite completion
order. Overcapacity calls forward unobserved with a saturating loss count. Snapshot/close
refuse while admitted calls are active; closed or foreign-scope calls still forward without
content projection. Foreign scope sets only an explicit loss flag. Settled snapshots preserve
admission holes as metadata loss and have no claim of complete model/cache/stream coverage.
No recording lock is held over projection/delegation/await.

Each opaque observation/facade owns its private working state. Global issuance tracking
holds only weak references, and metadata state keeps weak loop/task references. Keeping a
metadata handle therefore does not root abandoned delegates, clients, callbacks or task/loop
cycles. Active calls still belong to the caller's native task; observation introduces no
finalizer, implicit close/cancel or fabricated terminal record when work is abandoned.
Safe repr/copy/pickle behavior does not make host-level private-slot introspection a security
boundary. Settled metadata contains no delegate configuration or original request/response.

A real caught cancellation is re-raised by identity. If the delegate itself swallows it and
returns, that original return still reaches the caller; increased pending cancellation marks
the observer's response retired and skips projection. Cancellation is not process death.
Caller partitions and hashes stay sensitive local assertions, with no persistence/export,
headless credentials, automatic ingestion or tenant grant implied.

Synthetic conformance uses the actual unchanged LLMClient with explicitly injected fake
providers: Chat/Responses builders, rich opaque SDK objects, cache hits, structured reprompts,
nonstreaming tool loops, retry/provider/model fallback, budget/parse failures and cancellation.
The facade's direct object/exception identity tests are distinct from LLMClient's existing
upstream normalization/mapping. The shared
[typed value fixture](../../tests/fixtures/prompts/provider-input-v1.json) declares native
integer-versus-binary64 recipes explicitly; independent Node test vectors do not imply a Node
provider adapter or universal numeric equivalence after arbitrary host JSON parsing. These
controls use no actual provider/network/model execution or secret configuration.

## Explicit native stream observation

`await prepare_stream_observation(prepared_invocation)` creates a body-free partial metadata
owner from the exact issued local PreparedInvocation. `wrap_streaming_provider(original,
observation=observation, provider_slot=...)` is a separate explicit ProviderClient injection
facade. It selects no provider/config, credentials, defaults or telemetry/export behavior.
The existing `wrap_provider` retains its transparent, unobserved stream iterator identity.
Nonstreaming calls/auth/route/description methods on the stream facade forward unchanged.

The opt-in facade observes only exact native Python async generators. Unsupported custom
iterators/values forward untouched with bounded loss and no admitted/native-iterator claim.
Iterator and advance identity/type/introspection are intentionally different. The fixed
views forward native anext/asend/athrow/aclose and advance await/next/send/throw/close operations.
Genuine items, original exceptions/control instances, native overlap/reuse refusals and
backpressure remain native. No observer task, shield, driver, retry, prefetch, buffering or
implicit close is added. Low-level awaited Futures are not counted as raw stream items.

Input uses the unchanged allowlisted structured-value projection, bound under a distinct
StreamInputBinding domain to actual astream/aresponses_stream and the corresponding normalized
acompletion/aresponses schema. The argument snapshot occurs before the first driven data
advance attempt; construction alone reads no input. Mutable arguments may change before
that point. An invalid native advance may refuse before the generator body executes, so
neither construction nor driving proves a provider request. Closing before any data drive
requires no input projection. Extra observer argument references release after the single
projection attempt or retirement; native-bound arguments retain native ownership.

Metadata records only bounded counts, operation outcomes and timestamps, never item content,
item hashes, response buffers, usage/cost or raw errors. First raw item drive latency excludes
construction delay but can include caller/manual-driving pauses and observer/scheduling
work. It is not provider TTFT: a raw Responses status event may precede text and the client
parser may emit later. Only observed StopAsyncIteration or normal native aclose completion
establishes local iterator end. Closing an advance awaitable is not an observed generator
aclose; native effects may still exhaust the generator. Until exhaustion or aclose completion
is actually observed, end remains unknown. Cancellation, an error, garbage collection,
retirement or inactivity establishes neither successful output nor confirmed provider/process death.

snapshot() returns copied/frozen partial metadata during active work. close() retires only
observation, freezes the ledger and clears extra input references through at most64 weak
issued-view references and fixed internal slot access. It calls no user/native operation,
cancels no work and waits for nothing. Late native items/controls still reach the caller
without updating retired metadata. View-owned private slots and weak issuance introduce no
global working-value root; keeping metadata does not retain abandoned generator/advance,
argument/error/callback/task/loop cycles. Native GC/finalization remains unobserved. Working
repr/copy/pickle controls are not protection against explicit host-level introspection.

Admission reserves 4 KiB per record plus a 4 KiB header inside 256 KiB. This permits an
effective 63 lifetime entries beneath the MAX_STREAMS count cap of 64; no limit is widened.
Entry numbers are never reused. A failed admission allocation removes any partly published
record and counts that reserved entry as metadata loss, keeping ledger holes coherent. Saturating2^53-1 counters have an explicit saturation flag. Unsupported, overbudget,
foreign-scope, closed or observer-failed capture never skips, retries or replaces the
original factory/protocol call. Lost admitted metadata remains an explicit ledger hole.
The SDK cannot infer cache/fallback/tool-loop causal links, served model/revision, remote
source access, HTTP wire bytes or complete coverage from these local observations. Existing
LLMClient parsing, tools and call policies stay authoritative; no alternate renderer/compiler,
provider adapter, automatic ingestion or Node streaming implementation is supplied.

Synthetic conformance also drives the unchanged LLMClient Chat/Responses parsers and tool
loop through explicitly injected native fake streams. These controls preserve missing-final
Responses refusal, upstream error mapping, streaming cache/retry/fallback bypass, executable
tools, declaration suspension with caller-owned resumption, loop errors and iteration limits.
That client behavior is distinct from direct facade value/control identity. The
[shared metadata fixture](../../tests/fixtures/prompts/provider-stream-v1.json) records only
fictional body-free snapshots; Python checks closed DTOs while private Node tests check the
same portable bytes/hashes. Those vectors supply no Node streaming/provider implementation.
