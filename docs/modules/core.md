# `strata_forge.core` — cross-cutting primitives

`strata_forge.core` is the base of the dependency graph. Every other `strata_forge.*` module
imports from it; it imports from none of them. That rule is what keeps the package importable in
any order, keeps optional extras genuinely optional, and makes the module safe to depend on from
anywhere without thinking about cycles. Its only external dependencies are `structlog`,
`tenacity`, `anyio`, and the standard library.

Nothing here knows about LLMs. These are the primitives the LLM-shaped modules are built out of:

- **Errors** — the `ForgeError` hierarchy every Forge-raised exception inherits from
  ([ADR 0003](../architecture/adr/0003-exception-hierarchy.md)).
- **`@retry`** — a tenacity-backed decorator that works on sync and async callables, with
  predicates keyed to the exception hierarchy above.
- **Logging** — one-call structlog setup plus a `correlation_id` that rides a `ContextVar` across
  `await` boundaries so log lines correlate without manual plumbing.
- **Budgets** — `BudgetContext`, an async context manager enforcing USD and token ceilings, with
  nesting.
- **Reproducibility** — `set_seed`, `content_hash`, `env_snapshot` for run provenance.
- **IDs** — a hand-rolled RFC 9562 `uuid7()` (no third-party dependency) and correlation-ID
  helpers built on it.
- **Types** — the two shared aliases, `JSONValue` and `PathLike`.

Module rules:
[`src/strata_forge/core/CLAUDE.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/core/CLAUDE.md).
Source:
[`src/strata_forge/core/`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/core/).

---

## Contents

- [Quickstart](#quickstart)
- [Public API](#public-api)
- [The ForgeError hierarchy](#the-forgeerror-hierarchy)
- [Retry](#retry)
- [Logging](#logging)
- [Correlation IDs](#correlation-ids)
- [Budgets](#budgets)
- [Reproducibility](#reproducibility)
- [IDs](#ids)
- [Shared types](#shared-types)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```python
import asyncio

from strata_forge.core import BudgetContext, BudgetExceededError, configure_logging, get_logger


async def main() -> None:
    configure_logging(level="INFO")
    log = get_logger("demo")

    async with BudgetContext(max_usd=0.05) as budget:
        await budget.consume(usd=0.04, tokens=1200)
        log.info("spend", usd=budget.spent_usd, remaining_usd=budget.available_usd())
        try:
            await budget.consume(usd=0.02)
        except BudgetExceededError as exc:
            log.warning("budget.blocked", limit_usd=exc.limit_usd, would_spend=exc.spent_usd)


asyncio.run(main())
```

The second `consume` projects `0.04 + 0.02 = 0.06` against a `0.05` ceiling, so it raises and
records nothing — `budget.spent_usd` is still `0.04` afterwards.

---

## Public API

`strata_forge.core.__init__` re-exports a curated surface. Anything not listed here is internal,
even when it is importable from a submodule.

| Symbol | Defined in | What it is |
|---|---|---|
| `ForgeError` | `errors.py` | Root of the exception hierarchy. |
| `ConfigError` | `errors.py` | Missing, malformed, or contradictory configuration. |
| `ProviderError` + 6 subclasses | `errors.py` | Normalized provider failures. |
| `BudgetExceededError` | `errors.py` | A cost or token ceiling would be breached. |
| `ValidationError` | `errors.py` | Forge-side input/output validation failure. |
| `CacheError` | `errors.py` | Cache backend operation failed. |
| `RegistryError` | `errors.py` | Model registry lookup or validation failure. |
| `FallbackExhaustedError` | `errors.py` | Every entry in a fallback chain failed. |
| `retry` | `retry.py` | Decorator for sync and async callables. |
| `DEFAULT_RETRY_ON` | `retry.py` | The default retryable exception tuple. |
| `configure_logging` | `logging.py` | Process-wide structlog setup. |
| `get_logger` | `logging.py` | Return a structlog logger. |
| `traced_span` | `logging.py` | Sync context manager emitting `start`/`end` log records. |
| `BudgetContext` | `budget.py` | Async context manager enforcing spend ceilings. |
| `current_budget` | `budget.py` | The innermost active `BudgetContext`, or `None`. |
| `set_seed` | `repro.py` | Seed `random`, and `numpy`/`torch` when installed. |
| `content_hash` | `repro.py` | Stable SHA-256 over a JSON-canonical form. |
| `env_snapshot` | `repro.py` | Flat dict of runtime + package versions for provenance. |
| `uuid7` | `ids.py` | Time-ordered UUIDv7. |
| `new_correlation_id` | `ids.py` | Fresh correlation-ID string (UUIDv7 hex). |
| `get_correlation_id` | `ids.py` | Current correlation ID, or `None`. |
| `set_correlation_id` | `ids.py` | Set it; returns a `Token` for restoring. |
| `correlation_id_var` | `ids.py` | The raw `ContextVar`, exported for `reset`. |
| `JSONValue`, `PathLike` | `types.py` | Shared type aliases. |

One deliberate asymmetry: `add_correlation_id`, the structlog processor, is public in
`strata_forge.core.logging` but is **not** re-exported from the package. Import it as
`from strata_forge.core.logging import add_correlation_id` — you only need it if you are building
your own structlog pipeline instead of calling `configure_logging`.

---

## The ForgeError hierarchy

Every exception Forge raises deliberately inherits from `ForgeError`, so a caller can write one
`except ForgeError` for catch-all behaviour and still pattern-match subclasses for finer control.
Provider SDK and LiteLLM exceptions never bubble out of `strata_forge.llm`; they are normalized at
the seam in `strata_forge.llm.errors.map_litellm_exception` and chained via `raise ... from`, so
the original is always reachable on `__cause__`.

```
ForgeError
├── ConfigError               # source: the offending env var / YAML path
├── ProviderError             # model, provider, status_code
│   ├── ProviderAuthError
│   ├── ProviderRateLimitError
│   ├── ProviderTimeoutError
│   ├── ProviderBadRequestError
│   ├── ProviderServerError
│   └── ProviderContentFilterError
├── BudgetExceededError       # limit_usd, limit_tokens, spent_usd, spent_tokens
├── ValidationError
├── CacheError                # backend
├── RegistryError             # model, provider, reason
└── FallbackExhaustedError    # causes: [(model, provider, error), ...]
```

Structured attributes are the point. Each class carries the fields a caller needs to act on the
failure, and renders them into `str(exc)` so a log line is self-describing:

| Class | Attributes | `str(exc)` renders as |
|---|---|---|
| `ConfigError` | `source` | `missing key (source: LANGFUSE_SECRET_KEY)` |
| `ProviderError` | `model`, `provider`, `status_code` | `bad key (model=gpt-5.5, provider=openai, status=401)` |
| `RegistryError` | `model`, `provider`, `reason` | `Unknown model (reason=unknown_model, model=gpt-9)` |
| `CacheError` | `backend` | `set failed (backend: redis)` |
| `BudgetExceededError` | `limit_usd`, `limit_tokens`, `spent_usd`, `spent_tokens` | plain message |
| `FallbackExhaustedError` | `causes` | message plus one indented line per failed attempt |

`RegistryError.reason` is a short machine-readable tag rather than free text, so callers can
branch on it: `"unknown_model"`, `"missing_pricing"`, `"unsupported_route"`,
`"capability_missing"`, `"capability_unknown"`.

`FallbackExhaustedError.causes` is a list of `(model, provider, error)` triples in attempt order.
`provider` is `None` for entries that failed before route resolution.

Two classifications are load-bearing elsewhere and should not be changed casually. Retry
predicates select on `ProviderRateLimitError` / `ProviderTimeoutError` / `ProviderServerError`,
and fallback chains treat `ProviderContentFilterError` as terminal for the whole chain — the same
content will be refused by every other provider, so advancing is wasted spend. Changing the
hierarchy therefore changes retry and fallback behaviour, which is why
[ADR 0003](../architecture/adr/0003-exception-hierarchy.md) governs it.

`strata_forge.core.errors.ValidationError` is distinct from Pydantic's `ValidationError` on
purpose, so callers can catch Forge-validated failures specifically. When wrapping a Pydantic
error, chain it: `raise ValidationError(...) from pydantic_error`.

---

## Retry

`@retry` is a thin, typed wrapper over tenacity that supplies Forge-aware defaults. It works on
both `def` and `async def` callables — tenacity dispatches on the wrapped function — and the
PEP 695 typevars preserve the signature, so the decorated callable type-checks exactly like the
original.

```python
from strata_forge.core import DEFAULT_RETRY_ON, ProviderTimeoutError, retry


@retry
async def fetch() -> str: ...


@retry(max_attempts=3, initial_wait=0.5, max_wait=10.0, retry_on=(ProviderTimeoutError,))
def fetch_sync() -> str: ...
```

| Parameter | Default | Meaning |
|---|---|---|
| `max_attempts` | `5` | Total attempts, including the first call. |
| `initial_wait` | `1.0` | Base seconds for exponential backoff. |
| `max_wait` | `30.0` | Cap on the wait between attempts. |
| `retry_on` | `DEFAULT_RETRY_ON` | Exception types that trigger a retry. |

The wait policy is `tenacity.wait_exponential_jitter(initial=initial_wait, max=max_wait)`, and
`reraise=True` is set — when the attempts run out you get the last real exception, not a
`RetryError` wrapper. That matters: callers upstream still pattern-match on
`ProviderRateLimitError`.

`DEFAULT_RETRY_ON` is `(ProviderRateLimitError, ProviderTimeoutError, ProviderServerError)` —
transient failures only. `ProviderAuthError`, `ProviderBadRequestError`, and
`ProviderContentFilterError` are excluded deliberately: the request itself is the problem, so
retrying burns time and money without changing the outcome.

The decorator retries on exception *type*, not on return value. A function that swallows an error
and returns a sentinel will never be retried.

---

## Logging

`configure_logging` is called once per process — by your application entry point, a notebook
cell, or a test fixture. Nothing inside `strata_forge` calls it for you, so a library consumer
that never calls it gets structlog's stock configuration.

```python
from strata_forge.core import configure_logging, get_logger

configure_logging(level="INFO")          # json=None -> auto-detect
log = get_logger("myapp.ingest")
log.info("ingest.start", rows=1200, source="s3://bucket/key")
```

| Parameter | Default | Behaviour |
|---|---|---|
| `level` | `"INFO"` | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`. An unrecognized name silently falls back to `INFO`. |
| `json` | `None` | `None` auto-detects from `sys.stdout.isatty()`: pretty for a terminal, one JSON object per line otherwise. `True`/`False` force it. |

The processor chain is log level, then `add_correlation_id`, then an ISO-8601 UTC timestamp,
stack-info rendering, exception formatting, and finally either `JSONRenderer` or
`ConsoleRenderer(colors=True)`.

Two conventions the rest of the codebase follows, worth adopting in your own call sites: the
first positional argument is a dotted **event name** (`"ingest.start"`), not a sentence, and
everything else is a keyword — structured fields survive JSON rendering, interpolated strings do
not. Full prompts and responses go at `DEBUG`, never `INFO`, and credentials never go anywhere.

`traced_span` is the logging-side span helper: a **synchronous** context manager that emits
`<name>.start` and `<name>.end` records with `elapsed_s`, or `<name>.error` with `error_type` and
`error` before re-raising. Extra keyword arguments are attached to every record in the span.

```python
from strata_forge.core import traced_span

with traced_span("chunking", doc_id="d-17"):
    ...
```

> `strata_forge.core.traced_span` and `strata_forge.tracing.traced_span` are different tools with
> the same name. The core one is sync, writes structlog records, and needs no configuration. The
> tracing one is an **async** context manager that opens a Langfuse observation. Import one or the
> other explicitly; do not let them alias each other in a module that uses both.

---

## Correlation IDs

This is the mechanism most often misunderstood, so it is worth stating precisely.

`correlation_id_var` is a module-level `ContextVar[str | None]` declared in `ids.py` with a
default of `None`. `configure_logging` installs `add_correlation_id` as a structlog processor;
that processor reads the variable at the moment each record is emitted and, when it is not `None`,
adds a `correlation_id` key to the record. Nothing else is involved. There is no logger state, no
thread-local, and no argument to pass down a call stack.

```python
import asyncio

from strata_forge.core import (
    configure_logging,
    correlation_id_var,
    get_logger,
    new_correlation_id,
    set_correlation_id,
)

configure_logging(level="INFO", json=True)
log = get_logger("worker")


async def step() -> None:
    await asyncio.sleep(0)
    log.info("step.done")  # carries correlation_id without being told it


async def handle_request() -> None:
    token = set_correlation_id(new_correlation_id())
    try:
        await asyncio.gather(step(), step())
    finally:
        correlation_id_var.reset(token)


asyncio.run(handle_request())
```

Propagation follows Python's contextvars rules exactly, which produce four distinct behaviours:

| Boundary | Does the value propagate in? | Does a `set` inside leak out? |
|---|---|---|
| `await some_coro()` in the same task | Yes | **Yes** — same context object |
| `asyncio.create_task` / `gather` / `TaskGroup` | Yes, the child copies the context | No |
| `anyio.to_thread.run_sync` | Yes — anyio copies the context into the worker | No |
| `loop.run_in_executor`, bare `threading.Thread` | **No** — the value is `None` in the thread | No |

The second row is the useful one: a task started inside a correlation scope inherits the ID, and
anything it sets stays local. The first row is the surprising one: directly awaiting a coroutine
runs it in the *caller's* context, so a `set_correlation_id` inside that coroutine is still in
effect after the `await` returns. Helpers that set the ID should therefore always restore it,
which is what the `Token` returned by `set_correlation_id` is for.

Who sets it in practice:

- **You**, at the top of a request handler, CLI invocation, or batch job, via
  `set_correlation_id(new_correlation_id())`.
- **`strata_forge.tracing.traced`**, which sets it to the active Langfuse trace ID on entry and
  restores the previous value on exit. Every log line emitted inside a `@traced` function
  therefore carries the trace ID, which is what lets you jump from a log line to the trace.
- Nothing else. `strata_forge.llm` only *reads* it, stamping `get_correlation_id()` onto each
  NDJSON diagnostic record so dumps join back to logs and traces.

Because IDs are UUIDv7, they are k-sortable: the first 48 bits are a millisecond timestamp, so
sorting correlation IDs lexicographically sorts them by creation time.

---

## Budgets

`BudgetContext` is an async context manager that enforces a USD ceiling, a token ceiling, or
both, over a block of code. It is generic — anything expensive can charge against it — but in
practice the caller is `strata_forge.llm.LLMClient`.

```python
import asyncio

from strata_forge.core import BudgetContext, current_budget


async def main() -> None:
    async with BudgetContext(max_usd=1.00, max_tokens=500_000) as outer:
        assert current_budget() is outer
        async with BudgetContext(max_usd=0.10) as inner:
            await inner.consume(usd=0.04, tokens=1_000)
            # inner.spent_usd == 0.04 and outer.spent_usd == 0.04


asyncio.run(main())
```

| Member | Signature | Notes |
|---|---|---|
| `BudgetContext(...)` | `*, max_usd=None, max_tokens=None, isolated=False` | `None` means "track but do not enforce". |
| `consume` | `async (*, usd=None, tokens=None) -> None` | Check-then-commit across the chain. |
| `available_usd` | `() -> float \| None` | Remaining headroom; `None` with no USD ceiling. |
| `available_tokens` | `() -> int \| None` | Same for tokens. |
| `spent_usd`, `spent_tokens` | attributes | Running totals for this budget only. |
| `current_budget()` | `() -> BudgetContext \| None` | The innermost active budget. |

**Nesting.** On `__aenter__`, a budget links to whatever budget is currently bound and publishes
itself as the innermost. A child starts at zero spend — it never inherits the parent's running
total. `consume` walks the parent chain, checks *every* budget in it, and only then commits to
every budget in it. So a charge on an inner budget also charges each ancestor, a tight inner limit
and a broad outer limit are enforced simultaneously, and a rejection anywhere in the chain records
nothing anywhere — there is no partial spend. Pass `isolated=True` to skip the link entirely and
create a hard sub-budget whose spend never reaches the parent.

Concurrency is serialized through the root budget's `anyio.Lock`, so concurrent `consume` calls
within one tree cannot interleave a check with another coroutine's commit. Separate trees do not
share a root and do not block each other.

**What `BudgetExceededError` actually guarantees.** `consume` itself is strictly pre-spend: it
raises before recording anything, so `spent_usd` never exceeds `max_usd`. But `LLMClient` calls
`consume` *after* a provider call returns, because the true cost is only known then. The practical
consequence is that the call which trips the ceiling has already been made and billed by the
provider; what the error prevents is every call after it. Size ceilings with one call of slack, and
treat `BudgetExceededError` as "stop now", not "nothing was spent".

The budget is discovered through a contextvar, exactly like the correlation ID, which means the
same propagation table applies. A `BudgetContext` entered in one task is not visible to a task
that was created before it.

---

## Reproducibility

Three helpers back the provenance story. Any long-running operation — an eval run, a training job,
a sweep — should call `set_seed` at startup and persist `env_snapshot()` next to its outputs.

**`set_seed(seed)`** seeds Python's `random`, then `numpy.random` and `torch` (including
`torch.cuda.manual_seed_all` when CUDA is available) if those packages are importable. The heavy
imports happen inside the function, so calling it on a base install raises nothing. Route all
pseudo-randomness through it rather than calling `random.seed` directly, so one call covers every
library in play.

**`content_hash(obj)`** returns a 64-character SHA-256 hex digest of a JSON-canonical rendering:
keys sorted, separators collapsed, `ensure_ascii=False`. Logically-equivalent inputs therefore
hash identically regardless of key insertion order. Non-JSON values are coerced with `str`, which
keeps the function total, but means the digest is only stable if those objects have a stable
`__str__` — a plain object with a default `repr` embeds its memory address and will hash
differently on every run. This function is the basis of dataset versioning and cache keys.

```python
from strata_forge.core import content_hash

assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
```

**`env_snapshot()`** returns a flat `dict[str, str | None]` suitable for dropping straight into
run metadata: `python`, `platform`, `machine`, `system`, `cwd`, plus one `pkg.<name>` entry for
each of 24 tracked packages, resolved through `importlib.metadata.version` and set to `None` when
the package is absent. The tracked list spans the core stack (`pydantic`, `pydantic-settings`,
`litellm`, `structlog`, `tenacity`, `typer`, `anyio`, `pyyaml`, `python-dotenv`, `rich`) and the
heavy optional stack
(`torch`, `numpy`, `transformers`, `trl`, `peft`, `datasets`, `accelerate`, `openai`, `anthropic`,
`google-cloud-aiplatform`, `boto3`, `qdrant-client`, `vllm`, `skypilot`). It imports none of them
and it does not capture a git revision — record your own if you need one.

---

## IDs

`uuid7()` returns an RFC 9562 version-7 UUID built from the standard library alone: 48 bits of
Unix milliseconds in the high bits, the version nibble, 12 bits of `rand_a` for sub-millisecond
ordering, the variant bits, and 62 bits of `rand_b` from `secrets`. There is no `uuid7`
third-party dependency to install or audit.

The time prefix is why Forge uses v7 rather than v4 for correlation IDs and job identifiers:
IDs sort chronologically as strings, which makes them index-friendly and makes a directory listing
of ID-named files chronological for free.

`new_correlation_id()` is `uuid7().hex` — 32 lowercase hex characters, no dashes.
`get_correlation_id()`, `set_correlation_id(cid)`, and the raw `correlation_id_var` are described
in [Correlation IDs](#correlation-ids). `set_correlation_id` accepts `None` to explicitly clear
the value within the current context.

---

## Shared types

Two aliases, both PEP 695 `type` statements:

```python
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
type PathLike = str | os.PathLike[str]
```

`JSONValue` is the standard recursive JSON definition, used wherever a payload must survive a
round trip through `json.dumps`. `PathLike` is anything `pathlib.Path(...)` accepts, and is the
convention for every public parameter that names a file.

The bar for adding a third alias is deliberately high: it must be used by at least two modules.
Module-local types stay module-local.

---

## Troubleshooting

**My log lines have no `correlation_id` field.**
The processor omits the key entirely when the variable is `None`, and it is `None` until something
sets it. Either call `set_correlation_id(new_correlation_id())` at the top of your unit of work,
or run inside a `strata_forge.tracing.traced` function, which sets it to the Langfuse trace ID.
Also confirm `configure_logging` actually ran — without it, `add_correlation_id` is not in the
processor chain at all.

**The correlation ID is `None` inside a thread.**
`loop.run_in_executor` and a bare `threading.Thread` do not carry the context across. Use
`anyio.to_thread.run_sync`, which does, or pass the ID explicitly into the thread and set it there.

**A helper set the correlation ID and it stayed set after the call returned.**
Directly awaiting a coroutine runs it in the caller's context, so its `set` is visible to the
caller. Capture the `Token` that `set_correlation_id` returns and call
`correlation_id_var.reset(token)` in a `finally` block.

**`configure_logging` had no effect the second time I called it.**
structlog is configured with `cache_logger_on_first_use=True`, so a logger object that has already
emitted a record keeps its original level and renderer. Reconfigure before the first log call, or
re-fetch the logger with `get_logger(...)` after reconfiguring.

**I set `FORGE_LOG_LEVEL` and nothing changed.**
That variable populates `strata_forge.config.Settings.logging`; it is not read by
`configure_logging`. Wire the two together at your entry point:
`configure_logging(level=get_settings().logging.level)`.

**An unknown log level was accepted silently.**
`configure_logging(level="TRACE")` falls back to `INFO` rather than raising. Check the spelling
against `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`.

**`@retry` isn't retrying.**
It selects on exception type. If the failure is a `ProviderAuthError`,
`ProviderBadRequestError`, or `ProviderContentFilterError`, the default policy excludes it on
purpose. Pass `retry_on=(...)` to widen the set, and remember that a function returning an error
sentinel instead of raising will never retry.

**`BudgetExceededError` fired but I was still billed for that call.**
`LLMClient` charges the budget after the provider responds, since that is when the real cost is
known. The ceiling stops the *next* call. Leave one call of headroom when sizing it.

**A nested `BudgetContext` didn't stop the outer one from growing.**
That is the design: a child's `consume` charges every linked ancestor. If you want a sub-budget
whose spend does not reach the parent, construct it with `isolated=True`.

**`content_hash` returns a different digest each run for the same object.**
Something in the structure is not JSON-serializable and is being coerced with `str`. If its
`__str__` embeds an address or a timestamp, the digest changes every time. Convert to plain
JSON-compatible values — `model_dump(mode="json")` for a Pydantic model — before hashing.

**`env_snapshot()` has no git SHA.**
It does not collect one. It captures the interpreter, platform, working directory, and tracked
package versions only. Record the revision yourself if your provenance story needs it.
