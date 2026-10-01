# Agent rules — strata_forge.core

`strata_forge.core` is the strictly-upstream utility module. Every other `strata_forge.*` module depends on it; it depends on nothing inside `forge`.

## Purpose

Cross-cutting concerns that every module needs: exception hierarchy, retry decorator, structured logging with `trace_id` propagation, cost budgets, reproducibility helpers, UUIDv7 ids, secret redaction, shared type aliases. None of it is LLM-specific.

## Boundaries

- **Owns:** `errors.py`, `retry.py`, `logging.py`, `budget.py`, `repro.py`, `ids.py`, `redact.py`, `types.py`.
- **Imports from inside `forge`:** nothing. This is a hard rule.
- **Allowed external deps:** `pydantic`, `pydantic-settings`, `structlog`, `tenacity`, anyio, `uuid7` (or equivalent), stdlib. Anything heavier (numpy, torch) must be lazily imported inside the function that needs it.
- **Does NOT:** know about LLMs, read config (it accepts values; `strata_forge.config` produces them), make network calls.

## Public API

The module's `__init__.py` re-exports a curated surface. Treat the following as the supported public API; anything not in `__init__.py` is internal.

- Exception types from `errors.py`: `ForgeError`, `ConfigError`, `ProviderError` + subclasses, `BudgetExceededError`, `ValidationError`, `CacheError`, `RegistryError`, `FallbackExhaustedError`.
- `@retry` decorator and `DEFAULT_RETRY_ON` from `retry.py` (works for async + sync).
- `configure_logging`, `get_logger`, `traced_span` from `logging.py`.
- `BudgetContext` from `budget.py`.
- `set_seed`, `content_hash`, `env_snapshot` from `repro.py`.
- `uuid7`, `new_correlation_id`, `get_correlation_id`, `set_correlation_id`, `correlation_id_var` from `ids.py`.
- `Redactor`, `RedactingStream`, `TokenPattern`, `DEFAULT_PATTERNS`, `PLACEHOLDER` from `redact.py`. Every string a credential-holding process surfaces, and every console a relay forwards, goes through a `Redactor`; never write a second scrubber.
- Shared type aliases from `types.py` (e.g. `JSONValue`, `PathLike`).

## Internal patterns

- Exception classes have no fancy `__init__` — they accept a message and optional structured data (e.g. `ProviderError(message, route=..., status=...)`).
- `@retry` is tenacity-based; default policy is exponential backoff with jitter; predicates select on `ProviderError` subclasses. The decorator works for both async and sync callables.
- structlog setup: pretty renderer when stdout is a TTY, JSON renderer otherwise. `correlation_id` is a `ContextVar[str | None]` in `ids.py` — `logging.py`'s `_add_correlation_id` processor injects it into every record; it propagates across `await` without explicit passing.
- `BudgetContext` is an async context manager. Nested budgets share the parent's accumulated spend unless `isolated=True`.
- `content_hash` hashes a JSON-canonical form (sorted keys, no whitespace) to remain stable across logically-equivalent inputs.
- `env_snapshot` captures: installed package versions, git SHA (if in a repo), Python version, OS info. Lazy imports for `numpy` / `torch` versions when present.
- `Redactor` computes the union of every span any value form, token pattern or private-key block covers, and replaces each maximal covered run with one `PLACEHOLDER`. `RedactingStream` runs the same computation with a "decided" bound: a match starting at least `max_len - 1` characters before the end of what has arrived is final. That is what makes streamed output equal one-shot output, and it holds only because every `TokenPattern` declares a true `max_chars` and uses no lookbehind.

## Test expectations

- Unit tests under `tests/unit/core/`, one file per module (`test_errors.py`, `test_retry.py`, etc.).
- Coverage target: ≥ 85 % line.
- Hypothesis property tests for `content_hash` (stability under reorderings, sensitivity to value changes).
- Hypothesis property tests for `redact`: a secret split at any offset across any pieces never survives; streamed output equals one-shot output on text built from every pattern's edges; short values are refused; base64 at any alignment and the URL/JSON/`repr` encodings are caught.
- Tests must run without optional heavy deps installed; `set_seed` tests exercise the no-numpy and no-torch paths.
- The structlog setup test verifies that `trace_id` propagates across `await` by spawning a task and checking the logger output.

## Gotchas

- Do NOT add any `strata_forge.*` import here. If you find yourself reaching for `strata_forge.config` or `strata_forge.llm`, the wrong module is calling — refactor upward.
- Do NOT change the exception hierarchy without an ADR — many `@retry` predicates and fallback rules depend on the classification.
- The `@retry` decorator must remain compatible with both `async def` and regular `def` — when adding features, test both.
- `BudgetContext` accounting must be thread/coroutine safe; spend updates use an `anyio.Lock`.
- `redact.py` must never log, and no error message or `repr` from it may include a value it was given. A new `TokenPattern` needs a bounded quantifier on every repeat, a `max_chars` that covers its longest match plus any lookahead, and no lookbehind. Put the long variable part at the END: a bounded middle segment followed by a required delimiter stops matching a longer token altogether, so a middle bound must be one no real token reaches.

## When to update this file

- Adding a new sub-module (e.g. `strata_forge.core.profiling`).
- Adding or renaming exception classes.
- Changing the default `@retry` policy or predicate set.
- Changing the `trace_id` propagation mechanism.
- Tightening or loosening dependency rules (e.g. lifting the no-`strata_forge.*`-imports rule, which would require an ADR).
