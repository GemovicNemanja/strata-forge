# ADR 0003 — Exception-based error model with a unified `ForgeError` hierarchy

**Status:** Accepted
**Date:** Initial scaffolding
**Supersedes:** —
**Superseded by:** —

## Context

LLM-touching code paths fail in many distinguishable ways: rate limits, auth, timeouts, server errors, bad request, content filters, missing models, exceeded budgets, cache backend failures, fallback exhaustion. Callers need to write robust retry / fallback / cleanup logic that depends on accurately classifying these failures.

Without a uniform error model, two problems emerge: (1) every provider exposes its own exception classes, so retry predicates become a mess of `isinstance(e, ProviderASpecificException) or isinstance(e, ProviderBSpecificException) or ...`; (2) higher-level modules (evals, agents) can't decide what's recoverable without poking provider-specific internals.

Python has two idiomatic ways to model recoverable failures: exceptions, or `Result[T, E]` tagged unions (Rust-style). The latter shows up in errors-as-values libraries (`returns`, `result`); proponents argue it makes errors visible in types.

## Decision

Forge uses **exceptions** for all error paths. The hierarchy is rooted at `ForgeError` and lives in `strata_forge.core.errors`:

```
ForgeError
├── ConfigError              # missing env, malformed YAML, bad overlay
├── ProviderError            # all provider-side failures
│   ├── ProviderAuthError
│   ├── ProviderRateLimitError
│   ├── ProviderTimeoutError
│   ├── ProviderBadRequestError
│   ├── ProviderServerError
│   └── ProviderContentFilterError
├── BudgetExceededError      # cost / token ceiling breached
├── ValidationError          # input / output Pydantic validation
├── CacheError               # backend failure (Redis unreachable, etc.)
├── RegistryError            # unknown model, missing pricing, unsupported route
└── FallbackExhaustedError   # entire chain failed; carries underlying causes
```

LiteLLM exceptions are normalized at the seam (`strata_forge.llm.errors.map_litellm_exception`) into the `ProviderError` subtree before bubbling out of `strata_forge.llm`. Higher-level modules raise their own `ForgeError` subclasses; they never re-throw raw provider SDK or LiteLLM exceptions.

`Result`-style returns were explicitly considered and rejected.

## Consequences

**Positive**

- Decorators (`@retry`, eventually `@traced`) drive control flow with clean exception predicates — `retry_if_exception_type(ProviderRateLimitError | ProviderTimeoutError | ProviderServerError)`.
- Public function signatures stay clean: `async def complete(...) -> LLMResponse` instead of `-> Result[LLMResponse, LLMError]`.
- Interop with the rest of the Python ecosystem (pytest's `pytest.raises`, contextlib, tenacity, structlog's exception logging) is frictionless.
- Cross-module error semantics are explicit: a function that lists `BudgetExceededError` in its docstring tells the caller it respects budgets.

**Negative**

- Exceptions can be missed if callers don't handle them. Forge mitigates this for known failure modes by documenting which exceptions each public function can raise and by surfacing them in the type-checker output via `Raises:` docstring sections (informational; pyright doesn't enforce).
- Retry/fallback logic depends on accurate normalization at the LiteLLM seam — one place that must stay current with LiteLLM's exception surface.

**Mitigations**

- Snapshot tests over `map_litellm_exception` catch drift in LiteLLM's exception classes.
- `FallbackExhaustedError` carries every underlying `(model, provider, error)` triple so callers can introspect.
- The retry decorator's default predicates pin to `ForgeError` subclasses, never raw exceptions, so adding a new normalization rule automatically updates retry behavior.

## Alternatives considered

1. **`Result[T, E]` tagged-union returns.** Visible in signatures, but every callsite needs a `.unwrap()` / `match` and the friction adds up. Interop with async generators, decorators, and most of the Python ecosystem is awkward. Rejected.
2. **Mixed: exceptions for "exceptional," `Outcome` for evaluation outcomes (pass/fail/error/skip).** This was nearly chosen for evals specifically. Decision: keep evaluation outcomes as a separate tagged union (`Pass`, `Fail`, `Error`, `Skip`) at the grader return surface, but never let `Outcome` leak into the control-flow error path. Pragmatic middle ground that stays consistent with exception-first elsewhere.
3. **Provider-specific exception passthrough.** Forces every caller to know provider quirks. Rejected on coupling grounds.
