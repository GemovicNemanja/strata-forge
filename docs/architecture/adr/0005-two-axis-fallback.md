# ADR 0005 — Two-axis fallback (model-level + provider-level)

**Status:** Accepted
**Date:** Initial scaffolding
**Supersedes:** —
**Superseded by:** —

## Context

Provider availability and rate-limit headroom fluctuate; a robust LLM client needs a fallback strategy. A flat list of model names (`["claude-opus-4-7", "gpt-5.5", "gemini-3.1-pro"]`) treats every transition as "switch to a different model," which conflates two semantically different transitions:

1. **Provider failover for the same logical model.** Claude Opus 4.7 is available on Anthropic, Bedrock, and Vertex. If Anthropic is rate-limiting, the operator likely wants to try the same model on Bedrock before changing models — the quality and cost are the same, only the route differs.
2. **Model failover to a different model.** If every provider route for Opus is exhausted, the operator might want to drop to a cheaper or simpler model rather than fail outright. This is a genuine quality/cost trade-off.

A flat list can't express both axes; operators encode their intent imprecisely and end up with surprising behavior under provider stress.

## Decision

Fallback chains have two explicit axes:

- **`ModelFallback(model, providers=[...])`** — provider-level failover for a single logical model. Within one `ModelFallback`, providers are tried in order. The model identity stays constant; only the route changes.
- **The outer list of `ModelFallback` entries** — model-level failover. When all providers in a `ModelFallback` are exhausted, the next entry (potentially a different logical model) takes over.

```python
client = LLMClient.with_fallbacks([
    ModelFallback(model="claude-opus-4-7", providers=["anthropic", "bedrock", "vertex"]),
    ModelFallback(model="gpt-5.5",          providers=["openai", "azure"]),
    ModelFallback(model="gemini-3.1-pro",   providers=["vertex"]),
])
```

Per-error advance/abort rules:

| Error class | Advance to next provider? | Advance to next model? |
|---|---|---|
| `ProviderRateLimitError` | Yes | Only after providers exhausted |
| `ProviderTimeoutError` | Yes | Only after providers exhausted |
| `ProviderServerError` | Yes | Only after providers exhausted |
| `ProviderAuthError` | Yes (different provider may have valid creds) | Only after providers exhausted |
| `ProviderBadRequestError` | Configurable; default: advance | Only after providers exhausted |
| `ProviderContentFilterError` | **No** — short-circuit the whole chain | **No** |

Bare-string shorthand (`with_fallbacks(["claude-opus-4-7", "gpt-5.5"])`) expands to single-provider `ModelFallback` entries using the registry default route for each model — no provider-level failover. Operators opt into provider-level failover by writing `ModelFallback(...)` explicitly.

Final failure raises `FallbackExhaustedError` carrying every `(model, provider, error)` triple for diagnosis.

## Consequences

**Positive**

- Operators express intent precisely: "stay on Claude across providers before falling back to GPT" versus "try every available model in priority order."
- Provider outages are gracefully handled at constant cost/quality (same-model failover).
- Content-filter errors don't waste API calls trying providers that will give the same answer.
- Errors carry full route history, making post-hoc analysis straightforward.

**Negative**

- The API is slightly more complex than a flat list. The bare-string shorthand absorbs the common case.
- Two axes of error policy means more decisions for the implementer to get right.

**Mitigations**

- Default error-policy rules are conservative and documented in the table above.
- VCR cassettes cover the full matrix (provider-level failover, model-level failover, mixed, content-filter short-circuit) — the semantics are tested before they're trusted.

## Alternatives considered

1. **Flat list of (model, provider) tuples** (`with_fallbacks([("claude-opus-4-7", "anthropic"), ("claude-opus-4-7", "bedrock"), ...])`). Expressible but verbose, and loses the visual grouping of "all routes for this logical model."
2. **Single fallback axis, model-level only.** Loses provider failover semantics for the same model — operators can't say "try Claude harder before dropping to GPT."
3. **`model@provider` string syntax** (`"claude-opus-4-7@anthropic"`). Compact but harder to evolve when we add per-entry overrides (retry budgets, timeout overrides). The structured `ModelFallback` opens the door for those without further API churn.
