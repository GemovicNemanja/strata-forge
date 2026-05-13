# ADR 0001 — LiteLLM as the provider transport

**Status:** Accepted
**Date:** Initial scaffolding
**Supersedes:** —
**Superseded by:** —

## Context

AI Forge must reach six provider surfaces from day one: OpenAI, Anthropic, Google Vertex AI (Gemini and Anthropic-on-Vertex), AWS Bedrock (Anthropic-on-Bedrock), Azure OpenAI, and OpenAI-compatible servers (vLLM, TGI, SGLang). Each provider has its own SDK with idiosyncratic auth flows, request shapes, streaming protocols, error classes, and feature surfaces. Hand-rolling adapters per provider would mean six independent migration treadmills — every provider release nudges the abstraction.

Forge needs:
- Typed Pydantic in/out (the abstraction we own)
- Caching keyed on a canonical request hash
- A two-axis fallback chain (model-level and provider-level)
- A model registry with cost/capability metadata
- Token counting and cost computation per call
- Hooks for Langfuse observability

What it does NOT need to own: HTTP transport, authentication mechanics, streaming chunk parsing, retry-on-transient-error plumbing, or vendor-specific JSON-schema mappings (when those are well-handled upstream).

## Decision

Use **LiteLLM** as the underlying transport layer. Forge wraps LiteLLM in a thin, typed layer that owns the concerns above. Provider SDKs are imported only inside `src/forge/llm/providers/` — never elsewhere in the codebase.

Concretely:
- `forge.llm.providers.ProviderClient` is an abstract base over LiteLLM's `acompletion` and `astream`.
- Per-provider modules (`openai.py`, `anthropic.py`, etc.) handle provider-specific quirks (auth config shape, tool schema serialization, idiosyncratic error mapping).
- `forge.llm.errors.map_litellm_exception` normalizes LiteLLM exceptions into the `ProviderError` subtree.
- The `provider_extras={...}` argument on `LLMClient.complete` forwards verbatim to LiteLLM, giving callers a typed-but-not-too-typed escape hatch when they need provider-specific features (Anthropic's `thinking` parameter, OpenAI's Responses API, Vertex tuning, …).

## Consequences

**Positive**

- Provider breadth and protocol freshness come for free. New providers added to LiteLLM upstream are immediately reachable.
- ~30–40 % less code than hand-rolled adapters; we focus engineering effort on what differentiates Forge (typing, caching, fallback semantics, registry, observability).
- LiteLLM's Langfuse callback integrates cleanly — `forge.tracing` doesn't need to instrument LLM calls separately.
- One place to handle most provider-specific quirks (LiteLLM's normalization) plus one place to handle the remainder (our provider modules + error-mapping seam).

**Negative**

- We are pinned to LiteLLM's release cadence and feature-coverage choices. A provider feature LiteLLM hasn't wrapped yet is harder to expose.
- LiteLLM's abstractions sometimes leak — error classification, streaming chunk shapes, and JSON-schema serialization differ across providers in ways LiteLLM doesn't fully normalize. We handle these at our typed seam, which means the seam is non-trivial.
- LiteLLM has a sizable transitive dependency surface. Mitigation: it's a core dep (always installed); heavy optional features (vLLM serving, etc.) remain behind extras.

**Mitigations**

- The `provider_extras` escape hatch ensures callers can always reach LiteLLM-specific arguments.
- Each provider module retains direct access to its underlying SDK if LiteLLM coverage becomes blocking — we'd write a thin direct-SDK path inside that module rather than abandoning the abstraction.
- An ADR superseding this one is the path to a different transport.

## Alternatives considered

1. **Hand-rolled per-provider SDK adapters.** Maximum control and type fidelity, but ~3–4× the surface area and ongoing maintenance against provider drift. Rejected on cost.
2. **Pure direct LiteLLM use without a Forge typed layer.** Loses the typed Pydantic surface, makes structured output / tool calling / caching consistent only if every caller is disciplined. Rejected: the discipline doesn't scale.
3. **LangChain or similar mega-framework.** Brings opinions and surface area we don't want; harder to type strictly. Rejected.
