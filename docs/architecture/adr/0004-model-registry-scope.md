# ADR 0004 — Model registry scoped to latest OpenAI, Anthropic, Google foundation models

**Status:** Accepted
**Date:** Initial scaffolding
**Supersedes:** —
**Superseded by:** —

## Context

The LLM provider space is broad: there are dozens of vendors, hundreds of model variants across generations, and a thicket of pricing and capability metadata that drifts constantly. A registry that tries to cover everything quickly becomes a maintenance hotspot — stale pricing, drift between provider-reported capabilities and the registry, broken default routes.

At the same time, most experimental work targets a small set of flagship models. Operators choose models on a frontier-quality vs. cost spectrum, not by browsing a catalog of 200 entries.

## Decision

The initial model registry is scoped to the **latest foundation models from OpenAI, Anthropic, and Google** only.

| Vendor | Models | Provider routes |
|---|---|---|
| Anthropic | Claude Opus 4.7, Sonnet 4.6, Haiku 4.5 | `anthropic`, `bedrock`, `vertex` |
| OpenAI | GPT-5.5, GPT-5.5 Pro, GPT-5.5 Thinking, GPT-5.5 Instant | `openai`, `azure` |
| Google | Gemini 3.1 Pro, Gemini 3.1 Flash, Gemini 3.1 Flash Lite | `vertex` |

The registry data format (`src/forge/llm/registry_data.yaml`) is general — it does not constrain which models can be added. Adding new vendors (Mistral, Cohere, xAI, Meta open-weight via inference providers, etc.) or older variants requires a new ADR justifying the trade-off and updating this one's status to `Superseded by`.

OpenAI-compatible self-hosted endpoints (vLLM, TGI, SGLang) are addressed via the `openai_compat` provider, not via registry entries — those models are configured per-deployment.

## Consequences

**Positive**

- Curation effort focuses on the models people actually use.
- Pricing and capability metadata stay current.
- Less surface area for the cache key to collide on (logical model names are the cache key, so a smaller registry means fewer foot-guns around model aliasing).
- Clear precedent: future expansions go through an ADR, so we don't drift into a catalog-of-the-week problem.

**Negative**

- Users wanting older model variants or non-major-vendor models must either (a) propose a registry extension via ADR, (b) maintain their own registry overlay, or (c) route through `openai_compat` for self-hosted models.

**Mitigations**

- The registry loader merges `registry_data.yaml` with an optional user overlay path declared in `Settings.registry.overlay_path` — local extensions don't need an ADR.
- `forge doctor` reports the registry contents at runtime, making it obvious which models are reachable in this deployment.

## Alternatives considered

1. **Exhaustive registry covering every model from every supported provider.** Higher maintenance cost; pricing/capability drift is constant; we'd spend engineering time on curation rather than abstraction. Rejected.
2. **No registry; let LiteLLM's model info drive routing.** LiteLLM's data is good but inconsistent (pricing fields missing for some models, capability flags incomplete). Forge needs a canonical source of truth. Rejected.
3. **Per-user registries with no shipped defaults.** Forces every user to maintain pricing/capability data themselves; defeats the "batteries included" pillar. Rejected.
