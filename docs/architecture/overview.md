# Architecture overview

AI Forge is a typed, async-first library of primitives for running AI experiments — inference, evaluation, fine-tuning, RAG, and remote compute — across the latest foundation models from OpenAI, Anthropic, and Google. This document is the narrative companion to the rules in [`CLAUDE.md`](../../CLAUDE.md). Decision rationale lives in the [ADRs](adr/); per-module reference docs live in [`docs/modules/`](../modules/).

## Design pillars

1. **Vendor-neutral at the seam.** LiteLLM handles provider HTTP/auth/streaming/normalization. Forge owns a typed Pydantic layer on top. ([ADR 0001](adr/0001-litellm-as-transport.md))
2. **Async-first everywhere.** Every public function is `async`. Sync wrappers live only in `strata_forge.sync` for CLI / notebook ergonomics. ([ADR 0002](adr/0002-async-only-public-api.md))
3. **Strict typing.** pyright `strict` on `src/strata_forge`. No `Any` without a justified ignore. Pydantic for data, Protocol for behavior.
4. **Exception-based errors.** Every Forge-raised error inherits from `ForgeError`; provider errors normalize at the seam. ([ADR 0003](adr/0003-exception-hierarchy.md))
5. **Modules are independent.** `import strata_forge.<module>` works without optional extras installed; heavy deps are lazily imported inside the functions that need them.
6. **Observability and cost are first-class.** Every LLM call is traceable to Langfuse, dumpable to NDJSON, and respects an active `BudgetContext`.
7. **Reproducibility is built in.** `set_seed`, `content_hash`, `env_snapshot` are in `strata_forge.core.repro`; long-running operations capture them.
8. **No vendor lock-in upward.** Higher-level modules (evals, agents, RAG) talk only to `strata_forge.llm` — they never import provider SDKs.

## Layering

```
                ┌───────────────────────────┐
                │         strata_forge.cli         │
                │   (Typer entry points)    │
                └─────────────┬─────────────┘
                              │
   ┌──────────────────────────┴───────────────────────────┐
   │  strata_forge.evals   strata_forge.agents   strata_forge.rag   strata_forge.compute │
   │  strata_forge.training  strata_forge.datasets  strata_forge.prompts          │
   └──────────────────────────┬───────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │       strata_forge.llm           │
                │  + strata_forge.tracing          │
                │  + strata_forge.storage          │
                └─────────────┬─────────────┘
                              │
                ┌─────────────┴─────────────┐
                │      strata_forge.config         │
                └─────────────┬─────────────┘
                              │
                ┌─────────────┴─────────────┐
                │       strata_forge.core          │
                │  (errors, retry, logging, │
                │   budget, repro, ids)     │
                └───────────────────────────┘
```

- `core` depends on nothing inside `forge`.
- `config` depends only on `core`.
- `llm` + `tracing` + `storage` are the next layer; they may depend on `core`, `config`, and (for `tracing`/`storage`) on each other where natural.
- Higher modules (evals, agents, RAG, compute, training, datasets, prompts) sit on top of the LLM/tracing/storage layer.
- `cli` is purely a presentation layer.

See [`module-boundaries.md`](module-boundaries.md) for the strict import rules per module.

## Provider routing

A logical model (e.g. `claude-opus-4-7`) often lives on multiple provider routes (Anthropic + Bedrock + Vertex). The model registry encodes these routes; `ModelRoute(model, provider)` is the concrete dispatch unit. The cache key is provider-agnostic — a hit is valid regardless of which route served the original call.

Fallback is two-axis: provider-level (same model across providers) within a `ModelFallback`, model-level (different model) across the outer list. ([ADR 0005](adr/0005-two-axis-fallback.md))

## Tool calling, structured output, and multimodal

All three are first-class features of `strata_forge.llm` — they ship together, not as separate add-ons. The agent module (Phase 3) reuses these primitives rather than reimplementing them. ([ADR 0006](adr/0006-tool-calling-as-llm-primitive.md))

## Configuration

`strata_forge.config.Settings` is a Pydantic `BaseSettings` with sub-models per concern. Layering: defaults → `configs/<FORGE_PROFILE>.yaml` overlay → `.env` → process env → in-code override. `strata-forge doctor` validates everything at startup.

## Observability

Langfuse is the default tracing destination. The LiteLLM Langfuse callback handles automatic LLM-call tracing; `@traced` and `traced_span()` cover non-LLM code paths. `trace_id` propagates across `await` boundaries via contextvars — never pass it manually. When `FORGE_DIAGNOSTIC=1`, every LLM call also appends to a local NDJSON file, independent of Langfuse.

## Where things live (quick reference)

| Concern | Location |
|---|---|
| Cross-cutting utilities | `src/strata_forge/core/` |
| Runtime configuration | `src/strata_forge/config/` |
| LLM client + tools + registry | `src/strata_forge/llm/` ([reference](../modules/llm.md)) |
| Prompt templates + registry | `src/strata_forge/prompts/` ([reference](../modules/prompts.md)) |
| Langfuse tracing helpers | `src/strata_forge/tracing/` ([reference](../modules/tracing.md)) |
| Datasets bridge | `src/strata_forge/datasets/` ([reference](../modules/datasets.md)) |
| Evaluation runner | `src/strata_forge/evals/` ([reference](../modules/evals.md)) |
| Agent builder | `src/strata_forge/agents/` ([reference](../modules/agents.md)) |
| Retrieval-augmented generation | `src/strata_forge/rag/` ([reference](../modules/rag.md)) |
| Storage backends | `src/strata_forge/storage/` |
| Remote compute | `src/strata_forge/compute/` |
| Fine-tuning | `src/strata_forge/training/` |
| CLI entry points | `src/strata_forge/cli/` |
| Sync wrappers | `src/strata_forge/sync.py` |
| Decision records | `docs/architecture/adr/` |
| Module reference docs | `docs/modules/` |
| Recipes | `docs/recipes/` |
| Phase status | `docs/roadmap.md` |

## What's deliberately out of scope

- Hosted inference serving — that's vLLM / TGI / SGLang's job; we orchestrate them via `strata_forge.compute`.
- Provider catalog beyond OpenAI / Anthropic / Google — see [ADR 0004](adr/0004-model-registry-scope.md); new vendors require an ADR.
- Telemetry beyond Langfuse — OpenTelemetry / Prometheus support can be added later through `strata_forge.tracing` if needed.
- A Forge-defined prompt-engineering DSL — Jinja2 templates and Langfuse-stored prompts are enough; DSPy-style compilation is out of scope.
- Production-grade authentication/authorization — Forge is a library, not a service.
