# Architecture overview

strata-forge is a typed, async-first library of primitives for running AI experiments — inference,
evaluation, fine-tuning, RAG, and remote compute — across the latest foundation models from OpenAI,
Anthropic, and Google. The distribution is `strata-forge`; the import package is `strata_forge`.
This document is the narrative companion to the contributor rules in
[`CLAUDE.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CLAUDE.md). Decision
rationale lives in the [ADRs](adr/README.md); per-module reference docs live in
[`docs/modules/`](../modules/README.md).

## Design pillars

1. **Vendor-neutral at the seam.** LiteLLM handles provider HTTP, auth, streaming, and response
   normalization. strata-forge owns a typed Pydantic layer on top and nothing below it.
   ([ADR 0001](adr/0001-litellm-as-transport.md))
2. **Async-first.** Every public API that performs I/O is `async`. Pure helpers — `content_hash`,
   `set_seed`, template rendering, config access — stay synchronous because making them awaitable
   buys nothing. Sync facades over the LLM client live only in `strata_forge.sync`, for CLI and
   notebook ergonomics. ([ADR 0002](adr/0002-async-only-public-api.md))
3. **Strict typing.** pyright `strict` on `src/strata_forge`. No `Any` without a justified ignore.
   Pydantic v2 for data, `Protocol` for behaviour, `TypedDict` for narrow boundaries.
4. **Exception-based errors.** Failures raise; they are never encoded in return values. Library
   failures raise `ForgeError` subclasses from `strata_forge.core.errors`, and provider exceptions
   normalize into the `ProviderError` subtree at the LiteLLM seam. Ordinary argument mistakes still
   raise the builtins you would expect — `ValueError`, `TypeError`.
   ([ADR 0003](adr/0003-exception-hierarchy.md))
5. **Modules are independent.** `import strata_forge.<module>` works with no optional extras
   installed; heavy dependencies are imported inside the function that needs them and raise
   `ImportError` with the exact `pip install 'strata-forge[<extra>]'` hint when absent.
6. **Observability and cost are first-class.** Every LLM call is traceable to Langfuse, dumpable to
   NDJSON, and charged against any active `BudgetContext`.
7. **Reproducibility is built in.** `set_seed`, `content_hash`, and `env_snapshot` live in
   `strata_forge.core.repro`; long-running operations capture them into run metadata.
8. **No provider SDKs above the transport layer.** Chat completions reach a vendor only through
   `src/strata_forge/llm/providers/`; every module above it calls `LLMClient`. Two neighbouring
   SDK uses are deliberate rather than drift: `strata_forge.llm.errors` imports `openai` to name
   the exception classes LiteLLM re-raises, and `strata_forge.rag` calls `litellm.aembedding`
   directly for embeddings and lazy-imports `qdrant-client` and `cohere` for its own vector store
   and reranker. ([ADR 0012](adr/0012-rag-protocols-and-vector-store-relocation.md))

## Layering

```
               ┌─────────────────────────────────────────────────────┐
               │      strata_forge.cli        strata_forge.sync      │
               │       Typer commands      asyncio.run facades       │
               └──────────────────────────┬──────────────────────────┘
                                          │
               ┌──────────────────────────┴──────────────────────────┐
               │               strata_forge.pipelines                │
               │             env-driven run entrypoints              │
               └──────────────────────────┬──────────────────────────┘
                                          │
┌─────────────────────────────────────────┴─────────────────────────────────────────┐
│  strata_forge.evals  strata_forge.agents  strata_forge.rag  strata_forge.prompts  │
│        strata_forge.datasets  strata_forge.compute  strata_forge.training         │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
       ┌──────────────────────────────────┴──────────────────────────────────┐
       │ strata_forge.llm     strata_forge.tracing     strata_forge.storage  │
       │                 transport, observability, file I/O                  │
       └──────────────────────────────────┬──────────────────────────────────┘
                                          │
               ┌──────────────────────────┴──────────────────────────┐
               │                 strata_forge.config                 │
               │       Settings, .env loading, overlay helpers       │
               └──────────────────────────┬──────────────────────────┘
                                          │
               ┌──────────────────────────┴──────────────────────────┐
               │                  strata_forge.core                  │
               │     errors, retry, logging, budget, repro, ids      │
               └─────────────────────────────────────────────────────┘
```

The boxes describe how far up the stack a module is allowed to reach, not how many imports it
actually writes. Read against the real import graph, the picture is sparser than the drawing
suggests, and the gaps are load-bearing:

- `core` imports nothing from `strata_forge`. `config` imports only `core` (`ConfigError`,
  `PathLike`).
- `llm` sits on `core` and `config`; `storage` touches `config` only, lazily, when resolving a
  Hugging Face token.
- `tracing` imports `core.ids` and `config` and nothing else. It wraps the library from above
  rather than threading through it, so no module below it imports `tracing` — decorating a call
  site is always the caller's choice.
  ([ADR 0008](adr/0008-tracing-as-cross-cutting.md))
- `rag` has no intra-package imports at all; its embedder speaks to `litellm` directly.
- `compute` and `training` reach `llm` for types only — `compute` entirely under `TYPE_CHECKING`,
  `training` for the message models its chat-template formatter converts.
- Two edges run sideways inside the capability layer: `agents` imports `rag` at runtime for the
  vector-store primitives its episodic memory needs
  ([ADR 0012](adr/0012-rag-protocols-and-vector-store-relocation.md)), and `evals` imports
  `datasets` for typing only, since the runner is handed a resolved `Dataset` rather than fetching
  one.
- `pipelines` is the only module that composes across the capability layer — it wires `compute`,
  `llm`, `storage`, and `training` into one runnable entrypoint.
- `cli` and `sync` are sinks. Nothing imports them.

See [`module-boundaries.md`](module-boundaries.md) for the per-module import rules and ownership
statements.

## Provider routing

A logical model (e.g. `claude-opus-4-7`) often lives on multiple provider routes — Anthropic,
Bedrock, and Vertex all serve the same Claude weights. The model registry encodes those routes, and
`ModelRoute(model, provider)` is the concrete dispatch unit. The response cache key is
provider-agnostic: a hit stays valid regardless of which route served the original call.

Fallback is two-axis. Provider-level failover moves across routes for one logical model inside a
`ModelFallback`; model-level failover moves to the next `ModelFallback` in the outer list once its
providers are exhausted. A content-filter refusal short-circuits both axes, because retrying it
elsewhere is a waste of money. ([ADR 0005](adr/0005-two-axis-fallback.md))

## Tool calling, structured output, and multimodal

All three are primitives of `strata_forge.llm` — they ship together, not as separate add-ons.
`strata_forge.agents` composes them rather than reimplementing them: it holds no tool abstraction,
no message types, and no loop of its own.
([ADR 0006](adr/0006-tool-calling-as-llm-primitive.md),
[ADR 0011](adr/0011-agents-thin-wrapper-over-forge-llm.md))

The streaming counterpart, `LLMClient.stream_tool_loop`, yields a typed event union rather than raw
chunks, and suspends with a `PendingToolCalls` event when the model asks for a tool the library
cannot execute itself — leaving resumption to plain message input rather than a second API.
([ADR 0014](adr/0014-streaming-tool-loop-event-protocol.md),
[ADR 0015](adr/0015-client-executed-tools-suspend-the-streaming-loop.md))

## Configuration

`strata_forge.config.Settings` is a Pydantic `BaseSettings` root that aggregates one sub-model per
concern. Values resolve in this order, later winning over earlier: field defaults, then `.env`,
then process environment, then any in-code override passed to the constructor.

`strata_forge.config.overlays` ships standalone YAML helpers — `load_overlay`, `deep_merge`,
`overlay_path_for_profile` — for applications that want profile files. They are primitives for a
caller to apply at its own bootstrap; `Settings` does not read them, and `FORGE_PROFILE` only sets
the `Settings.profile` string.

`strata-forge doctor` is a diagnostic, not a gate. It prints the resolved settings, the versions of
the packages strata-forge tracks, and TCP reachability for Langfuse, Redis, and Qdrant, then always
exits 0.

## Observability

Langfuse is the default tracing destination. `install_litellm_callback()` registers LiteLLM's
Langfuse callback so every completion is traced without a line of call-site change; `@traced` and
`traced_span()` cover non-LLM code paths and are applied by whoever wants the span. When Langfuse
is unconfigured, the whole module degrades to a silent no-op.

A `correlation_id` propagates across `await` boundaries through a contextvar and is injected into
every structlog record — never pass it manually. Independently of Langfuse, setting
`FORGE_DIAGNOSTIC_ENABLED=1` appends a JSON record for every completed LLM call to the file named
by `FORGE_DIAGNOSTIC_PATH`.

## Where things live (quick reference)

| Concern | Location |
|---|---|
| Cross-cutting utilities | `src/strata_forge/core/` ([reference](../modules/core.md)) |
| Runtime configuration | `src/strata_forge/config/` ([reference](../modules/config.md)) |
| LLM client + tools + registry | `src/strata_forge/llm/` ([reference](../modules/llm.md)) |
| Prompt templates + registry | `src/strata_forge/prompts/` ([reference](../modules/prompts.md)) |
| Langfuse tracing helpers | `src/strata_forge/tracing/` ([reference](../modules/tracing.md)) |
| Datasets bridge | `src/strata_forge/datasets/` ([reference](../modules/datasets.md)) |
| Evaluation runner | `src/strata_forge/evals/` ([reference](../modules/evals.md)) |
| Agent builder | `src/strata_forge/agents/` ([reference](../modules/agents.md)) |
| Retrieval-augmented generation | `src/strata_forge/rag/` ([reference](../modules/rag.md)) |
| Storage backends | `src/strata_forge/storage/` ([reference](../modules/storage.md)) |
| Remote compute | `src/strata_forge/compute/` ([reference](../modules/compute.md)) |
| Fine-tuning | `src/strata_forge/training/` ([reference](../modules/training.md)) |
| Runnable pipeline entrypoints | `src/strata_forge/pipelines/` ([reference](../modules/pipelines.md)) |
| CLI entry points | `src/strata_forge/cli/` ([reference](../modules/cli.md)) |
| Sync wrappers | `src/strata_forge/sync.py` ([reference](../modules/sync.md)) |
| Decision records | [`docs/architecture/adr/`](adr/README.md) |
| Module reference docs | [`docs/modules/`](../modules/README.md) |
| Task-oriented recipes | [`docs/recipes/`](../recipes/README.md) |
| Runnable examples | [`examples/`](https://github.com/GemovicNemanja/strata-forge/tree/main/examples) |
| How to contribute | [`CONTRIBUTING.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CONTRIBUTING.md) |

## What's deliberately out of scope

- Hosted inference serving — that is vLLM / TGI / SGLang's job; strata-forge orchestrates them
  through `strata_forge.compute` and talks to the result over the `openai_compat` route.
- A provider catalogue beyond OpenAI, Anthropic, and Google — see
  [ADR 0004](adr/0004-model-registry-scope.md). New vendors require a new ADR; the registry data
  format itself imposes no such limit.
- Telemetry beyond Langfuse — OpenTelemetry or Prometheus support would go through
  `strata_forge.tracing` behind the same no-op contract.
- A prompt-engineering DSL — Jinja2 templates plus a versioned prompt registry are enough;
  DSPy-style compilation is out of scope.
- Production-grade authentication and authorization — strata-forge is a library, not a service.
