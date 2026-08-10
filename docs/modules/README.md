# Module guides

One page per `strata_forge.*` module, named after the module. Each page is the
canonical reference for that module's public API, its behavioural contracts, and the
gotchas that only show up once you've used it — deeper than a docstring, narrower than
the [architecture overview](../architecture/overview.md).

The modules are listed here roughly bottom-up: everything above a row may import from
the rows below it, and never the other way round. `strata_forge.core` and
`strata_forge.config` sit under everything; `strata_forge.tracing` wraps everything from
above without being imported by any of it.

| Module | What it does |
|---|---|
| [`strata_forge.core`](core.md) | The strictly-upstream primitives: the `ForgeError` hierarchy, a tenacity-backed `@retry`, structlog with correlation-id injection, `BudgetContext` cost ceilings, reproducibility helpers, UUIDv7 ids. |
| [`strata_forge.config`](config.md) | The single configuration entry point — a Pydantic Settings root with per-concern sub-models, `.env` loading, and standalone YAML-overlay primitives. |
| [`strata_forge.llm`](llm.md) | The async LLM client over LiteLLM: typed messages and responses, structured output, tool calling, streaming, two-axis fallback, a provider-agnostic cache, and the model registry. |
| [`strata_forge.sync`](sync.md) | Blocking `asyncio.run` facades over four `LLMClient` methods, for CLI and notebook use. |
| [`strata_forge.prompts`](prompts.md) | Prompts as sandboxed Jinja2 templates split into a stable prefix and a dynamic suffix, rendered to messages and stored through a pluggable versioned registry. |
| [`strata_forge.tracing`](tracing.md) | Langfuse observability layered over everything else — LiteLLM auto-tracing, `@traced`, spans, scores, metrics — degrading to a silent no-op when unconfigured. |
| [`strata_forge.datasets`](datasets.md) | Frozen dataset shapes with content-hash versioning and diffing, a pluggable async store, a Hugging Face Datasets bridge, and two synthetic-data helpers. |
| [`strata_forge.evals`](evals.md) | Experiments as data: models × prompts × dataset × graders run concurrently, then aggregated into metrics, reports, sweeps, trace replays, and a Wilson-bounded CI gate. |
| [`strata_forge.agents`](agents.md) | An agent runtime composed from `strata_forge.llm` primitives, with built-in tools, conversation and episodic memory, and hand-off / critic-refiner patterns. |
| [`strata_forge.rag`](rag.md) | Retrieval: five Protocols (embedder, chunker, retriever, vector store, reranker), concrete in-process and hosted implementations, and a pipeline that composes them. |
| [`strata_forge.storage`](storage.md) | One async surface over local disk, S3, GCS, Azure Blob, and the Hugging Face Hub, plus model and dataset push/pull. |
| [`strata_forge.compute`](compute.md) | Remote work as typed data, submitted through a uniform `Backend` Protocol (local subprocess, SSH, SkyPilot), with batch inference and self-hosted serving adapters. |
| [`strata_forge.training`](training.md) | Fine-tuning: typed configs rendered into TRL's SFT and preference trainers (DPO, ORPO, KTO, GRPO), LoRA/QLoRA adapters, chat templating, packing, and NDJSON progress events. |
| [`strata_forge.pipelines`](pipelines.md) | Runnable entrypoints that compose the library's serving, batch, and storage primitives into a single job you launch on a compute target. |
| [`strata_forge.cli`](cli.md) | The `strata-forge` console script: `doctor`, `chat`, `prompts`, `datasets`, `eval`, `experiments`, `compute`, `train`, `serve`. |

## Where else to look

- [Architecture overview](../architecture/overview.md) — how the layers fit together.
- [Module boundaries](../architecture/module-boundaries.md) — what each module may and
  may not import.
- [ADRs](../architecture/adr/README.md) — why a given design looks the way it does. Module pages
  link the relevant ones from their **See also** footers.
- [`examples/`](https://github.com/GemovicNemanja/strata-forge/tree/main/examples) —
  runnable scripts; most module pages link the ones that exercise them.
