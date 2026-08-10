# Module boundaries

Authoritative rules for what each `strata_forge.*` module owns, what it may import, and what it
explicitly does not do. These rules are enforced by code review; a violation that is genuinely
warranted needs an ADR before it merges. The narrative version of the same picture is in
[`overview.md`](overview.md).

## Dependency direction (allowed imports)

"May import" is a ceiling, not a description. The right-hand column is the rule that matters; the
middle column is the widest set a module is permitted to reach for. What each module imports
*today* is listed in its ownership section below, and is in most cases a strict subset.

| Module | May import from | Must NOT import from |
|---|---|---|
| `strata_forge.core` | — | any other `strata_forge.*` |
| `strata_forge.config` | `core` | everything else |
| `strata_forge.llm` | `core`, `config` | every module above it |
| `strata_forge.tracing` | `core`, `config`, `llm` and `prompts` (types only, for serializing trace payloads) | `datasets`, `evals`, `agents`, `rag`, `compute`, `training`, `pipelines` |
| `strata_forge.storage` | `core`, `config` | `llm` and everything above it |
| `strata_forge.prompts` | `core`, `config`, `llm` | `datasets`, `evals`, `agents`, `rag`, `compute`, `training` |
| `strata_forge.datasets` | `core`, `config`, `llm`, `storage` | `prompts`, `evals`, `agents`, `rag`, `compute`, `training` |
| `strata_forge.rag` | `core`, `config`, `llm`, `storage` | `agents`, `evals`, `datasets`, `prompts`, `compute`, `training` |
| `strata_forge.evals` | `core`, `config`, `llm`, `prompts`, `datasets`, `storage` | `agents`, `rag`, `compute`, `training` |
| `strata_forge.agents` | `core`, `config`, `llm`, `prompts`, `rag`, `storage` | `evals`, `datasets`, `compute`, `training` |
| `strata_forge.compute` | `core`, `config`, `llm`, `storage` | `evals`, `agents`, `rag`, `training` |
| `strata_forge.training` | `core`, `config`, `llm` (types only), `datasets`, `storage` | `evals`, `agents`, `rag`, `compute` |
| `strata_forge.pipelines` | any module below it | `cli`, `sync` |
| `strata_forge.cli` | any module | — (the CLI is a sink) |
| `strata_forge.sync` | `llm` | provider SDKs directly |

Three rules underpin the table:

- **`tracing` points down, never up.** No module imports `strata_forge.tracing`. It wraps the
  library from above through the LiteLLM callback and through `@traced` applied at call sites the
  caller chooses. ([ADR 0008](adr/0008-tracing-as-cross-cutting.md))
- **`cli` and `sync` are sinks.** Nothing imports them, so neither can create a cycle.
- **`pipelines` is the only cross-capability composer.** It exists precisely to wire `compute`,
  `llm`, `storage`, and `training` into one runnable entrypoint, which is why its ceiling is the
  whole stack below it. No capability module may import `pipelines` back.

Heavy dependencies (`torch`, `transformers`, `trl`, `peft`, `skypilot`, `asyncssh`,
`qdrant-client`, `vllm`, `pillow`, `redis`, `datasets`, `huggingface_hub`, `fsspec`, `langfuse`,
`cohere`, `sentence-transformers`) are imported **inside** the function or method that uses them,
never at module load, and the enclosing function raises `ImportError` with the matching
`pip install 'strata-forge[<extra>]'` hint when the extra is absent. The invariant this protects is
that `import strata_forge.<anything>` succeeds on a bare install.

## Per-module ownership

### `strata_forge.core`
**Owns:** the `ForgeError` hierarchy, the `@retry` decorator, structlog configuration with
contextvar `correlation_id` injection, `BudgetContext` cost and token ceilings, `set_seed` /
`content_hash` / `env_snapshot`, a hand-rolled RFC 9562 `uuid7`, and shared type aliases.

**Imports today:** nothing from `strata_forge`.

**Does NOT:** know about LLM specifics, touch the network, or read configuration. It accepts values;
`strata_forge.config` produces them.

### `strata_forge.config`
**Owns:** the `Settings` root and its per-concern sub-models, the cached `get_settings()` accessor,
`.env` loading, and the standalone YAML-overlay primitives (`load_overlay`, `deep_merge`,
`overlay_path_for_profile`).

**Imports today:** `core.errors.ConfigError` and `core.types.PathLike`.

**Does NOT:** construct provider clients, make network calls, or apply the YAML overlays itself. The
overlay functions are primitives a caller applies at its own bootstrap; `Settings` never consumes
them, and `FORGE_PROFILE` only populates the `Settings.profile` string.

### `strata_forge.llm`
**Owns:** `LLMClient`, the message and response models, tool-calling primitives (`Tool`,
`ToolDeclaration`, `@tool`, `run_tool_loop`, `stream_tool_loop`), structured output, multimodal
image input, streaming accumulators, two-axis fallback, the provider-agnostic response cache, the
model registry, cost and token accounting, the NDJSON diagnostic dump, and the LiteLLM-backed
provider clients under `providers/`.

**Imports today:** `core` (errors, budget, ids, repro) and `config` (lazily, in `diagnostic.py`).

**Does NOT:** manage prompts, store datasets, run evaluations, build agents, or retrieve documents.
It is consumed by those modules and never consumes them. It also does not own embeddings — see
`strata_forge.rag`.

### `strata_forge.tracing`
**Owns:** the lazily constructed Langfuse client singleton, `install_litellm_callback()`, the
`@traced` decorator, the `traced_span` context manager, and the score and metric helpers.

**Imports today:** `core.ids` and `config`. It imports neither `llm` nor `prompts`, though the table
permits both for payload types.

**Does NOT:** auto-instrument application code beyond LLM calls — an explicit `@traced` is the
contract for non-LLM spans. It also does not raise on misconfiguration: with Langfuse unconfigured
every entry point is a silent no-op.

### `strata_forge.storage`
**Owns:** `StorageGateway`, an `fsspec` façade over local disk, S3, GCS, Azure Blob, and the HF Hub;
and `HFHubClient` for Hub repo, model, and dataset transfer.

**Imports today:** `config`, lazily, only to resolve a Hugging Face token and endpoint.

**Does NOT:** cache LLM responses — that is `strata_forge.llm.cache`.

### `strata_forge.prompts`
**Owns:** the sandboxed Jinja2 environment and its safe filters, `PromptTemplate` with its
structural stable-prefix / dynamic-suffix split, `render()` into `strata_forge.llm` messages plus a
`CacheHints` record, and the `PromptRegistry` over pluggable in-memory and Langfuse stores.

**Imports today:** `core` (errors, logging, `content_hash`) and `llm` (message types,
`tokens.count_tokens` for the cache-hint token estimate).

**Does NOT:** call a model. Rendering produces messages; sending them is the caller's job via
`strata_forge.llm`. `CacheHints` is a value the caller reads — the LLM client does not consume it.
([ADR 0007](adr/0007-stable-prefix-dynamic-suffix-prompts.md))

### `strata_forge.datasets`
**Owns:** the frozen `Dataset` / `DatasetItem` models with content-hash identity, content-hash
versioning and diffing, the pluggable `DatasetStore` interface with in-memory and Langfuse backends,
the bidirectional Hugging Face Datasets bridge, and the `self_instruct` / `distill` synthetic
helpers.

**Imports today:** `core` (errors, `content_hash`), `llm` (message types and `LLMClient` for the
synthetic helpers), and `config` lazily inside the Langfuse store.

**Does NOT:** score or grade samples — that is `strata_forge.evals`.
([ADR 0009](adr/0009-datasets-langfuse-canonical-hf-exchange.md))

### `strata_forge.evals`
**Owns:** the concurrent experiment runner, deterministic and LLM-driven graders (exact match,
regex, JSON field and structure, LLM judge, pairwise, semantic similarity), the metric functions,
Markdown and HTML report rendering, parameter sweeps, Langfuse trace replay, and the Wilson-bounded
CI regression gate.

**Imports today:** `llm` at runtime, `datasets` for typing only, and `config` lazily inside trace
replay. It does not import `prompts`: prompt rendering arrives as a caller-supplied
`PromptRenderer` callable, which keeps the runner independent of any one templating layer.

**Does NOT:** define what a good output is. Graders are supplied by the caller; the runner only
orchestrates and aggregates them.
([ADR 0010](adr/0010-evals-experiment-as-data-pluggable-graders.md))

### `strata_forge.agents`
**Owns:** the `Agent` runtime and `AgentResult`, the built-in tools (`calculator`, `fetch_url`, and
the `fs_read_tool` / `web_search_tool` factories), `ConversationMemory` and vector-backed
`EpisodicMemory`, and the multi-agent patterns `handoff` and `critic_refiner_run`.

**Imports today:** `llm` and `rag` (the vector-store primitives its episodic memory builds on).

**Does NOT:** implement tool plumbing, message types, a tool loop, or a structured-output path —
all four come from `strata_forge.llm`. It is a composition layer, not a parallel runtime, and it is
not built on PydanticAI: that alternative was considered and rejected.
([ADR 0011](adr/0011-agents-thin-wrapper-over-forge-llm.md))

### `strata_forge.rag`
**Owns:** the five runtime-checkable Protocols (`Embedder`, `Chunker`, `Retriever`, `VectorStore`,
`Reranker`), the `LiteLLMEmbedder`, `RecursiveChunker`, dense / BM25 / hybrid-RRF retrievers, the
in-memory and Qdrant vector stores, Cohere and cross-encoder rerankers, and the composable
`RAGPipeline`. BM25 is a self-contained pure-Python implementation with no external dependency.

**Imports today:** nothing from `strata_forge`. `LiteLLMEmbedder` calls `litellm.aembedding`
directly rather than routing through `LLMClient` — a deliberate seam, because embeddings have a
different request shape, a different cost model, and no need for the completion client's fallback,
cache, or tool machinery. ([ADR 0012](adr/0012-rag-protocols-and-vector-store-relocation.md))

**Does NOT:** grow `LLMClient` to cover embeddings, and does not own the agent-facing re-exports —
`strata_forge.agents.memory` re-exports the relocated vector-store names, so the dependency arrow
stays `agents → rag`.

**Note:** the module does ship one prompt string. `DEFAULT_AUGMENT_TEMPLATE` and
`RAGPipeline.augment_prompt` produce a ready-to-send augmented prompt; callers who want their own
wording pass a different template.

### `strata_forge.compute`
**Owns:** `Task` and `ResourceSpec` as frozen, YAML-round-trippable data; `Job` / `JobStatus`; the
async `Backend` Protocol (`submit`, `status`, `logs`, `cancel`, `cleanup`, `read_file`) with local
subprocess, SSH, and SkyPilot implementations; the `safe_workdir_relpath` confinement guard;
`serving_endpoint` plus the vLLM / TGI / SGLang task builders; and the concurrency-bounded
`BatchInferenceRunner`.

**Imports today:** `llm`, entirely under `TYPE_CHECKING` — the batch runner is typed against
`LLMClient` but never constructs one. Job ids come from `uuid.uuid4()`, not `core.uuid7`, and no
backend reads `strata_forge.config`: SSH hosts, usernames, and keys are constructor arguments.

**Does NOT:** implement the workloads it launches. A `Task` is a shell command plus resources;
what runs on the far end is the caller's business.
([ADR 0013](adr/0013-compute-task-and-backend-shapes.md),
[ADR 0016](adr/0016-backend-read-file.md))

### `strata_forge.training`
**Owns:** frozen `SFTConfig` and the preference configs (DPO / ORPO / KTO / GRPO) that render into
TRL trainer kwargs, the `SFTRunner` and `PreferenceRunner`, `LoRAConfig` / `QLoRAConfig` PEFT
wiring, chat-template formatting, sequence packing, and the JSONL progress-event stream
(`ProgressEvent`, `JsonlProgressWriter`, `trainer_callback`).

**Imports today:** `llm.messages` only, for the chat-template formatter.

**Does NOT:** launch remote jobs — that is `strata_forge.compute` — and ships no trainer
entry-point script of its own. The runners' `train()` methods are synchronous, a deliberate
exception to the async-first rule: TRL's training loop is CPU-and-GPU-bound and blocking, so
wrapping it in `async def` would advertise a concurrency it cannot deliver.

### `strata_forge.pipelines`
**Owns:** runnable entrypoints that compose the rest of the library into one process you launch on
a machine you control. `inference_runner` reads an inert JSON run spec from the `STRATA_RUN_CONFIG`
environment variable, validates it into a frozen `RunSpec`, renders one prompt per dataset row,
starts a local vLLM server, fans the prompts through `BatchInferenceRunner`, and writes a Parquet
result set either to a Hugging Face dataset repo or to local disk — appending `ProgressEvent` lines
to a file an orchestrator can tail.

**Imports today:** `compute`, `llm`, `storage`, and `training.progress`.

**Does NOT:** provide an in-process API. These modules are launched with `python -m`, not imported
as a library surface; `RunSpec` and `render_template` are exported so callers can build and test a
spec, not so they can drive the run from inside their own event loop.

### `strata_forge.cli`
**Owns:** the Typer application installed as the `strata-forge` console script, with the commands
`doctor` and `chat` plus the `prompts`, `datasets`, `eval`, `experiments`, `compute`, `train`, and
`serve` sub-applications, and the shared helpers in `cli.helpers` (`run_async`, `error_exit`, and
the store factories that pick Langfuse when configured).

**Imports today:** `compute`, `config`, `core`, `datasets`, `evals`, `llm`, `prompts`, `training`.
It bridges to async through its own `run_async` helper rather than through `strata_forge.sync`.

**Does NOT:** define business logic. Every subcommand is a thin wrapper over its underlying module.

### `strata_forge.sync`
**Owns:** `asyncio.run` facades for exactly four `LLMClient` methods — `complete`,
`complete_structured`, `stream`, and `run_tool_loop`.

**Imports today:** `llm` only.

**Does NOT:** cover the whole library. `stream_tool_loop` has no facade, nothing outside
`strata_forge.llm` is wrapped, and the sync `stream()` drains every chunk before it returns, so it
is a convenience for scripts rather than a streaming API. Nothing here reimplements async logic:
each function is one `asyncio.run` over the async equivalent.
([ADR 0002](adr/0002-async-only-public-api.md))

## Cross-cutting guarantees

- **Provider SDKs stay at the transport seam.** Provider clients are constructed only inside
  `src/strata_forge/llm/providers/`, which is the only place that calls LiteLLM's `acompletion` and
  `astream`. One deliberate exception: `src/strata_forge/llm/errors.py` imports `openai` to name the
  exception classes LiteLLM re-raises, so the normalizer can catch them. Everywhere else uses
  `LLMClient` or the `provider_extras` passthrough.
  ([ADR 0001](adr/0001-litellm-as-transport.md))
- **Configuration flows through `get_settings()`.** Modules that need a setting call
  `strata_forge.config.get_settings()` rather than reading the environment. Four places read
  `os.environ` directly and each is a considered exception, not drift: the Langfuse prompt and
  dataset stores fall back to `LANGFUSE_*` when a caller injects its own SDK client and no settings
  are available; `training.progress` accepts `FORGE_PROGRESS_PATH` so a remote trainer can be
  pointed at a progress file without a config file; and `pipelines.inference_runner` is by design an
  environment-driven entrypoint (`STRATA_RUN_CONFIG`, `FORGE_PROGRESS_PATH`, `HF_WRITE_TOKEN`).
  Anything new that reads the environment directly needs a reason of the same shape.
- **Errors normalize once, at the seam.** LiteLLM exceptions become `ProviderError` subclasses
  inside `strata_forge.llm.errors` and never escape as raw SDK types. Higher modules raise their own
  `ForgeError` subclasses for domain failures and plain `ValueError` / `TypeError` for argument
  mistakes; nothing re-throws a provider SDK exception.
  ([ADR 0003](adr/0003-exception-hierarchy.md))
- **Persistent writes go through `strata_forge.storage`** whenever they outlive a single process.
  Process-local artefacts — the NDJSON diagnostic dump, the CLI's `~/.forge` state, a job's progress
  file — write directly and are expected to.
