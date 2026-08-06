# Module boundaries

Authoritative rules for what each `strata_forge.*` module owns, what it imports, and what it explicitly does not do. These rules are enforced by code review (and, where reasonable, by ruff lint configurations); violations require an ADR.

## Dependency direction (allowed imports)

| Module | May import from | Must NOT import from |
|---|---|---|
| `strata_forge.core` | — | any other `strata_forge.*` |
| `strata_forge.config` | `core` | `llm`, `prompts`, `tracing`, … |
| `strata_forge.llm` | `core`, `config` | higher-level modules |
| `strata_forge.tracing` | `core`, `config`, `llm` (for trace types) | `evals`, `agents`, `rag`, … |
| `strata_forge.storage` | `core`, `config` | LLM/eval/agent layers |
| `strata_forge.prompts` | `core`, `config`, `llm` (typing only), `tracing` | `evals`, `agents`, `rag` |
| `strata_forge.datasets` | `core`, `config`, `llm` (embedding API), `storage`, `tracing` | `evals`, `agents`, `rag` |
| `strata_forge.evals` | `core`, `config`, `llm`, `prompts`, `datasets`, `tracing`, `storage` | `agents` (other way around if needed) |
| `strata_forge.rag` | `core`, `config`, `llm`, `storage`, `tracing` | `agents`, `evals`, `training`, `compute` |
| `strata_forge.agents` | `core`, `config`, `llm`, `prompts`, `tracing`, `rag`, `storage` | `evals`, `training`, `compute` |
| `strata_forge.compute` | `core`, `config`, `llm`, `storage`, `tracing` | `evals`, `agents`, `training` (other way around) |
| `strata_forge.training` | `core`, `config`, `llm` (typing only), `datasets`, `storage`, `tracing`, `compute` | `evals`, `agents`, `rag` |
| `strata_forge.cli` | any module | (no inverse rule — cli is a sink) |
| `strata_forge.sync` | any async public surface | provider SDKs directly |

Heavy deps (`torch`, `transformers`, `trl`, `peft`, `skypilot`, `asyncssh`, `qdrant-client`, `vllm`, `pillow`, `redis`, `inspect-ai`) are imported **inside** functions that use them, never at module load.

## Per-module ownership

### `strata_forge.core`
**Owns:** Exception hierarchy (`ForgeError` + subclasses), `@retry`, structlog setup, `BudgetContext`, `set_seed` / `content_hash` / `env_snapshot`, UUIDv7 helpers, shared type aliases.

**Does NOT:** Know about LLM specifics. Talk to networks. Read configuration directly (it accepts values; `strata_forge.config` produces them).

### `strata_forge.config`
**Owns:** `Settings` (Pydantic BaseSettings) and its sub-models. YAML overlay loader. `.env` integration. The `get_settings()` cached accessor.

**Does NOT:** Construct provider clients. Read provider-specific env vars except through declared sub-models. Make network calls.

### `strata_forge.llm`
**Owns:** `LLMClient`, message and response types, tool calling primitives (`Tool`, `@tool`, `run_tool_loop`), structured output, multimodal image input, streaming, two-axis fallback, response cache (provider-agnostic), model registry, cost/token accounting, NDJSON diagnostic dump. Provider clients via LiteLLM wrappers in `providers/`.

**Does NOT:** Manage prompts, store datasets, run evaluations, build agents, perform retrieval. It is consumed by those modules; it never consumes them.

### `strata_forge.tracing`
**Owns:** Langfuse client setup, `@traced` decorator, `traced_span` context manager, custom metric/score helpers, structlog → Langfuse correlation.

**Does NOT:** Auto-instrument application code beyond LLM calls — explicit `@traced` is the contract for non-LLM spans.

### `strata_forge.storage`
**Owns:** `fsspec` gateway with auth wiring for S3, GCS, Azure Blob, HF Hub. Hugging Face Hub model push/pull. Dataset upload/download.

**Does NOT:** Cache LLM responses (that's `strata_forge.llm.cache`).

### `strata_forge.prompts`
**Owns:** Jinja2 environment with safe filters. Prompt registry (load/save/version) backed by Langfuse. Few-shot example injection from Langfuse datasets. Stable-prefix / dynamic-suffix split helpers for prompt caching.

**Does NOT:** Render prompts to LLM responses (that's the caller's job via `strata_forge.llm`).

### `strata_forge.datasets`
**Owns:** Dataset CRUD against Langfuse, HF Datasets bridge, synthetic data generators (self-instruct, distillation).

**Does NOT:** Score or grade samples — that's `strata_forge.evals`.

### `strata_forge.evals`
**Owns:** Experiment runner, graders (exact, JSON similarity, LLM-as-judge, pairwise, semantic), standard metrics, report rendering, parameter sweep machinery, trace replay, CI regression gate.

**Does NOT:** Define what a "good" output is — graders are user-supplied; the framework just orchestrates them.

### `strata_forge.agents`
**Owns:** PydanticAI agent builder, built-in tools (`web_search`, `fs_read`, `fetch_url`, `calculator`), `ConversationMemory`, vector-backed `EpisodicMemory`, multi-agent patterns (Hand-Off, Critic-Refiner stub).

**Does NOT:** Implement tool plumbing — reuses `Tool`, `@tool`, `run_tool_loop` from `strata_forge.llm`.

### `strata_forge.rag`
**Owns:** Embedder (via `strata_forge.llm`'s provider abstraction), Qdrant store wrapper, chunkers (recursive, token-based, semantic), retrieval (dense, BM25 via `bm25s`, hybrid RRF), rerankers (Cohere, cross-encoder), loaders (text + URL), composable pipeline.

**Does NOT:** Bundle a prompt template — pipelines hand documents off to callers, who own the prompt.

### `strata_forge.compute`
**Owns:** SkyPilot submission via `sky.api.sdk`. `asyncssh` SSH backend. SkyPilot task YAML templates. Async concurrency-controlled batch inference.

**Does NOT:** Implement the trainer scripts themselves — those live in `strata_forge.training` and are shipped as artifacts to remote machines.

### `strata_forge.training`
**Owns:** TRL `SFTTrainer` wrapper, preference tuning (DPO/ORPO/KTO/GRPO), PEFT (LoRA/QLoRA), chat-template formatting, sequence packing, trainer entry-point scripts.

**Does NOT:** Launch remote jobs — that's `strata_forge.compute`.

### `strata_forge.cli`
**Owns:** Typer app + entry point; subcommands `chat`, `eval`, `experiments`, `prompts`, `datasets`, `train`, `serve`, `compute`, `doctor`.

**Does NOT:** Define business logic — every subcommand is a thin wrapper over its underlying module.

### `strata_forge.sync`
**Owns:** Synchronous wrappers (`strata_forge.sync.complete`, `.stream`, `.run_tool_loop`, …) over async public APIs via `asyncio.run`.

**Does NOT:** Reimplement anything async. Never a separate code path.

## Cross-cutting guarantees

- **Imports of provider SDKs** (`anthropic`, `openai`, `google-cloud-aiplatform`, `boto3`) appear ONLY inside `src/strata_forge/llm/providers/`. Everywhere else uses `LLMClient` or the `provider_extras` passthrough.
- **Direct env-var reads** (`os.environ[...]`) appear ONLY inside `src/strata_forge/config/`. Other modules consume `strata_forge.config.get_settings()`.
- **Direct file-system writes for caches / dumps** route through `strata_forge.storage` when persisting beyond a single process.
