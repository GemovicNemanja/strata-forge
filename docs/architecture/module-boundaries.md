# Module boundaries

Authoritative rules for what each `forge.*` module owns, what it imports, and what it explicitly does not do. These rules are enforced by code review (and, where reasonable, by ruff lint configurations); violations require an ADR.

## Dependency direction (allowed imports)

| Module | May import from | Must NOT import from |
|---|---|---|
| `forge.core` | — | any other `forge.*` |
| `forge.config` | `core` | `llm`, `prompts`, `tracing`, … |
| `forge.llm` | `core`, `config` | higher-level modules |
| `forge.tracing` | `core`, `config`, `llm` (for trace types) | `evals`, `agents`, `rag`, … |
| `forge.storage` | `core`, `config` | LLM/eval/agent layers |
| `forge.prompts` | `core`, `config`, `llm` (typing only), `tracing` | `evals`, `agents`, `rag` |
| `forge.datasets` | `core`, `config`, `llm` (embedding API), `storage`, `tracing` | `evals`, `agents`, `rag` |
| `forge.evals` | `core`, `config`, `llm`, `prompts`, `datasets`, `tracing`, `storage` | `agents` (other way around if needed) |
| `forge.rag` | `core`, `config`, `llm`, `storage`, `tracing` | `agents`, `evals`, `training`, `compute` |
| `forge.agents` | `core`, `config`, `llm`, `prompts`, `tracing`, `rag`, `storage` | `evals`, `training`, `compute` |
| `forge.compute` | `core`, `config`, `llm`, `storage`, `tracing` | `evals`, `agents`, `training` (other way around) |
| `forge.training` | `core`, `config`, `llm` (typing only), `datasets`, `storage`, `tracing`, `compute` | `evals`, `agents`, `rag` |
| `forge.cli` | any module | (no inverse rule — cli is a sink) |
| `forge.sync` | any async public surface | provider SDKs directly |

Heavy deps (`torch`, `transformers`, `trl`, `peft`, `skypilot`, `asyncssh`, `qdrant-client`, `vllm`, `pillow`, `redis`, `inspect-ai`) are imported **inside** functions that use them, never at module load.

## Per-module ownership

### `forge.core`
**Owns:** Exception hierarchy (`ForgeError` + subclasses), `@retry`, structlog setup, `BudgetContext`, `set_seed` / `content_hash` / `env_snapshot`, UUIDv7 helpers, shared type aliases.

**Does NOT:** Know about LLM specifics. Talk to networks. Read configuration directly (it accepts values; `forge.config` produces them).

### `forge.config`
**Owns:** `Settings` (Pydantic BaseSettings) and its sub-models. YAML overlay loader. `.env` integration. The `get_settings()` cached accessor.

**Does NOT:** Construct provider clients. Read provider-specific env vars except through declared sub-models. Make network calls.

### `forge.llm`
**Owns:** `LLMClient`, message and response types, tool calling primitives (`Tool`, `@tool`, `run_tool_loop`), structured output, multimodal image input, streaming, two-axis fallback, response cache (provider-agnostic), model registry, cost/token accounting, NDJSON diagnostic dump. Provider clients via LiteLLM wrappers in `providers/`.

**Does NOT:** Manage prompts, store datasets, run evaluations, build agents, perform retrieval. It is consumed by those modules; it never consumes them.

### `forge.tracing`
**Owns:** Langfuse client setup, `@traced` decorator, `traced_span` context manager, custom metric/score helpers, structlog → Langfuse correlation.

**Does NOT:** Auto-instrument application code beyond LLM calls — explicit `@traced` is the contract for non-LLM spans.

### `forge.storage`
**Owns:** `fsspec` gateway with auth wiring for S3, GCS, Azure Blob, HF Hub. Hugging Face Hub model push/pull. Dataset upload/download.

**Does NOT:** Cache LLM responses (that's `forge.llm.cache`).

### `forge.prompts`
**Owns:** Jinja2 environment with safe filters. Prompt registry (load/save/version) backed by Langfuse. Few-shot example injection from Langfuse datasets. Stable-prefix / dynamic-suffix split helpers for prompt caching.

**Does NOT:** Render prompts to LLM responses (that's the caller's job via `forge.llm`).

### `forge.datasets`
**Owns:** Dataset CRUD against Langfuse, HF Datasets bridge, synthetic data generators (self-instruct, distillation).

**Does NOT:** Score or grade samples — that's `forge.evals`.

### `forge.evals`
**Owns:** Experiment runner, graders (exact, JSON similarity, LLM-as-judge, pairwise, semantic), standard metrics, report rendering, parameter sweep machinery, trace replay, CI regression gate.

**Does NOT:** Define what a "good" output is — graders are user-supplied; the framework just orchestrates them.

### `forge.agents`
**Owns:** PydanticAI agent builder, built-in tools (`web_search`, `fs_read`, `fetch_url`, `calculator`), `ConversationMemory`, vector-backed `EpisodicMemory`, multi-agent patterns (Hand-Off, Critic-Refiner stub).

**Does NOT:** Implement tool plumbing — reuses `Tool`, `@tool`, `run_tool_loop` from `forge.llm`.

### `forge.rag`
**Owns:** Embedder (via `forge.llm`'s provider abstraction), Qdrant store wrapper, chunkers (recursive, token-based, semantic), retrieval (dense, BM25 via `bm25s`, hybrid RRF), rerankers (Cohere, cross-encoder), loaders (text + URL), composable pipeline.

**Does NOT:** Bundle a prompt template — pipelines hand documents off to callers, who own the prompt.

### `forge.compute`
**Owns:** SkyPilot submission via `sky.api.sdk`. `asyncssh` SSH backend. SkyPilot task YAML templates. Async concurrency-controlled batch inference.

**Does NOT:** Implement the trainer scripts themselves — those live in `forge.training` and are shipped as artifacts to remote machines.

### `forge.training`
**Owns:** TRL `SFTTrainer` wrapper, preference tuning (DPO/ORPO/KTO/GRPO), PEFT (LoRA/QLoRA), chat-template formatting, sequence packing, trainer entry-point scripts.

**Does NOT:** Launch remote jobs — that's `forge.compute`.

### `forge.cli`
**Owns:** Typer app + entry point; subcommands `chat`, `eval`, `experiments`, `prompts`, `datasets`, `train`, `serve`, `compute`, `doctor`.

**Does NOT:** Define business logic — every subcommand is a thin wrapper over its underlying module.

### `forge.sync`
**Owns:** Synchronous wrappers (`forge.sync.complete`, `.stream`, `.run_tool_loop`, …) over async public APIs via `asyncio.run`.

**Does NOT:** Reimplement anything async. Never a separate code path.

## Cross-cutting guarantees

- **Imports of provider SDKs** (`anthropic`, `openai`, `google-cloud-aiplatform`, `boto3`) appear ONLY inside `src/forge/llm/providers/`. Everywhere else uses `LLMClient` or the `provider_extras` passthrough.
- **Direct env-var reads** (`os.environ[...]`) appear ONLY inside `src/forge/config/`. Other modules consume `forge.config.get_settings()`.
- **Direct file-system writes for caches / dumps** route through `forge.storage` when persisting beyond a single process.
