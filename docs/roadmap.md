# Roadmap

The current state of the project, broken down by phase. Phases that have
shipped are documented in detail (what landed and *why*); the active
phase carries its implementation plan; future phases get a one-paragraph
overview that will be expanded when each one starts.

Design rationale that's bigger than a single phase lives in
[`docs/architecture/overview.md`](architecture/overview.md) and the
[ADRs](architecture/adr/). Per-module reference docs live in
[`docs/modules/`](modules/).

| # | Phase | Status |
|---|---|---|
| 0 | Foundations | ✅ done |
| 1 | LLM abstraction (`strata_forge.llm`, `strata_forge.sync`) | ✅ done |
| 2 | Prompts, tracing, datasets, evals | ✅ done |
| 3 | Agents (`strata_forge.agents`) | ✅ done |
| 4 | RAG (`strata_forge.rag`) | ✅ done |
| 5 | Remote compute + inference + training (`strata_forge.compute`, `strata_forge.training`) | ✅ done |
| 6 | Storage (`strata_forge.storage`) | ✅ done |
| 7 | CLI completion (`strata_forge.cli`) | ✅ done |
| 8 | DX maturity (notebooks, Docker hardening, examples polish) | ✅ done |
| 9 | Testing maturity (cassette refresh CI, eval gate, security audit) | ✅ done |

---

## Phase 0 — Foundations ✅

Build the floor before any LLM code lands, so every later module can sit
on a finished base. Phase 0 is intentionally infrastructure-heavy: lint,
type, test, docs, agent rules, and the cross-cutting `strata_forge.core` +
`strata_forge.config` modules. Nothing here is "LLM-aware."

### What shipped

**Packaging & developer experience**

- `pyproject.toml` with `uv` as the package manager, Python `>=3.14`, and
  the full extras matrix (`[redis]`, `[multimodal]`, `[inspect]`,
  `[compute]`, `[serving]`, `[finetuning]`, `[all]`). Heavy deps are
  gated behind extras so a base install stays slim.
- `ruff` (lint + format, single config in pyproject) and `pyright`
  (`strict` mode on `src/strata_forge`) with a uniform line length of 100.
- `pre-commit` hooks: ruff, pyright, trailing-whitespace, EOF-newline,
  no-merge-markers.
- `Makefile` one-command entrypoints: `fmt`, `lint`, `type`, `test`,
  `vcr-replay`, `vcr-record`, `doctor`, `stack-up`, `stack-down`.
- GitHub Actions: `ci.yml` runs ruff + pyright + pytest + cassette
  replay on every PR; `nightly.yml` is scaffolded for live integration,
  the eval gate, cassette refresh, and a security audit.
- Docker Compose stack (`docker/compose.yaml`): Langfuse + Postgres +
  Qdrant + Redis. `make stack-up` boots it locally.
- The entire module tree exists as empty packages from day one (`agents/`,
  `rag/`, `training/`, …), each with a stub `README.md` and a
  module-specific `CLAUDE.md` — so contributors and agents see the
  intended shape before any code lands.

**`strata_forge.core`** — cross-cutting utilities, no `strata_forge.*` imports

- `errors.py` — the full exception hierarchy rooted at `ForgeError`:
  `ConfigError`, `ProviderError` + six subclasses (`Auth`, `RateLimit`,
  `Timeout`, `BadRequest`, `Server`, `ContentFilter`),
  `BudgetExceededError`, `ValidationError`, `CacheError`,
  `RegistryError`, `FallbackExhaustedError`. Centralizing the hierarchy
  here lets the `@retry` decorator and fallback runner pattern-match on
  class hierarchy without circular imports.
- `retry.py` — tenacity-backed `@retry` decorator that works for sync
  and async callables, with a default predicate matching only transient
  `ProviderError` subclasses. Configurable per call site.
- `ids.py` — UUIDv7 generator (time-ordered, k-sortable per RFC 9562),
  `correlation_id_var` ContextVar, and `new_correlation_id()` /
  `get_correlation_id()` / `set_correlation_id()` helpers. Time-ordered
  correlation IDs make distributed logs sortable without joins.
- `logging.py` — structlog configured for pretty terminal output when
  stdout is a TTY and JSON output when it isn't, with a processor that
  injects `correlation_id` into every record. Correlation IDs propagate
  across `await` boundaries via the ContextVar — never passed manually.
- `budget.py` — `BudgetContext` async context manager that enforces a
  USD and/or token ceiling. `consume()` pre-checks the chain before
  recording spend, so `BudgetExceededError` fires *before* the
  overage. Nested budgets share the parent ceiling unless explicitly
  `isolated=True`.
- `repro.py` — `set_seed(n)` seeds Python/numpy/torch (lazy imports for
  the optional ones); `content_hash(obj)` produces stable SHA-256
  digests over JSON-canonical forms (used as the cache key and dataset
  version primitive in later phases); `env_snapshot()` captures Python
  version + git SHA + tracked-package versions for run provenance.
- `types.py` — shared type aliases (`JSONValue`, `PathLike`).

**`strata_forge.config`** — single source of truth for runtime configuration

- `settings.py` — root `Settings(BaseSettings)` with one Pydantic
  sub-model per concern (`LangfuseConfig`, `RedisConfig`, `QdrantConfig`,
  `StorageConfig`, `LoggingConfig`, `DiagnosticConfig`, plus a
  placeholder `ProvidersConfig` that Phase 1 populates).
- `overlays.py` — YAML overlay loader with deep merge so a
  `configs/<FORGE_PROFILE>.yaml` file overrides individual fields without
  erasing siblings.
- `env.py` — `.env` file loader. Loading precedence is
  defaults → YAML overlay → `.env` → env vars → in-code overrides.
- `get_settings()` is `@functools.lru_cache(maxsize=1)`-d, with
  `reset_settings()` for the autouse test fixture in
  `tests/conftest.py`. Secrets stay in `SecretStr` so they don't print
  in error messages or logs.

**`strata_forge.cli`** — Typer entry point and the `doctor` command

- `strata-forge doctor` validates env-var presence, runs lightweight provider
  auth probes (stubbed where the LLM module didn't exist yet), reports
  package versions and service reachability (Langfuse/Redis/Qdrant). It
  *skips* unconfigured providers cleanly instead of failing — a missing
  Anthropic key reports "skipped: missing env" and moves on.
- The other CLI commands (`chat`, `eval`, `experiments`, `prompts`,
  `datasets`, `train`, `serve`, `compute`) are stubbed with "not yet
  implemented (phase X)" messages so the entry-point shape is committed.

**Architecture docs + ADRs**

- `docs/architecture/overview.md` — the design narrative.
- `docs/architecture/module-boundaries.md` — what can import what.
- Six ADRs:
  - 0001 — LiteLLM as the transport seam
  - 0002 — async-only public API (sync wrappers in `strata_forge.sync` only)
  - 0003 — exception-based error model rooted at `ForgeError`
  - 0004 — model registry scope (OpenAI + Anthropic + Google only)
  - 0005 — two-axis fallback (model-level + provider-level)
  - 0006 — tool calling as an LLM primitive (not deferred to agents)

**Agent rules + Cursor MDC**

- Root `CLAUDE.md` (north star, architecture principles, coding
  standards, testing/docs discipline, commit conventions, the
  self-maintenance protocol).
- `AGENTS.md` symlink for cross-tool compatibility (Cursor / Codex /
  Copilot).
- `.cursor/rules/000-core.mdc` through `500-docs.mdc` — glob-scoped MDC
  rules that mirror CLAUDE.md for the IDE's rule engine.
- Per-module `CLAUDE.md` stubs everywhere — each module gets one before
  any code lands, so module-specific conventions exist before
  contributors do.

### Why this order

- Doing tooling, errors, logging, and budgets *first* means every
  Phase-1 module can `raise ProviderError` knowing the class exists,
  `await budget.consume(...)` knowing the context manager works, and
  use the configured logger knowing `correlation_id` will propagate.
- Building the agent rules and ADRs before any module code commits the
  design ahead of the implementation — when Phase 1 ran into ambiguity
  (e.g., should tool calling live in `agents/` or `llm/`?), the ADR
  already had the answer.
- Module stubs from day one let module-specific `CLAUDE.md` files exist
  before the code does, so agents working in those directories pick up
  the right conventions on the first edit.

### Phase 0 verification

`uv sync` works on a fresh clone; `make fmt && make lint && make type
&& make test` are all green; pre-commit hooks are installed and
passing; `strata-forge doctor` reports cleanly against an empty `.env`;
`make stack-up` boots the local services.

---

## Phase 1 — LLM abstraction ✅

Phase 1 turns LiteLLM into a typed, async, Forge-flavored client with
caching, fallback, tools, structured output, multimodal, streaming,
cost tracking, and observability. Everything in higher-level modules
goes through this seam — the rest of the project never imports
provider SDKs directly.

Detailed reference: [`docs/modules/llm.md`](modules/llm.md).

### What shipped

**Foundation (1.1 – 1.4)**

- `errors.py` — `map_litellm_exception(e, *, model, provider)`
  normalizes every LiteLLM exception into a `ProviderError` subclass,
  annotated with the active `(model, provider)`. Most-specific-first
  dispatch over content-filter, auth, rate-limit, timeout, bad request,
  and server errors. Raw LiteLLM exceptions never bubble out.
- `registry.py` + `registry_data.yaml` — nine logical models curated
  per ADR 0004: Claude Opus 4.7 / Sonnet 4.6 / Haiku 4.5; GPT-5.5 /
  Pro / Thinking / Instant; Gemini 3.1 Pro / Flash-Lite. Each model has
  one or more `ProviderRoute(provider, provider_model_id, is_default)`
  entries, capabilities (tools, vision, prompt caching, structured
  output), pricing per million tokens, and aliases. YAML validation
  catches drift on import.
- `routing.py` — `ModelRoute` frozen-slotted dataclass and
  `resolve(model, provider=None)` that returns the concrete dispatch
  target. Aliases resolve before lookup; unknown model raises
  `RegistryError(reason="unknown_model")`; unsupported `(model,
  provider)` raises `RegistryError(reason="unsupported_route")`.
- `messages.py` + `responses.py` — Pydantic v2 message hierarchy
  (`SystemMessage`, `UserMessage`, `AssistantMessage`,
  `ToolResultMessage`) with a `Message` namespace for ergonomic
  construction. `LLMResponse` carries text, tool calls, finish reason,
  usage, cost, route, cache_hit, and latency. `validate_conversation()`
  enforces that every `tool_call_id` in a result references a prior
  call.

**Providers (1.5 – 1.11)**

- `providers/base.py` — `ProviderClient(ABC)` with `name`/`litellm_prefix`
  class vars and abstract `auth_kwargs()`. Concrete `acompletion` and
  `astream` wrap `litellm.acompletion` and forward verbatim.
- `providers/config.py` — `OpenAIConfig`, `AnthropicConfig`,
  `VertexConfig`, `BedrockConfig`, `AzureConfig`, `OpenAICompatConfig`,
  each with its own env-var prefix and `SecretStr` credentials.
- `providers/openai.py`, `anthropic.py`, `vertex.py`, `bedrock.py`,
  `azure.py`, `openai_compat.py` — one provider client per backend.
  `VertexProvider.litellm_model()` overrides routing to send Claude-on-
  Vertex through `anthropic_vertex/` instead of `vertex_ai/`. Each
  provider also exports a `to_<provider>_tool_schema` serializer (only
  three — OpenAI, Anthropic, Gemini — because Bedrock/Azure/compat
  reuse OpenAI's shape).

**Token & cost accounting (1.12)**

- `tokens.py` — `count_tokens(text, *, model=None)` with lazy tiktoken
  import. GPT-5 family uses `o200k_base`; everything else uses
  `cl100k_base` (an estimate for Claude/Gemini that's close enough for
  budgeting).
- `cost.py` — `compute_cost(usage, model)` against the registry's
  per-million-token rates, including cache-read / cache-write rates when
  the provider reports them. Falls back to the input rate when cache
  rates aren't set.

**Multimodal + streaming (1.13 – 1.14)**

- `multimodal.py` — `ImageContent` Pydantic model with `from_url`,
  `from_path` (auto-detects MIME), `from_bytes(data, mime)` factories
  and `to_openai_format` / `to_anthropic_format` / `to_gemini_format`
  serializers. `downscale_image(data, *, max_dimension, quality)` lazy-
  imports Pillow behind the `[multimodal]` extra.
- `streaming.py` — async iterator helpers: `accumulate_text(chunks)`,
  `accumulate_tool_calls(chunks)` (validates id+name+JSON args), and a
  `JSONAccumulator` class with `feed`, `is_complete`, `parse`, `reset`
  for partial-JSON during structured-output streaming.

**Tools + structured output (1.15 – 1.16)**

- `tools.py` — frozen `Tool` dataclass and `@tool` decorator that
  extracts the Pydantic args model from the function's annotation via
  `typing.get_type_hints` (works under `from __future__ import
  annotations`). `Tool.invoke(args_dict)` validates through the args
  model and calls the wrapped function. `ToolLoopExceededError` carries
  `max_iterations` + `iterations` for the loop helper.
- `schemas.py` — Pydantic → JSON-Schema canonicalization with
  `$defs`/`$ref` inlining; `to_openai_response_format` (strict mode
  with `additionalProperties: false` everywhere);
  `to_anthropic_forced_tool_schema` (returns `(tool, tool_choice)`);
  `to_gemini_response_schema` (strips OpenAPI-incompatible keys);
  `make_reprompt_instruction` + `parse_json_response` for the fallback
  retry path; `StructuredOutputError` for retry exhaustion.

**Cache + fallback + diagnostic (1.17 – 1.19)**

- `cache.py` — `cache_key()` computes a SHA-256 over the *logical*
  model plus the canonical request — **provider is intentionally not
  in the key** so cached hits survive provider-level failover.
  `CacheBackend(ABC)` + `InMemoryCache` (bounded LRU on `OrderedDict`)
  + `RedisCache` (lazy-imports `redis.asyncio`, pickle serialization,
  prefix support, optional TTL).
- `fallback.py` — `ModelFallback(model, providers=None)` entries plus
  `normalize_fallback_chain()` that expands bare strings. The runner
  `run_with_fallback(chain, call, ...)` loops provider-level (inner)
  inside each entry, then model-level (outer) across entries. Per-error
  rules per ADR 0005: rate-limit/timeout/server retry then advance;
  auth advances; bad-request advances by default (configurable to
  abort); content-filter short-circuits the whole chain. Every attempt
  is wrapped in `strata_forge.core.retry.retry` so transient blips retry
  before counting as exhausted. `FallbackExhaustedError.causes` carries
  every `(model, provider, error)` triple.
- `diagnostic.py` — `DiagnosticRecord` Pydantic model + async
  `write_diagnostic_record()` that appends NDJSON to
  `FORGE_DIAGNOSTIC_PATH` when `FORGE_DIAGNOSTIC_ENABLED=true`. File I/O
  runs on a worker thread via `anyio.to_thread.run_sync`; a module-level
  `anyio.Lock` serializes concurrent writers to avoid torn lines. Records
  are plain-JSON-serializable so replay/analytics consumers don't need
  to import Forge classes.

**Public client + sync wrappers (1.20 – 1.21)**

- `client.py` — `LLMClient` ties everything together. Constructor
  takes either `(model, provider)` for a single route or `chain=` for
  a fallback chain (also via `LLMClient.with_fallbacks(...)`). Methods:
  `complete()`, `complete_structured()` → `StructuredResponse[M]` with
  `.parsed: M`, `stream()` (bypasses cache + fallback —
  mid-stream provider switch is impractical), `run_tool_loop()` for
  multi-turn tool use with `is_error=True` results when a tool throws
  or doesn't exist. Capability gate against the registry fires before
  any HTTP when tools are requested against a non-tool-capable model.
- `sync.py` — `strata_forge.sync.complete`, `complete_structured`, `stream`,
  `run_tool_loop`. Each wraps a single `asyncio.run`. Pass `client=` to
  reuse a pre-built `LLMClient` or `model=` / `provider=` / `chain=`
  for one-shot. Sync `stream()` drains every chunk before returning the
  iterator (true sync streaming is structurally impossible through
  `asyncio.run`).

**Testing, examples, docs (1.22 – 1.24)**

- `tests/vcr/test_providers.py` — 48 cassette-replay tests parametrized
  across (provider × scenario). Each test skips cleanly with the
  expected cassette path in the reason when no cassette exists;
  `RECORD=1` triggers live recording with secret scrubbing per
  `tests/vcr/conftest.py`.
- `examples/01_basic_completion.py` through `examples/10_cache_hit.py`
  — ten runnable demos sharing `_common.py` for argparse + env-check +
  the route/usage/cost/latency footer. Each exits cleanly with a
  `[skip] <provider>: missing env vars: ...` message when run without
  keys.
- `docs/modules/llm.md` — full module reference (~600 lines): registry
  table, API matrix, message types, routing, fallback rules per error
  class, cache, structured output, tools, multimodal, streaming, cost
  + budgets + diagnostic, full error taxonomy, sync wrappers,
  troubleshooting.

### Why these decisions

- **LiteLLM at the seam** (ADR 0001) lets Forge get provider breadth
  for free — adding a new vendor means a new `ProviderClient` subclass
  and registry entries, not a new HTTP / streaming / auth implementation.
- **Async-only public API** (ADR 0002) scales to batch and agent
  workloads. Sync wrappers exist for CLI/notebook ergonomics; mixing
  sync `httpx` and `requests` into async paths is banned.
- **Exception hierarchy** (ADR 0003) gives retry predicates, fallback
  rules, and diagnostic dumps a single contract to pattern-match on.
- **Registry scoped to latest models from three vendors** (ADR 0004) is
  a curated choice — the abstraction supports adding more vendors but
  the registry refuses to grow without an ADR.
- **Two-axis fallback** (ADR 0005) handles both the "same model, this
  provider is rate-limited" failure mode (Claude on Anthropic 429s,
  succeed on Bedrock) and the "this model is exhausted, drop to a
  different model" mode in one chain.
- **Tool calling as an LLM primitive** (ADR 0006) lets Phase 3 agents
  reuse `Tool`, `@tool`, message types, and `run_tool_loop` directly
  without re-implementing tool plumbing.
- **Provider-agnostic cache key** — including the provider in the key
  would invalidate the cache on every provider failover, defeating its
  purpose. Excluding it means a hit cached on Anthropic is valid when
  the next call would have gone to Bedrock.
- **Streaming bypasses cache and fallback** — chunks have
  provider-specific shapes and no useful "final form" mid-stream, and
  transferring a stream mid-flight to a new provider isn't safe.

### Phase 1 verification

`make fmt && make lint && make type && make test && make vcr-replay`
all green; **752 tests** pass + 48 VCR tests skip with explicit
"no cassette" reasons; pyright 0 errors; coverage on `src/strata_forge/llm/`
is **96%** (target ≥ 90% per the LLM module's CLAUDE.md); `forge
doctor` reports the environment without crashing.

---

## Phase 2 — Prompts, tracing, datasets, evals ✅

Phase 1 gave us a typed LLM client. Phase 2 makes it useful for
*experiments*: prompts you can version, traces you can inspect, datasets
you can iterate against, and evaluations that produce reports + a CI
gate.

Four modules; ~50 sub-phases total. Ordered so each module only depends
on what's already shipped:

1. `strata_forge.prompts` — no `strata_forge.*` dependency outside `core` + `llm`.
2. `strata_forge.tracing` — cross-cutting; reuses `LangfuseConfig` from
   `strata_forge.config`; doesn't import `strata_forge.llm`.
3. `strata_forge.datasets` — uses `strata_forge.llm` (for synthetic data) and
   `strata_forge.tracing` (for run provenance).
4. `strata_forge.evals` — consumes all three.

### 2.1 — `strata_forge.prompts` ✅

Jinja2 templating with safe filters; a store-agnostic prompt registry;
explicit stable-prefix / dynamic-suffix split so provider prompt caching
works without per-call hand-tuning; rendering to `strata_forge.llm.Message`
instances.

| Sub-phase | Deliverable |
|---|---|
| 2.1.1 | Module skeleton + ADR: stable-prefix / dynamic-suffix design |
| 2.1.2 | `template.py` — Jinja2 environment (no I/O extensions, safe filter set), `PromptTemplate` wrapper |
| 2.1.3 | `cache_aware.py` — `StableDynamicSplit`, per-provider cache markers |
| 2.1.4 | `variables.py` — extract refs from body, validate against declared variables |
| 2.1.5 | `registry.py` — `PromptRegistry`, abstract `PromptStore` |
| 2.1.6 | `stores/memory.py` — in-process `PromptStore` |
| 2.1.7 | `stores/langfuse.py` — Langfuse-backed store, lazy import (`[langfuse]` extra) |
| 2.1.8 | `rendering.py` — render template + variables to `list[AnyMessage]` |
| 2.1.9 | Tests (templating, validation, cache split, mocked Langfuse) |
| 2.1.10 | Examples + `docs/modules/prompts.md` |
| 2.1.11 | Sign-off |

**Why a registry + a separate cache-aware split?** Versioning prompts
in Langfuse (rather than hard-coding them in Python) lets non-engineers
iterate on prompts without a code release; the cache-aware split is
orthogonal — it controls whether the stable system+exemplars chunk gets
flagged for provider-side prompt caching, which can cut cost by an
order of magnitude on repeated requests.

### 2.2 — `strata_forge.tracing` ✅

Langfuse observability layered on top of every other module via the
LiteLLM Langfuse callback (auto-traces every LLM call when keys are
configured); `@traced` decorator and `traced_span()` context manager
for non-LLM code paths; structlog `correlation_id` correlation; score +
metric helpers for attaching feedback to a trace.

| Sub-phase | Deliverable |
|---|---|
| 2.2.1 | Module skeleton + ADR: tracing-as-cross-cutting (no upward imports) |
| 2.2.2 | `client.py` — lazy Langfuse client from `LangfuseConfig`, cached `get_client()` |
| 2.2.3 | `litellm_callback.py` — wire LiteLLM's Langfuse callback when keys present |
| 2.2.4 | `decorator.py` — `@traced` (sync + async), structlog `correlation_id` integration |
| 2.2.5 | `span.py` — `traced_span()` async context manager with parent linkage |
| 2.2.6 | `score.py` — `score_trace`, `score_observation` feedback attachment |
| 2.2.7 | `metrics.py` — numeric / categorical metric attachment |
| 2.2.8 | Tests (mocked Langfuse) + an `@integration` test against the docker-compose Langfuse |
| 2.2.9 | Examples + `docs/modules/tracing.md` |
| 2.2.10 | Sign-off |

**Why tracing as cross-cutting (not a hard dep of `strata_forge.llm`)?** The
LiteLLM Langfuse callback is registered at app startup and intercepts
every LLM call without `strata_forge.llm` importing anything from
`strata_forge.tracing`. This keeps the dependency arrow pointing the right way
(`tracing → llm`, never the reverse) so the LLM module stays usable in
contexts where tracing isn't configured.

### 2.3 — `strata_forge.datasets` ✅

Pydantic dataset schema with content-hash IDs; Langfuse Datasets API as
the canonical store; Hugging Face Datasets as the exchange format;
content-hash version tracking with deltas; synthetic-data primitives
(self-instruct + distillation) built on `strata_forge.llm`.

| Sub-phase | Deliverable |
|---|---|
| 2.3.1 | Foundation — ADR 0009; `schema.py` (`DatasetItem`, `Dataset` with content-hash IDs); `store.py` (`DatasetStore` ABC + `DatasetNotFoundError`); `versioning.py` (`dataset_version` + `diff` + `DatasetDelta`); `stores/memory.py` (`InMemoryDatasetStore` with content-hash versioning + dedup); module CLAUDE.md + README; unit tests across all four files. |
| 2.3.2 | External integrations — `stores/langfuse.py` (Langfuse Datasets CRUD behind the `[langfuse]` extra, lazy import, mocked-client tests); `hf_bridge.py` (bidirectional `datasets.Dataset` conversion behind a new `[hf]` extra, lazy import, mocked tests). |
| 2.3.3 | Synthetic data — `synthetic/self_instruct.py` (using `strata_forge.llm.LLMClient.complete_structured`); `synthetic/distillation.py` (teacher-student primitive); unit tests with mocked `LLMClient`. |
| 2.3.4 | Closeout — cross-module integration tests, examples 17–19, `docs/modules/datasets.md`, CLAUDE.md + architecture overview link wiring, full sign-off verification matrix. |

The 4-bundle grouping (previously 11 fine-grained sub-phases) trades a
bit of per-commit granularity for fewer validation cycles. Each bundle
still has a coherent theme — foundation pieces operate on the schema,
external integrations share the lazy-import-wrap-an-SDK pattern, the
synthetic helpers both run on `strata_forge.llm`, and the closeout is the
standard tests-examples-docs-signoff trio.

**Why Langfuse-canonical with an HF bridge?** Langfuse is where eval
runs and trace data already live, so keeping the canonical dataset
there makes "annotate this trace and add it to the eval set" a
single-system operation. HF Datasets is the standard exchange format
for training and external consumption, so the bridge is bidirectional
rather than one-way export.

### 2.4 — `strata_forge.evals` ✅

Experiment runner over the (model × prompt × dataset × grader) matrix;
a grader Protocol with deterministic, LLM-judged, pairwise, and
semantic implementations; metrics (accuracy, F1, BLEU, ROUGE);
parameter sweeps; trace replay; Markdown + HTML reports; a CI
eval-regression gate with Wilson CI + cost cap; `strata-forge eval` CLI.

| Sub-phase | Deliverable |
|---|---|
| 2.4.1 | Module skeleton + ADR: experiment design (4-axis matrix) |
| 2.4.2 | `experiment.py` — `Experiment`, `Trial`, `Outcome` data shapes |
| 2.4.3 | `runner.py` — async runner, parallelism-bounded |
| 2.4.4 | `graders/base.py` — `Grader` Protocol |
| 2.4.5 | `graders/exact.py` — `ExactMatch`, `Regex` |
| 2.4.6 | `graders/json.py` — `JSONStructure`, `JSONField` |
| 2.4.7 | `graders/llm_judge.py` — single-output judge via `complete_structured` |
| 2.4.8 | `graders/pairwise.py` — A/B comparison |
| 2.4.9 | `graders/semantic.py` — embedding similarity |
| 2.4.10 | `metrics.py` — accuracy, F1, BLEU, ROUGE (lazy imports for BLEU/ROUGE) |
| 2.4.11 | `sweeps.py` — parameter sweep across (model, prompt, sampling params) |
| 2.4.12 | `trace_replay.py` — replay a Langfuse trace with a new model/prompt |
| 2.4.13 | `reports/markdown.py` |
| 2.4.14 | `reports/html.py` (side-by-side example outputs) |
| 2.4.15 | `ci_gate.py` — Wilson CI accuracy regression + cost cap |
| 2.4.16 | `strata-forge eval` CLI wiring |
| 2.4.17 | Tests |
| 2.4.18 | Examples + `docs/modules/evals.md` |
| 2.4.19 | Sign-off |

**Why Wilson CI + cost cap on the CI gate?** Naive accuracy comparison
("PR accuracy ≥ baseline") is noisy on small eval sets; Wilson CI gives
us a principled way to say "this PR's lower bound is above last
release's upper bound." Pairing it with a cost cap stops PRs from
silently doubling the per-experiment spend.

---

## Phase 3 — Agents (`strata_forge.agents`) ✅

PydanticAI-backed agent builder, built-in tools (web_search, fs_read,
fetch_url, calculator), `ConversationMemory` + vector-backed
`EpisodicMemory`, multi-agent patterns (Hand-off, Critic-Refiner). The
module **reuses** `Tool`, `@tool`, message types, and `run_tool_loop`
from `strata_forge.llm` per ADR 0006 — no re-implementation. MCP server/tool
support is the natural addition here too: an adapter that exposes
existing `Tool` instances over MCP and consumes external MCP tools as
`Tool`s.

## Phase 4 — RAG (`strata_forge.rag`) ✅

Embedder built on `strata_forge.llm`; Qdrant vector store; chunkers (recursive,
token, semantic); retrieval (dense, BM25, hybrid via RRF); rerankers
(Cohere API + a local cross-encoder); document loaders (text + URL); a
composable pipeline class that wires these together.

## Phase 5 — Remote compute, inference, training ✅

`strata_forge.compute` ships typed `Task` / `Job` / `Status` shapes,
a `Backend` Protocol with three implementations (`LocalBackend`,
`SSHBackend`, `SkyPilotBackend`), a concurrency-bounded
`BatchInferenceRunner`, and serving-task builders for
vLLM / TGI / SGLang plus a `serving_endpoint` async context
manager that launches + health-checks + cleans up the server
around an OpenAI-compatible `LLMClient` session. The SSH and
SkyPilot backends lazy-import their SDKs behind the `[compute]`
extra; `LocalBackend`, batch inference, and the serving task
builders have no extra requirements. Reference doc:
[`docs/modules/compute.md`](modules/compute.md).

`strata_forge.training` ships fine-tuning primitives: `SFTConfig` /
`SFTRunner` over TRL `SFTTrainer`; `DPOConfig` / `ORPOConfig` /
`KTOConfig` / `GRPOConfig` dispatched through `PreferenceRunner`;
`LoRAConfig` / `QLoRAConfig` PEFT wrappers with QLoRA's
bitsandbytes config builder; chat-template formatting
(`apply_chat_template`, `conversation_to_dicts`) and greedy
first-fit sequence packing (`pack_sequences`). All heavy deps
(`torch`, `transformers`, `trl`, `peft`, `datasets`,
`accelerate`) sit behind the `[finetuning]` extra and are
lazy-imported inside the runners' `train` methods. Reference
doc: [`docs/modules/training.md`](modules/training.md).

## Phase 6 — Storage (`strata_forge.storage`) ✅

`strata_forge.storage` ships two cooperating clients. `StorageGateway`
is an async fsspec wrapper covering read / write / list /
copy / move / delete across local, S3, GCS, Azure Blob, HTTP, and
any other fsspec-compatible target; URL → protocol auto-detection,
per-protocol option forwarding, filesystem caching, same-protocol
native copy/move plus cross-protocol byte streaming, and an
explicit `NotImplementedError` on cross-protocol recursive copy
so callers reach for vendor-native tooling on big transfers.

`HFHubClient` is the async wrapper over `huggingface_hub.HfApi`
with repo lifecycle (create / delete / list), single-file
download/upload, whole-repo snapshot download/upload, and
convenience push/pull wrappers for both models and datasets;
every method exposes an `extras={}` passthrough so the full
HfApi surface stays reachable. Both clients lazy-import the
heavy deps (`fsspec`, `s3fs`, `gcsfs`, `adlfs`,
`huggingface_hub`) behind the new `[storage]` extra, so
`import strata_forge.storage` succeeds without it. Reference doc:
[`docs/modules/storage.md`](modules/storage.md).

## Phase 7 — CLI completion (`strata_forge.cli`) ✅

Every Forge module with a useful operator workflow now surfaces
a `forge` subcommand:

- `strata-forge chat` — one-shot or interactive REPL against an
  `LLMClient` with optional `--provider`, `--system`,
  `--temperature`, `--max-tokens`.
- `strata-forge prompts list/show/render` — wraps `PromptRegistry`,
  parses `--vars` as JSON, renders via
  `strata_forge.prompts.rendering.render`.
- `strata-forge datasets list/show/head` — wraps `DatasetStore`;
  `head` prints first N items as JSON with a truncation count.
- `strata-forge eval run/list/show` — runs a quick single-model eval
  with `exact_match` / `regex:` graders, writes markdown to
  `~/.forge/experiments/<name>.md`.
- `strata-forge experiments list/show/delete` — mirror group over the
  same report directory.
- `strata-forge compute submit/status/logs/cancel/cleanup/list` over
  any of `local`, `ssh`, `skypilot`; persists Job + backend
  state to `~/.forge/jobs/<id>.json` so subsequent commands
  reconstruct the right backend automatically.
- `strata-forge train sft/dpo` — wraps `SFTRunner` and the DPO branch
  of `PreferenceRunner`; supports `--adapter none/lora/qlora`
  with `--adapter-rank`; heavy ML deps lazy-imported behind
  `[finetuning]`.
- `strata-forge serve vllm/tgi/sglang` — builds the corresponding
  `strata_forge.compute.serving` task; default `--submit none` prints
  YAML, `--submit local` runs it on `LocalBackend` and hands
  off monitoring to `strata-forge compute`.

Shared helpers (`run_async`, `error_exit`, store factories)
live in `src/strata_forge/cli/helpers.py`. Reference doc:
[`docs/modules/cli.md`](modules/cli.md).

## Phase 8 — DX maturity ✅

Three pieces of DX polish landed:

- An import-only smoke harness sweeps every script under
  `examples/` via `importlib.util.spec_from_file_location`, so
  drift between the 33 example files and the current `strata_forge.*`
  public surface surfaces as a pytest failure. All 33 examples
  pass and the no-credentials ones (29 — local compute, 30 — SFT
  config, 31 — serving task, 32 — storage gateway, 33 — HF Hub
  dry run) were re-run end-to-end.
- Three Marimo notebook templates under `notebooks/`:
  `00_chat_starter.py` (LLMClient with model dropdown + prompt +
  temperature slider), `01_eval_iterate.py` (inline dataset +
  multi-model selector + run_experiment + outcomes table), and
  `02_rag_prototype.py` (inline docs through
  RecursiveChunker + InMemoryVectorStore + DenseRetriever via
  RAGPipeline). Marimo is in the `[dev]` dependency group; a
  matching pytest smoke harness imports each notebook to catch
  API drift.
- Dockerfile reworked into a two-stage build: the builder stage
  carries `build-essential` and the toolchain; the runtime stage
  is `python:3.14-slim` plus CA certs and a non-root `forge`
  user (uid 1000). uv is pinned to a release tag (no
  `:latest`), uv install layers use BuildKit cache mounts, and
  the runtime image carries a `forge --help` HEALTHCHECK.
- Extras audit: added `cohere>=5.13` to `[rag]` so
  `CohereReranker` matches its documented install hint.
  `sentence-transformers` is deliberately left out of `[rag]`
  (it would pull `torch`); users install it themselves when
  they want `CrossEncoderReranker`.

## Phase 9 — Testing maturity ✅

The test infrastructure itself:

- **Nightly cassette refresh** (`.github/workflows/nightly.yml`,
  `cassette-refresh` job): gated on whether any provider secret
  is set; when one is, deletes the committed cassettes and re-
  records via `pytest tests/vcr -x` with `RECORD=1` against live
  providers. If anything changed, opens a PR labelled
  `automated,cassettes` via `gh pr create` from a timestamped
  branch. Clean no-op when secrets aren't configured.
- **Canonical eval regression gate**
  (`scripts/run_eval_gate.py` + nightly `eval-gate` job): five-
  item capital-cities dataset baked into the script (deterministic
  across hosts), one model through `run_experiment` with
  `ExactMatch`, then `evaluate_ci_gate` against the committed
  `CIGateThresholds(min_pass_rate=0.80, max_cost_usd=0.10,
  use_wilson_ci=True)`. Exits non-zero on regression. Gated on
  `ANTHROPIC_API_KEY`.
- **Security audit**: `pip-audit --strict` runs in the nightly
  with the `|| true` swallow removed, so new CVEs surface as a
  job failure. (A CodeQL workflow is deliberately absent: code
  scanning requires GitHub Advanced Security on a private repo,
  so its runs can never upload results here — restore one from
  the git history if the repo goes public or gains GHAS.)

Also picked up a tokenizer-test cleanup along the way: removed
the `test_concatenation_is_at_least_as_many_as_each_part`
hypothesis property — it was structurally false for BPE
tokenizers (`'tree'+'true' → 4 > 1+1+1`, `'a'+'mecb' → 2 < 3`).

This finishes the roadmap. Future work (new vendors, new
training methods, MCP tool support, guardrails layer, advanced
telemetry) attaches against the now-complete public surface;
none of it requires re-opening the phase plan.
