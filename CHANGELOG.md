# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is below `1.0.0` the
public API may change in any release; breaking changes are called out explicitly in the entry that
introduces them.

## [Unreleased]

Nothing yet.

## [0.0.1] - 2026-08-06

Initial public release. The library was built privately and is published as `strata-forge` on PyPI
(import package `strata_forge`) under Apache-2.0. Everything below already existed at the point the
repository was opened; this entry is a description of that starting state rather than a record of
changes against a previous version.

### Added

- **`strata_forge.core`** — the strictly-upstream layer every other module builds on: the
  `ForgeError` exception hierarchy, a tenacity-backed `@retry` decorator, structlog configuration
  that injects a contextvar `correlation_id` into every record, `BudgetContext` cost and token
  ceilings, reproducibility helpers (`set_seed`, `env_snapshot`, `content_hash`), and UUIDv7 ids.
- **`strata_forge.config`** — a Pydantic Settings root with per-concern sub-models, `.env` loading,
  a cached `get_settings()` accessor, and standalone YAML overlay primitives (load, deep-merge,
  profile path resolution) that a caller wires in at its own bootstrap site.
- **`strata_forge.llm`** — an async client over LiteLLM owning the typed layer: Pydantic messages
  and responses, structured output, tool calling with a multi-turn (and streaming) loop, image
  input, streaming accumulators, fallback across both models and providers, a provider-agnostic
  response cache with in-memory and Redis backends, a curated model registry, cost and token
  accounting, and an opt-in NDJSON diagnostic dump. Provider clients for Anthropic, OpenAI, Azure,
  Bedrock, Vertex, and OpenAI-compatible endpoints.
- **`strata_forge.prompts`** — prompts as sandboxed Jinja2 templates split into a stable prefix and
  a dynamic suffix, rendered into `strata_forge.llm` messages alongside a provider-agnostic
  cache-hint record, with a versioned registry over in-memory and Langfuse stores.
- **`strata_forge.tracing`** — Langfuse observability: the LiteLLM callback, a `@traced` decorator,
  span context managers, and score and metric helpers, all degrading to silent no-ops when Langfuse
  is unconfigured.
- **`strata_forge.datasets`** — frozen dataset shapes with content-hash ids and versions, an async
  store interface with in-memory and Langfuse backends, a bidirectional Hugging Face Datasets
  bridge, dataset diffing, and LLM-driven synthetic-data helpers (`self_instruct`, `distill`).
- **`strata_forge.evals`** — an experiment runner over models, prompts, datasets, and graders;
  deterministic graders (exact match, regex, JSON field and structure) and LLM-driven ones (judge,
  pairwise, semantic similarity); metrics, Markdown and HTML reports, parameter sweeps, Langfuse
  trace replay, and a Wilson-bounded CI regression gate.
- **`strata_forge.agents`** — an agent runtime composing an `LLMClient`, a system prompt, and
  `strata_forge.llm` tools; built-in calculator, URL-fetch, sandboxed filesystem-read, and
  backend-agnostic web-search tools; conversation and vector-backed episodic memory; hand-off and
  critic-refiner multi-agent patterns.
- **`strata_forge.rag`** — Protocols for embedders, chunkers, retrievers, vector stores, and
  rerankers, with concrete implementations: a LiteLLM embedder, a recursive chunker, dense, BM25,
  and hybrid RRF retrieval, in-memory and Qdrant stores, Cohere and cross-encoder rerankers, and a
  pipeline that composes ingestion, retrieval, reranking, and prompt augmentation.
- **`strata_forge.storage`** — an async `fsspec` gateway spanning local disk, S3, GCS, and Azure
  Blob, plus a Hugging Face Hub client for model and dataset transfer.
- **`strata_forge.compute`** — remote work described as frozen Pydantic data and submitted through a
  uniform async backend Protocol, with local-subprocess, SSH, and SkyPilot backends, YAML task
  round-tripping, self-hosted serving task builders for vLLM, TGI, and SGLang, a readiness probe,
  and concurrency-bounded batch inference.
- **`strata_forge.training`** — typed, frozen configs rendered into TRL's SFT and preference
  trainers (DPO, ORPO, KTO, GRPO), optional LoRA and QLoRA adapters, chat-template formatting,
  sequence packing, and JSONL progress streaming. Heavy ML imports are deferred until `train()`.
- **`strata_forge.pipelines`** — a runnable batch-inference entrypoint that reads its run
  specification from an environment variable, launches a local inference server, fans prompts out
  through the batch runner, and writes results to the Hub or to local disk.
- **`strata_forge.sync`** — `asyncio.run` facades over `LLMClient.complete`, `complete_structured`,
  `stream`, and `run_tool_loop` for CLI and notebook use.
- **`strata_forge.cli`** — the `strata-forge` console script: `doctor`, `chat`, `prompts`,
  `datasets`, `eval`, `experiments`, `compute`, `train`, and `serve`.
- **Packaging** — Apache-2.0, Python 3.14+, `py.typed` for downstream type checkers, and optional
  extras (`redis`, `multimodal`, `inspect`, `langfuse`, `hf`, `evals`, `rag`, `compute`, `serving`,
  `bedrock`, `storage`, `finetuning`, `all`) so the base install stays light. Publishing runs
  through PyPI Trusted Publishing, with the built archives checked for excluded files before upload.

[Unreleased]: https://github.com/GemovicNemanja/strata-forge/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/GemovicNemanja/strata-forge/releases/tag/v0.0.1
