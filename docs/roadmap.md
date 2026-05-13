# Roadmap

Current status of implementation phases. Updated as work lands. Design rationale lives in `docs/architecture/overview.md` and the ADRs under `docs/architecture/adr/`.

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations — packaging, DX, `core/`, `config/`, CLI skeleton + `doctor`, testing infra | in progress |
| 1 | LLM abstraction — providers, registry, routing, fallback, tools, structured output, multimodal, streaming, cache, diagnostic | pending |
| 2 | Langfuse integration — prompts, tracing, datasets, evals | pending |
| 3 | Agents — PydanticAI builder, memory, multi-agent patterns | pending |
| 4 | RAG — embed, vector, chunk, retrieve, rerank | pending |
| 5 | Remote compute — SkyPilot, SSH backend, inference, training | pending |
| 6 | Storage — fsspec, Hugging Face Hub | pending |
| 7 | CLI — remaining subcommands | pending |
| 8 | DX maturity — notebooks, Docker hardening, examples polish | pending |
| 9 | Testing maturity — cassette refresh CI, eval gate, security audit | pending |
