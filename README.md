# AI Forge

Typed, async-first baseline repository for AI/LLM experiments — inference, evaluation, fine-tuning, and remote compute orchestration across the major foundation-model providers (OpenAI, Anthropic, Google).

Built on **LiteLLM** for provider breadth and **Pydantic v2** for strict typing. Async-first public API with sync wrappers in `forge.sync` for CLI / notebook ergonomics. Heavy optional dependencies (training, serving, remote compute) live behind extras so the base install stays light.

> **Status:** Phase 0 scaffolding. Proprietary; not open source.

## Quickstart

```bash
uv sync                  # materialize the venv + lockfile
make doctor              # check env vars + reachability of configured services
make test                # unit tests
make check               # lint (ruff) + type-check (pyright strict)
```

## Documentation

- `docs/architecture/overview.md` — design overview and module map
- `docs/architecture/adr/` — Architecture Decision Records
- `docs/modules/<name>.md` — per-module API reference
- `docs/roadmap.md` — phase status
- `CLAUDE.md` — agent guidance (read via the `AGENTS.md` symlink by Cursor / Codex / Copilot, and directly by Claude Code)

## Layout

```
src/forge/
  core/         cross-cutting (errors, retry, logging, budget, repro)
  config/       Pydantic Settings + YAML overlays + .env
  llm/          provider abstraction, structured output, tools, fallback   (Phase 1)
  prompts/      Jinja2 + Langfuse-backed prompt registry                   (Phase 2.1)
  tracing/      Langfuse observability                                     (Phase 2.2)
  datasets/     dataset CRUD + Hugging Face bridge                         (Phase 2.3)
  evals/        experiment runner + graders + metrics                      (Phase 2.4)
  agents/       PydanticAI builder, memory, multi-agent                    (Phase 3)
  rag/          embed + vector + chunk + retrieve + rerank                 (Phase 4)
  storage/      fsspec gateway, Hugging Face Hub                           (Phase 6)
  compute/      SkyPilot + SSH backends + inference                        (Phase 5)
  training/     SFT / DPO / ORPO / KTO / GRPO with PEFT                    (Phase 5.3)
  cli/          Typer entry points                                         (Phase 7)
```

See `docs/roadmap.md` for current phase status.
