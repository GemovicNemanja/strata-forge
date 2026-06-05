# CLAUDE.md — agent rules for AI Forge

This file is the single source of truth for cross-cutting conventions that every contributor (human or agent — Claude Code, Cursor, Codex, Copilot, …) follows in this codebase. Tools that read `AGENTS.md` get the same content via the symlink at the repo root.

Module-specific rules live in `src/forge/<module>/CLAUDE.md`. If a module rule conflicts with this file, **fix the conflict** before merging — the module file is authoritative for itself, this file is authoritative for cross-cutting concerns.

Glob-scoped reinforcement of specific patterns lives in `.cursor/rules/*.mdc` for Cursor's rule engine, mirroring the relevant sections of this file.

---

## 1. North star and non-goals

**Goal.** AI Forge is a typed, async-first, batteries-included baseline that supports common AI/LLM experimentation workflows out of the box (inference, evaluation, fine-tuning, RAG, remote compute) and stays modular enough to extend into bespoke research without ripping apart its foundations.

**Non-goals.**
- Not a SaaS or hosted product — purely a library.
- Not a one-off research script — generality over single-use convenience.
- Not a wrapper around any one provider — vendor-neutral via LiteLLM at the transport seam.
- Not opinionated about prompt design or grader curation — Forge ships primitives, not opinions.
- Not aimed at production-grade inference serving (that's vLLM/TGI/SGLang's job; we orchestrate them via `forge.compute`).

---

## 2. Project map

| Module | Purpose | Module rules |
|---|---|---|
| `forge.core` | Cross-cutting utilities: errors (`ForgeError` hierarchy), retry, structlog logging with `trace_id` propagation, `BudgetContext`, reproducibility helpers, UUIDv7 ids, shared types. Strictly upstream — does not import from any other `forge.*` module. | [`src/forge/core/CLAUDE.md`](src/forge/core/CLAUDE.md) |
| `forge.config` | Pydantic Settings root with sub-models per concern; YAML profile overlays (`FORGE_PROFILE`); `.env` loading. The single configuration entry point — never read env vars directly elsewhere. | [`src/forge/config/CLAUDE.md`](src/forge/config/CLAUDE.md) |
| `forge.llm` | Provider-abstracted async LLM client over LiteLLM. Owns the typed layer: Pydantic messages/responses, structured output, tools (`Tool`, `@tool`, `run_tool_loop`), multimodal, streaming, two-axis fallback, provider-agnostic cache, model registry. Reference: [`docs/modules/llm.md`](docs/modules/llm.md). | [`src/forge/llm/CLAUDE.md`](src/forge/llm/CLAUDE.md) |
| `forge.prompts` | Jinja2 templating with safe filters + Langfuse-backed prompt registry. Templates model the stable-prefix / dynamic-suffix split so provider prompt caching just works. Reference: [`docs/modules/prompts.md`](docs/modules/prompts.md). | [`src/forge/prompts/CLAUDE.md`](src/forge/prompts/CLAUDE.md) |
| `forge.tracing` | Langfuse observability. Auto-tracing via LiteLLM callback, `@traced` decorator, span context managers, score/metric helpers. Reference: [`docs/modules/tracing.md`](docs/modules/tracing.md). | [`src/forge/tracing/CLAUDE.md`](src/forge/tracing/CLAUDE.md) |
| `forge.datasets` | Dataset CRUD + versioning; Langfuse ↔ Hugging Face Datasets bridge; synthetic-data primitives. | [`src/forge/datasets/CLAUDE.md`](src/forge/datasets/CLAUDE.md) |
| `forge.evals` | Experiment runner (model × prompt × dataset × graders); graders, metrics, reports, parameter sweeps; trace replay; CI eval-regression gate. | [`src/forge/evals/CLAUDE.md`](src/forge/evals/CLAUDE.md) |
| `forge.agents` | PydanticAI agent builder + built-in tools + memory + multi-agent patterns. **Reuses** `Tool`, `@tool`, message types, `run_tool_loop` from `forge.llm` — does NOT re-implement tool plumbing. | [`src/forge/agents/CLAUDE.md`](src/forge/agents/CLAUDE.md) |
| `forge.rag` | Embedders (via `forge.llm`), Qdrant store, chunkers, retrieval (dense / BM25 / hybrid RRF), rerankers, composable pipeline. | [`src/forge/rag/CLAUDE.md`](src/forge/rag/CLAUDE.md) |
| `forge.storage` | `fsspec` gateway (local, S3, GCS, Azure Blob, HF Hub); HF Hub model push/pull; dataset handling. | [`src/forge/storage/CLAUDE.md`](src/forge/storage/CLAUDE.md) |
| `forge.compute` | Remote compute orchestration: SkyPilot via `sky.api.sdk`, raw `asyncssh` SSH backend, YAML task templates, async batch inference. | [`src/forge/compute/CLAUDE.md`](src/forge/compute/CLAUDE.md) |
| `forge.training` | Fine-tuning: SFT, DPO/ORPO/KTO/GRPO, PEFT (LoRA/QLoRA), chat-template formatting, sequence packing. | [`src/forge/training/CLAUDE.md`](src/forge/training/CLAUDE.md) |
| `forge.cli` | Typer entry points: `chat`, `eval`, `experiments`, `prompts`, `datasets`, `train`, `serve`, `compute`, `doctor`. | [`src/forge/cli/CLAUDE.md`](src/forge/cli/CLAUDE.md) |

Per-module reference docs live under `docs/modules/<name>.md`. The phase roadmap lives in [`docs/roadmap.md`](docs/roadmap.md); architectural rationale lives in `docs/architecture/`.

---

## 3. Architecture principles

These apply everywhere in `src/forge/`. Code that violates them is wrong by default; if a violation is genuinely necessary, write an ADR before merging.

### 3.1 Async-first public API
- Every public function on every module is `async`.
- Sync wrappers live **only** in `forge.sync` and exist for CLI / notebook ergonomics. They wrap async functions via `asyncio.run`.
- Internal calls are async end-to-end. Do not mix `requests` / sync `httpx` with async code paths.
- HTTP I/O uses `httpx.AsyncClient` (often indirectly through LiteLLM).

### 3.2 Strict typing
- pyright runs in `strict` mode on `src/forge/`. New code must be strict-clean.
- No `Any` without a single-line `# pyright: ignore[reportXxx]` justification.
- Data types: Pydantic v2 models. Behavior types: `Protocol`. Narrow boundaries: `TypedDict`.
- `from __future__ import annotations` at the top of modules using generics — keeps annotations lazy on Python 3.14.

### 3.3 Exception-based errors
- Every Forge-raised error inherits from `forge.core.errors.ForgeError`.
- Provider exceptions are normalized at the seam (`forge.llm.errors.map_litellm_exception`) into `ProviderError` subclasses.
- Never `except:`. Never swallow exceptions. Decorators (`@retry`) drive control flow; predicates select on `ProviderError` subclasses.
- Validation errors raise `ValidationError` (or a subclass), never silently coerce.

### 3.4 Module independence + lazy heavy imports
- `import forge.<module>` MUST NOT crash when an optional extra is uninstalled.
- Heavy deps (`torch`, `transformers`, `trl`, `peft`, `skypilot`, `asyncssh`, `qdrant-client`, `vllm`, `pillow`, `redis`, `inspect-ai`) live behind PyPI extras — `[compute]`, `[finetuning]`, `[serving]`, `[multimodal]`, `[redis]`, `[inspect]`.
- Heavy deps are imported **inside the function that uses them**, never at module load. The function raises `ImportError` with an installation hint if the extra is missing.

### 3.5 Composability + escape hatches
- Modules expose a small core API plus an escape hatch for advanced cases (e.g. `LLMClient.complete(..., provider_extras={"anthropic": {...}})` forwards verbatim to LiteLLM).
- Never block users from accessing the underlying transport. The escape hatch is a feature, not a leak.

### 3.6 Observability built in
- Every LLM call is traced (when Langfuse is configured) and dumpable to NDJSON (env-gated via `FORGE_DIAGNOSTIC=1`).
- Use the structlog logger from `forge.core.logging`; `trace_id` propagates via contextvars across `await` boundaries — never pass it manually.
- Do not log full prompts at `INFO`; use `DEBUG`. Never log raw API keys, AWS signatures, or session tokens.

### 3.7 Cost awareness
- LLM-touching code paths must respect any active `BudgetContext`. Pre-call estimation + post-call true-up.
- Don't silently truncate prompts; raise `BudgetExceededError` instead.

### 3.8 Reproducibility
- Long-running operations capture `forge.core.repro.env_snapshot()` in their run metadata.
- Pseudo-randomness goes through `forge.core.repro.set_seed()` — never call `random.seed` directly.
- Dataset content is hashed via `content_hash()` for run provenance.

### 3.9 No vendor lock-in in higher-level modules
- `forge.evals`, `forge.agents`, `forge.rag` use `forge.llm` for all model calls. They never import provider SDKs directly.
- Vector store, prompt registry, dataset bridge — all behind interfaces that can be swapped.

---

## 4. Coding standards

Most of these are enforced by `ruff` and `pyright`. Local violations without justification fail pre-commit.

- **Ruff** is the lint + format authority. Single config in `pyproject.toml`. Line length 100. `target-version = "py314"`. Selected rule sets include E, F, W, I, N, UP, B, A, C4, PIE, SIM, RUF, ASYNC, S (bandit), PT (pytest), TID, TCH, PTH, ERA.
- **Pyright** strict on `src/forge`. Tests are typed but not strict (pragmatic). `Any` requires a justified ignore comment.
- **Naming:** `snake_case` for functions / variables / modules; `PascalCase` for classes; `SCREAMING_SNAKE` for constants. Pydantic models follow `XxxConfig`, `XxxRequest`, `XxxResponse`, `XxxError`.
- **Imports:** ruff isort. First-party is `forge`. Avoid `from forge.<other_module> import *` re-exports without explicit `__all__`.
- **Docstrings:** module-level summary mandatory. Public functions / classes: one-line summary; multi-line only when the *why* is non-obvious. No paragraph-long docstrings — long-form belongs in `docs/modules/<name>.md`.
- **Comments:** rare. Only when the *why* is non-obvious. Never describe *what* — names already do that. Never reference current tasks, PRs, or scaffolding metadata.
- **Emojis:** none in source.
- **`from __future__ import annotations`** at the top of modules that use generics in signatures.

---

## 5. Testing discipline

- Every new code path gets a test in the same PR.
- Prefer fast unit tests (no network) for most logic. Use VCR for provider-touching tests; live integration is env-gated and runs only nightly.
- Coverage thresholds enforced in CI:
  - `src/forge/llm/` ≥ 90 % line
  - `src/forge/core/`, `src/forge/config/` ≥ 85 % line
- No skipped tests committed without a tracking issue link in the skip reason.
- VCR cassettes scrub secrets in `before_record_request`. Recording without scrubbing = leaking credentials; rotate them.
- Snapshot tests (`syrupy`) for provider format mappings — a failing snapshot is an early-warning signal, not noise to suppress.
- Marker discipline: `live`, `redis`, `slow`, `integration` (declared in `pyproject.toml`; new markers go there).

---

## 6. Documentation discipline

- Every module has `docs/modules/<name>.md` kept in sync with the public API — update it in the same PR as any API change.
- Architectural decisions go into `docs/architecture/adr/NNNN-short-slug.md`. Numbered sequentially, immutable once merged — supersede via a new ADR that points back to the old one.
- ADR template: **Context → Decision → Consequences**. Keep under ~200 lines.
- Module READMEs describe what the module does/will do in product terms. **Phase numbers do NOT appear there** — the phase roadmap is the only place for those.
- User-facing changes update `README.md` and `docs/roadmap.md`.
- Cross-link liberally: module README ↔ docs/modules/ ↔ ADRs that motivated the design.

---

## 7. Self-maintenance protocol

The single most important rule. When you (the agent) notice that this file, a module-level `CLAUDE.md`, or a `.cursor/rules/*.mdc` file no longer matches the code (renamed module, changed API, deprecated pattern, new convention emerging across PRs):

1. **Propose an update to the rule file in the same PR as the code change.**
2. If a pattern in code isn't yet captured in any rule, **propose adding it.**
3. Never silently ignore a stale rule — either follow it or update it.
4. If a module rule conflicts with this root file, **fix the conflict** before merging.
5. ADRs are immutable; supersede via a new ADR, don't edit existing ones.

This protocol applies to memories under `~/.claude/projects/.../memory/` too — when a saved memory contradicts current code or rules, update or delete it.

Agents that silently ignore stale rules are worse than agents that don't read them at all.

---

## 8. Commit and PR conventions

### Conventional Commits — required

Every commit subject MUST follow the Conventional Commits format:

```
<type>(<optional-scope>): <imperative summary>
```

The subject is a single line, lowercase, imperative, under 70 characters. `<type>` is one of the values below — anything else is rejected at review.

| Type | Use when… | Example |
|---|---|---|
| `feat` | a user-visible feature is added | `feat(llm): add structured-output reprompt fallback` |
| `fix` | a bug in existing behavior is corrected | `fix(cache): handle Redis timeout without crashing` |
| `docs` | only documentation changes (READMEs, ADRs, CLAUDE.md, module guides, docstrings) | `docs(rag): document the BM25 retriever options` |
| `style` | whitespace, formatting, missing semicolons, no code change | `style: apply ruff format to recent edits` |
| `refactor` | code change that neither fixes a bug nor adds a feature | `refactor(llm): split client.py into client + tool_loop` |
| `perf` | a measurable performance improvement | `perf(tokens): cache tiktoken encoders across calls` |
| `test` | adding or correcting tests; no production code change | `test(fallback): cover content-filter short-circuit` |
| `build` | changes to packaging, dependencies, build backend | `build: bump pydantic to >=2.11` |
| `ci` | changes to CI configuration or workflows | `ci: add nightly cassette-refresh job` |
| `chore` | routine housekeeping: tooling, configs, releases, repo plumbing — anything not in the categories above | `chore: prune unused make targets` |
| `revert` | undo a previous commit; reference the original in the body | `revert: feat(llm) — add structured-output reprompt fallback` |

### Scope (optional)

Use a parenthesized scope when the change is local to one module — usually the module name (`core`, `config`, `llm`, `prompts`, `evals`, `agents`, `rag`, `storage`, `compute`, `training`, `cli`) or a focused sub-area inside it. Omit the scope only when the change spans the whole repo (root tooling, multi-module refactors, cross-cutting docs).

### Body and footer

- Body (optional): wrapped at ~72 chars, explains the *why* and any non-obvious *how*. Bullet points are fine.
- Reference issues / PRs in a trailer line (`Closes #123`, `Refs PR-456`).
- Breaking changes go in a `BREAKING CHANGE:` trailer or use the `!` marker (`feat(llm)!: replace LLMClient.complete signature`).

### Time-stable language

**Commit messages describe changes in time-stable terms.** They MUST NOT reference phase numbers, sub-phase numbers, step numbers, plan-file slugs, or any other scaffolding metadata — those exist only during construction and become meaningless afterward. Keep the construction process out of the long-term git log. The same rule applies to PR titles, branch names, and any other long-lived artifact.

### Other rules

- One commit per cohesive unit of work — never bundle unrelated changes for convenience.
- PR titles follow the same Conventional Commits format as the subject line. PR body has `## Summary`, `## Why`, `## Test plan`.
- Never `--no-verify`. Never `--force-push` to `main` or `dev`. Never `--amend` a commit that has been pushed somewhere shared.

### Branching and deployment

A three-tier flow — **every** change follows it; never commit directly to `dev` or `main`:

1. **Topic branch** off `dev`, named `<type>/<short-title>` (same `<type>` vocabulary as commits) — e.g. `feat/streaming-tool-loop`. One cohesive unit of work per branch.
2. **Merge into `dev`** (via PR) when done. `dev` is the persistent integration branch.
3. **Delete the topic branch once it's merged into `dev`** — local AND remote: `git branch -d <branch> && git push origin --delete <branch>`. **`dev` and `main` are the only persistent branches**; never leave a merged topic branch on the remote. (Sweep: `git branch -r --merged origin/dev --format='%(refname:short)' | grep -vE '^origin/(dev|main)$' | sed 's#^origin/##' | xargs -r git push origin --delete`.)
4. **Promote `dev` into `main`** once integrated — via a `dev` → `main` PR that a maintainer merges.

**`main` only ever receives `dev` — never a topic branch, never a direct push.** Enforced, not just convention: `main` is protected (PR required, direct pushes blocked, no admin bypass) and a required CI check (`.github/workflows/promotion-guard.yml`) **fails any PR into `main` whose source branch isn't `dev`**. Promotion is always a `dev` → `main` **PR** — agents may open and merge it themselves (via the GitHub MCP or `gh`); what's forbidden is a *direct push* to `main` and any non-`dev` source.

forge is a library, not a service, so it has no deploy of its own — but its branches feed the sibling **strata-server**'s deploys: the server's **staging** (`strata-server` `dev`) bundles forge **`dev`**, and the server's **production** (`strata-server` `main`) bundles forge **`main`**. So promote a forge change to `main` only once the server staging that consumes forge `dev` looks good, and keep forge `dev` green — it gates the whole staging chain.

---

## 9. How to add a new top-level module

1. Create `src/forge/<module>/` with `__init__.py` (module docstring), `README.md` (what + why), `CLAUDE.md` (module-specific rules).
2. Add a row to the **Project map** table in §2 of this file.
3. Add `docs/modules/<module>.md` reference doc.
4. If the module introduces new heavy deps, add an extra to `pyproject.toml` and gate the imports lazily inside the using function.
5. Add at least one unit test under `tests/unit/<module>/`.
6. If it touches provider APIs, add VCR cassettes under `tests/vcr/cassettes/<module>/`.
7. If the change is architecturally significant, write an ADR.
8. Update `docs/roadmap.md` if the new module materially shifts phase status.
9. Open a PR following §8.

---

## 10. Where to look when stuck

- This file → cross-cutting rules.
- `src/forge/<module>/CLAUDE.md` → module-specific rules.
- `docs/architecture/overview.md` → high-level architecture narrative.
- `docs/architecture/adr/` → "why does this look the way it does?"
- `docs/modules/<name>.md` → public API reference for a module.
- `docs/roadmap.md` → what's done, what's next, what's deferred.
- `pyproject.toml` → ground truth for deps, extras, ruff config, pyright config, pytest markers.
- `Makefile` → all the things you can run with one command.

When this file disagrees with code that's already merged, **the code wins** — and you update this file. When this file disagrees with in-flight work, **this file wins** — and you update the code.
