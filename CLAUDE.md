# CLAUDE.md — agent rules for strata-forge

This file is the single source of truth for cross-cutting conventions that every contributor (human
or agent — Claude Code, Cursor, Codex, Copilot, …) follows in this codebase. Tools that read
`AGENTS.md` get the same content via the symlink at the repo root.

Module-specific rules live in `src/strata_forge/<module>/CLAUDE.md`. If a module rule conflicts with
this file, **fix the conflict** before merging — the module file is authoritative for itself, this
file is authoritative for cross-cutting concerns.

Glob-scoped reinforcement of specific patterns lives in `.cursor/rules/*.mdc` for Cursor's rule
engine, mirroring the relevant sections of this file.

This is a public repository published to PyPI. Everything under `src/` — including docstrings and
module READMEs — is read by strangers who cannot see anything else. Write for them.

## Contents

1. [North star and non-goals](#1-north-star-and-non-goals)
2. [Project map](#2-project-map) · [2b. Repository map](#2b-repository-map)
3. [Architecture principles](#3-architecture-principles)
4. [Coding standards](#4-coding-standards)
5. [Testing discipline](#5-testing-discipline)
6. [Documentation discipline](#6-documentation-discipline)
7. [Self-maintenance protocol](#7-self-maintenance-protocol) ·
   [7b. Contributing as an outside agent](#7b-contributing-as-an-outside-agent)
8. [Commit and PR conventions](#8-commit-and-pr-conventions)
9. [How to add a new top-level module](#9-how-to-add-a-new-top-level-module) ·
   [9b. Publishing to PyPI](#9b-publishing-to-pypi)
10. [Where to look when stuck](#10-where-to-look-when-stuck)

---

## 1. North star and non-goals

**Goal.** strata-forge is a typed, async-first, batteries-included baseline that supports common
AI/LLM experimentation workflows out of the box (inference, evaluation, fine-tuning, RAG, remote
compute) and stays modular enough to extend into bespoke research without ripping apart its
foundations.

**Non-goals.**
- Not a SaaS or hosted product — purely a library.
- Not a one-off research script — generality over single-use convenience.
- Not a wrapper around any one provider — vendor-neutral via LiteLLM at the transport seam.
- Not opinionated about prompt design or grader curation — the library ships primitives, not
  opinions.
- Not aimed at production-grade inference serving (that's vLLM/TGI/SGLang's job; we orchestrate them
  via `strata_forge.compute`).

The library is feature-complete against its original design; there is no roadmap document and none
is to be written (§6). Direction now comes from the applications that consume it. Describe maturity
by saying what the code does today.

---

## 2. Project map

| Module | Purpose | Module rules | Reference |
|---|---|---|---|
| `strata_forge.core` | Cross-cutting primitives every other module builds on: the `ForgeError` hierarchy, a tenacity-backed `@retry`, structlog configuration that injects `correlation_id` from a contextvar, `BudgetContext` cost/token ceilings, reproducibility helpers (`set_seed`, `content_hash`, `env_snapshot`), UUIDv7 ids, shared type aliases. Strictly upstream — imports nothing from `strata_forge.*`. | [`core/CLAUDE.md`](src/strata_forge/core/CLAUDE.md) | [`docs/modules/core.md`](docs/modules/core.md) |
| `strata_forge.config` | Pydantic Settings root with a `BaseSettings` sub-model per concern (`langfuse`, `redis`, `qdrant`, `huggingface`, `storage`, `logging`, `diagnostic`) plus `providers`, a cached `get_settings()`, and `.env` loading. YAML overlay helpers (`load_overlay`, `deep_merge`, `overlay_path_for_profile`) are standalone primitives — nothing layers them into `Settings`, and `FORGE_PROFILE` only populates `Settings.profile`. The single configuration entry point — never read env vars directly elsewhere. | [`config/CLAUDE.md`](src/strata_forge/config/CLAUDE.md) | [`docs/modules/config.md`](docs/modules/config.md) |
| `strata_forge.llm` | Provider-abstracted async LLM client over LiteLLM. Owns the typed layer: Pydantic messages/responses, structured output, tools (`Tool`, `@tool`, `run_tool_loop`, streaming tool loop), multimodal image input, streaming accumulators, two-axis fallback, provider-agnostic cache, curated model registry, cost/token accounting, NDJSON diagnostic dump. | [`llm/CLAUDE.md`](src/strata_forge/llm/CLAUDE.md) | [`docs/modules/llm.md`](docs/modules/llm.md) |
| `strata_forge.sync` | Sync facades over four `LLMClient` methods — `complete`, `complete_structured`, `stream`, `run_tool_loop` — via `asyncio.run`, for script and notebook ergonomics. Nothing outside `strata_forge.llm` is wrapped, and the CLI does not use it (see §3.1). | — | [`docs/modules/sync.md`](docs/modules/sync.md) |
| `strata_forge.prompts` | Sandboxed Jinja2 templates split into a stable prefix and a dynamic suffix, rendered into `strata_forge.llm` messages plus a `CacheHints` record the caller can act on, with a versioned registry over in-memory and Langfuse stores. The split is what makes provider prompt caching reachable; the LLM client does not consume the hints itself. | [`prompts/CLAUDE.md`](src/strata_forge/prompts/CLAUDE.md) | [`docs/modules/prompts.md`](docs/modules/prompts.md) |
| `strata_forge.tracing` | Langfuse observability applied from above the stack: LiteLLM callback install, `@traced` decorator, span context managers, score/metric helpers. A silent no-op whenever Langfuse is unconfigured. | [`tracing/CLAUDE.md`](src/strata_forge/tracing/CLAUDE.md) | [`docs/modules/tracing.md`](docs/modules/tracing.md) |
| `strata_forge.datasets` | Frozen dataset shapes with content-hash ids and versions, diffing, an async store interface with in-memory and Langfuse backends, a bidirectional Hugging Face Datasets bridge, and LLM-driven synthetic data (`self_instruct`, `distill`). | [`datasets/CLAUDE.md`](src/strata_forge/datasets/CLAUDE.md) | [`docs/modules/datasets.md`](docs/modules/datasets.md) |
| `strata_forge.evals` | Experiment runner (models × prompts × dataset × graders); deterministic and LLM-driven graders, metrics, Markdown/HTML reports, parameter sweeps, Langfuse trace replay, Wilson-bounded CI eval-regression gate. | [`evals/CLAUDE.md`](src/strata_forge/evals/CLAUDE.md) | [`docs/modules/evals.md`](docs/modules/evals.md) |
| `strata_forge.agents` | Agent runtime composing an `LLMClient`, a system prompt and `strata_forge.llm` tools, plus built-in tools, conversation/episodic memory, and hand-off / critic-refiner patterns. **Reuses** `Tool`, `@tool`, message types and the tool loop from `strata_forge.llm` — does NOT re-implement tool plumbing, and is deliberately not built on PydanticAI ([ADR 0011](docs/architecture/adr/0011-agents-thin-wrapper-over-forge-llm.md)). | [`agents/CLAUDE.md`](src/strata_forge/agents/CLAUDE.md) | [`docs/modules/agents.md`](docs/modules/agents.md) |
| `strata_forge.rag` | Runtime-checkable Protocols (Embedder, Chunker, Retriever, VectorStore, Reranker) plus concrete implementations: a LiteLLM embedder, recursive chunker, in-memory and Qdrant stores, dense / BM25 / hybrid-RRF retrieval, Cohere and cross-encoder rerankers, composable `RAGPipeline`. The embedder calls `litellm.aembedding` directly rather than routing through `strata_forge.llm` ([ADR 0012](docs/architecture/adr/0012-rag-protocols-and-vector-store-relocation.md)). | [`rag/CLAUDE.md`](src/strata_forge/rag/CLAUDE.md) | [`docs/modules/rag.md`](docs/modules/rag.md) |
| `strata_forge.storage` | Async `fsspec` gateway (local, S3, GCS, Azure Blob) and a Hugging Face Hub client for model / dataset push and pull. | [`storage/CLAUDE.md`](src/strata_forge/storage/CLAUDE.md) | [`docs/modules/storage.md`](docs/modules/storage.md) |
| `strata_forge.compute` | Remote compute orchestration: frozen `Task` / `ResourceSpec` / `Job` shapes submitted through one async `Backend` protocol backed by a local subprocess, raw `asyncssh` SSH, or SkyPilot via `sky.api.sdk`; YAML task round-trip, serving-endpoint helpers for vLLM / TGI / SGLang, concurrency-bounded batch inference. | [`compute/CLAUDE.md`](src/strata_forge/compute/CLAUDE.md) | [`docs/modules/compute.md`](docs/modules/compute.md) |
| `strata_forge.training` | Typed, frozen configs rendered into TRL trainers: SFT, preference methods (DPO/ORPO/KTO/GRPO), PEFT (LoRA/QLoRA), chat-template formatting, sequence packing, JSONL progress streaming. Every heavy ML import is deferred to `train()`. | [`training/CLAUDE.md`](src/strata_forge/training/CLAUDE.md) | [`docs/modules/training.md`](docs/modules/training.md) |
| `strata_forge.pipelines` | Runnable entrypoints launched *on* a compute target rather than imported in-process: `inference_runner` reads an inert JSON run spec from the environment, composes the serving / batch / storage primitives, and appends progress events to a file the launcher tails. | [`pipelines/CLAUDE.md`](src/strata_forge/pipelines/CLAUDE.md) | [`docs/modules/pipelines.md`](docs/modules/pipelines.md) |
| `strata_forge.cli` | Typer application installed as the `strata-forge` console script: `doctor`, `chat`, `prompts`, `datasets`, `eval`, `experiments`, `compute`, `train`, `serve`. | [`cli/CLAUDE.md`](src/strata_forge/cli/CLAUDE.md) | [`docs/modules/cli.md`](docs/modules/cli.md) |

Per-module reference docs live under `docs/modules/<name>.md`; architectural rationale lives in
[`docs/architecture/`](docs/architecture/) and its ADRs.

---

## 2b. Repository map

Where things live and when you touch them. Paths are repo-relative.

| Path | What lives there | When you go there |
|---|---|---|
| `src/strata_forge/` | The library. The only tree published to PyPI. | Any behaviour change. Obeys §3 and §4 without exception. |
| `src/strata_forge/<module>/CLAUDE.md` | Module-specific rules; excluded from the wheel and sdist. | Read before editing a module; update in the same PR when its API or layout changes (§7). |
| `src/strata_forge/<module>/README.md` | Product-level description of the module. **Ships inside the wheel.** | Keep it present-tense and true; no status hedges, no phase numbers, no dangling links. |
| `docs/` | The published documentation set, rendered by MkDocs + Material to GitHub Pages. `docs/` is the `docs_dir`. | Any user-visible change. Link rules in §6. |
| `docs/modules/<name>.md` | Canonical public-API reference, one per module. | Same PR as any public API change (§6). |
| `docs/architecture/overview.md` | Narrative architecture: layering, routing, the seams and why they sit where they do. | Orientation before a cross-module change. |
| `docs/architecture/module-boundaries.md` | The import rules — what each module may and may not import, and what it owns. | Before adding an intra-package import. |
| `docs/architecture/adr/NNNN-*.md` | Decision records: Context → Decision → Consequences. Immutable once merged. | Read before changing a seam; supersede with a new ADR, never edit an old one. |
| `docs/recipes/` | Cross-module how-to walkthroughs (a workflow that needs more than one module). | Add one when a real workflow spans modules and no single module page can host it. |
| `docs/agent-guide.md` | The long-form agent playbook: repo orientation, task recipes, the verification loop. | Read alongside this file on your first task here. |
| `examples/NN_*.py` | Self-contained runnable scripts, environment-gated, one per documented capability. | Add one when you add a public capability; `tests/e2e` imports every file here. |
| `notebooks/*.py` | [marimo](https://marimo.io/) notebooks — notebook state stored as reviewable Python. | Interactive exploration templates; `tests/e2e` imports these too. |
| `scripts/` | Operational helpers run by hand or by CI — `run_eval_gate.py` is the eval-regression gate. | Rare. Prefer a CLI subcommand over a new script. |
| `tests/unit/<module>/` | Fast, offline, no network. The default home for a new test. | Every new code path (§5). |
| `tests/integration/` | Cross-module tests. The `test_cross_module.py` files need no services; the `integration`-marked ones need the local Docker stack. | Behaviour that only appears when two modules meet. |
| `tests/e2e/` | Import smoke over `examples/` and `notebooks/` — catches drift between them and the public API. | Runs automatically; fix it rather than skipping it when it fires. |
| `tests/vcr/` | Provider-touching tests replayed from cassettes in `tests/vcr/cassettes/<provider>/`. | Provider request/response shape changes. Secrets are scrubbed in `before_record_request` (§5). |
| `docker/compose.yaml` | Local dev services (Postgres, Qdrant, Redis) behind `make stack-up`. | Integration work that needs a real backing service. |
| `docker/Dockerfile` | Container image for the library and its CLI. | Packaging changes. |
| `.github/workflows/` | `ci.yml` (lint, type, unit tests + coverage, cassette replay), `nightly.yml`, `promotion-guard.yml` (only `dev` may target `main`), `release.yml` (tag-triggered PyPI publish). | Changing what CI enforces. |
| `.cursor/rules/*.mdc` | Glob-scoped restatements of these rules for Cursor's rule engine. | Keep in sync whenever the corresponding section here changes (§7). |
| `pyproject.toml` | Ground truth for dependencies, extras, ruff, pyright, pytest markers, coverage, and the build allow-list. | Deps, tooling config, packaging (§9b). |
| `Makefile` | Every routine command: `check`, `test`, `test-all`, `integration`, `vcr-replay`, `eval-gate`, `docs-serve`, `docs-build`, `stack-up`. | First stop for "how do I run this?". |
| `README.md` | The landing page: what it is, install, quickstart, module table. | User-visible capability changes. |
| `CONTRIBUTING.md` | The human-facing contribution guide — environment setup, the local check loop, PR expectations. | Before your first PR. |
| `CHANGELOG.md` | Release-worthy changes, newest first. | Any user-visible change (§6). |
| `SECURITY.md`, `CODE_OF_CONDUCT.md` | Reporting policy and community standards. Both route to `gemovic@strataml.com`. | Never file a vulnerability as a public issue. |
| `mkdocs.yml` | Docs-site config: nav, theme, `docs_dir`. | Adding or renaming a page under `docs/`. |
| `.env.example` | The documented surface of every environment variable the library reads. | Adding or renaming a setting (§6). |
| `LICENSE`, `NOTICE` | Apache-2.0 text and attribution. | Don't. |
| `AGENTS.md` | Symlink to this file. | Never edit or replace the symlink. |

---

## 3. Architecture principles

These apply everywhere in `src/strata_forge/`. Code that violates them is wrong by default; if a
violation is genuinely necessary, write an ADR before merging.

### 3.1 Async-first public API
- Every public function that performs I/O is `async`. Pure helpers stay synchronous — `get_settings`,
  `render`, `content_hash`, `retry`, registry lookups, config construction. Don't wrap a pure
  function in `async` for symmetry.
- Deliberate exceptions are documented at the seam, not silently taken. Today the only one is
  `SFTRunner.train` / `PreferenceRunner.train`: TRL's training loop is synchronous and blocking, so
  the runner is too. A new exception needs the same treatment in the module's docs and rules file.
- Sync wrappers live **only** in `strata_forge.sync`, wrapping four `LLMClient` methods via
  `asyncio.run` for script and notebook ergonomics. The CLI does **not** go through them — it bridges
  with its own `run_async` helper in `cli/helpers.py`.
- Internal calls are async end-to-end. Do not mix `requests` / sync `httpx` with async code paths.
- HTTP I/O uses `httpx.AsyncClient` (often indirectly through LiteLLM).

### 3.2 Strict typing
- pyright runs in `strict` mode on `src/strata_forge/`. New code must be strict-clean.
- No `Any` without a single-line `# pyright: ignore[reportXxx]` justification.
- Data types: Pydantic v2 models. Behavior types: `Protocol`. Narrow boundaries: `TypedDict`.
- `from __future__ import annotations` at the top of modules using generics — keeps annotations lazy
  on Python 3.14.

### 3.3 Exception-based errors
- Library-level runtime failures — the ones a caller might reasonably catch — raise a
  `strata_forge.core.errors.ForgeError` subclass: provider faults, config faults, budget ceilings,
  registry misses, cache faults, validation. A new failure mode of that kind gets a `ForgeError`
  subclass, not a bare builtin.
- Programmer error at the call boundary (a bad argument, an impossible combination of kwargs) raises
  the builtin `ValueError` / `TypeError`. That is the standing convention across the modules; don't
  convert those to `ForgeError` retroactively without an ADR.
- Provider exceptions are normalized at the seam (`strata_forge.llm.errors.map_litellm_exception`)
  into `ProviderError` subclasses.
- Never `except:`. Never swallow exceptions — with one sanctioned exception: `strata_forge.tracing`
  swallows its own failures by design, because telemetry must never break a caller's hot path
  ([ADR 0008](docs/architecture/adr/0008-tracing-as-cross-cutting.md)).
- Decorators (`@retry`) drive control flow; predicates select on `ProviderError` subclasses.
- Validation errors raise `ValidationError` (or a subclass), never silently coerce.

### 3.4 Module independence + lazy heavy imports
- `import strata_forge.<module>` MUST NOT crash when an optional extra is uninstalled.
- Heavy deps (`torch`, `transformers`, `trl`, `peft`, `datasets`, `skypilot`, `asyncssh`,
  `qdrant-client`, `cohere`, `sentence-transformers`, `fsspec` + cloud filesystems,
  `huggingface_hub`, `langfuse`, `redis`, `pillow`, `nltk`, `boto3`) live behind PyPI extras:
  `[compute]`, `[finetuning]`, `[serving]`, `[storage]`, `[hf]`, `[langfuse]`, `[evals]`, `[rag]`,
  `[multimodal]`, `[redis]`, `[bedrock]`, and the aggregate `[all]`.
- Heavy deps are imported **inside the function that uses them**, never at module load. The function
  raises `ImportError` with an installation hint if the extra is missing.
- The hint names the real distribution: `pip install 'strata-forge[<extra>]'`. `ai-forge` is a
  different project and must never appear in a hint, a doc, or an example.

### 3.5 Composability + escape hatches
- Modules expose a small core API plus an escape hatch for advanced cases (e.g.
  `LLMClient.complete(..., provider_extras={"anthropic": {...}})` forwards verbatim to LiteLLM).
- Never block users from accessing the underlying transport. The escape hatch is a feature, not a
  leak.

### 3.6 Observability built in
- Every LLM call is traced (when Langfuse is configured) and dumpable to NDJSON, env-gated via
  `FORGE_DIAGNOSTIC_ENABLED=1` with the destination in `FORGE_DIAGNOSTIC_PATH`.
- Use the structlog logger from `strata_forge.core.logging`; `correlation_id` propagates via
  contextvars across `await` boundaries — never pass it manually.
- Do not log full prompts at `INFO`; use `DEBUG`. Never log raw API keys, AWS signatures, or session
  tokens.

### 3.7 Cost awareness
- LLM-touching code paths must respect any active `BudgetContext`.
- Accounting is post-call: `LLMClient` consumes against the budget once a call returns, so a ceiling
  stops the *next* call rather than the one in flight. Don't document it as pre-flight, and don't
  assume a budget can cancel work already dispatched.
- Don't silently truncate prompts; raise `BudgetExceededError` instead.

### 3.8 Reproducibility
- Long-running operations capture `strata_forge.core.repro.env_snapshot()` in their run metadata —
  interpreter, platform, and the versions of the tracked package set.
- Pseudo-randomness goes through `strata_forge.core.repro.set_seed()` — never call `random.seed`
  directly.
- Dataset content is hashed via `content_hash()` for run provenance.

### 3.9 No vendor lock-in in higher-level modules
- `strata_forge.evals` and `strata_forge.agents` route every model call through `strata_forge.llm`.
  They never import a model-provider SDK.
- `strata_forge.rag` is the one sanctioned exception, at two narrow seams: `LiteLLMEmbedder` calls
  `litellm.aembedding` directly ([ADR 0012](docs/architecture/adr/0012-rag-protocols-and-vector-store-relocation.md)),
  and the Qdrant / Cohere adapters lazily import their own SDKs because they are storage and ranking
  services, not model providers. Nothing else outside `strata_forge.llm` imports a provider SDK.
- Vector store, prompt registry, dataset bridge — all behind interfaces that can be swapped.

---

## 4. Coding standards

Most of these are enforced by `ruff` and `pyright`. Local violations without justification fail
pre-commit.

- **Ruff** is the lint + format authority. Single config in `pyproject.toml`. Line length 100.
  `target-version = "py314"`. Selected rule sets include E, F, W, I, N, UP, B, A, C4, PIE, SIM, RUF,
  ASYNC, S (bandit), PT (pytest), TID, TCH, PTH, ERA.
- **Pyright** strict on `src/strata_forge`. Tests are typed but not strict (pragmatic). `Any` requires
  a justified ignore comment.
- **Naming:** `snake_case` for functions / variables / modules; `PascalCase` for classes;
  `SCREAMING_SNAKE` for constants. Pydantic models follow `XxxConfig`, `XxxRequest`, `XxxResponse`,
  `XxxError`.
- **Imports:** ruff isort. First-party is `strata_forge`. Avoid
  `from strata_forge.<other_module> import *` re-exports without explicit `__all__`.
- **Docstrings:** module-level summary mandatory. Public functions / classes: one-line summary;
  multi-line only when the *why* is non-obvious. No paragraph-long docstrings — long-form belongs in
  `docs/modules/<name>.md`. Docstrings ship to PyPI, so write them for a reader who has only the
  installed package: no phase numbers, no "lands later", no internal infrastructure (§9b).
- **Comments:** rare. Only when the *why* is non-obvious. Never describe *what* — names already do
  that. Never reference current tasks, PRs, or scaffolding metadata.
- **Emojis:** none anywhere in the repository — not in source, docs, headings, tables, or commit
  messages. The badges at the top of the root `README.md` are the only exception.
- **`from __future__ import annotations`** at the top of modules that use generics in signatures.
- **Every symbol you write into a doc, example, or error message must exist.** Verify against
  `src/strata_forge` before writing an import or a call. A snippet that doesn't run is worse than no
  snippet.

---

## 5. Testing discipline

- Every new code path gets a test in the same PR.
- Prefer fast unit tests (no network) for most logic. Use VCR for provider-touching tests; live
  integration is env-gated and runs only nightly.
- Coverage: a single package-wide floor of 85 % line coverage (`fail_under` in `pyproject.toml`),
  measured by CI over `tests/unit`. `llm/`, `core/` and `config/` are held to a higher bar by
  convention — treat a drop there as a regression even though no gate enforces it per-module.
- `make test` runs the unit suite; `make test-all` runs everything, which is what catches
  `tests/e2e` and the offline cross-module tests. Run it before a PR that touches the public API.
- No skipped tests committed without a tracking issue link in the skip reason.
- VCR cassettes scrub secrets in `before_record_request`. Recording without scrubbing = leaking
  credentials; rotate them.
- Snapshot tests (`syrupy`) for provider format mappings — a failing snapshot is an early-warning
  signal, not noise to suppress.
- Marker discipline: `live`, `redis`, `slow`, `integration` (declared in `pyproject.toml`; new markers
  go there).

---

## 6. Documentation discipline

- Every module has `docs/modules/<name>.md` kept in sync with the public API — including `core`,
  `config`, `sync` and `pipelines`. Update it in the same PR as any API change, and add the page to
  `mkdocs.yml`'s nav when you create one.
- **The docs site is real.** `docs/` is built by MkDocs + Material and published to GitHub Pages
  (`make docs-serve` locally, `make docs-build` for the strict build). MkDocs cannot resolve paths
  above its `docs_dir`, so: links between files inside `docs/` are **relative `.md` links**, and links
  to anything outside `docs/` (`examples/`, `src/`, root `CONTRIBUTING.md`) are **absolute**
  `https://github.com/GemovicNemanja/strata-forge/blob/main/...` URLs. Both forms then work on GitHub
  and on the site.
- `docs/recipes/` holds cross-module walkthroughs. When a workflow spans modules and no single module
  page can host it, that is where it goes.
- Architectural decisions go into `docs/architecture/adr/NNNN-short-slug.md`. Numbered sequentially,
  immutable once merged — supersede via a new ADR that points back to the old one.
- ADR template: **Context → Decision → Consequences**. Keep under ~200 lines.
- Module READMEs describe what the module does in product terms, in the present tense. They ship
  inside the wheel, so every link in them must resolve for someone who only installed the package.
- **No phase numbers, no roadmap, anywhere.** Phase / sub-phase / step numbers, plan-file slugs and
  build-order narration are construction scaffolding: they are forbidden in source, docstrings,
  READMEs, `docs/`, error strings, commit messages, PR titles and branch names. There is no roadmap
  file and none is to be created. Equally forbidden: describing something that exists as "not yet
  implemented", "in progress" or "lands later", and advertising something that does not exist.
- User-facing changes update `README.md` and add an entry to `CHANGELOG.md`.
- New or renamed environment variables are documented in `.env.example` with a leading comment.
- Cross-link liberally: module README ↔ `docs/modules/` ↔ the ADRs that motivated the design.

---

## 7. Self-maintenance protocol

The single most important rule. When you (the agent) notice that this file, a module-level
`CLAUDE.md`, or a `.cursor/rules/*.mdc` file no longer matches the code (renamed module, changed API,
deprecated pattern, new convention emerging across PRs):

1. **Propose an update to the rule file in the same PR as the code change.**
2. If a pattern in code isn't yet captured in any rule, **propose adding it.**
3. Never silently ignore a stale rule — either follow it or update it.
4. If a module rule conflicts with this root file, **fix the conflict** before merging.
5. ADRs are immutable; supersede via a new ADR, don't edit existing ones.

This protocol applies to memories under `~/.claude/projects/.../memory/` too — when a saved memory
contradicts current code or rules, update or delete it.

Agents that silently ignore stale rules are worse than agents that don't read them at all.

---

## 7b. Contributing as an outside agent

If you are new to this repository: this file is the rules,
[`CONTRIBUTING.md`](CONTRIBUTING.md) is the human-facing contribution guide (environment setup, the
local check loop, what a reviewable PR looks like), and [`docs/agent-guide.md`](docs/agent-guide.md)
is the long-form agent playbook (how to orient, which file to open for which task, how to verify).
Read this file first, then the `CLAUDE.md` of whichever module you are about to touch.

The non-negotiables, in priority order:

- **Never commit directly to `dev` or `main`.** Topic branch → PR, every time (§8).
- **Never `--no-verify`**, never force-push a shared branch, never `--amend` a pushed commit.
- **Never weaken a type, an assertion, a threshold, or a test to make CI green.** Deleting a check
  does not fix a failure — it hides one. A red check is information; read it.
- **Never widen the sdist allow-list, and never commit anything from `.env`** (§9b). The build config
  is what keeps live provider keys out of a permanent public archive.
- **Update `docs/modules/<name>.md` and the module's `CLAUDE.md` in the same PR as an API change**
  (§6, §7). A doc that lags the code is a defect, not a follow-up.
- **Every new code path gets a test in the same PR** (§5).
- **Verify every symbol you write.** grep `src/strata_forge` before putting an import, a call, or an
  error message into a document.

Report security vulnerabilities and code-of-conduct concerns privately to `gemovic@strataml.com`
(see [`SECURITY.md`](SECURITY.md) and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)). Never open a
public issue for a vulnerability.

---

## 8. Commit and PR conventions

### Conventional Commits — required

Every commit subject MUST follow the Conventional Commits format:

```
<type>(<optional-scope>): <imperative summary>
```

The subject is a single line, lowercase, imperative, under 70 characters. `<type>` is one of the
values below — anything else is rejected at review.

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

Use a parenthesized scope when the change is local to one module — usually the module name (`core`,
`config`, `llm`, `prompts`, `tracing`, `datasets`, `evals`, `agents`, `rag`, `storage`, `compute`,
`training`, `pipelines`, `cli`) or a focused sub-area inside it. Omit the scope only when the change
spans the whole repo (root tooling, multi-module refactors, cross-cutting docs).

### Body and footer

- Body (optional): wrapped at ~72 chars, explains the *why* and any non-obvious *how*. Bullet points
  are fine.
- Reference issues / PRs in a trailer line (`Closes #123`, `Refs PR-456`).
- Breaking changes go in a `BREAKING CHANGE:` trailer or use the `!` marker (`feat(llm)!: replace
  LLMClient.complete signature`).

### Time-stable language

**Commit messages describe changes in time-stable terms.** They MUST NOT reference phase numbers,
sub-phase numbers, step numbers, plan-file slugs, or any other scaffolding metadata — those exist
only during construction and become meaningless afterward. Keep the construction process out of the
long-term git log. The same rule applies to PR titles, branch names, and any other long-lived
artifact.

### Other rules

- One commit per cohesive unit of work — never bundle unrelated changes for convenience.
- PR titles follow the same Conventional Commits format as the subject line. PR body has
  `## Summary`, `## Why`, `## Test plan`.
- Never `--no-verify`. Never `--force-push` to `main` or `dev`. Never `--amend` a commit that has been
  pushed somewhere shared.

### Branching

A three-tier flow — **every** change follows it; never commit directly to `dev` or `main`:

1. **Topic branch** off `dev`, named `<type>/<short-title>` (same `<type>` vocabulary as commits) —
   e.g. `feat/streaming-tool-loop`. One cohesive unit of work per branch.
2. **Merge into `dev`** (via PR) when done. `dev` is the persistent integration branch.
3. **Delete the topic branch once it's merged into `dev`** — local AND remote:
   `git branch -d <branch> && git push origin --delete <branch>`. **`dev` and `main` are the only
   persistent branches**; never leave a merged topic branch on the remote. (Sweep:
   `git branch -r --merged origin/dev --format='%(refname:short)' | grep -vE '^origin/(dev|main)$' | sed 's#^origin/##' | xargs -r git push origin --delete`.)
4. **Promote `dev` into `main`** once integrated — via a `dev` → `main` PR that a maintainer merges.

**`main` only ever receives `dev` — never a topic branch, never a direct push.** Enforced, not just
convention: `main` is protected (PR required, direct pushes blocked, no admin bypass) and a required
CI check (`.github/workflows/promotion-guard.yml`) **fails any PR into `main` whose source branch
isn't `dev`**. The guard runs on `pull_request_target`, so its definition comes from `main` and a PR
cannot disable it by editing the file in its own diff. Promotion is always a `dev` → `main` **PR** —
agents may open and merge it themselves (via the GitHub MCP or `gh`); what's forbidden is a *direct
push* to `main` and any non-`dev` source.

**`dev` must stay green.** It is the branch downstream consumers track and the only source `main`
accepts, so a broken `dev` stalls every release behind it. Run `make check && make test` before
opening a PR into it, and don't merge one whose CI is red.

**`main` is the released line.** It moves only by promotion, and release tags (`vX.Y.Z`) are cut from
it (§9b). strata-forge is a library, not a service — there is nothing to deploy; publishing to PyPI
is the only release action.

---

## 9. How to add a new top-level module

1. Create `src/strata_forge/<module>/` with `__init__.py` (module docstring), `README.md` (what + why,
   present tense, ships in the wheel), `CLAUDE.md` (module-specific rules).
2. Add a row to the **Project map** table in §2 of this file.
3. Add `docs/modules/<module>.md` and register it in `mkdocs.yml`'s nav.
4. Add a section to `docs/architecture/module-boundaries.md` stating what the module owns, what it
   may import, and what it must not.
5. If the module introduces new heavy deps, add an extra to `pyproject.toml` and gate the imports
   lazily inside the using function (§3.4).
6. Add at least one unit test under `tests/unit/<module>/`, and an `examples/NN_*.py` script for the
   headline capability.
7. If it touches provider APIs, add VCR cassettes under `tests/vcr/cassettes/<module>/`.
8. If the change is architecturally significant, write an ADR.
9. Update the module table in `README.md` and add a `CHANGELOG.md` entry.
10. Open a PR following §8.

---

## 9b. Publishing to PyPI

`strata-forge` is published to PyPI under **Apache-2.0**. The distribution is `strata-forge`; the
import package is `strata_forge`. Only the `strata-forge` console script is installed — never
reintroduce a bare `forge` binary or a bare `forge` import package, both of which collide with an
unrelated PyPI project.

**An sdist is public and permanent.** The build config is an allow-list, not a filter:
`[tool.hatch.build.targets.sdist] only-include` ships **only** `src/strata_forge` (hatchling adds
the readme, licence, `NOTICE`, and `pyproject.toml`), and both targets exclude `CLAUDE.md` /
`AGENTS.md` / `.env*`. If you add a build `include`/`force-include`, re-check what lands in the
archive — `.env` holds live provider keys and is kept out only by the allow-list plus VCS ignores.
`release.yml` re-asserts this before every upload and fails the job rather than publishing.

Anything under `src/` ships to a public index, so it must not name anything a reader outside this
repository can't see — internal services, private repositories, orchestration architecture. Write
module docstrings for that outside reader.

**Cutting a release:** bump `version` in `pyproject.toml` **and** `__version__` in
`src/strata_forge/__init__.py` (they are separate strings and will drift if you forget), add the
`CHANGELOG.md` entry, promote `dev` → `main` per §8, then tag `main` with `vX.Y.Z`. The tag triggers
`release.yml`, which verifies the tag matches the packaged version, builds, checks the archives, and
uploads via PyPI **Trusted Publishing** (OIDC — there is no API token in this repo).
`workflow_dispatch` publishes to TestPyPI for a rehearsal.

---

## 10. Where to look when stuck

- This file → cross-cutting rules.
- §2b above → which directory holds what, and when to open it.
- `src/strata_forge/<module>/CLAUDE.md` → module-specific rules.
- `docs/agent-guide.md` → the agent playbook: how to orient and how to verify.
- `CONTRIBUTING.md` → environment setup and the human-facing contribution loop.
- `docs/architecture/overview.md` → high-level architecture narrative.
- `docs/architecture/module-boundaries.md` → what each module may import.
- `docs/architecture/adr/` → "why does this look the way it does?"
- `docs/modules/<name>.md` → public API reference for a module.
- `docs/recipes/` → workflows that span more than one module.
- `examples/` → a runnable version of nearly every documented capability.
- `pyproject.toml` → ground truth for deps, extras, ruff config, pyright config, pytest markers.
- `Makefile` → all the things you can run with one command.
- `CHANGELOG.md` → what changed, and when.

When this file disagrees with code that's already merged, **the code wins** — and you update this
file. When this file disagrees with in-flight work, **this file wins** — and you update the code.
