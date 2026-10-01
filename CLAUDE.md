# CLAUDE.md — agent rules for AI Forge

This file is the single source of truth for cross-cutting conventions that every contributor (human or agent — Claude Code, Cursor, Codex, Copilot, …) follows in this codebase. Tools that read `AGENTS.md` get the same content via the symlink at the repo root.

Module-specific rules live in `src/strata_forge/<module>/CLAUDE.md`. If a module rule conflicts with this file, **fix the conflict** before merging — the module file is authoritative for itself, this file is authoritative for cross-cutting concerns.

Glob-scoped reinforcement of specific patterns lives in `.cursor/rules/*.mdc` for Cursor's rule engine, mirroring the relevant sections of this file.

---

## 1. North star and non-goals

**Goal.** AI Forge is a typed, async-first, batteries-included baseline that supports common AI/LLM experimentation workflows out of the box (inference, evaluation, fine-tuning, RAG, remote compute) and stays modular enough to extend into bespoke research without ripping apart its foundations.

**Non-goals.**
- Not a SaaS or hosted product — purely a library.
- Not a one-off research script — generality over single-use convenience.
- Not a wrapper around any one provider — vendor-neutral via LiteLLM at the transport seam.
- Not opinionated about prompt design or grader curation — Forge ships primitives, not opinions.
- Not aimed at production-grade inference serving (that's vLLM/TGI/SGLang's job; we orchestrate them via `strata_forge.compute`).

---

## 2. Project map

| Module | Purpose | Module rules |
|---|---|---|
| `strata_forge.core` | Cross-cutting utilities: errors (`ForgeError` hierarchy), retry, structlog logging with `trace_id` propagation, `BudgetContext`, reproducibility helpers, UUIDv7 ids, secret redaction (`Redactor`, the one scrubber for every surfaced string and relayed console), shared types. Strictly upstream — does not import from any other `strata_forge.*` module. | [`src/strata_forge/core/CLAUDE.md`](src/strata_forge/core/CLAUDE.md) |
| `strata_forge.config` | Pydantic Settings root with sub-models per concern; YAML profile overlays (`FORGE_PROFILE`); `.env` loading. The single configuration entry point — never read env vars directly elsewhere. | [`src/strata_forge/config/CLAUDE.md`](src/strata_forge/config/CLAUDE.md) |
| `strata_forge.llm` | Provider-abstracted async LLM client over LiteLLM. Owns the typed layer: Pydantic messages/responses, structured output, tools (`Tool`, `@tool`, `run_tool_loop`), multimodal, streaming, two-axis fallback, provider-agnostic cache, model registry. Reference: [`docs/modules/llm.md`](docs/modules/llm.md). | [`src/strata_forge/llm/CLAUDE.md`](src/strata_forge/llm/CLAUDE.md) |
| `strata_forge.prompts` | Jinja2 templating with safe filters + Langfuse-backed prompt registry. Templates model the stable-prefix / dynamic-suffix split so provider prompt caching just works. Reference: [`docs/modules/prompts.md`](docs/modules/prompts.md). | [`src/strata_forge/prompts/CLAUDE.md`](src/strata_forge/prompts/CLAUDE.md) |
| `strata_forge.tracing` | Langfuse observability. Auto-tracing via LiteLLM callback, `@traced` decorator, span context managers, score/metric helpers. Reference: [`docs/modules/tracing.md`](docs/modules/tracing.md). | [`src/strata_forge/tracing/CLAUDE.md`](src/strata_forge/tracing/CLAUDE.md) |
| `strata_forge.datasets` | Dataset CRUD + versioning; Langfuse ↔ Hugging Face Datasets bridge; synthetic-data primitives. | [`src/strata_forge/datasets/CLAUDE.md`](src/strata_forge/datasets/CLAUDE.md) |
| `strata_forge.evals` | Experiment runner (model × prompt × dataset × graders); graders, metrics, reports, parameter sweeps; trace replay; CI eval-regression gate. | [`src/strata_forge/evals/CLAUDE.md`](src/strata_forge/evals/CLAUDE.md) |
| `strata_forge.agents` | PydanticAI agent builder + built-in tools + memory + multi-agent patterns. **Reuses** `Tool`, `@tool`, message types, `run_tool_loop` from `strata_forge.llm` — does NOT re-implement tool plumbing. | [`src/strata_forge/agents/CLAUDE.md`](src/strata_forge/agents/CLAUDE.md) |
| `strata_forge.rag` | Embedders (via `strata_forge.llm`), Qdrant store, chunkers, retrieval (dense / BM25 / hybrid RRF), rerankers, composable pipeline. | [`src/strata_forge/rag/CLAUDE.md`](src/strata_forge/rag/CLAUDE.md) |
| `strata_forge.storage` | `fsspec` gateway (local, S3, GCS, Azure Blob, HF Hub); HF Hub model push/pull; dataset handling. | [`src/strata_forge/storage/CLAUDE.md`](src/strata_forge/storage/CLAUDE.md) |
| `strata_forge.compute` | Remote compute orchestration: SkyPilot via `sky.api.sdk`, raw `asyncssh` SSH backend, YAML task templates, async batch inference. | [`src/strata_forge/compute/CLAUDE.md`](src/strata_forge/compute/CLAUDE.md) |
| `strata_forge.training` | Fine-tuning: SFT, DPO/ORPO/KTO/GRPO, PEFT (LoRA/QLoRA), chat-template formatting, sequence packing. | [`src/strata_forge/training/CLAUDE.md`](src/strata_forge/training/CLAUDE.md) |
| `strata_forge.cli` | Typer entry points: `chat`, `eval`, `experiments`, `prompts`, `datasets`, `train`, `serve`, `compute`, `doctor`. | [`src/strata_forge/cli/CLAUDE.md`](src/strata_forge/cli/CLAUDE.md) |

Per-module reference docs live under `docs/modules/<name>.md`. The phase roadmap lives in [`docs/roadmap.md`](docs/roadmap.md); architectural rationale lives in `docs/architecture/`.

---

## 3. Architecture principles

These apply everywhere in `src/strata_forge/`. Code that violates them is wrong by default; if a violation is genuinely necessary, write an ADR before merging.

### 3.1 Async-first public API
- Every public function on every module is `async`.
- Sync wrappers live **only** in `strata_forge.sync` and exist for CLI / notebook ergonomics. They wrap async functions via `asyncio.run`.
- Internal calls are async end-to-end. Do not mix `requests` / sync `httpx` with async code paths.
- HTTP I/O uses `httpx.AsyncClient` (often indirectly through LiteLLM).

### 3.2 Strict typing
- pyright runs in `strict` mode on `src/strata_forge/`. New code must be strict-clean.
- No `Any` without a single-line `# pyright: ignore[reportXxx]` justification.
- Data types: Pydantic v2 models. Behavior types: `Protocol`. Narrow boundaries: `TypedDict`.
- `from __future__ import annotations` at the top of modules using generics — keeps annotations lazy on Python 3.14.

### 3.3 Exception-based errors
- Every Forge-raised error inherits from `strata_forge.core.errors.ForgeError`.
- Provider exceptions are normalized at the seam (`strata_forge.llm.errors.map_litellm_exception`) into `ProviderError` subclasses.
- Never `except:`. Never swallow exceptions. Decorators (`@retry`) drive control flow; predicates select on `ProviderError` subclasses.
- Validation errors raise `ValidationError` (or a subclass), never silently coerce.

### 3.4 Module independence + lazy heavy imports
- `import strata_forge.<module>` MUST NOT crash when an optional extra is uninstalled.
- Heavy deps (`torch`, `transformers`, `trl`, `peft`, `skypilot`, `asyncssh`, `qdrant-client`, `vllm`, `pillow`, `redis`, `inspect-ai`) live behind PyPI extras — `[compute]`, `[finetuning]`, `[serving]`, `[multimodal]`, `[redis]`, `[inspect]`.
- Heavy deps are imported **inside the function that uses them**, never at module load. The function raises `ImportError` with an installation hint if the extra is missing.

### 3.5 Composability + escape hatches
- Modules expose a small core API plus an escape hatch for advanced cases (e.g. `LLMClient.complete(..., provider_extras={"anthropic": {...}})` forwards verbatim to LiteLLM).
- Never block users from accessing the underlying transport. The escape hatch is a feature, not a leak.

### 3.6 Observability built in
- Every LLM call is traced (when Langfuse is configured) and dumpable to NDJSON (env-gated via `FORGE_DIAGNOSTIC=1`).
- Use the structlog logger from `strata_forge.core.logging`; `trace_id` propagates via contextvars across `await` boundaries — never pass it manually.
- Do not log full prompts at `INFO`; use `DEBUG`. Never log raw API keys, AWS signatures, or session tokens.
- A credential a compute job needs travels in `Task.secrets`, delivered as a private file the job reads and deletes, never in `Task.env`, a command line, a generated script or a child process's environment ([ADR 0019](docs/architecture/adr/0019-secrets-travel-beside-the-task.md)).

### 3.7 Cost awareness
- LLM-touching code paths must respect any active `BudgetContext`. Pre-call estimation + post-call true-up.
- Don't silently truncate prompts; raise `BudgetExceededError` instead.

### 3.8 Reproducibility
- Long-running operations capture `strata_forge.core.repro.env_snapshot()` in their run metadata.
- Pseudo-randomness goes through `strata_forge.core.repro.set_seed()` — never call `random.seed` directly.
- Dataset content is hashed via `content_hash()` for run provenance.

### 3.9 No vendor lock-in in higher-level modules
- `strata_forge.evals`, `strata_forge.agents`, `strata_forge.rag` use `strata_forge.llm` for all model calls. They never import provider SDKs directly.
- Vector store, prompt registry, dataset bridge — all behind interfaces that can be swapped.

---

## 4. Coding standards

Most of these are enforced by `ruff` and `pyright`. Local violations without justification fail pre-commit.

- **Ruff** is the lint + format authority. Single config in `pyproject.toml`. Line length 100. `target-version = "py314"`. Selected rule sets include E, F, W, I, N, UP, B, A, C4, PIE, SIM, RUF, ASYNC, S (bandit), PT (pytest), TID, TCH, PTH, ERA.
- **Pyright** strict on `src/strata_forge`. Tests are typed but not strict (pragmatic). `Any` requires a justified ignore comment.
- **Naming:** `snake_case` for functions / variables / modules; `PascalCase` for classes; `SCREAMING_SNAKE` for constants. Pydantic models follow `XxxConfig`, `XxxRequest`, `XxxResponse`, `XxxError`.
- **Imports:** ruff isort. First-party is `forge`. Avoid `from strata_forge.<other_module> import *` re-exports without explicit `__all__`.
- **Docstrings:** module-level summary mandatory. Public functions / classes: one-line summary; multi-line only when the *why* is non-obvious. No paragraph-long docstrings — long-form belongs in `docs/modules/<name>.md`.
- **Comments:** rare. Only when the *why* is non-obvious. Never describe *what* — names already do that. Never reference current tasks, PRs, or scaffolding metadata.
- **Emojis:** none in source.
- **`from __future__ import annotations`** at the top of modules that use generics in signatures.

---

## 5. Testing discipline

- Every new code path gets a test in the same PR.
- Prefer fast unit tests (no network) for most logic. Use VCR for provider-touching tests; live integration is env-gated and runs only nightly.
- Coverage thresholds enforced in CI:
  - `src/strata_forge/llm/` ≥ 90 % line
  - `src/strata_forge/core/`, `src/strata_forge/config/` ≥ 85 % line
- No skipped tests committed without a tracking issue link in the skip reason.
- VCR cassettes scrub secrets in `before_record_request`. Recording without scrubbing = leaking credentials; rotate them.
- Snapshot tests (`syrupy`) for provider format mappings — a failing snapshot is an early-warning signal, not noise to suppress.
- Marker discipline: `live`, `redis`, `slow`, `integration` (declared in `pyproject.toml`; new markers go there).
- Dependency pins are proven by `.github/workflows/clean-install.yml`, not by the lockfile: a run's machine installs this package fresh, so the job installs the same extras strings into empty venvs with no constraints and builds every config a run builds (`scripts/smoke_clean_install.py`). Its matrix is the extras strings the orchestrator installs on a run's machine, verbatim; when those change, the matrix changes in the same step. A new extras string a runner needs, or a new thing a runner constructs before loading weights, goes into the matrix or the smoke script in the same PR. That job executes upstream code no lockfile vouches for, so it never runs in `main`'s context (the schedule dispatches it on `dev`); and a job that holds a secret, a write token or a publishing identity never restores the Actions cache (`enable-cache: false`), because every ref restores what the default branch's scope holds.

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

### Native model routes — the registry ships first

`src/strata_forge/llm/registry_data.yaml` is the **allowlist for every vendor-native chat route the Strata app offers**. strata-server runs chat in-process on the forge it bundles, and `routing.resolve()` refuses (`RegistryError(reason="unknown_model")`) any id on the `anthropic` / `openai` / `vertex` / `bedrock` / `azure` routes that the registry does not carry; only `openai_compat` (OpenRouter) passes ids through. An app catalog entry naming a native route for an unregistered model therefore fails with "Unknown model" for every user who holds only that vendor's key. When a vendor's lineup changes:

1. **Register the models in forge.** Copy ids, context window, max output, pricing and capability flags from the vendor's own documentation, never from memory, the app's catalog, LiteLLM's model map or OpenRouter, and name the pages in the entry's source comment. For OpenAI that is the model page AND the GPT family / migration guide (`developers.openai.com/api/docs/guides/latest-model`) and the reasoning guide (`.../guides/reasoning`), which carry the endpoint and parameter caveats the model page omits; for Anthropic it is the models overview, the pricing page and the thinking page (`platform.claude.com/docs/en/build-with-claude/thinking`, its tool-use and sampling-parameter limits). A capability flag states what the model does on its route's wire API: every OpenAI `openai` route speaks the Responses API (`wire_api: responses`, [ADR 0018](docs/architecture/adr/0018-openai-routes-speak-the-responses-api.md)), where current OpenAI models call tools at any reasoning effort, and `sampling_params: false` marks a model that rejects `temperature` / `top_p` at its default effort. Update `CURRENT_LINEUP` in `tests/unit/llm/test_routing.py` in the same change.
2. **Smoke every new native route live** with a real key and the parameters the server actually sends (tools, its `max_tokens` budget): `scripts/smoke_responses.py` for OpenAI routes (two legs: a tool call that suspends, then the resumed turn with its replayed reasoning), and strata-server's staging smoke once staging bundles the change. A registry entry that has never answered a tool call is not done.
3. **Release in one sequence across the three repos.** forge PR into `dev`, carrying the version bump §9b assigns and renaming `CHANGELOG.md`'s **Unreleased** heading to it: a **patch** for registry entries alone (the model registry is not §9b's method or dataset-format registry), a **minor** when the change also does something §9b lists, such as raising the dependency pin a new wire API needs → strata-server re-locks against forge `dev` and redeploys staging (its CLAUDE.md §3.8; the `strata-server-dev` redeploy §9b requires after every forge merge to `dev`), and the staging smoke passes → forge `dev` → `main`, tagged `vX.Y.Z` the same day (PyPI; §9b never leaves `main` untagged) → strata-server `dev` → `main`, whose production deploy rebuilds against that forge release → only then the app catalog PR (its `native-models` check green) → app `dev` → `main`.
4. **The app may offer a native route only for a model whose entry has `tool_calling: true`**, because the app's chat always sends tools. A model forge cannot carry natively stays on the app's OpenRouter route. `CURRENT_LINEUP` is forge-local and cannot see the app's catalog; the recurrence guard for "the app names a native id forge lacks" is the app's `native-models` CI check, which resolves every native catalog route against the forge registry.

---

## 9. How to add a new top-level module

1. Create `src/strata_forge/<module>/` with `__init__.py` (module docstring), `README.md` (what + why), `CLAUDE.md` (module-specific rules).
2. Add a row to the **Project map** table in §2 of this file.
3. Add `docs/modules/<module>.md` reference doc.
4. If the module introduces new heavy deps, add an extra to `pyproject.toml` and gate the imports lazily inside the using function.
5. Add at least one unit test under `tests/unit/<module>/`.
6. If it touches provider APIs, add VCR cassettes under `tests/vcr/cassettes/<module>/`.
7. If the change is architecturally significant, write an ADR.
8. Update `docs/roadmap.md` if the new module materially shifts phase status.
9. Open a PR following §8.

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

Anything under `src/` ships to a public index, so it must not name private sibling repos or
internal infrastructure — write module docstrings for an outside reader.

**Cutting a release:** bump `version` in `pyproject.toml` **and** `__version__` in
`src/strata_forge/__init__.py` (they are separate strings and will drift if you forget), re-run
`uv lock` (the lockfile records the project's own version; `uv lock --check` catches a stale
one), record the release under its own heading in `CHANGELOG.md` (entries accumulate under
**Unreleased** as they merge to `dev`; the bump renames that heading), promote `dev` → `main` per
§8, then tag `main` with `vX.Y.Z`. The tag triggers `release.yml`, which verifies the tag matches
the packaged version, builds, checks the archives, and uploads via PyPI **Trusted Publishing**
(OIDC — there is no API token in this repo). `workflow_dispatch` publishes to TestPyPI for a rehearsal.

**When a release is required — the release rule.** The engine runs on machines the control plane
installs it onto at launch, pinned to the exact version the control plane validated the run's spec
against (`strata_forge.pipelines.SPEC_VERSION`, sent as the spec's `engine_version`; the runner
refuses any other installed engine with `engine version mismatch`). A change the control plane must
see therefore has to be a version the control plane can pin:

- Every promotion of `dev` → `main` that changes a **runner spec** (`RunSpec`, `FinetuneSpec`, a
  `Hyperparams` field), the **method or dataset-format registry**, the **`Task` shape** or a
  **dependency pin** is a **minor** bump, tagged the same day through `release.yml`. A fix that
  changes none of those is a **patch** release. Never promote such a change to `main` untagged:
  production installs a released version, so an untagged `main` is a change nothing can run.
- The server's **production** promotion always follows the forge tag — the pin makes that
  enforceable, because the production install string names the release the server bundled, and
  a release that is not on PyPI cannot be installed.
- **Staging** needs no release: it installs from `forge_ref=dev`, pinned to the exact commit the
  staging server image bundled (`engine_version` is then `"<version>+<commit>"`). But the image
  bundles forge only when the server is deployed, so **every forge merge to `dev` (and always a
  version bump) is followed by a `strata-server-dev` redeploy** (`workflow_dispatch` of the
  server's staging deploy workflow); until then staging keeps running the commit the image
  bundled, and a capability that landed on forge `dev` is not on staging yet.

---

## 10. Where to look when stuck

- This file → cross-cutting rules.
- `src/strata_forge/<module>/CLAUDE.md` → module-specific rules.
- `docs/architecture/overview.md` → high-level architecture narrative.
- `docs/architecture/adr/` → "why does this look the way it does?"
- `docs/modules/<name>.md` → public API reference for a module.
- `docs/roadmap.md` → what's done, what's next, what's deferred.
- `pyproject.toml` → ground truth for deps, extras, ruff config, pyright config, pytest markers.
- `Makefile` → all the things you can run with one command.

When this file disagrees with code that's already merged, **the code wins** — and you update this file. When this file disagrees with in-flight work, **this file wins** — and you update the code.
