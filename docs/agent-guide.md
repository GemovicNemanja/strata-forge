# Agent guide

This is the playbook for a coding agent working in strata-forge for the first time.
[`CLAUDE.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CLAUDE.md) is the rules —
what is allowed, what is forbidden, what a commit has to look like. This page is the how-to: the
order to read files in, which file to open for which task, which files have to change together, and
how to prove your change is correct before you open a pull request. Nothing here overrides the
rules; where the two disagree, the rules win and this page is the thing that needs fixing.

Tools that read `AGENTS.md` get `CLAUDE.md` — it is a symlink, not a second document. Humans get
[`CONTRIBUTING.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CONTRIBUTING.md),
which covers environment setup and the same expectations in prose.

## Contents

- [Orientation ladder](#orientation-ladder)
- [Navigation: I want to X](#navigation-i-want-to-x)
- [The import graph, and how to check yourself](#the-import-graph-and-how-to-check-yourself)
- [Conventions that fail quietly](#conventions-that-fail-quietly)
- [The verification loop](#the-verification-loop)
- [Definition of done](#definition-of-done)
- [Anti-patterns](#anti-patterns)
- [When the rules and the code disagree](#when-the-rules-and-the-code-disagree)

---

## Orientation ladder

Read in this order. Steps 1 to 4 are a one-time cost of roughly twenty minutes and give you the
whole shape of the repository; steps 5 to 8 are paid again for each new area you touch.

| # | Read | Size | The question it answers |
|---|---|---|---|
| 1 | `README.md` | ~90 lines | What is this library, what does `pip install` give you, which module does what? |
| 2 | `CLAUDE.md` §1–§4 | ~200 lines | What am I allowed to do, and what will be rejected on sight? |
| 3 | `docs/architecture/overview.md` | ~190 lines | Why is it shaped like this? Which layer does my change sit in? |
| 4 | `docs/architecture/module-boundaries.md` | ~260 lines | What may this module import, what does it own, and what does it deliberately not do? |
| 5 | `docs/modules/<name>.md` | 240–850 lines | What is the public contract I must not break? |
| 6 | `src/strata_forge/<name>/CLAUDE.md` | 60–150 lines | What do I need to know that the reference page does not say — the gotchas, the test patterns, the local invariants? |
| 7 | `tests/unit/<name>/` | varies | What does "working" mean here, concretely? |
| 8 | The ADRs linked from steps 3–6 | <200 lines each | Why was the obvious alternative rejected? |

Two shortcuts worth knowing:

- **Do not read a module's source top to bottom as your first move.** The reference page plus the
  unit tests give you an accurate model for a fraction of the tokens, and the tests are the part
  that cannot lie. Open source files when you are about to change them — and when you do, read the
  file you are changing *and* the package `__init__.py`, because its `__all__` is the public-surface
  contract that a rename silently breaks.
- **Skip straight to the navigation table below when the task is a known shape.** The four
  always-read documents describe the repository; the table describes your task.

A note on the module reference pages: every claim in them is meant to be verifiable against
`src/strata_forge`. If a snippet does not run or a symbol does not exist, that is a defect to fix in
the same PR, not a quirk to work around (§7 of the rules).

---

## Navigation: I want to X

**Read first** is the minimum context that makes the change safe. **Changes together** is the set a
reviewer expects to see in one pull request; a row is not done until every file in it is consistent.
Two entries are implied everywhere and therefore omitted from the rows: the module's
`docs/modules/<name>.md` when the public surface moves, and `CHANGELOG.md` under `## [Unreleased]`
when the change is user-visible.

### Extending a module

| Goal | Read first | Changes together |
|---|---|---|
| Add or reprice a model | `src/strata_forge/llm/registry_data.yaml`, `llm/registry.py`, the registry section of `docs/modules/llm.md` | `registry_data.yaml` (pricing gets a source link in a YAML comment — never hard-code pricing in Python), `tests/unit/llm/test_registry.py`, the model table in `docs/modules/llm.md` |
| Add a provider route | `llm/providers/base.py`, one existing sibling (`providers/anthropic.py` is the shortest), `providers/config.py`, `llm/client.py::_default_provider_clients`, ADR 0001, ADR 0004 | `llm/providers/<name>.py`, `providers/config.py`, `providers/__init__.py`, `llm/__init__.py`, `registry_data.yaml` routes, `llm/errors.py` if the provider raises new exception shapes, `tests/unit/llm/providers/`, `tests/vcr/cassettes/<name>/`, `.env.example`. A new **vendor** (not just a route to an existing one) needs an ADR superseding 0004 |
| Add a grader | `evals/graders/base.py` (the `Grader` Protocol: a `name` property and `async grade(*, item, response)`), `graders/exact.py` as the simplest implementation, `evals/experiment.py` for `GraderResult`, ADR 0010, and [Evaluate a model with a custom grader](recipes/evaluate-a-model-with-a-custom-grader.md), which is the worked example for this exact task | `evals/graders/<name>.py`, `graders/__init__.py`, `evals/__init__.py`, `src/strata_forge/evals/README.md` (it enumerates every shipped grader and ships inside the wheel), `tests/unit/evals/graders/test_<name>.py`. A grader that should also be selectable from the CLI needs `cli/eval_cmd.py::_make_grader` **and** its `unknown grader ... Supported:` message, the supported-grader list in `docs/modules/cli.md`, and `tests/unit/cli/test_eval_cmd.py` — without those, `strata-forge eval` cannot reach it |
| Add a metric | `evals/metrics.py` (note `bleu` / `rouge` lazily import the `[evals]` extra), `tests/unit/evals/test_metrics.py` | `evals/metrics.py`, `evals/__init__.py`, `tests/unit/evals/test_metrics.py` |
| Add a chunker, retriever, or reranker | The Protocol you are implementing — `rag/chunking.py::Chunker` (sync, CPU-bound), `rag/retrieval.py::Retriever`, `rag/rerankers.py::Reranker` — plus the nearest concrete sibling and [Build a RAG pipeline over Qdrant](recipes/build-a-rag-pipeline-over-qdrant.md). Also `rag/pipeline.py::IndexableRetriever`: a retriever that owns its corpus implements `async index(chunks)` too, or `RAGPipeline.ingest` raises `"retriever of type X doesn't satisfy IndexableRetriever"` (`DenseRetriever` satisfies it; `BM25Retriever` deliberately does not, because it takes its corpus at construction) | `rag/<name>.py`, `rag/__init__.py`, `src/strata_forge/rag/README.md` (it enumerates the shipped retrievers and ships inside the wheel), `tests/unit/rag/test_<name>.py`. A new third-party backend also needs a lazy import, an extra in `pyproject.toml`, and the `ImportError` hint |
| Add a vector-store backend | `rag/vector_store.py` (`VectorStore` Protocol: `add` / `search` / `delete` / `clear`), `rag/qdrant.py` as the hosted reference | `rag/<name>.py`, `rag/__init__.py`, `tests/unit/rag/test_<name>.py`. `strata_forge.agents.memory` re-exports these primitives, so check `agents/memory/vector_store.py` still lines up |
| Add a compute backend | `compute/backends/base.py` (six methods, including `read_file` and the `safe_workdir_relpath` guard), `backends/local.py` for the in-process shape, `backends/ssh.py` for the remote shape, ADR 0013 and ADR 0016 | `compute/backends/<name>.py`, `backends/__init__.py`, `compute/__init__.py`, `tests/unit/compute/test_<name>_backend.py` (unit tests inject a fake SDK through `sys.modules`) |
| Add a built-in agent tool | `llm/tools.py` for `Tool` and the `@tool` decorator, `agents/tools/calculator.py` (module-level instance) versus `agents/tools/fs_read.py` (factory, because it needs caller configuration) | `agents/tools/<name>.py`, `agents/tools/__init__.py`, `agents/__init__.py`, `tests/unit/agents/tools/test_<name>.py`. Never define a new tool abstraction — reuse `strata_forge.llm` |
| Add a prompt or dataset store backend | `prompts/registry.py::PromptStore` or `datasets/store.py::DatasetStore` (both are ABCs, not Protocols), the in-memory backend for semantics, the Langfuse backend for the async-over-sync-SDK pattern | `<module>/stores/<name>.py`, `stores/__init__.py`, the package `__init__.py`, `tests/unit/<module>/stores/`, and `cli/helpers.py` if the CLI should be able to pick it |
| Add a training method | `training/preference.py` (all four preference methods share one config base and one runner), `training/sft.py`, `training/peft.py` | `training/preference.py`, `training/__init__.py`, `tests/unit/training/`. Remember `train()` is synchronous on purpose (§3.1) and every heavy import stays inside it |

### Cross-cutting changes

| Goal | Read first | Changes together |
|---|---|---|
| Add or change an error type | `core/errors.py`, ADR 0003, and `llm/errors.py::map_litellm_exception` if it is provider-shaped | `core/errors.py`, `core/__init__.py`, `tests/unit/core/test_errors.py`, plus the raising module and its docs. Runtime failures a caller might catch get a `ForgeError` subclass; a bad argument stays `ValueError` / `TypeError` |
| Add a config setting | `config/settings.py` (each sub-model is its own `BaseSettings` with an `env_prefix`), `config/CLAUDE.md`, `.env.example` | `config/settings.py` (field plus the sub-model's `__all__`), `config/__init__.py` re-export — the two lists have drifted before — `.env.example` with a leading comment, `tests/unit/config/test_settings.py`, and `cli/doctor.py` if the value belongs in the diagnostic. Nothing else in the tree may read the variable directly |
| Add an optional dependency | `pyproject.toml` `[project.optional-dependencies]`, any existing lazy-import guard (`rag/qdrant.py::_build_client` is the canonical one) | `pyproject.toml` (the extra **and** the `all` aggregate), the guarded import inside the using function, the `ImportError` hint naming `strata-forge[<extra>]`, a test that asserts the hint, and the extras list in `README.md` |
| Add a CLI command | `cli/main.py` (how sub-apps attach), one existing command module, `cli/helpers.py` (`run_async`, `error_exit`, the store factories) | `cli/<name>.py`, `cli/main.py`, `tests/unit/cli/test_<name>_cmd.py`, `docs/modules/cli.md`. The command stays a thin wrapper: business logic belongs in the module underneath |
| Add a new top-level module | `CLAUDE.md` §9 — it is a checklist, follow it literally | `src/strata_forge/<module>/` with `__init__.py`, `README.md` and `CLAUDE.md`; a row in the §2 project map and the §2b repository map; `docs/modules/<module>.md`; a section in `docs/architecture/module-boundaries.md`; `tests/unit/<module>/`; `mkdocs.yml` nav; an ADR if the module introduces a new seam |
| Change a public API | The module page, the module `CLAUDE.md`, and `git grep` for every call site including `examples/` and `notebooks/` | The source, the tests, `docs/modules/<name>.md`, the module `README.md` if the product description shifts, any affected example or notebook (`tests/e2e` imports all of them), and `CHANGELOG.md` |

### Docs, examples, and tests

| Goal | Read first | Changes together |
|---|---|---|
| Add a runnable example | `examples/README.md`, `examples/_common.py` (`parse_args`, `require_env`, `print_summary`), the nearest numbered sibling | `examples/NN_<slug>.py` with the body under `if __name__ == "__main__":`, a link from the owning `docs/modules/<name>.md`, and any inventory in `examples/README.md`. `tests/e2e/test_example_imports.py` picks the file up automatically and fails if an import drifts |
| Add a notebook | `notebooks/README.md`, `notebooks/01_eval_iterate.py` | `notebooks/NN_<slug>.py` (marimo, stored as reviewable Python), the `notebooks/README.md` table. `tests/e2e/test_notebook_imports.py` imports it |
| Record a VCR cassette | `tests/vcr/README.md`, `tests/vcr/conftest.py` (the scrubbing filter), `tests/vcr/test_providers.py` | The test function, then `make vcr-record` with live keys and `RECORD=1`. Inspect the produced YAML before committing — an unscrubbed cassette is a credential leak and the key must be rotated |
| Write an ADR | Any recent ADR for the shape; `docs/architecture/adr/` for the next free number | `docs/architecture/adr/NNNN-short-slug.md` (Context → Decision → Consequences, under ~200 lines), plus links to it from the module page and the module `CLAUDE.md`. ADRs are immutable once merged — supersede with a new one that links back |
| Add a recipe | `docs/recipes/README.md` and the existing recipe | `docs/recipes/<slug>.md`, the `docs/recipes/README.md` index, `mkdocs.yml` nav. Recipes are for workflows that no single module page can host |
| Fix a docstring | The module in question, `CLAUDE.md` §4 and §9b | The docstring only. Remember docstrings ship inside the wheel and the sdist: they are read by strangers who have the package and nothing else — no internal infrastructure, no build-order language, no dangling links into `docs/` |

---

## The import graph, and how to check yourself

The dependency rule in one sentence: **imports point down, never up, and never sideways except at
two documented seams.**

- `strata_forge.core` imports nothing from `strata_forge`. It is the only module with that property,
  and it is what makes the rest of the graph acyclic.
- `strata_forge.config` imports `core` and nothing else.
- `strata_forge.llm`, `strata_forge.tracing`, and `strata_forge.storage` sit on top of those two.
  `tracing` is special: nothing imports it, because it wraps the library from above through the
  LiteLLM callback and through `@traced` applied wherever the *caller* wants a span (ADR 0008).
- The capability modules — `prompts`, `datasets`, `evals`, `agents`, `rag`, `compute`, `training` —
  sit above the transport layer. The two sanctioned sideways edges are `agents` importing `rag` for
  vector-store primitives (ADR 0012) and `evals` importing `datasets` for typing.
- `pipelines` is the only module that composes across the capability layer. `cli` and `sync` are
  sinks: nothing imports them, so neither can create a cycle.
- Model-provider SDKs (`anthropic`, `openai`, `boto3`, Google's clients) are reachable only from
  `src/strata_forge/llm/providers/`. Higher modules call `LLMClient`, or use the
  `provider_extras={...}` passthrough when they need something the typed layer does not model.

[`module-boundaries.md`](architecture/module-boundaries.md) is the authoritative table, including
the "may import" ceiling for each module and what each one imports today. Before you add an
intra-package import, check the row. After you add one, run the checks below.

```bash
# 1. core imports nothing from the package except itself. Expect no output.
grep -rEn "^(from|import) strata_forge" src/strata_forge/core --include='*.py' \
  | grep -v "strata_forge\.core"

# 2. config imports only core. Expect no output.
grep -rEn "^(from|import) strata_forge" src/strata_forge/config --include='*.py' \
  | grep -vE "strata_forge\.(config|core)"

# 3. Provider SDKs outside the transport seam. Expect exactly one line:
#    src/strata_forge/llm/errors.py imports `openai` to name the exception classes
#    LiteLLM re-raises, so the normalizer can catch them.
grep -rEn "^[[:space:]]*(from|import) (anthropic|openai|boto3|vertexai|google\.(cloud|generativeai))" \
  src/strata_forge --include='*.py' | grep -v "/llm/providers/"

# 4. Direct environment reads outside strata_forge.config. Expect only the
#    considered exceptions: the Langfuse prompt and dataset stores (credential
#    fallback when a caller injects its own SDK client), training/progress.py
#    (FORGE_PROGRESS_PATH), pipelines/inference_runner.py (an env-driven
#    entrypoint by design), and compute/backends/local.py, which copies the
#    parent environment into a subprocess rather than reading a setting.
grep -rn "os\.environ\|os\.getenv" src/strata_forge --include='*.py' \
  | grep -v "^src/strata_forge/config/"

# 5. A heavy dependency imported at module scope. Expect no output.
grep -rEn "^(import|from) (torch|transformers|trl|peft|datasets|sky|asyncssh|qdrant_client|cohere|sentence_transformers|fsspec|huggingface_hub|langfuse|redis|PIL|nltk|boto3)\b" \
  src/strata_forge --include='*.py'
```

Check 5 has a second, stronger form that catches what grep cannot — importing the module in a
bare environment:

```bash
uv run python -c "
import importlib
for m in ('core','config','llm','prompts','tracing','datasets','evals','agents','rag','storage','compute','training','pipelines','cli','sync'):
    importlib.import_module(f'strata_forge.{m}')
print('all modules import on a bare install')
"
```

---

## Conventions that fail quietly

These are the rules that lint will not catch and a reviewer skimming a diff can miss. Each one has
bitten this codebase or is one careless edit away from doing so.

**Public I/O is `async`.** Every public function that touches a network, a disk, or a subprocess is
`async def`. Pure helpers stay synchronous on purpose — `get_settings`, `render`, `content_hash`,
registry lookups, `Chunker.chunk`. Do not wrap a pure function in `async` for symmetry, and do not
make an I/O function sync for convenience. The one standing exception is `SFTRunner.train` /
`PreferenceRunner.train`, because TRL's loop is blocking; a new exception needs the same explicit
treatment in the module docs and rules file.

**Sync wrappers live only in `strata_forge.sync`.** It wraps exactly four `LLMClient` methods via
`asyncio.run`. Never add a `*_sync` twin next to an async function, and never reach for
`asyncio.run` inside a library code path: the only sanctioned sites in `src/` are `sync.py`, the
CLI's `run_async` helper in `cli/helpers.py`, and the `python -m` entrypoint in `pipelines/`, each
of which is a boundary between an async library and a synchronous caller.

**Heavy imports go inside the function, behind an extra, with a hint.** Three things travel
together and a missing one is a silent regression: the import inside the using function, the extra
in `pyproject.toml`, and an `ImportError` whose text names the real distribution. The house idiom:

```python
def _build_client(self) -> Any:
    try:
        qdrant_mod: Any = __import__("qdrant_client", fromlist=["AsyncQdrantClient"])
    except ImportError as exc:
        msg = (
            "The [rag] extra is required for QdrantVectorStore. "
            "Install it with: pip install 'strata-forge[rag]'."
        )
        raise ImportError(msg) from exc
    return qdrant_mod.AsyncQdrantClient(**self._connect_kwargs)
```

`ai-forge` is a different project on PyPI. It must never appear in a hint, a doc, or an example.

**Never `except:`, and never swallow.** Catch the narrowest exception you can name and re-raise with
`from exc`. The single sanctioned swallow is `strata_forge.tracing`, which must never break a
caller's hot path (ADR 0008).

**Errors inherit `ForgeError` — but only the ones that should.** A runtime failure a caller might
reasonably catch (provider fault, budget ceiling, registry miss, cache fault, validation) gets a
`ForgeError` subclass. A bad argument at the call boundary stays `ValueError` or `TypeError`; that
is the standing convention across every module, and converting existing ones needs an ADR.

**`Any` needs a justification on the same line.** The house form is the suppression code, an em
dash, and the reason: `# noqa: S307 — AST-whitelisted`, `# pyright: ignore[reportMissingTypeStubs]`.
Lazily imported SDKs are the legitimate source of `Any` — annotate the boundary explicitly and keep
the `Any` from spreading past it.

**`from __future__ import annotations` plus `TYPE_CHECKING` has a trap.** Ruff's TC rules want
type-only imports moved into an `if TYPE_CHECKING:` block, which breaks any Pydantic model whose
field annotation needs runtime resolution. When ruff is wrong about that, the fix is
`# noqa: TC001 — Pydantic needs runtime resolution`, not deleting the annotation. Typer command
signatures have the same problem for a different reason (it evaluates annotations at runtime), and
use `# noqa: TC003`.

**Paths are `pathlib`.** Ruff's PTH rules enforce it, but only where they can see it — string
concatenation of paths passes lint and is still wrong. `strata_forge.core.types.PathLike` is the
parameter type for anything that accepts a caller-supplied path.

**Budgets raise; they never truncate.** If a prompt will not fit under an active `BudgetContext`,
raise `BudgetExceededError`. Silently dropping messages to fit produces a result the caller cannot
distinguish from a good one. Note the accounting is post-call: a ceiling stops the *next* call, not
the one in flight.

**Logging.** Use `strata_forge.core.logging.get_logger`. `correlation_id` propagates through a
contextvar across `await` boundaries — never thread it through call signatures. Full prompts go at
`DEBUG`, never `INFO`. API keys, AWS signatures, and session tokens never get logged at all, and
`Settings` is never logged whole.

**Comments explain why.** Names explain what. No comment should reference a task, a PR, a plan, or
a build order; those are meaningless to the next reader. No emojis anywhere in the repository.

---

## The verification loop

Run these in order. Each one is cheap relative to the one after it, so failing early saves time.

| Step | Command | What it catches |
|---|---|---|
| 1 | `make fmt` | Formatting. Ruff format is authoritative — never hand-format around it |
| 2 | `make lint` | The selected ruff rule sets (E, F, W, I, N, UP, B, A, C4, PIE, SIM, RUF, ASYNC, S, PT, TID, TCH, PTH, ERA) |
| 3 | `make type` | pyright strict over `src/strata_forge` |
| 4 | `make test` | The unit suite (`tests/unit`), offline and fast |
| 5 | `make test-cov` | The same suite with coverage; the gate is a package-wide 85 % line floor |
| 6 | `make test-all` | Everything, including `tests/e2e` (imports every example and notebook) and the offline cross-module tests. Run this before any PR that moves the public API |
| 7 | `make vcr-replay` | Provider request shapes against recorded cassettes. Relevant when you touch `strata_forge.llm` |
| 8 | `make docs-build` | The MkDocs site with `--strict`: a broken link or a page missing from the nav fails the build |

`make check` is steps 2 and 3 together and is the usual inner loop. `make integration` needs the
local Docker services (`make stack-up`); `make eval-gate` makes live LLM calls and costs money.

**What pyright strict rejects** that ordinary Python review does not:

- an unannotated parameter or return type anywhere in `src/strata_forge`;
- a value whose type it cannot fully infer flowing into a typed API — this is why lazily imported
  SDK handles are annotated `Any` at the boundary and immediately narrowed;
- access to another module's `_private` name (`reportPrivateUsage`) — the test suite does this
  deliberately in a few places and carries an ignore comment for each;
- a structural mismatch against a `Protocol`. `Grader`, `Retriever`, `Chunker`, `Reranker`,
  `VectorStore`, `Embedder`, and `Backend` are `runtime_checkable` Protocols: adding a method to one
  breaks every implementation in the tree and in user code, which makes it an API-breaking change
  and usually an ADR.

Coverage: one package-wide floor of 85 % declared in `pyproject.toml`, measured by CI over
`tests/unit`. `llm/`, `core/`, and `config/` are held to a higher bar by convention — a drop there
is a regression even though no per-module gate enforces it.

Snapshot tests (`syrupy`) exist for provider format mappings and token accounting. **A failing
snapshot is a signal, not noise.** It means a serialization you depend on changed shape. Read the
diff, decide whether the new shape is correct, and re-record deliberately — never regenerate the
snapshot to make the suite green.

One thing `make vcr-replay` does not tell you: a test whose cassette is absent skips rather than
fails, so a green run over a directory with no cassettes proves nothing. Check that the scenario you
care about actually has a recording under `tests/vcr/cassettes/<provider>/`.

---

## Definition of done

A change is done when all of these are true:

1. **The code follows the module's rules**, not just the root ones. Read
   `src/strata_forge/<module>/CLAUDE.md` before you edit and after you finish.
2. **Every new code path has a test** in the same PR, in `tests/unit/<module>/` unless it genuinely
   needs another tier.
3. **`docs/modules/<name>.md` matches the code.** A doc that lags the code is a defect, not a
   follow-up. Every symbol, signature, and error string you write into it must exist — grep first.
4. **`CHANGELOG.md` has an entry under `## [Unreleased]`** when the change is user-visible. Skip it
   for internal refactors that no caller can observe.
5. **An ADR accompanies an architectural change** — a new seam, a new dependency direction, a
   changed Protocol, a rejected alternative worth recording. Context → Decision → Consequences,
   under ~200 lines, next sequential number, immutable once merged.
6. **Rule files that your change made stale are updated in the same PR** — root `CLAUDE.md`, the
   module `CLAUDE.md`, `.cursor/rules/*.mdc`.
7. **The work is on a topic branch off `dev`**, named `<type>/<short-title>`, with a Conventional
   Commits title (`feat(llm): ...`) in time-stable language — no phase numbers, step numbers, or
   plan-file slugs anywhere in a branch name, commit, or PR title.
8. **`make check` and `make test` pass locally**, and `make test-all` too if you touched the public
   API. Never `--no-verify`.

The PR template encodes most of this as a checklist. It is there to be used, not deleted.

---

## Anti-patterns

Stated bluntly, because each of these has a tempting local justification and a bad global outcome.

- **Do not re-implement tool plumbing in `strata_forge.agents`.** `Tool`, `@tool`, the message
  types, and the tool loop belong to `strata_forge.llm`. The agents module composes them; it holds
  no tool abstraction, no message types, and no loop of its own (ADR 0011). It is also deliberately
  not built on PydanticAI — do not add that dependency.
- **Do not read environment variables outside `strata_forge.config`.** Add a field to a sub-model,
  document it in `.env.example`, and read it through `get_settings()`. The four existing direct
  reads are each a considered exception with a documented reason; new ones need a reason of the same
  shape, not convenience.
- **Do not add a model-provider SDK dependency to a higher-level module.** `evals`, `agents`, and
  anything above route model calls through `LLMClient`. If the typed layer will not express what you
  need, use `provider_extras={...}` — that escape hatch exists so the boundary does not have to
  break.
- **Do not add a top-level heavy import.** `import strata_forge.<anything>` must succeed on a bare
  install. One module-scope `import torch` breaks that for every user of every other module.
- **Do not silently truncate a prompt to fit a budget.** Raise. A quietly shortened prompt produces
  a plausible wrong answer and no way to detect it.
- **Do not commit an unscrubbed cassette.** If keys, AWS signatures, or bearer tokens reach a
  cassette, they are leaked the moment the branch is pushed and the credentials must be rotated.
- **Do not weaken a type, an assertion, a threshold, or a test to make a checker happy.** Casting to
  `Any`, loosening a signature, lowering a coverage floor, or deleting an assertion does not fix a
  failure — it hides one. If the checker is genuinely wrong, add the narrowest possible ignore with
  a reason on the same line.
- **Do not widen the sdist allow-list.** `[tool.hatch.build.targets.sdist] only-include` is what
  keeps `.env` and its live provider keys out of a permanent public archive.
- **Do not write a status hedge into a long-lived file.** No "coming soon", no "implementation
  pending", no phase or step numbers, no roadmap links. Describe what the code does today; the repo
  has no roadmap document and is not getting one.
- **Do not edit an ADR's reasoning.** Supersede it with a new one that links back.
- **Do not commit directly to `dev` or `main`.** `main` only ever receives `dev`, through a
  promotion PR, and CI enforces it.

---

## When the rules and the code disagree

This is §7 of the rules, restated as a procedure. It is the single most important habit in this
repository, because the docs and the rule files are read as ground truth by the next agent.

1. **Establish which is authoritative.** Merged code that works wins over a rule that describes
   something else: the rule is stale, and you fix the rule. In-flight work loses to the rule: the
   code is wrong, and you fix the code.
2. **Fix it in the same PR as the change that exposed it.** Not a follow-up issue. A stale rule left
   in place will be followed by something that cannot read the code.
3. **Fix it in the right file.** Cross-cutting convention goes in root `CLAUDE.md`; module-local
   rules go in `src/strata_forge/<module>/CLAUDE.md`; the glob-scoped restatements in
   `.cursor/rules/*.mdc` mirror both and must not drift from them. If a module rule contradicts the
   root file, resolve the contradiction rather than picking a side silently.
4. **Never edit an ADR's Decision or Consequences.** They record what was decided at a point in
   time. If reality moved, write a new ADR that supersedes the old one and links back to it.
5. **A pattern that has emerged in the code but is written down nowhere is also drift.** Propose
   adding it to the appropriate rule file rather than leaving the next reader to infer it.
6. **Silently ignoring a stale rule is the worst outcome available.** An agent that never read the
   rules does less damage than one that read them, noticed they were wrong, and left them wrong.

If you find something that is not a documentation defect but a security problem — a leaked
credential, a path traversal, an injection — do not open a public issue or a pull request describing
it. Report it privately to `gemovic@strataml.com`, as
[`SECURITY.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/SECURITY.md) sets out.
