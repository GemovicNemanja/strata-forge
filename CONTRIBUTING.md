# Contributing to strata-forge

Thanks for being here. strata-forge is a small library with a large surface, and the fastest way it
gets better is people using it for real work and pushing back on the parts that get in the way. A
bug report that says "the error message pointed me the wrong direction" is a genuine contribution;
so is a grader you wrote for your own evals, a chunker tuned to your documents, or a paragraph of
documentation that would have saved you an hour. You do not need to be an AI researcher to
contribute here — most of this codebase is ordinary typed async Python with strong opinions about
where things live.

Those opinions are the one thing worth reading before you write code. The conventions below are
strict, but they are strict in service of a single property: a stranger (or an agent) should be able
to open any file in `src/strata_forge/` and predict what it contains. This document is the human
version of the rules; the machine-readable version that coding agents load lives in
[`AGENTS.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CLAUDE.md).

**Contents**

- [Good first contributions](#good-first-contributions)
- [Getting set up](#getting-set-up)
- [The check loop](#the-check-loop)
- [Architecture rules](#architecture-rules)
- [Coding standards](#coding-standards)
- [Testing](#testing)
- [Commits and pull requests](#commits-and-pull-requests)
- [Branching and promotion](#branching-and-promotion)
- [Adding a new module](#adding-a-new-module)
- [When to write an ADR](#when-to-write-an-adr)
- [Documentation](#documentation)
- [Contributing with AI coding agents](#contributing-with-ai-coding-agents)
- [Reporting security issues](#reporting-security-issues)
- [Code of Conduct and license](#code-of-conduct-and-license)

---

## Good first contributions

Every one of these is small, self-contained, and lands in a single reviewable PR. Pick the one that
overlaps with what you were doing anyway — a contribution that came out of your own use is worth
more than one picked off a list.

**Write a grader.** `strata_forge.evals` takes any object satisfying the `Grader` Protocol
([`src/strata_forge/evals/graders/base.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/evals/graders/base.py)) —
a `name` property and an `async grade(*, item, response) -> GraderResult`. No subclassing, no
registration. The shipped set covers exact match, regex, JSON structure, LLM judge, pairwise
preference, and semantic similarity; anything domain-specific (citation presence, schema
conformance for your API, refusal detection) is missing and welcome.

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from strata_forge.evals import GraderResult

if TYPE_CHECKING:
    from strata_forge.datasets import DatasetItem
    from strata_forge.llm import LLMResponse


class WordBudget:
    """Passes when the response stays inside a word budget."""

    def __init__(self, *, max_words: int) -> None:
        self._max_words = max_words

    @property
    def name(self) -> str:
        return "word_budget"

    async def grade(self, *, item: DatasetItem, response: LLMResponse) -> GraderResult:
        words = len(response.text.split())
        within = words <= self._max_words
        return GraderResult(
            grader_name=self.name,
            score=1.0 if within else 0.0,
            passed=within,
            explanation=f"{words} words against a budget of {self._max_words}",
        )
```

`Grader` (importable from `strata_forge.evals`) is `runtime_checkable`, so
`isinstance(WordBudget(max_words=50), Grader)` is a real assertion you can put in the test.

**Write a chunker.** `strata_forge.rag` ships exactly one splitter, `RecursiveChunker`. The
`Chunker` Protocol is a single synchronous `chunk(document) -> Sequence[Chunk]`
([`src/strata_forge/rag/chunking.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/rag/chunking.py)).
A Markdown-heading-aware splitter, a token-budgeted splitter, or a code-aware splitter would all
slot in without touching the pipeline. Populate `character_start` / `character_end` in chunk
metadata the way `RecursiveChunker` does so callers can highlight the source span.

**Write a recipe.** [`docs/recipes/`](https://github.com/GemovicNemanja/strata-forge/tree/main/docs/recipes)
is for walkthroughs that cross module boundaries — the things the per-module reference pages
deliberately do not cover. "Index a document tree into Qdrant and query it", "run batch inference
against a self-hosted vLLM endpoint", "replay production traces against a candidate model". Write
the one you had to figure out yourself, and follow the shape of the recipe already there.

**Record a provider cassette.** `tests/vcr/` replays recorded HTTP exchanges so CI exercises real
provider wire formats without live keys. Several provider and scenario slots are still empty. If
you have credentials for a provider we do not cover, recording its cassettes is high-value work —
read [Testing](#testing) first, particularly the part about scrubbing.

**Add an example.** [`examples/`](https://github.com/GemovicNemanja/strata-forge/tree/main/examples)
scripts are runnable, self-contained, and environment-gated (they print a clear skip message when
credentials are absent). They double as an import-level smoke test of the public API via
`tests/e2e/test_example_imports.py`, so an example that drifts from the library fails a full
`uv run pytest` loudly.

**Fix documentation.** If a docstring, module guide, or rule file disagrees with the code, that is a
bug. See [Documentation](#documentation) for which one wins.

---

## Getting set up

You need **Python 3.14 or newer** (the repo pins `3.14` in `.python-version`) and
[uv](https://docs.astral.sh/uv/). uv can supply the interpreter itself:

```bash
git clone https://github.com/GemovicNemanja/strata-forge.git
cd strata-forge
uv python install 3.14
make install          # runs `uv sync` — creates .venv from the lockfile
uv run pre-commit install
```

`make install` is exactly `uv sync`: base dependencies plus the `dev` dependency group (ruff,
pyright, pytest and friends). That is enough to run the entire unit suite — module tests inject
fakes for heavy optional dependencies rather than requiring them.

Copy the environment template and fill in whatever you actually need:

```bash
cp .env.example .env
```

`.env` is git-ignored and is read by `strata_forge.config`. Nothing in the base install requires a
key; you only need provider credentials for the tests and examples that talk to a provider.

### Optional extras

Heavy dependencies live behind extras so the base install stays small. Install only the ones your
change touches:

```bash
uv sync --extra rag --extra langfuse
```

| Extra | Pulls in | Needed for |
|---|---|---|
| `langfuse` | `langfuse` | Tracing, the Langfuse prompt and dataset stores |
| `hf` | `datasets` | The Hugging Face dataset bridge |
| `evals` | `nltk`, `rouge-score` | BLEU and ROUGE metrics |
| `rag` | `qdrant-client`, `cohere` | Qdrant vector store, Cohere reranking |
| `redis` | `redis` | The Redis LLM response cache backend |
| `multimodal` | `pillow` | Image downscaling for multimodal input |
| `storage` | `fsspec`, `s3fs`, `gcsfs`, `adlfs`, `huggingface_hub` | The storage gateway and Hub client |
| `compute` | `skypilot`, `asyncssh` | The SSH and SkyPilot backends |
| `bedrock` | `boto3` | The Bedrock provider route |
| `finetuning` | `torch`, `transformers`, `trl`, `peft`, `accelerate`, `datasets` | Running a real training job |
| `serving` | `vllm` | Running a vLLM server on the same host as your code |

Two notes on that table. `serving` is the odd one out: no module imports `vllm`.
`strata_forge.compute.serving` builds the shell command that launches a server, so you need the
extra only if that server runs on your own machine. And `rag` also pins `rank-bm25`, which nothing
imports — `BM25Retriever` is a self-contained pure-Python implementation.

`pyproject.toml` is the ground truth for this list — check it if the table looks stale, and fix the
table in the same PR.

### Local services

```bash
make stack-up      # docker compose up -d
make stack-logs    # tail the stack
make stack-down    # tear it down
```

[`docker/compose.yaml`](https://github.com/GemovicNemanja/strata-forge/blob/main/docker/compose.yaml)
brings up **Postgres, Qdrant, and Redis** on their default ports. Langfuse is not part of the local
stack: point `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` at a Langfuse instance
you control or at Langfuse Cloud. Every Langfuse-touching code path degrades to a silent no-op when
those keys are absent, so you can develop most of the library without one.

`make doctor` runs `strata-forge doctor`, which prints resolved settings, tracked package versions,
and TCP reachability for Langfuse, Redis, and Qdrant. It is a diagnostic, not a gate — it always
exits 0.

---

## The check loop

Run this before you push. It is the same set of checks CI runs, minus the parts that need
credentials.

```bash
make fmt        # ruff format
make lint       # ruff check
make type       # pyright (strict on src/strata_forge)
make check      # lint + type
make test       # pytest tests/unit
make test-cov   # pytest tests/unit with a term-missing coverage report
```

| Target | What it does |
|---|---|
| `make install` | `uv sync` |
| `make fmt` | `ruff format` over `src tests examples scripts` |
| `make lint` | `ruff check` over the same directories |
| `make type` | `pyright` over `src` and `tests` |
| `make check` | `lint` then `type` |
| `make test` | Unit tests only — no network, no services |
| `make test-cov` | Unit tests with coverage and missing-line output |
| `make integration` | `pytest -m integration` (see [Testing](#testing)) |
| `make vcr-replay` | Replays committed cassettes; no keys needed |
| `make vcr-record` | Re-records cassettes; **needs live keys**, sets `RECORD=1` for you |
| `make refresh-cassettes` | Alias for `vcr-record` |
| `make doctor` | `strata-forge doctor` |
| `make stack-up` / `stack-down` / `stack-logs` | The Docker services |
| `make clean` | Removes caches and build artifacts |

Two things the make targets do not cover:

- **`notebooks/` is outside `PY_DIRS`.** `make fmt` and `make lint` skip it, while CI runs a bare
  `ruff check` and `ruff format --check` across the whole tree. If you touch a notebook, run
  `uv run ruff format notebooks && uv run ruff check notebooks` explicitly.
- **CI's test job runs `tests/unit` only.** The end-to-end import smoke tests (`tests/e2e/`) and the
  unmarked cross-module tests under `tests/integration/` run only under a bare `uv run pytest`. Run
  that at least once before opening a PR that touches `examples/`, `notebooks/`, or a public API
  signature.

The pre-commit hooks (ruff with `--fix`, ruff-format, whitespace and YAML/TOML checks, and pyright)
run on every commit once you have installed them. Do not bypass them with `--no-verify`; if a hook
is wrong, fix the hook.

---

## Architecture rules

These are the rules a reviewer will hold your PR to. Code that breaks one of them is wrong by
default. If a violation is genuinely necessary, it needs an
[ADR](#when-to-write-an-adr) before it merges — not a comment explaining the exception.

The long-form rationale for each lives in
[the architecture overview](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/overview.md)
and the numbered ADRs.

### Async-first public API

Public functions that do I/O are `async`. Internal call chains stay async end to end — never mix
`requests` or a synchronous `httpx` call into an async path. HTTP goes through `httpx.AsyncClient`,
usually via LiteLLM.

Synchronous facades live in **one place only**: `strata_forge.sync`, where each wrapper is a thin
`asyncio.run` over its async counterpart. Do not add a `foo_sync()` next to `foo()` anywhere else.
Not every async surface has a facade, and that is fine — `strata_forge.sync` exists for CLI and
notebook ergonomics, not for API symmetry.

Purely computational helpers (`content_hash`, `count_tokens`, `render`, chunking) are synchronous
and should stay that way. `async def` on a function that never awaits buys nothing.

### Strict typing

pyright runs in `strict` mode over `src/strata_forge/`. New code must be strict-clean before it
merges. Tests are typed but not held to strict.

- **Pydantic v2 models** for data. Frozen (`model_config = ConfigDict(frozen=True, extra="forbid")`)
  unless mutation is the point.
- **`Protocol`** for behaviour. The pluggable seams in this library — graders, chunkers, retrievers,
  vector stores, rerankers, stores, compute backends — are all Protocols, so a user's class works
  without importing a base class from us. Mark them `runtime_checkable` when tests want
  `isinstance`.
- **`TypedDict`** for narrow boundaries such as provider payload fragments.
- **No bare `Any`.** Where it is unavoidable (an untyped SDK, a lazily imported module), silence the
  specific rule on that line with a justified `# pyright: ignore[reportXxx]`.
- **`from __future__ import annotations`** at the top of any module using generics in signatures.
- Type-only imports belong in an `if TYPE_CHECKING:` block — ruff's `TCH` rules enforce this.

### Errors

Everything raised from library code inherits from `strata_forge.core.errors.ForgeError`. A caller
should be able to write one `except ForgeError` and catch anything the library considers its own
fault domain, then pattern-match subclasses for finer control.

- Provider exceptions are normalized **at the seam**:
  `strata_forge.llm.errors.map_litellm_exception`
  turns a LiteLLM exception into a `ProviderError` subclass (`ProviderRateLimitError`,
  `ProviderAuthError`, `ProviderTimeoutError`, `ProviderContentFilterError`, and so on). Nothing
  above `strata_forge.llm` should ever catch a raw provider exception type.
- Never write a bare `except:`. Never swallow an exception to keep going.
- Argument validation raises `ValidationError` (or a builtin `ValueError` / `TypeError` for plain
  signature misuse) — never silently coerce a bad value into a plausible one.
- Retries are declared, not hand-rolled: the `@retry` decorator from `strata_forge.core` drives
  control flow, and its predicates select on `ProviderError` subclasses.

### Lazy heavy imports and extras

`import strata_forge.<module>` must succeed on a base install with no extras. That invariant is what
lets `strata_forge.cli` import every subcommand at startup.

So: a heavy dependency is imported **inside the function that uses it**, never at module load, and
the function raises `ImportError` with the exact install command when it is missing:

```python
def _import_fsspec(self) -> Any:
    try:
        return __import__("fsspec")
    except ImportError as exc:
        msg = (
            "The [storage] extra is required for StorageGateway. "
            "Install it with: pip install 'strata-forge[storage]'."
        )
        raise ImportError(msg) from exc
```

That is the shape used throughout the codebase. `__import__` returning `Any` is deliberate: these
SDKs are untyped, and funnelling them through one annotated accessor keeps the rest of the module
strict-clean instead of scattering ignores.

Get the distribution name right — it is `strata-forge`, and that message is what a stuck user pastes
into a terminal. If you add a heavy dependency, add it to an extra in `pyproject.toml` and write the
import guard in the same PR.

### Module boundaries

The dependency graph is a DAG and the direction is documented in
[`docs/architecture/module-boundaries.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/module-boundaries.md).
The load-bearing parts:

- `strata_forge.core` is strictly upstream. It imports from no other `strata_forge.*` module, ever.
- `strata_forge.config` is the only place that defines settings. Do not read environment variables
  ad hoc elsewhere — add a field to the relevant sub-model and consume `get_settings()`.
- **Chat-completion provider SDKs (`anthropic`, `openai`, `boto3`, Google's clients) appear only
  under `src/strata_forge/llm/providers/`.** Higher-level modules — `evals`, `agents`, `rag`,
  `datasets` — make every model call through `strata_forge.llm`. Where a module legitimately needs a
  non-chat vendor SDK (Qdrant for vectors, Cohere for reranking, LiteLLM directly for embeddings),
  that seam is deliberate and recorded in an ADR; do not open a new one without writing one.
- `strata_forge.cli` is a sink: it may import anything, nothing imports it, and every subcommand
  is a thin wrapper over the module underneath.

### Composability and escape hatches

Each module exposes a small core API plus a way out.
`LLMClient.complete(..., provider_extras={...})` forwards verbatim to LiteLLM; backends accept
explicit clients for injection; stores take a pre-built SDK client. Never design an API that leaves
a user stuck because we did not anticipate their provider's newest parameter. The escape hatch is a
feature.

### Observability, cost, reproducibility

- Log through `strata_forge.core.logging.get_logger`. The correlation id propagates across `await`
  boundaries via a contextvar (`correlation_id`) — never thread it through call signatures by hand.
- Never log full prompts at `INFO` (use `DEBUG`), and never log API keys, AWS signatures, or session
  tokens at any level.
- LLM-touching code respects the active `BudgetContext`. When a call would exceed the ceiling, raise
  `BudgetExceededError` — do not quietly truncate the prompt to fit.
- Pseudo-randomness goes through `strata_forge.core.repro.set_seed`, never `random.seed` directly.
  Long-running operations capture `env_snapshot()` into their run metadata, and dataset content is
  identified by `content_hash()`.

---

## Coding standards

Most of this is enforced by ruff and pyright; the rest is enforced at review.

- **Line length 100.** ruff is the formatter and the linter; there is one config, in
  `pyproject.toml`. Do not add per-file formatter directives.
- **Naming.** `snake_case` for functions, variables, and modules. `PascalCase` for classes.
  `SCREAMING_SNAKE` for constants. Pydantic models follow `XxxConfig`, `XxxRequest`, `XxxResponse`,
  `XxxError`.
- **Imports.** ruff's isort profile, first-party is `strata_forge`. Avoid star re-exports; declare
  an explicit `__all__`.
- **Docstrings.** Module-level summary is mandatory. Public functions and classes get a one-line
  summary, and more only when the *why* is non-obvious. Long-form explanation belongs in
  `docs/modules/<name>.md`, not in a docstring.
- **Docstrings ship to PyPI.** Anything under `src/` lands in a public sdist, so write for an
  outside reader: no internal infrastructure, no private repositories, no "when X lands" framing.
- **Comments are rare.** Explain *why*, never *what* — the name already says what. Never reference a
  task, a PR, or planning scaffolding in a comment.
- **No emojis** anywhere in source or documentation.
- **No planning vocabulary in long-lived files.** Phase numbers, step numbers, and plan slugs do not
  belong in source, documentation, commit messages, branch names, or PR titles. If you need to
  describe maturity, describe what the code does today.

---

## Testing

**Every new code path gets a test in the same PR.** Not a follow-up, not a tracking issue — the
same PR.

### Layout

| Directory | What lives there | When it runs |
|---|---|---|
| `tests/unit/` | Fast, offline tests mirroring `src/strata_forge/` package by package | `make test`, every CI run |
| `tests/integration/` | Cross-module tests plus the service-backed ones | Cross-module tests run under a bare `uv run pytest`; the marked ones under `make integration` |
| `tests/e2e/` | Import-level smoke tests for `examples/` and `notebooks/` | Bare `uv run pytest` |
| `tests/vcr/` | Provider wire-format tests replaying recorded cassettes | `make vcr-replay`, CI |

**Unit tests must not touch the network.** Fake the provider — `respx` for HTTP, `AsyncMock` for an
injected `LLMClient`, or a `sys.modules`-injected stub for a heavy SDK. That last pattern is how the
compute, training, and rag suites test code paths whose dependencies are not installed in the dev
environment; copy the shape from the nearest existing test.

### Markers

Four markers are declared in `pyproject.toml`:

| Marker | Meaning |
|---|---|
| `live` | Needs live API keys; skipped by default |
| `redis` | Needs a running Redis (`make stack-up`) |
| `slow` | Long-running |
| `integration` | Needs an external service |

`--strict-markers` is on, so an undeclared marker is a hard error. A new marker goes into
`pyproject.toml` in the same PR that first uses it.

Today only `integration` is used, and it selects the two Langfuse-backed tests, which skip
themselves unless `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. The other cross-module
tests under `tests/integration/` are deliberately unmarked because they need nothing external.

Do not commit a skipped test without a tracking issue link in the skip reason.

### Coverage

The enforced gate is repo-wide: `fail_under = 85` in `[tool.coverage.report]`. On top of that,
reviewers hold two module-level expectations that are conventions rather than automated checks:

- `src/strata_forge/llm/` at 90 % line coverage or better
- `src/strata_forge/core/` and `src/strata_forge/config/` at 85 % or better

Run `make test-cov` and read the missing-line output for the files you touched. A PR that drops
coverage on the module it edits needs a sentence in the description explaining why.

### Property and snapshot tests

`hypothesis` is in the dev group and is used where invariants matter more than examples — token
counting, cache-key canonicalization, metric aggregation, prompt rendering. Reach for it when a
function has an algebraic property ("reordering the input does not change the hash") rather than a
handful of interesting cases.

`syrupy` is available for snapshot tests and is the right tool for provider format mappings, where
the assertion is "this payload shape did not move". If you add one, treat a failing snapshot as an
early warning that an upstream wire format changed — investigate before you regenerate.

### VCR cassettes

`tests/vcr/` replays recorded provider exchanges so CI exercises real wire formats without keys.
Cassettes live at `tests/vcr/cassettes/<provider>/<scenario>.yaml`; a test skips gracefully when its
cassette is absent.

```bash
make vcr-replay    # record_mode="none" — fails rather than reaching the network
make vcr-record    # sets RECORD=1; needs live credentials for the providers you cover
```

Scrubbing happens in `before_record_request` in `tests/vcr/conftest.py`: every auth-bearing header
(`authorization`, `x-api-key`, `anthropic-api-key`, the AWS SigV4 set, and more) and sensitive query
parameter is replaced with `[REDACTED]` before the cassette is written.

**If you record without scrubbing, treat the exposed credentials as leaked and rotate them
immediately.** After recording, read the produced YAML before you commit it — look for
model-specific headers, request ids, account identifiers, or anything else you do not recognize.
A cassette is a permanent, public artifact once it is pushed.

---

## Commits and pull requests

### Conventional Commits

Every commit subject follows:

```
<type>(<optional-scope>): <imperative summary>
```

One line, lowercase, imperative mood, under 70 characters. `<type>` is one of:

| Type | Use when | Example |
|---|---|---|
| `feat` | A user-visible feature is added | `feat(llm): add structured-output reprompt fallback` |
| `fix` | A bug in existing behaviour is corrected | `fix(cache): handle Redis timeout without crashing` |
| `docs` | Documentation only — READMEs, ADRs, module guides, docstrings | `docs(rag): document the BM25 retriever options` |
| `style` | Whitespace or formatting; no behaviour change | `style: apply ruff format to recent edits` |
| `refactor` | Neither fixes a bug nor adds a feature | `refactor(llm): split client.py into client + tool_loop` |
| `perf` | A measurable performance improvement | `perf(tokens): cache tiktoken encoders across calls` |
| `test` | Adds or corrects tests; no production change | `test(fallback): cover content-filter short-circuit` |
| `build` | Packaging, dependencies, build backend | `build: bump pydantic to >=2.11` |
| `ci` | CI configuration or workflows | `ci: add nightly cassette-refresh job` |
| `chore` | Routine housekeeping not covered above | `chore: prune unused make targets` |
| `revert` | Undoes a previous commit; reference it in the body | `revert: feat(llm) — add structured-output reprompt fallback` |

**Scope** is optional and parenthesized. Use the module name (`core`, `config`, `llm`, `prompts`,
`tracing`, `datasets`, `evals`, `agents`, `rag`, `storage`, `compute`, `training`, `pipelines`,
`cli`) or a focused sub-area inside one. Omit it when the change genuinely spans the repo — root
tooling, a multi-module refactor, cross-cutting documentation.

**Body and footer.** Optional body wrapped at about 72 characters, explaining the *why* and any
non-obvious *how*; bullets are fine. Reference issues in a trailer (`Closes #123`). Breaking changes
go in a `BREAKING CHANGE:` trailer or use the `!` marker (`feat(llm)!: replace complete signature`).

**Time-stable language.** Commit messages, branch names, and PR titles describe the change in terms
that still make sense in a year. No phase numbers, no step numbers, no plan-file slugs, no
references to the process that produced the change. The git log outlives the planning.

**Other rules.** One commit per cohesive unit of work — never bundle unrelated changes for
convenience. Never `--no-verify`. Never force-push to `dev` or `main`. Never `--amend` a commit that
has already been pushed somewhere shared.

### Pull requests

PR titles follow the same Conventional Commits format as a commit subject. The body has three
sections, which GitHub pre-fills from the pull request template:

```markdown
## Summary
What changed, in two or three sentences. Describe the end state, not the journey.

## Why
The problem this solves, or the behaviour that was wrong.

## Test plan
The commands you ran and what you observed. Include the failing case if this is a fix.
```

A reviewer should be able to read the Test plan and reproduce your confidence without guessing.
"CI is green" is not a test plan. The template also carries a short checklist — work through it
rather than ticking it blind; each line maps to a rule in this document.

---

## Branching and promotion

Three tiers. Every change follows them; nothing is committed directly to `dev` or `main`.

1. **Topic branch off `dev`**, named `<type>/<short-title>` using the same type vocabulary as
   commits — `feat/streaming-tool-loop`, `fix/redis-timeout`, `docs/rag-retriever-options`. One
   cohesive unit of work per branch.
2. **Open a PR into `dev`** when it is ready. `dev` is the persistent integration branch and is kept
   green.
3. **Delete the topic branch once it merges** — local and remote:
   `git branch -d <branch> && git push origin --delete <branch>`. `dev` and `main` are the only
   persistent branches; a merged topic branch never lingers on the remote.
4. **Promotion to `main` happens only through a `dev` → `main` PR.**

`main` only ever receives `dev`. This is enforced rather than trusted: `main` is protected against
direct pushes, and the `promotion-guard` workflow fails any PR into `main` whose source branch is
not `dev`. A tag of the form `vX.Y.Z` on `main` triggers the release workflow, which verifies that
the tag matches the packaged version and publishes to PyPI via Trusted Publishing.

Cutting a release bumps two separate strings that will drift if you forget either: `version` in
`pyproject.toml` and `__version__` in `src/strata_forge/__init__.py`.

---

## Adding a new module

A new top-level module is a large change; open an issue and agree on the seam before you build it.
Once you do:

1. Create `src/strata_forge/<module>/` with `__init__.py` (module docstring plus an explicit
   `__all__`), `README.md` (what it does and why, in product terms), and `CLAUDE.md` (rules specific
   to the module, for both humans and agents).
2. Add a row to the project map in
   [`CLAUDE.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CLAUDE.md) and to the
   module table in the root `README.md`.
3. Add `docs/modules/<module>.md` — the public API reference.
4. Add a row to
   [`docs/architecture/module-boundaries.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/module-boundaries.md)
   stating what the module owns, what it may import, and what it explicitly does not do.
5. If it introduces heavy dependencies, add an extra in `pyproject.toml` and gate every import
   lazily inside the function that uses it.
6. Add unit tests under `tests/unit/<module>/`.
7. If it touches provider APIs directly, add cassette coverage under `tests/vcr/`.
8. If the design is architecturally significant, write an ADR.
9. Open the PR per the rules above.

---

## When to write an ADR

Write an Architecture Decision Record when a change alters a load-bearing seam: a new module, a
dependency direction, a public Protocol, a vendor boundary, or a deliberate exception to one of the
rules in this document. If a reviewer would reasonably ask "why is it like this?" six months from
now, the answer belongs in an ADR rather than a code comment.

ADRs live in
[`docs/architecture/adr/`](https://github.com/GemovicNemanja/strata-forge/tree/main/docs/architecture/adr),
numbered sequentially as `NNNN-short-slug.md`. The template is **Context → Decision →
Consequences**, ideally under 200 lines. Record the alternatives you rejected and why — that is
usually the most valuable part to a later reader.

**A merged ADR is immutable as to its reasoning and conclusions.** If a decision changes, write a
new ADR that supersedes it and cross-link both. Correcting a typo or repairing a dead link in an old
ADR is fine; rewriting its decision is not.

---

## Documentation

Documentation is part of the change, not a follow-up.

- Public API change means the matching `docs/modules/<name>.md` changes in the **same PR**.
- User-facing change means the root `README.md` changes too.
- Module `README.md` files describe what the module does in product terms. They ship inside the
  installed wheel, so they must not link to files that only exist in the repository.
- **Describe what exists today.** No status hedging, no "coming soon", no phase numbers, no
  roadmaps. If something is genuinely absent, say so plainly in the present tense and stop there.
- **Link policy.** Between files inside `docs/`, use relative `.md` links so they work both on
  GitHub and in the rendered documentation site. To anything outside `docs/` — `src/`, `examples/`,
  root-level files — use an absolute `https://github.com/GemovicNemanja/strata-forge/blob/main/...`
  URL, because the site generator cannot resolve paths above its docs directory.
- **Every code snippet must run.** Verify each symbol against the source before you write it. A
  snippet that raises `ImportError` or `TypeError` on paste is worse than no snippet at all. Async
  examples need `asyncio.run` or an explicit await context.

The docs index is at
[`docs/README.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/README.md).

**When documentation and code disagree:** if the code is already merged, the code wins and you fix
the documentation. If the change is still in flight, the documented rule wins and you fix the code.
Either way, do not leave the contradiction standing — a stale rule is worse than no rule, because
the next reader will follow it.

---

## Contributing with AI coding agents

Agent-assisted contributions are welcome, and this repository is deliberately set up for them.

- [`AGENTS.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CLAUDE.md) (a symlink to
  `CLAUDE.md`) is the root instruction file. Claude Code, Cursor, Codex, and Copilot all read one of
  those two names, so they get identical rules.
- Module-specific rules live in `src/strata_forge/<module>/CLAUDE.md` and are authoritative for that
  module.
- Glob-scoped reinforcement for Cursor's rule engine lives in `.cursor/rules/*.mdc`.
- [`docs/agent-guide.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/agent-guide.md)
  is the orientation page: where to look first, which file answers which question, and how to
  navigate the codebase efficiently.

Two expectations if you work this way:

**You are the author of record.** Read every line before you open the PR. Agents are confidently
wrong about API signatures in particular — if a snippet or a call was generated, run it. The
"verify every symbol" rule in [Documentation](#documentation) exists because generated
documentation is the most common source of code that does not run.

**Fix stale rules rather than working around them.** If an agent (or you) notices that a rule file
no longer matches the code — a renamed symbol, a changed signature, a convention that has quietly
shifted — update the rule file in the same PR as the code change. Silently ignoring a stale rule is
the one failure mode that compounds: it teaches the next agent to ignore the rules too.

---

## Reporting security issues

**Never report a vulnerability in a public issue, discussion, or pull request.** Use GitHub's
private vulnerability reporting, or email **gemovic@strataml.com** if you cannot. The full policy —
what to include, what response times to expect, and the disclosure process — is in
[SECURITY.md](https://github.com/GemovicNemanja/strata-forge/blob/main/SECURITY.md).

The same address handles Code of Conduct reports.

If you believe credentials have been exposed — in a cassette, a log, an example, or a test
fixture — treat them as compromised, rotate them, and say so in the report. Rotation first, cleanup
second. Never put a live key in a report.

---

## Code of Conduct and license

Participation is governed by the
[Contributor Covenant Code of Conduct](https://github.com/GemovicNemanja/strata-forge/blob/main/CODE_OF_CONDUCT.md).
Read it before your first interaction; it is short.

strata-forge is licensed under
[Apache License 2.0](https://github.com/GemovicNemanja/strata-forge/blob/main/LICENSE). By
contributing, you agree that your contributions are licensed under the same terms. There is no CLA.
