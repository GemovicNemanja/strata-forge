# strata-forge

[![PyPI version](https://img.shields.io/pypi/v/strata-forge.svg)](https://pypi.org/project/strata-forge/)
[![Python versions](https://img.shields.io/pypi/pyversions/strata-forge.svg)](https://pypi.org/project/strata-forge/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/GemovicNemanja/strata-forge/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/GemovicNemanja/strata-forge/ci.yml?branch=main&label=CI)](https://github.com/GemovicNemanja/strata-forge/actions/workflows/ci.yml)

Typed, async-first Python library for the whole arc of an LLM experiment: calling a model,
authoring the prompt, tracing the call, building the dataset, grading the output, running the
agent, retrieving the context, fine-tuning the weights, and shipping the job to a remote machine.
Built on [LiteLLM](https://github.com/BerriAI/litellm) for provider breadth and
[Pydantic v2](https://docs.pydantic.dev/) for strict typing, with heavy dependencies behind extras
so the base install stays small.

This is a pre-1.0 release: the public API may change in any release until `1.0.0`, and every
breaking change is called out in the entry that introduces it in the
[changelog](https://github.com/GemovicNemanja/strata-forge/blob/main/CHANGELOG.md). Pin a version
if you depend on it.

## Why this exists

Most libraries in this space are frameworks: they own your control flow, and you write callbacks
and config that they call back into. That trade is worth it right up until the moment your problem
stops looking like the framework's example, and then you are fighting an abstraction to get at an
HTTP request you could have written yourself.

strata-forge is the other trade. It is a set of primitives you call, not a runtime that calls you.
The seams are deliberate:

- **Vendor-neutral at the transport seam.** One `LLMClient` over LiteLLM covers Anthropic, OpenAI,
  Google Vertex, AWS Bedrock, Azure OpenAI, and any OpenAI-compatible endpoint. Provider SDKs stay
  inside `llm/providers/`. Switching providers is a string, and failing over between two is a list.
- **Strict about types and errors.** Every data shape is a frozen Pydantic model, every behavioural
  seam is a `Protocol`, and pyright runs in strict mode over the whole package. Failures raise from
  one `ForgeError` hierarchy rather than returning `None` or a sentinel; provider exceptions are
  normalised into that hierarchy at the seam so retry and fallback can select on them.
- **Async end to end.** Public I/O is `async`, so batching a thousand calls is `asyncio.gather` and
  not a thread pool. Blocking facades for the four most common calls live in one module,
  `strata_forge.sync`, so there is never a second code path to keep in step.
- **Modular, and honest about its edges.** `import strata_forge.rag` works without Qdrant
  installed; the `ImportError` arrives when you first use the thing that needs it, and it names the
  extra to install. The escape hatch down to the transport is a feature rather than a leak —
  `provider_extras={"anthropic": {...}}` is forwarded verbatim to LiteLLM — because the library
  being in the way is worse than the library being incomplete.

It is a baseline to build on, not a product to adopt. If you want an opinionated agent framework
with a graph runtime and a hosted control plane, this is not that.

## Contents

- [Why this exists](#why-this-exists)
- [Install](#install)
- [Quickstart](#quickstart)
- [Modules](#modules)
- [Repository map](#repository-map)
- [Documentation](#documentation)
- [Development](#development)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)
- [Acknowledgements](#acknowledgements)

Beyond this page:

- [Documentation site](https://gemovicnemanja.github.io/strata-forge/) — everything under `docs/`,
  rendered and searchable.
- [Module reference](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/README.md)
  — one page per module, the canonical description of each public API.
- [Recipes](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/recipes/README.md) —
  end-to-end walkthroughs that cross module boundaries.
- [Architecture overview](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/overview.md)
  and [decision records](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/adr/README.md)
  — how the layers fit together, and why each seam is where it is.
- [Agent guide](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/agent-guide.md) — the
  orientation path for a coding agent working in this repository.
- [Contributing](https://github.com/GemovicNemanja/strata-forge/blob/main/CONTRIBUTING.md),
  [security policy](https://github.com/GemovicNemanja/strata-forge/blob/main/SECURITY.md),
  [changelog](https://github.com/GemovicNemanja/strata-forge/blob/main/CHANGELOG.md).

## Install

```bash
pip install strata-forge
```

Requires **Python 3.14 or newer**. The floor is deliberate, not incidental: the package is written
against the PEP 649 deferred-annotation semantics that became the interpreter default in 3.14, and
uses PEP 695 type-parameter syntax throughout (`class StructuredResponse[M: BaseModel]`,
`type AnyMessage = ...`); ruff and pyright are both pinned to `py314`. There is no compatibility
shim for older interpreters, so on 3.13 and below the install will not resolve. The distribution is
`strata-forge`; the import package is `strata_forge`. Installing it also puts a `strata-forge`
console script on your `PATH` — start with `strata-forge doctor`, which prints the resolved settings
and probes whatever services you have configured.

The base install is enough for prompting, model calls, structured output, tool calling, streaming,
fallback, in-process caching, datasets, evals, agents, in-process retrieval, and the CLI. Anything
with a heavy or optional dependency lives behind an extra:

| Extra | Adds | Unlocks |
|---|---|---|
| `langfuse` | `langfuse` | `strata_forge.tracing`, the Langfuse-backed prompt and dataset stores, and trace replay in `strata_forge.evals` |
| `redis` | `redis` | `RedisCache`, the cross-process response cache. `InMemoryCache` needs nothing |
| `multimodal` | `pillow` | `downscale_image`. Sending images with `ImageContent` works on the base install |
| `hf` | `datasets` | The Hugging Face Datasets bridge (`to_hf_dataset` / `from_hf_dataset`) and dataset loading in `strata_forge.pipelines` |
| `evals` | `nltk`, `rouge-score` | The `bleu` and `rouge` metrics. Every grader and every other metric is dependency-free |
| `rag` | `qdrant-client`, `cohere` | `QdrantVectorStore` and `CohereReranker`. Chunking, dense retrieval, BM25, and hybrid RRF are pure Python |
| `storage` | `fsspec`, `s3fs`, `gcsfs`, `adlfs`, `huggingface_hub` | `StorageGateway` over S3 / GCS / Azure Blob, and `HFHubClient` for Hub push and pull |
| `bedrock` | `boto3` | The AWS Bedrock provider route |
| `compute` | `skypilot`, `asyncssh` | `SkyPilotBackend` and `SSHBackend`. `LocalBackend` needs nothing |
| `serving` | `vllm` | vLLM on the machine that will host the server. The library only builds the launch command, so you need this where the server runs, not where you build the task |
| `finetuning` | `torch`, `transformers`, `trl`, `peft`, `accelerate`, `datasets` | `SFTRunner` and `PreferenceRunner` — everything in `strata_forge.training` past config construction |
| `all` | every extra at once | Convenience for a scratch environment; not what you want in production |

```bash
pip install "strata-forge[langfuse,rag]"
```

One further extra, `inspect`, installs `inspect-ai` for use alongside the library; no strata-forge
module imports it. One dependency deliberately has no extra at all: `CrossEncoderReranker` needs
`sentence-transformers`, which pulls in torch, so `[rag]` leaves it out and you install it yourself
when you want a local cross-encoder rather than the hosted `CohereReranker`. The authoritative list,
with version pins, is `[project.optional-dependencies]` in
[`pyproject.toml`](https://github.com/GemovicNemanja/strata-forge/blob/main/pyproject.toml).

## Quickstart

Provider credentials are read from the process environment — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`AZURE_OPENAI_API_KEY`, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`,
`GOOGLE_APPLICATION_CREDENTIALS`. Nothing is passed in code unless you want it to be. Library
settings (Langfuse, Redis, Qdrant, logging, diagnostics) additionally read a `.env` file in the
working directory, and `strata_forge.config.load_env_file()` merges one into `os.environ` when you
want your provider keys picked up from there too.
[`.env.example`](https://github.com/GemovicNemanja/strata-forge/blob/main/.env.example) lists the
variables the library reads.

A model name is a logical name from the built-in registry, which resolves it to a provider route,
a provider-specific model id, and a pricing table. The registry is a closed, curated list of ten
models — `claude-opus-4-8`, `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5`, `gpt-5.5`,
`gpt-5.5-pro`, `gpt-5.5-thinking`, `gpt-5.5-instant`, `gemini-3.1-pro`, `gemini-3.1-flash-lite`,
plus aliases like `opus` and `sonnet` — and any other name raises `RegistryError` on the first
call, however valid it is in the provider's own documentation. Anything outside that list goes
through the `openai_compat` route, which skips the registry and forwards the model id verbatim:
`LLMClient("llama-3.3-70b", provider="openai_compat")` reaches Ollama, Groq, OpenRouter, a vLLM
deployment, or any other OpenAI-shaped endpoint, with `FORGE_OPENAI_COMPAT_BASE_URL` pointing at it.
The full table, with routes and prices, is in the
[module reference](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/llm.md#model-registry).

```python
import asyncio

from strata_forge.llm import LLMClient, Message


async def main() -> None:
    client = LLMClient("claude-opus-4-8")
    response = await client.complete([Message.user("Name three uses for a paperclip.")])

    print(response.text)
    print(response.route.provider, response.usage.total_tokens, f"${response.cost_usd:.5f}")


asyncio.run(main())
```

Now something the base install is genuinely good at: a validated Pydantic result, a fallback chain
across two providers and then a second model, and a hard ceiling on what the block is allowed to
spend.

```python
import asyncio

from pydantic import BaseModel

from strata_forge.core import BudgetContext, BudgetExceededError
from strata_forge.llm import LLMClient, Message, ModelFallback


class Incident(BaseModel):
    summary: str
    severity: int
    components: list[str]


async def main() -> None:
    # Two axes of failover: Anthropic then Bedrock for the same model, then a different
    # model entirely. Rate limits, timeouts, 5xx, and auth failures advance the chain; a
    # content-filter refusal short-circuits it, because retrying it elsewhere is not a fix.
    client = LLMClient(
        chain=[
            ModelFallback("claude-opus-4-8", ("anthropic", "bedrock")),
            "gpt-5.5",
        ]
    )

    async with BudgetContext(max_usd=0.10) as budget:
        try:
            response = await client.complete_structured(
                [Message.user("Triage: the checkout service is returning 503s for EU traffic.")],
                schema=Incident,
            )
        except BudgetExceededError as exc:
            print(f"stopped at ${exc.limit_usd}")
            return

    incident = response.parsed  # typed as Incident, already validated
    print(incident.severity, incident.components)
    print(f"spent ${budget.spent_usd:.5f} of ${budget.max_usd}")


asyncio.run(main())
```

`complete_structured` picks the right mechanism per provider — native `response_format` on OpenAI,
forced-tool emulation on Anthropic, `response_schema` on Gemini — and reprompts with the parse
error when the model returns something that fails validation. The constraint on the output comes
from the schema rather than from sampling parameters, so there is no `temperature` here; several
current flagship models reject an explicit `temperature=0.0` outright. Spend is charged against the
active budget after each call; when a call's cost would take the block past the ceiling, the spend
is refused and `BudgetExceededError` is raised, so the next call never goes out.

Only the head of the chain runs unless it fails, so this needs one `ANTHROPIC_API_KEY` to work. The
legs below it have their own requirements when they are reached: the Bedrock leg needs
`pip install "strata-forge[bedrock]"` and AWS credentials, the `gpt-5.5` leg an `OPENAI_API_KEY`.
Without those, a chain that gets that far exhausts itself and raises `FallbackExhaustedError`.

More of both:
[`examples/`](https://github.com/GemovicNemanja/strata-forge/tree/main/examples) has 34 numbered,
self-contained scripts that skip cleanly when credentials are missing.

## Modules

Listed roughly bottom-up: anything in a row may import from the rows below it, never the other way
round.

| Module | What it does |
|---|---|
| [`strata_forge.core`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/core.md) | The strictly-upstream primitives: the `ForgeError` hierarchy, a tenacity-backed `@retry`, structlog with correlation-id injection, `BudgetContext` ceilings, reproducibility helpers, UUIDv7 ids |
| [`strata_forge.config`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/config.md) | The single configuration entry point: a Pydantic Settings root with per-concern sub-models, `.env` loading, and standalone YAML-overlay primitives |
| [`strata_forge.llm`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/llm.md) | The async client over LiteLLM: typed messages and responses, structured output, tool calling, streaming, two-axis fallback, a provider-agnostic cache, the model registry |
| [`strata_forge.sync`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/sync.md) | Blocking `asyncio.run` facades over four `LLMClient` methods, for CLI and notebook use |
| [`strata_forge.prompts`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/prompts.md) | Prompts as sandboxed Jinja2 templates split into a stable prefix and a dynamic suffix, rendered to messages and stored in a versioned registry |
| [`strata_forge.tracing`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/tracing.md) | Langfuse observability layered over everything else — LiteLLM auto-tracing, `@traced`, spans, scores, metrics — silently no-op when unconfigured |
| [`strata_forge.datasets`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/datasets.md) | Frozen dataset shapes with content-hash versioning and diffing, a pluggable async store, a Hugging Face bridge, synthetic-data helpers |
| [`strata_forge.evals`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/evals.md) | Experiments as data: models × prompts × dataset × graders run concurrently, then aggregated into metrics, reports, sweeps, trace replays, and a Wilson-bounded CI gate |
| [`strata_forge.agents`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/agents.md) | An agent runtime composed from `strata_forge.llm` primitives, with built-in tools, conversation and episodic memory, and hand-off / critic-refiner patterns |
| [`strata_forge.rag`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/rag.md) | Retrieval: five Protocols (embedder, chunker, retriever, vector store, reranker), concrete implementations, and a pipeline that composes them |
| [`strata_forge.storage`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/storage.md) | One async surface over local disk, S3, GCS, Azure Blob, and the Hugging Face Hub, plus model and dataset push/pull |
| [`strata_forge.compute`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/compute.md) | Remote work as typed data, submitted through a uniform `Backend` Protocol (local subprocess, SSH, SkyPilot), with batch inference and self-hosted serving adapters |
| [`strata_forge.training`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/training.md) | Fine-tuning: typed configs rendered into TRL's SFT and preference trainers (DPO, ORPO, KTO, GRPO), LoRA/QLoRA, chat templating, packing, NDJSON progress events |
| [`strata_forge.pipelines`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/pipelines.md) | Runnable entrypoints that compose serving, batch, and storage primitives into one job you launch on a compute target |
| [`strata_forge.cli`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/cli.md) | The `strata-forge` console script: `doctor`, `chat`, `prompts`, `datasets`, `eval`, `experiments`, `compute`, `train`, `serve` |

## Repository map

| Path | What is in it |
|---|---|
| [`src/strata_forge/`](https://github.com/GemovicNemanja/strata-forge/tree/main/src/strata_forge) | The library. One directory per module, each with its own `README.md` (what it does) and `CLAUDE.md` (rules for anyone editing it) |
| [`docs/`](https://github.com/GemovicNemanja/strata-forge/tree/main/docs) | The documentation site source: `modules/` reference, `recipes/` walkthroughs, `architecture/` narrative and ADRs, and the agent guide |
| [`examples/`](https://github.com/GemovicNemanja/strata-forge/tree/main/examples) | 34 numbered, runnable scripts. Each is self-contained and skips with a clear message when a key or extra is missing |
| [`notebooks/`](https://github.com/GemovicNemanja/strata-forge/tree/main/notebooks) | [marimo](https://marimo.io/) notebook templates — chat, eval iteration, RAG prototyping. Plain Python files, so they diff and lint like code |
| [`tests/`](https://github.com/GemovicNemanja/strata-forge/tree/main/tests) | `unit/` (fast, no network), `e2e/` (imports every example and notebook to catch API drift), `integration/` (needs the local service stack), `vcr/` (recorded provider exchanges) |
| [`scripts/`](https://github.com/GemovicNemanja/strata-forge/tree/main/scripts) | Operational helpers run on a schedule or on demand — today, the eval-regression gate the nightly workflow runs |
| [`docker/`](https://github.com/GemovicNemanja/strata-forge/tree/main/docker) | `compose.yaml`, the local Qdrant / Redis / Postgres stack you run beside a checkout, plus an image that packages the library |
| [`.github/`](https://github.com/GemovicNemanja/strata-forge/tree/main/.github) | CI, docs deploy, nightly, release, and promotion-guard workflows; issue and pull-request templates |
| [`.cursor/rules/`](https://github.com/GemovicNemanja/strata-forge/tree/main/.cursor/rules) | Glob-scoped restatements of the house conventions for Cursor's rule engine |
| [`CLAUDE.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CLAUDE.md) / [`AGENTS.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CLAUDE.md) | The conventions every contributor follows, written for agents. `AGENTS.md` is a symlink to the same file |
| [`Makefile`](https://github.com/GemovicNemanja/strata-forge/blob/main/Makefile) | The index of everything you can run with one command. `make help` prints it |

## Documentation

- **[Documentation site](https://gemovicnemanja.github.io/strata-forge/)** — the whole of `docs/`,
  rendered with MkDocs Material and searchable. Built from `main` on every push.
- **[`docs/`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/README.md)** — the same
  content as Markdown in the repository, if you would rather read it beside the code.
- **[Module reference](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/README.md)**
  — one page per module: the public API, the behavioural contracts, and the gotchas that only show
  up once you have used it.
- **[Recipes](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/recipes/README.md)** —
  end-to-end walkthroughs that cross module boundaries: evaluating a model with a custom grader,
  building a RAG pipeline over Qdrant, running a tool-calling agent, controlling cost and
  reliability, fine-tuning on remote compute and serving the result.
- **[Architecture overview](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/overview.md)**
  and **[module boundaries](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/module-boundaries.md)**
  — how the layers stack, and what each module may and may not import.
- **[Decision records](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/adr/README.md)**
  — sixteen ADRs covering why LiteLLM is the transport, why the API is async-only, why errors raise
  instead of returning, and the rest of the load-bearing choices.
- **[`examples/`](https://github.com/GemovicNemanja/strata-forge/tree/main/examples)** and
  **[`notebooks/`](https://github.com/GemovicNemanja/strata-forge/tree/main/notebooks)** — runnable
  scripts and interactive templates.

## Development

```bash
git clone https://github.com/GemovicNemanja/strata-forge.git
cd strata-forge
uv sync            # create .venv from the lockfile
make check         # ruff lint + pyright strict over src/strata_forge
make test          # unit tests (fast, no network)
```

`make help` lists every target: `make test-all` adds the example and notebook import harness,
`make test-cov` reports coverage, `make stack-up` brings up Qdrant, Redis, and Postgres for the
integration tests, `make docs-serve` serves this documentation locally, and `make doctor` runs the
CLI diagnostic. Formatting, linting, and the type check also run as pre-commit hooks
(`uv run pre-commit install`).

## Contributing

Bug reports, graders, chunkers, provider routes, and documentation fixes are all welcome — read
[**CONTRIBUTING.md**](https://github.com/GemovicNemanja/strata-forge/blob/main/CONTRIBUTING.md)
first. It covers environment setup, the check loop, the architecture rules a change has to respect,
the Conventional Commits format, the branch and promotion flow, and when a change needs an ADR.
Report bugs and request features through
[GitHub Issues](https://github.com/GemovicNemanja/strata-forge/issues) — there is a template for
each. Participation is governed by the
[Code of Conduct](https://github.com/GemovicNemanja/strata-forge/blob/main/CODE_OF_CONDUCT.md).

**Working with a coding agent?** Point it at
[`AGENTS.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CLAUDE.md) for the rules and
[`docs/agent-guide.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/agent-guide.md)
for the how-to: the order to read files in, which file to open for which task, which files have to
change together, and how to prove a change is correct before opening a pull request.

## Security

Do not report vulnerabilities in a public issue or pull request. Use
[GitHub's private advisory form](https://github.com/GemovicNemanja/strata-forge/security/advisories/new),
or email **gemovic@strataml.com** if you cannot. The full policy — supported versions, what a
useful report contains, response times, and the credential-handling rules this library follows — is
in [SECURITY.md](https://github.com/GemovicNemanja/strata-forge/blob/main/SECURITY.md).

## License

[Apache License 2.0](https://github.com/GemovicNemanja/strata-forge/blob/main/LICENSE). See
[`NOTICE`](https://github.com/GemovicNemanja/strata-forge/blob/main/NOTICE) for attribution
requirements.

## Acknowledgements

strata-forge is a thin, opinionated layer over other people's hard work.
[LiteLLM](https://github.com/BerriAI/litellm) carries every provider request, so this library never
had to write six HTTP clients. [Pydantic](https://docs.pydantic.dev/) makes strict typing across a
process boundary practical rather than aspirational. [Langfuse](https://langfuse.com/) is the
observability, prompt, and dataset backend the tracing and store layers target.
[marimo](https://marimo.io/) makes the notebook templates ordinary Python files that lint, diff,
and review like the rest of the codebase. Fine-tuning stands on
[TRL](https://github.com/huggingface/trl), [PEFT](https://github.com/huggingface/peft), and
[Transformers](https://github.com/huggingface/transformers); remote compute on
[SkyPilot](https://github.com/skypilot-org/skypilot); retrieval on
[Qdrant](https://qdrant.tech/).
