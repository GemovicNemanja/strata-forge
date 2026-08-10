# strata-forge documentation

strata-forge is a typed, async-first Python library for AI/LLM work: inference across the major
foundation-model providers, prompting, tracing, datasets, evaluation, agents, retrieval,
fine-tuning, and remote compute. It is built on LiteLLM for provider breadth and Pydantic v2 for
strict typing, and it ships on PyPI as `strata-forge` (the import package is `strata_forge`).

If you have not installed it yet, start with
[Install](https://github.com/GemovicNemanja/strata-forge/blob/main/README.md#install) and the
[quickstart](https://github.com/GemovicNemanja/strata-forge/blob/main/README.md#quickstart) in the
repository README. Source, issues, and releases live on
[GitHub](https://github.com/GemovicNemanja/strata-forge).

---

## Table of contents

### Start here

| Page | What it answers |
|---|---|
| [Install and extras](https://github.com/GemovicNemanja/strata-forge/blob/main/README.md#install) | What `pip install` command do I run, and which optional extras do I actually need? |
| [Quickstart](https://github.com/GemovicNemanja/strata-forge/blob/main/README.md#quickstart) | What is the shortest program that calls a model and prints the answer? |
| [`strata_forge.llm`](modules/llm.md) | How do I make a real call — messages, structured output, tools, streaming, fallback, cost? |
| [Evaluate a model with a custom grader](recipes/evaluate-a-model-with-a-custom-grader.md) | How do I get from "it seems fine" to a number I can put in CI? |
| [`strata_forge.cli`](modules/cli.md) | What can I do from a terminal without writing any Python? |
| [`examples/`](https://github.com/GemovicNemanja/strata-forge/tree/main/examples) | Is there a runnable script for this? (Numbered, self-contained, skips cleanly without credentials.) |

### Recipes — task-oriented walkthroughs

Start at the [recipes index](recipes/README.md) for the suggested reading order.

| Recipe | What it answers |
|---|---|
| [Evaluate a model with a custom grader](recipes/evaluate-a-model-with-a-custom-grader.md) | How do I build a dataset, write my own grader, run the matrix, read the report, and gate CI on the result? |
| [Build a RAG pipeline over Qdrant](recipes/build-a-rag-pipeline-over-qdrant.md) | How do I chunk, embed, and retrieve in process — then swap in Qdrant, fuse BM25, and rerank without rewriting? |
| [Run a tool-calling agent](recipes/run-a-tool-calling-agent.md) | How does the tool loop actually work, and what does `Agent` add on top of it? |
| [Control cost and reliability in production](recipes/control-cost-and-reliability.md) | In what order do budgets, fallback, cache, and retry compose inside a single call? |
| [Fine-tune on remote compute and serve the result](recipes/fine-tune-on-remote-compute-and-serve.md) | How do I get from a dataset to a trained adapter on a GPU box to an endpoint I can call? |

### Module reference — the per-module API pages

One page per module, listed bottom-up: everything below is upstream of everything above it. The
[module index](modules/README.md) carries the same list with fuller summaries.

| Module | What it answers |
|---|---|
| [`strata_forge.core`](modules/core.md) | Errors, retry, structlog logging, correlation ids, budget ceilings, reproducibility — the primitives every other module imports. |
| [`strata_forge.config`](modules/config.md) | Where every environment variable is read, and what `Settings` actually layers. |
| [`strata_forge.llm`](modules/llm.md) | The async client: typed messages, structured output, tool calling, streaming, two-axis fallback, caching, the model registry. |
| [`strata_forge.sync`](modules/sync.md) | The blocking facades for scripts and notebooks, and when they are the wrong tool. |
| [`strata_forge.prompts`](modules/prompts.md) | Stable-prefix / dynamic-suffix Jinja2 templates, safe filters, and a versioned prompt registry. |
| [`strata_forge.tracing`](modules/tracing.md) | Langfuse traces, spans, scores, and metrics — and why nothing breaks when Langfuse is absent. |
| [`strata_forge.datasets`](modules/datasets.md) | Frozen dataset shapes, content-hash versioning and diffing, the Hugging Face bridge, synthetic data. |
| [`strata_forge.evals`](modules/evals.md) | Experiments as data: the runner, graders, metrics, sweeps, reports, trace replay, the CI gate. |
| [`strata_forge.agents`](modules/agents.md) | The agent runtime over `strata_forge.llm`: built-in tools, conversation and episodic memory, hand-off and critic-refiner. |
| [`strata_forge.rag`](modules/rag.md) | The five retrieval Protocols and their implementations — chunkers, embedders, dense/BM25/hybrid retrieval, rerankers, pipeline. |
| [`strata_forge.storage`](modules/storage.md) | One async surface over local disk, S3, GCS, Azure Blob, and the Hugging Face Hub. |
| [`strata_forge.compute`](modules/compute.md) | Remote work as typed data, submitted through one `Backend` Protocol: local subprocess, SSH, SkyPilot. |
| [`strata_forge.training`](modules/training.md) | Typed configs rendered into TRL's SFT and preference trainers, with LoRA/QLoRA and progress events. |
| [`strata_forge.pipelines`](modules/pipelines.md) | The runnable entrypoints you launch on a compute target, and the run spec they read. |
| [`strata_forge.cli`](modules/cli.md) | Every `strata-forge` command, its flags, and where it keeps local state. |

### Architecture — how the pieces fit and why

| Page | What it answers |
|---|---|
| [Architecture overview](architecture/overview.md) | What are the design pillars, how do the layers stack, and what is deliberately out of scope? |
| [Module boundaries](architecture/module-boundaries.md) | What does each module own, what may it import, and what is it explicitly not allowed to do? |
| [Decision records](architecture/adr/README.md) | Why does a given seam look the way it does? Sixteen accepted ADRs, each linked from the index and from the module pages it governs. |

### Contributing

| Page | What it answers |
|---|---|
| [Agent guide](agent-guide.md) | I am a coding agent: what do I read first, which file do I open for which task, and how do I prove the change is right? |
| [`CONTRIBUTING.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CONTRIBUTING.md) | How do I set up the environment, run the checks, and shape a commit and a pull request? |
| [`CLAUDE.md` / `AGENTS.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CLAUDE.md) | The house rules themselves — the authority both of the pages above defer to. |
| [`CHANGELOG.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CHANGELOG.md) | What changed in each release, and which changes are breaking? |
| [`SECURITY.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/SECURITY.md) | How do I report a vulnerability, and which versions get fixes? |
| [`CODE_OF_CONDUCT.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CODE_OF_CONDUCT.md) | What behaviour is expected in this project's spaces, and how is it enforced? |

---

## Choose your path

### I want to call a model

1. [Install and quickstart](https://github.com/GemovicNemanja/strata-forge/blob/main/README.md#install)
   — base install, one provider key, one call.
2. [`strata_forge.llm`](modules/llm.md) — the client surface end to end: routing,
   structured output, streaming, errors.
3. [Run a tool-calling agent](recipes/run-a-tool-calling-agent.md) — the tool loop, then what the
   agent layer packages on top of it.
4. [Control cost and reliability in production](recipes/control-cost-and-reliability.md)
   — budgets, fallback, cache, and retry before you put it anywhere real.

### I want to evaluate or fine-tune

1. [`strata_forge.datasets`](modules/datasets.md) — the frozen shapes, content-hash versions,
   and the Hugging Face bridge everything downstream consumes.
2. [Evaluate a model with a custom grader](recipes/evaluate-a-model-with-a-custom-grader.md) — one
   dataset, one hand-written grader, one report, one CI gate.
3. [`strata_forge.evals`](modules/evals.md) — the full runner surface: sweeps, metrics,
   trace replay, Wilson bounds.
4. [Fine-tune on remote compute and serve the result](recipes/fine-tune-on-remote-compute-and-serve.md)
   — training configs, a GPU host, weight transfer, and vLLM behind the same client.

### I want to contribute or extend

1. [Architecture overview](architecture/overview.md) — the layering and the reasons for it.
2. [Module boundaries](architecture/module-boundaries.md) — the import rules a review holds you to.
3. [`CONTRIBUTING.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CONTRIBUTING.md#getting-set-up)
   — environment, check loop, commit and PR conventions.
4. [Agent guide](agent-guide.md) — the same ground for a coding agent, plus which files
   must change together.

---

## Conventions worth knowing up front

Three properties hold everywhere in the library. Every page below assumes them.

- **Async-first, with one sync escape hatch.** Every public API that performs I/O is `async`.
  Blocking convenience exists in exactly one place — [`strata_forge.sync`](modules/sync.md) — as
  `asyncio.run` facades over the LLM client, so there is never a second code path to keep in step
  ([ADR 0002](architecture/adr/0002-async-only-public-api.md)).
- **Heavy dependencies live behind extras.** `import strata_forge.<module>` works on the base
  install; a feature that needs `torch`, `qdrant-client`, `skypilot`, or similar imports it inside
  the function that uses it and raises `ImportError` with the exact
  [`pip install 'strata-forge[<extra>]'`](https://github.com/GemovicNemanja/strata-forge/blob/main/README.md#install)
  hint when it is missing.
- **Failures raise, and they raise one family.** Library errors inherit from `ForgeError`, and
  provider exceptions normalize into the `ProviderError` subtree at the LiteLLM seam. Ordinary
  argument mistakes still raise the builtins you would expect. See
  [the `ForgeError` hierarchy](modules/core.md#the-forgeerror-hierarchy) and
  [ADR 0003](architecture/adr/0003-exception-hierarchy.md).
