# Recipes

A recipe is an end-to-end walkthrough of a task that crosses module boundaries — the thing you
actually want to do, rather than the surface of any single module. The per-module reference under
[`docs/modules/`](../modules/README.md) is where you look up a signature once you know which one
you need; these pages are where you find out.

Each recipe is opinionated: it picks one way through, explains why that way, and names the sharp
edges rather than routing around them. Every snippet uses symbols that exist in the published
package. Where a complete runnable script already lives in `examples/`, the recipe links it
instead of reprinting it.

| Recipe | What it covers |
|---|---|
| [Evaluate a model with a custom grader](evaluate-a-model-with-a-custom-grader.md) | A dataset, a hand-written grader satisfying the `Grader` Protocol, an `LLMJudge`, the runner, the Markdown and HTML reports, and a Wilson-bounded CI gate. |
| [Build a RAG pipeline over Qdrant](build-a-rag-pipeline-over-qdrant.md) | Chunk, embed, store in process, swap in Qdrant, fuse dense retrieval with BM25, rerank, and hand the context to a model. |
| [Run a tool-calling agent](run-a-tool-calling-agent.md) | The `@tool` decorator, `run_tool_loop` and its streaming counterpart, client-executed tools, then what `Agent`, memory, and the multi-agent patterns add on top. |
| [Control cost and reliability in production](control-cost-and-reliability.md) | How budgets, two-axis fallback, the response cache, and retry compose inside one call — and what each of them does not cover. |
| [Fine-tune on remote compute and serve the result](fine-tune-on-remote-compute-and-serve.md) | Training data, an `SFTConfig` with a LoRA adapter, a task on SSH or SkyPilot, progress tailing, weight transfer, and vLLM behind an `LLMClient`. |

If you are new to the library, read them in that order: the first two need only an API key, the
third introduces the loop everything else is built on, and the last two assume you have already
seen a call go out and come back.

## Related reading

- [`docs/modules/`](../modules/README.md) — per-module API reference.
- [`docs/architecture/overview.md`](../architecture/overview.md) — how the modules fit together.
- [`docs/architecture/module-boundaries.md`](../architecture/module-boundaries.md) — what each
  module owns and what it is not allowed to import. The numbered ADRs beside it record why each
  seam looks the way it does; every recipe links the ones relevant to its subject.
- [`examples/`](https://github.com/GemovicNemanja/strata-forge/tree/main/examples) — numbered,
  self-contained scripts that skip cleanly when credentials are missing.
