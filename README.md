# strata-forge

Typed, async-first toolkit for AI/LLM work — inference, evaluation, fine-tuning, retrieval, and
remote compute orchestration across the major foundation-model providers.

Built on **LiteLLM** for provider breadth and **Pydantic v2** for strict typing. The public API is
async, with sync wrappers in `strata_forge.sync` for CLI and notebook ergonomics. Heavy optional
dependencies (training, serving, remote compute) live behind extras, so the base install stays light.

## Install

```bash
pip install strata-forge
```

Requires **Python 3.14+**.

Optional extras pull in the heavy dependencies only when you need them:

```bash
pip install "strata-forge[rag]"        # embeddings, vector store, reranking
pip install "strata-forge[compute]"    # SkyPilot + SSH remote execution
pip install "strata-forge[serving]"    # vLLM
pip install "strata-forge[finetuning]" # torch, transformers, trl, peft
pip install "strata-forge[all]"        # everything
```

Full set: `redis`, `multimodal`, `inspect`, `langfuse`, `hf`, `evals`, `rag`, `compute`, `serving`,
`bedrock`, `storage`, `finetuning`, `all`.

## Quickstart

```python
import asyncio

from strata_forge.llm import LLMClient, Message


async def main() -> None:
    client = LLMClient("gpt-5.5", provider="openai")
    response = await client.complete([Message.user("Say hello in one short sentence.")])
    print(response.text)


asyncio.run(main())
```

Provider credentials are read from the environment (for example `OPENAI_API_KEY`). A CLI is
installed as `strata-forge`.

## What's in it

| Module | Purpose |
|---|---|
| `strata_forge.llm` | Provider abstraction, structured output, tool calling, two-axis fallback |
| `strata_forge.core` | Cross-cutting errors, retry, logging, budget ceilings, reproducibility |
| `strata_forge.config` | Pydantic Settings root with YAML overlays and `.env` support |
| `strata_forge.prompts` | Jinja2 and Langfuse-backed prompt registry |
| `strata_forge.tracing` | Langfuse observability |
| `strata_forge.datasets` | Dataset CRUD with a Hugging Face bridge |
| `strata_forge.evals` | Experiment runner, graders, metrics |
| `strata_forge.agents` | Agent runtime, memory, multi-agent composition |
| `strata_forge.rag` | Embed, chunk, store, retrieve, rerank |
| `strata_forge.storage` | fsspec gateway and Hugging Face Hub |
| `strata_forge.compute` | SkyPilot and SSH backends, serving, batch inference |
| `strata_forge.training` | SFT / DPO / ORPO / KTO / GRPO with PEFT |
| `strata_forge.pipelines` | Runnable entrypoints executed on a compute target |

## Documentation

Reference docs live in the repository:

- [Architecture overview](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/overview.md)
- [Architecture Decision Records](https://github.com/GemovicNemanja/strata-forge/tree/main/docs/architecture/adr)
- [Per-module API reference](https://github.com/GemovicNemanja/strata-forge/tree/main/docs/modules)

## Development

```bash
uv sync        # materialize the venv + lockfile
make check     # lint (ruff) + type-check (pyright strict)
make test      # unit tests
```

## License

[Apache License 2.0](https://github.com/GemovicNemanja/strata-forge/blob/main/LICENSE).
