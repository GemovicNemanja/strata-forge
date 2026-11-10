# strata-forge

Typed, async-first toolkit for AI/LLM work — inference, evaluation, fine-tuning, retrieval, and
remote compute orchestration across the major foundation-model providers.

Built on **LiteLLM** for provider breadth and **Pydantic v2** for strict typing. The public API is
async for execution and I/O, with sync wrappers in `strata_forge.sync` for CLI and notebook ergonomics. Pure classification metrics are synchronous. Heavy optional
dependencies (training, serving, remote compute) live behind extras, so the base install stays light.

Pinned classification evaluation uses explicit stable-ID joins and separate per-config reports,
with native classifier or imported-probability inputs. The installed `eval schema` and
`eval validate-spec` commands expose its inert contract. Execution requires a trusted read
policy; the installed adapter consumes a run/spec-bound private Task authority frame. Publishing
additionally requires a separate destination fence and accepted actual-commit receipt. Standalone
launch denies access without installed authority. See [pipeline workers](docs/modules/pipelines.md).

Primary classifier training has an async SDK over resolved objects, with ordinary CE defaults,
exact trusted head initialization and complete local safetensors reload. It performs no Hub or
remote operations; worker/adapter/calibration integration remains separate. See [training](docs/modules/training.md).
A separate local auxiliary verifier SDK preserves the primary path while adding exact PID-offset
geometry, independent flag/bilinear support heads and strict safe export/reload. It does not
register a worker or change the primary training defaults.
The separate local population profile scans a complete named source before optimizer use,
caches selected unpadded inputs on disk and consumes exactly one recorded epoch. Its trusted
Python policy seam does not enable remote dispatch or change v1 worker limits.
A separate bounded local CPU benchmark SDK reports explicit scorer/verifier call boundaries,
warmups, latency percentiles and two throughput denominators. It does not load models, run
remote jobs or make hardware/research performance claims.

A separate local temperature-fit SDK consumes exact raw logits from a complete registered
named source under an explicit trusted policy. It retains numerical and fit provenance,
never infers held-out eligibility or enables a worker/model call.

## Install

```bash
pip install strata-forge
```

Requires **Python 3.14+**.

Optional extras pull in the heavy dependencies only when you need them:

```bash
pip install "strata-forge[rag]"        # embeddings, vector store, reranking
pip install "strata-forge[compute]"    # SkyPilot + SSH remote execution
pip install "strata-forge[serving]"    # vLLM and exact HF scoring dependencies
pip install "strata-forge[finetuning]" # torch, transformers, trl, peft, bitsandbytes
pip install "strata-forge[scoring]"    # NumPy for temperature application
pip install "strata-forge[all]"        # everything
```

Full set: `redis`, `multimodal`, `inspect`, `langfuse`, `hf`, `evals`, `rag`, `compute`, `serving`,
`bedrock`, `storage`, `finetuning`, `scoring`, `all`.

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
| `strata_forge.prompts` | Jinja2 templates and immutable modular prompt builds with exact pins and UTF-8 attribution |
| `strata_forge.tracing` | Langfuse observability |
| `strata_forge.datasets` | Dataset CRUD, Hugging Face bridge, and [isolated local SQL](docs/modules/headless-sql.md) over provided pinned Parquet |
| `strata_forge.metrics` | Pure classification metrics, explicit populations, and paired/family comparisons |
| `strata_forge.evals` | Experiment runner, graders, metrics |
| `strata_forge.agents` | Agent runtime, memory, multi-agent composition |
| `strata_forge.rag` | Embed, chunk, store, retrieve, rerank |
| `strata_forge.storage` | fsspec gateway and Hugging Face Hub |
| `strata_forge.compute` | SkyPilot and SSH backends, serving, batch inference |
| `strata_forge.training` | SFT / DPO / ORPO / KTO / GRPO with PEFT |
| `strata_forge.pipelines` | Runnable entrypoints executed on a compute target |

The inference pipeline accepts the default `vllm_chat` engine and HF forward-pass engines:
`hf_classify` scores premise/hypothesis pairs; `hf_label_logits` reads exact single-token label
probabilities and full-vocabulary outside mass. Both preserve explicit overflow failures and
write Parquet plus input-commit provenance. See the [scoring contract](docs/modules/compute.md#exact-hugging-face-scoring)
for required knobs, safe loading, length policy and result artifacts.
Trusted Python consumers of a loaded scorer can select `score(..., overflow_policy="error")`
to refuse overlength calls without truncating inputs. Run specs/workers keep their existing
length policy and do not forward this SDK option.

## Documentation

Reference docs live in the repository:

- [Architecture overview](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/overview.md)
- [Architecture Decision Records](https://github.com/GemovicNemanja/strata-forge/tree/main/docs/architecture/adr)
- [Per-module API reference](https://github.com/GemovicNemanja/strata-forge/tree/main/docs/modules)

## Development

Maintained by [Nemanja Gemovic](https://github.com/GemovicNemanja), with package
contact [gemovic.nemanja@gmail.com](mailto:gemovic.nemanja@gmail.com).

```bash
uv sync        # materialize the venv + lockfile
make check     # lint (ruff) + type-check (pyright strict)
make test      # unit tests
```

## License

[Apache License 2.0](https://github.com/GemovicNemanja/strata-forge/blob/main/LICENSE).

### Provided native datasets

The opt-in `strata_forge.datasets.LocalDatasetExecutor` validates supplied CSV/JSONL or
Parquet, queries exact named Parquet inputs and returns bounded immutable materialized
Parquet with execution receipts. It requires explicit pinned operator image setup and has
no platform grants, background job scheduler or parent native-engine fallback. See
[the execution contract](docs/modules/native-dataset-execution.md). Materialized columns
require explicit distinct names; ambiguous projections refuse instead of silently renaming.
