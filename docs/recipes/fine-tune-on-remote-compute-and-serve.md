# Fine-tune on remote compute and serve the result

This is the longest path through the library: prepare data with `strata_forge.datasets`, describe
the run with `strata_forge.training`, ship it to a GPU with `strata_forge.compute`, move weights
with `strata_forge.storage`, then stand up vLLM and talk to it through the same `LLMClient` you
use for hosted models.

**What strata-forge does and does not do.** It renders typed configs into TRL's surface, ships a
task description to a machine over a uniform async `Backend` interface, tails progress, and gives
you an OpenAI-compatible client pointed at whatever you launched. It does **not** provision GPUs,
install CUDA, choose hyperparameters, or ship a packaged trainer script. The training entrypoint
that runs on the remote host is yours to write; the library gives you the configs it consumes and
the plumbing around it.

**Extras.**

| Extra | Installs | Needed where |
|---|---|---|
| `[finetuning]` | torch, transformers, trl, peft, accelerate, datasets | On the **training host**. Everything is lazy-imported, so your laptop can build configs without it. |
| `[compute]` | skypilot, asyncssh | Wherever you **submit** from. `LocalBackend` needs neither. |
| `[hf]` | datasets | Wherever you convert a Forge `Dataset` to a Hugging Face one. |
| `[storage]` | fsspec, s3fs, gcsfs, adlfs, huggingface_hub | Wherever you push or pull weights. |
| `[serving]` | vllm | Only if vLLM runs in **this** environment. When you launch a serving task on a remote host, the task's own `setup` installs it there. |

**Running the snippets.** Every block below is a fragment, not a complete program: anything with an
`await` belongs inside an `async def main()` invoked with `asyncio.run(main())`, or in a notebook
cell, where top-level `await` is legal.

---

## 1. Prepare the training data

TRL trains on a Hugging Face dataset with one text column. `to_hf_dataset` converts a Forge
`Dataset` into HF rows (`id`, `input`, `expected_output`, `metadata`) but it does not invent a
training column — the prompt format is yours to decide, so you build it.

```python
from datasets import Dataset as HFDataset          # from the [hf] extra
from transformers import AutoTokenizer             # from the [finetuning] extra

from strata_forge.llm import AssistantMessage, SystemMessage, UserMessage
from strata_forge.training import apply_chat_template

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

conversations = [
    [
        SystemMessage(content="You are a terse support agent."),
        UserMessage(content=str(item.input["question"])),
        AssistantMessage(content=str(item.expected_output)),
    ]
    for item in dataset.items
]

train_dataset = HFDataset.from_dict(
    {"text": [apply_chat_template(c, tokenizer=tokenizer) for c in conversations]}
)
```

`apply_chat_template` returns a string with `tokenize=False` (the default) and token ids with
`tokenize=True`. When no tokenizer is available — a quick smoke test, or a base model with no
chat template — `conversation_to_text` renders `role: content` blocks instead, and
`conversation_to_dicts` gives you the plain `[{role, content}]` shape.

The column name must match `SFTConfig.dataset_text_field`, which defaults to `"text"`.

## 2. Describe the run

Configs are frozen Pydantic models that render into TRL kwargs. Build and inspect them anywhere;
nothing heavy is imported until you call `train()`.

```python
from strata_forge.training import LoRAConfig, SFTConfig, SFTRunner

config = SFTConfig(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="./checkpoints/support-sft",
    max_seq_length=4096,
    packing=True,
    num_epochs=2,
    per_device_batch_size=2,
    gradient_accumulation_steps=8,     # effective batch = 2 x 8 x n_devices
    learning_rate=1e-4,
    precision="bf16",
    save_steps=200,
    progress_jsonl="progress.jsonl",
)

runner = SFTRunner(config, peft_config=LoRAConfig(r=32, alpha=64, target_modules=("q_proj", "v_proj")))
print(config.to_trl_kwargs())          # exactly what TRL receives
```

`to_trl_kwargs()` is the honesty check: it shows the rendered dict, including the Forge-specific
renames (`max_seq_length` becomes TRL's `max_length`, `save_steps=0` becomes
`save_strategy="no"`). Anything Forge does not model goes through `extra_trainer_args`, which is
merged last and therefore wins over Forge's defaults.

Swap `LoRAConfig` for `QLoRAConfig` to train an adapter on a 4-bit base (nf4, double quant, bf16
compute by default). `PreferenceRunner` with `DPOConfig` / `ORPOConfig` / `KTOConfig` /
`GRPOConfig` covers the preference methods; `reward_funcs` is required for GRPO, and passing it
to the other three is silently ignored rather than rejected.

On the GPU host, with `[finetuning]` installed:

```python
result = runner.train(train_dataset=train_dataset)
print(result.output_dir, result.train_loss, result.metrics)
```

`train()` is **synchronous** — a deliberate exception to the library's async-first rule, because
TRL's loop is blocking and compute-bound, and wrapping it in a coroutine would suggest a
concurrency that does not exist.

See
[`examples/31_training_sft_config.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/31_training_sft_config.py)
for the full rendered-kwargs walkthrough, which runs without a GPU.

## 3. Ship it to a machine

A `Task` is a frozen description of remote work: a shell command, optional setup, a working
directory to upload, environment variables, and a hardware request. Backends consume it; none of
them interpret it.

```python
import os

from strata_forge.compute import ResourceSpec, Task

task = Task(
    name="support-sft",
    setup="pip install 'strata-forge[finetuning]'",
    run="python train_entrypoint.py --config config.json",
    workdir="./trainer",                       # uploaded; becomes the job's cwd
    env={
        "FORGE_PROGRESS_PATH": "progress.jsonl",
        "HF_TOKEN": os.environ["HF_TOKEN"],
    },
    resources=ResourceSpec(accelerators="A100:1", cloud="aws", disk_gb=200),
)
```

`train_entrypoint.py` is your script — it loads `config.json` into an `SFTConfig`, builds an
`SFTRunner`, and calls `train()`. Keep it small; everything interesting is already in the config.

`env` values are rendered verbatim into the task YAML and shipped to the worker. Read secrets
from your own environment as above rather than committing them, and do not persist a rendered
task that contains one.

Three backends satisfy the same `Backend` Protocol:

```python
from strata_forge.compute import LocalBackend, SSHBackend, SkyPilotBackend

backend = LocalBackend()                                       # subprocess, no extra
backend = SSHBackend(host="gpu-01", username="ubuntu")         # [compute] → asyncssh
backend = SkyPilotBackend(cluster_prefix="forge-")             # [compute] → skypilot
```

`SSHBackend` runs the task under `nohup` in a per-job directory below `~/.forge-compute/`,
tracking the PID and exit code in files there, so a dropped connection does not kill the job.
`SkyPilotBackend` provisions from `ResourceSpec` and manages the cluster. `LocalBackend` runs a
subprocess in `task.workdir` (or the parent's cwd) and buffers stdout and stderr in memory —
convenient for smoke tests, not a job store.

The imports are lazy and happen on **first use**, not in the constructor: `SSHBackend(...)`
constructs fine without asyncssh, and the `ImportError` with the install hint arrives at the
first `submit`. Do not use construction as a capability probe.

`Task.to_yaml()` and `Task.from_yaml()` round-trip a SkyPilot-YAML subset, so a task can be
reviewed, committed, and submitted from the CLI:

```bash
strata-forge compute submit task.yaml --backend ssh --ssh-host gpu-01 --ssh-user ubuntu
strata-forge compute status <job-id>
strata-forge compute logs <job-id>
```

Unknown top-level keys in the YAML raise `ValueError` rather than being silently dropped — Forge
will not pretend to honour a SkyPilot knob it does not model.

See
[`examples/30_compute_local.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/30_compute_local.py).

## 4. Watch it run

Set `progress_jsonl` on the config (or `FORGE_PROGRESS_PATH` in the environment) and both runners
attach a callback that appends one `ProgressEvent` per training event — step, epoch, loss,
learning rate, and any extra metrics — flushing after each line so a tailing reader sees them
live.

`Backend.read_file` reads a file the job produced inside its own working directory:

```python
import asyncio

job = await backend.submit(task)
while not (status := await backend.status(job)).is_terminal:
    print(await backend.read_file(job, "progress.jsonl", tail=5))
    await asyncio.sleep(30)
print(status.state, status.exit_code)
```

Paths are workdir-relative and validated: absolute paths, backslashes, and `..` traversal are
rejected before anything is read, and the read is size-capped. `read_file` is implemented by
`LocalBackend` and `SSHBackend`; `SkyPilotBackend` raises `NotImplementedError`, so use `logs()`
there. See [ADR 0016](../architecture/adr/0016-backend-read-file.md).

## 5. Collect the weights

`HFHubClient` wraps `huggingface_hub` behind an async surface. `push_model` creates the repo if
needed and uploads the directory; `pull_model` fetches a snapshot to a local path.

```python
from strata_forge.storage import HFHubClient

hub = HFHubClient()      # token from HF_TOKEN via strata_forge.config, or a cached login
url = await hub.push_model("./checkpoints/support-sft", "my-org/support-sft", private=True)
local = await hub.pull_model("my-org/support-sft", "./weights/support-sft")
```

For object storage rather than the Hub, `StorageGateway` is one `fsspec`-backed interface over
local disk, S3, GCS, and Azure Blob — the protocol comes from the URL scheme, so the same call
works everywhere:

```python
from strata_forge.storage import StorageGateway

gateway = StorageGateway()
await gateway.write_text("s3://my-bucket/runs/support-sft/manifest.json", manifest)
```

Both need the `[storage]` extra; the import is lazy, so the `ImportError` with the install hint
arrives on the first networked call. See
[`examples/33_storage_gateway.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/33_storage_gateway.py)
and
[`examples/34_storage_hf_hub.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/34_storage_hf_hub.py).

## 6. Serve it

The serving builders render a `Task` that launches an OpenAI-compatible server. They build shell
commands — nothing imports vLLM, TGI, or SGLang in-process — so the target host is what needs the
server installed, which the task's `setup` handles by default.

```python
from strata_forge.compute import build_vllm_task

task = build_vllm_task(
    "my-org/support-sft",
    port=8000,
    tensor_parallel_size=1,
    max_model_len=8192,
    dtype="bfloat16",
)
```

Pass `setup=""` when the host already has vLLM. `build_tgi_task` and `build_sglang_task` are the
equivalents for the other two servers.

`serving_endpoint` submits the task, polls `{base_url}/models` until it answers 200, and yields a
`ServingEndpoint` for the lifetime of the block, cancelling and cleaning up the job on exit. Forge
does not guess the URL from the task — you configured the port, so you supply it.

```python
from strata_forge.compute import serving_endpoint
from strata_forge.llm import LLMClient, Message, OpenAICompatConfig, OpenAICompatProvider

async with serving_endpoint(
    backend,
    task,
    base_url="http://localhost:8000/v1",
    wait_timeout_s=900,
) as endpoint:
    client = LLMClient(
        model="my-org/support-sft",
        provider="openai_compat",
        provider_clients={
            "openai_compat": OpenAICompatProvider(
                OpenAICompatConfig(base_url=endpoint.base_url)
            )
        },
    )
    response = await client.complete([Message.user("Where is my order?")])
    print(response.text)
```

Pinning the endpoint goes through `provider_clients`, which replaces the auto-instantiated
provider client for that route. That is the whole mechanism: `LLMClient` has no separate
"custom endpoint" concept.

`wait_for_endpoint` is exported separately if you manage the server's lifecycle yourself and only
want the readiness probe.

See
[`examples/32_serving_vllm_task.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/32_serving_vllm_task.py)
for the rendered YAML of all three servers, and `strata-forge serve vllm --model ... --submit local`
for the CLI equivalent.

## 7. Run a batch through it

Self-hosted serving is usually about throughput. `BatchInferenceRunner` fans a prompt list across
one client with bounded concurrency:

```python
from strata_forge.compute import BatchInferenceRunner

runner = BatchInferenceRunner(client, concurrency=16, on_error="collect")
results = await runner.run([[Message.user(p)] for p in prompts])

succeeded = [r.response.text for r in results if r.succeeded]
failed = [r.error for r in results if not r.succeeded]
```

`on_error="collect"` returns one aligned `BatchInferenceResult` per prompt, each holding either a
response or an exception — the right mode for an unattended batch where one bad row should not
lose the other thousand. With the default `on_error="raise"`, the first exception propagates out
of `run()` and you get no tuple at all; note that the sibling calls already in flight continue to
completion and still cost money, so `"raise"` is a reporting choice, not a kill switch.

---

## Registry note

The curated model registry ([ADR 0004](../architecture/adr/0004-model-registry-scope.md)) tracks
hosted models only, so it has no pricing or capability entry for `my-org/support-sft`. Cost
accounting through an `openai_compat` route therefore has nothing to compute against — the
useful signals for a self-hosted server are latency and token counts, not `cost_usd`.

## Where to go next

- [`docs/modules/training.md`](../modules/training.md) — every config field, the preference
  methods, packing, progress events.
- [`docs/modules/compute.md`](../modules/compute.md) — the `Backend` Protocol, job states, YAML
  round-trip, serving helpers.
- [`docs/modules/storage.md`](../modules/storage.md) — the gateway and Hub client in full.
- [ADR 0013](../architecture/adr/0013-compute-task-and-backend-shapes.md) — why `Task` is inert
  data and `Backend` is a Protocol.
- [Control cost and reliability](control-cost-and-reliability.md) — the batch above is exactly
  where a `BudgetContext` earns its keep.
