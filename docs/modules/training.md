# `strata_forge.training` — SFT, preference tuning, PEFT, chat-template, packing

`strata_forge.training` ships fine-tuning primitives: SFT via TRL's
`SFTTrainer`, preference tuning (DPO, ORPO, KTO, GRPO)
via the TRL preference trainer family, and PEFT (LoRA, QLoRA)
adapters via peft. Supporting helpers cover chat-template
formatting (`strata_forge.llm.AnyMessage` → HF tokenizer dict shape)
and sequence packing for SFT throughput.

The module wraps each TRL trainer with a typed Pydantic
config and a small runner class. Configs are frozen Pydantic models, so
`model_dump()` / `model_dump_json()` give you a reproducible record you
can ship to a remote `strata_forge.compute` run and rebuild there. Runners
defer every heavy import until `train()` is called.

**The two `train()` methods are synchronous** — a deliberate exception to
the library's async-first rule. A training run is a long, CPU/GPU-bound,
single-threaded call with no I/O concurrency to exploit; making it a
coroutine would only invite callers to block an event loop for hours. Call
them from a worker process, or from `asyncio.to_thread` if you must reach
them from async code.

Integration points:

- **Configs:** `SFTConfig`, `DPOConfig`,
  `ORPOConfig`, `KTOConfig`, `GRPOConfig`,
  `LoRAConfig`, `QLoRAConfig`.
- **Runners:** `SFTRunner`, `PreferenceRunner`.
- **Result shapes:** `SFTRunResult`,
  `PreferenceRunResult`.
- **Type alias:** `AnyPreferenceConfig` =
  `DPOConfig` | `ORPOConfig` | `KTOConfig` |
  `GRPOConfig`.
- **Helpers:** `apply_chat_template`,
  `conversation_to_dicts`, `conversation_to_text`,
  `pack_sequences`, `PackedSequence`.
- **Progress streaming:** `ProgressEvent`, `JsonlProgressWriter`,
  `trainer_callback` — structured NDJSON metrics an orchestrator can tail
  while a run is in flight.

Module rules: [`src/strata_forge/training/CLAUDE.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/training/CLAUDE.md).
Source: [`src/strata_forge/training/`](https://github.com/GemovicNemanja/strata-forge/tree/main/src/strata_forge/training/).

---

## Contents

- [Quickstart](#quickstart)
- [SFT](#sft)
- [Preference tuning](#preference-tuning)
- [PEFT (LoRA / QLoRA)](#peft-lora--qlora)
- [Chat-template formatting](#chat-template-formatting)
- [Sequence packing](#sequence-packing)
- [Progress streaming](#progress-streaming)
- [Lazy-import contract](#lazy-import-contract)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```python
from strata_forge.training import LoRAConfig, SFTConfig, SFTRunner

sft = SFTConfig(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="./checkpoints/sft",
    max_seq_length=4096,
    num_epochs=3,
    learning_rate=2e-4,
    precision="bf16",
)
lora = LoRAConfig(r=32, alpha=64)

runner = SFTRunner(sft, peft_config=lora)
result = runner.train(train_dataset=your_dataset)  # HF Dataset
print(result.train_loss, result.train_runtime_s, result.output_dir)
```

For preference tuning:

```python
from strata_forge.training import DPOConfig, PreferenceRunner

dpo = DPOConfig(
    model_id="./checkpoints/sft",  # the SFT-tuned model
    output_dir="./checkpoints/dpo",
    beta=0.1,
    loss_type="sigmoid",
)
runner = PreferenceRunner(dpo)
result = runner.train(train_dataset=preference_dataset)
```

## SFT

`SFTConfig` mirrors the TRL `SFTConfig` knobs this module
exposes by default. The `extra_trainer_args` field is a
verbatim passthrough to TRL — the wrapper never blocks access to the
underlying surface.

Key fields:

| Field | Default | Notes |
|---|---|---|
| `model_id` | required | HF id or local path |
| `output_dir` | required | Where to save checkpoints + adapter |
| `dataset_text_field` | `"text"` | Column with formatted text; `None` for a custom collator |
| `max_seq_length` | 2048 | Per-example token budget |
| `packing` | `False` | Use TRL's online packing |
| `num_epochs` | 1.0 | Float — fractional epochs allowed |
| `per_device_batch_size` | 1 | |
| `gradient_accumulation_steps` | 8 | Effective batch = device × accumulation × N devices |
| `learning_rate` | 2e-4 | |
| `warmup_ratio` | 0.03 | Fraction of total steps |
| `precision` | `"bf16"` | Or `"fp16"` / `"fp32"` |
| `gradient_checkpointing` | `True` | |
| `save_steps` | 0 | 0 = no intermediate saves |
| `logging_steps` | 10 | How often loss is logged (and progress events emitted) |
| `seed` | 42 | Forwarded to `transformers.set_seed` |
| `progress_jsonl` | `None` | Path for NDJSON progress events — see [Progress streaming](#progress-streaming) |
| `extra_trainer_args` | `{}` | Verbatim passthrough |

`SFTRunner.train` does:

1. Lazy-imports `transformers` / `trl` / `datasets`.
2. Loads model + tokenizer via `AutoModelForCausalLM` /
   `AutoTokenizer` (or uses the `model=` / `tokenizer=`
   the caller provided).
3. Builds `trl.SFTTrainer` with the rendered kwargs + optional
   PEFT config.
4. Runs `trainer.train()`, calls `trainer.save_model`.
5. Returns an `SFTRunResult` with `train_loss`,
   `train_runtime_s`, etc.

## Preference tuning

Each method has its own config class so the type hints reflect
method-specific knobs (`loss_type` for DPO, `num_generations`
+ `reward_funcs` for GRPO, etc.). `PreferenceRunner`
reads `config.method` and dispatches to the right
`trl.XxxTrainer` / `trl.XxxConfig` pair.

```python
from strata_forge.training import (
    DPOConfig, ORPOConfig, KTOConfig, GRPOConfig, PreferenceRunner
)

# DPO — needs preference pairs (prompt, chosen, rejected).
dpo = DPOConfig(model_id="...", output_dir="...", beta=0.1)

# ORPO — preference pairs, no ref model.
orpo = ORPOConfig(model_id="...", output_dir="...", beta=0.1)

# KTO — binary signals per response.
kto = KTOConfig(
    model_id="...", output_dir="...",
    desirable_weight=1.0, undesirable_weight=1.0,
)

# GRPO — reward-driven, no preference pairs.
def my_reward(completions, **_):
    return [score(c) for c in completions]

grpo = GRPOConfig(
    model_id="...", output_dir="...",
    num_generations=8,
    max_completion_length=256,
)
runner = PreferenceRunner(grpo)
runner.train(train_dataset=prompt_only_dataset, reward_funcs=my_reward)
```

Forwarding rules:

- `ref_model` flows into the trainer only for `dpo` and
  `kto` (ORPO doesn't use a reference model; GRPO doesn't
  either).
- `reward_funcs` is **required** for `grpo` — omitting it raises. For
  `dpo`, `orpo`, and `kto` it is **silently dropped**, not rejected: it
  never reaches the trainer and no error is raised. Passing a reward
  function to DPO expecting a loud failure gets you a quiet no-op
  instead.
- `peft_config` flows into every trainer when set; QLoRA's
  bnb config is applied to `from_pretrained` automatically.

## PEFT (LoRA / QLoRA)

```python
from strata_forge.training import LoRAConfig, QLoRAConfig

lora = LoRAConfig(
    r=32, alpha=64, dropout=0.05,
    target_modules=("q_proj", "v_proj", "o_proj"),
)

qlora = QLoRAConfig(
    lora=LoRAConfig(r=64, alpha=16),
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16",
)
```

Both configs are Pydantic `frozen=True` with `extra="forbid"`.
`to_peft_config()` builds the actual `peft.LoraConfig`; for
QLoRA, `to_bnb_config()` additionally builds the
`transformers.BitsAndBytesConfig` (passed via
`quantization_config=` to `from_pretrained`).

## Chat-template formatting

```python
from strata_forge.llm import Message
from strata_forge.training import apply_chat_template, conversation_to_dicts

conversation = [
    Message.system("You are a helpful assistant."),
    Message.user("What's 2 + 2?"),
    Message.assistant("4."),
]

dicts = conversation_to_dicts(conversation)
# [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]

# With a HF tokenizer:
formatted = apply_chat_template(conversation, tokenizer=tokenizer)
token_ids = apply_chat_template(conversation, tokenizer=tokenizer, tokenize=True)
```

The formatter targets text-only training. Multimodal content
parts (image messages from `strata_forge.llm.multimodal`) collapse
to `[non-text: ClassName]` placeholders rather than being
silently dropped — a clear signal that vision-instruction
training needs its own path.

## Sequence packing

For SFT throughput on short conversations:

```python
from strata_forge.training import pack_sequences

tokenized = [tokenizer.encode(text) for text in texts]
packs = pack_sequences(
    tokenized,
    max_length=4096,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
for pack in packs:
    # Each pack is a fixed-length (max_length) tuple of token IDs.
    print(len(pack.input_ids), pack.source_indices)
```

The algorithm is greedy first-fit: each source sequence is
treated atomically (never split across packs); oversize
sequences are truncated to `max_length - 1` to leave room for
the EOS. The remaining slack at the end of each pack is filled
with pad tokens, and `attention_mask` is 1 for real tokens, 0
for pad.

## Progress streaming

A training run is opaque from the outside: TRL logs to stdout, and stdout
is a stream of prose. Set `progress_jsonl` on any config and both runners
attach a `TrainerCallback` that appends one JSON object per lifecycle
event to that file, so an orchestrator can `tail` it and render a real
progress bar.

```python
sft = SFTConfig(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="./checkpoints/sft",
    progress_jsonl="./progress.jsonl",
    logging_steps=10,
)
SFTRunner(sft).train(train_dataset=your_dataset)
```

Each line is a `ProgressEvent`: `kind` (one of `start`, `step`, `eval`,
`checkpoint`, `end`, `error`), plus `step`, `total_steps`, `epoch`,
`loss`, `learning_rate`, a `metrics` dict for anything else numeric, an
optional `message`, and a UTC `ts`. `step` events land on the
`logging_steps` cadence.

When `progress_jsonl` is unset, the runners fall back to the
`FORGE_PROGRESS_PATH` environment variable; when neither is set, no
callback is attached and nothing is written. The two lower-level pieces
are public if you want to drive them yourself: `JsonlProgressWriter(path)`
owns the append-only file, and `trainer_callback(writer)` builds the
`TrainerCallback` that feeds it.

This is also how `strata_forge.compute` backends surface live metrics —
point `progress_jsonl` at a file inside the job's workdir and read it back
with [`Backend.read_file`](compute.md#read_file-and-the-workdir-guard).

## Lazy-import contract

Importing `strata_forge.training` works without the `[finetuning]`
extra installed. The heavy deps are imported inside the methods
that need them:

- `transformers` / `trl` / `datasets` inside
  `SFTRunner._load_modules` and
  `PreferenceRunner._load_modules`.
- `peft` inside `LoRAConfig.to_peft_config`.
- `transformers` + `torch` inside
  `QLoRAConfig.to_bnb_config`.

When any of these is missing, the corresponding method raises
`ImportError` with the install hint
`pip install 'strata-forge[finetuning]'`.

## Troubleshooting

- **`ImportError: The [finetuning] extra is required`:** install
  the extra; on CPU-only macOS, expect lengthy wheel builds for
  `torch`.
- **OOM in SFT:** lower `per_device_batch_size`, raise
  `gradient_accumulation_steps`, enable
  `gradient_checkpointing` (default on), or switch to QLoRA.
- **Training metrics not surfaced:** `SFTRunResult` filters
  non-numeric metrics. Inspect the full `trainer.state.log_history`
  on the trainer instance returned by
  `SFTRunner.build_trainer` if you need everything.
- **GRPO `reward_funcs` confusion:** TRL accepts either a single
  callable or a list. The runner forwards verbatim — match TRL's
  expectations for the version you have installed.
- **`reward_funcs` had no effect on a DPO/ORPO/KTO run:** expected, and
  silent. Only GRPO consumes it.
- **DPO with PEFT + no `ref_model`:** TRL handles this by disabling
  the adapter on the base model to derive a reference. No extra
  config required; just don't pass `ref_model`.
- **Progress file never appears:** the callback attaches only when
  `progress_jsonl` is set on the config or `FORGE_PROGRESS_PATH` is in the
  environment. With neither, progress streaming is off.

---

## See also

- [`strata_forge.datasets`](datasets.md) — `to_hf_dataset` produces the
  HF `Dataset` the runners take.
- [`strata_forge.llm`](llm.md) — the `Message` / `AnyMessage` types
  `apply_chat_template` converts.
- [`strata_forge.compute`](compute.md) — running a trainer on a GPU host
  and tailing its progress file.
- [`strata_forge.storage`](storage.md) — pushing the finished checkpoint
  to the HF Hub.
- [`strata_forge.cli`](cli.md) — `strata-forge train sft` and
  `strata-forge train dpo`.
