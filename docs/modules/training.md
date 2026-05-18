# `forge.training` — SFT, preference tuning, PEFT, chat-template, packing

`forge.training` ships fine-tuning primitives: SFT via TRL's
:class:`SFTTrainer`, preference tuning (DPO, ORPO, KTO, GRPO)
via the TRL preference trainer family, and PEFT (LoRA, QLoRA)
adapters via peft. Supporting helpers cover chat-template
formatting (Forge :class:`AnyMessage` → HF tokenizer dict shape)
and sequence packing for SFT throughput.

The module wraps each TRL trainer with a typed Pydantic
config and a small runner class. Configs serialize cleanly to
YAML for reproducibility and ship across to remote
:mod:`forge.compute` runs; runners defer every heavy import
until ``train()`` is called.

Integration points:

- **Configs:** :class:`SFTConfig`, :class:`DPOConfig`,
  :class:`ORPOConfig`, :class:`KTOConfig`, :class:`GRPOConfig`,
  :class:`LoRAConfig`, :class:`QLoRAConfig`.
- **Runners:** :class:`SFTRunner`, :class:`PreferenceRunner`.
- **Result shapes:** :class:`SFTRunResult`,
  :class:`PreferenceRunResult`.
- **Type alias:** :data:`AnyPreferenceConfig` =
  :class:`DPOConfig` | :class:`ORPOConfig` | :class:`KTOConfig` |
  :class:`GRPOConfig`.
- **Helpers:** :func:`apply_chat_template`,
  :func:`conversation_to_dicts`, :func:`conversation_to_text`,
  :func:`pack_sequences`, :class:`PackedSequence`.

Module rules: [`src/forge/training/CLAUDE.md`](../../src/forge/training/CLAUDE.md).
Source: [`src/forge/training/`](../../src/forge/training/).

---

## Contents

- [Quickstart](#quickstart)
- [SFT](#sft)
- [Preference tuning](#preference-tuning)
- [PEFT (LoRA / QLoRA)](#peft-lora--qlora)
- [Chat-template formatting](#chat-template-formatting)
- [Sequence packing](#sequence-packing)
- [Lazy-import contract](#lazy-import-contract)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```python
from forge.training import LoRAConfig, SFTConfig, SFTRunner

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
from forge.training import DPOConfig, PreferenceRunner

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

:class:`SFTConfig` mirrors the TRL ``SFTConfig`` knobs Forge
exposes by default. The ``extra_trainer_args`` field is a
verbatim passthrough to TRL — Forge never blocks access to the
underlying surface.

Key fields:

| Field | Default | Notes |
|---|---|---|
| ``model_id`` | required | HF id or local path |
| ``output_dir`` | required | Where to save checkpoints + adapter |
| ``dataset_text_field`` | ``"text"`` | Column with formatted text; ``None`` for a custom collator |
| ``max_seq_length`` | 2048 | Per-example token budget |
| ``packing`` | ``False`` | Use TRL's online packing |
| ``num_epochs`` | 1.0 | Float — fractional epochs allowed |
| ``per_device_batch_size`` | 1 | |
| ``gradient_accumulation_steps`` | 8 | Effective batch = device × accumulation × N devices |
| ``learning_rate`` | 2e-4 | |
| ``warmup_ratio`` | 0.03 | Fraction of total steps |
| ``precision`` | ``"bf16"`` | Or ``"fp16"`` / ``"fp32"`` |
| ``gradient_checkpointing`` | ``True`` | |
| ``save_steps`` | 0 | 0 = no intermediate saves |
| ``seed`` | 42 | Forwarded to ``transformers.set_seed`` |
| ``extra_trainer_args`` | ``{}`` | Verbatim passthrough |

:class:`SFTRunner.train` does:

1. Lazy-imports ``transformers`` / ``trl`` / ``datasets``.
2. Loads model + tokenizer via ``AutoModelForCausalLM`` /
   ``AutoTokenizer`` (or uses the ``model=`` / ``tokenizer=``
   the caller provided).
3. Builds ``trl.SFTTrainer`` with the rendered kwargs + optional
   PEFT config.
4. Runs ``trainer.train()``, calls ``trainer.save_model``.
5. Returns an :class:`SFTRunResult` with ``train_loss``,
   ``train_runtime_s``, etc.

## Preference tuning

Each method has its own config class so the type hints reflect
method-specific knobs (``loss_type`` for DPO, ``num_generations``
+ ``reward_funcs`` for GRPO, etc.). :class:`PreferenceRunner`
reads ``config.method`` and dispatches to the right
``trl.XxxTrainer`` / ``trl.XxxConfig`` pair.

```python
from forge.training import (
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

- ``ref_model`` flows into the trainer only for ``dpo`` and
  ``kto`` (ORPO doesn't use a reference model; GRPO doesn't
  either).
- ``reward_funcs`` is required for ``grpo`` and rejected
  elsewhere.
- ``peft_config`` flows into every trainer when set; QLoRA's
  bnb config is applied to ``from_pretrained`` automatically.

## PEFT (LoRA / QLoRA)

```python
from forge.training import LoRAConfig, QLoRAConfig

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

Both configs are Pydantic ``frozen=True`` with ``extra="forbid"``.
``to_peft_config()`` builds the actual ``peft.LoraConfig``; for
QLoRA, ``to_bnb_config()`` additionally builds the
``transformers.BitsAndBytesConfig`` (passed via
``quantization_config=`` to ``from_pretrained``).

## Chat-template formatting

```python
from forge.llm import Message
from forge.training import apply_chat_template, conversation_to_dicts

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
parts (image messages from :mod:`forge.llm.multimodal`) collapse
to ``[non-text: ClassName]`` placeholders rather than being
silently dropped — a clear signal that vision-instruction
training needs its own path.

## Sequence packing

For SFT throughput on short conversations:

```python
from forge.training import pack_sequences

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
sequences are truncated to ``max_length - 1`` to leave room for
the EOS. The remaining slack at the end of each pack is filled
with pad tokens, and ``attention_mask`` is 1 for real tokens, 0
for pad.

## Lazy-import contract

Importing ``forge.training`` works without the ``[finetuning]``
extra installed. The heavy deps are imported inside the methods
that need them:

- ``transformers`` / ``trl`` / ``datasets`` inside
  :meth:`SFTRunner._load_modules` and
  :meth:`PreferenceRunner._load_modules`.
- ``peft`` inside :meth:`LoRAConfig.to_peft_config`.
- ``transformers`` + ``torch`` inside
  :meth:`QLoRAConfig.to_bnb_config`.

When any of these is missing, the corresponding method raises
:class:`ImportError` with the install hint
``pip install 'ai-forge[finetuning]'``.

## Troubleshooting

- **`ImportError: The [finetuning] extra is required`:** install
  the extra; on CPU-only macOS, expect lengthy wheel builds for
  ``torch``.
- **OOM in SFT:** lower ``per_device_batch_size``, raise
  ``gradient_accumulation_steps``, enable
  ``gradient_checkpointing`` (default on), or switch to QLoRA.
- **Training metrics not surfaced:** :class:`SFTRunResult` filters
  non-numeric metrics. Inspect the full ``trainer.state.log_history``
  on the trainer instance returned by
  :meth:`SFTRunner.build_trainer` if you need everything.
- **GRPO `reward_funcs` confusion:** TRL accepts either a single
  callable or a list. Forge forwards verbatim — match TRL's
  expectations for the version you have installed.
- **DPO with PEFT + no `ref_model`:** TRL handles this by disabling
  the adapter on the base model to derive a reference. No special
  Forge config required; just don't pass ``ref_model``.
