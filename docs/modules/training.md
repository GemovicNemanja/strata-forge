# `strata_forge.training` — SFT, preference tuning, PEFT, chat-template, packing

`strata_forge.training` ships fine-tuning primitives: SFT via TRL's
:class:`SFTTrainer`, preference tuning (DPO, ORPO, KTO, GRPO)
via the TRL preference trainer family, and PEFT (LoRA, QLoRA)
adapters via peft. Supporting helpers cover chat-template
formatting (Forge :class:`AnyMessage` → HF tokenizer dict shape)
and sequence packing for SFT throughput.

The module wraps each TRL trainer with a typed Pydantic
config and a small runner class. Configs serialize cleanly to
YAML for reproducibility and ship across to remote
:mod:`strata_forge.compute` runs; runners defer every heavy import
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
- **Declaration layer:** :data:`METHODS` / :func:`pick_method` /
  :func:`enabled_methods` / :func:`check_format` and
  :data:`FORMATS` / :func:`validate_mapping` /
  :func:`build_training_rows` — for callers that hold a job spec
  rather than Python. See
  [Driving training from a declaration](#driving-training-from-a-declaration).

Module rules: [`src/strata_forge/training/CLAUDE.md`](../../src/strata_forge/training/CLAUDE.md).
Source: [`src/strata_forge/training/`](../../src/strata_forge/training/).

---

## Contents

- [Quickstart](#quickstart)
- [SFT](#sft)
- [Preference tuning](#preference-tuning)
- [PEFT (LoRA / QLoRA)](#peft-lora--qlora)
- [Chat-template formatting](#chat-template-formatting)
- [Sequence packing](#sequence-packing)
- [Driving training from a declaration](#driving-training-from-a-declaration)
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
| ``warmup_ratio`` | 0.03 | Fraction of total steps; reaches TRL as ``warmup_steps`` |
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

- ``ref_model`` flows into the trainer only for ``dpo`` and
  ``kto`` (ORPO doesn't use a reference model; GRPO doesn't
  either).
- ``reward_funcs`` is required for ``grpo`` and rejected
  elsewhere.
- ``peft_config`` flows into every trainer when set; QLoRA's
  bnb config is applied to ``from_pretrained`` automatically.

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

Both configs are Pydantic ``frozen=True`` with ``extra="forbid"``.
``to_peft_config()`` builds the actual ``peft.LoraConfig``; for
QLoRA, ``to_bnb_config()`` additionally builds the
``transformers.BitsAndBytesConfig`` (passed via
``quantization_config=`` to ``from_pretrained``).

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
parts (image messages from :mod:`strata_forge.llm.multimodal`) collapse
to ``[non-text: ClassName]`` placeholders rather than being
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
sequences are truncated to ``max_length - 1`` to leave room for
the EOS. The remaining slack at the end of each pack is filled
with pad tokens, and ``attention_mask`` is 1 for real tokens, 0
for pad.

## Driving training from a declaration

The API above assumes a caller writing Python: it picks a config class, constructs a runner, and
hands it a `Dataset`. An orchestrator cannot do any of that — it holds a JSON job spec that was
allowed to carry data and nothing else. Two small modules close that gap, and
[`strata_forge.pipelines.finetune_runner`](../../src/strata_forge/pipelines/finetune_runner.py) is
their first consumer.

**The method registry** (`methods.py`) is the single table mapping a method NAME to what it is:
its config class, its runner class, the head it trains (`task_type`), and the dataset formats it
accepts. `pick_method(name)` is the only supported way in.

```python
from strata_forge.training import pick_method, check_format, enabled_methods

spec = pick_method("dpo")          # UnsupportedMethodError for an unknown method
check_format(spec, "preference")   # ...or one the method cannot train on
[m.name for m in enabled_methods()]  # ["sft", "dpo", "orpo", "kto"]
```

`GRPO` is registered with `enabled=False`. It is fully implemented in
:class:`PreferenceRunner`, but it needs `reward_funcs` — callables — and a spec that carries no
code cannot supply one. `pick_method("grpo")` therefore raises with that explanation rather than
behaving like an unknown method. Reach for :class:`PreferenceRunner` directly to use it.

**The dataset formats** (`dataset_format.py`) declare which of a dataset's columns plays which
role, because TRL decides what kind of run it is doing by looking at the column NAMES it was
handed and real datasets never use those names.

| Format | Required roles | Optional | Methods |
|---|---|---|---|
| `text` | `text` | — | sft |
| `prompt_completion` | `prompt`, `completion` | — | sft |
| `conversational` | `messages` | — | sft |
| `preference` | `chosen`, `rejected` | `prompt` | dpo, orpo |
| `unpaired_preference` | `prompt`, `completion`, `label` | — | kto |

```python
from strata_forge.training import validate_mapping, build_training_rows

mapping = {"prompt": "question", "completion": "answer"}
validate_mapping("prompt_completion", mapping, dataset.column_names)  # before anything expensive
rows = build_training_rows(dataset, "prompt_completion", mapping)
```

`validate_mapping` is the point of the whole thing: it checks the declaration against the split's
real columns and names the missing role, the bad column and what is available. Without it a wrong
mapping is a TRL `KeyError` on a rented GPU, minutes into a job that has already downloaded a
model. `build_training_rows` then projects each row onto the roles and **drops every other
column** — a leftover `id` or `source` is not inert, it can change the format TRL infers.

`preference` keeps `prompt` optional because TRL accepts both spellings (an explicit prompt beside
the two completions, or the prompt embedded in both) and datasets in the wild use each about
equally. A `label` cell is coerced to a real boolean rather than trusted: a column of non-empty
strings would otherwise read as every-row-true, which trains a model on the premise that nothing
is bad — silently wrong rather than failed.

**A spec's `hyperparams` reach the method's config verbatim, with three exceptions.** The config's
own `extra="forbid"` decides what each method accepts, so an inapplicable knob is a named error
rather than a silent drop, and no second list of field names has to be kept in sync. But
`extra="forbid"` only rejects keys the config does not declare — it waves through
**`model_id`, `output_dir` and `progress_jsonl`**, which the runner *derives*: `model_id` is the
value it validated and the value the run is recorded as, `output_dir` is the artifact directory the
push/merge/cleanup paths address, and `progress_jsonl` is the orchestrator's channel. A spec naming
one of them is refused (`hyperparams may not set …: the runner derives these`), because otherwise a
submitted hyperparam could train a different model than the record names and write checkpoints and
the progress log to any absolute path — walking past the very `validate_repo_id` re-check the VM
side exists to perform. `extra_trainer_args` is checked for the same three keys, since
`to_trl_kwargs` applies it last and it reaches TRL's own `output_dir`.

This bounds only the **declaration** path. `extra_trainer_args` remains an unrestricted escape
hatch for a caller driving `SFTRunner` / `PreferenceRunner` from Python — there the caller *is* the
operator, and Forge does not block access to the underlying TRL surface. The distinction is who
submitted the values, not what they are.

## The upstream contract

A config renders to a kwargs dict that is splatted into TRL's constructor, so the
names it emits are part of this package's contract with a specific pair of upstream
majors — pinned in `pyproject.toml` as `transformers>=5.5.3,<6` and `trl>=1.0,<2`,
and frozen as an explicit key set in `tests/unit/training/test_trl_contract.py`.
Two places where the emitted name deliberately differs from the field:

- ``warmup_ratio`` reaches TRL as **``warmup_steps``**. transformers 5 removed
  `warmup_ratio` and folded it into `warmup_steps`, which reads a float in
  ``[0, 1)`` as exactly that fraction — identical semantics, so the field keeps
  the name that describes what it is.
- ORPO's config/trainer are resolved from **``trl.experimental.orpo``** when the
  top-level namespace does not export them, which is where TRL 1.x moved them.
  That namespace carries no semver promise, so ORPO is the one method the
  ``<2`` ceiling does not fully protect.

``max_prompt_length`` is **gone** from the preference configs: TRL removed it
across 0.27–0.29 with no replacement. Bound prompt length by filtering the
dataset before training; ``max_length`` still applies to the full sequence.

These pins are not decoration. This package is `pip install`-ed fresh onto a
user's VM at run time with no lockfile, so an unbounded specifier means every
run resolves against whatever shipped that morning, and a breaking upstream
release surfaces as a crash on the user's hardware rather than a red build.

## Lazy-import contract

Importing ``strata_forge.training`` works without the ``[finetuning]``
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
