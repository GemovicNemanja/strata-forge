# `strata_forge.training` — SFT, preference tuning, sequence classification, PEFT, chat-template, packing

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
- **QLoRA precondition:** :func:`require_bitsandbytes`,
  :class:`MissingBitsAndBytesError`.
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
- [Sequence classification (`seq_cls`)](#sequence-classification-seq_cls)
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
verbatim passthrough to TRL — Forge never blocks a Python caller's
access to the underlying surface (a run spec is narrower; see
[Driving training from a declaration](#driving-training-from-a-declaration)).

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
   the caller provided), on the terms in
   [Loading from the Hub](#loading-from-the-hub).
3. Builds ``trl.SFTTrainer`` with the rendered kwargs + optional
   PEFT config.
4. Runs ``trainer.train()``, calls ``trainer.save_model``.
5. Returns an :class:`SFTRunResult` with ``train_loss``,
   ``train_runtime_s``, etc.

## Loading from the Hub

Every model and tokenizer load the runners make states its terms
explicitly, from :mod:`strata_forge.training.loading`
(``model_load_kwargs`` / ``tokenizer_load_kwargs``):

| Argument | Value | Why |
|---|---|---|
| ``trust_remote_code`` | ``False`` | A repo's own Python never runs. |
| ``use_safetensors`` | ``True`` (models) | A pickle checkpoint (``*.bin`` / ``*.pt``) can execute code when unpickled; a repo that ships only those fails to load instead. |
| ``token`` | the caller's | ``build_trainer(..., token=...)`` / ``train(..., token=...)``. |
| ``revision`` | the caller's pin, when given | ``build_trainer(..., revision=...)``: a full commit id reads exactly that commit. |

A load that reads a checkpoint into a model class it must fill
completely also passes ``output_loading_info=True`` and hands the
report to ``check_loading_info(info, head_prefixes=...)``. It raises
:class:`LoadRefusedError` (a ``ForgeError``) for a missing weight (the
library would have initialised it at random), a shape mismatch, a
reported load error, or an ignored checkpoint tensor under one of
``head_prefixes``, naming the keys; it returns the other ignored
checkpoint keys (a pooler, a ``position_ids`` buffer) so the caller can
record them. The inference runner's ``hf_classify`` engine uses it with
``classifier.`` / ``score.`` / ``head.``.

``token`` is a :data:`HubToken`: a token string, ``False`` to send
no credential at all, or ``None`` (the default) for the library's
own lookup (an ``HF_TOKEN`` variable or a cached login). It is a
keyword argument of the load and is **never stored on a config**:
every ``XxxConfig`` rejects a ``token`` field, and nothing in
``to_trl_kwargs`` carries one, so it cannot reach TRL's arguments or
the ``training_args.bin`` TRL pickles beside each checkpoint (a
pushed output directory would carry that file to the Hub).

The VM runner in :mod:`strata_forge.pipelines.finetune_runner`
passes the run's delivered token, or ``False`` when none was
delivered, to every load, including the merge step's base model,
adapter and tokenizer, so a run never falls back to a credential
the machine happens to hold.

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
- When the runner loads the policy itself for ``dpo`` / ``kto``
  **without** an adapter, it loads the reference model too, with
  the same load arguments (token, no remote code, safetensors
  only, same precision). Left to TRL, the reference is re-downloaded
  by the policy's name with none of them, so a gated base would fail
  there. With an adapter (TRL disables it to recover the reference)
  or ``precompute_ref_log_probs`` no reference is loaded. A caller
  that passes ``model=`` owns that load and should pass
  ``ref_model=`` with it. Both models load with no device placement
  of their own, as the policy always has, so a full-weight DPO / KTO
  run holds two copies of the base in host memory until the trainer
  moves them to the accelerator (TRL placed its own reference load
  with ``device_map="auto"``): size host RAM for twice the model.
- ``reward_funcs`` is required for ``grpo`` and rejected
  elsewhere.
- ``peft_config`` flows into every trainer when set; QLoRA's
  bnb config is applied to ``from_pretrained`` automatically.

## Sequence classification (`seq_cls`)

`SeqClsConfig` + `SeqClsRunner` (`seq_cls.py`) train a premise/hypothesis classifier on
`transformers.Trainer`: a stock `AutoModelForSequenceClassification` primary head plus optional
auxiliary heads, from a local safetensors checkpoint directory and a dataset whose columns are the
`nli_pair` roles. The registry entry is `seq_cls` (`task_type="SEQ_CLS"`, format `nli_pair`); it
trains full weights only (a LoRA or QLoRA adapter is refused).

```python
from strata_forge.training import SeqClsConfig, SeqClsRunner

config = SeqClsConfig(
    model_id="org/base",                     # recorded; the runner loads `checkpoint`
    output_dir="/data/run/checkpoints",      # intermediate checkpoints only
    labels=("entailment", "contradiction", "neutral"),
    two_way_positive_label="entailment",
    two_way_labels=("entailed", "not_entailed"),
    head_init="decoder_lm_head",
    verbalizers={"entailment": " True", "contradiction": " False", "neutral": " Unknown"},
    brier_weight=0.1,
)
result = SeqClsRunner(config).run(
    checkpoint=local_snapshot, artifact_dir=out, train_dataset=train, eval_dataset=dev,
    on_receipt=lambda name, receipt: print(name, receipt),
)
```

### Rows

| Role | Cell | Null |
|---|---|---|
| `premises` | a non-empty list of `{pid, text}` whose pids are `p1`, `p2`, ... in order (a string or a list of strings is refused) | refused |
| `hypothesis` | a string | refused |
| `label` | three-way: a declared label, an alias, or an integer through `label_index_map` (integer labels need one, because index orders differ); two-way: exactly one of the declared `two_way_labels` | refused |
| `label_space` | `3way` or `2way_entail_vs_not` | refused (unmapped: every row is `3way`) |
| `label_dist` | `len(labels)` non-negative numbers summing to 1 (three-way rows) | the one-hot label |
| `inconsistent_premises` | a boolean | masked |
| `support_set` | a list of the row's premise ids | masked |

`premise_format="ids"` renders each premise as the line `"{pid}: {text}"` and marks it at the first
token overlapping the pid's first character (the support head needs this); `plain` renders the text
alone. An encoder sees the tokenizer's pair encoding of (premises, hypothesis); a decoder sees
`decoder_template` (default `"{premises}\n{hypothesis}"`) with the tokenizer's own special tokens,
and its answer position is the final token. Rows are never truncated.

### Objective

`classifier_loss` (`classifier_loss.py`), vectorised over the batch. A three-way row contributes
`CE + brier_weight * Brier` against its target distribution (Brier in the sum form). A two-way row
asks only whether the positive label holds. Both halves are declared, with no default: which label
the positive is (`two_way_positive_label`) and how a row spells its positive and negative
(`two_way_labels`, e.g. `("entailed", "not_entailed")`); a split with a two-way row and neither
declared refuses. The negative is the aggregate of every other label (with three labels,
P(not_entailed) = P(C) + P(U)), `log(1 - q) = LSE(z_neg) - LSE(z)`, so the negative
labels are never trained against each other, and its Brier is `2 (q - t)^2`. The primary term is
the mean over every row. The flag head adds `flag_loss_weight` times the mean BCE over the rows
whose flag is not null; the support head adds `support_loss_weight` times the mean, over eligible
rows, of each row's mean BCE over its premises. A term with no eligible row is exactly zero.
Every mean is over the declared batch, which is why `gradient_accumulation_steps` is refused by
name: accumulating would change the denominators. There is no label smoothing and no class or
sample weight. There is no named preset: the weights are plain parameters (research runs use
`brier_weight=0.1`, `flag_loss_weight=1.0`, `support_loss_weight=1.0`), and each run's manifest
records the three weights beside `LOSS_FORM`, the objective in words.

### Heads and the load guard

`head_init` declares where the primary head comes from (`classifier_heads.py`):

| `head_init` | Base | Allowed missing keys | Initialisation |
|---|---|---|---|
| `new` | encoder (`modernbert`) | exactly `classifier.weight`, `classifier.bias` | stock random classifier |
| `decoder_lm_head` | decoder (`modernbert-decoder`) | exactly the prediction head's parameters and `classifier.weight` | `lm_head.*` -> `head.*`; classifier rows and bias = the verbalizer rows of `decoder.weight` (or the tied embedding) and `decoder.bias` |
| `existing` | a checkpoint that already has this classifier (`modernbert` or `modernbert-decoder`) | none; its `id2label` must equal `labels` | as saved; auxiliary heads load strictly; a decoder keeps template rendering and last-token pooling, and a rendering that differs from the saved `strata_verifier.json` (premise format, decoder template) refuses |

Any other missing key, any unexpected key outside the architecture's declared set and any shape
mismatch refuses the run. The tables are explicit per `model_type`; there is no key-name heuristic
and no repository code. The stock decoder classifier has no bias, so `decoder_lm_head` rebuilds it
with one to carry `decoder.bias`; because that architecture's `classifier_bias` also gives the
prediction head's dense layer a bias, that one is added as zero and frozen, and listed in
`strata_verifier.json` as a frozen parameter (an `existing` run re-freezes it). The saved model
then reloads with every key present and computes the same function.

The auxiliary heads exist exactly when their role is mapped. Both read one pooled vector, the first
token for an encoder and the answer position for a decoder: the flag head is `Linear(hidden, 1)`,
the support head is `h^T W m_pid + b` over each premise's marker token. They start from
`normal(0, initializer_range)` with a zero bias and are saved as `strata_aux_heads.safetensors`.

### Checks before step 0

`SeqClsRunner.prepare` runs these in order, calling `on_receipt(name, receipt)` as each passes;
a failure raises `PreflightError` (or `Step0Error`) carrying its receipt. The two checks that
compare logits (step 0, and reload parity below) run both sides in float32 with TF32 matmul and
cuDNN switched off for the comparison, and require the same predicted label on every compared row
besides the tolerance.

1. **`overflow`**: every row resolves its label (a soft `label_dist` must be finite, non-negative
   and sum to 1) and tokenises without truncation; a refusal names the row index and the role,
   never the cell; a row longer than
   `max_length` refuses the run with the count and the first row indices. The receipt records the
   row count, the over-cap count, the template's token overhead (special tokens plus template
   literals), and the per-row token-length distribution (`min`, `p50`, `p95`, `max`, nearest
   rank, overhead included).
2. **`load`**: the load guard above.
3. **`step0`** (`decoder_lm_head`): on the first `step0_rows` training rows, the classifier's label
   logits must equal the base model's own causal-LM logits for the verbalizer tokens within
   `step0_tolerance` (absolute, float32, TF32 off) with the same predicted label on every row. The
   receipt records the largest logit and probability differences, the agreeing rows and the
   resolved verbalizer token ids; every verbalizer must be one token.
4. **`memory`**: one full optimizer step at the split's longest real batch (the
   `per_device_batch_size` longest rows, padded together), under the run's precision and
   activation-checkpointing setting, measured by the CUDA allocator's peak and then undone exactly
   (weights restored, gradients and optimizer state dropped). With an eval split, the in-run
   evaluation is measured as well: a forward pass at the eval batch size over the eval split's
   longest rows while the optimizer state is still allocated, as when the trainer evaluates
   mid-run. The larger peak above `memory_fraction` of the device, or an out-of-memory error,
   refuses the run with the measured peaks. Off a GPU the step still runs and the peaks are
   recorded as unmeasured.

The run also refuses a machine exposing more than one GPU, since the trainer would split the batch.

### Training and the artifact

`bf16` precision is autocast over float32 master weights. `gradient_checkpointing=True` enables
`torch.utils.checkpoint` with `use_reentrant=False` and `preserve_rng_state=True`; it is off by
default. Batches are grouped by length (`group_by_length`). An eval split is evaluated each epoch
(or every `eval_steps`). `save_steps > 0` writes intermediate checkpoints to `output_dir`, never to
the artifact directory. The artifact holds the stock checkpoint (safetensors, `id2label` in label
order), the tokenizer, `strata_aux_heads.safetensors`, and `strata_verifier.json`: the label order,
aliases and index map, the pooling position, the premise and template rendering (and the
template's SHA-256), `max_length`, the verbalizers and their token ids, the trained outputs and the
frozen parameters, so an evaluator renders rows exactly as training did.

### After training: reload parity and calibration

`SeqClsRunner.finish` checks the saved artifact before anything else reads it
(`artifact_checks`). Every parameter the method holds at zero (the decoder's `head.dense.bias`) is
asserted exactly zero at save and again after reload. The artifact is reloaded the way a scorer
loads it: stock `AutoModelForSequenceClassification` from its own safetensors, no remote code,
float32, under `check_loading_info`. Its logits on a fixed set of rows (the first `step0_rows` of
the eval split, else of the training split) must equal the trained classifier's in-memory float32
logits within `step0_tolerance`, with the same predicted label on every row (`reload_parity`
receipt). A failure raises `ArtifactCheckError`; the run is refused and nothing is pushed. When a
decoder's classifier gained a bias, the load receipt records `classifier_bias: true` and why.

With a declared calibration split (the run spec's `calibration_split`, an in-distribution dev
split), one temperature is then fitted on the reloaded model's float32 logits over the split's
single-label three-way rows (two-way and soft-label rows are excluded and counted) with
`strata_forge.metrics.calibration_fit.fit_temperature`, the same fit the evaluation runner uses,
and written to `calibration.json` in the artifact: the fit (temperature, NLL before and after, a
boundary hit), the split and its dataset commit, the row counts, and `rows_sha256` (SHA-256 over
one JSON line `[input_ids, label]` per fitted row, in split order). An engine without that fit
refuses a spec that declares a calibration split.

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

QLoRA needs **bitsandbytes** on the machine that loads the model: transformers quantises the
weights inside ``from_pretrained`` and refuses to start without it. The ``[finetuning]`` extra
installs it (``bitsandbytes>=0.49,<1``), and it is also the one dependency whose absence would
otherwise surface last, because ``BitsAndBytesConfig`` constructs without it. So the check is made
up front instead:

```python
from strata_forge.training import MissingBitsAndBytesError, require_bitsandbytes

require_bitsandbytes()  # MissingBitsAndBytesError when it is missing or below 0.49
```

``to_bnb_config()`` calls it before building anything, the ``finetune_runner`` calls it while
validating the spec (before the dataset download) for ``adapter="qlora"``, and the
``strata-forge train ... --adapter qlora`` commands call it before resolving the dataset.
:class:`MissingBitsAndBytesError` is a ``ForgeError`` and an ``ImportError``. The check looks the
package up without importing it, since importing it loads its native library, and reads its version
from the distribution's metadata: a bitsandbytes below the extra's floor (one installed outside the
extra, which pip would otherwise have upgraded) is refused the same way, naming the installed
version. A version the metadata does not record is not refused.

Why ``0.49``: the floor has to carry a CUDA 13.0 kernel (what the ``torch`` pin resolves to on
Linux; first shipped in 0.48) and publish a wheel on each platform that ``torch>=2.9`` publishes a
Python 3.14 wheel for (Linux x86_64 and aarch64, Windows x64, macOS arm64; the macOS wheel first
shipped in 0.49). Because every one of those platforms has a wheel, the pin needs no platform
marker, and QLoRA is not ruled out on any of them (bitsandbytes lists CUDA, Apple MPS and CPU among
its devices). The cost is macOS 12 and 13: bitsandbytes' macOS wheel needs macOS 14 and it
publishes no sdist to build from, so an Apple Silicon Mac on 12 or 13, where ``torch`` would settle
on an older macOS 11 wheel, cannot install ``[finetuning]``. Both are past Apple's support window.
A marker such as ``sys_platform != 'darwin' or platform_release >= '23'`` would keep them
installable without QLoRA; it is not used, since those releases are out of support and the marker
would be one more thing a resolver has to get right.

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
[m.name for m in enabled_methods()]  # ["sft", "dpo", "orpo", "kto", "seq_cls"]
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
| `nli_pair` | `premises`, `hypothesis`, `label` | `label_space`, `label_dist`, `inconsistent_premises`, `support_set` | seq_cls |

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
equally. A KTO `label` cell is coerced to a real boolean rather than trusted (an `nli_pair` label
is left as it is; the classifier resolves it through its declared labels): a column of non-empty
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

**A spec's `extra_trainer_args` may not set the TRL knobs that act under other terms than the
run's.** It is TRL's and `transformers`' whole argument surface, and some of it loads, pushes or
reports outside everything the runner sets on its own loads (see
[Loading from the Hub](#loading-from-the-hub)): `chat_template_path` (a tokenizer load of any repo
or local path, with the machine's own credential), `trust_remote_code` and `model_init_kwargs` /
`ref_model_init_kwargs` (remote code, or a load configured elsewhere), `push_to_hub` and every
`hub_*` / `trackio_*` setting (a push with the machine's own credential), `resume_from_checkpoint`,
`deepspeed`, `fsdp_config` and `accelerator_config` (files on the machine, unpickled in the
checkpoint case), `logging_dir` (outside the artifact directory) and `report_to` (a third-party
service). From a spec each is refused by name
(`hyperparams may not set extra_trainer_args.…`), all of them in one message; every other key
passes through.

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

The nightly clean-install job (`.github/workflows/clean-install.yml`) is the
control for that. Every night it installs, into empty Python 3.14 venvs with no
lockfile and no constraints, each extras string a run's machine installs
(`[finetuning,storage]`, `[serving,storage,hf]`) plus `[all]`, and then
`scripts/smoke_clean_install.py` builds every enabled method x dataset format x
adapter through the runner's own spec path and its own `build_trainer`, down to
the TRL config, the peft config, the bitsandbytes config and the trainer call,
whose keywords are bound against the installed trainer's signature (a model and
a tokenizer are passed in, so nothing is downloaded). A renamed kwarg fails there
on the day the release ships. A QLoRA case first checks the machine can
quantise: it fails when `[finetuning]` stops naming `bitsandbytes` or the install
lacks it, before it builds anything, so the gap is reported as itself rather than
as the runner's own refusal. It also runs on any branch push (main excepted) that
changes `pyproject.toml`, so a pin change is resolved fresh before it merges.

Because it executes upstream code no lockfile vouches for, the install never
runs in the default branch's context, whose Actions cache every ref restores:
the nightly schedule (which GitHub runs from `main`) dispatches the job on `dev`
and reports that run's result. `release.yml`'s build and the `nightly.yml` jobs
do not restore the Actions cache at all.

## Lazy-import contract

Importing ``strata_forge.training`` works without the ``[finetuning]``
extra installed. The heavy deps are imported inside the methods
that need them:

- ``transformers`` / ``trl`` / ``datasets`` inside
  :meth:`SFTRunner._load_modules` and
  :meth:`PreferenceRunner._load_modules`.
- ``peft`` inside :meth:`LoRAConfig.to_peft_config`.
- ``transformers`` + ``torch`` inside
  :meth:`QLoRAConfig.to_bnb_config`, after
  :func:`require_bitsandbytes` has looked ``bitsandbytes`` up
  (never imported: transformers does that when it quantises).

When any of these is missing, the corresponding method raises
:class:`ImportError` with the install hint
``pip install 'strata-forge[finetuning]'`` (for ``bitsandbytes``,
the :class:`MissingBitsAndBytesError` subclass).

## Troubleshooting

- **`ImportError: The [finetuning] extra is required`:** install
  the extra; on CPU-only macOS, expect lengthy wheel builds for
  ``torch``.
- **`adapter 'qlora' needs bitsandbytes`:** the machine has the
  training stack but not ``bitsandbytes`` (an install that predates
  it joining ``[finetuning]``, or one made without the extra).
  Reinstall ``strata-forge[finetuning]``, or train with
  ``adapter="lora"``. When the message names an installed version,
  that ``bitsandbytes`` is older than the extra's floor; upgrade it
  with the ``pip install`` line the message prints.
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

## Run telemetry

`strata_forge.training.progress` is the channel an orchestrator tails (one `ProgressEvent` per
JSONL line, read via `Backend.read_file` — see
[ADR 0016](../architecture/adr/0016-backend-read-file.md)). Beyond the step counters it carries two
things a run *watcher* needs, decided in
[ADR 0017](../architecture/adr/0017-run-telemetry-on-the-progress-channel.md).

**`stage` — where the run is, as an id.** One of `RUN_STAGES`:

```
provision → install_engine → load_model → run → push
```

`stage` is independent of `kind`: `kind` says what sort of record an event is, `stage` says where in
the run it sits. Render an ordered stepper from `RUN_STAGES` and `stage` — never by matching on
`message`, whose wording is prose and free to change.

A runner reports only the last three. `provision` and `install_engine` describe the VM *before the
runner's process exists*, so the orchestrator that submitted the job owns them. Dataset loading
reports as `load_model`: not literally the model, but the same "getting ready" milestone from the
watcher's side.

**`metrics` — the numbers.** A free-form `dict[str, float]`, so a new counter is a key rather than a
schema change and an older consumer ignores what it does not recognise.

| Key | Where it comes from |
|---|---|
| `grad_norm`, `eval_*`, … | whatever the TRL/`transformers` log dict carried, minus the promoted fields |
| `steps_per_s` | derived from the wall clock between logged steps |
| `tokens_per_s` | training: derived, *only* when the trainer counts tokens (`include_num_input_tokens_seen`). Batch inference: from each row's `usage` |
| `succeeded`, `failed` | batch inference, per chunk |
| `rows_per_s`, `latency_p50_ms` | batch inference, from each row's own `LLMResponse` |
| `gpu_count`, `gpu_util_pct`, `gpu_mem_used_mb`, `gpu_mem_total_mb`, `gpu_temp_c` | `strata_forge.training.hardware` |

A `phase` event may carry `stage` and `metrics` too — neither is a step count, and both stay true
during exactly the long uncountable stretches where nothing else does.

### GPU counters

`hardware.GpuSampler.sample()` **never blocks**: it returns the last reading and hands a refresh
to a daemon thread when that reading has gone stale (`min_interval_s`, 5s default). Three
properties are deliberate:

- **No new dependency.** `pynvml` would be tidier in-process, but forge's dependencies install
  *fresh on the user's VM* at the start of every run, so each pin is another package that can
  publish a breaking release between a green CI run and someone's four-hour fine-tune.
- **It never raises.** No binary, no driver, a timeout, a changed CSV shape, a CPU-only box — every
  path yields no counters. A missing gauge is cosmetic; an exception out of a metrics call is not.
- **It never hangs the run.** Callers are coroutines, and `nvidia-smi` can block for an unbounded
  time — `subprocess.run(timeout=...)` does *not* bound it, because its POSIX timeout path kills
  the child and then waits on it with no timeout, which never returns for the uninterruptible
  `D` state a wedged driver produces. Sampling therefore happens off the loop, and the probe
  abandons an unkillable child rather than waiting for it.

On a multi-GPU box: mean utilization, summed memory, **max** temperature. Max because one card
cooking is the fact worth surfacing, and a mean would hide it behind its healthy neighbours.
Missing values are handled per *field*, so a card that reports `[N/A]` for temperature alone still
contributes its utilization and memory. A failed refresh keeps the last good reading rather than
blanking the gauge.

Non-finite numbers never reach the channel. `loss=nan` and `grad_norm=inf` are routine in fp16, and
pydantic serializes them to JSON `null` — which `metrics: dict[str, float]` then refuses to parse
back, so an ordinary gradient explosion would emit a line the orchestrator could not read.
`coerce_float` drops them, and metric keys are capped in length and count.
