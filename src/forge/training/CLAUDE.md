# Agent rules — forge.training

`forge.training` ships fine-tuning primitives: SFT, preference
tuning (DPO / ORPO / KTO / GRPO), and PEFT (LoRA / QLoRA), plus
the supporting chat-template formatting and sequence-packing
helpers. The module wraps TRL / peft / transformers with typed
Pydantic configs and runner classes that defer all heavy imports
until ``train()`` is called.

## Purpose

- :class:`SFTConfig` + :class:`SFTRunner` + :class:`SFTRunResult`
  — TRL ``SFTTrainer`` wrapper.
- :class:`DPOConfig` / :class:`ORPOConfig` / :class:`KTOConfig` /
  :class:`GRPOConfig` + :class:`PreferenceRunner` +
  :class:`PreferenceRunResult` — preference-tuning dispatcher
  over TRL's family of preference trainers.
- :class:`LoRAConfig` / :class:`QLoRAConfig` — adapter configs
  with sensible Forge defaults; both expose ``to_peft_config``,
  QLoRA additionally exposes ``to_bnb_config``.
- :func:`apply_chat_template` / :func:`conversation_to_dicts` /
  :func:`conversation_to_text` — bridge Forge :class:`AnyMessage`
  conversations into the formats HF tokenizers and TRL want.
- :func:`pack_sequences` + :class:`PackedSequence` — greedy
  first-fit sequence packing for SFT throughput.

## Boundaries

- **Owns:** ``sft.py``, ``preference.py``, ``peft.py``,
  ``chat_template.py``, ``packing.py``.
- **Imports from inside ``forge``:** :mod:`forge.llm.messages`
  for the conversation shapes consumed by
  :mod:`forge.training.chat_template`. Nothing else.
- **Does NOT import** :mod:`forge.tracing`, :mod:`forge.agents`,
  :mod:`forge.evals`, :mod:`forge.rag`, :mod:`forge.datasets`,
  :mod:`forge.compute`. Trainer scripts that ride on
  :mod:`forge.compute` live in ``examples/`` or in
  user-controlled scripts; the dependency arrow stays unbroken.
- **External deps:** Pydantic in the core install. ``torch``,
  ``transformers``, ``trl``, ``peft``, ``datasets``,
  ``accelerate`` live behind the ``[finetuning]`` extra and are
  lazy-imported inside the runner methods that touch them.

## Public API

The module's ``__init__.py`` re-exports:

- Configs: :class:`SFTConfig`, :class:`DPOConfig`,
  :class:`ORPOConfig`, :class:`KTOConfig`, :class:`GRPOConfig`,
  :class:`LoRAConfig`, :class:`QLoRAConfig`.
- Runners: :class:`SFTRunner`, :class:`PreferenceRunner`.
- Result types: :class:`SFTRunResult`,
  :class:`PreferenceRunResult`.
- Type alias: :data:`AnyPreferenceConfig`.
- Helpers: :func:`apply_chat_template`,
  :func:`conversation_to_dicts`, :func:`conversation_to_text`,
  :func:`pack_sequences`, :class:`PackedSequence`.

Errors raised from this module are :class:`ValueError` for
input validation or :class:`ImportError` when the
``[finetuning]`` extra is missing.

## Internal patterns

- **Configs are declarative data.** Pydantic ``frozen=True``,
  ``extra="forbid"``, ``Field(default=..., ...)`` everywhere.
  Each config has a ``to_trl_kwargs()`` (or ``to_peft_config()``)
  method that renders the typed shape into the kwargs TRL / peft
  expects. ``extra_trainer_args`` is the verbatim passthrough
  escape hatch.
- **Runners are thin orchestration.** No work in the constructor.
  ``_load_modules()`` lazy-imports the heavy stack; ``train()``
  builds the trainer, runs it, saves, and returns a Pydantic
  result. Callers can supply ``model=`` / ``tokenizer=`` /
  ``ref_model=`` to skip the default ``from_pretrained`` calls.
- **PEFT is opt-in.** Runners accept ``peft_config=None``;
  when set, the adapter config flows into TRL's
  ``peft_config=`` and (for QLoRA) the BitsAndBytes config flows
  into ``from_pretrained`` via ``quantization_config``.
- **Method-specific dispatch on the preference side.**
  :class:`PreferenceRunner` reads ``config.method`` and selects
  the right TRL ``XxxTrainer`` / ``XxxConfig`` pair. ``ref_model``
  is forwarded only when the method needs it; ``reward_funcs``
  is required for GRPO and rejected otherwise.

## Test expectations

- Unit tests under ``tests/unit/training/``, one file per source
  module. ``transformers`` / ``trl`` / ``peft`` / ``torch`` /
  ``datasets`` are faked via ``sys.modules`` injection — no
  real ML imports in unit tests.
- Coverage target: ≥ 90 % line.
- Live integration tests (``@pytest.mark.live``) that exercise
  TRL against a tiny model can land in Phase 9.

## Gotchas

- **Heavy imports stay inside ``train``.** A top-level
  ``import transformers`` would break the "import ``forge.training``
  without the extra" invariant. Always import inside the runner
  method that needs the dep.
- **Don't manually validate ``ref_model``-vs-PEFT.** TRL has
  branched semantics here (it'll synthesize a ref by disabling
  the adapter); :class:`PreferenceRunner` just forwards whatever
  the caller passes.
- **Pydantic ``Field`` uses keyword ``default=``** rather than
  positional defaults, so pyright recognises the field as
  optional.
- **Chat-template formatting is text-only.** Non-text content
  parts (images) collapse to ``[non-text: ClassName]``
  placeholders. Vision-instruction training needs its own path —
  out of scope here.

## When to update this file

- Adding a new preference method.
- Adding a new top-level public class/function to
  ``__init__.py``.
- Changing the lazy-import boundary.
- Adding a new heavy dep to the ``[finetuning]`` extra.
- Adding a vision-aware training path (it would need its own
  module, not a chat-template extension).
