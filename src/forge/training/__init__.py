"""Fine-tuning: SFT, preference tuning (DPO/ORPO/KTO/GRPO), and PEFT (LoRA/QLoRA).

The public surface is:

- :class:`SFTConfig` / :class:`SFTRunner` / :class:`SFTRunResult`
- :class:`DPOConfig` / :class:`ORPOConfig` / :class:`KTOConfig` /
  :class:`GRPOConfig` / :class:`PreferenceRunner` /
  :class:`PreferenceRunResult`
- :class:`LoRAConfig` / :class:`QLoRAConfig`
- :func:`apply_chat_template` / :func:`conversation_to_dicts` /
  :func:`conversation_to_text`
- :func:`pack_sequences` / :class:`PackedSequence`

Heavy ML dependencies (``torch``, ``transformers``, ``trl``,
``peft``, ``datasets``) ride behind the ``[finetuning]`` extra
and are lazy-imported inside the runners' ``train`` methods, so
``import forge.training`` works without them.
"""

from forge.training.chat_template import (
    apply_chat_template,
    conversation_to_dicts,
    conversation_to_text,
)
from forge.training.packing import PackedSequence, pack_sequences
from forge.training.peft import LoRAConfig, QLoRAConfig
from forge.training.preference import (
    AnyPreferenceConfig,
    DPOConfig,
    GRPOConfig,
    KTOConfig,
    ORPOConfig,
    PreferenceRunner,
    PreferenceRunResult,
)
from forge.training.progress import (
    JsonlProgressWriter,
    ProgressEvent,
    trainer_callback,
)
from forge.training.sft import SFTConfig, SFTRunner, SFTRunResult

__all__ = [
    "AnyPreferenceConfig",
    "DPOConfig",
    "GRPOConfig",
    "JsonlProgressWriter",
    "KTOConfig",
    "LoRAConfig",
    "ORPOConfig",
    "PackedSequence",
    "PreferenceRunResult",
    "PreferenceRunner",
    "ProgressEvent",
    "QLoRAConfig",
    "SFTConfig",
    "SFTRunResult",
    "SFTRunner",
    "apply_chat_template",
    "conversation_to_dicts",
    "conversation_to_text",
    "pack_sequences",
    "trainer_callback",
]
