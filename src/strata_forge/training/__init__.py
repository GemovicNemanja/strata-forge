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
- :data:`METHODS` / :class:`MethodSpec` / :func:`pick_method` /
  :func:`enabled_methods` / :func:`check_format` /
  :class:`UnsupportedMethodError` — the method registry, for
  callers that drive training from a declaration rather than
  from Python.
- :data:`FORMATS` / :class:`FormatSpec` / :func:`pick_format` /
  :func:`validate_mapping` / :func:`build_training_rows` /
  :func:`sft_text_field` / :class:`DatasetFormatError` — the
  dataset-shape declaration those callers map columns with.

Heavy ML dependencies (``torch``, ``transformers``, ``trl``,
``peft``, ``datasets``) ride behind the ``[finetuning]`` extra
and are lazy-imported inside the runners' ``train`` methods, so
``import strata_forge.training`` works without them.
"""

from strata_forge.training.chat_template import (
    apply_chat_template,
    conversation_to_dicts,
    conversation_to_text,
)
from strata_forge.training.dataset_format import (
    FORMATS,
    DatasetFormat,
    DatasetFormatError,
    FormatSpec,
    build_training_rows,
    pick_format,
    sft_text_field,
    validate_mapping,
)
from strata_forge.training.methods import (
    METHODS,
    MethodName,
    MethodSpec,
    UnsupportedMethodError,
    check_format,
    enabled_methods,
    pick_method,
)
from strata_forge.training.packing import PackedSequence, pack_sequences
from strata_forge.training.peft import LoRAConfig, QLoRAConfig
from strata_forge.training.preference import (
    AnyPreferenceConfig,
    DPOConfig,
    GRPOConfig,
    KTOConfig,
    ORPOConfig,
    PreferenceRunner,
    PreferenceRunResult,
)
from strata_forge.training.progress import (
    JsonlProgressWriter,
    ProgressEvent,
    trainer_callback,
)
from strata_forge.training.sft import SFTConfig, SFTRunner, SFTRunResult

__all__ = [
    "FORMATS",
    "METHODS",
    "AnyPreferenceConfig",
    "DPOConfig",
    "DatasetFormat",
    "DatasetFormatError",
    "FormatSpec",
    "GRPOConfig",
    "JsonlProgressWriter",
    "KTOConfig",
    "LoRAConfig",
    "MethodName",
    "MethodSpec",
    "ORPOConfig",
    "PackedSequence",
    "PreferenceRunResult",
    "PreferenceRunner",
    "ProgressEvent",
    "QLoRAConfig",
    "SFTConfig",
    "SFTRunResult",
    "SFTRunner",
    "UnsupportedMethodError",
    "apply_chat_template",
    "build_training_rows",
    "check_format",
    "conversation_to_dicts",
    "conversation_to_text",
    "enabled_methods",
    "pack_sequences",
    "pick_format",
    "pick_method",
    "sft_text_field",
    "trainer_callback",
    "validate_mapping",
]
