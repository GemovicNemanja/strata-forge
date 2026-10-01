"""PEFT (LoRA / QLoRA) configuration shapes.

Forge wraps ``peft.LoraConfig`` with Pydantic models that ship
sensible defaults for typical Forge fine-tuning runs and add the
quantization toggles that distinguish QLoRA from plain LoRA. The
peft / transformers packages are imported lazily inside the ``to_*``
builders so importing this module works without the ``[finetuning]``
extra; bitsandbytes is only ever looked up, never imported, because
transformers imports it itself when it quantises the model.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.core.errors import ForgeError

__all__ = [
    "LoRAConfig",
    "MissingBitsAndBytesError",
    "QLoRAConfig",
    "require_bitsandbytes",
]


class MissingBitsAndBytesError(ForgeError, ImportError):
    """QLoRA was asked for on a machine without a usable bitsandbytes.

    Raised when the package is not installed, or when its recorded version is below the floor the
    ``[finetuning]`` extra pins. An ``ImportError`` as well, so a caller already handling a
    missing extra handles this too.
    """


# The ``[finetuning]`` extra's lower bound; a test holds the two equal.
_BITSANDBYTES_FLOOR = (0, 49)


def _bitsandbytes_version() -> str | None:
    """The installed distribution's version, or ``None`` when no metadata records one."""
    try:
        return importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError:
        return None


def require_bitsandbytes() -> None:
    """Refuse QLoRA up front when bitsandbytes is missing or older than the extra's floor.

    ``transformers.BitsAndBytesConfig`` constructs without the package, so the absence otherwise
    surfaces only inside ``from_pretrained`` -- after the dataset and the model weights have been
    downloaded. A lookup rather than an import: importing bitsandbytes loads its CUDA library,
    which is the model load's job, not a precondition check's. The version comes from the
    distribution's metadata for the same reason; a version that metadata does not record, or
    that does not start with ``major.minor``, is not refused.
    """
    if importlib.util.find_spec("bitsandbytes") is None:
        msg = (
            "adapter 'qlora' needs bitsandbytes to quantise the model, and it is not installed. "
            "Install it with: pip install 'strata-forge[finetuning]' (which includes it), "
            "or train with adapter 'lora'."
        )
        raise MissingBitsAndBytesError(msg)
    installed = _bitsandbytes_version()
    release = re.match(r"(\d+)\.(\d+)", installed or "")
    if release is not None and (int(release[1]), int(release[2])) < _BITSANDBYTES_FLOOR:
        floor = ".".join(str(part) for part in _BITSANDBYTES_FLOOR)
        msg = (
            f"adapter 'qlora' needs bitsandbytes>={floor} to quantise the model, and "
            f"{installed} is installed. Upgrade it with: pip install 'bitsandbytes>={floor},<1', "
            "or train with adapter 'lora'."
        )
        raise MissingBitsAndBytesError(msg)


type _TaskType = Literal[
    "CAUSAL_LM",
    "SEQ_CLS",
    "SEQ_2_SEQ_LM",
    "TOKEN_CLS",
    "QUESTION_ANS",
    "FEATURE_EXTRACTION",
]


class LoRAConfig(BaseModel):
    """Adapter configuration for plain LoRA fine-tuning.

    Attributes:
        r: LoRA rank. 16-64 covers most use cases.
        alpha: LoRA alpha scaling factor (often set to ``2 * r``).
        dropout: Dropout applied to the LoRA path.
        target_modules: Linear-layer names to adapt. ``None`` uses
            peft's per-architecture defaults.
        bias: Whether to train bias parameters.
        task_type: peft TaskType. ``"CAUSAL_LM"`` for SFT/DPO/etc.
        modules_to_save: Extra (non-LoRA) modules to mark trainable
            and save with the adapter (e.g. ``["embed_tokens", "lm_head"]``
            when you've resized the vocabulary).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    r: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    target_modules: tuple[str, ...] | None = None
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: _TaskType = "CAUSAL_LM"
    modules_to_save: tuple[str, ...] = ()

    def to_peft_config(self) -> Any:
        """Build a ``peft.LoraConfig`` from this spec.

        Raises:
            ImportError: When the ``[finetuning]`` extra (which
                installs ``peft``) is not available.
        """
        try:
            peft_mod: Any = __import__("peft")
        except ImportError as exc:
            msg = (
                "The [finetuning] extra is required for LoRAConfig.to_peft_config. "
                "Install it with: pip install 'strata-forge[finetuning]'."
            )
            raise ImportError(msg) from exc
        kwargs: dict[str, Any] = {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "bias": self.bias,
            "task_type": self.task_type,
        }
        if self.target_modules is not None:
            kwargs["target_modules"] = list(self.target_modules)
        if self.modules_to_save:
            kwargs["modules_to_save"] = list(self.modules_to_save)
        return peft_mod.LoraConfig(**kwargs)


class QLoRAConfig(BaseModel):
    """QLoRA: LoRA on a 4-bit quantized base model.

    Wraps :class:`LoRAConfig` and adds the bitsandbytes
    quantization knobs. :meth:`to_bnb_config` returns the
    ``transformers.BitsAndBytesConfig`` to pass to
    ``from_pretrained``; :meth:`to_peft_config` delegates to the
    embedded :class:`LoRAConfig`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lora: LoRAConfig = Field(default_factory=lambda: LoRAConfig(r=64, alpha=16))
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"

    def to_peft_config(self) -> Any:
        return self.lora.to_peft_config()

    def to_bnb_config(self) -> Any:
        """Build a ``transformers.BitsAndBytesConfig``.

        Raises:
            MissingBitsAndBytesError: When bitsandbytes is not
                installed, which every model load with this config
                would otherwise discover only after downloading.
            ImportError: When the ``[finetuning]`` extra is not
                available.
        """
        require_bitsandbytes()
        try:
            transformers_mod: Any = __import__("transformers")
        except ImportError as exc:
            msg = (
                "The [finetuning] extra is required for QLoRAConfig.to_bnb_config. "
                "Install it with: pip install 'strata-forge[finetuning]'."
            )
            raise ImportError(msg) from exc
        try:
            torch_mod: Any = __import__("torch")
        except ImportError as exc:
            msg = (
                "The [finetuning] extra is required for QLoRAConfig.to_bnb_config. "
                "Install it with: pip install 'strata-forge[finetuning]'."
            )
            raise ImportError(msg) from exc
        dtype = getattr(torch_mod, self.bnb_4bit_compute_dtype)
        return transformers_mod.BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
