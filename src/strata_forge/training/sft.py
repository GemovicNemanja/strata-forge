"""Supervised fine-tuning via TRL's ``SFTTrainer``.

:class:`SFTConfig` is a Pydantic shape that captures the
fine-tuning hyperparameters Forge cares about; it serializes
cleanly to YAML for reproducibility and ships across to remote
:mod:`strata_forge.compute` runs. :class:`SFTRunner` wires the config
together with a :class:`strata_forge.training.peft.LoRAConfig` or
:class:`strata_forge.training.peft.QLoRAConfig` (optional) and invokes
TRL.

All heavy imports — ``torch``, ``transformers``, ``trl``,
``datasets`` — are deferred until :meth:`SFTRunner.train` is
called, so importing :mod:`strata_forge.training.sft` succeeds without
the ``[finetuning]`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.training.progress import attach as _attach_progress
from strata_forge.training.progress import coerce_int, numeric_metrics

if TYPE_CHECKING:
    from strata_forge.training.peft import LoRAConfig, QLoRAConfig

__all__ = [
    "SFTConfig",
    "SFTRunResult",
    "SFTRunner",
]


type _Precision = Literal["fp32", "fp16", "bf16"]


class SFTConfig(BaseModel):
    """Hyperparameters for an SFT run.

    Mirrors the TRL ``SFTConfig`` knobs Forge exposes by default.
    Anything you need that isn't here goes through
    :attr:`extra_trainer_args` verbatim — Forge never blocks
    access to the underlying TRL surface.

    Attributes:
        model_id: HuggingFace model id or local path.
        output_dir: Where to write checkpoints + final adapter / weights.
        dataset_text_field: The column on the dataset that holds the
            already-formatted text. When ``None``, Forge expects the
            caller's collator / formatting fn to do it.
        max_seq_length: Per-example token budget.
        packing: Use TRL's packing (different from Forge's
            :mod:`strata_forge.training.packing` — that one is for offline
            preprocessing).
        num_epochs: Training epochs.
        per_device_batch_size: Per-device batch size.
        gradient_accumulation_steps: Effective batch size = device
            x accumulation x N devices.
        learning_rate: Optimizer learning rate.
        warmup_ratio: Fraction of total steps used for LR warmup. Reaches TRL as
            ``warmup_steps``, which takes a float in ``[0, 1)`` as exactly this
            fraction.
        weight_decay: AdamW weight decay.
        precision: Training precision.
        gradient_checkpointing: Trade memory for compute.
        logging_steps: How often to log loss to stdout / Langfuse.
        save_steps: How often to checkpoint (``0`` = no intermediate
            saves).
        seed: PRNG seed; forwarded to ``transformers.set_seed``.
        extra_trainer_args: Verbatim passthrough to TRL's
            ``SFTConfig``.
        progress_jsonl: When set, the runner attaches a callback that
            writes one :class:`strata_forge.training.progress.ProgressEvent` per
            training event to this path (for an orchestrator to tail). The
            ``FORGE_PROGRESS_PATH`` env var is used as a fallback. Not a TRL
            knob — it never reaches ``to_trl_kwargs``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    output_dir: str
    dataset_text_field: str | None = "text"
    max_seq_length: int = Field(default=2048, ge=8)
    packing: bool = False
    num_epochs: float = Field(default=1.0, gt=0)
    per_device_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    precision: _Precision = "bf16"
    gradient_checkpointing: bool = True
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=0, ge=0)
    seed: int = 42
    extra_trainer_args: dict[str, Any] = Field(default_factory=dict)
    progress_jsonl: str | None = None

    def to_trl_kwargs(self) -> dict[str, Any]:
        """Render this config as kwargs for TRL's ``SFTConfig``."""
        kwargs: dict[str, Any] = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            # Forwarded under a DIFFERENT name than the field deliberately. transformers 5 removed
            # `warmup_ratio` and folded it into `warmup_steps`, which reads a float in [0, 1) as a
            # fraction of total steps -- identical semantics, so the knob this package exposes keeps
            # the name that describes what it is.
            "warmup_steps": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "logging_steps": self.logging_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "seed": self.seed,
            "max_length": self.max_seq_length,
            "packing": self.packing,
        }
        if self.dataset_text_field is not None:
            kwargs["dataset_text_field"] = self.dataset_text_field
        if self.save_steps > 0:
            kwargs["save_steps"] = self.save_steps
            kwargs["save_strategy"] = "steps"
        else:
            kwargs["save_strategy"] = "no"
        if self.precision == "bf16":
            kwargs["bf16"] = True
        elif self.precision == "fp16":
            kwargs["fp16"] = True
        # extra args win over Forge defaults to keep the escape hatch sharp.
        kwargs.update(self.extra_trainer_args)
        return kwargs


class SFTRunResult(BaseModel):
    """Result returned by :meth:`SFTRunner.train`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    output_dir: str
    train_loss: float | None = None
    train_runtime_s: float | None = None
    train_samples_per_second: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    # Optimizer steps actually taken. TRL reports it beside the metrics rather than inside them,
    # so an orchestrator summarising a finished run has no other way to say how far it got.
    steps: int | None = None


class SFTRunner:
    """TRL :class:`SFTTrainer` wrapper.

    Args:
        config: The :class:`SFTConfig` describing the run.
        peft_config: Optional :class:`LoRAConfig` /
            :class:`QLoRAConfig`. When set, TRL trains an adapter
            instead of the full model.

    The runner does no work in the constructor; all heavy imports
    happen in :meth:`train`.
    """

    def __init__(
        self,
        config: SFTConfig,
        *,
        peft_config: LoRAConfig | QLoRAConfig | None = None,
    ) -> None:
        self._config = config
        self._peft_config = peft_config

    @property
    def config(self) -> SFTConfig:
        return self._config

    @property
    def peft_config(self) -> LoRAConfig | QLoRAConfig | None:
        return self._peft_config

    def _load_modules(self) -> tuple[Any, Any, Any]:
        try:
            transformers_mod: Any = __import__("transformers")
            trl_mod: Any = __import__("trl")
            datasets_mod: Any = __import__("datasets")
        except ImportError as exc:
            msg = (
                "The [finetuning] extra is required for SFTRunner. "
                "Install it with: pip install 'strata-forge[finetuning]'."
            )
            raise ImportError(msg) from exc
        return transformers_mod, trl_mod, datasets_mod

    def build_trainer(
        self,
        *,
        train_dataset: Any,
        eval_dataset: Any = None,
        tokenizer: Any = None,
        model: Any = None,
    ) -> Any:
        """Construct the TRL ``SFTTrainer`` without starting it.

        Useful when callers want to inspect or mutate the trainer
        before calling ``.train()``. :meth:`train` does this for you.
        """
        transformers_mod, trl_mod, _ = self._load_modules()
        if model is None:
            model_load_kwargs: dict[str, Any] = {}
            if self._peft_config is not None and hasattr(self._peft_config, "to_bnb_config"):
                # QLoRA branch: pass the bnb config.
                model_load_kwargs["quantization_config"] = self._peft_config.to_bnb_config()  # type: ignore[union-attr]
            model = transformers_mod.AutoModelForCausalLM.from_pretrained(
                self._config.model_id, **model_load_kwargs
            )
        if tokenizer is None:
            tokenizer = transformers_mod.AutoTokenizer.from_pretrained(self._config.model_id)
        trl_config = trl_mod.SFTConfig(**self._config.to_trl_kwargs())
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": trl_config,
            "train_dataset": train_dataset,
            "processing_class": tokenizer,
        }
        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset
        if self._peft_config is not None:
            trainer_kwargs["peft_config"] = self._peft_config.to_peft_config()
        trainer = trl_mod.SFTTrainer(**trainer_kwargs)
        _attach_progress(trainer, self._config.progress_jsonl)
        return trainer

    def train(
        self,
        *,
        train_dataset: Any,
        eval_dataset: Any = None,
        tokenizer: Any = None,
        model: Any = None,
    ) -> SFTRunResult:
        """Run the SFT loop end to end.

        Returns:
            An :class:`SFTRunResult` summarising the run. The model
            / adapter weights live under
            :attr:`SFTConfig.output_dir`.
        """
        trainer = self.build_trainer(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model=model,
        )
        train_output: Any = trainer.train()
        trainer.save_model(self._config.output_dir)
        metrics = numeric_metrics(train_output)
        return SFTRunResult(
            output_dir=self._config.output_dir,
            train_loss=metrics.get("train_loss"),
            train_runtime_s=metrics.get("train_runtime"),
            train_samples_per_second=metrics.get("train_samples_per_second"),
            metrics=metrics,
            steps=coerce_int(getattr(train_output, "global_step", None)),
        )
