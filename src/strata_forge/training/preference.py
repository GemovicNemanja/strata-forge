"""Preference fine-tuning: DPO / ORPO / KTO / GRPO via TRL.

A small family of Pydantic configs and runners that mirror the
shape of :class:`strata_forge.training.sft.SFTConfig` /
:class:`strata_forge.training.sft.SFTRunner`. Each preference method
gets its own ``XxxConfig`` (so the type hints reflect each
method's distinct knobs) and a single :class:`PreferenceRunner`
that dispatches based on the config type.

All heavy imports defer to :meth:`PreferenceRunner.train`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.training.progress import attach as _attach_progress
from strata_forge.training.progress import coerce_int, numeric_metrics

if TYPE_CHECKING:
    from strata_forge.training.peft import LoRAConfig, QLoRAConfig

__all__ = [
    "AnyPreferenceConfig",
    "DPOConfig",
    "GRPOConfig",
    "KTOConfig",
    "ORPOConfig",
    "PreferenceRunResult",
    "PreferenceRunner",
]


type _Precision = Literal["fp32", "fp16", "bf16"]


class _BasePreferenceConfig(BaseModel):
    """Shared knobs across DPO / ORPO / KTO / GRPO configs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    output_dir: str
    num_epochs: float = Field(default=1.0, gt=0)
    per_device_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=5e-6, gt=0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    precision: _Precision = "bf16"
    gradient_checkpointing: bool = True
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=0, ge=0)
    seed: int = 42
    extra_trainer_args: dict[str, Any] = Field(default_factory=dict)
    progress_jsonl: str | None = None
    """When set, write live progress events to this JSONL path (env fallback:
    ``FORGE_PROGRESS_PATH``). Not a TRL knob — never reaches ``to_trl_kwargs``."""

    def _base_trl_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            # See SFTConfig.to_trl_kwargs: transformers 5 folded `warmup_ratio` into `warmup_steps`,
            # which reads a float in [0, 1) as a fraction of total steps.
            "warmup_steps": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "logging_steps": self.logging_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "seed": self.seed,
        }
        if self.save_steps > 0:
            kwargs["save_steps"] = self.save_steps
            kwargs["save_strategy"] = "steps"
        else:
            kwargs["save_strategy"] = "no"
        if self.precision == "bf16":
            kwargs["bf16"] = True
        elif self.precision == "fp16":
            kwargs["fp16"] = True
        return kwargs


class DPOConfig(_BasePreferenceConfig):
    """Direct Preference Optimization hyperparameters."""

    method: Literal["dpo"] = "dpo"
    beta: float = Field(default=0.1, gt=0)
    loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid"
    max_length: int = Field(default=2048, ge=8)

    def to_trl_kwargs(self) -> dict[str, Any]:
        kwargs = self._base_trl_kwargs()
        kwargs.update(
            {
                "beta": self.beta,
                "loss_type": self.loss_type,
                "max_length": self.max_length,
            }
        )
        kwargs.update(self.extra_trainer_args)
        return kwargs


class ORPOConfig(_BasePreferenceConfig):
    """Odds-Ratio Preference Optimization."""

    method: Literal["orpo"] = "orpo"
    beta: float = Field(default=0.1, gt=0)
    max_length: int = Field(default=2048, ge=8)

    def to_trl_kwargs(self) -> dict[str, Any]:
        kwargs = self._base_trl_kwargs()
        kwargs.update(
            {
                "beta": self.beta,
                "max_length": self.max_length,
            }
        )
        kwargs.update(self.extra_trainer_args)
        return kwargs


class KTOConfig(_BasePreferenceConfig):
    """Kahneman-Tversky Optimization (binary preference signals)."""

    method: Literal["kto"] = "kto"
    beta: float = Field(default=0.1, gt=0)
    desirable_weight: float = Field(default=1.0, gt=0)
    undesirable_weight: float = Field(default=1.0, gt=0)
    max_length: int = Field(default=2048, ge=8)

    def to_trl_kwargs(self) -> dict[str, Any]:
        kwargs = self._base_trl_kwargs()
        kwargs.update(
            {
                "beta": self.beta,
                "desirable_weight": self.desirable_weight,
                "undesirable_weight": self.undesirable_weight,
                "max_length": self.max_length,
            }
        )
        kwargs.update(self.extra_trainer_args)
        return kwargs


class GRPOConfig(_BasePreferenceConfig):
    """Group Relative Policy Optimization (reward-driven, no preference pairs)."""

    method: Literal["grpo"] = "grpo"
    beta: float = Field(default=0.04, gt=0)
    num_generations: int = Field(default=8, ge=2)
    max_completion_length: int = Field(default=256, ge=8)

    def to_trl_kwargs(self) -> dict[str, Any]:
        kwargs = self._base_trl_kwargs()
        kwargs.update(
            {
                "beta": self.beta,
                "num_generations": self.num_generations,
                "max_completion_length": self.max_completion_length,
            }
        )
        kwargs.update(self.extra_trainer_args)
        return kwargs


AnyPreferenceConfig = DPOConfig | ORPOConfig | KTOConfig | GRPOConfig
"""Discriminated union over the four preference-method configs."""


class PreferenceRunResult(BaseModel):
    """Result returned by :meth:`PreferenceRunner.train`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal["dpo", "orpo", "kto", "grpo"]
    output_dir: str
    train_loss: float | None = None
    train_runtime_s: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    # Optimizer steps actually taken. TRL reports it beside the metrics rather than inside them,
    # so an orchestrator summarising a finished run has no other way to say how far it got.
    steps: int | None = None


_TRAINER_CLASS: dict[str, str] = {
    "dpo": "DPOTrainer",
    "orpo": "ORPOTrainer",
    "kto": "KTOTrainer",
    "grpo": "GRPOTrainer",
}
_TRL_CONFIG_CLASS: dict[str, str] = {
    "dpo": "DPOConfig",
    "orpo": "ORPOConfig",
    "kto": "KTOConfig",
    "grpo": "GRPOConfig",
}

#: Methods whose classes TRL keeps outside its top-level namespace, and the submodule they moved to.
#: ORPO left the mainline API and now lives only under ``trl.experimental``. That namespace carries
#: NO semantic-versioning promise, so this is the one method a major-version ceiling does not
#: protect -- a minor TRL release may move or change it again.
_TRL_FALLBACK_MODULE: dict[str, str] = {"orpo": "trl.experimental.orpo"}


def _resolve_trl_class(trl_mod: Any, method: str, name: str) -> Any:
    """Find a TRL class by name, following the method's move out of the top-level namespace.

    Looks in ``trl`` first so a version that still exports the class mainline keeps working, then
    falls back to the submodule TRL relegated it to.
    """
    found = getattr(trl_mod, name, None)
    if found is not None:
        return found
    module_path = _TRL_FALLBACK_MODULE.get(method)
    if module_path is None:
        msg = f"trl has no {name}; the installed TRL version does not support {method}"
        raise AttributeError(msg)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - depends on the installed TRL layout
        msg = f"trl has no {name} and {module_path} is unavailable: {exc}"
        raise AttributeError(msg) from exc
    try:
        return getattr(module, name)
    except AttributeError as exc:
        msg = f"neither trl nor {module_path} provides {name}"
        raise AttributeError(msg) from exc


class PreferenceRunner:
    """Dispatcher for DPO / ORPO / KTO / GRPO TRL trainers.

    Args:
        config: One of :class:`DPOConfig`, :class:`ORPOConfig`,
            :class:`KTOConfig`, :class:`GRPOConfig`.
        peft_config: Optional adapter config; when set, the
            preference trainer trains a LoRA / QLoRA adapter.
    """

    def __init__(
        self,
        config: AnyPreferenceConfig,
        *,
        peft_config: LoRAConfig | QLoRAConfig | None = None,
    ) -> None:
        self._config = config
        self._peft_config = peft_config

    @property
    def config(self) -> AnyPreferenceConfig:
        return self._config

    @property
    def peft_config(self) -> LoRAConfig | QLoRAConfig | None:
        return self._peft_config

    def _load_modules(self) -> tuple[Any, Any]:
        try:
            transformers_mod: Any = __import__("transformers")
            trl_mod: Any = __import__("trl")
        except ImportError as exc:
            msg = (
                "The [finetuning] extra is required for PreferenceRunner. "
                "Install it with: pip install 'strata-forge[finetuning]'."
            )
            raise ImportError(msg) from exc
        return transformers_mod, trl_mod

    def build_trainer(
        self,
        *,
        train_dataset: Any,
        eval_dataset: Any = None,
        tokenizer: Any = None,
        model: Any = None,
        ref_model: Any = None,
        reward_funcs: Any = None,
    ) -> Any:
        """Construct the TRL preference trainer for this config."""
        transformers_mod, trl_mod = self._load_modules()
        method = self._config.method
        if model is None:
            model_load_kwargs: dict[str, Any] = {}
            if self._peft_config is not None and hasattr(self._peft_config, "to_bnb_config"):
                model_load_kwargs["quantization_config"] = self._peft_config.to_bnb_config()  # type: ignore[union-attr]
            model = transformers_mod.AutoModelForCausalLM.from_pretrained(
                self._config.model_id, **model_load_kwargs
            )
        if tokenizer is None:
            tokenizer = transformers_mod.AutoTokenizer.from_pretrained(self._config.model_id)

        trl_config_cls: Any = _resolve_trl_class(trl_mod, method, _TRL_CONFIG_CLASS[method])
        trainer_cls: Any = _resolve_trl_class(trl_mod, method, _TRAINER_CLASS[method])
        trl_config = trl_config_cls(**self._config.to_trl_kwargs())

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
        # DPO and KTO need a ref model unless using PEFT (TRL handles
        # that case by branching off the adapter-less base model).
        if method in ("dpo", "kto") and ref_model is not None:
            trainer_kwargs["ref_model"] = ref_model
        if method == "grpo":
            if reward_funcs is None:
                err = (
                    "PreferenceRunner: GRPO requires reward_funcs= to be set "
                    "(callable or list of callables)."
                )
                raise ValueError(err)
            trainer_kwargs["reward_funcs"] = reward_funcs
        trainer = trainer_cls(**trainer_kwargs)
        _attach_progress(trainer, self._config.progress_jsonl)
        return trainer

    def train(
        self,
        *,
        train_dataset: Any,
        eval_dataset: Any = None,
        tokenizer: Any = None,
        model: Any = None,
        ref_model: Any = None,
        reward_funcs: Any = None,
    ) -> PreferenceRunResult:
        """Run the preference loop end to end."""
        trainer = self.build_trainer(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model=model,
            ref_model=ref_model,
            reward_funcs=reward_funcs,
        )
        train_output: Any = trainer.train()
        trainer.save_model(self._config.output_dir)
        metrics = numeric_metrics(train_output)
        return PreferenceRunResult(
            method=self._config.method,
            output_dir=self._config.output_dir,
            train_loss=metrics.get("train_loss"),
            train_runtime_s=metrics.get("train_runtime"),
            metrics=metrics,
            steps=coerce_int(getattr(train_output, "global_step", None)),
        )
