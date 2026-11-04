"""Primary sequence-classifier SDK over resolved objects, with explicit objectives and exports."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strata_forge.core.errors import ValidationError
from strata_forge.core.repro import env_snapshot, set_seed
from strata_forge.training._classifier_heads import (
    HeadInitializationReceipt,
    InitializedClassifier,
    classifier_modules,
    decoder_classifier_class,
    initialize_classifier,
    validate_classifier_labels,
)
from strata_forge.training._load_guard import _guard_loading_result
from strata_forge.training.verifier_config import VerifierConfig, render_verifier_input

if TYPE_CHECKING:
    from pathlib import Path

# Optional ML objects cross one dynamic Transformers boundary, never an inert worker spec.
type _ML = Any

__all__ = [
    "ClassifierConfig",
    "ClassifierExample",
    "ClassifierRunResult",
    "ClassifierRunner",
    "InitializedClassifier",
    "initialize_classifier",
    "load_classifier",
]


class ClassifierExample(BaseModel):
    """An explicit primary training row; integers require the config's declared index map."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    item_id: str = Field(min_length=1, max_length=1024)
    text: str = Field(min_length=1, max_length=200_000)
    text_pair: str | None = Field(default=None, max_length=200_000)
    label: str | int
    label_dist: tuple[float, ...] | None = None
    sample_weight: float = 1.0
    premises: tuple[str, ...] | None = None
    hypothesis: str | None = None

    @model_validator(mode="after")
    def _unsupported(self) -> ClassifierExample:
        if not self.item_id.strip():
            raise ValueError("classifier_item_id_invalid")
        if self.sample_weight != 1.0:
            raise ValueError("classifier_sample_weights_unsupported")
        if type(self.label) not in (str, int):
            raise ValueError("classifier_label_invalid")
        return self

    @classmethod
    def from_verifier(
        cls,
        *,
        rendering: VerifierConfig,
        item_id: str,
        premises: list[str],
        hypothesis: str,
        label: str | int,
    ) -> ClassifierExample:
        """Use the artifact renderer exactly; unsupported premise shapes refuse without truncation."""
        rendered = render_verifier_input(rendering, premises, hypothesis)
        if rendered.error_code is not None or rendered.text is None:
            raise ValidationError(rendered.error_code or "classifier_rendering_invalid")
        return cls(
            item_id=item_id,
            text=rendered.text,
            text_pair=rendered.text_pair,
            label=label,
            premises=tuple(premises),
            hypothesis=hypothesis,
        )


class ClassifierConfig(BaseModel):
    """A bounded primary CE objective; research objectives must be selected explicitly."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    labels: tuple[str, ...]
    objective: Literal["cross_entropy", "ce_brier"] = "cross_entropy"
    brier_weight: float = Field(default=0.0, ge=0, le=100, allow_inf_nan=False)
    class_weights: tuple[float, ...] | None = None
    label_smoothing: float = 0.0
    source_objective: Literal["ordinary_classification", "nli_binary_grouped"] = (
        "ordinary_classification"
    )
    label_aliases: dict[str, str] = Field(default_factory=dict)
    label_index_map: dict[int, str] = Field(default_factory=lambda: dict[int, str]())
    max_length: int = Field(default=512, ge=1, le=8192)
    overflow: Literal["error", "drop"] = "error"
    epochs: float = Field(default=1.0, gt=0, le=100, allow_inf_nan=False)
    max_steps: int = Field(default=-1, ge=-1, le=1_000_000)
    learning_rate: float = Field(default=5e-5, gt=0, le=1, allow_inf_nan=False)
    batch_size: int = Field(default=8, ge=1, le=1024)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    precision: Literal["fp32", "bf16"] = "fp32"
    gradient_checkpointing: bool = False
    use_cpu: bool = False
    scheduler: Literal["linear", "constant", "cosine"] = "linear"
    warmup_steps: int = Field(default=0, ge=0, le=1_000_000)
    checkpoint_selection: Literal["final", "metric"] = "final"
    selection_split: str | None = None
    evaluation_tiers: dict[str, Literal["development", "gate", "test"]] = Field(
        default_factory=dict
    )
    rendering: VerifierConfig | None = None

    @model_validator(mode="after")
    def _contract(self) -> ClassifierConfig:
        validate_classifier_labels(self.labels)
        if self.class_weights is not None or self.label_smoothing != 0:
            raise ValueError("classifier_weighting_or_smoothing_unsupported")
        if self.source_objective != "ordinary_classification":
            raise ValueError("classifier_nli_binary_objective_undefined")
        if (self.objective == "cross_entropy" and self.brier_weight != 0) or (
            self.objective == "ce_brier" and self.brier_weight <= 0
        ):
            raise ValueError("classifier_objective_coefficient_invalid")
        if self.selection_split is not None and self.evaluation_tiers.get(self.selection_split) in (
            "gate",
            "test",
        ):
            raise ValueError("classifier_gate_selection_forbidden")
        if self.checkpoint_selection != "final" or self.selection_split is not None:
            raise ValueError("classifier_metric_selection_unsupported")
        if self.max_steps == 0:
            raise ValueError("classifier_zero_steps_invalid")
        if any(
            label not in self.labels
            for label in (*self.label_aliases.values(), *self.label_index_map.values())
        ):
            raise ValueError("classifier_label_mapping_invalid")
        if self.rendering is not None and self.rendering.labels != self.labels:
            raise ValueError("classifier_rendering_label_order_mismatch")
        if self.rendering is not None and self.rendering.output_capabilities != ("classification",):
            raise ValueError("classifier_auxiliary_outputs_unsupported")
        if len(self.model_dump_json().encode("utf-8")) > 32_768:
            raise ValueError("classifier_config_too_large")
        return self

    @classmethod
    def research_ce_brier(cls, *, labels: tuple[str, ...], **kwargs: _ML) -> ClassifierConfig:
        """Explicitly request the research CE+0.1 Brier preset; generic default remains CE."""
        return cls(labels=labels, objective="ce_brier", brier_weight=0.1, **kwargs)


class ClassifierRunResult(BaseModel):
    """Local artifact result, not a remote launch, calibration or checkpoint-selection claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    artifact_dir: str
    steps: int
    trained_rows: int
    overflow_ids: tuple[str, ...]
    checkpoint_selection: Literal["final"] = "final"


def _targets(config: ClassifierConfig, row: ClassifierExample) -> tuple[float, ...]:
    label = (
        config.label_index_map.get(row.label)
        if type(row.label) is int
        else config.label_aliases.get(str(row.label), str(row.label))
    )
    if label not in config.labels:
        raise ValidationError("classifier_label_mapping_required")
    if row.label_dist is None:
        return tuple(float(value == label) for value in config.labels)
    import math

    values = row.label_dist
    if (
        len(values) != len(config.labels)
        or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values)
        or not math.isclose(sum(values), 1.0, abs_tol=1e-6, rel_tol=0)
    ):
        raise ValidationError("classifier_distribution_invalid")
    return values


def _loss(logits: _ML, targets: _ML, brier_weight: float) -> _ML:
    torch, _ = classifier_modules()
    values = logits.float()
    ce = -(targets * torch.nn.functional.log_softmax(values, dim=-1)).sum(-1).mean()
    return ce + brier_weight * ((values.softmax(-1) - targets) ** 2).sum(-1).mean()


def _rows(
    config: ClassifierConfig, rows: tuple[ClassifierExample, ...], tokenizer: _ML
) -> tuple[list[dict[str, _ML]], tuple[str, ...]]:
    if not rows or len(rows) > 100_000 or len({row.item_id for row in rows}) != len(rows):
        raise ValidationError("classifier_population_invalid")
    targets = [_targets(config, row) for row in rows]
    output: list[dict[str, _ML]] = []
    dropped: list[str] = []
    for row, target in zip(rows, targets, strict=True):
        if config.rendering is not None:
            rendered = render_verifier_input(config.rendering, row.premises, row.hypothesis)
            if (
                rendered.error_code is not None
                or rendered.text != row.text
                or rendered.text_pair != row.text_pair
            ):
                raise ValidationError("classifier_rendering_mismatch")
        elif row.premises is not None or row.hypothesis is not None:
            raise ValidationError("classifier_rendering_required")
        encoded = dict(tokenizer(row.text, text_pair=row.text_pair, truncation=False))
        if len(encoded["input_ids"]) > config.max_length:
            if config.overflow == "error":
                raise ValidationError("classifier_overflow")
            dropped.append(row.item_id)
            continue
        output.append({**encoded, "targets": target})
    if not output:
        raise ValidationError("classifier_population_empty_after_overflow")
    return output, tuple(dropped)


def _train(
    config: ClassifierConfig,
    initialized: InitializedClassifier,
    tokenizer: _ML,
    rows: tuple[ClassifierExample, ...],
    output_dir: Path,
    cancelled: threading.Event,
) -> ClassifierRunResult:
    config = ClassifierConfig.model_validate(config.model_dump())
    rows = tuple(ClassifierExample.model_validate(row.model_dump()) for row in rows)
    if initialized.receipt.family != "bert" and any(row.text_pair is not None for row in rows):
        raise ValidationError("classifier_decoder_pair_unsupported")
    if initialized.receipt.labels != config.labels:
        raise ValidationError("classifier_initialized_label_order_mismatch")
    if config.rendering is not None and config.rendering.label_tokens is not None:
        if initialized.receipt.initializer_id != "decoder_logit_head":
            raise ValidationError("classifier_verbalizer_metadata_unsupported")
        encoded_labels = tuple(
            tokenizer.encode(config.rendering.label_tokens[label], add_special_tokens=False)
            for label in config.labels
        )
        if (
            any(len(ids) != 1 for ids in encoded_labels)
            or tuple(ids[0] for ids in encoded_labels) != initialized.receipt.verbalizer_ids
        ):
            raise ValidationError("classifier_verbalizer_metadata_mismatch")
    # Validate labels/targets before constructing the trainer or starting an optimizer.
    encoded, dropped = _rows(config, rows, tokenizer)
    if initialized.receipt.family != "bert" and any(
        set(row) - {"input_ids", "attention_mask", "targets"} for row in encoded
    ):
        raise ValidationError("classifier_decoder_input_fields_unsupported")
    if output_dir.is_symlink() or (output_dir.exists() and any(output_dir.iterdir())):
        raise ValidationError("classifier_output_not_empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".classifier-attempt.json").open("x") as file:
        json.dump({"status": "started"}, file)
    torch, transformers = classifier_modules()
    set_seed(config.seed)
    if tokenizer.pad_token_id is None:
        raise ValidationError("classifier_padding_required")
    initialized.model.config.pad_token_id = tokenizer.pad_token_id
    if config.gradient_checkpointing:
        initialized.model.config.use_cache = False

    class ObjectiveTrainer(transformers.Trainer):
        def compute_loss(
            self, model: _ML, inputs: _ML, return_outputs: bool = False, **kwargs: _ML
        ) -> _ML:
            targets = inputs.pop("targets")
            outputs = model(**inputs)
            loss = _loss(outputs.logits, targets, config.brier_weight)
            if not bool(torch.isfinite(loss)):
                raise ValidationError("classifier_loss_nonfinite")
            return (loss, outputs) if return_outputs else loss

    def collate(batch: list[dict[str, _ML]]) -> dict[str, _ML]:
        targets = torch.tensor([row["targets"] for row in batch], dtype=torch.float32)
        inputs = tokenizer.pad(
            [{key: value for key, value in row.items() if key != "targets"} for row in batch],
            return_tensors="pt",
        )
        return {**inputs, "targets": targets}

    manifest = {
        "version": 1,
        "implementation": "forge_primary_classifier_v1",
        "config": config.model_dump(mode="json"),
        "initialization": initialized.receipt.model_dump(mode="json"),
        "checkpoint_selection": "final",
        "steps": 1_000_000_000_000,
        "trained_rows": len(encoded),
        "overflow_ids": dropped[:32],
        "overflow_count": len(dropped),
        "overflow_omitted": max(0, len(dropped) - 32),
        "environment": env_snapshot(),
    }
    if (
        len(json.dumps(manifest, allow_nan=False, ensure_ascii=False, indent=2).encode("utf-8"))
        > 65_536
    ):
        raise ValidationError("classifier_metadata_too_large")
    args = transformers.TrainingArguments(
        output_dir=str(output_dir / "trainer-state"),
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        seed=config.seed,
        data_seed=config.seed,
        bf16=config.precision == "bf16",
        gradient_checkpointing=config.gradient_checkpointing,
        use_cpu=config.use_cpu,
        lr_scheduler_type=config.scheduler,
        warmup_steps=config.warmup_steps,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
        label_names=["targets"],
        dataloader_pin_memory=False,
    )

    class CancellationCallback(transformers.TrainerCallback):
        def on_step_end(self, args: _ML, state: _ML, control: _ML, **kwargs: _ML) -> _ML:
            if cancelled.is_set():
                control.should_training_stop = True
            return control

    trainer = ObjectiveTrainer(
        model=initialized.model,
        args=args,
        train_dataset=encoded,
        data_collator=collate,
        callbacks=[CancellationCallback()],
    )
    trainer.train()
    if cancelled.is_set():
        raise asyncio.CancelledError
    if trainer.state.global_step < 1:
        raise ValidationError("classifier_no_optimizer_steps")
    artifact = output_dir / "artifact"
    initialized.model.save_pretrained(artifact, safe_serialization=True)
    tokenizer.save_pretrained(artifact)
    manifest["steps"] = trainer.state.global_step
    raw = json.dumps(manifest, allow_nan=False, ensure_ascii=False, indent=2) + "\n"
    if len(raw.encode("utf-8")) > 65_536:
        raise ValidationError("classifier_metadata_too_large")
    (artifact / "strata_classifier.json").write_text(raw)
    if config.rendering is not None:
        (artifact / "strata_verifier.json").write_text(
            config.rendering.model_dump_json(indent=2) + "\n"
        )
    return ClassifierRunResult(
        artifact_dir=str(artifact),
        steps=trainer.state.global_step,
        trained_rows=len(encoded),
        overflow_ids=tuple(dropped),
    )


class ClassifierRunner:
    """Train a supported already-initialized primary classifier without hidden Hub reads."""

    def __init__(self, config: ClassifierConfig) -> None:
        self.config = config

    async def train(
        self,
        *,
        initialized: InitializedClassifier,
        tokenizer: _ML,
        rows: tuple[ClassifierExample, ...],
        output_dir: Path,
    ) -> ClassifierRunResult:
        """Run Trainer off-loop and save only the final safe artifact in a fresh local directory."""
        cancelled = threading.Event()
        work = asyncio.create_task(
            asyncio.to_thread(
                _train, self.config, initialized, tokenizer, rows, output_dir, cancelled
            )
        )
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            cancelled.set()
            try:
                await asyncio.wait_for(asyncio.shield(work), timeout=10)
            except Exception, asyncio.CancelledError:
                cancelled.set()
            raise
        except ValidationError, ImportError:
            raise
        except Exception:
            raise ValidationError("classifier_training_failed") from None


def _load(path: Path) -> InitializedClassifier:
    metadata = path / "strata_classifier.json"
    with metadata.open("rb") as file:
        raw = file.read(65_537)
    if len(raw) > 65_536:
        raise ValidationError("classifier_metadata_too_large")
    value: object = json.loads(raw)
    if type(value) is not dict:
        raise ValidationError("classifier_artifact_unsupported")
    manifest = cast("dict[str, object]", value)
    if (
        set(manifest)
        != {
            "version",
            "implementation",
            "config",
            "initialization",
            "checkpoint_selection",
            "steps",
            "trained_rows",
            "overflow_ids",
            "overflow_count",
            "overflow_omitted",
            "environment",
        }
        or type(manifest.get("version")) is not int
        or manifest.get("version") != 1
        or manifest.get("implementation") != "forge_primary_classifier_v1"
    ):
        raise ValidationError("classifier_artifact_unsupported")
    receipt = HeadInitializationReceipt.model_validate_json(json.dumps(manifest["initialization"]))
    config = ClassifierConfig.model_validate_json(json.dumps(manifest["config"]))
    if receipt.labels != config.labels:
        raise ValidationError("classifier_artifact_label_order_mismatch")
    _, transformers = classifier_modules()
    native_config = transformers.AutoConfig.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    if (
        native_config.model_type != receipt.family
        or tuple(native_config.id2label[i] for i in range(native_config.num_labels))
        != receipt.labels
    ):
        raise ValidationError("classifier_artifact_config_mismatch")
    factory = (
        transformers.BertForSequenceClassification
        if receipt.family == "bert"
        else decoder_classifier_class()
    )
    model, _ = _guard_loading_result(
        factory.from_pretrained(
            path,
            config=native_config,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            output_loading_info=True,
        )
    )
    return InitializedClassifier(model=model, receipt=receipt)


async def load_classifier(path: Path) -> InitializedClassifier:
    """Reload only a complete local safetensors artifact through fixed installed classes."""
    try:
        return await asyncio.to_thread(_load, path)
    except ValidationError, ImportError:
        raise
    except Exception:
        raise ValidationError("classifier_artifact_invalid") from None
