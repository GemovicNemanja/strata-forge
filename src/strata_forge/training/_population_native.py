"""Fixed native local optimizer over already-prepared unpadded records; imports stay lazy."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from strata_forge.core.repro import set_seed
from strata_forge.training._classifier_heads import (
    HeadInitializationReceipt,
    InitializedClassifier,
    classifier_modules,
    decoder_classifier_class,
)
from strata_forge.training._verifier_heads import verifier_classifier_class
from strata_forge.training.classifier import (
    _loss,  # pyright: ignore[reportPrivateUsage] — same differentiable SDK objective
)
from strata_forge.training.population import (
    PopulationError,
    PrimaryPopulationTask,
    TrainingPopulationPlanV2,
)
from strata_forge.training.verifier_classifier import (
    InitializedVerifierClassifier,
    _validate_binding,  # pyright: ignore[reportPrivateUsage] — exact installed SDK binding validation
    _validate_state,  # pyright: ignore[reportPrivateUsage] — exact installed SDK state validation
)
from strata_forge.training.verifier_loss import ResearchLossRow, research_verifier_loss

if TYPE_CHECKING:
    import threading

    from strata_forge.training._population_store import PopulationRecord


# Resolved native modules and tensors have dynamic optional-library APIs.
type _ML = Any


def _validate_model(
    plan: TrainingPopulationPlanV2, initialized: _ML, tokenizer: _ML
) -> tuple[_ML, _ML]:
    torch, transformers = classifier_modules()
    if isinstance(plan.task, PrimaryPopulationTask):
        if type(initialized) is not InitializedClassifier:
            raise PopulationError("population_training_unsupported")
        receipt = HeadInitializationReceipt.model_validate(
            {
                name: getattr(initialized.receipt, name)
                for name in HeadInitializationReceipt.model_fields
            }
        )
        expected = (
            transformers.BertForSequenceClassification
            if plan.family == "bert"
            else decoder_classifier_class()
        )
        if type(initialized.model) is not expected or receipt.labels != plan.task.config.labels:
            raise PopulationError("population_training_unsupported")
        rendering = plan.task.config.rendering
        if (
            rendering is not None
            and rendering.label_tokens is not None
            and tuple(
                tokenizer.encode(rendering.label_tokens[label], add_special_tokens=False)
                for label in receipt.labels
            )
            != tuple([value] for value in receipt.verbalizer_ids)
        ):
            raise PopulationError("population_training_unsupported")
    else:
        if (
            type(initialized) is not InitializedVerifierClassifier
            or type(initialized.model) is not verifier_classifier_class()
            or initialized.geometry != plan.task.geometry
        ):
            raise PopulationError("population_training_unsupported")
        _validate_binding(initialized.model.config, initialized.geometry, initialized.receipt)
        _validate_state(initialized.model, initialized.receipt)
        receipt = initialized.receipt.primary
    model = initialized.model
    if (
        receipt.family != plan.family
        or receipt.seed != plan.seed
        or model.config.model_type != plan.family
        or model.config.pad_token_id != tokenizer.pad_token_id
    ):
        raise PopulationError("population_training_unsupported")
    labels = (
        plan.task.config.labels
        if isinstance(plan.task, PrimaryPopulationTask)
        else plan.task.geometry.labels
    )
    if tuple(model.config.id2label[index] for index in range(len(labels))) != labels:
        raise PopulationError("population_training_unsupported")
    parameters = tuple(model.parameters())
    if (
        not parameters
        or len({value.device for value in parameters}) != 1
        or any(value.dtype != torch.float32 or not value.requires_grad for value in parameters)
        or (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() != 1
        )
    ):
        raise PopulationError("population_training_unsupported")
    if (
        isinstance(plan.task, PrimaryPopulationTask)
        and plan.task.config.use_cpu
        and parameters[0].device.type != "cpu"
    ):
        raise PopulationError("population_training_unsupported")
    return torch, transformers


class NativePopulationBackend:
    def __init__(
        self,
        plan: TrainingPopulationPlanV2,
        initialized: _ML,
        tokenizer: _ML,
        cancelled: threading.Event,
    ) -> None:
        torch, transformers = _validate_model(plan, initialized, tokenizer)
        defaults = transformers.TrainingArguments.__dataclass_fields__
        expected = {
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
        }
        if (
            any(defaults[key].default != value for key, value in expected.items())
            or defaults["optim"].default != "adamw_torch_fused"
        ):
            raise PopulationError("population_training_unsupported")
        self.plan, self.model, self.tokenizer = plan, initialized.model, tokenizer
        self.torch = torch
        self.device = next(self.model.parameters()).device
        settings = (
            plan.task.config
            if isinstance(plan.task, PrimaryPopulationTask)
            else plan.task.optimizer
        )
        set_seed(plan.seed)
        if cancelled.is_set():
            raise PopulationError("population_cancelled")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=settings.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            fused=True,
        )
        self.scheduler = transformers.get_scheduler(
            settings.scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=settings.warmup_steps,
            num_training_steps=plan.steps,
        )
        self.profile = {
            "optimizer": "adamw_torch_fused",
            "learning_rate": settings.learning_rate,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "scheduler": settings.scheduler,
            "warmup_steps": settings.warmup_steps,
            "batch_size": 32,
            "gradient_accumulation": 1,
            "epochs": 1,
        }
        self.versions = {"torch": torch.__version__, "transformers": transformers.__version__}
        if cancelled.is_set():
            raise PopulationError("population_cancelled")
        self.model.train()

    def apply(self, rows: tuple[PopulationRecord, ...]) -> None:
        torch = self.torch
        size = max(len(row.input_ids) for row in rows)
        ids = torch.full(
            (len(rows), size), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device
        )
        attention = torch.zeros_like(ids)
        types = torch.zeros_like(ids)
        has_types = any(row.token_type_ids is not None for row in rows)
        if has_types and any(row.token_type_ids is None for row in rows):
            raise PopulationError("population_cache_invalid")
        for index, row in enumerate(rows):
            length = len(row.input_ids)
            ids[index, :length] = torch.tensor(row.input_ids, device=self.device)
            attention[index, :length] = 1
            if row.token_type_ids is not None:
                types[index, :length] = torch.tensor(row.token_type_ids, device=self.device)
        inputs = {"input_ids": ids, "attention_mask": attention}
        if has_types:
            inputs["token_type_ids"] = types
        self.optimizer.zero_grad(set_to_none=True)
        if isinstance(self.plan.task, PrimaryPopulationTask):
            targets = torch.tensor(
                [json.loads(row.targets_json) for row in rows],
                dtype=torch.float32,
                device=self.device,
            )
            if (
                targets.shape != (len(rows), len(self.plan.task.config.labels))
                or not bool(torch.isfinite(targets).all())
                or not bool(((targets >= 0) & (targets <= 1)).all())
                or not bool(
                    torch.isclose(
                        targets.sum(-1),
                        torch.ones(len(rows), device=self.device),
                        atol=1e-6,
                        rtol=0,
                    ).all()
                )
            ):
                raise PopulationError("population_cache_invalid")
            loss = _loss(self.model(**inputs).logits, targets, self.plan.task.config.brier_weight)
        else:
            targets = tuple(ResearchLossRow.model_validate_json(row.targets_json) for row in rows)
            if any(
                target.premise_ids != row.premise_ids
                for target, row in zip(targets, rows, strict=True)
            ):
                raise PopulationError("population_cache_invalid")
            output = self.model(
                **inputs,
                marker_positions=tuple(row.marker_positions for row in rows),
                premise_ids=tuple(row.premise_ids for row in rows),
            )
            loss = research_verifier_loss(
                output.logits,
                class_order=self.plan.task.class_order,
                rows=targets,
                flag_logits=output.inconsistent_logit,
                support_logits=output.support_logits,
            ).total
        if not bool(torch.isfinite(loss)):
            raise PopulationError("population_training_failed")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        if not bool(torch.isfinite(norm)):
            raise PopulationError("population_training_failed")
        self.optimizer.step()

    def after_step(self) -> None:
        self.scheduler.step()
