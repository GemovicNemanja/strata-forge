"""Trusted in-memory head initialization and fixed installed decoder wrapper classes."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.core.errors import ValidationError
from strata_forge.core.repro import set_seed

# Native ML classes are optional, dynamically imported and version-dependent at this seam.
type _ML = Any


class HeadInitializationReceipt(BaseModel):
    """Exact head transformations; source objects are caller-resolved, not Hub attestations."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    version: Literal[1] = 1
    initializer_id: Literal["encoder_random_head", "decoder_logit_head"]
    family: Literal["bert", "gpt2", "modernbert-decoder"]
    source_resolution: Literal["caller_resolved_object"] = "caller_resolved_object"
    labels: tuple[str, ...]
    created_keys: tuple[str, ...] = Field(min_length=1, max_length=32)
    removed_keys: tuple[str, ...] = Field(max_length=32)
    body_key_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    body_key_count: int = Field(ge=1)
    verbalizer_ids: tuple[int, ...] = ()
    bias_copied: bool = False
    pooling: Literal["native_encoder", "last_attended_token"]
    seed: int
    step_zero_max_probability_error: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @field_validator("version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("classifier_receipt_version_invalid")
        return value

    @model_validator(mode="after")
    def _exact(self) -> HeadInitializationReceipt:
        validate_classifier_labels(self.labels)
        for keys in (self.created_keys, self.removed_keys):
            if len(set(keys)) != len(keys) or any(
                len(key) > 192 or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", key) is None
                for key in keys
            ):
                raise ValueError("classifier_receipt_keys_invalid")
        if self.initializer_id == "decoder_logit_head" and (
            self.family == "bert"
            or self.step_zero_max_probability_error is None
            or self.pooling != "last_attended_token"
            or len(self.verbalizer_ids) != len(self.labels)
        ):
            raise ValueError("classifier_decoder_receipt_invalid")
        if self.initializer_id == "encoder_random_head" and (
            self.family != "bert" or self.verbalizer_ids or self.pooling != "native_encoder"
        ):
            raise ValueError("classifier_encoder_receipt_invalid")
        return self


@dataclass(frozen=True, repr=False)
class InitializedClassifier:
    """A model and its trusted initializer receipt; never a serialized loader exemption."""

    model: _ML
    receipt: HeadInitializationReceipt


def classifier_modules() -> tuple[_ML, _ML]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError("Install 'strata-forge[finetuning]' for classifier training.") from exc
    return torch, transformers


def validate_classifier_labels(labels: tuple[str, ...]) -> None:
    if (
        not 2 <= len(labels) <= 16
        or len(set(labels)) != len(labels)
        or any(type(label) is not str or not label.strip() or len(label) > 64 for label in labels)
    ):
        raise ValidationError("classifier_labels_invalid")


@cache
def decoder_classifier_class() -> _ML:
    torch, transformers = classifier_modules()
    from transformers.modeling_outputs import SequenceClassifierOutput

    class ForgeDecoderClassifier(transformers.PreTrainedModel):
        supports_gradient_checkpointing = True
        _supports_sdpa = True

        def __init__(self, config: _ML) -> None:
            super().__init__(config)  # pyright: ignore[reportUnknownMemberType] — optional dynamic native superclass
            if config.model_type == "gpt2":
                self.backbone = transformers.GPT2Model(config)
                self.head = torch.nn.Identity()
                bias = False
            elif config.model_type == "modernbert-decoder":
                from transformers.models.modernbert_decoder.modeling_modernbert_decoder import (
                    ModernBertDecoderModel,
                    ModernBertDecoderPredictionHead,
                )

                self.backbone = ModernBertDecoderModel(config)
                self.head = ModernBertDecoderPredictionHead(config)
                bias = config.decoder_bias
            else:
                raise ValidationError("classifier_family_unsupported")
            self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels, bias=bias)
            self.post_init()

        def forward(
            self,
            input_ids: _ML,
            attention_mask: _ML,
        ) -> _ML:
            if (
                attention_mask.shape != input_ids.shape
                or not bool(((attention_mask == 0) | (attention_mask == 1)).all())
                or not bool((attention_mask.sum(-1) > 0).all())
            ):
                raise ValidationError("classifier_attention_mask_invalid")
            output = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            states = self.head(output.last_hidden_state)
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            last = (positions * attention_mask.long()).max(-1).values
            pooled = states[torch.arange(input_ids.shape[0], device=input_ids.device), last]
            return SequenceClassifierOutput(logits=self.classifier(pooled))

    return ForgeDecoderClassifier


def _initialize(
    source: _ML,
    *,
    labels: tuple[str, ...],
    initializer_id: Literal["encoder_random_head", "decoder_logit_head"],
    verbalizer_ids: tuple[int, ...],
    seed: int,
    equivalence_inputs: dict[str, _ML] | None,
) -> InitializedClassifier:
    validate_classifier_labels(labels)
    if type(seed) is not int or seed < 0 or seed > 2**32 - 1:
        raise ValidationError("classifier_seed_invalid")
    torch, transformers = classifier_modules()
    config = copy.deepcopy(source.config)
    config.num_labels = len(labels)
    config.id2label = dict(enumerate(labels))
    config.label2id = {label: i for i, label in enumerate(labels)}
    set_seed(seed)
    removed: tuple[str, ...] = ()
    bias_copied = False
    step_zero_error = None
    if initializer_id == "encoder_random_head" and type(source) is transformers.BertModel:
        if verbalizer_ids or source.pooler is None:
            raise ValidationError("classifier_encoder_initialization_unsupported")
        model = transformers.BertForSequenceClassification(config)
        body = source.state_dict()
        model.bert.load_state_dict(body, strict=True)
        created = ("classifier.weight", "classifier.bias")
        pooling = "native_encoder"
    elif initializer_id == "decoder_logit_head" and type(source) in (
        transformers.GPT2LMHeadModel,
        transformers.ModernBertDecoderForCausalLM,
    ):
        if (
            len(verbalizer_ids) != len(labels)
            or len(set(verbalizer_ids)) != len(labels)
            or any(
                type(token) is not int or not 0 <= token < config.vocab_size
                for token in verbalizer_ids
            )
        ):
            raise ValidationError("classifier_verbalizer_ids_invalid")
        model = decoder_classifier_class()(config)
        source_body = source.transformer if config.model_type == "gpt2" else source.model
        body = source_body.state_dict()
        model.backbone.load_state_dict(body, strict=True)
        projection = source.lm_head if config.model_type == "gpt2" else source.decoder
        if config.model_type == "modernbert-decoder":
            model.head.load_state_dict(source.lm_head.state_dict(), strict=True)
            removed = tuple("lm_head." + key for key in source.lm_head.state_dict())
        with torch.no_grad():
            indices = torch.tensor(verbalizer_ids, device=projection.weight.device)
            model.classifier.weight.copy_(projection.weight.index_select(0, indices))
            if projection.bias is not None:
                if model.classifier.bias is None:
                    raise ValidationError("classifier_bias_mapping_unsupported")
                model.classifier.bias.copy_(projection.bias.index_select(0, indices))
                bias_copied = True
        prefix = "lm_head" if config.model_type == "gpt2" else "decoder"
        removed += tuple(prefix + "." + key for key in projection.state_dict())
        body_prefix = "transformer" if config.model_type == "gpt2" else "model"
        expected_source = {body_prefix + "." + key for key in body} | set(removed)
        if set(source.state_dict()) != expected_source:
            raise ValidationError("classifier_undeclared_source_keys")
        created = tuple("head." + key for key in model.head.state_dict()) + tuple(
            "classifier." + key for key in model.classifier.state_dict()
        )
        pooling = "last_attended_token"
        if equivalence_inputs is None or set(equivalence_inputs) != {"input_ids", "attention_mask"}:
            raise ValidationError("classifier_step_zero_inputs_required")
        was_training = source.training
        try:
            source.eval()
            model.eval()
            with torch.no_grad():
                expected_logits = source(**equivalence_inputs).logits
                attention = equivalence_inputs["attention_mask"]
                last = (
                    (torch.arange(attention.shape[1], device=attention.device) * attention)
                    .max(-1)
                    .values
                )
                expected = (
                    expected_logits[
                        torch.arange(attention.shape[0], device=attention.device), last
                    ][:, list(verbalizer_ids)]
                    .float()
                    .softmax(-1)
                )
                actual = model(**equivalence_inputs).logits.float().softmax(-1)
                step_zero_error = float((expected - actual).abs().max())
                if not torch.allclose(expected, actual, atol=1e-6, rtol=1e-5):
                    raise ValidationError("classifier_step_zero_mismatch")
        finally:
            source.train(was_training)
    else:
        raise ValidationError("classifier_initializer_unsupported")
    receipt = HeadInitializationReceipt(
        initializer_id=initializer_id,
        family=config.model_type,
        labels=labels,
        created_keys=created,
        removed_keys=removed,
        body_key_sha256=hashlib.sha256(
            json.dumps(sorted(body), separators=(",", ":")).encode()
        ).hexdigest(),
        body_key_count=len(body),
        verbalizer_ids=verbalizer_ids,
        bias_copied=bias_copied,
        pooling=pooling,
        seed=seed,
        step_zero_max_probability_error=step_zero_error,
    )
    return InitializedClassifier(model=model, receipt=receipt)


async def initialize_classifier(
    source: _ML,
    *,
    labels: tuple[str, ...],
    initializer_id: Literal["encoder_random_head", "decoder_logit_head"],
    verbalizer_ids: tuple[int, ...] = (),
    seed: int = 0,
    equivalence_inputs: dict[str, _ML] | None = None,
) -> InitializedClassifier:
    """Initialize supported native in-memory objects; never fetch a model or accept remote code."""
    try:
        return await asyncio.to_thread(
            _initialize,
            source,
            labels=labels,
            initializer_id=initializer_id,
            verbalizer_ids=verbalizer_ids,
            seed=seed,
            equivalence_inputs=equivalence_inputs,
        )
    except ValidationError, ImportError:
        raise
    except Exception:
        raise ValidationError("classifier_initialization_failed") from None
