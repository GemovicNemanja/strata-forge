"""Fixed native verifier wrappers with explicit parameter sharing and exact auxiliary keys."""

from __future__ import annotations

import math
from functools import cache
from typing import Any

from strata_forge.training._classifier_heads import classifier_modules, decoder_classifier_class
from strata_forge.training.verifier_geometry import VerifierGeometryError

# Optional native ML objects are resolved only inside the fixed installed class factory.
type _ML = Any


def auxiliary_keys(outputs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        name + suffix
        for output, name in (
            ("inconsistent", "inconsistency_head"),
            ("support_set", "support_head"),
        )
        if output in outputs
        for suffix in (".weight", ".bias")
    )


@cache
def verifier_classifier_class() -> _ML:
    torch, transformers = classifier_modules()

    class SupportHead(torch.nn.Module):
        def __init__(self, hidden: int, *, device: _ML, dtype: _ML) -> None:
            super().__init__()  # pyright: ignore[reportUnknownMemberType] — optional native nn.Module superclass
            self.weight = torch.nn.Parameter(
                torch.empty(hidden, hidden, device=device, dtype=dtype)
            )
            self.bias = torch.nn.Parameter(torch.zeros(1, device=device, dtype=dtype))

        def forward(self, pooled: _ML, markers: _ML) -> _ML:
            return ((pooled @ self.weight) * markers).sum(-1) + self.bias

    class ForgeAuxiliaryVerifier(transformers.PreTrainedModel):
        supports_gradient_checkpointing = True
        _supports_sdpa = True

        def __init__(self, config: _ML, *, primary: _ML = None) -> None:
            super().__init__(config)  # pyright: ignore[reportUnknownMemberType] — dynamic optional native superclass
            outputs = config.strata_verifier_outputs
            if (
                type(outputs) not in (list, tuple)
                or not outputs
                or outputs[0] != "label"
                or len(set(outputs)) != len(outputs)
                or any(x not in ("label", "inconsistent", "support_set") for x in outputs)
            ):
                raise VerifierGeometryError("verifier_model_unsupported")
            if config.model_type == "bert":
                primary = (
                    transformers.BertForSequenceClassification(config)
                    if primary is None
                    else primary
                )
                if type(primary) is not transformers.BertForSequenceClassification:
                    raise VerifierGeometryError("verifier_model_unsupported")
                self.bert, self.dropout, self.classifier = (
                    primary.bert,
                    primary.dropout,
                    primary.classifier,
                )
            elif config.model_type in ("gpt2", "modernbert-decoder"):
                native = decoder_classifier_class()
                primary = native(config) if primary is None else primary
                if type(primary) is not native:
                    raise VerifierGeometryError("verifier_model_unsupported")
                self.backbone, self.head, self.classifier = (
                    primary.backbone,
                    primary.head,
                    primary.classifier,
                )
            else:
                raise VerifierGeometryError("verifier_model_unsupported")
            # Native lifecycle metadata is finalized below without reinitializing shared modules.
            parameter = next(primary.parameters())
            std = config.initializer_range
            if type(std) not in (float, int) or not math.isfinite(std) or std < 0:
                raise VerifierGeometryError("verifier_initialization_invalid")
            if "inconsistent" in outputs:
                self.inconsistency_head = torch.nn.Linear(
                    config.hidden_size, 1, device=parameter.device, dtype=parameter.dtype
                )
                torch.nn.init.normal_(self.inconsistency_head.weight, mean=0, std=std)
                torch.nn.init.zeros_(self.inconsistency_head.bias)
            if "support_set" in outputs:
                self.support_head = SupportHead(
                    config.hidden_size, device=parameter.device, dtype=parameter.dtype
                )
                torch.nn.init.normal_(self.support_head.weight, mean=0, std=std)
            self.outputs = tuple(outputs)
            self.post_init()

        def init_weights(self) -> None:
            # post_init must collect native tied-key/placement metadata, but every
            # parameter is already initialized by the primary or exact head rules.
            return None

        def forward(
            self,
            input_ids: _ML,
            attention_mask: _ML,
            *,
            marker_positions: tuple[tuple[int, ...], ...],
            premise_ids: tuple[tuple[str, ...], ...],
            token_type_ids: _ML = None,
        ) -> _ML:
            if (
                input_ids.ndim != 2
                or attention_mask.shape != input_ids.shape
                or not bool(((attention_mask == 0) | (attention_mask == 1)).all())
                or not bool((attention_mask.sum(-1) > 0).all())
            ):
                raise VerifierGeometryError("verifier_batch_invalid")
            if len(marker_positions) != input_ids.shape[0] or len(premise_ids) != len(
                marker_positions
            ):
                raise VerifierGeometryError("verifier_markers_invalid")
            for row, (positions, ids) in enumerate(zip(marker_positions, premise_ids, strict=True)):
                if (
                    len(positions) != len(ids)
                    or len(set(positions)) != len(positions)
                    or len(set(ids)) != len(ids)
                    or any(type(x) is not int or not 0 <= x < input_ids.shape[1] for x in positions)
                    or any(not bool(attention_mask[row, p]) for p in positions)
                ):
                    raise VerifierGeometryError("verifier_markers_invalid")
            if self.config.model_type == "bert":
                if not bool((attention_mask[:, 0] == 1).all()):
                    raise VerifierGeometryError("verifier_batch_invalid")
                output = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    return_dict=True,
                )
                states = output.last_hidden_state
                pooled = states[:, 0]
                logits = self.classifier(self.dropout(output.pooler_output))
            else:
                if token_type_ids is not None:
                    raise VerifierGeometryError("verifier_batch_invalid")
                output = self.backbone(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                )
                states = output.last_hidden_state
                last = (
                    (
                        torch.arange(input_ids.shape[1], device=input_ids.device)
                        * attention_mask.long()
                    )
                    .max(-1)
                    .values
                )
                indices = torch.arange(input_ids.shape[0], device=input_ids.device)
                pooled = states[indices, last]
                logits = self.classifier(self.head(states)[indices, last])
            for row, positions in enumerate(marker_positions):
                if any(not bool(attention_mask[row, position]) for position in positions):
                    raise VerifierGeometryError("verifier_markers_invalid")
            flag = (
                self.inconsistency_head(pooled).squeeze(-1)
                if "inconsistent" in self.outputs
                else None
            )
            support = (
                tuple(
                    self.support_head(pooled[row], states[row, list(positions)])
                    for row, positions in enumerate(marker_positions)
                )
                if "support_set" in self.outputs
                else None
            )
            return VerifierForward(logits, flag, support, premise_ids)

    return ForgeAuxiliaryVerifier


class VerifierForward:
    """Graph-connected numerical outputs; no calibrated/thresholded interpretation is added."""

    def __init__(
        self,
        logits: _ML,
        inconsistent_logit: _ML,
        support_logits: tuple[_ML, ...] | None,
        premise_ids: tuple[tuple[str, ...], ...],
    ) -> None:
        self.logits = logits
        self.inconsistent_logit = inconsistent_logit
        self.support_logits = support_logits
        self.premise_ids = premise_ids
