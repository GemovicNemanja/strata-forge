"""Fixed grouped classifier and null-masked auxiliary losses over supplied batch logits."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strata_forge.core.errors import ValidationError

# Optional native tensors cross this numerical seam; importing it must not import Torch.
type _Tensor = Any
type _Torch = Any
type LogicalClass = Literal["E", "C", "U"]

OBJECTIVE_PRESET = "verifier_grouped_aux_v1"
_ERRORS = frozenset(
    {
        "verifier_loss_class_order_invalid",
        "verifier_loss_targets_invalid",
        "verifier_loss_logits_invalid",
        "verifier_loss_logits_required",
        "verifier_loss_nonfinite",
    }
)

__all__ = [
    "OBJECTIVE_PRESET",
    "EntailmentTarget",
    "ResearchLossRow",
    "ResearchLossTerms",
    "ThreeWayTarget",
    "VerifierLossError",
    "research_verifier_loss",
]


class VerifierLossError(ValidationError):
    """A fixed content-free refusal from the supplied-logit loss boundary."""


class _Target(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        hide_input_in_errors=True,
        revalidate_instances="always",
    )


class ThreeWayTarget(_Target):
    """A complete distribution in the caller's explicit E/C/U class order."""

    label_space: Literal["3way"] = "3way"
    label_dist: tuple[float, float, float]

    @model_validator(mode="after")
    def _distribution(self) -> ThreeWayTarget:
        if any(
            not math.isfinite(v) or not 0 <= v <= 1 for v in self.label_dist
        ) or not math.isclose(
            math.fsum(self.label_dist),
            1.0,
            abs_tol=1e-6,
            rel_tol=0,
        ):
            raise ValueError("verifier_distribution_invalid")
        return self


class EntailmentTarget(_Target):
    """An entailment target whose negative aggregates C and U without distinguishing them."""

    label_space: Literal["2way_entail_vs_not"] = "2way_entail_vs_not"
    entailment_target: int = Field(ge=0, le=1)


class ResearchLossRow(_Target):
    """Resolved supervision; null auxiliary values explicitly exclude that row's targets."""

    primary: Annotated[ThreeWayTarget | EntailmentTarget, Field(discriminator="label_space")]
    flag_target: bool | None = None
    premise_ids: tuple[str, ...] = ()
    support_set: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _ids(self) -> ResearchLossRow:
        if len(set(self.premise_ids)) != len(self.premise_ids) or any(
            not pid.strip() for pid in self.premise_ids
        ):
            raise ValueError("verifier_premise_ids_invalid")
        if self.support_set is not None and (
            not self.premise_ids
            or len(set(self.support_set)) != len(self.support_set)
            or not set(self.support_set) <= set(self.premise_ids)
        ):
            raise ValueError("verifier_support_ids_invalid")
        return self

    @property
    def flag_eligible(self) -> bool:
        return self.flag_target is not None

    @property
    def support_eligible(self) -> bool:
        return self.support_set is not None


@dataclass(frozen=True, repr=False)
class ResearchLossTerms:
    """Differentiable fixed-coefficient terms with their actual row denominators."""

    total: _Tensor
    primary: _Tensor
    flag: _Tensor
    support: _Tensor
    primary_rows: int
    flag_rows: int
    support_rows: int
    preset_id: Literal["verifier_grouped_aux_v1"] = field(default=OBJECTIVE_PRESET, init=False)


def _torch() -> _Torch:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install 'strata-forge[finetuning]' for verifier tensor losses.") from exc
    return torch


def _tensor(value: _Tensor, shape: tuple[int, ...], torch: _Torch, device: _Tensor) -> _Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.layout != torch.strided
        or not value.is_floating_point()
        or value.device != device
        or not bool(torch.isfinite(value).all())
    ):
        raise VerifierLossError("verifier_loss_logits_invalid")
    return value.to(dtype=torch.float64 if value.dtype == torch.float64 else torch.float32)


def _zero(values: tuple[_Tensor, ...], fallback: _Tensor) -> _Tensor:
    # Selecting one finite element avoids overflow from summing large logits before *0.
    result = fallback.reshape(-1)[0] * 0
    for value in values:
        result = result + (value.reshape(-1)[0] * 0 if value.numel() else value.sum())
    return result


def _compute(
    primary_logits: _Tensor,
    class_order: tuple[LogicalClass, LogicalClass, LogicalClass],
    rows: tuple[ResearchLossRow, ...],
    flag_logits: _Tensor,
    support_logits: tuple[_Tensor, ...] | None,
    torch: _Torch,
) -> ResearchLossTerms:
    if (
        type(class_order) is not tuple
        or len(class_order) != 3
        or set(class_order) != {"E", "C", "U"}
    ):
        raise VerifierLossError("verifier_loss_class_order_invalid")
    if type(rows) is not tuple or not rows or any(type(row) is not ResearchLossRow for row in rows):
        raise VerifierLossError("verifier_loss_targets_invalid")
    rows = tuple(ResearchLossRow.model_validate(row) for row in rows)
    if not isinstance(primary_logits, torch.Tensor):
        raise VerifierLossError("verifier_loss_logits_invalid")
    values = _tensor(primary_logits, (len(rows), 3), torch, primary_logits.device)
    flags = (
        None if flag_logits is None else _tensor(flag_logits, (len(rows),), torch, values.device)
    )
    support = None
    if support_logits is not None:
        if type(support_logits) is not tuple or len(support_logits) != len(rows):
            raise VerifierLossError("verifier_loss_logits_invalid")
        support = tuple(
            _tensor(value, (len(row.premise_ids),), torch, values.device)
            for row, value in zip(rows, support_logits, strict=True)
        )
    flag_count = sum(row.flag_eligible for row in rows)
    support_count = sum(row.support_eligible for row in rows)
    if (flag_count and flags is None) or (support_count and support is None):
        raise VerifierLossError("verifier_loss_logits_required")
    e, c, u = (class_order.index(label) for label in ("E", "C", "U"))
    log_p = values.log_softmax(dim=-1)
    p = log_p.exp()
    primary_terms: list[_Tensor] = []
    for index, row in enumerate(rows):
        if isinstance(row.primary, ThreeWayTarget):
            target = values.new_tensor(row.primary.label_dist)
            ce = -(target * log_p[index]).sum()
            brier = ((p[index] - target) ** 2).sum()
        else:
            t = row.primary.entailment_target
            ce = -log_p[index, e] if t else -torch.logsumexp(log_p[index, [c, u]], dim=0)
            brier = 2 * (p[index, e] - t) ** 2
        primary_terms.append(ce + 0.1 * brier)
    primary = torch.stack(primary_terms).mean()
    flag = _zero(() if flags is None else (flags,), values)
    if flag_count and flags is not None:
        positions = [index for index, row in enumerate(rows) if row.flag_eligible]
        targets = flags.new_tensor([float(rows[index].flag_target is True) for index in positions])
        flag = flag + torch.nn.functional.binary_cross_entropy_with_logits(
            flags[positions],
            targets,
            reduction="mean",
        )
    support_loss = _zero(() if support is None else support, values)
    if support_count and support is not None:
        support_terms: list[_Tensor] = []
        for index, row in enumerate(rows):
            if row.support_set is None:
                continue
            selected = set(row.support_set)
            targets = support[index].new_tensor([float(pid in selected) for pid in row.premise_ids])
            support_terms.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    support[index],
                    targets,
                    reduction="mean",
                )
            )
        support_loss = support_loss + torch.stack(support_terms).mean()
    total = primary + flag + support_loss
    if not bool(torch.isfinite(total)):
        raise VerifierLossError("verifier_loss_nonfinite")
    return ResearchLossTerms(
        total, primary, flag, support_loss, len(rows), flag_count, support_count
    )


def research_verifier_loss(
    primary_logits: _Tensor,
    *,
    class_order: tuple[LogicalClass, LogicalClass, LogicalClass],
    rows: tuple[ResearchLossRow, ...],
    flag_logits: _Tensor = None,
    support_logits: tuple[_Tensor, ...] | None = None,
) -> ResearchLossTerms:
    """Compute verifier_grouped_aux_v1 for one complete supplied batch, without model I/O."""
    torch = _torch()
    result = None
    code = "verifier_loss_targets_invalid"
    try:
        result = _compute(primary_logits, class_order, rows, flag_logits, support_logits, torch)
    except VerifierLossError as exc:
        code = str(exc) if str(exc) in _ERRORS else "verifier_loss_targets_invalid"
    except Exception:
        result = None
    if result is None:
        # Raise outside the handler so native/Pydantic errors cannot retain caller values.
        raise VerifierLossError(code)
    return result
