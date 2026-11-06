"""Nominal two-human-rater agreement and a separate registered-reference selector."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strata_forge.core.errors import ValidationError

AGREEMENT_VERSION = "nominal-two-human-raters-v1"
REFERENCE_VERSION = "registered-reference-prevalence-v1"
GATE_VERSION = "reference-prevalence-agreement-gate-v1"
RESEARCH_ANNOTATION_THRESHOLD = 0.70
type CoefficientName = Literal["cohen_kappa", "gwet_ac1"]
type UndefinedReason = Literal["empty", "chance_agreement_one"]
type _GateReason = Literal["empty", "selected_coefficient_undefined"]

__all__ = [
    "AGREEMENT_VERSION",
    "GATE_VERSION",
    "REFERENCE_VERSION",
    "RESEARCH_ANNOTATION_THRESHOLD",
    "AgreementEstimate",
    "AgreementGate",
    "HumanRatings",
    "PairwiseAgreement",
    "ReferencePrevalence",
    "RegisteredReference",
    "ResolvedReference",
    "pairwise_agreement",
    "reference_prevalence",
    "research_annotation_gate",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        hide_input_in_errors=True,
        revalidate_instances="always",
    )


def _nonempty(values: tuple[str, ...]) -> bool:
    return all(value.strip() for value in values)


def _unique(values: tuple[str, ...]) -> bool:
    return _nonempty(values) and len(set(values)) == len(values)


def _categories(labels: tuple[str, ...]) -> bool:
    return len(labels) >= 2 and _unique(labels)


def _sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


class HumanRatings(_Frozen):
    """Caller-declared human ratings; this model does not attest rater identity or blinding."""

    rater_id: str = Field(min_length=1)
    rater_kind: Literal["human"]
    item_ids: tuple[str, ...]
    ratings: tuple[str, ...]

    @model_validator(mode="after")
    def _complete(self) -> HumanRatings:
        if (
            not self.rater_id.strip()
            or not _unique(self.item_ids)
            or not _nonempty(self.ratings)
            or len(self.item_ids) != len(self.ratings)
        ):
            raise ValueError("agreement_input_invalid")
        return self


class AgreementEstimate(_Frozen):
    """A finite point estimate or an explicit undefined reason; never a confidence interval."""

    value: float | None
    chance_agreement: float | None
    reason: UndefinedReason | None

    @model_validator(mode="after")
    def _consistent(self) -> AgreementEstimate:
        if self.value is not None and (not math.isfinite(self.value) or not -1 <= self.value <= 1):
            raise ValueError("agreement_estimate_invalid")
        if self.chance_agreement is not None and (
            not math.isfinite(self.chance_agreement) or not 0 <= self.chance_agreement <= 1
        ):
            raise ValueError("agreement_estimate_invalid")
        if self.reason == "empty":
            valid = self.value is None and self.chance_agreement is None
        elif self.reason == "chance_agreement_one":
            valid = self.value is None and self.chance_agreement == 1
        else:
            valid = self.value is not None and self.chance_agreement != 1
        if not valid:
            raise ValueError("agreement_estimate_invalid")
        return self


def _ratios(
    counts: tuple[tuple[int, ...], ...], n: int
) -> tuple[Fraction | None, Fraction | None, Fraction | None, Fraction | None, Fraction | None]:
    if not n:
        return None, None, None, None, None
    size = len(counts)
    a = tuple(sum(row) for row in counts)
    b = tuple(sum(row[column] for row in counts) for column in range(size))
    observed = Fraction(sum(counts[i][i] for i in range(size)), n)
    kappa_chance = Fraction(sum(x * y for x, y in zip(a, b, strict=True)), n * n)
    pooled = tuple(x + y for x, y in zip(a, b, strict=True))
    ac1_chance = Fraction(sum(x * (2 * n - x) for x in pooled), 4 * n * n * (size - 1))
    kappa = (observed - kappa_chance) / (1 - kappa_chance) if kappa_chance != 1 else None
    ac1 = (observed - ac1_chance) / (1 - ac1_chance)
    return observed, kappa, ac1, kappa_chance, ac1_chance


def _estimate(value: Fraction | None, chance: Fraction | None = None) -> AgreementEstimate:
    return AgreementEstimate(
        value=float(value) if value is not None else None,
        chance_agreement=float(chance) if chance is not None else None,
        reason=("chance_agreement_one" if chance == 1 else "empty") if value is None else None,
    )


class PairwiseAgreement(_Frozen):
    """Complete paired population, contingency counts and all three point estimates."""

    metric_version: Literal["nominal-two-human-raters-v1"] = AGREEMENT_VERSION
    labels: tuple[str, ...]
    item_ids: tuple[str, ...]
    rater_a_id: str
    rater_b_id: str
    rater_kind: Literal["human"] = "human"
    counts: tuple[tuple[int, ...], ...]
    n: int = Field(ge=0)
    agreement_count: int = Field(ge=0)
    percent_agreement: AgreementEstimate
    cohen_kappa: AgreementEstimate
    gwet_ac1: AgreementEstimate
    confidence_method: Literal["not_produced"] = "not_produced"

    @model_validator(mode="after")
    def _conserves(self) -> PairwiseAgreement:
        size = len(self.labels)
        if (
            not _categories(self.labels)
            or not _unique(self.item_ids)
            or len(self.item_ids) != self.n
            or not _unique((self.rater_a_id, self.rater_b_id))
            or len(self.counts) != size
            or any(len(row) != size or any(value < 0 for value in row) for row in self.counts)
            or sum(map(sum, self.counts)) != self.n
            or self.agreement_count != sum(self.counts[i][i] for i in range(size))
        ):
            raise ValueError("agreement_report_invalid")
        observed, kappa, ac1, pk, pg = _ratios(self.counts, self.n)
        if (
            self.percent_agreement != _estimate(observed)
            or self.cohen_kappa != _estimate(kappa, pk)
            or self.gwet_ac1 != _estimate(ac1, pg)
        ):
            raise ValueError("agreement_report_invalid")
        return self


class RegisteredReference(_Frozen):
    """Caller-owned reference provenance; registration and solver correctness are not attested."""

    item_id: str
    solver_gold: str | None
    adjudicated: str | None
    source_ref: str
    version_ref: str


class ResolvedReference(_Frozen):
    """The explicit reference chosen by gold precedence, retaining its supplied provenance."""

    item_id: str
    label: str
    basis: Literal["solver_gold", "adjudicated"]
    source_ref: str
    version_ref: str


class ReferencePrevalence(_Frozen):
    """Reference-only class counts over a caller-declared registered item set and version."""

    metric_version: Literal["registered-reference-prevalence-v1"] = REFERENCE_VERSION
    labels: tuple[str, ...]
    item_ids: tuple[str, ...]
    item_set_ref: str
    reference_version: str
    references: tuple[ResolvedReference, ...]
    counts: tuple[int, ...]
    n: int = Field(ge=0)

    @model_validator(mode="after")
    def _conserves(self) -> ReferencePrevalence:
        if (
            not _categories(self.labels)
            or not _unique(self.item_ids)
            or not _nonempty((self.item_set_ref, self.reference_version))
            or len(self.item_ids) != self.n
            or tuple(item.item_id for item in self.references) != self.item_ids
            or any(
                item.label not in self.labels or not _nonempty((item.source_ref, item.version_ref))
                for item in self.references
            )
            or self.counts
            != tuple(sum(item.label == label for item in self.references) for label in self.labels)
        ):
            raise ValueError("agreement_reference_invalid")
        return self


def _threshold(value: float) -> float:
    result: float | None = None
    try:
        if type(value) in {int, float}:
            number = float(value)
            if math.isfinite(number):
                result = number
    except OverflowError, ValueError:
        pass
    if result is None:
        raise ValidationError("agreement_threshold_invalid")
    return result


def _gate_values(
    agreement: PairwiseAgreement, prevalence: ReferencePrevalence, threshold: float
) -> tuple[
    CoefficientName | None, float | None, bool | None, _GateReason | None, UndefinedReason | None
]:
    if agreement.labels != prevalence.labels or agreement.item_ids != prevalence.item_ids:
        raise ValidationError("agreement_population_mismatch")
    if not agreement.n:
        return None, None, None, "empty", "empty"
    selected: CoefficientName = (
        "gwet_ac1" if 5 * max(prevalence.counts) >= 4 * prevalence.n else "cohen_kappa"
    )
    _, kappa, ac1, _, _ = _ratios(agreement.counts, agreement.n)
    value = ac1 if selected == "gwet_ac1" else kappa
    if value is None:
        return selected, None, None, "selected_coefficient_undefined", "chance_agreement_one"
    # The serialized threshold's decimal spelling is its policy value; no epsilon or rounded
    # displayed coefficient can move an inclusive gate boundary.
    return selected, float(value), value >= Fraction(str(threshold)), None, None


class AgreementGate(_Frozen):
    """A reference-selected point comparison; not annotation attestation or a completed gate."""

    gate_version: Literal["reference-prevalence-agreement-gate-v1"] = GATE_VERSION
    agreement: PairwiseAgreement
    prevalence: ReferencePrevalence
    threshold: float = Field(allow_inf_nan=False)
    selected_coefficient: CoefficientName | None
    selected_value: float | None
    passed: bool | None
    reason: _GateReason | None
    selected_reason: UndefinedReason | None
    confidence_method: Literal["not_produced"] = "not_produced"

    @model_validator(mode="after")
    def _consistent(self) -> AgreementGate:
        expected = _gate_values(self.agreement, self.prevalence, self.threshold)
        actual = (
            self.selected_coefficient,
            self.selected_value,
            self.passed,
            self.reason,
            self.selected_reason,
        )
        if actual != expected:
            raise ValueError("agreement_gate_invalid")
        return self


def _checked[T: _Frozen](schema: type[T], value: T, code: str) -> T:
    result: T | None = None
    try:
        if type(value) is schema:
            result = schema.model_validate(value.model_dump(mode="python", warnings=False))
    except ValueError, TypeError:
        pass
    if result is None:
        raise ValidationError(code)
    return result


def _label_order(labels: Sequence[str]) -> tuple[str, ...]:
    if not _sequence(labels):
        raise ValidationError("agreement_labels_invalid")
    names = tuple(labels)
    if any(type(label) is not str for label in names) or not _categories(names):
        raise ValidationError("agreement_labels_invalid")
    return names


def pairwise_agreement(
    rater_a: HumanRatings, rater_b: HumanRatings, *, labels: Sequence[str]
) -> PairwiseAgreement:
    """Calculate nominal agreement for two complete, identically ordered human declarations."""
    names = _label_order(labels)
    a = _checked(HumanRatings, rater_a, "agreement_input_invalid")
    b = _checked(HumanRatings, rater_b, "agreement_input_invalid")
    if (
        a.rater_id == b.rater_id
        or a.item_ids != b.item_ids
        or any(label not in names for label in (*a.ratings, *b.ratings))
    ):
        raise ValidationError("agreement_input_invalid")
    order = {label: i for i, label in enumerate(names)}
    matrix = [[0] * len(names) for _ in names]
    for left, right in zip(a.ratings, b.ratings, strict=True):
        matrix[order[left]][order[right]] += 1
    counts = tuple(tuple(row) for row in matrix)
    observed, kappa, ac1, pk, pg = _ratios(counts, len(a.item_ids))
    return PairwiseAgreement(
        labels=names,
        item_ids=a.item_ids,
        rater_a_id=a.rater_id,
        rater_b_id=b.rater_id,
        counts=counts,
        n=len(a.item_ids),
        agreement_count=sum(counts[i][i] for i in range(len(names))),
        percent_agreement=_estimate(observed),
        cohen_kappa=_estimate(kappa, pk),
        gwet_ac1=_estimate(ac1, pg),
    )


def reference_prevalence(
    references: Sequence[RegisteredReference],
    *,
    labels: Sequence[str],
    item_set_ref: str,
    reference_version: str,
) -> ReferencePrevalence:
    """Count every registered reference, choosing supplied gold before adjudication."""
    names = _label_order(labels)
    if not _sequence(references):
        raise ValidationError("agreement_reference_invalid")
    if any(
        type(value) is not str or not value.strip() for value in (item_set_ref, reference_version)
    ):
        raise ValidationError("agreement_reference_invalid")
    resolved: list[ResolvedReference] = []
    for raw in references:
        item = _checked(RegisteredReference, raw, "agreement_reference_invalid")
        if not _nonempty((item.item_id, item.source_ref, item.version_ref)) or any(
            label is not None and label not in names
            for label in (item.solver_gold, item.adjudicated)
        ):
            raise ValidationError("agreement_reference_invalid")
        label = item.solver_gold if item.solver_gold is not None else item.adjudicated
        if label is None:
            raise ValidationError("agreement_reference_invalid")
        resolved.append(
            ResolvedReference(
                item_id=item.item_id,
                label=label,
                basis="solver_gold" if item.solver_gold is not None else "adjudicated",
                source_ref=item.source_ref,
                version_ref=item.version_ref,
            )
        )
    ids = tuple(item.item_id for item in resolved)
    if not _unique(ids):
        raise ValidationError("agreement_reference_invalid")
    return ReferencePrevalence(
        labels=names,
        item_ids=ids,
        item_set_ref=item_set_ref,
        reference_version=reference_version,
        references=tuple(resolved),
        counts=tuple(sum(item.label == label for item in resolved) for label in names),
        n=len(resolved),
    )


def research_annotation_gate(
    agreement: PairwiseAgreement,
    prevalence: ReferencePrevalence,
    *,
    threshold: float = RESEARCH_ANNOTATION_THRESHOLD,
) -> AgreementGate:
    """Select AC1 at inclusive 80% reference prevalence, otherwise kappa; compare inclusively."""
    paired = _checked(PairwiseAgreement, agreement, "agreement_report_invalid")
    reference = _checked(ReferencePrevalence, prevalence, "agreement_reference_invalid")
    limit = _threshold(threshold)
    selected, value, passed, reason, selected_reason = _gate_values(paired, reference, limit)
    return AgreementGate(
        agreement=paired,
        prevalence=reference,
        threshold=limit,
        selected_coefficient=selected,
        selected_value=value,
        passed=passed,
        reason=reason,
        selected_reason=selected_reason,
    )
