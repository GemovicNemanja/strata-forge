"""Nominal two-human-rater agreement, a registered-reference selector and its bootstrap."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strata_forge.core.errors import ValidationError

AGREEMENT_VERSION = "nominal-two-human-raters-v1"
REFERENCE_VERSION = "registered-reference-prevalence-v1"
GATE_VERSION = "reference-prevalence-agreement-gate-v1"
BOOTSTRAP_VERSION = "item-percentile-bootstrap-v1"
BOOTSTRAP_RNG = "numpy.random.Generator(PCG64)"
BOOTSTRAP_QUANTILE_RULE = "hyndman-fan-7"
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 10_000
MAX_BOOTSTRAP_WORK = 20_000_000
MAX_BOOTSTRAP_SEED = 2**53 - 1
RESEARCH_ANNOTATION_THRESHOLD = 0.70
HUMAN_MODEL_VERSION = "nominal-human-vs-model-v1"
MAX_HUMAN_SKIP_FRACTION = 0.10
MAX_MODEL_NULL_FRACTION = 0.10
type CoefficientName = Literal["cohen_kappa", "gwet_ac1"]
type UndefinedReason = Literal["empty", "chance_agreement_one"]
type _GateReason = Literal["empty", "selected_coefficient_undefined"]

type HumanModelStatus = Literal["pass", "fail", "refused"]
type HumanModelReason = Literal[
    "below_threshold",
    "human_skips_exceeded",
    "empty",
    "reference_missing",
    "model_nulls_exceeded",
    "no_used_items",
    "coefficient_undefined",
]

__all__ = [
    "AGREEMENT_VERSION",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_QUANTILE_RULE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_RNG",
    "BOOTSTRAP_VERSION",
    "GATE_VERSION",
    "HUMAN_MODEL_VERSION",
    "MAX_BOOTSTRAP_SEED",
    "MAX_BOOTSTRAP_WORK",
    "MAX_HUMAN_SKIP_FRACTION",
    "MAX_MODEL_NULL_FRACTION",
    "REFERENCE_VERSION",
    "RESEARCH_ANNOTATION_THRESHOLD",
    "AgreementBootstrap",
    "AgreementEstimate",
    "AgreementGate",
    "BootstrapInterval",
    "ExactValue",
    "HumanAnnotations",
    "HumanModelAgreement",
    "HumanModelReason",
    "HumanModelStatus",
    "HumanRatings",
    "ModelRatings",
    "PairwiseAgreement",
    "ReferencePrevalence",
    "RegisteredReference",
    "ReportOnlyKappa",
    "ResolvedReference",
    "agreement_bootstrap",
    "human_model_agreement",
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


def _select_from_counts(counts: Sequence[int], n: int) -> CoefficientName:
    return "gwet_ac1" if 5 * max(counts) >= 4 * n else "cohen_kappa"


def _select_coefficient(prevalence: ReferencePrevalence) -> CoefficientName:
    return _select_from_counts(prevalence.counts, prevalence.n)


def _gate_values(
    agreement: PairwiseAgreement, prevalence: ReferencePrevalence, threshold: float
) -> tuple[
    CoefficientName | None, float | None, bool | None, _GateReason | None, UndefinedReason | None
]:
    if agreement.labels != prevalence.labels or agreement.item_ids != prevalence.item_ids:
        raise ValidationError("agreement_population_mismatch")
    if not agreement.n:
        return None, None, None, "empty", "empty"
    selected = _select_coefficient(prevalence)
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


def _selected_estimate(
    agreement: PairwiseAgreement, coefficient: CoefficientName
) -> AgreementEstimate:
    return agreement.gwet_ac1 if coefficient == "gwet_ac1" else agreement.cohen_kappa


class BootstrapInterval(_Frozen):
    """The approved item-level percentile interval block, without any item identity."""

    bootstrap_version: Literal["item-percentile-bootstrap-v1"] = BOOTSTRAP_VERSION
    rng: Literal["numpy.random.Generator(PCG64)"] = BOOTSTRAP_RNG
    quantile_rule: Literal["hyndman-fan-7"] = BOOTSTRAP_QUANTILE_RULE
    coefficient_selection: Literal["full_sample_reference"] = "full_sample_reference"
    confidence_level: float = Field(default=BOOTSTRAP_CONFIDENCE, allow_inf_nan=False)
    seed: int = Field(ge=0, le=MAX_BOOTSTRAP_SEED)
    resamples: int = Field(ge=1, le=MAX_BOOTSTRAP_WORK)
    undefined_resamples: int = Field(ge=0)
    lower: float | None
    upper: float | None
    reason: Literal["all_resamples_undefined"] | None
    numpy_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> BootstrapInterval:
        if self.confidence_level != BOOTSTRAP_CONFIDENCE or (
            self.undefined_resamples > self.resamples
        ):
            raise ValueError("agreement_bootstrap_invalid")
        if self.undefined_resamples == self.resamples:
            valid = self.lower is None and self.upper is None and self.reason is not None
        else:
            valid = (
                self.reason is None
                and self.lower is not None
                and self.upper is not None
                and all(math.isfinite(bound) for bound in (self.lower, self.upper))
                and -1 <= self.lower <= self.upper <= 1
            )
        if not valid:
            raise ValueError("agreement_bootstrap_invalid")
        return self


class AgreementBootstrap(_Frozen):
    """An item-level percentile interval for the coefficient the full sample selected."""

    agreement: PairwiseAgreement
    prevalence: ReferencePrevalence
    coefficient: CoefficientName
    estimate: AgreementEstimate
    interval: BootstrapInterval

    @model_validator(mode="after")
    def _consistent(self) -> AgreementBootstrap:
        paired, reference = self.agreement, self.prevalence
        if (
            paired.labels != reference.labels
            or paired.item_ids != reference.item_ids
            or not paired.n
            or self.coefficient != _select_coefficient(reference)
            or self.estimate != _selected_estimate(paired, self.coefficient)
            or self.interval.resamples * max(paired.n, len(paired.labels) ** 2) > MAX_BOOTSTRAP_WORK
            or (self.estimate.value is None and self.interval.reason is None)
        ):
            raise ValueError("agreement_bootstrap_invalid")
        return self


def _bootstrap_count(value: object, low: int, high: int, code: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValidationError(code)
    return value


def _resampled_coefficients(
    cells: Sequence[int], size: int, coefficient: CoefficientName, *, seed: int, resamples: int
) -> tuple[tuple[float, ...], int, str]:
    """Return sorted defined resample values, the undefined count and the NumPy version."""
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "agreement_bootstrap requires numpy: install 'strata-forge[scoring]'"
        ) from exc
    n = len(cells)
    population = np.asarray(cells, dtype=np.int64)
    draws = np.random.Generator(np.random.PCG64(seed)).integers(
        0, n, size=(resamples, n), dtype=np.int64
    )
    offsets = (np.arange(resamples, dtype=np.int64) * (size * size))[:, None]
    counts = np.bincount(
        (population[draws] + offsets).ravel(), minlength=resamples * size * size
    ).reshape(resamples, size, size)
    diagonal = np.trace(counts, axis1=1, axis2=2)
    row_totals = counts.sum(axis=2)
    column_totals = counts.sum(axis=1)
    # Both coefficients are reduced to one exact integer ratio per resample, so a single
    # correctly rounded division equals float(Fraction) of the point-estimate arithmetic.
    if coefficient == "cohen_kappa":
        chance = (row_totals * column_totals).sum(axis=1)
        numerator = n * diagonal - chance
        denominator = n * n - chance
    else:
        pooled = row_totals + column_totals
        chance = (pooled * (2 * n - pooled)).sum(axis=1)
        numerator = 4 * n * (size - 1) * diagonal - chance
        denominator = 4 * n * n * (size - 1) - chance
    defined = denominator != 0
    values = np.sort(numerator[defined] / denominator[defined])
    undefined = resamples - int(np.count_nonzero(defined))
    return tuple(float(value) for value in values.tolist()), undefined, str(np.__version__)


def _percentile(values: tuple[float, ...], probability: Fraction) -> float:
    position = (len(values) - 1) * probability
    index = math.floor(position)
    low = values[index]
    if index + 1 == len(values):
        return low
    high = values[index + 1]
    weight = float(position - index)
    return min(max(low + weight * (high - low), low), high)


def _interval(
    cells: Sequence[int], size: int, coefficient: CoefficientName, *, seed: int, resamples: int
) -> BootstrapInterval:
    if resamples * max(len(cells), size * size) > MAX_BOOTSTRAP_WORK or (
        4 * len(cells) * len(cells) * (size - 1) > 2**53
    ):
        raise ValidationError("agreement_bootstrap_too_large")
    values, undefined, numpy_version = _resampled_coefficients(
        cells, size, coefficient, seed=seed, resamples=resamples
    )
    tail = (1 - Fraction(str(BOOTSTRAP_CONFIDENCE))) / 2
    return BootstrapInterval(
        seed=seed,
        resamples=resamples,
        undefined_resamples=undefined,
        lower=_percentile(values, tail) if values else None,
        upper=_percentile(values, 1 - tail) if values else None,
        reason=None if values else "all_resamples_undefined",
        numpy_version=numpy_version,
    )


def _bootstrap_settings(seed: object, resamples: object) -> tuple[int, int]:
    return (
        _bootstrap_count(seed, 0, MAX_BOOTSTRAP_SEED, "agreement_bootstrap_seed_invalid"),
        _bootstrap_count(resamples, 1, MAX_BOOTSTRAP_WORK, "agreement_bootstrap_resamples_invalid"),
    )


def agreement_bootstrap(
    rater_a: HumanRatings,
    rater_b: HumanRatings,
    prevalence: ReferencePrevalence,
    *,
    labels: Sequence[str],
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> AgreementBootstrap:
    """Bootstrap items with replacement for the coefficient the full-sample reference selects."""
    paired = pairwise_agreement(rater_a, rater_b, labels=labels)
    reference = _checked(ReferencePrevalence, prevalence, "agreement_reference_invalid")
    if paired.labels != reference.labels or paired.item_ids != reference.item_ids:
        raise ValidationError("agreement_population_mismatch")
    if not paired.n:
        raise ValidationError("agreement_bootstrap_empty")
    chosen_seed, count = _bootstrap_settings(seed, resamples)
    size = len(paired.labels)
    coefficient = _select_coefficient(reference)
    order = {label: i for i, label in enumerate(paired.labels)}
    cells = tuple(
        order[left] * size + order[right]
        for left, right in zip(rater_a.ratings, rater_b.ratings, strict=True)
    )
    return AgreementBootstrap(
        agreement=paired,
        prevalence=reference,
        coefficient=coefficient,
        estimate=_selected_estimate(paired, coefficient),
        interval=_interval(cells, size, coefficient, seed=chosen_seed, resamples=count),
    )


class HumanAnnotations(_Frozen):
    """One caller-declared human rater whose `None` entries are explicit skips."""

    rater_id: str = Field(min_length=1)
    rater_kind: Literal["human"]
    item_ids: tuple[str, ...]
    ratings: tuple[str | None, ...]

    @model_validator(mode="after")
    def _complete(self) -> HumanAnnotations:
        if (
            not self.rater_id.strip()
            or not _unique(self.item_ids)
            or not _nonempty(tuple(label for label in self.ratings if label is not None))
            or len(self.item_ids) != len(self.ratings)
        ):
            raise ValueError("agreement_input_invalid")
        return self


class ModelRatings(_Frozen):
    """One model rater with its prompt digest and run reference; `None` is a null label."""

    rater_id: str = Field(min_length=1)
    rater_kind: Literal["model"]
    item_ids: tuple[str, ...]
    ratings: tuple[str | None, ...]
    model_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_run_ref: str | None

    @model_validator(mode="after")
    def _complete(self) -> ModelRatings:
        if (
            not self.rater_id.strip()
            or not _unique(self.item_ids)
            or not _nonempty(tuple(label for label in self.ratings if label is not None))
            or len(self.item_ids) != len(self.ratings)
            or (self.model_run_ref is not None and not self.model_run_ref.strip())
        ):
            raise ValueError("agreement_input_invalid")
        return self


class ExactValue(_Frozen):
    """A coefficient as its exact rational spelling and the float nearest to it."""

    exact: str = Field(pattern=r"^-?[0-9]+(/[0-9]+)?$")
    value: float = Field(ge=-1, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _consistent(self) -> ExactValue:
        rational = Fraction(self.exact)
        if str(rational) != self.exact or float(rational) != self.value:
            raise ValueError("agreement_value_invalid")
        return self

    @classmethod
    def of(cls, value: Fraction) -> ExactValue:
        """Spell a rational coefficient exactly."""
        return cls(exact=str(value), value=float(value))


class ReportOnlyKappa(_Frozen):
    """User-versus-solver-gold Cohen kappa; reported beside the gate and never gating."""

    role: Literal["report_only"] = "report_only"
    n: int = Field(ge=0)
    value: ExactValue | None
    reason: UndefinedReason | None

    @model_validator(mode="after")
    def _consistent(self) -> ReportOnlyKappa:
        if (self.value is None) == (self.reason is None) or (self.reason == "empty") != (
            self.n == 0
        ):
            raise ValueError("agreement_report_invalid")
        return self


def _exceeds(count: int, items: int, fraction: float) -> bool:
    return count > Fraction(str(fraction)) * items


class HumanModelAgreement(_Frozen):
    """Counted exclusions, a reference-selected gated point value and its reported interval."""

    statistic_version: Literal["nominal-human-vs-model-v1"] = HUMAN_MODEL_VERSION
    items: int = Field(ge=0)
    human_skipped: int = Field(ge=0)
    model_null: int = Field(ge=0)
    used: int = Field(ge=0)
    max_human_skip_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    max_model_null_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    status: HumanModelStatus
    reason: HumanModelReason | None
    coefficient: CoefficientName | None
    value: ExactValue | None
    cohen_kappa: ExactValue | None
    cohen_kappa_reason: Literal["chance_agreement_one"] | None
    threshold: float = Field(allow_inf_nan=False)
    interval: BootstrapInterval | None
    user_solver_kappa: ReportOnlyKappa
    model_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_run_ref: str | None

    @model_validator(mode="after")
    def _consistent(self) -> HumanModelAgreement:
        excluded = self.items - self.used
        skips = _exceeds(self.human_skipped, self.items, self.max_human_skip_fraction)
        nulls = _exceeds(self.model_null, self.items, self.max_model_null_fraction)
        refused = self.status == "refused"
        valued = self.value is not None
        checks = (
            self.used <= self.items,
            max(self.human_skipped, self.model_null) <= excluded,
            excluded <= self.human_skipped + self.model_null,
            refused == (self.reason not in {None, "below_threshold", "human_skips_exceeded"}),
            (self.status == "pass") == (self.reason is None),
            self.reason != "empty" or self.items == 0,
            self.reason != "model_nulls_exceeded" or nulls,
            self.reason in {"empty", "reference_missing", "model_nulls_exceeded"} or not nulls,
            self.reason != "no_used_items" or self.used == 0,
            self.reason != "human_skips_exceeded" or skips,
            refused or self.reason == "human_skips_exceeded" or not skips,
            valued == (not refused),
            (self.interval is not None) == valued,
            (self.coefficient is None) == (self.reason in {"empty", "reference_missing"}),
            self.coefficient == "gwet_ac1"
            or (self.cohen_kappa is None and self.cohen_kappa_reason is None),
            not (self.cohen_kappa is not None and self.cohen_kappa_reason is not None),
            not valued
            or self.coefficient != "gwet_ac1"
            or (self.cohen_kappa is None) != (self.cohen_kappa_reason is None),
            valued or (self.cohen_kappa is None and self.cohen_kappa_reason is None),
        )
        if not all(checks):
            raise ValueError("agreement_human_model_invalid")
        if self.status == "pass" or self.reason == "below_threshold":
            met = self.value is not None and Fraction(self.value.exact) >= Fraction(
                str(self.threshold)
            )
            if met != (self.status == "pass"):
                raise ValueError("agreement_human_model_invalid")
        return self


def _limit(value: object) -> float:
    if type(value) not in {int, float}:
        raise ValidationError("agreement_exclusion_limit_invalid")
    number = float(cast("float", value))
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValidationError("agreement_exclusion_limit_invalid")
    return number


def _counts(pairs: Sequence[tuple[str, str]], order: dict[str, int]) -> tuple[tuple[int, ...], ...]:
    matrix = [[0] * len(order) for _ in order]
    for left, right in pairs:
        matrix[order[left]][order[right]] += 1
    return tuple(tuple(row) for row in matrix)


def _user_solver(
    human: HumanAnnotations, references: Sequence[RegisteredReference], order: dict[str, int]
) -> ReportOnlyKappa:
    pairs = [
        (label, reference.solver_gold)
        for label, reference in zip(human.ratings, references, strict=True)
        if label is not None and reference.solver_gold is not None
    ]
    _, kappa, _, chance, _ = _ratios(_counts(pairs, order), len(pairs))
    reason: UndefinedReason | None = None
    if kappa is None:
        reason = "chance_agreement_one" if chance == 1 else "empty"
    return ReportOnlyKappa(
        n=len(pairs), value=ExactValue.of(kappa) if kappa is not None else None, reason=reason
    )


def human_model_agreement(
    human: HumanAnnotations,
    model: ModelRatings,
    references: Sequence[RegisteredReference],
    *,
    labels: Sequence[str],
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    threshold: float = RESEARCH_ANNOTATION_THRESHOLD,
    max_human_skip_fraction: float = MAX_HUMAN_SKIP_FRACTION,
    max_model_null_fraction: float = MAX_MODEL_NULL_FRACTION,
) -> HumanModelAgreement:
    """Gate one human against one model on a slice, excluding and counting skips and nulls."""
    names = _label_order(labels)
    person = _checked(HumanAnnotations, human, "agreement_input_invalid")
    machine = _checked(ModelRatings, model, "agreement_input_invalid")
    if (
        person.rater_id == machine.rater_id
        or person.item_ids != machine.item_ids
        or any(
            label is not None and label not in names
            for label in (*person.ratings, *machine.ratings)
        )
    ):
        raise ValidationError("agreement_input_invalid")
    if not _sequence(references):
        raise ValidationError("agreement_reference_invalid")
    refs = tuple(
        _checked(RegisteredReference, item, "agreement_reference_invalid") for item in references
    )
    if tuple(item.item_id for item in refs) != person.item_ids or any(
        label is not None and label not in names
        for item in refs
        for label in (item.solver_gold, item.adjudicated)
    ):
        raise ValidationError("agreement_reference_invalid")
    limit = _threshold(threshold)
    skip_limit = _limit(max_human_skip_fraction)
    null_limit = _limit(max_model_null_fraction)
    chosen_seed, count = _bootstrap_settings(seed, resamples)
    order = {label: i for i, label in enumerate(names)}
    items = len(person.item_ids)
    pairs = [
        (left, right)
        for left, right in zip(person.ratings, machine.ratings, strict=True)
        if left is not None and right is not None
    ]
    skipped = sum(label is None for label in person.ratings)
    nulls = sum(label is None for label in machine.ratings)
    reference_labels = [
        item.solver_gold if item.solver_gold is not None else item.adjudicated for item in refs
    ]
    coefficient: CoefficientName | None = None
    reason: HumanModelReason | None = None
    if not items:
        reason = "empty"
    elif any(label is None for label in reference_labels):
        reason = "reference_missing"
    else:
        coefficient = _select_from_counts([reference_labels.count(label) for label in names], items)
        if _exceeds(nulls, items, null_limit):
            reason = "model_nulls_exceeded"
        elif not pairs:
            reason = "no_used_items"
    value: Fraction | None = None
    kappa: Fraction | None = None
    kappa_chance: Fraction | None = None
    if reason is None and coefficient is not None:
        _, kappa, ac1, kappa_chance, _ = _ratios(_counts(pairs, order), len(pairs))
        value = ac1 if coefficient == "gwet_ac1" else kappa
        if value is None:
            reason = "coefficient_undefined"
    interval: BootstrapInterval | None = None
    status: HumanModelStatus = "refused"
    if value is not None and coefficient is not None:
        size = len(names)
        cells = tuple(order[left] * size + order[right] for left, right in pairs)
        interval = _interval(cells, size, coefficient, seed=chosen_seed, resamples=count)
        if _exceeds(skipped, items, skip_limit):
            status, reason = "fail", "human_skips_exceeded"
        elif value >= Fraction(str(limit)):
            status = "pass"
        else:
            status, reason = "fail", "below_threshold"
    with_kappa = value is not None and coefficient == "gwet_ac1"
    return HumanModelAgreement(
        items=items,
        human_skipped=skipped,
        model_null=nulls,
        used=len(pairs),
        max_human_skip_fraction=skip_limit,
        max_model_null_fraction=null_limit,
        status=status,
        reason=reason,
        coefficient=coefficient,
        value=ExactValue.of(value) if value is not None else None,
        cohen_kappa=ExactValue.of(kappa) if with_kappa and kappa is not None else None,
        cohen_kappa_reason="chance_agreement_one"
        if with_kappa and kappa is None and kappa_chance == 1
        else None,
        threshold=limit,
        interval=interval,
        user_solver_kappa=_user_solver(person, refs, order),
        model_prompt_sha256=machine.model_prompt_sha256,
        model_run_ref=machine.model_run_ref,
    )
