"""Deterministic classification point metrics with explicit labels and denominators."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.core.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

METRIC_VERSION = "classification-point-v1"
PROBABILITY_SUM_TOLERANCE = 1e-6
NLL_CLIP_EPSILON = 1e-15
type ProbabilityKind = Literal["class_probabilities", "restricted_softmax", "censored_topk"]

__all__ = [
    "METRIC_VERSION",
    "NLL_CLIP_EPSILON",
    "PROBABILITY_SUM_TOLERANCE",
    "ClassScore",
    "ClassificationReport",
    "ConfusionMatrix",
    "LabelSpace",
    "MetricValue",
    "ProbabilityKind",
    "accuracy",
    "brier",
    "classification_report",
    "confusion_matrix",
    "invalid_rate",
    "macro_f1",
    "nll",
]


class LabelSpace(BaseModel):
    """An explicit class order; input indexes never infer this order."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)
    labels: tuple[str, ...] = Field(min_length=2)

    @field_validator("labels")
    @classmethod
    def _valid_labels(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
            raise ValueError("labels must be unique nonempty strings")
        return labels


class MetricValue(BaseModel):
    """A value, its actual denominator, and omissions from the eligible population."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)
    value: float | None
    n: int = Field(ge=0)
    excluded: int = Field(default=0, ge=0)
    reason: Literal["empty", "no_probabilities"] | None = None
    probability_kind: Literal["class_probabilities", "restricted_softmax"] | None = None
    calibration_ref: str | None = Field(default=None, min_length=1, max_length=256)
    clip_epsilon: float | None = None
    clipped: int = Field(default=0, ge=0)
    normalization_tolerance: float | None = None
    coverage: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _consistent_value(self) -> MetricValue:
        if self.value is not None and (
            not math.isfinite(self.value) or self.n == 0 or self.reason is not None
        ):
            raise ValueError(
                "a metric value requires a positive denominator and no undefined reason"
            )
        if self.value is None and (self.n != 0 or self.reason is None):
            raise ValueError("an absent metric requires a reason and zero denominator")
        if self.clipped > self.n:
            raise ValueError("clipping count exceeds the metric denominator")
        if self.clip_epsilon is not None and (
            not math.isfinite(self.clip_epsilon) or not 0 < self.clip_epsilon < 1
        ):
            raise ValueError("clipping epsilon must be finite in (0, 1)")
        if self.normalization_tolerance is not None and (
            not math.isfinite(self.normalization_tolerance)
            or not 0 <= self.normalization_tolerance < 1
        ):
            raise ValueError("normalization tolerance must be finite in [0, 1)")
        if self.reason == "empty" and self.excluded:
            raise ValueError("an empty population cannot contain excluded rows")
        if self.reason == "no_probabilities" and not self.excluded:
            raise ValueError("missing probabilities require an eligible population")
        total = self.n + self.excluded
        coverage = self.n / total if total else None
        if self.coverage is not None and self.coverage != coverage:
            raise ValueError("coverage must equal n / (n + excluded)")
        object.__setattr__(self, "coverage", coverage)
        return self


class ConfusionMatrix(BaseModel):
    """Rows are gold and columns predicted; invalid outputs remain in gold support."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)
    labels: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]
    invalid_by_gold: tuple[int, ...]
    n: int = Field(ge=0)

    @model_validator(mode="after")
    def _conserves_population(self) -> ConfusionMatrix:
        LabelSpace(labels=self.labels)
        size = len(self.labels)
        if (
            len(self.counts) != size
            or any(len(row) != size for row in self.counts)
            or len(self.invalid_by_gold) != size
        ):
            raise ValueError("confusion matrix dimensions must match the label order")
        values = [value for row in self.counts for value in row] + list(self.invalid_by_gold)
        if any(value < 0 for value in values) or sum(values) != self.n:
            raise ValueError("confusion counts and invalid outputs must conserve the population")
        return self


class ClassScore(BaseModel):
    """One class's support, including invalid outputs as false negatives."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    label: str
    support: int = Field(ge=0)
    predicted: int = Field(ge=0)
    precision: float = Field(ge=0, le=1, allow_inf_nan=False)
    recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    f1: float = Field(ge=0, le=1, allow_inf_nan=False)


class ClassificationReport(BaseModel):
    """Point metrics for one explicitly declared population and label space."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    metric_version: Literal["classification-point-v1"] = METRIC_VERSION
    confusion: ConfusionMatrix
    per_class: tuple[ClassScore, ...]
    accuracy: MetricValue
    macro_f1: MetricValue
    invalid_rate: MetricValue


def _label_order(labels: Sequence[str]) -> tuple[str, ...]:
    from pydantic import ValidationError as ModelValidationError

    if isinstance(labels, str | bytes):
        raise ValidationError("labels must be an ordered collection, not a single string")
    try:
        return LabelSpace(labels=tuple(labels)).labels
    except ModelValidationError as exc:
        raise ValidationError(
            "an explicit unique ordered label space of at least two classes is required"
        ) from exc


def _index_map(index_map: Mapping[int, str] | None, labels: tuple[str, ...]) -> dict[int, str]:
    out: dict[int, str] = {}
    if index_map is not None:
        for key, label in index_map.items():
            if isinstance(key, bool) or not isinstance(key, Integral) or label not in labels:
                raise ValidationError("index_map requires integer keys and declared output labels")
            out[int(key)] = label
    return out


def _label_index(
    value: object, order: dict[str, int], indexes: dict[int, str], *, gold: bool
) -> int | None:
    if not isinstance(value, bool) and isinstance(value, Integral):
        if not indexes:
            raise ValidationError("integer labels require an explicit index_map")
        value = indexes.get(int(value))
    if isinstance(value, str) and value in order:
        return order[value]
    if gold:
        raise ValidationError("every gold label must belong to the declared label space")
    return None


def _gold_indexes(
    gold: Iterable[object], labels: tuple[str, ...], index_map: Mapping[int, str] | None
) -> tuple[list[int], dict[int, str]]:
    indexes = _index_map(index_map, labels)
    order = {label: index for index, label in enumerate(labels)}
    result: list[int] = []
    for value in gold:
        index = _label_index(value, order, indexes, gold=True)
        if index is None:
            raise ValidationError("every gold label must belong to the declared label space")
        result.append(index)
    return result, indexes


def confusion_matrix(
    gold: Iterable[object],
    predictions: Iterable[object],
    *,
    labels: Sequence[str],
    index_map: Mapping[int, str] | None = None,
) -> ConfusionMatrix:
    """Count valid predictions and preserve invalid outputs in each gold class's support."""
    names = _label_order(labels)
    actual, indexes = _gold_indexes(gold, names, index_map)
    predicted = list(predictions)
    if len(actual) != len(predicted):
        raise ValidationError("gold and prediction populations must have the same length")
    order = {label: index for index, label in enumerate(names)}
    counts = [[0] * len(names) for _ in names]
    invalid = [0] * len(names)
    for target, value in zip(actual, predicted, strict=True):
        guess = _label_index(value, order, indexes, gold=False)
        if guess is None:
            invalid[target] += 1
        else:
            counts[target][guess] += 1
    return ConfusionMatrix(
        labels=names,
        counts=tuple(tuple(row) for row in counts),
        invalid_by_gold=tuple(invalid),
        n=len(actual),
    )


def _value(value: float, n: int) -> MetricValue:
    return MetricValue(value=value if n else None, n=n, reason=None if n else "empty")


def _report(matrix: ConfusionMatrix) -> ClassificationReport:
    scores: list[ClassScore] = []
    for index, name in enumerate(matrix.labels):
        tp = matrix.counts[index][index]
        support = sum(matrix.counts[index]) + matrix.invalid_by_gold[index]
        predicted = sum(row[index] for row in matrix.counts)
        denominator = support + predicted
        scores.append(
            ClassScore(
                label=name,
                support=support,
                predicted=predicted,
                precision=tp / predicted if predicted else 0.0,
                recall=tp / support if support else 0.0,
                f1=2 * tp / denominator if denominator else 0.0,
            )
        )
    correct = sum(matrix.counts[index][index] for index in range(len(matrix.labels)))
    return ClassificationReport(
        confusion=matrix,
        per_class=tuple(scores),
        accuracy=_value(correct / matrix.n if matrix.n else 0.0, matrix.n),
        macro_f1=_value(math.fsum(score.f1 for score in scores) / len(scores), matrix.n),
        invalid_rate=_value(sum(matrix.invalid_by_gold) / matrix.n if matrix.n else 0.0, matrix.n),
    )


def classification_report(
    gold: Iterable[object],
    predictions: Iterable[object],
    *,
    labels: Sequence[str],
    index_map: Mapping[int, str] | None = None,
) -> ClassificationReport:
    """Summarize one population; absent declared classes contribute zero to macro-F1."""
    return _report(confusion_matrix(gold, predictions, labels=labels, index_map=index_map))


def accuracy(
    gold: Iterable[object],
    predictions: Iterable[object],
    *,
    labels: Sequence[str],
    index_map: Mapping[int, str] | None = None,
) -> MetricValue:
    """Return correct predictions / all gold rows, including invalid predictions as wrong."""
    return classification_report(gold, predictions, labels=labels, index_map=index_map).accuracy


def macro_f1(
    gold: Iterable[object],
    predictions: Iterable[object],
    *,
    labels: Sequence[str],
    index_map: Mapping[int, str] | None = None,
) -> MetricValue:
    """Average F1 across every declared class, with absent-class F1 defined as zero."""
    return classification_report(gold, predictions, labels=labels, index_map=index_map).macro_f1


def invalid_rate(
    gold: Iterable[object],
    predictions: Iterable[object],
    *,
    labels: Sequence[str],
    index_map: Mapping[int, str] | None = None,
) -> MetricValue:
    """Return invalid/missing outputs / all gold rows; invalid rows are not exclusions."""
    return classification_report(gold, predictions, labels=labels, index_map=index_map).invalid_rate


def _probability_rows(
    gold: Iterable[object],
    probabilities: Iterable[object],
    *,
    labels: Sequence[str],
    index_map: Mapping[int, str] | None,
    probability_kind: ProbabilityKind,
    tolerance: float,
) -> tuple[list[tuple[int, tuple[float, ...]]], int]:
    names = _label_order(labels)
    actual, _ = _gold_indexes(gold, names, index_map)
    rows = list(probabilities)
    if len(actual) != len(rows):
        raise ValidationError("gold and probability populations must have the same length")
    if probability_kind not in ("class_probabilities", "restricted_softmax"):
        raise ValidationError(
            "censored or unknown probability provenance cannot be scored as exact probabilities"
        )
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValidationError("probability normalization tolerance must be finite in [0, 1)")
    valid: list[tuple[int, tuple[float, ...]]] = []
    for target, row in zip(actual, rows, strict=True):
        if row is None:
            continue
        if isinstance(row, str | bytes) or not isinstance(row, Iterable):
            raise ValidationError("each probability row must be numeric or explicitly missing")
        values: list[float] = []
        for value in cast("Iterable[object]", row):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValidationError("probabilities must be finite real numbers in [0, 1]")
            number = float(value)
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise ValidationError("probabilities must be finite real numbers in [0, 1]")
            values.append(number)
        if len(values) != len(names) or abs(math.fsum(values) - 1) > tolerance:
            raise ValidationError(
                "probability rows must match the label order and sum to one; no normalization is inferred"
            )
        valid.append((target, tuple(values)))
    return valid, len(actual)


def _probability_value(
    values: Sequence[float],
    total: int,
    *,
    kind: ProbabilityKind,
    calibration_ref: str | None,
    tolerance: float,
    epsilon: float | None = None,
    clipped: int = 0,
) -> MetricValue:
    if kind not in ("class_probabilities", "restricted_softmax"):
        raise ValidationError("censored probability provenance cannot produce an exact metric")
    n = len(values)
    return MetricValue(
        value=math.fsum(values) / n if n else None,
        n=n,
        excluded=total - n,
        reason=None if n else "empty" if total == 0 else "no_probabilities",
        probability_kind=kind,
        calibration_ref=calibration_ref,
        normalization_tolerance=tolerance,
        clip_epsilon=epsilon,
        clipped=clipped,
    )


def brier(
    gold: Iterable[object],
    probabilities: Iterable[object],
    *,
    labels: Sequence[str],
    probability_kind: ProbabilityKind,
    index_map: Mapping[int, str] | None = None,
    calibration_ref: str | None = None,
    normalization_tolerance: float = PROBABILITY_SUM_TOLERANCE,
) -> MetricValue:
    """Mean multiclass sum of squared errors; missing probabilities expose reduced coverage."""
    rows, total = _probability_rows(
        gold,
        probabilities,
        labels=labels,
        index_map=index_map,
        probability_kind=probability_kind,
        tolerance=normalization_tolerance,
    )
    values = [
        math.fsum(
            (probability - int(index == target)) ** 2 for index, probability in enumerate(row)
        )
        for target, row in rows
    ]
    return _probability_value(
        values,
        total,
        kind=probability_kind,
        calibration_ref=calibration_ref,
        tolerance=normalization_tolerance,
    )


def nll(
    gold: Iterable[object],
    probabilities: Iterable[object],
    *,
    labels: Sequence[str],
    probability_kind: ProbabilityKind,
    index_map: Mapping[int, str] | None = None,
    calibration_ref: str | None = None,
    normalization_tolerance: float = PROBABILITY_SUM_TOLERANCE,
    clip_epsilon: float = NLL_CLIP_EPSILON,
) -> MetricValue:
    """Mean negative log gold probability; report the explicit clipping policy and count."""
    if (
        isinstance(clip_epsilon, bool)
        or not math.isfinite(clip_epsilon)
        or not 0 < clip_epsilon < 1
    ):
        raise ValidationError("NLL clip_epsilon must be finite in (0, 1)")
    rows, total = _probability_rows(
        gold,
        probabilities,
        labels=labels,
        index_map=index_map,
        probability_kind=probability_kind,
        tolerance=normalization_tolerance,
    )
    values = [-math.log(max(row[target], clip_epsilon)) for target, row in rows]
    return _probability_value(
        values,
        total,
        kind=probability_kind,
        calibration_ref=calibration_ref,
        tolerance=normalization_tolerance,
        epsilon=clip_epsilon,
        clipped=sum(row[target] < clip_epsilon for target, row in rows),
    )
