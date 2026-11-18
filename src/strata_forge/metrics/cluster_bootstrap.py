"""Seeded cluster percentile bootstrap for classification point metrics, single and paired."""

from __future__ import annotations

import math
import statistics
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strata_forge.core.errors import ValidationError
from strata_forge.metrics.classification import (
    ClassificationReport,
    classification_report,
    confusion_matrix,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray

CLUSTER_BOOTSTRAP_VERSION = "cluster-percentile-bootstrap-v1"
CLUSTER_BOOTSTRAP_RNG = "numpy.random.Generator(PCG64)"
CLUSTER_BOOTSTRAP_QUANTILE_RULE = "hyndman-fan-7"
CLUSTER_BOOTSTRAP_CONFIDENCE = 0.95
CLUSTER_BOOTSTRAP_RESAMPLES = 10_000
MAX_CLUSTER_BOOTSTRAP_RESAMPLES = 1_000_000
MAX_CLUSTER_BOOTSTRAP_SEED = 2**53 - 1
_CHUNK_CELLS = 1 << 20

__all__ = [
    "CLUSTER_BOOTSTRAP_CONFIDENCE",
    "CLUSTER_BOOTSTRAP_QUANTILE_RULE",
    "CLUSTER_BOOTSTRAP_RESAMPLES",
    "CLUSTER_BOOTSTRAP_RNG",
    "CLUSTER_BOOTSTRAP_VERSION",
    "MAX_CLUSTER_BOOTSTRAP_RESAMPLES",
    "MAX_CLUSTER_BOOTSTRAP_SEED",
    "ClusterBootstrap",
    "ClusterBootstrapSettings",
    "ClusterMeanBootstrap",
    "MetricInterval",
    "PairedClusterBootstrap",
    "PairedClusterMeanBootstrap",
    "SystemIntervals",
    "cluster_bootstrap",
    "cluster_mean_bootstrap",
    "paired_cluster_bootstrap",
    "paired_cluster_mean_bootstrap",
]

ABSENT_GOLD_CLASS = "absent_gold_class:"

# Per system: accuracy, macro-F1, invalid rate and present-class macro-F1 per resample.
type _Values = tuple[list[float], list[float], list[float], list[float]]
# Per system: the point values (macro-F1 None when a declared class is absent) and absent classes.
type _Point = tuple[float, float | None, float, float | None, tuple[str, ...]]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)


class ClusterBootstrapSettings(_Frozen):
    """The resampling contract one interval block was produced under."""

    bootstrap_version: Literal["cluster-percentile-bootstrap-v1"] = CLUSTER_BOOTSTRAP_VERSION
    rng: Literal["numpy.random.Generator(PCG64)"] = CLUSTER_BOOTSTRAP_RNG
    quantile_rule: Literal["hyndman-fan-7"] = CLUSTER_BOOTSTRAP_QUANTILE_RULE
    confidence_level: float = Field(default=CLUSTER_BOOTSTRAP_CONFIDENCE, allow_inf_nan=False)
    seed: int = Field(ge=0, le=MAX_CLUSTER_BOOTSTRAP_SEED)
    resamples: int = Field(ge=1, le=MAX_CLUSTER_BOOTSTRAP_RESAMPLES)
    n: int = Field(ge=1)
    clusters: int = Field(ge=1)
    numpy_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _fixed_confidence(self) -> ClusterBootstrapSettings:
        if self.confidence_level != CLUSTER_BOOTSTRAP_CONFIDENCE or self.clusters > self.n:
            raise ValueError("cluster_bootstrap_invalid")
        return self


class MetricInterval(_Frozen):
    """A full-sample point value and its percentile bounds over the defined resamples.

    A resample whose metric is undefined (macro-F1 when its gold has no row of a declared
    class) is counted in ``undefined_resamples`` and left out of the quantiles;
    it is never dropped silently or scored as zero. When every resample is undefined the bounds
    are null with a reason. A one-sided 97.5% bound is the matching end of this interval.
    """

    point: float | None = Field(allow_inf_nan=False)
    lower: float | None = Field(allow_inf_nan=False)
    upper: float | None = Field(allow_inf_nan=False)
    undefined_resamples: int = Field(default=0, ge=0)
    # ``all_resamples_undefined``, or ``absent_gold_class:<label>[,<label>…]`` when the population's
    # GOLD has no row of a declared class: the metric then has no value for any system (never one
    # averaged with a zero), whatever the predictions were.
    reason: str | None = Field(default=None, max_length=1024)
    # With an ``absent_gold_class`` reason: those classes, in declared order.
    absent_classes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _ordered(self) -> MetricInterval:
        if self.reason not in (None, "all_resamples_undefined") and self.reason != (
            ABSENT_GOLD_CLASS + ",".join(self.absent_classes)
        ):
            raise ValueError("cluster_bootstrap_invalid")
        if self.point is None:
            if self.lower is not None or self.upper is not None or not self.absent_classes:
                raise ValueError("cluster_bootstrap_invalid")
            return self
        if self.absent_classes:
            raise ValueError("cluster_bootstrap_invalid")
        if self.lower is None or self.upper is None:
            if self.lower is not None or self.upper is not None or self.reason is None:
                raise ValueError("cluster_bootstrap_invalid")
        elif self.lower > self.upper or self.reason is not None:
            raise ValueError("cluster_bootstrap_invalid")
        return self


class SystemIntervals(_Frozen):
    """Accuracy, macro-F1 and invalid rate for one system, or a candidate-minus-baseline delta.

    ``macro_f1_present_classes`` is a DIAGNOSTIC: macro-F1 averaged over ``present_classes`` only
    (the declared classes the population's GOLD has, one set for every system), under the same
    undefined-resample rule. It is never a gate read.
    """

    accuracy: MetricInterval
    macro_f1: MetricInterval
    invalid_rate: MetricInterval
    macro_f1_present_classes: MetricInterval
    present_classes: tuple[str, ...]


class ClusterBootstrap(_Frozen):
    """One system's point metrics with cluster percentile intervals."""

    settings: ClusterBootstrapSettings
    intervals: SystemIntervals


class PairedClusterBootstrap(_Frozen):
    """Two systems on one population; ``delta`` is candidate minus baseline on shared resamples."""

    settings: ClusterBootstrapSettings
    baseline: SystemIntervals
    candidate: SystemIntervals
    delta: SystemIntervals


def _setting(value: object, high: int, code: str, *, low: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValidationError(code)
    return value


def _cluster_codes(clusters: Iterable[object], n: int) -> tuple[list[int], int]:
    """Row -> canonical cluster index; clusters are ordered by their sorted string keys."""
    keys: list[str] = []
    for value in clusters:
        if not isinstance(value, str) or not value:
            raise ValidationError("cluster_bootstrap_cluster_invalid")
        keys.append(value)
    if len(keys) != n:
        raise ValidationError("cluster_bootstrap_population_mismatch")
    index = {key: i for i, key in enumerate(sorted(set(keys)))}
    return [index[key] for key in keys], len(index)


def _cluster_cells(
    gold: list[object],
    predictions: list[object],
    codes: list[int],
    clusters: int,
    *,
    labels: tuple[str, ...],
    index_map: Mapping[int, str] | None,
) -> list[list[int]]:
    """Per-cluster confusion cells (K*K counts, then K invalid), in canonical cluster order."""
    members: list[list[int]] = [[] for _ in range(clusters)]
    for row, code in enumerate(codes):
        members[code].append(row)
    out: list[list[int]] = []
    for rows in members:
        matrix = confusion_matrix(
            [gold[i] for i in rows],
            [predictions[i] for i in rows],
            labels=labels,
            index_map=index_map,
        )
        out.append([value for row in matrix.counts for value in row] + list(matrix.invalid_by_gold))
    return out


def _require_numpy() -> str:
    try:
        import numpy
    except ImportError as exc:
        raise ImportError(
            "cluster bootstrap requires numpy: install 'strata-forge[scoring]'"
        ) from exc
    return str(numpy.__version__)


def _weights(clusters: int, *, seed: int, resamples: int) -> Iterator[NDArray[np.float64]]:
    """Cluster multiplicities per resample, in row blocks of one C-order ``(resamples, C)`` draw.

    Consecutive blocks from one generator reproduce the single draw, so chunking bounds memory
    without changing the stream.
    """
    import numpy as np

    generator = np.random.Generator(np.random.PCG64(seed))
    rows = max(1, _CHUNK_CELLS // clusters)
    for start in range(0, resamples, rows):
        count = min(rows, resamples - start)
        draws = generator.integers(0, clusters, size=(count, clusters), dtype=np.int64)
        offsets = (np.arange(count, dtype=np.int64) * clusters)[:, None]
        counts = np.bincount((draws + offsets).ravel(), minlength=count * clusters)
        yield counts.reshape(count, clusters).astype(np.float64)


def _metrics(
    counts: NDArray[np.float64], size: int, present: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Accuracy, macro-F1 and invalid rate per resample, from ``(R, K*K + K)`` cell counts.

    The definitions of ``classification_report`` (invalid outputs stay in gold support), except
    that a resample whose GOLD has no row of a declared class has no macro-F1 (whatever the
    predictions): it is NaN here and counted as undefined, never scored as zero. The
    present-class macro-F1 averages the ``present`` classes and is undefined when the resample's
    gold lacks one of them.
    """
    import numpy as np

    matrix = counts[:, : size * size].reshape(-1, size, size)
    invalid = counts[:, size * size :]
    n = matrix.sum(axis=(1, 2)) + invalid.sum(axis=1)
    tp = np.diagonal(matrix, axis1=1, axis2=2)
    denominator = matrix.sum(axis=2) + invalid + matrix.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    in_gold = (matrix.sum(axis=2) + invalid) > 0
    macro = np.where(in_gold.all(axis=1), f1.sum(axis=1) / size, np.nan)
    subset = f1[:, present]
    macro_present = np.where(
        in_gold[:, present].all(axis=1), subset.sum(axis=1) / max(int(present.sum()), 1), np.nan
    )
    return tp.sum(axis=1) / n, macro, invalid.sum(axis=1) / n, macro_present


def _resampled(
    per_system: list[list[list[int]]],
    size: int,
    present: list[bool],
    *,
    seed: int,
    resamples: int,
) -> list[_Values]:
    """Per system, the resampled accuracy, macro-F1, invalid-rate and present-class macro-F1
    values, unsorted."""
    import numpy as np

    mask = np.asarray(present, dtype=np.bool_)
    tables = [np.asarray(cells, dtype=np.float64) for cells in per_system]
    out: list[_Values] = [([], [], [], []) for _ in tables]
    for weights in _weights(len(per_system[0]), seed=seed, resamples=resamples):
        for table, values in zip(tables, out, strict=True):
            # Integer counts summed in float64 stay exact below 2**53 in any summation order.
            counts = np.rint(weights @ table)
            for target, column in zip(values, _metrics(counts, size, mask), strict=True):
                target.extend(float(value) for value in column.tolist())
    return out


def _percentile(values: Sequence[float], probability: Fraction) -> float:
    """Hyndman-Fan type 7 at an exact rational position over sorted values."""
    position = (len(values) - 1) * probability
    index = math.floor(position)
    low = values[index]
    if index + 1 == len(values):
        return low
    high = values[index + 1]
    weight = float(position - index)
    return min(max(low + weight * (high - low), low), high)


def _interval(
    point: float | None, values: list[float], absent: tuple[str, ...] = ()
) -> MetricInterval:
    if point is None:
        # Null by rule, not unstable: a class the population's gold lacks is in no resample of it.
        return MetricInterval(
            point=None,
            lower=None,
            upper=None,
            undefined_resamples=len(values),
            reason=ABSENT_GOLD_CLASS + ",".join(absent),
            absent_classes=absent,
        )
    ordered = sorted(value for value in values if not math.isnan(value))
    undefined = len(values) - len(ordered)
    if not ordered:
        return MetricInterval(
            point=point,
            lower=None,
            upper=None,
            undefined_resamples=undefined,
            reason="all_resamples_undefined",
        )
    tail = (1 - Fraction(str(CLUSTER_BOOTSTRAP_CONFIDENCE))) / 2
    return MetricInterval(
        point=point,
        lower=_percentile(ordered, tail),
        upper=_percentile(ordered, 1 - tail),
        undefined_resamples=undefined,
    )


def _points(report: ClassificationReport, present: tuple[str, ...]) -> _Point:
    """Point values; ``present`` are the classes the GOLD has, the same for every system."""
    values = (report.accuracy.value, report.invalid_rate.value)
    if any(value is None for value in values):  # unreachable: the population is nonempty
        raise ValidationError("cluster_bootstrap_empty")
    scores = {score.label: score for score in report.per_class}
    absent = tuple(label for label in scores if label not in present)
    macro = None if absent else float(report.macro_f1.value or 0)
    macro_present = math.fsum(scores[label].f1 for label in present) / len(present)
    return (float(values[0] or 0), macro, float(values[1] or 0), macro_present, absent)


def _system(point: _Point, values: _Values, present: tuple[str, ...]) -> SystemIntervals:
    accuracy, macro, invalid, macro_present, absent = point
    return SystemIntervals(
        accuracy=_interval(accuracy, values[0]),
        macro_f1=_interval(macro, values[1], absent),
        invalid_rate=_interval(invalid, values[2]),
        macro_f1_present_classes=_interval(macro_present, values[3]),
        present_classes=present,
    )


def _difference(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _run(
    gold: Iterable[object],
    systems: Sequence[Iterable[object]],
    *,
    labels: Sequence[str],
    clusters: Iterable[object],
    index_map: Mapping[int, str] | None,
    seed: object,
    resamples: object,
) -> tuple[ClusterBootstrapSettings, list[_Point], list[_Values], tuple[str, ...]]:
    chosen_seed = _setting(
        seed, MAX_CLUSTER_BOOTSTRAP_SEED, "cluster_bootstrap_seed_invalid", low=0
    )
    count = _setting(
        resamples, MAX_CLUSTER_BOOTSTRAP_RESAMPLES, "cluster_bootstrap_resamples_invalid", low=1
    )
    actual = list(gold)
    if not actual:
        raise ValidationError("cluster_bootstrap_empty")
    rows = [list(system) for system in systems]
    if any(len(system) != len(actual) for system in rows):
        raise ValidationError("cluster_bootstrap_population_mismatch")
    codes, cluster_count = _cluster_codes(clusters, len(actual))
    reports = [
        classification_report(actual, system, labels=labels, index_map=index_map) for system in rows
    ]
    names = reports[0].confusion.labels
    numpy_version = _require_numpy()
    per_system = [
        _cluster_cells(actual, system, codes, cluster_count, labels=names, index_map=index_map)
        for system in rows
    ]
    # The classes the population's GOLD has: one set for every system, whatever it predicted.
    seen = {score.label for score in reports[0].per_class if score.support > 0}
    present = tuple(label for label in names if label in seen)
    values = _resampled(
        per_system,
        len(names),
        [label in seen for label in names],
        seed=chosen_seed,
        resamples=count,
    )
    settings = ClusterBootstrapSettings(
        seed=chosen_seed,
        resamples=count,
        n=len(actual),
        clusters=cluster_count,
        numpy_version=numpy_version,
    )
    return settings, [_points(report, present) for report in reports], values, present


def cluster_bootstrap(
    gold: Iterable[object],
    predictions: Iterable[object],
    *,
    labels: Sequence[str],
    clusters: Iterable[object],
    seed: int,
    resamples: int = CLUSTER_BOOTSTRAP_RESAMPLES,
    index_map: Mapping[int, str] | None = None,
) -> ClusterBootstrap:
    """Resample whole clusters with replacement and bound accuracy, macro-F1 and invalid rate.

    ``clusters`` gives each row's nonempty string cluster key (for example a base problem id).
    Invalid or missing predictions follow ``classification_report``: wrong, and in gold support.
    """
    settings, points, values, present = _run(
        gold,
        [predictions],
        labels=labels,
        clusters=clusters,
        index_map=index_map,
        seed=seed,
        resamples=resamples,
    )
    return ClusterBootstrap(settings=settings, intervals=_system(points[0], values[0], present))


def paired_cluster_bootstrap(
    gold: Iterable[object],
    baseline: Iterable[object],
    candidate: Iterable[object],
    *,
    labels: Sequence[str],
    clusters: Iterable[object],
    seed: int,
    resamples: int = CLUSTER_BOOTSTRAP_RESAMPLES,
    index_map: Mapping[int, str] | None = None,
) -> PairedClusterBootstrap:
    """Bootstrap two systems on the same rows and clusters; every resample scores both.

    ``delta`` is candidate minus baseline, differenced per resample before the quantiles are
    taken, so its interval reflects the pairing rather than two independent intervals.
    """
    settings, points, values, present = _run(
        gold,
        [baseline, candidate],
        labels=labels,
        clusters=clusters,
        index_map=index_map,
        seed=seed,
        resamples=resamples,
    )
    (base_point, cand_point), (base_values, cand_values) = points, values
    # Absence is decided by the shared gold, so both sides and the delta agree on it.
    absent = base_point[4]
    delta_point: _Point = (
        cand_point[0] - base_point[0],
        _difference(cand_point[1], base_point[1]),
        cand_point[2] - base_point[2],
        _difference(cand_point[3], base_point[3]),
        absent,
    )
    accuracy, macro, invalid, macro_present = (
        [c - b for b, c in zip(bv, cv, strict=True)]
        for bv, cv in zip(base_values, cand_values, strict=True)
    )
    delta_values: _Values = (accuracy, macro, invalid, macro_present)
    return PairedClusterBootstrap(
        settings=settings,
        baseline=_system(base_point, base_values, present),
        candidate=_system(cand_point, cand_values, present),
        delta=_system(delta_point, delta_values, present),
    )


class ClusterMeanBootstrap(_Frozen):
    """The mean of a per-row score (for example a Brier term) with a cluster percentile interval."""

    settings: ClusterBootstrapSettings
    interval: MetricInterval


class PairedClusterMeanBootstrap(_Frozen):
    """Two systems' per-row scores on the same rows; ``delta`` is candidate minus baseline."""

    settings: ClusterBootstrapSettings
    baseline: MetricInterval
    candidate: MetricInterval
    delta: MetricInterval
    # The sample SD (n - 1) of the per-row differences, with its interval over the same resamples.
    delta_sd: MetricInterval


def _row_values(values: Iterable[object], what: str) -> list[float]:
    out: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValidationError(f"cluster_bootstrap_{what}_invalid")
        out.append(float(value))
    return out


def _resampled_means(
    systems: list[list[float]], codes: list[int], clusters: int, *, seed: int, resamples: int
) -> list[list[float]]:
    """Per system, the resampled row-weighted means: cluster sums over cluster sizes."""
    import numpy as np

    index = np.asarray(codes, dtype=np.int64)
    sizes = np.bincount(index, minlength=clusters).astype(np.float64)
    sums = [
        np.bincount(index, weights=np.asarray(values, dtype=np.float64), minlength=clusters)
        for values in systems
    ]
    out: list[list[float]] = [[] for _ in systems]
    for weights in _weights(clusters, seed=seed, resamples=resamples):
        rows = weights @ sizes
        for total, target in zip(sums, out, strict=True):
            target.extend(float(v) for v in ((weights @ total) / rows).tolist())
    return out


def _mean_run(
    systems: list[Iterable[object]], clusters: Iterable[object], *, seed: object, resamples: object
) -> tuple[ClusterBootstrapSettings, list[float], list[list[float]]]:
    chosen_seed = _setting(
        seed, MAX_CLUSTER_BOOTSTRAP_SEED, "cluster_bootstrap_seed_invalid", low=0
    )
    count = _setting(
        resamples, MAX_CLUSTER_BOOTSTRAP_RESAMPLES, "cluster_bootstrap_resamples_invalid", low=1
    )
    rows = [_row_values(values, "value") for values in systems]
    if not rows[0]:
        raise ValidationError("cluster_bootstrap_empty")
    if any(len(values) != len(rows[0]) for values in rows):
        raise ValidationError("cluster_bootstrap_population_mismatch")
    codes, cluster_count = _cluster_codes(clusters, len(rows[0]))
    numpy_version = _require_numpy()
    resampled = _resampled_means(rows, codes, cluster_count, seed=chosen_seed, resamples=count)
    settings = ClusterBootstrapSettings(
        seed=chosen_seed,
        resamples=count,
        n=len(rows[0]),
        clusters=cluster_count,
        numpy_version=numpy_version,
    )
    return settings, [math.fsum(values) / len(values) for values in rows], resampled


def cluster_mean_bootstrap(
    values: Iterable[object],
    *,
    clusters: Iterable[object],
    seed: int,
    resamples: int = CLUSTER_BOOTSTRAP_RESAMPLES,
) -> ClusterMeanBootstrap:
    """Bound the mean of finite per-row scores by resampling whole clusters (same contract)."""
    settings, points, resampled = _mean_run([values], clusters, seed=seed, resamples=resamples)
    return ClusterMeanBootstrap(settings=settings, interval=_interval(points[0], resampled[0]))


def paired_cluster_mean_bootstrap(
    baseline: Iterable[object],
    candidate: Iterable[object],
    *,
    clusters: Iterable[object],
    seed: int,
    resamples: int = CLUSTER_BOOTSTRAP_RESAMPLES,
) -> PairedClusterMeanBootstrap:
    """Two systems' per-row scores on the same rows, differenced per resample (candidate minus
    baseline) before the quantiles, and the SD of the per-row differences. Needs two rows."""
    baseline_rows, candidate_rows, cluster_rows = list(baseline), list(candidate), list(clusters)
    if len(baseline_rows) < 2:
        raise ValidationError("cluster_bootstrap_too_few_rows")
    settings, points, resampled = _mean_run(
        [baseline_rows, candidate_rows], cluster_rows, seed=seed, resamples=resamples
    )
    delta = [c - b for b, c in zip(resampled[0], resampled[1], strict=True)]
    diffs = [
        c - b
        for b, c in zip(
            _row_values(baseline_rows, "value"), _row_values(candidate_rows, "value"), strict=True
        )
    ]
    return PairedClusterMeanBootstrap(
        settings=settings,
        baseline=_interval(points[0], resampled[0]),
        candidate=_interval(points[1], resampled[1]),
        delta=_interval(points[1] - points[0], delta),
        delta_sd=_interval(
            statistics.stdev(diffs),
            _resampled_sd(diffs, cluster_rows, seed=settings.seed, resamples=settings.resamples),
        ),
    )


def _resampled_sd(
    values: list[float], clusters: list[object], *, seed: int, resamples: int
) -> list[float]:
    """The sample SD of ``values`` in every resample, from weighted sums of x and x squared."""
    import numpy as np

    codes, count = _cluster_codes(clusters, len(values))
    index = np.asarray(codes, dtype=np.int64)
    x = np.asarray(values, dtype=np.float64)
    sizes = np.bincount(index, minlength=count).astype(np.float64)
    first = np.bincount(index, weights=x, minlength=count)
    second = np.bincount(index, weights=x * x, minlength=count)
    out: list[float] = []
    for weights in _weights(count, seed=seed, resamples=resamples):
        n = weights @ sizes
        s1 = weights @ first
        s2 = weights @ second
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.where(n > 1, (s2 - s1 * s1 / n) / (n - 1), np.nan)
        out.extend(float(v) for v in np.sqrt(np.maximum(var, 0.0)).tolist())
    return out
