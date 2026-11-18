"""Seeded cluster percentile bootstrap for classification point metrics, single and paired."""

from __future__ import annotations

import math
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
    "MetricInterval",
    "PairedClusterBootstrap",
    "SystemIntervals",
    "cluster_bootstrap",
    "paired_cluster_bootstrap",
]

type _Values = tuple[list[float], list[float], list[float]]


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
    """A full-sample point value and its percentile bounds over the resamples."""

    point: float = Field(allow_inf_nan=False)
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _ordered(self) -> MetricInterval:
        if self.lower > self.upper:
            raise ValueError("cluster_bootstrap_invalid")
        return self


class SystemIntervals(_Frozen):
    """Accuracy, macro-F1 and invalid rate for one system, or a candidate-minus-baseline delta."""

    accuracy: MetricInterval
    macro_f1: MetricInterval
    invalid_rate: MetricInterval


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
    counts: NDArray[np.float64], size: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Accuracy, macro-F1 and invalid rate per resample, from ``(R, K*K + K)`` cell counts.

    The definitions of ``classification_report``: invalid outputs stay in gold support, and a
    class with no support and no predictions contributes F1 zero.
    """
    import numpy as np

    matrix = counts[:, : size * size].reshape(-1, size, size)
    invalid = counts[:, size * size :]
    n = matrix.sum(axis=(1, 2)) + invalid.sum(axis=1)
    tp = np.diagonal(matrix, axis1=1, axis2=2)
    denominator = matrix.sum(axis=2) + invalid + matrix.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return tp.sum(axis=1) / n, f1.sum(axis=1) / size, invalid.sum(axis=1) / n


def _resampled(
    per_system: list[list[list[int]]], size: int, *, seed: int, resamples: int
) -> list[_Values]:
    """Per system, the resampled accuracy, macro-F1 and invalid-rate values, unsorted."""
    import numpy as np

    tables = [np.asarray(cells, dtype=np.float64) for cells in per_system]
    out: list[_Values] = [([], [], []) for _ in tables]
    for weights in _weights(len(per_system[0]), seed=seed, resamples=resamples):
        for table, values in zip(tables, out, strict=True):
            # Integer counts summed in float64 stay exact below 2**53 in any summation order.
            counts = np.rint(weights @ table)
            for target, column in zip(values, _metrics(counts, size), strict=True):
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


def _interval(point: float, values: list[float]) -> MetricInterval:
    ordered = sorted(values)
    tail = (1 - Fraction(str(CLUSTER_BOOTSTRAP_CONFIDENCE))) / 2
    return MetricInterval(
        point=point, lower=_percentile(ordered, tail), upper=_percentile(ordered, 1 - tail)
    )


def _points(report: ClassificationReport) -> tuple[float, float, float]:
    values = (report.accuracy.value, report.macro_f1.value, report.invalid_rate.value)
    if any(value is None for value in values):  # unreachable: the population is nonempty
        raise ValidationError("cluster_bootstrap_empty")
    return (float(values[0] or 0), float(values[1] or 0), float(values[2] or 0))


def _system(point: tuple[float, float, float], values: _Values) -> SystemIntervals:
    return SystemIntervals(
        accuracy=_interval(point[0], values[0]),
        macro_f1=_interval(point[1], values[1]),
        invalid_rate=_interval(point[2], values[2]),
    )


def _run(
    gold: Iterable[object],
    systems: Sequence[Iterable[object]],
    *,
    labels: Sequence[str],
    clusters: Iterable[object],
    index_map: Mapping[int, str] | None,
    seed: object,
    resamples: object,
) -> tuple[ClusterBootstrapSettings, list[tuple[float, float, float]], list[_Values]]:
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
    values = _resampled(per_system, len(names), seed=chosen_seed, resamples=count)
    settings = ClusterBootstrapSettings(
        seed=chosen_seed,
        resamples=count,
        n=len(actual),
        clusters=cluster_count,
        numpy_version=numpy_version,
    )
    return settings, [_points(report) for report in reports], values


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
    settings, points, values = _run(
        gold,
        [predictions],
        labels=labels,
        clusters=clusters,
        index_map=index_map,
        seed=seed,
        resamples=resamples,
    )
    return ClusterBootstrap(settings=settings, intervals=_system(points[0], values[0]))


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
    settings, points, values = _run(
        gold,
        [baseline, candidate],
        labels=labels,
        clusters=clusters,
        index_map=index_map,
        seed=seed,
        resamples=resamples,
    )
    (base_point, cand_point), (base_values, cand_values) = points, values
    delta_point = (
        cand_point[0] - base_point[0],
        cand_point[1] - base_point[1],
        cand_point[2] - base_point[2],
    )
    accuracy, macro, invalid = (
        [c - b for b, c in zip(bv, cv, strict=True)]
        for bv, cv in zip(base_values, cand_values, strict=True)
    )
    delta_values: _Values = (accuracy, macro, invalid)
    return PairedClusterBootstrap(
        settings=settings,
        baseline=_system(base_point, base_values),
        candidate=_system(cand_point, cand_values),
        delta=_system(delta_point, delta_values),
    )
