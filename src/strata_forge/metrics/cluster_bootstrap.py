"""Bounded paired row-pooled cluster percentiles; supplied counts confer no authority."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.core.errors import ValidationError
from strata_forge.metrics.classification import (
    ConfusionMatrix,
    LabelSpace,
    MetricValue,
    brier,
    classification_report_from_confusion,
    confusion_matrix,
    nll,
)

MAX_ROWS = 100_000
MAX_CLUSTERS = 10_000
MAX_WORK = 100_000_000
MAX_ARRAY_BYTES = 256 * 1024**2
RESAMPLES = 10_000
SEED = 20261104

type MetricName = Literal["accuracy", "macro_f1", "invalid_rate", "brier", "nll"]
type Family = Literal["class_probabilities", "restricted_softmax"]


def _planned_bytes(m: int, c: int, k: int, q: int) -> int:
    return 8 * (
        m * (5 + 2 * k)
        + 2 * c * (k * k + k)
        + c * (3 + 4 * q)
        + 2 * k * (k + 1)
        + 3 * RESAMPLES * q * 2
        + q * 1024
    )


class ClusterBootstrapError(ValidationError):
    """A content-free mathematical input/computation refusal."""

    def __init__(self) -> None:
        super().__init__("cluster_bootstrap_invalid")


class _Value(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    @field_validator(
        "schema_version",
        "seed",
        "resamples",
        "resamples_completed",
        "lower_rank",
        "upper_rank",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("cluster_bootstrap_invalid")
        return value

    def canonical(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")


def _text(value: str, limit: int = 1024) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > limit
        or len(value.encode("utf-8")) > limit
    ):
        raise ValueError("cluster_bootstrap_invalid")
    return value


class PairedClusterObservation(_Value):
    item_id: str
    base_problem_id: str
    gold_index: int
    baseline_pred_index: int | None
    candidate_pred_index: int | None
    baseline_probabilities: tuple[float | int, ...] | None
    candidate_probabilities: tuple[float | int, ...] | None

    _ids = field_validator("item_id", "base_problem_id")(_text)


class PairedClusterProfile(_Value):
    schema_version: Literal[1]
    labels: tuple[str, ...] = Field(min_length=2, max_length=16)
    metrics: tuple[MetricName, ...] = Field(min_length=1, max_length=5)
    probability_kind: Family | None
    resamples: Literal[10000]
    seed: Literal[20261104]
    confidence_level: Literal["0.95"]
    weighting: Literal["row_pooled"]
    interval: Literal["percentile_nearest_rank_v1"]
    rng_profile: Literal["numpy_generator_pcg64_int64_v1"]

    @model_validator(mode="after")
    def _valid(self) -> PairedClusterProfile:
        LabelSpace(labels=self.labels)
        for label in self.labels:
            _text(label, 64)
        if len(set(self.metrics)) != len(self.metrics) or bool(
            set(self.metrics) & {"brier", "nll"}
        ) != (self.probability_kind is not None):
            raise ValueError("cluster_bootstrap_invalid")
        return self


class PercentileInterval(_Value):
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)
    lower_rank: Literal[250]
    upper_rank: Literal[9750]
    resamples: Literal[10000]

    @model_validator(mode="after")
    def _order(self) -> PercentileInterval:
        if self.lower > self.upper:
            raise ValueError("cluster_bootstrap_invalid")
        return self


class BootstrapMetric(_Value):
    metric: MetricName
    baseline: MetricValue
    candidate: MetricValue
    delta: float | None = Field(allow_inf_nan=False)
    baseline_interval: PercentileInterval | None
    candidate_interval: PercentileInterval | None
    delta_interval: PercentileInterval | None
    interval_reason: Literal["no_metric_population", "insufficient_clusters"] | None


class ClusterBootstrapResult(_Value):
    schema_version: Literal[1]
    method: Literal["paired_cluster_percentile_v1"]
    status: Literal["complete", "undefined"]
    profile: PairedClusterProfile
    metrics: tuple[BootstrapMetric, ...]
    paired_rows: int = Field(ge=0, le=MAX_ROWS)
    clusters: int = Field(ge=0, le=MAX_CLUSTERS)
    minimum_cluster_rows: int | None = Field(ge=1)
    maximum_cluster_rows: int | None = Field(ge=1)
    membership_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    resamples_completed: Literal[0, 10000]
    minimum_resample_rows: int | None = Field(ge=1)
    maximum_resample_rows: int | None = Field(ge=1)
    draws_sha256: str | None = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    replicates_sha256: str | None = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    planned_draw_work_units: int = Field(ge=0, le=MAX_WORK)
    draw_work_units: int = Field(ge=0, le=MAX_WORK)
    planned_numeric_bytes: int = Field(ge=0, le=MAX_ARRAY_BYTES)
    numpy_version: str | None = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def _coherent(self) -> ClusterBootstrapResult:
        complete = self.clusters >= 2
        if (
            (self.clusters == 0) != (self.paired_rows == 0)
            or self.clusters > self.paired_rows
            or tuple(item.metric for item in self.metrics) != self.profile.metrics
        ):
            raise ValueError("cluster_bootstrap_invalid")
        if (
            self.status != ("complete" if complete else "undefined")
            or self.resamples_completed != (RESAMPLES if complete else 0)
            or self.planned_draw_work_units != (RESAMPLES * self.clusters if complete else 0)
            or self.draw_work_units != self.resamples_completed * self.clusters
        ):
            raise ValueError("cluster_bootstrap_invalid")
        if self.planned_numeric_bytes != _planned_bytes(
            self.paired_rows, self.clusters, len(self.profile.labels), len(self.profile.metrics)
        ):
            raise ValueError("cluster_bootstrap_invalid")
        for value in (self.minimum_cluster_rows, self.maximum_cluster_rows):
            if (value is None) != (not self.paired_rows):
                raise ValueError("cluster_bootstrap_invalid")
        if self.paired_rows and (
            cast("int", self.maximum_cluster_rows) > self.paired_rows
            or cast("int", self.minimum_cluster_rows) > cast("int", self.maximum_cluster_rows)
            or cast("int", self.minimum_cluster_rows) * self.clusters > self.paired_rows
            or cast("int", self.maximum_cluster_rows) * self.clusters < self.paired_rows
        ):
            raise ValueError("cluster_bootstrap_invalid")
        for value in (
            self.draws_sha256,
            self.replicates_sha256,
            self.minimum_resample_rows,
            self.maximum_resample_rows,
            self.numpy_version,
        ):
            if (value is not None) != complete:
                raise ValueError("cluster_bootstrap_invalid")
        if (
            complete
            and not cast("int", self.minimum_cluster_rows) * self.clusters
            <= cast("int", self.minimum_resample_rows)
            <= cast("int", self.maximum_resample_rows)
            <= cast("int", self.maximum_cluster_rows) * self.clusters
        ):
            raise ValueError("cluster_bootstrap_invalid")
        reason = (
            None
            if complete
            else "no_metric_population"
            if not self.paired_rows
            else "insufficient_clusters"
        )
        for item in self.metrics:
            expected_delta = (
                None
                if not self.paired_rows
                else cast("float", item.candidate.value) - cast("float", item.baseline.value)
            )
            if (
                item.baseline.n != self.paired_rows
                or item.candidate.n != self.paired_rows
                or item.baseline.excluded
                or item.candidate.excluded
                or item.delta != expected_delta
                or item.interval_reason != reason
            ):
                raise ValueError("cluster_bootstrap_invalid")
            upper = (
                2.000004
                if item.metric == "brier"
                else -math.log(1e-15)
                if item.metric == "nll"
                else 1.0
            )
            for point in (item.baseline, item.candidate):
                if point.value is not None and not 0 <= point.value <= upper:
                    raise ValueError("cluster_bootstrap_invalid")
                if point.calibration_ref is not None:
                    raise ValueError("cluster_bootstrap_invalid")
            if item.delta is not None and not -upper <= item.delta <= upper:
                raise ValueError("cluster_bootstrap_invalid")
            for interval in (item.baseline_interval, item.candidate_interval):
                if interval is not None and not 0 <= interval.lower <= interval.upper <= upper:
                    raise ValueError("cluster_bootstrap_invalid")
            if (
                item.delta_interval is not None
                and not -upper <= item.delta_interval.lower <= item.delta_interval.upper <= upper
            ):
                raise ValueError("cluster_bootstrap_invalid")
            for interval in (item.baseline_interval, item.candidate_interval, item.delta_interval):
                if (interval is not None) != complete:
                    raise ValueError("cluster_bootstrap_invalid")
            for metric in (item.baseline, item.candidate):
                if item.metric in {"brier", "nll"}:
                    if (
                        metric.probability_kind != self.profile.probability_kind
                        or metric.normalization_tolerance != 1e-6
                        or metric.clip_epsilon != (1e-15 if item.metric == "nll" else None)
                    ):
                        raise ValueError("cluster_bootstrap_invalid")
                elif metric.probability_kind is not None:
                    raise ValueError("cluster_bootstrap_invalid")
        return self


_MODELS = (
    PairedClusterObservation,
    PairedClusterProfile,
    PercentileInterval,
    BootstrapMetric,
    ClusterBootstrapResult,
    MetricValue,
)


def _copy[T: BaseModel](value: T) -> T:
    def native(item: object, depth: int = 0) -> object:
        if depth > 16:
            raise ClusterBootstrapError()
        if any(type(item) is model_type for model_type in _MODELS):
            model = cast("BaseModel", item)
            storage: object = object.__getattribute__(model, "__dict__")
            if type(storage) is not dict or model.__pydantic_extra__ is not None:
                raise ClusterBootstrapError()
            data = cast("dict[object, object]", storage).copy()
            fields = type(model).model_fields
            if any(type(key) is not str for key in data) or set(data) != set(fields):
                raise ClusterBootstrapError()
            return {key: native(data[key], depth + 1) for key in fields}
        if type(item) is tuple:
            if len(cast("tuple[object, ...]", item)) > MAX_ROWS:
                raise ClusterBootstrapError()
            return tuple(native(child, depth + 1) for child in cast("tuple[object, ...]", item))
        if item is None or any(
            type(item) is native_type for native_type in (str, int, float, bool)
        ):
            return item
        raise ClusterBootstrapError()

    if not any(type(value) is model_type for model_type in _MODELS):
        raise ClusterBootstrapError()
    return type(value).model_validate(native(value))


def _validate(
    rows: tuple[PairedClusterObservation, ...], profile: PairedClusterProfile
) -> tuple[tuple[PairedClusterObservation, ...], PairedClusterProfile]:
    if type(profile) is not PairedClusterProfile or type(rows) is not tuple or len(rows) > MAX_ROWS:
        raise ClusterBootstrapError()
    # Literal equality must not admit bool or scalar subclasses via preconstructed models.
    for name, expected in (("schema_version", 1), ("resamples", RESAMPLES), ("seed", SEED)):
        if type(getattr(profile, name)) is not int or getattr(profile, name) != expected:
            raise ClusterBootstrapError()
    profile = _copy(profile)
    copied: list[PairedClusterObservation] = []
    ids: set[str] = set()
    for original in rows:
        if type(original) is not PairedClusterObservation:
            raise ClusterBootstrapError()
        row = _copy(original)
        for index in (row.gold_index, row.baseline_pred_index, row.candidate_pred_index):
            if index is not None and (
                type(index) is not int or not 0 <= index < len(profile.labels)
            ):
                raise ClusterBootstrapError()
        if row.item_id in ids:
            raise ClusterBootstrapError()
        ids.add(row.item_id)
        for values in (row.baseline_probabilities, row.candidate_probabilities):
            if profile.probability_kind is None:
                if values is not None:
                    raise ClusterBootstrapError()
            elif (
                values is None
                or len(values) != len(profile.labels)
                or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in values)
                or not math.isclose(math.fsum(values), 1, rel_tol=0, abs_tol=1e-6)
            ):
                raise ClusterBootstrapError()
        copied.append(row)
    return tuple(sorted(copied, key=lambda row: row.item_id.encode("utf-8"))), profile


class _ClusterBootstrapState:
    """Owned incremental numerical state; no source, I/O or serialized callback."""

    def __init__(
        self, rows: tuple[PairedClusterObservation, ...], profile: PairedClusterProfile
    ) -> None:
        rows, self.profile = _validate(rows, profile)
        self.n = len(rows)
        names = sorted({row.base_problem_id for row in rows}, key=lambda key: key.encode("utf-8"))
        self.c = len(names)
        if self.c > MAX_CLUSTERS or (self.c >= 2 and RESAMPLES * self.c > MAX_WORK):
            raise ClusterBootstrapError()
        k, q, m = len(profile.labels), len(profile.metrics), len(rows)
        self.planned = _planned_bytes(m, self.c, k, q)
        if self.planned > MAX_ARRAY_BYTES:
            raise ClusterBootstrapError()
        membership = {
            "labels": profile.labels,
            "members": [(r.item_id, profile.labels[r.gold_index], r.base_problem_id) for r in rows],
        }
        self.membership = hashlib.sha256(
            b"strata.analysis.cluster.members.v1\0"
            + json.dumps(
                membership, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
        grouped: dict[str, list[PairedClusterObservation]] = {name: [] for name in names}
        for row in rows:
            grouped[row.base_problem_id].append(row)
        self.sizes = tuple(len(grouped[name]) for name in names)
        self.matrices: list[tuple[ConfusionMatrix, ...]] = []
        self.losses: list[dict[str, tuple[float, ...]]] = []
        self.clipped: list[tuple[int, ...]] = []
        for arm in ("baseline", "candidate"):
            matrices: list[ConfusionMatrix] = []
            losses: dict[str, list[float]] = {
                name: [] for name in profile.metrics if name in {"brier", "nll"}
            }
            clipped: list[int] = []
            for name in names:
                members = grouped[name]
                gold = [profile.labels[row.gold_index] for row in members]
                predictions = [
                    row.baseline_pred_index if arm == "baseline" else row.candidate_pred_index
                    for row in members
                ]
                matrices.append(
                    confusion_matrix(
                        gold,
                        [None if pred is None else profile.labels[pred] for pred in predictions],
                        labels=profile.labels,
                    )
                )
                probabilities = [
                    row.baseline_probabilities if arm == "baseline" else row.candidate_probabilities
                    for row in members
                ]
                for metric in losses:
                    function = brier if metric == "brier" else nll
                    result = function(
                        gold,
                        probabilities,
                        labels=profile.labels,
                        probability_kind=cast("Family", profile.probability_kind),
                    )
                    if result.value is None or result.n != len(members) or result.excluded:
                        raise ClusterBootstrapError()
                    losses[metric].append(result.value * result.n)
                    if metric == "nll":
                        clipped.append(result.clipped)
            self.matrices.append(tuple(matrices))
            self.losses.append({key: tuple(values) for key, values in losses.items()})
            self.clipped.append(tuple(clipped))
        self.completed = 0
        self.low_n: int | None = None
        self.high_n: int | None = None
        self.version: str | None = None
        self.draw_hash = hashlib.sha256(
            b"strata.analysis.cluster.draws.v1\0" + struct.pack("<QQ", RESAMPLES, self.c)
        )
        header = {
            "metrics": profile.metrics,
            "resamples": RESAMPLES,
            "series_order": ["baseline", "candidate", "delta"],
        }
        self.value_hash = hashlib.sha256(
            b"strata.analysis.cluster.replicates.v1\0"
            + json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
        )
        self.values: object = None
        self.rng: object = None
        self.count_arrays: object = None
        if self.c >= 2:
            import numpy as np

            self.version = np.__version__
            self.count_arrays = np.array(
                [
                    [
                        [(*row, matrix.invalid_by_gold[i]) for i, row in enumerate(matrix.counts)]
                        for matrix in matrices
                    ]
                    for matrices in self.matrices
                ],
                dtype=np.int64,
            )
            self.rng = np.random.Generator(np.random.PCG64(SEED))
            self.values = np.empty((RESAMPLES, q, 3), dtype=np.float64)
        self.points = self._metrics((1,) * self.c)

    def _metrics(
        self, weights: tuple[int, ...]
    ) -> tuple[tuple[MetricValue, ...], tuple[MetricValue, ...]]:
        total = sum(weight * size for weight, size in zip(weights, self.sizes, strict=True))
        output: list[tuple[MetricValue, ...]] = []
        k = len(self.profile.labels)
        for arm in range(2):
            matrices = self.matrices[arm]
            if self.count_arrays is None:
                counts = tuple(
                    tuple(
                        sum(w * x.counts[i][j] for w, x in zip(weights, matrices, strict=True))
                        for j in range(k)
                    )
                    for i in range(k)
                )
                invalid = tuple(
                    sum(w * x.invalid_by_gold[i] for w, x in zip(weights, matrices, strict=True))
                    for i in range(k)
                )
            else:
                import numpy as np

                arrays = cast(
                    "np.ndarray[tuple[int,int,int,int],np.dtype[np.int64]]", self.count_arrays
                )
                weighted = cast(
                    "np.ndarray[tuple[int,int],np.dtype[np.int64]]",
                    np.tensordot(np.asarray(weights, dtype=np.int64), arrays[arm], axes=(0, 0)),
                )
                counts = tuple(tuple(int(weighted[i, j]) for j in range(k)) for i in range(k))
                invalid = tuple(int(weighted[i, k]) for i in range(k))
            matrix = ConfusionMatrix(
                labels=self.profile.labels, counts=counts, invalid_by_gold=invalid, n=total
            )
            report = classification_report_from_confusion(matrix)
            values: list[MetricValue] = []
            for name in self.profile.metrics:
                if name not in {"brier", "nll"}:
                    values.append(cast("MetricValue", getattr(report, name)))
                else:
                    numerator = math.fsum(
                        w * x for w, x in zip(weights, self.losses[arm][name], strict=True)
                    )
                    clipped = (
                        sum(w * x for w, x in zip(weights, self.clipped[arm], strict=True))
                        if name == "nll"
                        else 0
                    )
                    values.append(
                        MetricValue(
                            value=numerator / total if total else None,
                            n=total,
                            reason=None if total else "empty",
                            probability_kind=self.profile.probability_kind,
                            clip_epsilon=1e-15 if name == "nll" else None,
                            clipped=clipped,
                            normalization_tolerance=1e-6,
                        )
                    )
            output.append(tuple(values))
        return output[0], output[1]

    @property
    def done(self) -> bool:
        return self.c < 2 or self.completed == RESAMPLES

    def advance(self) -> None:
        if self.done:
            return
        import numpy as np
        from numpy.random import Generator

        values = cast("np.ndarray[tuple[int, int, int], np.dtype[np.float64]]", self.values)
        if type(self.rng) is not Generator:
            raise ClusterBootstrapError()
        for _ in range(min(100, RESAMPLES - self.completed)):
            draw = self.rng.integers(0, self.c, size=self.c, dtype=np.int64, endpoint=False)
            self.draw_hash.update(draw.astype("<i8", copy=False).tobytes(order="C"))
            weights = tuple(int(value) for value in np.bincount(draw, minlength=self.c))
            total = sum(w * s for w, s in zip(weights, self.sizes, strict=True))
            self.low_n = total if self.low_n is None else min(self.low_n, total)
            self.high_n = total if self.high_n is None else max(self.high_n, total)
            arms = self._metrics(weights)
            for j in range(len(self.profile.metrics)):
                a, b = arms[0][j].value, arms[1][j].value
                if a is None or b is None:
                    raise ClusterBootstrapError()
                triple = (a, b, b - a)
                if not all(math.isfinite(value) for value in triple):
                    raise ClusterBootstrapError()
                values[self.completed, j, :] = triple
                self.value_hash.update(struct.pack("<ddd", *triple))
            self.completed += 1

    def result(self) -> ClusterBootstrapResult:
        if not self.done:
            raise ClusterBootstrapError()
        metrics: list[BootstrapMetric] = []
        for j, name in enumerate(self.profile.metrics):
            intervals: list[PercentileInterval | None] = [None, None, None]
            if self.c >= 2:
                import numpy as np

                values = cast("np.ndarray[tuple[int,int,int], np.dtype[np.float64]]", self.values)
                for series in range(3):
                    ordered = np.sort(values[:, j, series])
                    intervals[series] = PercentileInterval(
                        lower=float(ordered[249]),
                        upper=float(ordered[9749]),
                        lower_rank=250,
                        upper_rank=9750,
                        resamples=RESAMPLES,
                    )
            a, b = self.points[0][j], self.points[1][j]
            metrics.append(
                BootstrapMetric(
                    metric=name,
                    baseline=a,
                    candidate=b,
                    delta=None if a.value is None or b.value is None else b.value - a.value,
                    baseline_interval=intervals[0],
                    candidate_interval=intervals[1],
                    delta_interval=intervals[2],
                    interval_reason=None
                    if self.c >= 2
                    else "insufficient_clusters"
                    if self.n
                    else "no_metric_population",
                )
            )
        return ClusterBootstrapResult(
            schema_version=1,
            method="paired_cluster_percentile_v1",
            status="complete" if self.c >= 2 else "undefined",
            profile=self.profile,
            metrics=tuple(metrics),
            paired_rows=self.n,
            clusters=self.c,
            minimum_cluster_rows=min(self.sizes) if self.sizes else None,
            maximum_cluster_rows=max(self.sizes) if self.sizes else None,
            membership_sha256=self.membership,
            resamples_completed=RESAMPLES if self.completed == RESAMPLES else 0,
            minimum_resample_rows=self.low_n,
            maximum_resample_rows=self.high_n,
            draws_sha256=self.draw_hash.hexdigest() if self.c >= 2 else None,
            replicates_sha256=self.value_hash.hexdigest() if self.c >= 2 else None,
            planned_draw_work_units=RESAMPLES * self.c if self.c >= 2 else 0,
            draw_work_units=self.completed * self.c,
            planned_numeric_bytes=self.planned,
            numpy_version=self.version,
        )


def paired_cluster_percentile(
    observations: tuple[PairedClusterObservation, ...], *, profile: PairedClusterProfile
) -> ClusterBootstrapResult:
    """Compute only the fixed mathematical profile; source authority belongs to its caller."""
    result = None
    missing = False
    try:
        state = _ClusterBootstrapState(observations, profile)
        while not state.done:
            state.advance()
        result = state.result()
    except ImportError:
        missing = True
    except Exception:
        result = None
    if missing:
        raise ImportError("Install 'strata-forge[scoring]' for paired cluster percentiles.")
    if result is None:
        raise ClusterBootstrapError()
    return result
