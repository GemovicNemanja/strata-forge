"""Latency percentiles over timed passes, and an opt-in iteration bootstrap for them.

The point statistics follow one protocol: each timed pass yields its own p50/p95/p99 (Hyndman-Fan
type 7, which is ``numpy.percentile``'s default ``linear`` method, reproduced here operation for
operation so no NumPy is needed), the headline value of each percentile is the median of the
per-pass values with their minimum and maximum beside it, and the pooled percentile over every
measured iteration is reported too. Nothing is reported for a pass with no measurement.

:func:`latency_bootstrap` is a separate, opt-in interval for the pooled percentiles: a
nonparametric percentile bootstrap that resamples iterations, with the RNG, index order, quantile
rule and confidence of the approved agreement bootstrap. Iterations of one pass are not
independent draws (a warm cache drifts), so its population is a declared assumption, recorded in
the result, and not part of the point protocol above.
"""

from __future__ import annotations

import math
from fractions import Fraction
from statistics import median
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.core.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "LATENCY_BOOTSTRAP_VERSION",
    "LATENCY_QUANTILES",
    "LATENCY_VERSION",
    "MAX_LATENCY_BOOTSTRAP_DRAWS",
    "LatencyBootstrap",
    "LatencySummary",
    "PassQuantiles",
    "QuantileAcrossPasses",
    "QuantileInterval",
    "latency_bootstrap",
    "latency_summary",
    "percentile",
]

LATENCY_VERSION = "latency-pass-percentiles-v1"
LATENCY_BOOTSTRAP_VERSION = "latency-iteration-percentile-bootstrap-v1"
#: The percentiles every latency summary reports, in this order.
LATENCY_QUANTILES: tuple[int, ...] = (50, 95, 99)
#: Bounds the bootstrap's work: resamples times iterations, the index draws it makes.
MAX_LATENCY_BOOTSTRAP_DRAWS = 2_000_000_000
_MAX_RESAMPLES = 1_000_000
_MAX_SEED = 2**53 - 1
_CONFIDENCE = 0.95
# Index draws per generator call: consecutive row blocks of one generator reproduce the single
# (resamples, n) draw, so the chunk bounds memory without changing the stream.
_CHUNK_DRAWS = 1 << 22


class _Frozen(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, hide_input_in_errors=True, allow_inf_nan=False
    )


class PassQuantiles(_Frozen):
    """One timed pass: how many iterations it measured and its percentiles (``None`` when none)."""

    pass_index: int = Field(ge=0)
    n: int = Field(ge=0)
    p50: float | None
    p95: float | None
    p99: float | None


class QuantileAcrossPasses(_Frozen):
    """One percentile across passes: the median of the per-pass values, with min and max."""

    q: int
    median: float | None
    min: float | None
    max: float | None
    passes: int = Field(ge=0)


class LatencySummary(_Frozen):
    """The point summary of one boundary of one configuration, in milliseconds."""

    version: Literal["latency-pass-percentiles-v1"] = LATENCY_VERSION
    unit: Literal["ms"] = "ms"
    quantile_rule: Literal["hyndman-fan-7"] = "hyndman-fan-7"
    n: int = Field(ge=0)
    mean: float | None
    min: float | None
    max: float | None
    pooled: dict[str, float | None]
    across_passes: tuple[QuantileAcrossPasses, ...]
    passes: tuple[PassQuantiles, ...]


class QuantileInterval(_Frozen):
    """A bootstrap interval for one pooled percentile."""

    q: int
    lower: float | None
    upper: float | None


class LatencyBootstrap(_Frozen):
    """The interval block and the contract that produced it."""

    version: Literal["latency-iteration-percentile-bootstrap-v1"] = LATENCY_BOOTSTRAP_VERSION
    population: Literal["measured_iterations_pooled_across_passes"] = (
        "measured_iterations_pooled_across_passes"
    )
    rng: Literal["numpy.random.Generator(PCG64)"] = "numpy.random.Generator(PCG64)"
    quantile_rule: Literal["hyndman-fan-7"] = "hyndman-fan-7"
    confidence: float = _CONFIDENCE
    seed: int = Field(ge=0)
    resamples: int = Field(ge=1)
    n: int = Field(ge=0)
    intervals: tuple[QuantileInterval, ...]
    numpy_version: str | None


def _finite(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        raw = cast("object", value)
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValidationError("latency values must be numbers")
        number = float(raw)
        if not math.isfinite(number) or number < 0:
            raise ValidationError("latency values must be finite and non-negative")
        out.append(number)
    return out


def _lerp(a: float, b: float, t: float) -> float:
    # NumPy's `_lerp`: from the lower neighbour below the midpoint, from the upper one at or above
    # it, so the result is exact at both ends and monotone between them.
    diff = b - a
    if t >= 0.5:
        return b - diff * (1 - t)
    return a + diff * t


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile of ``values`` by Hyndman-Fan type 7 (``numpy.percentile`` default).

    Computed with the same float operations NumPy's ``linear`` method uses, so the two agree
    exactly rather than to within rounding. ``values`` must be non-empty.
    """
    data = sorted(_finite(values))
    if not data:
        raise ValidationError("a percentile needs at least one value")
    if not 0 <= q <= 100:
        raise ValidationError("q must be in [0, 100]")
    virtual = (q / 100) * (len(data) - 1)
    lower = math.floor(virtual)
    upper = min(lower + 1, len(data) - 1)
    return _lerp(data[lower], data[upper], virtual - lower)


def _quantiles(values: Sequence[float]) -> dict[int, float | None]:
    if not values:
        return dict.fromkeys(LATENCY_QUANTILES)
    return {q: percentile(values, q) for q in LATENCY_QUANTILES}


def latency_summary(passes: Sequence[Sequence[float]]) -> LatencySummary:
    """Per-pass percentiles, their median across passes, and the pooled percentiles.

    ``passes`` holds each timed pass's measured latencies in milliseconds, in measurement order.
    A pass that measured nothing is kept, with ``n=0`` and no percentiles, and is not counted in
    the across-pass median.
    """
    clean = [_finite(values) for values in passes]
    per_pass: list[PassQuantiles] = []
    for index, values in enumerate(clean):
        point = _quantiles(values)
        per_pass.append(
            PassQuantiles(
                pass_index=index, n=len(values), p50=point[50], p95=point[95], p99=point[99]
            )
        )
    across: list[QuantileAcrossPasses] = []
    for q in LATENCY_QUANTILES:
        observed = [value for item in per_pass if (value := getattr(item, f"p{q}")) is not None]
        across.append(
            QuantileAcrossPasses(
                q=q,
                median=float(median(observed)) if observed else None,
                min=min(observed) if observed else None,
                max=max(observed) if observed else None,
                passes=len(observed),
            )
        )
    pooled = [value for values in clean for value in values]
    return LatencySummary(
        n=len(pooled),
        mean=math.fsum(pooled) / len(pooled) if pooled else None,
        min=min(pooled) if pooled else None,
        max=max(pooled) if pooled else None,
        pooled={f"p{q}": value for q, value in _quantiles(pooled).items()},
        across_passes=tuple(across),
        passes=tuple(per_pass),
    )


def _bootstrap_setting(value: object, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValidationError(f"latency_bootstrap {name} must be an integer in [{low}, {high}]")
    return value


def latency_bootstrap(values: Sequence[float], *, seed: int, resamples: int) -> LatencyBootstrap:
    """Percentile-bootstrap intervals for the pooled p50/p95/p99 of ``values`` (milliseconds).

    ``values`` is every measured iteration of one boundary of one configuration, in measurement
    order (pass by pass). The stream is one C-order ``(resamples, n)`` index draw from
    ``Generator(PCG64(seed))`` over that order, drawn in row blocks that reproduce it; each
    resample's percentiles use type 7, and the interval ends are the type-7 2.5th and 97.5th
    percentiles of the resampled values. NumPy is imported here only (``[scoring]``).
    """
    data = _finite(values)
    chosen_seed = _bootstrap_setting(seed, 0, _MAX_SEED, "seed")
    count = _bootstrap_setting(resamples, 1, _MAX_RESAMPLES, "resamples")
    n = len(data)
    if n * count > MAX_LATENCY_BOOTSTRAP_DRAWS:
        raise ValidationError(
            f"latency_bootstrap would draw {n * count} indices; the limit is "
            f"{MAX_LATENCY_BOOTSTRAP_DRAWS}"
        )
    if not n:
        return LatencyBootstrap(
            seed=chosen_seed,
            resamples=count,
            n=0,
            intervals=tuple(
                QuantileInterval(q=q, lower=None, upper=None) for q in LATENCY_QUANTILES
            ),
            numpy_version=None,
        )
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "latency_bootstrap requires numpy: install 'strata-forge[scoring]'"
        ) from exc
    population = np.asarray(data, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(chosen_seed))
    rows = max(1, _CHUNK_DRAWS // n)
    resampled: dict[int, list[float]] = {q: [] for q in LATENCY_QUANTILES}
    for start in range(0, count, rows):
        block = min(rows, count - start)
        draws = generator.integers(0, n, size=(block, n), dtype=np.int64)
        values_block = np.percentile(population[draws], LATENCY_QUANTILES, axis=1)
        for position, q in enumerate(LATENCY_QUANTILES):
            resampled[q].extend(float(v) for v in values_block[position].tolist())
    # From the confidence's decimal spelling, so the tail is exactly 2.5 rather than 2.5000000000000022.
    tail = float((1 - Fraction(str(_CONFIDENCE))) / 2 * 100)
    return LatencyBootstrap(
        seed=chosen_seed,
        resamples=count,
        n=n,
        intervals=tuple(
            QuantileInterval(
                q=q,
                lower=percentile(resampled[q], tail),
                upper=percentile(resampled[q], 100 - tail),
            )
            for q in LATENCY_QUANTILES
        ),
        numpy_version=str(np.__version__),
    )
