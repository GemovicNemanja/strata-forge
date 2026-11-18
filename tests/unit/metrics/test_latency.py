"""Latency percentiles and the opt-in iteration bootstrap, against NumPy and known values."""

from __future__ import annotations

import random

import numpy as np
import pytest

from strata_forge.core.errors import ValidationError
from strata_forge.metrics import latency_bootstrap, latency_summary, percentile
from strata_forge.metrics.latency import MAX_LATENCY_BOOTSTRAP_DRAWS


@pytest.mark.parametrize("n", [1, 2, 3, 7, 100, 1999, 2000])
@pytest.mark.parametrize("q", [0, 1, 50, 95, 99, 99.9, 100])
def test_percentile_equals_numpy_percentile_exactly(n: int, q: float) -> None:
    rng = random.Random(n * 1000 + int(q * 10))  # noqa: S311 - synthetic test data
    values = [rng.lognormvariate(3.0, 0.6) for _ in range(n)]
    assert percentile(values, q) == float(np.percentile(values, q))


def test_percentile_known_values() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([1.0, 2.0], 100) == 2.0
    # Order of the input does not matter.
    assert percentile([40.0, 10.0, 30.0, 20.0], 25) == percentile([10.0, 20.0, 30.0, 40.0], 25)


@pytest.mark.parametrize("bad", [[], [float("nan")], [-1.0], [float("inf")]])
def test_percentile_refuses_what_is_not_a_latency(bad: list[float]) -> None:
    with pytest.raises(ValidationError):
        percentile(bad, 50)


def test_percentile_refuses_a_q_out_of_range() -> None:
    with pytest.raises(ValidationError):
        percentile([1.0], 101)


def test_a_boolean_is_not_a_latency() -> None:
    with pytest.raises(ValidationError):
        latency_summary([[True]])  # pyright: ignore[reportArgumentType]


def test_summary_reports_the_median_of_per_pass_percentiles_with_min_and_max() -> None:
    passes = [[10.0, 20.0, 30.0], [11.0, 21.0, 41.0], [9.0, 19.0, 29.0], [100.0], []]
    summary = latency_summary(passes)
    per_pass_p95 = [percentile(p, 95) for p in passes if p]
    across = {item.q: item for item in summary.across_passes}
    assert across[95].median == float(np.median(per_pass_p95))
    assert across[95].min == min(per_pass_p95)
    assert across[95].max == max(per_pass_p95)
    # The empty pass is kept but contributes no percentile.
    assert across[95].passes == 4
    assert summary.passes[4].n == 0
    assert summary.passes[4].p95 is None
    pooled = [v for p in passes for v in p]
    assert summary.n == len(pooled)
    assert summary.pooled["p99"] == float(np.percentile(pooled, 99))
    assert summary.min == 9.0
    assert summary.max == 100.0
    assert summary.unit == "ms"
    assert summary.quantile_rule == "hyndman-fan-7"


def test_a_summary_of_nothing_reports_nothing() -> None:
    summary = latency_summary([[], []])
    assert summary.n == 0
    assert summary.mean is None
    assert summary.pooled == {"p50": None, "p95": None, "p99": None}
    assert all(item.median is None and item.passes == 0 for item in summary.across_passes)


def test_bootstrap_is_reproducible_by_seed_and_brackets_the_point() -> None:
    rng = random.Random(7)  # noqa: S311 - synthetic test data
    values = [rng.lognormvariate(3.0, 0.5) for _ in range(500)]
    first = latency_bootstrap(values, seed=20261104, resamples=400)
    again = latency_bootstrap(values, seed=20261104, resamples=400)
    other = latency_bootstrap(values, seed=1, resamples=400)
    assert first == again
    assert first != other
    for interval in first.intervals:
        point = percentile(values, interval.q)
        assert interval.lower is not None
        assert interval.upper is not None
        assert interval.lower <= point <= interval.upper
    assert first.population == "measured_iterations_pooled_across_passes"
    assert first.rng == "numpy.random.Generator(PCG64)"
    assert first.n == 500
    assert first.numpy_version == np.__version__


def test_bootstrap_draws_one_c_order_index_stream() -> None:
    # The resampled p50s are those of the rows of one (resamples, n) draw, whatever the chunking.
    values = [float(v) for v in range(1, 40)]
    result = latency_bootstrap(values, seed=3, resamples=50)
    draws = np.random.Generator(np.random.PCG64(3)).integers(0, 39, size=(50, 39), dtype=np.int64)
    medians = np.percentile(np.asarray(values)[draws], 50, axis=1).tolist()
    expected_lower = percentile(medians, 2.5)
    assert result.intervals[0].q == 50
    assert result.intervals[0].lower == expected_lower


def test_bootstrap_of_nothing_estimates_nothing() -> None:
    result = latency_bootstrap([], seed=0, resamples=10)
    assert result.n == 0
    assert all(i.lower is None and i.upper is None for i in result.intervals)


@pytest.mark.parametrize(
    ("seed", "resamples"), [(-1, 10), (0, 0), (True, 10), (0, 2_000_000), (2**60, 10)]
)
def test_bootstrap_settings_are_bounded(seed: object, resamples: object) -> None:
    with pytest.raises(ValidationError):
        latency_bootstrap([1.0], seed=seed, resamples=resamples)  # pyright: ignore[reportArgumentType]


def test_bootstrap_work_is_bounded() -> None:
    too_many = MAX_LATENCY_BOOTSTRAP_DRAWS // 1000 + 1
    with pytest.raises(ValidationError, match="limit"):
        latency_bootstrap([1.0] * too_many, seed=0, resamples=1000)
