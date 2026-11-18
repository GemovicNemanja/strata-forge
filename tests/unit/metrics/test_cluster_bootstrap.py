"""Seeded cluster percentile bootstrap for accuracy, macro-F1 and invalid rate."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
import pytest

from strata_forge.core.errors import ValidationError
from strata_forge.metrics.classification import classification_report
from strata_forge.metrics.cluster_bootstrap import (
    CLUSTER_BOOTSTRAP_RESAMPLES,
    CLUSTER_BOOTSTRAP_VERSION,
    ClusterBootstrap,
    MetricInterval,
    PairedClusterBootstrap,
    cluster_bootstrap,
    paired_cluster_bootstrap,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

LABELS = ("E", "C", "U")
SEED = 20261104


def _population(
    clusters: int = 40, *, seed: int = 3
) -> tuple[list[str], list[str | None], list[str | None], list[str]]:
    """Synthetic gold, two correlated systems and 1-4 rows per cluster."""
    rng = random.Random(seed)  # noqa: S311 — synthetic fixture, not a security context.
    gold: list[str] = []
    base: list[str | None] = []
    cand: list[str | None] = []
    keys: list[str] = []
    for c in range(clusters):
        hard = rng.random() < 0.4
        for _ in range(rng.randint(1, 4)):
            label = rng.choice(LABELS)
            gold.append(label)
            keys.append(f"base-{c:03d}")
            wrong = rng.choice([x for x in LABELS if x != label])
            b_ok = rng.random() < (0.45 if hard else 0.8)
            base.append(label if b_ok else (None if rng.random() < 0.1 else wrong))
            cand.append(label if (b_ok or rng.random() < 0.4) else wrong)
    return gold, base, cand, keys


def _bounds(interval: MetricInterval) -> tuple[float, float]:
    assert interval.lower is not None
    assert interval.upper is not None
    return interval.lower, interval.upper


def _type7(values: Sequence[float], probability: Fraction) -> float:
    """Hyndman-Fan type 7 at an exact rational position, written independently of the module."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    index = position.numerator // position.denominator
    if index == len(ordered) - 1:
        return ordered[index]
    low, high = ordered[index], ordered[index + 1]
    return low + float(position - index) * (high - low)


def _expanded_values(
    gold: Sequence[str],
    predictions: Sequence[str | None],
    keys: Sequence[str],
    *,
    seed: int,
    resamples: int,
) -> tuple[list[float], list[float]]:
    """Every resample rebuilt as explicit rows and scored by ``classification_report``."""
    order = sorted(set(keys))
    members = {key: [i for i, k in enumerate(keys) if k == key] for key in order}
    draws = np.random.Generator(np.random.PCG64(seed)).integers(
        0, len(order), size=(resamples, len(order)), dtype=np.int64
    )
    accuracy: list[float] = []
    macro: list[float] = []
    for row in draws.tolist():
        rows = [i for c in row for i in members[order[c]]]
        report = classification_report(
            [gold[i] for i in rows], [predictions[i] for i in rows], labels=LABELS
        )
        accuracy.append(report.accuracy.value or 0.0)
        macro.append(report.macro_f1.value or 0.0)
    return accuracy, macro


def test_contract_is_recorded_and_seeded() -> None:
    gold, base, _, keys = _population()
    first = cluster_bootstrap(gold, base, labels=LABELS, clusters=keys, seed=SEED)
    second = cluster_bootstrap(gold, base, labels=LABELS, clusters=keys, seed=SEED)
    assert first == second
    settings = first.settings
    assert settings.bootstrap_version == CLUSTER_BOOTSTRAP_VERSION
    assert settings.rng == "numpy.random.Generator(PCG64)"
    assert settings.quantile_rule == "hyndman-fan-7"
    assert settings.confidence_level == 0.95
    assert settings.resamples == CLUSTER_BOOTSTRAP_RESAMPLES
    assert (settings.n, settings.clusters) == (len(gold), 40)
    assert settings.numpy_version == np.__version__
    other = cluster_bootstrap(gold, base, labels=LABELS, clusters=keys, seed=SEED + 1)
    assert other.intervals.macro_f1 != first.intervals.macro_f1
    assert ClusterBootstrap.model_validate_json(first.model_dump_json()) == first


def test_points_equal_the_classification_report() -> None:
    gold, base, _, keys = _population()
    result = cluster_bootstrap(gold, base, labels=LABELS, clusters=keys, seed=SEED, resamples=50)
    report = classification_report(gold, base, labels=LABELS)
    assert result.intervals.accuracy.point == report.accuracy.value
    assert result.intervals.macro_f1.point == report.macro_f1.value
    assert result.intervals.invalid_rate.point == report.invalid_rate.value


def test_resamples_equal_explicitly_expanded_rows() -> None:
    gold, base, _, keys = _population(clusters=25)
    resamples = 200
    result = cluster_bootstrap(
        gold, base, labels=LABELS, clusters=keys, seed=SEED, resamples=resamples
    )
    accuracy, macro = _expanded_values(gold, base, keys, seed=SEED, resamples=resamples)
    tail = Fraction(1, 40)
    assert result.intervals.accuracy.lower == pytest.approx(_type7(accuracy, tail), abs=1e-12)
    assert result.intervals.accuracy.upper == pytest.approx(_type7(accuracy, 1 - tail), abs=1e-12)
    assert result.intervals.macro_f1.lower == pytest.approx(_type7(macro, tail), abs=1e-12)
    assert result.intervals.macro_f1.upper == pytest.approx(_type7(macro, 1 - tail), abs=1e-12)


def test_row_order_and_cluster_spelling_order_do_not_matter() -> None:
    gold, base, _, keys = _population()
    order = list(range(len(gold)))
    random.Random(11).shuffle(order)  # noqa: S311 — synthetic permutation.
    first = cluster_bootstrap(gold, base, labels=LABELS, clusters=keys, seed=SEED, resamples=300)
    shuffled = cluster_bootstrap(
        [gold[i] for i in order],
        [base[i] for i in order],
        labels=LABELS,
        clusters=[keys[i] for i in order],
        seed=SEED,
        resamples=300,
    )
    assert shuffled == first


def test_singleton_clusters_equal_an_item_bootstrap() -> None:
    gold, base, _, _ = _population()
    keys = [f"row-{i:04d}" for i in range(len(gold))]
    result = cluster_bootstrap(gold, base, labels=LABELS, clusters=keys, seed=SEED, resamples=120)
    accuracy, _ = _expanded_values(gold, base, keys, seed=SEED, resamples=120)
    assert result.intervals.accuracy.lower == pytest.approx(_type7(accuracy, Fraction(1, 40)))
    assert result.settings.clusters == len(gold)


def test_clustering_widens_the_interval_for_correlated_rows() -> None:
    """Rows duplicated within a cluster carry one cluster's information, not k rows'."""
    rng = random.Random(5)  # noqa: S311 — synthetic fixture.
    gold: list[str] = []
    pred: list[str] = []
    keys: list[str] = []
    for c in range(60):
        label = rng.choice(LABELS)
        guess = label if rng.random() < 0.7 else rng.choice([x for x in LABELS if x != label])
        for r in range(5):
            gold.append(label)
            pred.append(guess)
            keys.append(f"b{c}-{r}")
    items = cluster_bootstrap(gold, pred, labels=LABELS, clusters=keys, seed=SEED, resamples=2000)
    clustered = cluster_bootstrap(
        gold,
        pred,
        labels=LABELS,
        clusters=[key.split("-")[0] for key in keys],
        seed=SEED,
        resamples=2000,
    )

    def width(result: ClusterBootstrap) -> float:
        return _bounds(result.intervals.accuracy)[1] - _bounds(result.intervals.accuracy)[0]

    assert width(clustered) > 1.5 * width(items)


def test_paired_delta_is_narrower_than_marginal_differences() -> None:
    gold, base, cand, keys = _population(clusters=80)
    paired = paired_cluster_bootstrap(
        gold, base, cand, labels=LABELS, clusters=keys, seed=SEED, resamples=2000
    )
    alone = cluster_bootstrap(gold, base, labels=LABELS, clusters=keys, seed=SEED, resamples=2000)
    assert paired.baseline == alone.intervals
    delta = paired.delta.accuracy
    cand_point, base_point = paired.candidate.accuracy.point, paired.baseline.accuracy.point
    assert cand_point is not None
    assert base_point is not None
    assert delta.point == pytest.approx(cand_point - base_point)
    cand_lo, cand_hi = _bounds(paired.candidate.accuracy)
    base_lo, base_hi = _bounds(paired.baseline.accuracy)
    delta_lo, delta_hi = _bounds(delta)
    marginal = (cand_hi - base_lo) - (cand_lo - base_hi)
    assert delta_hi - delta_lo < marginal / 2
    assert delta_lo > 0  # the candidate only ever fixes baseline errors
    assert PairedClusterBootstrap.model_validate_json(paired.model_dump_json()) == paired


def test_identical_systems_have_a_zero_delta() -> None:
    gold, base, _, keys = _population()
    paired = paired_cluster_bootstrap(
        gold, base, base, labels=LABELS, clusters=keys, seed=SEED, resamples=100
    )
    for interval in (paired.delta.accuracy, paired.delta.macro_f1, paired.delta.invalid_rate):
        assert (interval.point, interval.lower, interval.upper) == (0.0, 0.0, 0.0)


def test_a_class_absent_from_both_sides_leaves_macro_f1_undefined() -> None:
    gold = ["E", "E", "C", "C"]
    pred = ["E", "E", "C", "C"]
    result = cluster_bootstrap(
        gold, pred, labels=LABELS, clusters=["a", "a", "b", "b"], seed=1, resamples=50
    )
    # U is declared but absent from the whole population: macro-F1 has no value at all (never one
    # averaged with a zero), and no resample has one either.
    macro = result.intervals.macro_f1
    assert (macro.point, macro.lower, macro.upper) == (None, None, None)
    assert (macro.reason, macro.absent_classes) == ("absent_classes", ("U",))
    assert macro.undefined_resamples == 50
    assert result.intervals.accuracy.undefined_resamples == 0
    # The diagnostic averages the classes present (E and C), under the same undefined rule.
    present = result.intervals.macro_f1_present_classes
    assert result.intervals.present_classes == ("E", "C")
    assert present.point == 1.0
    assert present.reason is None


def test_a_delta_is_undefined_where_either_side_lacks_a_class() -> None:
    gold = ["E", "E", "C", "C"]
    base = ["E", "E", "C", "C"]
    cand = ["E", "U", "C", "C"]
    paired = paired_cluster_bootstrap(
        gold, base, cand, labels=LABELS, clusters=["a", "b", "c", "d"], seed=1, resamples=50
    )
    assert paired.baseline.macro_f1.absent_classes == ("U",)
    assert paired.candidate.macro_f1.point is not None
    delta = paired.delta.macro_f1
    assert (delta.point, delta.reason, delta.absent_classes) == (None, "absent_classes", ("U",))
    # Present classes are those seen in gold or ANY compared system: E, C and U here.
    assert paired.delta.present_classes == ("E", "C", "U")
    assert paired.delta.macro_f1_present_classes.point is None


def test_undefined_resamples_are_counted_not_zeroed() -> None:
    gold = ["E", "C", "U", "U"]
    pred = ["E", "C", "U", "E"]
    keys = ["a", "b", "c", "d"]
    resamples = 400
    result = cluster_bootstrap(
        gold, pred, labels=LABELS, clusters=keys, seed=SEED, resamples=resamples
    )
    draws = np.random.Generator(np.random.PCG64(SEED)).integers(0, 4, size=(resamples, 4)).tolist()
    expected_undefined = 0
    defined: list[float] = []
    for row in draws:
        g = [gold[i] for i in row]
        p = [pred[i] for i in row]
        if any(label not in g and label not in p for label in LABELS):
            expected_undefined += 1
        else:
            defined.append(classification_report(g, p, labels=LABELS).macro_f1.value or 0.0)
    macro = result.intervals.macro_f1
    assert macro.undefined_resamples == expected_undefined > 0
    assert macro.lower == pytest.approx(_type7(defined, Fraction(1, 40)), abs=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"seed": -1}, "cluster_bootstrap_seed_invalid"),
        ({"seed": True}, "cluster_bootstrap_seed_invalid"),
        ({"resamples": 0}, "cluster_bootstrap_resamples_invalid"),
        ({"resamples": 10**7}, "cluster_bootstrap_resamples_invalid"),
    ],
)
def test_settings_are_checked(kwargs: dict[str, int], code: str) -> None:
    options: dict[str, int] = {"seed": 1, "resamples": 10, **kwargs}
    with pytest.raises(ValidationError, match=code):
        cluster_bootstrap(
            ["E"],
            ["E"],
            labels=LABELS,
            clusters=["a"],
            seed=options["seed"],
            resamples=options["resamples"],
        )


def test_population_errors() -> None:
    with pytest.raises(ValidationError, match="cluster_bootstrap_empty"):
        cluster_bootstrap([], [], labels=LABELS, clusters=[], seed=1)
    with pytest.raises(ValidationError, match="cluster_bootstrap_population_mismatch"):
        cluster_bootstrap(["E"], ["E"], labels=LABELS, clusters=["a", "b"], seed=1)
    with pytest.raises(ValidationError, match="cluster_bootstrap_population_mismatch"):
        paired_cluster_bootstrap(["E"], ["E"], [], labels=LABELS, clusters=["a"], seed=1)
    with pytest.raises(ValidationError, match="cluster_bootstrap_cluster_invalid"):
        cluster_bootstrap(["E"], ["E"], labels=LABELS, clusters=[""], seed=1)
    with pytest.raises(ValidationError, match="gold label"):
        cluster_bootstrap(["X"], ["E"], labels=LABELS, clusters=["a"], seed=1)


def test_golden_bounds_pin_the_numpy_stream() -> None:
    """Exact bounds for one fixed input, so a change in NumPy's PCG64 stream fails CI."""
    gold = ["E", "C", "U", "E", "C", "U", "E", "E", "C", "U", "U", "C"]
    base = ["E", "C", "E", "E", None, "U", "C", "E", "C", "E", "U", "C"]
    cand = ["E", "C", "U", "E", "C", "U", "C", "E", "C", "U", "U", "E"]
    keys = ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4", "p5", "p5", "p6", "p6"]
    result = paired_cluster_bootstrap(
        gold, base, cand, labels=LABELS, clusters=keys, seed=SEED, resamples=1000
    )
    assert result.baseline.macro_f1.lower == 0.42735042735042733
    assert result.baseline.macro_f1.upper == 0.8690740740740741
    assert result.baseline.macro_f1.undefined_resamples == 2
    assert result.delta.macro_f1.lower == -0.1662037037037037
    assert result.delta.macro_f1.upper == 0.4453703703703704
    assert result.delta.macro_f1.undefined_resamples == 4
    assert result.delta.accuracy.lower == -0.16666666666666674
    assert result.delta.accuracy.upper == 0.41666666666666663
