"""Temperature fitting, cluster cross-fitting and expected calibration error."""

from __future__ import annotations

import math
import random

import pytest

from strata_forge.core.errors import ValidationError
from strata_forge.metrics.calibration import temperature_scale
from strata_forge.metrics.calibration_fit import (
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    cross_fit_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
)

LABELS = ("no", "yes")


def _binary(n: int, margin: float, share_yes: float) -> tuple[list[list[float]], list[str]]:
    """Every row says 'yes' with logit margin ``margin``; ``share_yes`` of the gold labels agree."""
    yes = round(n * share_yes)
    return [[0.0, margin] for _ in range(n)], ["yes"] * yes + ["no"] * (n - yes)


def test_analytic_optimum_for_a_single_margin() -> None:
    # sigmoid(beta * d) must equal the gold share q, so T = d / logit(q).
    logits, gold = _binary(100, math.log(3), 0.6)
    fit = fit_temperature(logits, gold, labels=LABELS)
    assert fit.temperature == pytest.approx(math.log(3) / math.log(1.5), rel=1e-9)
    assert fit.at_bound is None
    assert fit.nll_after < fit.nll_before
    assert fit.nll_after == pytest.approx(-(0.6 * math.log(0.6) + 0.4 * math.log(0.4)), rel=1e-9)


def test_a_constant_shift_per_row_changes_nothing() -> None:
    logits, gold = _binary(50, 1.2, 0.8)
    shifted = [[a + 7.0, b + 7.0] for a, b in logits]
    moved = fit_temperature(shifted, gold, labels=LABELS)
    base = fit_temperature(logits, gold, labels=LABELS)
    assert moved.temperature == pytest.approx(base.temperature, rel=1e-9)
    assert moved.nll_after == pytest.approx(base.nll_after, rel=1e-9)


def test_bounds_are_taken_and_reported() -> None:
    sharp = fit_temperature(*_binary(10, 2.0, 1.0), labels=LABELS)
    assert (sharp.temperature, sharp.at_bound) == (pytest.approx(MIN_TEMPERATURE), "min")
    flat = fit_temperature(*_binary(10, 2.0, 0.0), labels=LABELS)
    assert (flat.temperature, flat.at_bound) == (pytest.approx(MAX_TEMPERATURE), "max")


def test_fit_refusals() -> None:
    with pytest.raises(ValidationError, match="calibration_gold_invalid"):
        fit_temperature([[0.0, 1.0]], ["maybe"], labels=LABELS)
    with pytest.raises(ValidationError, match="calibration_logits_invalid"):
        fit_temperature([[0.0, math.inf]], ["yes"], labels=LABELS)
    with pytest.raises(ValidationError, match="calibration_logits_invalid"):
        fit_temperature([[0.0, 1.0, 2.0]], ["yes"], labels=LABELS)
    with pytest.raises(ValidationError, match="calibration_empty"):
        fit_temperature([], [], labels=LABELS)
    with pytest.raises(ValidationError, match="calibration_population_mismatch"):
        fit_temperature([[0.0, 1.0]], ["yes", "no"], labels=LABELS)


def _population(seed: int = 4) -> tuple[list[list[float]], list[str], list[str]]:
    rng = random.Random(seed)  # noqa: S311 — synthetic fixture.
    logits: list[list[float]] = []
    gold: list[str] = []
    keys: list[str] = []
    for c in range(30):
        for _ in range(rng.randint(1, 3)):
            label = rng.choice(LABELS)
            margin = rng.uniform(0.5, 3.0) * (1 if rng.random() < 0.7 else -1)
            logits.append([0.0, margin] if label == "yes" else [margin, 0.0])
            gold.append(label)
            keys.append(f"c{c:02d}")
    return logits, gold, keys


def test_cross_fit_is_seeded_and_holds_each_fold_out() -> None:
    logits, gold, keys = _population()
    first = cross_fit_temperature(logits, gold, labels=LABELS, clusters=keys, seed=20261104)
    assert first == cross_fit_temperature(logits, gold, labels=LABELS, clusters=keys, seed=20261104)
    assert sum(first.fold_rows) == len(gold)
    assert len(first.folds) == 2
    # Every row's probabilities are softmax(z / T) with ONE of the two fold temperatures.
    temps = [fit.temperature for fit in first.folds]
    for row, probs in zip(logits, first.probabilities, strict=True):
        options = [temperature_scale([row], temperature=t)[0] for t in temps]
        assert any(probs == pytest.approx(option, abs=1e-12) for option in options)
    other = cross_fit_temperature(logits, gold, labels=LABELS, clusters=keys, seed=7)
    assert other.fold_rows != first.fold_rows or other.folds != first.folds


def test_a_cluster_never_straddles_folds() -> None:
    logits, gold, keys = _population()
    result = cross_fit_temperature(logits, gold, labels=LABELS, clusters=keys, seed=1)
    temps = [fit.temperature for fit in result.folds]
    used: dict[str, set[int]] = {}
    for key, row, probs in zip(keys, logits, result.probabilities, strict=True):
        for index, t in enumerate(temps):
            if probs == pytest.approx(temperature_scale([row], temperature=t)[0], abs=1e-12):
                used.setdefault(key, set()).add(index)
    assert all(len(found) == 1 for found in used.values())


def test_cross_fit_refusals() -> None:
    with pytest.raises(ValidationError, match="calibration_too_few_clusters"):
        cross_fit_temperature([[0.0, 1.0]], ["yes"], labels=LABELS, clusters=["a"], seed=1)
    with pytest.raises(ValidationError, match="calibration_cluster_invalid"):
        cross_fit_temperature([[0.0, 1.0]], ["yes"], labels=LABELS, clusters=[""], seed=1)
    with pytest.raises(ValidationError, match="calibration_seed_invalid"):
        cross_fit_temperature([[0.0, 1.0]], ["yes"], labels=LABELS, clusters=["a"], seed=-1)


def test_ece_known_values() -> None:
    # Four rows at confidence 0.9, all right: |1.0 - 0.9| = 0.1 in one bin.
    probs = [[0.1, 0.9]] * 4
    assert expected_calibration_error(["yes"] * 4, probs, labels=LABELS).value == pytest.approx(0.1)
    # Calibrated: confidence 0.75 and three of four right.
    gold = ["yes", "yes", "yes", "no"]
    probs = [[0.25, 0.75]] * 4
    ece = expected_calibration_error(gold, probs, labels=LABELS, scheme="equal_width")
    assert ece.value == pytest.approx(0.0)
    # Equal mass, two bins of two: {0.6 wrong, 0.6 right} -> |0.5-0.6|; {0.9, 0.9 right} -> 0.1.
    probs = [[0.4, 0.6], [0.4, 0.6], [0.1, 0.9], [0.1, 0.9]]
    gold = ["no", "yes", "yes", "yes"]
    two = expected_calibration_error(gold, probs, labels=LABELS, bins=2)
    assert two.value == pytest.approx((2 * 0.1 + 2 * 0.1) / 4)
    assert (two.bins, two.scheme, two.n) == (2, "equal_mass", 4)


def test_ece_refusals() -> None:
    with pytest.raises(ValidationError, match="calibration_probabilities_invalid"):
        expected_calibration_error(["yes"], [[0.5, 0.6]], labels=LABELS)
    with pytest.raises(ValidationError, match="calibration_bins_invalid"):
        expected_calibration_error(["yes"], [[0.5, 0.5]], labels=LABELS, bins=0)


def test_fold_zero_holds_the_first_ceil_half_of_the_permuted_sorted_clusters() -> None:
    import numpy as np

    logits, gold, keys = _population()
    seed = 20261104
    result = cross_fit_temperature(logits, gold, labels=LABELS, clusters=keys, seed=seed)
    order = sorted(set(keys))
    permuted = np.random.Generator(np.random.PCG64(seed)).permutation(len(order)).tolist()
    fold0 = {order[c] for c in permuted[: math.ceil(len(order) / 2)]}
    assert result.fold_rows[0] == sum(1 for key in keys if key in fold0)


def test_ece_ties_break_by_item_id() -> None:
    # Three rows tie at 0.6 (right, wrong, right) beside one right at 0.9; with 3 bins the
    # array_split sizes are 2, 1, 1, so which tied row comes first decides the bins.
    gold = ["yes", "no", "yes", "yes"]
    probs = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.1, 0.9]]
    wrong_first = expected_calibration_error(
        gold, probs, labels=LABELS, bins=3, item_ids=["b", "a", "c", "d"]
    )
    # {wrong, right} -> |0.5 - 0.6|, {right} -> 0.4, {right @ 0.9} -> 0.1
    assert wrong_first.value == pytest.approx((2 * 0.1 + 0.4 + 0.1) / 4)
    wrong_last = expected_calibration_error(
        gold, probs, labels=LABELS, bins=3, item_ids=["a", "c", "b", "d"]
    )
    # {right, right} -> 0.4 each, {wrong} -> 0.6, {right @ 0.9} -> 0.1
    assert wrong_last.value == pytest.approx((2 * 0.4 + 0.6 + 0.1) / 4)


def test_reliability_bins_are_the_bins_the_ece_weighs() -> None:
    # The tie case above: three rows at 0.6 (right, wrong, right) and one right at 0.9.
    gold = ["yes", "no", "yes", "yes"]
    probs = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.1, 0.9]]
    ids = ["b", "a", "c", "d"]
    got = reliability_bins(gold, probs, labels=LABELS, bins=3, item_ids=ids)
    assert [(b.n, b.accuracy) for b in got] == [(2, 0.5), (1, 1.0), (1, 1.0)]
    assert [(b.lo, b.hi) for b in got] == [(0.6, 0.6), (0.6, 0.6), (0.9, 0.9)]
    assert got[0].mean_confidence == pytest.approx(0.6)
    ece = expected_calibration_error(gold, probs, labels=LABELS, bins=3, item_ids=ids)
    weighted = sum(b.n * abs((b.accuracy or 0) - (b.mean_confidence or 0)) for b in got) / 4
    assert ece.value == pytest.approx(weighted)


def test_reliability_bins_keep_empty_bins() -> None:
    probs = [[0.4, 0.6], [0.1, 0.9]]
    mass = reliability_bins(["yes", "no"], probs, labels=LABELS, bins=3)
    assert [b.n for b in mass] == [1, 1, 0]
    assert (mass[2].lo, mass[2].hi, mass[2].mean_confidence, mass[2].accuracy) == (
        None,
        None,
        None,
        None,
    )
    width = reliability_bins(["yes", "no"], probs, labels=LABELS, bins=4, scheme="equal_width")
    assert [(b.lo, b.hi, b.n) for b in width] == [
        (0.0, 0.25, 0),
        (0.25, 0.5, 0),
        (0.5, 0.75, 1),
        (0.75, 1.0, 1),
    ]
    assert width[3].accuracy == 0.0
