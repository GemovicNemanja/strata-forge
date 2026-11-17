"""Seeded item-level percentile bootstrap for the full-sample-selected agreement coefficient."""

from __future__ import annotations

import math
import random
import sys
import time
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError as ModelValidationError

from strata_forge.core.errors import ValidationError
from strata_forge.metrics.agreement import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_VERSION,
    MAX_BOOTSTRAP_SEED,
    AgreementBootstrap,
    HumanRatings,
    ReferencePrevalence,
    RegisteredReference,
    agreement_bootstrap,
    pairwise_agreement,
    reference_prevalence,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from strata_forge.metrics.agreement import CoefficientName

LABELS = ("E", "C", "U")
SEED = 20261104


def _human(name: str, ratings: Sequence[str], ids: Sequence[str] | None = None) -> HumanRatings:
    return HumanRatings(
        rater_id=name,
        rater_kind="human",
        item_ids=tuple(ids) if ids is not None else tuple(f"item-{i}" for i in range(len(ratings))),
        ratings=tuple(ratings),
    )


def _references(
    labels: Sequence[str], ids: Sequence[str] | None = None, *, names: Sequence[str] = LABELS
) -> ReferencePrevalence:
    item_ids = tuple(ids) if ids is not None else tuple(f"item-{i}" for i in range(len(labels)))
    return reference_prevalence(
        tuple(
            RegisteredReference(
                item_id=item_id,
                solver_gold=label,
                adjudicated=None,
                source_ref="synthetic-solver",
                version_ref="synthetic-1",
            )
            for item_id, label in zip(item_ids, labels, strict=True)
        ),
        labels=names,
        item_set_ref="synthetic-set",
        reference_version="synthetic-references-1",
    )


def _pilot(
    n: int = 120, *, seed: int = 7
) -> tuple[HumanRatings, HumanRatings, ReferencePrevalence]:
    rng = random.Random(seed)  # noqa: S311 — synthetic fixture, not a security context.
    a = [rng.choice(LABELS) for _ in range(n)]
    b = [label if rng.random() < 0.7 else rng.choice(LABELS) for label in a]
    return _human("user", a), _human("second", b), _references(a)


def _exact_resample_values(
    a: Sequence[str],
    b: Sequence[str],
    coefficient: CoefficientName,
    *,
    labels: Sequence[str],
    seed: int,
    resamples: int,
) -> tuple[list[float], int, list[list[int]]]:
    """Recompute every resample through the exact Fraction point path, independently."""
    n = len(a)
    draws = np.random.Generator(np.random.PCG64(seed)).integers(
        0, n, size=(resamples, n), dtype=np.int64
    )
    values: list[float] = []
    undefined = 0
    rows: list[list[int]] = [[int(index) for index in row] for row in draws]
    for row in rows:
        ids = [f"draw-{j}" for j in range(n)]
        report = pairwise_agreement(
            _human("a", [a[i] for i in row], ids),
            _human("b", [b[i] for i in row], ids),
            labels=labels,
        )
        estimate = report.gwet_ac1 if coefficient == "gwet_ac1" else report.cohen_kappa
        if estimate.value is None:
            undefined += 1
        else:
            values.append(estimate.value)
    return sorted(values), undefined, rows


def test_fixed_seed_is_deterministic_and_records_its_contract() -> None:
    a, b, refs = _pilot()
    first = agreement_bootstrap(a, b, refs, labels=LABELS, seed=SEED)
    second = agreement_bootstrap(a, b, refs, labels=LABELS, seed=SEED)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert first.interval.bootstrap_version == BOOTSTRAP_VERSION
    assert first.interval.rng == "numpy.random.Generator(PCG64)"
    assert first.interval.quantile_rule == "hyndman-fan-7"
    assert first.interval.coefficient_selection == "full_sample_reference"
    assert first.interval.seed == SEED
    assert first.interval.resamples == BOOTSTRAP_RESAMPLES
    assert first.interval.confidence_level == 0.95
    assert first.interval.numpy_version == np.__version__
    other = agreement_bootstrap(a, b, refs, labels=LABELS, seed=SEED + 1)
    assert (other.interval.lower, other.interval.upper) != (
        first.interval.lower,
        first.interval.upper,
    )
    assert AgreementBootstrap.model_validate_json(first.model_dump_json()) == first


def _type7(values: Sequence[float], probability: Fraction) -> float:
    """Hyndman-Fan type 7 at an exact rational position, written independently of the module."""
    position = (len(values) - 1) * probability
    index = position.numerator // position.denominator
    if index == len(values) - 1:
        return values[index]
    low, high = values[index], values[index + 1]
    return min(max(low + float(position - index) * (high - low), low), high)


TAIL = Fraction(1, 40)


def test_resamples_match_exact_point_arithmetic_and_type_7_percentiles() -> None:
    a, b, refs = _pilot(30, seed=3)
    result = agreement_bootstrap(a, b, refs, labels=LABELS, seed=SEED, resamples=400)
    assert result.coefficient == "cohen_kappa"
    expected, undefined, _ = _exact_resample_values(
        a.ratings, b.ratings, "cohen_kappa", labels=LABELS, seed=SEED, resamples=400
    )
    assert undefined == result.interval.undefined_resamples == 0
    assert result.interval.lower is not None
    assert result.interval.upper is not None
    assert result.interval.lower == _type7(expected, TAIL)
    assert result.interval.upper == _type7(expected, 1 - TAIL)
    numpy_bounds = np.quantile(np.asarray(expected), [0.025, 0.975], method="linear")
    low, high = (float(value) for value in numpy_bounds)
    assert math.isclose(result.interval.lower, low, abs_tol=1e-12)
    assert math.isclose(result.interval.upper, high, abs_tol=1e-12)
    assert result.estimate == pairwise_agreement(a, b, labels=LABELS).cohen_kappa
    assert result.estimate.value is not None
    assert result.interval.lower <= result.estimate.value <= result.interval.upper


def test_type_7_positions_are_exact_order_statistics_at_41_values() -> None:
    assert _type7([float(i) for i in range(41)], TAIL) == 1.0
    assert _type7([float(i) for i in range(41)], 1 - TAIL) == 39.0
    assert _type7([0.0, 1.0], TAIL) == 0.025


def test_perfect_agreement_counts_single_class_resamples_as_undefined() -> None:
    ratings = ("E", "E", "C", "C")
    a, b = _human("a", ratings), _human("b", ratings)
    result = agreement_bootstrap(
        a, b, _references(ratings), labels=LABELS, seed=SEED, resamples=2000
    )
    assert result.coefficient == "cohen_kappa"
    assert result.estimate.value == 1
    _, _, rows = _exact_resample_values(
        ratings, ratings, "cohen_kappa", labels=LABELS, seed=SEED, resamples=2000
    )
    single_class = sum(len({ratings[i] for i in row}) == 1 for row in rows)
    assert 0 < single_class < 2000
    assert result.interval.undefined_resamples == single_class
    assert (result.interval.lower, result.interval.upper, result.interval.reason) == (
        1.0,
        1.0,
        None,
    )


def test_all_resamples_undefined_when_selected_kappa_is_undefined() -> None:
    ratings = ("E",) * 6
    result = agreement_bootstrap(
        _human("a", ratings),
        _human("b", ratings),
        _references(("E", "E", "E", "C", "C", "C")),
        labels=LABELS,
        seed=SEED,
        resamples=50,
    )
    assert result.coefficient == "cohen_kappa"
    assert result.estimate.reason == "chance_agreement_one"
    assert result.interval.undefined_resamples == 50
    assert (result.interval.lower, result.interval.upper) == (None, None)
    assert result.interval.reason == "all_resamples_undefined"


@pytest.mark.parametrize(
    ("reference", "selected"),
    [
        (("E",) * 16 + ("C",) * 4, "gwet_ac1"),
        (("E",) * 15 + ("C",) * 5, "cohen_kappa"),
    ],
)
def test_coefficient_stays_fixed_by_the_full_sample_selection(
    reference: tuple[str, ...], selected: CoefficientName
) -> None:
    rng = random.Random(11)  # noqa: S311 — synthetic fixture, not a security context.
    a = [rng.choice(("E", "C")) for _ in reference]
    b = [rng.choice(("E", "C")) for _ in reference]
    result = agreement_bootstrap(
        _human("a", a),
        _human("b", b),
        _references(reference),
        labels=LABELS,
        seed=SEED,
        resamples=300,
    )
    assert result.coefficient == selected
    expected, undefined, rows = _exact_resample_values(
        a, b, selected, labels=LABELS, seed=SEED, resamples=300
    )
    other: CoefficientName = "cohen_kappa" if selected == "gwet_ac1" else "gwet_ac1"
    reselected = [
        (5 * max(sum(reference[i] == label for i in row) for label in LABELS) >= 4 * len(row))
        != (selected == "gwet_ac1")
        for row in rows
    ]
    assert any(reselected), "the fixture must contain resamples a per-resample rule would flip"
    assert result.interval.undefined_resamples == undefined
    assert (result.interval.lower, result.interval.upper) == (
        _type7(expected, TAIL),
        _type7(expected, 1 - TAIL),
    )
    flipped, _, _ = _exact_resample_values(a, b, other, labels=LABELS, seed=SEED, resamples=300)
    assert (result.interval.lower, result.interval.upper) != (
        _type7(flipped, TAIL),
        _type7(flipped, 1 - TAIL),
    )


def test_population_and_label_validation_refuses_with_fixed_codes() -> None:
    a, b, refs = _pilot(10)
    with pytest.raises(ValidationError, match="agreement_labels_invalid"):
        agreement_bootstrap(a, b, refs, labels=("E",), seed=SEED)
    with pytest.raises(ValidationError, match="agreement_labels_invalid"):
        agreement_bootstrap(a, b, refs, labels=("E", "E", "C"), seed=SEED)
    with pytest.raises(ValidationError, match="agreement_input_invalid"):
        agreement_bootstrap(a, b, refs, labels=("E", "C"), seed=SEED)
    shorter = _human("second", b.ratings[:-1], b.item_ids[:-1])
    with pytest.raises(ValidationError, match="agreement_input_invalid"):
        agreement_bootstrap(a, shorter, refs, labels=LABELS, seed=SEED)
    with pytest.raises(ModelValidationError):
        HumanRatings(rater_id="x", rater_kind="human", item_ids=("1", "2"), ratings=("E",))
    reordered = _references(
        tuple(reversed([r.label for r in refs.references])), tuple(reversed(refs.item_ids))
    )
    with pytest.raises(ValidationError, match="agreement_population_mismatch"):
        agreement_bootstrap(a, b, reordered, labels=LABELS, seed=SEED)
    wider = _references(["E"] * 10, names=("E", "C", "U", "X"))
    with pytest.raises(ValidationError, match="agreement_population_mismatch"):
        agreement_bootstrap(a, b, wider, labels=LABELS, seed=SEED)
    with pytest.raises(ValidationError, match="agreement_bootstrap_empty"):
        agreement_bootstrap(
            _human("a", ()), _human("b", ()), _references(()), labels=LABELS, seed=SEED
        )


@pytest.mark.parametrize("seed", [-1, True, 1.0, "1", MAX_BOOTSTRAP_SEED + 1])
def test_invalid_seed_refuses(seed: object) -> None:
    a, b, refs = _pilot(10)
    with pytest.raises(ValidationError, match="agreement_bootstrap_seed_invalid"):
        agreement_bootstrap(a, b, refs, labels=LABELS, seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("resamples", [0, -5, True, 10.0, None])
def test_invalid_resample_count_refuses(resamples: object) -> None:
    a, b, refs = _pilot(10)
    with pytest.raises(ValidationError, match="agreement_bootstrap_resamples_invalid"):
        agreement_bootstrap(a, b, refs, labels=LABELS, seed=SEED, resamples=resamples)  # type: ignore[arg-type]


def test_work_bound_refuses_before_drawing() -> None:
    a, b, refs = _pilot(120)
    with pytest.raises(ValidationError, match="agreement_bootstrap_too_large"):
        agreement_bootstrap(a, b, refs, labels=LABELS, seed=SEED, resamples=200_000)


def test_missing_numpy_has_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b, refs = _pilot(10)
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(ImportError, match=r"strata-forge\[scoring\]"):
        agreement_bootstrap(a, b, refs, labels=LABELS, seed=SEED, resamples=10)


def test_result_refuses_forged_or_inconsistent_fields() -> None:
    a, b, refs = _pilot(30, seed=3)
    result = agreement_bootstrap(a, b, refs, labels=LABELS, seed=SEED, resamples=100)
    payload = result.model_dump(mode="python")
    top: list[dict[str, object]] = [
        {"coefficient": "gwet_ac1"},
        {"estimate": result.agreement.gwet_ac1},
        {"interval": {**payload["interval"], "resamples": 10**9}},
    ]
    nested: list[dict[str, object]] = [
        {"undefined_resamples": 101},
        {"undefined_resamples": 100},
        {"lower": result.interval.upper, "upper": result.interval.lower},
        {"lower": None},
        {"upper": 1.5},
        {"confidence_level": 0.9},
        {"reason": "all_resamples_undefined"},
        {"seed": -1},
        {"bootstrap_version": "other"},
    ]
    for change in top:
        with pytest.raises(ModelValidationError):
            AgreementBootstrap.model_validate({**payload, **change})
    for change in nested:
        with pytest.raises(ModelValidationError):
            AgreementBootstrap.model_validate(
                {**payload, "interval": {**payload["interval"], **change}}
            )


def test_pilot_scale_runtime_bound() -> None:
    a, b, refs = _pilot(120)
    agreement_bootstrap(a, b, refs, labels=LABELS, seed=SEED, resamples=10)
    start = time.perf_counter()
    result = agreement_bootstrap(a, b, refs, labels=LABELS, seed=SEED, resamples=10_000)
    elapsed = time.perf_counter() - start
    assert result.interval.resamples == 10_000
    assert elapsed < 1.0
