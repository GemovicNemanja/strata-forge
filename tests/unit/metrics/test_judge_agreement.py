from __future__ import annotations

import pytest

from strata_forge.core.errors import ValidationError
from strata_forge.metrics.judge_agreement import judge_agreement


def test_pass_fail_known_values() -> None:
    got = judge_agreement([1, 1, 0, 0], [1, 0, 0, 0], values=[0, 1], scale="pass_fail")
    # gold rows (fail, pass) x judge columns (fail, pass)
    assert got.counts == ((2, 0), (1, 1))
    assert got.percent_agreement == 0.75
    # chance = (2*3 + 2*1) / 16 = 1/2, so kappa = (3/4 - 1/2) / (1/2)
    assert got.cohen_kappa == 0.5
    assert got.mean_absolute_error is None
    assert got.quadratic_weighted_kappa is None
    assert got.quadratic_weighted_kappa_undefined is None


def test_ordinal_known_values() -> None:
    got = judge_agreement([1, 2, 3], [1, 3, 3], values=[1, 2, 3], scale="ordinal")
    assert got.mean_absolute_error == pytest.approx(1 / 3)
    assert got.percent_agreement == pytest.approx(2 / 3)
    # observed weighted disagreement 1/12 over expected 5/12
    assert got.quadratic_weighted_kappa == pytest.approx(0.8)


def test_undefined_values_are_null_with_a_reason() -> None:
    same = judge_agreement([1, 1], [1, 1], values=[0, 1], scale="pass_fail")
    assert (same.percent_agreement, same.cohen_kappa) == (1.0, None)
    assert same.cohen_kappa_undefined == "chance_agreement_one"
    empty = judge_agreement([], [], values=[1, 2, 3], scale="ordinal")
    assert empty.n == 0
    assert (empty.percent_agreement, empty.mean_absolute_error) == (None, None)
    assert (empty.cohen_kappa_undefined, empty.quadratic_weighted_kappa_undefined) == (
        "empty",
        "empty",
    )
    flat = judge_agreement([2, 2], [2, 2], values=[1, 2, 3], scale="ordinal")
    assert flat.quadratic_weighted_kappa_undefined == "chance_agreement_one"


def test_refusals() -> None:
    with pytest.raises(ValidationError, match="judge_agreement_population_mismatch"):
        judge_agreement([1], [1, 0], values=[0, 1], scale="pass_fail")
    with pytest.raises(ValidationError, match="judge_agreement_score_invalid"):
        judge_agreement([2], [1], values=[0, 1], scale="pass_fail")
    with pytest.raises(ValidationError, match="judge_agreement_score_invalid"):
        judge_agreement([True], [1], values=[0, 1], scale="pass_fail")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValidationError, match="judge_agreement_values_invalid"):
        judge_agreement([1], [1], values=[1], scale="ordinal")
