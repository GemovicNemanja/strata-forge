"""Independent analytic scalar-logit fixtures, not real-model calibration results."""

from __future__ import annotations

import math
from typing import cast

import pytest

from strata_forge.metrics.temperature_fit import (
    TemperatureFitBounds,
    TemperatureFitError,
    fit_temperature_logits,
)


def bounds(low: float = 0.5, high: float = 4.0) -> TemperatureFitBounds:
    return TemperatureFitBounds(minimum=low, maximum=high)


def test_analytic_three_of_four_binary_optimum_and_unclipped_objective() -> None:
    gap = 2 * math.log(3)
    result = fit_temperature_logits(((gap, 0.0),) * 4, (0, 0, 0, 1), bounds=bounds())
    assert result.temperature == pytest.approx(2.0, abs=1e-7)
    assert result.objective_before == pytest.approx(
        -(3 * math.log(0.9) + math.log(0.1)) / 4, abs=1e-14
    )
    assert result.objective_after == pytest.approx(
        -(3 * math.log(0.75) + math.log(0.25)) / 4, abs=1e-14
    )
    assert result.identification == "not_claimed"
    assert result.derivative_passes <= 67
    assert result.iterations <= 64


@pytest.mark.parametrize(
    ("low", "high", "location"), [(0.5, 1.0, "maximum"), (3.0, 4.0, "minimum")]
)
def test_explicit_boundary_optimum(low: float, high: float, location: str) -> None:
    if low > 1:
        # This profile explicitly requires the neutral T=1 to be in its bounds.
        with pytest.raises(ValueError, match="less_than_equal"):
            bounds(low, high)
        return
    result = fit_temperature_logits(
        ((2 * math.log(3), 0.0),) * 4, (0, 0, 0, 1), bounds=bounds(low, high)
    )
    assert result.temperature == high
    assert result.selected_location == location
    assert result.search_status == "bounded_optimum"


def test_separable_data_selects_cold_boundary_without_requiring_zero_gradient() -> None:
    result = fit_temperature_logits(((1.0, 0.0), (0.0, 1.0)), (0, 1), bounds=bounds())
    assert result.temperature == 0.5
    assert result.search_status == "bounded_optimum"


def test_actual_equal_logits_are_flat_but_underflow_only_is_not_identified() -> None:
    flat = fit_temperature_logits(((4.0, 4.0), (-2.0, -2.0)), (0, 1), bounds=bounds())
    assert flat.temperature == 1
    assert flat.search_status == "flat_unidentified"
    saturated = fit_temperature_logits(((1_000_000.0, -1_000_000.0),), (0,), bounds=bounds())
    assert saturated.temperature == 1
    assert saturated.search_status == "numerical_stationary"
    assert saturated.underflowed_weights == 1
    assert saturated.identification == "not_claimed"


def test_class_order_and_per_row_shift_preserve_exact_input_semantics() -> None:
    logits = ((2.0, -1.0, 0.0), (0.0, 2.0, -1.0), (0.0, 0.0, 2.0))
    gold = (0, 1, 2)
    a = fit_temperature_logits(logits, gold, bounds=bounds())
    shifted = tuple(
        tuple(value + (index + 1) * 100 for value in row) for index, row in enumerate(logits)
    )
    b = fit_temperature_logits(shifted, gold, bounds=bounds())
    perm = (2, 0, 1)
    c = fit_temperature_logits(
        tuple(tuple(row[i] for i in perm) for row in logits),
        tuple(perm.index(i) for i in gold),
        bounds=bounds(),
    )
    assert (
        (a.temperature, a.objective_after)
        == (b.temperature, b.objective_after)
        == (c.temperature, c.objective_after)
    )


@pytest.mark.parametrize(
    ("logits", "gold"),
    [
        ((), ()),
        (((1.0,),), (0,)),
        (((1.0, 0.0),), ()),
        (((1.0, 0.0),), (True,)),
        (((1.0, 0.0),), (2,)),
        (((1.0, 0.0), (1.0, 0.0, 2.0)), (0, 1)),
        (((True, 0.0),), (0,)),
        (((None, 0.0),), (0,)),
        (((float("inf"), 0.0),), (0,)),
        (((float("nan"), 0.0),), (0,)),
        (((10**1000, 0.0),), (0,)),
        (((-1_000_001, 0.0),), (0,)),
    ],
)
def test_malformed_math_refuses_without_input_or_context(logits: object, gold: object) -> None:
    with pytest.raises(TemperatureFitError, match=r"^temperature_fit_invalid$") as caught:
        fit_temperature_logits(
            cast("tuple[tuple[object,...],...]", logits),
            cast("tuple[int,...]", gold),
            bounds=bounds(),
        )
    assert caught.value.__cause__ is caught.value.__context__ is None


def test_complete_declared_class_set_and_actual_nonzero_logit_extremes() -> None:
    result = fit_temperature_logits(
        ((1_000_000.0, -1_000_000.0, 0.0),) * 2, (0, 2), bounds=bounds(0.001, 1000.0)
    )
    assert math.isfinite(result.objective_before)
    assert math.isfinite(result.objective_after)
    assert result.objective_after <= result.objective_before
    assert result.search_status != "flat_unidentified"


def test_versioned_analytic_fixture_matches_public_fitter() -> None:
    import json
    from pathlib import Path

    value = json.loads((Path(__file__).parent / "fixtures" / "temperature_fit_v1.json").read_text())
    result = fit_temperature_logits(
        tuple(tuple(row) for row in value["logits"]),
        tuple(value["gold_indexes"]),
        bounds=TemperatureFitBounds.model_validate(value["temperature_bounds"]),
    )
    assert result.temperature == pytest.approx(value["expected_temperature"], abs=1e-7)
    assert result.objective_before == pytest.approx(
        value["expected_unclipped_nll_before"], abs=1e-14
    )
    assert result.objective_after == pytest.approx(value["expected_unclipped_nll_after"], abs=1e-14)
