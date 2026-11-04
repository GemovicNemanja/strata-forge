"""Independent analytic fixtures and exact-rational references for comparison primitives."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import pytest
from pydantic import ValidationError as ModelValidationError

from strata_forge.core.errors import ValidationError
from strata_forge.metrics import (
    COMPARISON_VERSION,
    MAX_EXACT_DISCORDANT,
    AdjustedPValue,
    HolmCorrection,
    McNemarResult,
    ProportionInterval,
    holm_correction,
    mcnemar_exact,
    wilson_interval,
)


class WilsonCase(TypedDict):
    successes: int
    n: int
    confidence: float
    low: float
    high: float


class McNemarCase(TypedDict):
    baseline_only: int
    candidate_only: int
    n: int
    p_value: float
    delta: float
    log_p: NotRequired[float]


class HolmCase(TypedDict):
    raw: list[float]
    adjusted: list[float]


class Fixture(TypedDict):
    comparison_version: str
    wilson: list[WilsonCase]
    mcnemar: list[McNemarCase]
    holm: list[HolmCase]


@pytest.fixture
def reference() -> Fixture:
    path = Path(__file__).parent / "fixtures" / "classification_comparison_v1.json"
    return cast("Fixture", json.loads(path.read_text()))


def test_wilson_analytic_fixtures(reference: Fixture) -> None:
    for case in reference["wilson"]:
        result = wilson_interval(case["successes"], case["n"], confidence_level=case["confidence"])
        assert result.low == pytest.approx(case["low"], rel=5e-14, abs=1e-25)
        assert result.high == pytest.approx(case["high"], rel=5e-14, abs=1e-25)
        assert result.estimate == case["successes"] / case["n"]
        assert result.comparison_version == reference["comparison_version"] == COMPARISON_VERSION
        assert ProportionInterval.model_validate_json(result.model_dump_json()) == result


def test_wilson_empty_is_undefined_and_extreme_confidence_is_finite() -> None:
    empty = wilson_interval(0, 0)
    assert empty.estimate is empty.low is empty.high is None
    assert empty.reason == "empty"
    extreme = wilson_interval(5, 10, confidence_level=math.nextafter(1.0, 0.0))
    assert extreme.low is not None
    assert extreme.high is not None
    assert math.isfinite(extreme.low)
    assert math.isfinite(extreme.high)


@pytest.mark.parametrize("count", [-1, 1.0, True, None, 2**53])
def test_invalid_counts_are_never_coerced(count: object) -> None:
    with pytest.raises(ValidationError, match="counts"):
        wilson_interval(cast("int", count), 10)
    with pytest.raises(ValidationError, match="counts"):
        mcnemar_exact(0, 0, n_pairs=cast("int", count))


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, math.inf, math.nan, True])
def test_invalid_confidence_refused(confidence: float) -> None:
    with pytest.raises(ValidationError, match="probabilities"):
        wilson_interval(1, 10, confidence_level=confidence)


def test_counts_cannot_exceed_the_declared_population() -> None:
    with pytest.raises(ValidationError, match="eligible population"):
        wilson_interval(2, 1)
    with pytest.raises(ValidationError, match="paired population"):
        mcnemar_exact(2, 1, n_pairs=2)
    with pytest.raises(ValidationError, match="numerical range"):
        mcnemar_exact(MAX_EXACT_DISCORDANT + 1, 0, n_pairs=MAX_EXACT_DISCORDANT + 1)


def test_mcnemar_exact_rational_and_high_precision_fixtures(reference: Fixture) -> None:
    for case in reference["mcnemar"]:
        result = mcnemar_exact(case["baseline_only"], case["candidate_only"], n_pairs=case["n"])
        assert result.p_value == pytest.approx(case["p_value"], rel=5e-14, abs=0)
        assert result.accuracy_delta == case["delta"]
        if "log_p" in case:
            assert result.log_p_value == pytest.approx(case["log_p"], rel=0, abs=3e-14)
        assert result.n == case["n"]
        assert McNemarResult.model_validate_json(result.model_dump_json()) == result


def test_large_central_probability_avoids_log_gamma_cancellation() -> None:
    expected = float(1 - Fraction(math.comb(100_000, 50_000), 1 << 100_000))
    result = mcnemar_exact(50_001, 49_999, n_pairs=100_000)
    assert result.p_value == pytest.approx(expected, rel=0, abs=2e-15)


@pytest.mark.parametrize(
    ("discordant", "lower"), [(4096, 2047), (4097, 2047), (5000, 2498), (5001, 2499), (10000, 4990)]
)
def test_large_central_tails_against_integer_complement(discordant: int, lower: int) -> None:
    middle = sum(math.comb(discordant, index) for index in range(lower + 1, discordant - lower))
    expected = float(1 - Fraction(middle, 1 << discordant))
    result = mcnemar_exact(discordant - lower, lower, n_pairs=discordant)
    assert result.p_value == pytest.approx(expected, rel=0, abs=3e-15)


def test_all_small_paired_tables_against_exact_integer_binomial_sum() -> None:
    for discordant in range(1, 61):
        for baseline in range(discordant + 1):
            candidate = discordant - baseline
            lower = min(baseline, candidate)
            expected = min(
                Fraction(1),
                Fraction(
                    2 * sum(math.comb(discordant, i) for i in range(lower + 1)), 1 << discordant
                ),
            )
            result = mcnemar_exact(baseline, candidate, n_pairs=discordant + 17)
            assert result.p_value == pytest.approx(float(expected), rel=1e-14, abs=0)
            assert result.accuracy_delta == (candidate - baseline) / (discordant + 17)


def test_empty_pairs_and_no_discordance_are_different() -> None:
    empty = mcnemar_exact(0, 0, n_pairs=0)
    assert empty.reason == "empty"
    assert empty.p_value is empty.log_p_value is empty.accuracy_delta is None
    complete = mcnemar_exact(0, 0, n_pairs=20)
    assert complete.reason is None
    assert complete.p_value == 1
    assert complete.log_p_value == 0
    assert complete.accuracy_delta == 0


def test_swapping_models_preserves_p_and_reverses_effect() -> None:
    first = mcnemar_exact(9, 1, n_pairs=1000)
    second = mcnemar_exact(1, 9, n_pairs=1000)
    assert first.p_value == second.p_value
    assert first.accuracy_delta == -0.008
    assert second.accuracy_delta == 0.008


@pytest.mark.parametrize(
    ("discordant", "expected"), [(1075, math.ulp(0.0)), (1076, 0.0), (2000, 0.0)]
)
def test_full_two_sided_log_survives_subnormal_and_underflow(
    discordant: int, expected: float
) -> None:
    result = mcnemar_exact(discordant, 0, n_pairs=discordant)
    assert result.p_value == expected
    assert result.underflowed == (expected == 0)
    assert result.log_p_value == (1 - discordant) * math.log(2)
    assert result.log_p_value is not None
    assert math.isfinite(result.log_p_value)


def test_holm_reference_families_keep_caller_order(reference: Fixture) -> None:
    for case in reference["holm"]:
        family = {f"hypothesis-{index}": value for index, value in enumerate(case["raw"])}
        result = holm_correction(family, family_id="candidate:v1/evaluation")
        assert tuple(item.hypothesis_id for item in result.hypotheses) == tuple(family)
        assert [item.adjusted_p_value for item in result.hypotheses] == pytest.approx(
            case["adjusted"]
        )
        assert result.n == len(family)
        assert result.family_id == "candidate:v1/evaluation"
        assert HolmCorrection.model_validate_json(result.model_dump_json()) == result


def test_holm_empty_singleton_and_extreme_represented_values() -> None:
    empty = holm_correction({}, family_id="explicit-family")
    assert empty.n == 0
    assert empty.reason == "empty"
    assert empty.hypotheses == ()
    single = holm_correction({"one": 0.3}, family_id="single")
    assert single.hypotheses[0].adjusted_p_value == 0.3
    values = holm_correction(
        {"zero": 0.0, "min": math.ulp(0.0), "one": 1.0}, family_id="represented"
    )
    assert [item.adjusted_p_value for item in values.hypotheses] == [0.0, 2 * math.ulp(0.0), 1.0]
    underflow = mcnemar_exact(1076, 0, n_pairs=1076)
    assert underflow.p_value is not None
    adjusted = holm_correction(
        {"underflowed": underflow.p_value, "other": 1.0}, family_id="float-input"
    )
    assert adjusted.hypotheses[0].adjusted_p_value == 0.0


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, True])
def test_invalid_raw_p_values_refused(value: float) -> None:
    with pytest.raises(ValidationError, match="probabilities"):
        holm_correction({"test": value}, family_id="family")


@pytest.mark.parametrize("identifier", ["", "line\nbreak", "a" * 129, " space", chr(0x3B1)])
def test_explicit_family_and_hypothesis_ids_are_bounded(identifier: str) -> None:
    with pytest.raises(ValidationError, match="identifiers"):
        holm_correction({"test": 0.2}, family_id=identifier)
    with pytest.raises(ValidationError, match="identifiers"):
        holm_correction({identifier: 0.2}, family_id="family")


def test_bounded_family_size() -> None:
    with pytest.raises(ValidationError, match="family exceeds"):
        holm_correction({str(index): 0.2 for index in range(10001)}, family_id="family")


def test_serialized_results_refuse_inconsistent_populations_and_hidden_alpha() -> None:
    with pytest.raises(ModelValidationError):
        ProportionInterval(successes=1, n=0, confidence_level=0.95, reason="empty")
    with pytest.raises(ModelValidationError):
        McNemarResult(
            n=20,
            baseline_only_correct=0,
            candidate_only_correct=0,
            discordant=0,
            accuracy_delta=0.0,
            p_value=0.5,
            log_p_value=math.log(0.5),
        )
    with pytest.raises(ModelValidationError):
        AdjustedPValue(hypothesis_id="test", raw_p_value=0.3, adjusted_p_value=0.2)
    with pytest.raises(ModelValidationError):
        HolmCorrection(
            family_id="family",
            n=2,
            hypotheses=(AdjustedPValue(hypothesis_id="one", raw_p_value=0.2, adjusted_p_value=0.4),)
            * 2,
        )
    with pytest.raises(ModelValidationError):
        HolmCorrection.model_validate(
            {"family_id": "family", "n": 0, "hypotheses": (), "reason": "empty", "alpha": 0.05}
        )


def test_empty_and_nonempty_result_artifacts_cannot_exchange_undefined_values() -> None:
    with pytest.raises(ModelValidationError):
        ProportionInterval(successes=0, n=0, confidence_level=0.95, estimate=0.0, reason="empty")
    with pytest.raises(ModelValidationError):
        ProportionInterval(successes=1, n=2, confidence_level=0.95, estimate=0.3, low=0.1, high=0.9)
    with pytest.raises(ModelValidationError):
        McNemarResult(
            n=0,
            baseline_only_correct=0,
            candidate_only_correct=0,
            discordant=0,
            log_p_value=0.0,
            reason="empty",
        )
    with pytest.raises(ModelValidationError):
        McNemarResult(n=5, baseline_only_correct=2, candidate_only_correct=1, discordant=2)
    with pytest.raises(ModelValidationError):
        AdjustedPValue(hypothesis_id="not an id", raw_p_value=0.2, adjusted_p_value=0.4)


def test_comparisons_are_shared_evaluation_aliases() -> None:
    from strata_forge.evals import classification as evaluation

    assert evaluation.wilson_interval is wilson_interval
    assert evaluation.mcnemar_exact is mcnemar_exact
    assert evaluation.holm_correction is holm_correction
