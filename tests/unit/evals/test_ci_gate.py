"""Unit tests for `forge.evals.ci_gate`."""

from __future__ import annotations

from typing import Any

import pytest

from forge.evals.ci_gate import (
    CIGateThresholds,
    evaluate_ci_gate,
    wilson_lower_bound,
)
from forge.evals.experiment import GraderResult, Outcome, Trial
from forge.llm.responses import Usage


def _trial(*, model: str = "m", cost_usd: float = 0.001) -> Trial:
    return Trial(
        experiment_name="e",
        model=model,
        provider="p",
        item_id="i",
        response_text="x",
        usage=Usage(input_tokens=1, output_tokens=1),
        cost_usd=cost_usd,
        cache_hit=False,
        latency_ms=1.0,
    )


def _outcome(*results: tuple[str, float, bool], **trial_kwargs: Any) -> Outcome:
    return Outcome(
        trial=_trial(**trial_kwargs),
        grader_results=tuple(
            GraderResult(grader_name=name, score=score, passed=passed)
            for name, score, passed in results
        ),
    )


# ---------------------------------------------------------------------------
# wilson_lower_bound
# ---------------------------------------------------------------------------


class TestWilsonLowerBound:
    def test_all_passed_high_sample(self) -> None:
        # 100/100 → very high lower bound (close to 1 but not 1.0)
        lb = wilson_lower_bound(100, 100)
        assert 0.96 < lb < 1.0

    def test_all_passed_small_sample(self) -> None:
        # 5/5 → much lower Wilson bound; small samples are penalized.
        lb = wilson_lower_bound(5, 5)
        assert lb < 0.7

    def test_all_failed(self) -> None:
        lb = wilson_lower_bound(0, 100)
        assert lb == 0.0

    def test_zero_total_returns_zero(self) -> None:
        # Defensive default — undefined statistically.
        assert wilson_lower_bound(0, 0) == 0.0
        assert wilson_lower_bound(5, 0) == 0.0
        assert wilson_lower_bound(0, -1) == 0.0

    def test_half_passing(self) -> None:
        # 50/100 → Wilson lower bound is below 0.5.
        lb = wilson_lower_bound(50, 100)
        assert lb < 0.5
        assert lb > 0.3

    def test_z_parameter_changes_result(self) -> None:
        # 99% CI (z=2.576) should give a tighter (lower) bound than 95% CI.
        ci95 = wilson_lower_bound(50, 100, z=1.96)
        ci99 = wilson_lower_bound(50, 100, z=2.576)
        assert ci99 < ci95


# ---------------------------------------------------------------------------
# CIGateThresholds shape
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_defaults(self) -> None:
        t = CIGateThresholds()
        assert t.min_pass_rate == 0.85
        assert t.max_cost_usd == 1.0
        assert t.use_wilson_ci is True
        assert t.per_grader_pass_rate == {}

    def test_is_frozen(self) -> None:
        from pydantic import ValidationError

        t = CIGateThresholds()
        with pytest.raises(ValidationError, match="frozen"):
            t.min_pass_rate = 0.5  # type: ignore[misc]

    def test_invalid_min_pass_rate_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CIGateThresholds(min_pass_rate=1.5)
        with pytest.raises(ValidationError):
            CIGateThresholds(min_pass_rate=-0.1)


# ---------------------------------------------------------------------------
# evaluate_ci_gate
# ---------------------------------------------------------------------------


class TestEvaluateGate:
    def test_empty_outcomes_fails(self) -> None:
        result = evaluate_ci_gate([])
        assert result.passed is False
        assert any("no outcomes" in i for i in result.issues)

    def test_all_passing_above_threshold_passes(self) -> None:
        # 50 trials → Wilson lower bound ~0.93 for 100% point estimate,
        # comfortably above the 0.85 default threshold.
        outcomes = [_outcome(("g", 1.0, True))] * 50
        result = evaluate_ci_gate(
            outcomes,
            thresholds=CIGateThresholds(min_pass_rate=0.85),
        )
        assert result.passed is True
        assert result.issues == ()

    def test_low_pass_rate_fails(self) -> None:
        outcomes = [_outcome(("g", 1.0, True))] * 5 + [_outcome(("g", 0.0, False))] * 15
        result = evaluate_ci_gate(
            outcomes,
            thresholds=CIGateThresholds(min_pass_rate=0.85, use_wilson_ci=False),
        )
        assert result.passed is False
        # The issue mentions which grader failed.
        assert any("g" in i for i in result.issues)

    def test_wilson_ci_penalizes_small_samples(self) -> None:
        # 5/5 passing — point estimate is 1.0 but Wilson lower is ~0.57.
        outcomes = [_outcome(("g", 1.0, True))] * 5
        result_point = evaluate_ci_gate(
            outcomes,
            thresholds=CIGateThresholds(min_pass_rate=0.85, use_wilson_ci=False),
        )
        result_wilson = evaluate_ci_gate(
            outcomes,
            thresholds=CIGateThresholds(min_pass_rate=0.85, use_wilson_ci=True),
        )
        # Point estimate passes (1.0 >= 0.85).
        assert result_point.passed is True
        # Wilson fails (0.57 < 0.85).
        assert result_wilson.passed is False

    def test_cost_cap_failure(self) -> None:
        outcomes = [_outcome(("g", 1.0, True), cost_usd=0.5)] * 10
        # Total cost = 5.0, cap = 1.0.
        result = evaluate_ci_gate(
            outcomes,
            thresholds=CIGateThresholds(max_cost_usd=1.0),
        )
        assert result.passed is False
        assert any("cost" in i for i in result.issues)
        assert result.total_cost_usd == 5.0

    def test_per_grader_threshold_override(self) -> None:
        # g1 passes 8/10 (above 0.7 but below 0.95 specific threshold).
        # g2 passes 9/10 (above default 0.85).
        outcomes = []
        for i in range(10):
            outcomes.append(
                _outcome(
                    ("g1", 1.0 if i < 8 else 0.0, i < 8),
                    ("g2", 1.0 if i < 9 else 0.0, i < 9),
                )
            )
        result = evaluate_ci_gate(
            outcomes,
            thresholds=CIGateThresholds(
                min_pass_rate=0.7,
                per_grader_pass_rate={"g1": 0.95},
                use_wilson_ci=False,
            ),
        )
        # g1 fails its specific 0.95 threshold; g2 clears default 0.7.
        assert result.passed is False
        assert any("'g1'" in i for i in result.issues)
        assert not any("'g2'" in i for i in result.issues)

    def test_per_grader_via_call_kwarg_merges(self) -> None:
        # The kwarg overrides match the per_grader_pass_rate from thresholds.
        outcomes = [_outcome(("g", 1.0, True))] * 20
        # Default threshold is 0.85; we override to 0.99 just for g.
        result = evaluate_ci_gate(
            outcomes,
            thresholds=CIGateThresholds(use_wilson_ci=False),
            per_grader_pass_rate={"g": 0.99},
        )
        # Point estimate is 1.0 → still passes 0.99.
        assert result.passed is True

    def test_total_cost_returned(self) -> None:
        outcomes = [_outcome(("g", 1.0, True), cost_usd=0.005)] * 4
        result = evaluate_ci_gate(outcomes)
        assert abs(result.total_cost_usd - 0.020) < 1e-9
        assert result.total_trials == 4

    def test_pass_rates_returned(self) -> None:
        outcomes = [_outcome(("g", 1.0, True))] * 7 + [_outcome(("g", 0.0, False))] * 3
        result = evaluate_ci_gate(outcomes)
        assert abs(result.pass_rates_by_grader["g"] - 0.7) < 1e-9
        assert "g" in result.wilson_lower_bounds_by_grader

    def test_unknown_grader_in_thresholds_surfaces(self) -> None:
        outcomes = [_outcome(("g", 1.0, True))] * 20
        result = evaluate_ci_gate(
            outcomes,
            thresholds=CIGateThresholds(per_grader_pass_rate={"typo_grader": 0.9}),
        )
        # A typo in the grader threshold dict should surface, not silently pass.
        assert result.passed is False
        assert any("typo_grader" in i for i in result.issues)


# ---------------------------------------------------------------------------
# CIGateResult shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_is_frozen(self) -> None:
        from pydantic import ValidationError

        result = evaluate_ci_gate([_outcome(("g", 1.0, True))])
        with pytest.raises(ValidationError, match="frozen"):
            result.passed = False  # type: ignore[misc]

    def test_issues_is_tuple(self) -> None:
        result = evaluate_ci_gate([])
        assert isinstance(result.issues, tuple)
