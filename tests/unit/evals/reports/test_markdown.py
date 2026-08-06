"""Unit tests for `strata_forge.evals.reports.markdown`."""

from __future__ import annotations

from typing import Any

from strata_forge.evals.experiment import GraderResult, Outcome, Trial
from strata_forge.evals.reports.markdown import render_markdown
from strata_forge.llm.responses import Usage


def _trial(
    *,
    model: str = "m",
    item_id: str = "item-1",
    response_text: str = "answer",
    cost_usd: float = 0.001,
    latency_ms: float = 50.0,
) -> Trial:
    return Trial(
        experiment_name="e",
        model=model,
        provider="p",
        item_id=item_id,
        response_text=response_text,
        usage=Usage(input_tokens=1, output_tokens=1),
        cost_usd=cost_usd,
        cache_hit=False,
        latency_ms=latency_ms,
    )


def _outcome(*results: tuple[str, float, bool, str], **trial_kwargs: Any) -> Outcome:
    return Outcome(
        trial=_trial(**trial_kwargs),
        grader_results=tuple(
            GraderResult(grader_name=name, score=score, passed=passed, explanation=expl)
            for name, score, passed, expl in results
        ),
    )


class TestSummarySection:
    def test_empty_outcomes(self) -> None:
        out = render_markdown([])
        assert "No outcomes" in out

    def test_total_trials_in_summary(self) -> None:
        outcomes = [_outcome(("g", 1.0, True, "")), _outcome(("g", 0.0, False, ""))]
        out = render_markdown(outcomes)
        assert "Total trials" in out
        assert "2" in out

    def test_total_cost_summed(self) -> None:
        outcomes = [
            _outcome(("g", 1.0, True, ""), cost_usd=0.005),
            _outcome(("g", 1.0, True, ""), cost_usd=0.010),
        ]
        out = render_markdown(outcomes)
        assert "0.015" in out

    def test_mean_latency(self) -> None:
        outcomes = [
            _outcome(("g", 1.0, True, ""), latency_ms=100.0),
            _outcome(("g", 1.0, True, ""), latency_ms=200.0),
        ]
        out = render_markdown(outcomes)
        assert "150.0" in out

    def test_title_used(self) -> None:
        outcomes = [_outcome(("g", 1.0, True, ""))]
        out = render_markdown(outcomes, title="Custom Title")
        assert "# Custom Title" in out


class TestPassRateTables:
    def test_grader_pass_rate_table(self) -> None:
        outcomes = [
            _outcome(("g1", 1.0, True, ""), ("g2", 0.0, False, "")),
            _outcome(("g1", 1.0, True, ""), ("g2", 1.0, True, "")),
        ]
        out = render_markdown(outcomes)
        assert "Pass rate by grader" in out
        assert "100.0%" in out  # g1 perfect
        assert "50.0%" in out  # g2 half

    def test_per_model_breakdown(self) -> None:
        outcomes = [
            _outcome(("g", 1.0, True, ""), model="m1"),
            _outcome(("g", 0.0, False, ""), model="m2"),
        ]
        out = render_markdown(outcomes)
        assert "model" in out.lower()
        assert "m1" in out
        assert "m2" in out


class TestTrialsTable:
    def test_response_truncation(self) -> None:
        long_response = "x" * 200
        outcome = _outcome(("g", 1.0, True, ""), response_text=long_response)
        out = render_markdown([outcome], response_truncate=20)
        # The truncated response carries ellipsis but not the full text.
        assert "…" in out
        assert "x" * 200 not in out

    def test_pipe_in_response_escaped(self) -> None:
        outcome = _outcome(("g", 1.0, True, ""), response_text="has | pipe")
        out = render_markdown([outcome])
        # Pipe-in-content would break the table; we escape it.
        assert "has \\| pipe" in out

    def test_newlines_in_response_collapsed(self) -> None:
        outcome = _outcome(("g", 1.0, True, ""), response_text="line1\nline2")
        out = render_markdown([outcome])
        # Newlines would break the table; we collapse them.
        assert "line1 line2" in out

    def test_pass_glyph_for_passed(self) -> None:
        outcomes = [_outcome(("g", 1.0, True, ""))]
        out = render_markdown(outcomes)
        # Find the trials section and inspect the row.
        assert "✓" in out

    def test_fail_glyph_for_failed(self) -> None:
        outcomes = [_outcome(("g", 0.0, False, ""))]
        out = render_markdown(outcomes)
        assert "✗" in out


class TestFailuresSection:
    def test_failures_section_appears(self) -> None:
        outcomes = [
            _outcome(("g", 1.0, True, "")),
            _outcome(("g", 0.0, False, "this is why it failed")),
        ]
        out = render_markdown(outcomes)
        assert "## Failures" in out
        assert "this is why it failed" in out

    def test_failures_section_omitted_when_all_pass(self) -> None:
        outcomes = [_outcome(("g", 1.0, True, ""))]
        out = render_markdown(outcomes)
        assert "## Failures" not in out

    def test_failures_section_can_be_disabled(self) -> None:
        outcomes = [_outcome(("g", 0.0, False, "x"))]
        out = render_markdown(outcomes, include_failures=False)
        assert "## Failures" not in out

    def test_failure_includes_grader_name(self) -> None:
        outcomes = [
            _outcome(("the_grader", 0.2, False, "reason")),
        ]
        out = render_markdown(outcomes)
        assert "the_grader" in out
        assert "reason" in out

    def test_failure_handles_empty_explanation(self) -> None:
        outcomes = [_outcome(("g", 0.0, False, ""))]
        out = render_markdown(outcomes)
        assert "(no explanation)" in out
