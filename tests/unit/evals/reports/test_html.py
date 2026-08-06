"""Unit tests for `strata_forge.evals.reports.html`."""

from __future__ import annotations

from typing import Any

from strata_forge.evals.experiment import GraderResult, Outcome, Trial
from strata_forge.evals.reports.html import render_html
from strata_forge.llm.responses import Usage


def _trial(*, model: str = "m", response_text: str = "answer") -> Trial:
    return Trial(
        experiment_name="e",
        model=model,
        provider="p",
        item_id="item",
        response_text=response_text,
        usage=Usage(input_tokens=1, output_tokens=1),
        cost_usd=0.001,
        cache_hit=False,
        latency_ms=10.0,
    )


def _outcome(*results: tuple[str, float, bool, str], **trial_kwargs: Any) -> Outcome:
    return Outcome(
        trial=_trial(**trial_kwargs),
        grader_results=tuple(
            GraderResult(grader_name=name, score=score, passed=passed, explanation=expl)
            for name, score, passed, expl in results
        ),
    )


class TestDocumentStructure:
    def test_complete_html_document(self) -> None:
        out = render_html([_outcome(("g", 1.0, True, ""))])
        assert out.startswith("<!doctype html>")
        assert "<html" in out
        assert "</html>" in out
        assert "<style>" in out

    def test_empty_outcomes(self) -> None:
        out = render_html([])
        assert "No outcomes" in out
        assert "<h1>" in out

    def test_title_in_h1_and_head(self) -> None:
        out = render_html([_outcome(("g", 1.0, True, ""))], title="My Eval")
        assert "<title>My Eval</title>" in out
        assert "<h1>My Eval</h1>" in out


class TestSummaryRendering:
    def test_summary_table_when_outcomes(self) -> None:
        out = render_html([_outcome(("g", 1.0, True, ""))])
        assert "Summary" in out
        assert "Total trials" in out
        assert "Total cost" in out
        assert "Mean latency" in out

    def test_pass_rate_table_per_grader(self) -> None:
        out = render_html(
            [
                _outcome(("g1", 1.0, True, "")),
                _outcome(("g1", 0.0, False, "")),
            ]
        )
        assert "Pass rate by grader" in out
        # 50% pass rate for g1
        assert "50.0%" in out

    def test_per_model_breakdown(self) -> None:
        out = render_html(
            [
                _outcome(("g", 1.0, True, ""), model="m1"),
                _outcome(("g", 0.0, False, ""), model="m2"),
            ]
        )
        assert "m1" in out
        assert "m2" in out


class TestPerTrialCards:
    def test_response_included_in_card(self) -> None:
        out = render_html([_outcome(("g", 1.0, True, ""), response_text="the-answer")])
        assert "the-answer" in out

    def test_html_in_response_escaped(self) -> None:
        # XSS protection: any HTML in the response text must be escaped.
        out = render_html(
            [_outcome(("g", 1.0, True, ""), response_text="<script>alert('x')</script>")]
        )
        assert "<script>alert" not in out
        assert "&lt;script&gt;" in out

    def test_explanation_included(self) -> None:
        out = render_html([_outcome(("g", 1.0, True, "well done"))])
        assert "well done" in out

    def test_pass_class_on_passing_grader(self) -> None:
        out = render_html([_outcome(("g", 1.0, True, ""))])
        assert 'class="pass"' in out

    def test_fail_class_on_failing_grader(self) -> None:
        out = render_html([_outcome(("g", 0.0, False, ""))])
        assert 'class="fail"' in out

    def test_grader_name_in_html(self) -> None:
        out = render_html([_outcome(("my-special-grader", 1.0, True, ""))])
        assert "my-special-grader" in out
