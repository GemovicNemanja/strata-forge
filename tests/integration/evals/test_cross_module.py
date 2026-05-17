"""Cross-module evaluation workflows that don't need external services.

Each test wires multiple :mod:`forge.evals` pieces together —
runner + graders + metrics, runner + reports, runner + CI gate —
to confirm the seams hold. Unit tests cover each piece in isolation;
these tests prove the pieces compose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from forge.datasets.schema import Dataset, DatasetItem
from forge.evals.ci_gate import CIGateThresholds, evaluate_ci_gate
from forge.evals.experiment import Experiment, SamplingParams
from forge.evals.graders.exact import ExactMatch
from forge.evals.metrics import pass_rate, pass_rate_by_grader
from forge.evals.reports.html import render_html
from forge.evals.reports.markdown import render_markdown
from forge.evals.runner import run_experiment
from forge.llm.responses import LLMResponse, Usage
from forge.llm.routing import ModelRoute

if TYPE_CHECKING:
    from forge.llm.registry import ProviderName


def _response(text: str, *, provider: ProviderName = "anthropic") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(input_tokens=5, output_tokens=3),
        cost_usd=0.001,
        route=ModelRoute(
            model="claude-opus-4-7",
            provider=provider,
            provider_model_id="claude-opus-4-7",
        ),
    )


def _client(*texts: str) -> AsyncMock:
    client = AsyncMock()
    client.complete = AsyncMock(side_effect=[_response(t) for t in texts])
    return client


def _dataset() -> Dataset:
    return Dataset(
        name="qa",
        items=(
            DatasetItem.from_input({"q": "what is 2+2?"}, expected_output="4"),
            DatasetItem.from_input({"q": "capital of france?"}, expected_output="paris"),
        ),
    )


def _experiment() -> Experiment:
    return Experiment(
        name="cross-module",
        models=("claude-opus-4-7",),
        dataset_name="qa",
        grader_names=("exact_match",),
        sampling=SamplingParams(temperature=0.0),
    )


# ---------------------------------------------------------------------------
# Runner → metrics
# ---------------------------------------------------------------------------


class TestRunnerThroughMetrics:
    async def test_full_pass_metric(self) -> None:
        # Both responses match the references → 100% pass rate.
        client = _client("4", "paris")
        outcomes = await run_experiment(
            _experiment(),
            dataset=_dataset(),
            clients={"claude-opus-4-7": client},
            graders=[ExactMatch(case_sensitive=False)],
        )
        assert pass_rate(outcomes, grader_name="exact_match") == 1.0
        assert pass_rate_by_grader(outcomes) == {"exact_match": 1.0}

    async def test_half_pass_metric(self) -> None:
        client = _client("4", "London")  # second is wrong
        outcomes = await run_experiment(
            _experiment(),
            dataset=_dataset(),
            clients={"claude-opus-4-7": client},
            graders=[ExactMatch(case_sensitive=False)],
        )
        assert pass_rate(outcomes, grader_name="exact_match") == 0.5


# ---------------------------------------------------------------------------
# Runner → reports
# ---------------------------------------------------------------------------


class TestRunnerThroughReports:
    async def test_markdown_report_renders(self) -> None:
        client = _client("4", "paris")
        outcomes = await run_experiment(
            _experiment(),
            dataset=_dataset(),
            clients={"claude-opus-4-7": client},
            graders=[ExactMatch(case_sensitive=False)],
        )
        md = render_markdown(outcomes, title="Cross-module test")
        assert "Cross-module test" in md
        assert "Total trials" in md
        assert "100.0%" in md  # 2/2 pass rate

    async def test_html_report_renders(self) -> None:
        client = _client("4", "wrong")
        outcomes = await run_experiment(
            _experiment(),
            dataset=_dataset(),
            clients={"claude-opus-4-7": client},
            graders=[ExactMatch(case_sensitive=False)],
        )
        html = render_html(outcomes)
        assert html.startswith("<!doctype html>")
        assert 'class="pass"' in html  # the "4" trial passed
        assert 'class="fail"' in html  # the "wrong" trial failed


# ---------------------------------------------------------------------------
# Runner → CI gate
# ---------------------------------------------------------------------------


class TestRunnerThroughCIGate:
    async def test_gate_passes_with_perfect_run(self) -> None:
        # Use enough trials to clear the Wilson lower bound at 100% pass.
        items = tuple(
            DatasetItem.from_input({"q": str(i)}, expected_output=str(i)) for i in range(60)
        )
        dataset = Dataset(name="big", items=items)
        client = AsyncMock()
        client.complete = AsyncMock(side_effect=[_response(str(i)) for i in range(60)])
        experiment = Experiment(
            name="big-run",
            models=("claude-opus-4-7",),
            dataset_name="big",
            grader_names=("exact_match",),
        )
        outcomes = await run_experiment(
            experiment,
            dataset=dataset,
            clients={"claude-opus-4-7": client},
            graders=[ExactMatch(case_sensitive=False)],
        )
        result = evaluate_ci_gate(
            outcomes,
            thresholds=CIGateThresholds(min_pass_rate=0.85),
        )
        assert result.passed is True

    async def test_gate_fails_on_low_pass_rate(self) -> None:
        # 1/2 pass — point estimate 0.5 well below 0.85.
        client = _client("4", "wrong")
        outcomes = await run_experiment(
            _experiment(),
            dataset=_dataset(),
            clients={"claude-opus-4-7": client},
            graders=[ExactMatch(case_sensitive=False)],
        )
        result = evaluate_ci_gate(
            outcomes,
            thresholds=CIGateThresholds(min_pass_rate=0.85, use_wilson_ci=False),
        )
        assert result.passed is False


# ---------------------------------------------------------------------------
# Empty / boundary
# ---------------------------------------------------------------------------


class TestEmptyDatasetAcrossPipeline:
    async def test_empty_dataset_runs_cleanly(self) -> None:
        client = AsyncMock()
        client.complete = AsyncMock()
        outcomes = await run_experiment(
            _experiment(),
            dataset=Dataset(name="empty"),
            clients={"claude-opus-4-7": client},
            graders=[ExactMatch()],
        )
        assert outcomes == ()
        client.complete.assert_not_called()
        # Reports and CI gate must handle empty outcomes gracefully.
        assert "No outcomes" in render_markdown(outcomes)
        assert "No outcomes" in render_html(outcomes)
        result = evaluate_ci_gate(outcomes)
        assert result.passed is False
        assert any("no outcomes" in i for i in result.issues)


_unused_any: Any = None  # keep `Any` import used
