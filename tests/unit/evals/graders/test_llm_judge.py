"""Unit tests for `strata_forge.evals.graders.llm_judge`."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ConfigDict

from strata_forge.datasets.schema import DatasetItem
from strata_forge.evals.graders.llm_judge import JudgeVerdict, LLMJudge
from strata_forge.llm.responses import LLMResponse, Usage
from strata_forge.llm.routing import ModelRoute


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(input_tokens=10, output_tokens=5),
        cost_usd=0.0,
        route=ModelRoute(
            model="claude-opus-4-7",
            provider="anthropic",
            provider_model_id="claude-opus-4-7",
        ),
    )


def _item(*, expected: Any = None, input_dict: dict[str, Any] | None = None) -> DatasetItem:
    return DatasetItem(
        id="a",
        input=input_dict or {"q": "what is 2+2?"},
        expected_output=expected,
    )


def _judge_client(*, score: float, reasoning: str = "ok") -> AsyncMock:
    """Build a mock LLMClient whose complete_structured returns the verdict."""
    client = AsyncMock()
    verdict_response = AsyncMock()
    verdict_response.parsed = JudgeVerdict(score=score, reasoning=reasoning)
    verdict_response.cost_usd = 0.0001
    verdict_response.latency_ms = 5.0
    client.complete_structured = AsyncMock(return_value=verdict_response)
    return client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    async def test_high_score_passes(self) -> None:
        client = _judge_client(score=0.9, reasoning="excellent answer")
        grader = LLMJudge(client, criteria="accuracy", pass_threshold=0.7)
        result = await grader.grade(item=_item(expected="4"), response=_response("4"))
        assert result.passed is True
        assert result.score == 0.9
        assert result.explanation == "excellent answer"
        assert result.grader_name == "llm_judge"

    async def test_low_score_fails(self) -> None:
        client = _judge_client(score=0.3, reasoning="wrong answer")
        grader = LLMJudge(client, criteria="accuracy", pass_threshold=0.7)
        result = await grader.grade(item=_item(expected="4"), response=_response("5"))
        assert result.passed is False
        assert result.score == 0.3

    async def test_threshold_boundary(self) -> None:
        # Score exactly at threshold counts as passing.
        client = _judge_client(score=0.7)
        grader = LLMJudge(client, criteria="x", pass_threshold=0.7)
        result = await grader.grade(item=_item(expected="x"), response=_response("y"))
        assert result.passed is True

    async def test_judge_cost_and_latency_in_metadata(self) -> None:
        client = _judge_client(score=0.8)
        grader = LLMJudge(client, criteria="x")
        result = await grader.grade(item=_item(expected="x"), response=_response("y"))
        assert result.metadata["judge_cost_usd"] == 0.0001
        assert result.metadata["judge_latency_ms"] == 5.0

    async def test_custom_name(self) -> None:
        grader = LLMJudge(_judge_client(score=0.9), criteria="x", name="my-judge")
        assert grader.name == "my-judge"
        result = await grader.grade(item=_item(expected="x"), response=_response("y"))
        assert result.grader_name == "my-judge"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    async def test_default_prompt_includes_criteria_and_response(self) -> None:
        client = _judge_client(score=1.0)
        grader = LLMJudge(client, criteria="must be exactly correct")
        await grader.grade(
            item=_item(expected="4", input_dict={"q": "2+2"}),
            response=_response("4"),
        )
        messages = client.complete_structured.call_args.kwargs["messages"]
        assert len(messages) == 2
        user_msg = messages[1].content
        assert "must be exactly correct" in user_msg
        assert "2+2" in user_msg
        assert "4" in user_msg

    async def test_missing_expected_rendered_as_none_sentinel(self) -> None:
        client = _judge_client(score=0.5)
        grader = LLMJudge(client, criteria="x")
        await grader.grade(item=_item(expected=None), response=_response("y"))
        messages = client.complete_structured.call_args.kwargs["messages"]
        assert "(none)" in messages[1].content

    async def test_custom_system_prompt(self) -> None:
        client = _judge_client(score=0.9)
        grader = LLMJudge(client, criteria="x", system_prompt="my-custom-system")
        await grader.grade(item=_item(expected="x"), response=_response("y"))
        messages = client.complete_structured.call_args.kwargs["messages"]
        assert messages[0].content == "my-custom-system"

    async def test_custom_user_template(self) -> None:
        client = _judge_client(score=0.9)
        grader = LLMJudge(
            client,
            criteria="x",
            user_template="Criteria: {criteria}\nResponse: {response}",
        )
        await grader.grade(item=_item(expected="ref"), response=_response("got"))
        messages = client.complete_structured.call_args.kwargs["messages"]
        assert "Criteria: x" in messages[1].content
        assert "Response: got" in messages[1].content

    async def test_schema_is_judge_verdict_by_default(self) -> None:
        client = _judge_client(score=0.9)
        grader = LLMJudge(client, criteria="x")
        await grader.grade(item=_item(expected="x"), response=_response("y"))
        kwargs = client.complete_structured.call_args.kwargs
        assert kwargs["schema"] is JudgeVerdict

    async def test_custom_schema_passed_through(self) -> None:
        class _CustomVerdict(BaseModel):
            model_config = ConfigDict(extra="forbid")
            score: float
            reasoning: str
            extra_field: str = "default"

        client = AsyncMock()
        verdict_resp = AsyncMock()
        verdict_resp.parsed = _CustomVerdict(score=0.8, reasoning="ok")
        verdict_resp.cost_usd = 0.0
        verdict_resp.latency_ms = 0.0
        client.complete_structured = AsyncMock(return_value=verdict_resp)
        grader = LLMJudge(client, criteria="x", verdict_schema=_CustomVerdict)
        result = await grader.grade(item=_item(expected="x"), response=_response("y"))
        assert result.score == 0.8


# ---------------------------------------------------------------------------
# Threshold validation
# ---------------------------------------------------------------------------


class TestThresholdValidation:
    def test_negative_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="pass_threshold"):
            LLMJudge(AsyncMock(), criteria="x", pass_threshold=-0.1)

    def test_threshold_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="pass_threshold"):
            LLMJudge(AsyncMock(), criteria="x", pass_threshold=1.5)

    def test_pass_threshold_property_exposed(self) -> None:
        grader = LLMJudge(AsyncMock(), criteria="x", pass_threshold=0.5)
        assert grader.pass_threshold == 0.5


# ---------------------------------------------------------------------------
# Verdict validation
# ---------------------------------------------------------------------------


class TestJudgeVerdict:
    def test_score_in_zero_one_accepted(self) -> None:
        v = JudgeVerdict(score=0.5, reasoning="x")
        assert v.score == 0.5

    def test_score_below_zero_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JudgeVerdict(score=-0.1, reasoning="x")

    def test_score_above_one_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JudgeVerdict(score=1.5, reasoning="x")
