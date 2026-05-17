"""Unit tests for `forge.evals.graders.pairwise`."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import AsyncMock

from forge.datasets.schema import DatasetItem
from forge.evals.graders.pairwise import PairwiseGrader, PairwiseVerdict
from forge.llm.responses import LLMResponse, Usage
from forge.llm.routing import ModelRoute


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1),
        cost_usd=0.0,
        route=ModelRoute(
            model="claude-opus-4-7",
            provider="anthropic",
            provider_model_id="claude-opus-4-7",
        ),
    )


def _item(*, expected: Any = None) -> DatasetItem:
    return DatasetItem(id="a", input={"q": "x"}, expected_output=expected)


def _judge_client(*, winner: Literal["A", "B", "tie"], reasoning: str = "ok") -> AsyncMock:
    client = AsyncMock()
    verdict_resp = AsyncMock()
    verdict_resp.parsed = PairwiseVerdict(winner=winner, reasoning=reasoning)
    verdict_resp.cost_usd = 0.0
    client.complete_structured = AsyncMock(return_value=verdict_resp)
    return client


# ---------------------------------------------------------------------------
# Candidate wins / loses / ties
# ---------------------------------------------------------------------------


class TestVerdictMapping:
    async def test_candidate_wins_when_slot_matches(self) -> None:
        # No randomization → candidate goes into slot A; judge picks A
        # → candidate wins.
        client = _judge_client(winner="A", reasoning="candidate is better")
        grader = PairwiseGrader(client, criteria="x", randomize_positions=False)
        result = await grader.grade(
            item=_item(expected="reference"),
            response=_response("candidate"),
        )
        assert result.passed is True
        assert result.score == 1.0
        assert result.explanation == "candidate is better"
        assert result.metadata["candidate_slot"] == "A"
        assert result.metadata["winner"] == "A"

    async def test_reference_wins_when_judge_picks_other_slot(self) -> None:
        # No randomization → reference goes into slot B; judge picks B
        # → reference wins.
        client = _judge_client(winner="B", reasoning="reference is better")
        grader = PairwiseGrader(client, criteria="x", randomize_positions=False)
        result = await grader.grade(
            item=_item(expected="reference"),
            response=_response("candidate"),
        )
        assert result.passed is False
        assert result.score == 0.0

    async def test_tie_scores_half_and_passes(self) -> None:
        client = _judge_client(winner="tie", reasoning="equivalent")
        grader = PairwiseGrader(client, criteria="x", randomize_positions=False)
        result = await grader.grade(
            item=_item(expected="reference"),
            response=_response("candidate"),
        )
        assert result.passed is True
        assert result.score == 0.5


# ---------------------------------------------------------------------------
# Position randomization
# ---------------------------------------------------------------------------


class TestPositionRandomization:
    async def test_randomization_can_swap_slots(self) -> None:
        # With randomize_positions=True, the candidate may end up in
        # slot B. Run many grades and confirm both slot assignments
        # appear (probabilistically — secrets.randbelow is uniform).
        client = _judge_client(winner="A")
        grader = PairwiseGrader(client, criteria="x", randomize_positions=True)
        slots: set[str] = set()
        for _ in range(50):
            result = await grader.grade(
                item=_item(expected="reference"), response=_response("candidate")
            )
            slots.add(result.metadata["candidate_slot"])
            if len(slots) == 2:
                break
        # With 50 trials at 50/50 probability, seeing only one slot is
        # ~1 in 2^49 — effectively zero. If we see both, randomization
        # works.
        assert slots == {"A", "B"}

    async def test_pinned_candidate_is_slot_a(self) -> None:
        client = _judge_client(winner="A")
        grader = PairwiseGrader(client, criteria="x", randomize_positions=False)
        result = await grader.grade(
            item=_item(expected="reference"), response=_response("candidate")
        )
        assert result.metadata["candidate_slot"] == "A"


# ---------------------------------------------------------------------------
# Missing expected_output
# ---------------------------------------------------------------------------


class TestMissingExpected:
    async def test_returns_failed_when_no_expected(self) -> None:
        # No expected_output → grader can't compare; fails with explanation.
        client = AsyncMock()
        grader = PairwiseGrader(client, criteria="x")
        result = await grader.grade(item=_item(expected=None), response=_response("candidate"))
        assert result.passed is False
        assert result.score == 0.0
        assert "no expected_output" in result.explanation
        # Judge was never invoked.
        client.complete_structured.assert_not_called()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    async def test_both_answers_appear_in_user_message(self) -> None:
        client = _judge_client(winner="A")
        grader = PairwiseGrader(client, criteria="my-criteria", randomize_positions=False)
        await grader.grade(
            item=_item(expected="reference-text"),
            response=_response("candidate-text"),
        )
        messages = client.complete_structured.call_args.kwargs["messages"]
        user_msg = messages[1].content
        assert "candidate-text" in user_msg
        assert "reference-text" in user_msg
        assert "my-criteria" in user_msg

    async def test_custom_system_prompt(self) -> None:
        client = _judge_client(winner="A")
        grader = PairwiseGrader(
            client,
            criteria="x",
            system_prompt="my-system",
            randomize_positions=False,
        )
        await grader.grade(item=_item(expected="r"), response=_response("c"))
        messages = client.complete_structured.call_args.kwargs["messages"]
        assert messages[0].content == "my-system"

    async def test_custom_user_template(self) -> None:
        client = _judge_client(winner="A")
        grader = PairwiseGrader(
            client,
            criteria="x",
            user_template="A={answer_a} | B={answer_b}",
            randomize_positions=False,
        )
        await grader.grade(item=_item(expected="ref"), response=_response("cand"))
        messages = client.complete_structured.call_args.kwargs["messages"]
        # Slot A is candidate when randomize_positions=False
        assert "A=cand" in messages[1].content
        assert "B=ref" in messages[1].content


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


class TestNaming:
    def test_default_name(self) -> None:
        grader = PairwiseGrader(AsyncMock(), criteria="x")
        assert grader.name == "pairwise"

    def test_custom_name(self) -> None:
        grader = PairwiseGrader(AsyncMock(), criteria="x", name="my-pairwise")
        assert grader.name == "my-pairwise"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestPairwiseVerdict:
    def test_valid_winner_values_accepted(self) -> None:
        for winner in ("A", "B", "tie"):
            v = PairwiseVerdict(winner=winner, reasoning="x")  # type: ignore[arg-type]
            assert v.winner == winner

    def test_invalid_winner_rejected(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PairwiseVerdict(winner="other", reasoning="x")  # type: ignore[arg-type]
