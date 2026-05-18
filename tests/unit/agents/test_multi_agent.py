"""Unit tests for `forge.agents.multi_agent`."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from forge.agents import (
    Agent,
    AgentResult,
    CritiqueVerdict,
    RouterChoice,
    critic_refiner_run,
    handoff,
)
from forge.llm.client import StructuredResponse
from forge.llm.responses import LLMResponse, Usage
from forge.llm.routing import ModelRoute


def _route() -> ModelRoute:
    return ModelRoute(
        model="claude-opus-4-7",
        provider="anthropic",
        provider_model_id="claude-opus-4-7",
    )


def _llm_response(text: str = "ans") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1),
        cost_usd=0.0,
        route=_route(),
        latency_ms=1.0,
    )


def _agent_with_plain_response(text: str = "ans") -> Agent:
    """Build an Agent whose underlying client returns a fixed text response."""
    client = AsyncMock()
    client.complete = AsyncMock(return_value=_llm_response(text))
    client.run_tool_loop = AsyncMock(return_value=_llm_response(text))
    return Agent("x", client=client)


def _agent_with_structured_response(parsed: Any, text: str = "ans") -> Agent:
    """Build an Agent whose run_structured returns ``parsed``."""
    client = AsyncMock()
    sr = StructuredResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1),
        cost_usd=0.0,
        route=_route(),
        cache_hit=False,
        latency_ms=1.0,
        parsed=parsed,
    )
    client.complete_structured = AsyncMock(return_value=sr)
    client.complete = AsyncMock(return_value=_llm_response(text))
    return Agent("x", client=client)


# ---------------------------------------------------------------------------
# handoff
# ---------------------------------------------------------------------------


class TestHandoff:
    async def test_routes_to_chosen_specialist(self) -> None:
        router = _agent_with_structured_response(
            RouterChoice(specialist="math", reasoning="it's a math problem")
        )
        math_agent = _agent_with_plain_response("the math answer")
        text_agent = _agent_with_plain_response("the text answer")

        result = await handoff(
            router=router,
            specialists={"math": math_agent, "text": text_agent},
            user_input="what is 2 + 2?",
        )
        # The math specialist answered.
        assert result.text == "the math answer"
        # text_agent.run was not called.
        text_agent.client.complete.assert_not_called()  # type: ignore[attr-defined]

    async def test_empty_specialists_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            await handoff(
                router=_agent_with_structured_response(RouterChoice(specialist="x", reasoning="")),
                specialists={},
                user_input="hi",
            )

    async def test_unknown_specialist_rejected(self) -> None:
        router = _agent_with_structured_response(
            RouterChoice(specialist="nonexistent", reasoning="")
        )
        with pytest.raises(ValueError, match="nonexistent"):
            await handoff(
                router=router,
                specialists={"a": _agent_with_plain_response()},
                user_input="hi",
            )

    async def test_router_sees_catalog(self) -> None:
        router = _agent_with_structured_response(RouterChoice(specialist="alpha", reasoning=""))
        await handoff(
            router=router,
            specialists={
                "alpha": _agent_with_plain_response(),
                "beta": _agent_with_plain_response(),
            },
            user_input="hi",
        )
        # The routing prompt sent to the router includes both names.
        call_args = router.client.complete_structured.call_args  # type: ignore[attr-defined]
        messages = call_args.kwargs["messages"]
        user_msg = messages[-1].content
        assert "alpha" in user_msg
        assert "beta" in user_msg

    async def test_specialist_receives_original_input(self) -> None:
        router = _agent_with_structured_response(RouterChoice(specialist="x", reasoning=""))
        specialist = _agent_with_plain_response()
        await handoff(
            router=router,
            specialists={"x": specialist},
            user_input="please summarize Hamlet in two sentences",
        )
        call_args = specialist.client.complete.call_args  # type: ignore[attr-defined]
        messages = call_args.kwargs["messages"]
        # Last user message is the original input verbatim.
        assert messages[-1].content == "please summarize Hamlet in two sentences"


# ---------------------------------------------------------------------------
# critic_refiner_run
# ---------------------------------------------------------------------------


class TestCriticRefinerRun:
    async def test_returns_after_first_approval(self) -> None:
        drafter = _agent_with_plain_response("first draft")
        critic = _agent_with_structured_response(CritiqueVerdict(approved=True, feedback=""))

        result = await critic_refiner_run(
            drafter=drafter,
            critic=critic,
            user_input="write a haiku",
            max_rounds=3,
        )
        assert result.text == "first draft"
        # Drafter called exactly once.
        assert drafter.client.complete.call_count == 1  # type: ignore[attr-defined]
        # Critic called exactly once.
        assert critic.client.complete_structured.call_count == 1  # type: ignore[attr-defined]

    async def test_loops_until_approved(self) -> None:
        drafter = _agent_with_plain_response("draft")
        critic = AsyncMock(spec=Agent)
        critic.client = AsyncMock()
        # First two rounds: rejected; third: approved.
        critic.run_structured = AsyncMock(
            side_effect=[
                _make_agent_result(CritiqueVerdict(approved=False, feedback="needs more punch")),
                _make_agent_result(
                    CritiqueVerdict(approved=False, feedback="closer, but be specific")
                ),
                _make_agent_result(CritiqueVerdict(approved=True, feedback="")),
            ]
        )
        critic.name = "critic"

        result = await critic_refiner_run(
            drafter=drafter,
            critic=critic,
            user_input="write something",
            max_rounds=5,
        )
        # Drafter called 3 times (one per round).
        assert drafter.client.complete.call_count == 3  # type: ignore[attr-defined]
        # Critic called 3 times.
        assert critic.run_structured.call_count == 3
        # Final result is the third draft.
        assert result.text == "draft"

    async def test_max_rounds_returns_last_draft(self) -> None:
        drafter = _agent_with_plain_response("final draft")
        critic = AsyncMock(spec=Agent)
        critic.run_structured = AsyncMock(
            return_value=_make_agent_result(
                CritiqueVerdict(approved=False, feedback="never satisfied")
            )
        )

        result = await critic_refiner_run(
            drafter=drafter,
            critic=critic,
            user_input="x",
            max_rounds=2,
        )
        # Critic never approves; we get the last draft after 2 rounds.
        assert result.text == "final draft"
        assert drafter.client.complete.call_count == 2  # type: ignore[attr-defined]
        assert critic.run_structured.call_count == 2

    async def test_invalid_max_rounds_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_rounds"):
            await critic_refiner_run(
                drafter=_agent_with_plain_response(),
                critic=_agent_with_structured_response(CritiqueVerdict(approved=True)),
                user_input="x",
                max_rounds=0,
            )

    async def test_feedback_passed_to_drafter_on_refinement(self) -> None:
        drafter = _agent_with_plain_response("refined")
        critic = AsyncMock(spec=Agent)
        critic.run_structured = AsyncMock(
            side_effect=[
                _make_agent_result(CritiqueVerdict(approved=False, feedback="add more detail")),
                _make_agent_result(CritiqueVerdict(approved=True)),
            ]
        )

        await critic_refiner_run(
            drafter=drafter,
            critic=critic,
            user_input="describe a cat",
            max_rounds=3,
        )
        # On the second call, the drafter sees the previous draft + critique.
        second_call = drafter.client.complete.call_args_list[1]  # type: ignore[attr-defined]
        messages = second_call.kwargs["messages"]
        # The user message includes the previous draft and the feedback.
        user_content = messages[-1].content
        assert "add more detail" in user_content
        assert "describe a cat" in user_content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_result(parsed: Any) -> AgentResult:
    """Wrap ``parsed`` in an AgentResult so AsyncMock side_effects look like real returns."""
    return AgentResult(
        text="dummy",
        messages=(),
        final_response=_llm_response(),
        parsed=parsed,
    )
