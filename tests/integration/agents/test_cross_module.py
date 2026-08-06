"""Cross-module agent workflows that don't need external services.

Each test wires multiple :mod:`strata_forge.agents` pieces together so the
seams hold under realistic compositions: agent + built-in tool,
agent + memory, two agents in a critic-refiner loop, and so on.
Unit tests cover each piece in isolation; these tests prove the
pieces compose.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from strata_forge.agents import (
    Agent,
    AgentResult,
    ConversationMemory,
    CritiqueVerdict,
    EpisodicMemory,
    InMemoryVectorStore,
    RouterChoice,
    SystemMessage,
    UserMessage,
    calculator,
    critic_refiner_run,
    handoff,
)
from strata_forge.llm.client import StructuredResponse
from strata_forge.llm.responses import LLMResponse, Usage
from strata_forge.llm.routing import ModelRoute


def _route() -> ModelRoute:
    return ModelRoute(
        model="claude-opus-4-7",
        provider="anthropic",
        provider_model_id="claude-opus-4-7",
    )


def _response(text: str = "answer") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(input_tokens=10, output_tokens=5),
        cost_usd=0.001,
        route=_route(),
        latency_ms=10.0,
    )


def _structured(parsed: Any, text: str = "x") -> StructuredResponse[Any]:
    return StructuredResponse(
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


def _plain_client(text: str = "answer") -> AsyncMock:
    client = AsyncMock()
    client.complete = AsyncMock(return_value=_response(text))
    client.run_tool_loop = AsyncMock(return_value=_response(text))
    return client


def _structured_client(parsed: Any, text: str = "x") -> AsyncMock:
    client = AsyncMock()
    client.complete_structured = AsyncMock(return_value=_structured(parsed, text))
    client.complete = AsyncMock(return_value=_response(text))
    return client


# ---------------------------------------------------------------------------
# Agent + tool
# ---------------------------------------------------------------------------


class TestAgentWithTool:
    async def test_agent_with_calculator_uses_run_tool_loop(self) -> None:
        client = _plain_client("4")
        agent = Agent("math", client=client, tools=[calculator])
        result = await agent.run("what is 2 + 2?")
        # Routed through run_tool_loop because tools are present.
        client.run_tool_loop.assert_called_once()
        client.complete.assert_not_called()
        assert result.text == "4"


# ---------------------------------------------------------------------------
# Agent + ConversationMemory
# ---------------------------------------------------------------------------


class TestAgentWithMemory:
    async def test_multi_turn_with_trimming(self) -> None:
        client = _plain_client("response")
        agent = Agent("chatbot", client=client, system_prompt="be helpful")
        memory = ConversationMemory(system_message="be helpful")

        for i in range(3):
            memory.append_user(f"turn {i}")
            await agent.run(memory.non_system_messages)
            memory.append_assistant("response")

        # After 3 turns, memory has 6 non-system messages.
        assert len(memory.non_system_messages) == 6

        # Trim to a tight budget; system survives.
        memory.trim_to_messages(2)
        assert len(memory.non_system_messages) == 2
        assert memory.system_message is not None

    async def test_agent_run_accepts_memory_messages(self) -> None:
        # Run an agent over the exact tuple ConversationMemory returns.
        client = _plain_client("got it")
        agent = Agent("x", client=client, system_prompt="sys")
        memory = ConversationMemory(system_message="sys")
        memory.append_user("hello")

        result = await agent.run(memory.non_system_messages)
        # System prompt prepended; user-history follows.
        sent = client.complete.call_args.kwargs["messages"]
        assert isinstance(sent[0], SystemMessage)
        assert isinstance(sent[1], UserMessage)
        assert result.text == "got it"


# ---------------------------------------------------------------------------
# Agent + EpisodicMemory
# ---------------------------------------------------------------------------


class TestAgentWithEpisodicMemory:
    async def test_recall_seeds_next_run(self) -> None:
        # The agent looks up relevant facts before answering. Wiring
        # the memory into the agent is the caller's responsibility;
        # this test confirms the pieces compose.
        store = InMemoryVectorStore()

        async def _embed(text: str) -> list[float]:
            # Trivially-different vectors based on text identity.
            return [float(len(text)), 0.0]

        memory = EpisodicMemory(store=store, embed=_embed)
        await memory.add("user is allergic to peanuts")
        await memory.add("user prefers concise answers")

        # Search returns the two facts in score order.
        results = await memory.search("allergies", top_k=2)
        assert len(results) == 2

        client = _plain_client("noted")
        agent = Agent("dietary-assistant", client=client)
        # Caller prepends recalled facts as system context, then asks
        # the underlying agent.
        recalled = "Relevant memory: " + "; ".join(r.item.text for r in results)
        await agent.run(
            [SystemMessage(content=recalled), UserMessage(content="what should I avoid?")]
        )
        sent = client.complete.call_args.kwargs["messages"]
        assert "allergic" in sent[0].content or "allergic" in sent[1].content


# ---------------------------------------------------------------------------
# handoff cross-module
# ---------------------------------------------------------------------------


class TestHandoffCrossModule:
    async def test_router_picks_specialist_with_tools(self) -> None:
        # Specialist agent has a tool; handoff still dispatches correctly.
        router = Agent(
            "router",
            client=_structured_client(RouterChoice(specialist="math", reasoning="arithmetic")),
        )
        math_client = _plain_client("the math answer")
        math_agent = Agent("math", client=math_client, tools=[calculator])

        result = await handoff(
            router=router,
            specialists={"math": math_agent},
            user_input="what is 7 * 8?",
        )
        assert result.text == "the math answer"
        # Specialist invoked through run_tool_loop because it has tools.
        math_client.run_tool_loop.assert_called_once()


# ---------------------------------------------------------------------------
# critic-refiner cross-module
# ---------------------------------------------------------------------------


class TestCriticRefinerCrossModule:
    async def test_loops_and_returns_final_draft(self) -> None:
        drafter = Agent("d", client=_plain_client("draft"))

        # Build a critic whose run_structured cycles through reject/reject/approve.
        critic_client = AsyncMock()
        verdicts = [
            CritiqueVerdict(approved=False, feedback="rev 1"),
            CritiqueVerdict(approved=False, feedback="rev 2"),
            CritiqueVerdict(approved=True, feedback=""),
        ]
        critic_client.complete_structured = AsyncMock(
            side_effect=[_structured(v) for v in verdicts]
        )
        critic_client.complete = AsyncMock(return_value=_response("dummy"))
        critic = Agent("c", client=critic_client)

        result = await critic_refiner_run(
            drafter=drafter,
            critic=critic,
            user_input="describe a sunset",
            max_rounds=5,
        )
        assert isinstance(result, AgentResult)
        # Drafter called once per round (3 rounds).
        assert drafter.client.complete.call_count == 3  # type: ignore[attr-defined]
        # Critic called once per round.
        assert critic_client.complete_structured.call_count == 3
