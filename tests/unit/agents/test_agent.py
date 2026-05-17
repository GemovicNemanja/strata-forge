"""Unit tests for `forge.agents.agent`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ValidationError

from forge.agents import Agent, AgentResult, AssistantMessage, SystemMessage, UserMessage, tool
from forge.llm.responses import LLMResponse, Usage
from forge.llm.routing import ModelRoute

if TYPE_CHECKING:
    from forge.llm.registry import ProviderName as ProviderNameType  # type: ignore[unused-import]


def _response(text: str = "answer", *, provider: ProviderNameType = "anthropic") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(input_tokens=10, output_tokens=5),
        cost_usd=0.001,
        route=ModelRoute(
            model="claude-opus-4-7",
            provider=provider,
            provider_model_id="claude-opus-4-7",
        ),
        latency_ms=42.0,
    )


def _client(text: str = "answer") -> AsyncMock:
    client = AsyncMock()
    client.complete = AsyncMock(return_value=_response(text))
    client.run_tool_loop = AsyncMock(return_value=_response(text))
    client.complete_structured = AsyncMock()
    return client


class _ArgsModel(BaseModel):
    query: str


@tool
async def _dummy_tool(args: _ArgsModel) -> str:
    """A test tool."""
    return f"echo:{args.query}"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic_construction(self) -> None:
        agent = Agent("my-agent", client=_client())
        assert agent.name == "my-agent"
        assert agent.system_prompt is None
        assert agent.tools == ()
        assert agent.max_iterations == 8

    def test_construction_with_all_fields(self) -> None:
        client = _client()
        agent = Agent(
            "researcher",
            client=client,
            system_prompt="you are a helpful researcher",
            tools=[_dummy_tool],
            max_iterations=4,
        )
        assert agent.name == "researcher"
        assert agent.system_prompt == "you are a helpful researcher"
        assert agent.tools == (_dummy_tool,)
        assert agent.max_iterations == 4
        assert agent.client is client

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Agent("", client=_client())

    def test_zero_max_iterations_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            Agent("x", client=_client(), max_iterations=0)

    def test_negative_max_iterations_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            Agent("x", client=_client(), max_iterations=-1)

    def test_tools_stored_as_tuple(self) -> None:
        # Even when callers pass a list, the agent stores tools as a tuple.
        agent = Agent("x", client=_client(), tools=[_dummy_tool])
        assert isinstance(agent.tools, tuple)


# ---------------------------------------------------------------------------
# run() — no tools
# ---------------------------------------------------------------------------


class TestRunNoTools:
    async def test_no_tools_calls_complete(self) -> None:
        client = _client(text="hello world")
        agent = Agent("x", client=client, system_prompt="be helpful")
        result = await agent.run("hi there")
        assert isinstance(result, AgentResult)
        assert result.text == "hello world"
        client.complete.assert_called_once()
        client.run_tool_loop.assert_not_called()

    async def test_system_prompt_prepended(self) -> None:
        client = _client()
        agent = Agent("x", client=client, system_prompt="be helpful")
        await agent.run("hi")
        kwargs = client.complete.call_args.kwargs
        messages = kwargs["messages"]
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "be helpful"
        assert isinstance(messages[1], UserMessage)
        assert messages[1].content == "hi"

    async def test_no_system_prompt_omits_system_message(self) -> None:
        client = _client()
        agent = Agent("x", client=client)
        await agent.run("hi")
        messages = client.complete.call_args.kwargs["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)

    async def test_message_sequence_input_passed_through(self) -> None:
        client = _client()
        agent = Agent("x", client=client, system_prompt="sys")
        await agent.run(
            [
                UserMessage(content="first"),
                AssistantMessage(content="response"),
                UserMessage(content="second"),
            ]
        )
        messages = client.complete.call_args.kwargs["messages"]
        # System + 3 input messages.
        assert len(messages) == 4
        assert isinstance(messages[0], SystemMessage)
        assert messages[1].content == "first"
        assert messages[3].content == "second"

    async def test_sampling_params_forwarded(self) -> None:
        client = _client()
        agent = Agent("x", client=client)
        await agent.run("hi", temperature=0.7, max_tokens=200, top_p=0.95)
        kwargs = client.complete.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 200
        assert kwargs["top_p"] == 0.95

    async def test_provider_extras_forwarded(self) -> None:
        client = _client()
        agent = Agent("x", client=client)
        extras: dict[ProviderNameType, dict[str, Any]] = {
            "anthropic": {"thinking": {"type": "enabled"}}
        }
        await agent.run("hi", provider_extras=extras)
        kwargs = client.complete.call_args.kwargs
        assert kwargs["provider_extras"] == extras


# ---------------------------------------------------------------------------
# run() — with tools
# ---------------------------------------------------------------------------


class TestRunWithTools:
    async def test_tools_route_through_run_tool_loop(self) -> None:
        client = _client(text="done")
        agent = Agent("x", client=client, tools=[_dummy_tool])
        result = await agent.run("hi")
        client.run_tool_loop.assert_called_once()
        client.complete.assert_not_called()
        assert result.text == "done"

    async def test_run_tool_loop_receives_tools_and_max_iterations(self) -> None:
        client = _client()
        agent = Agent("x", client=client, tools=[_dummy_tool], max_iterations=3)
        await agent.run("hi")
        kwargs = client.run_tool_loop.call_args.kwargs
        assert kwargs["tools"] == (_dummy_tool,)
        assert kwargs["max_iterations"] == 3

    async def test_run_tool_loop_sees_system_and_user(self) -> None:
        client = _client()
        agent = Agent("x", client=client, system_prompt="sys", tools=[_dummy_tool])
        await agent.run("hi")
        call_args = client.run_tool_loop.call_args
        messages = call_args.args[0]  # positional `messages` argument
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------


class TestAgentResult:
    async def test_result_includes_input_messages(self) -> None:
        client = _client(text="ok")
        agent = Agent("x", client=client, system_prompt="sys")
        result = await agent.run("hello")
        assert len(result.messages) == 2
        assert isinstance(result.messages[0], SystemMessage)
        assert isinstance(result.messages[1], UserMessage)

    async def test_result_carries_final_response(self) -> None:
        client = _client(text="ok")
        agent = Agent("x", client=client)
        result = await agent.run("hi")
        assert result.final_response.text == "ok"
        assert result.final_response.cost_usd == 0.001

    async def test_cost_convenience_accessor(self) -> None:
        client = _client()
        agent = Agent("x", client=client)
        result = await agent.run("hi")
        assert result.cost_usd == result.final_response.cost_usd

    async def test_latency_convenience_accessor(self) -> None:
        client = _client()
        agent = Agent("x", client=client)
        result = await agent.run("hi")
        assert result.latency_ms == 42.0

    async def test_parsed_is_none_on_plain_run(self) -> None:
        client = _client()
        agent = Agent("x", client=client)
        result = await agent.run("hi")
        assert result.parsed is None

    async def test_is_frozen(self) -> None:
        client = _client()
        agent = Agent("x", client=client)
        result = await agent.run("hi")
        with pytest.raises(ValidationError, match="frozen"):
            result.text = "other"  # type: ignore[misc]

    async def test_messages_is_tuple(self) -> None:
        client = _client()
        agent = Agent("x", client=client)
        result = await agent.run("hi")
        assert isinstance(result.messages, tuple)


# ---------------------------------------------------------------------------
# run_structured
# ---------------------------------------------------------------------------


class _Output(BaseModel):
    answer: str
    confidence: float


class TestRunStructured:
    async def test_calls_complete_structured(self) -> None:
        client = _client()
        # AgentResult.final_response needs a real LLMResponse, so we
        # build a StructuredResponse (a subclass) and hand it back.
        from forge.llm.client import StructuredResponse

        real_resp = _response(text="the answer is x")
        sr = StructuredResponse(
            text="the answer is x",
            tool_calls=[],
            finish_reason="stop",
            usage=real_resp.usage,
            cost_usd=real_resp.cost_usd,
            route=real_resp.route,
            cache_hit=False,
            latency_ms=real_resp.latency_ms,
            parsed=_Output(answer="x", confidence=0.9),
        )
        client.complete_structured = AsyncMock(return_value=sr)

        agent = Agent("x", client=client, system_prompt="be helpful")
        result = await agent.run_structured("question", output_schema=_Output)
        assert result.text == "the answer is x"
        assert isinstance(result.parsed, _Output)
        assert result.parsed.answer == "x"
        client.complete_structured.assert_called_once()

    async def test_tools_not_used_in_structured(self) -> None:
        # run_structured ignores agent tools; complete_structured is called.
        client = _client()
        from forge.llm.client import StructuredResponse

        sr = StructuredResponse(
            text="x",
            tool_calls=[],
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1),
            cost_usd=0.0,
            route=ModelRoute(
                model="claude-opus-4-7",
                provider="anthropic",
                provider_model_id="claude-opus-4-7",
            ),
            cache_hit=False,
            latency_ms=1.0,
            parsed=_Output(answer="x", confidence=0.5),
        )
        client.complete_structured = AsyncMock(return_value=sr)

        agent = Agent("x", client=client, tools=[_dummy_tool])
        await agent.run_structured("hi", output_schema=_Output)
        client.complete_structured.assert_called_once()
        client.run_tool_loop.assert_not_called()

    async def test_schema_forwarded(self) -> None:
        client = _client()
        from forge.llm.client import StructuredResponse

        sr = StructuredResponse(
            text="x",
            tool_calls=[],
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1),
            cost_usd=0.0,
            route=ModelRoute(
                model="m",
                provider="anthropic",
                provider_model_id="m",
            ),
            cache_hit=False,
            latency_ms=1.0,
            parsed=_Output(answer="x", confidence=0.5),
        )
        client.complete_structured = AsyncMock(return_value=sr)

        agent = Agent("x", client=client)
        await agent.run_structured("hi", output_schema=_Output)
        kwargs = client.complete_structured.call_args.kwargs
        assert kwargs["schema"] is _Output

    async def test_max_reprompt_attempts_forwarded(self) -> None:
        client = _client()
        from forge.llm.client import StructuredResponse

        sr = StructuredResponse(
            text="x",
            tool_calls=[],
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1),
            cost_usd=0.0,
            route=ModelRoute(
                model="m",
                provider="anthropic",
                provider_model_id="m",
            ),
            cache_hit=False,
            latency_ms=1.0,
            parsed=_Output(answer="x", confidence=0.5),
        )
        client.complete_structured = AsyncMock(return_value=sr)

        agent = Agent("x", client=client)
        await agent.run_structured("hi", output_schema=_Output, max_reprompt_attempts=5)
        kwargs = client.complete_structured.call_args.kwargs
        assert kwargs["max_reprompt_attempts"] == 5
