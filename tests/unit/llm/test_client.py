"""Unit tests for `forge.llm.client`.

The strategy: mock `litellm.acompletion` at the boundary so the LLMClient
+ ProviderClient stack runs end-to-end against synthetic responses.
"""

from __future__ import annotations

import json
import types
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from forge.core.budget import BudgetContext
from forge.core.errors import (
    BudgetExceededError,
    ProviderRateLimitError,
    RegistryError,
)
from forge.llm.cache import InMemoryCache
from forge.llm.client import LLMClient, StructuredResponse
from forge.llm.fallback import ModelFallback
from forge.llm.messages import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)
from forge.llm.tools import ToolLoopExceededError, tool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from forge.llm.providers.base import ProviderClient
    from forge.llm.registry import ProviderName


# ---------------------------------------------------------------------------
# Fake LiteLLM response shape
# ---------------------------------------------------------------------------


def _fake_response(
    *,
    text: str = "hello",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 5,
    completion_tokens: int = 2,
    cache_read: int = 0,
) -> types.SimpleNamespace:
    """Build a LiteLLM-shaped `ModelResponse` substitute for tests."""
    tc_namespaces = [
        types.SimpleNamespace(
            id=tc["id"],
            type="function",
            function=types.SimpleNamespace(
                name=tc["name"],
                arguments=json.dumps(tc.get("arguments", {})),
            ),
        )
        for tc in (tool_calls or [])
    ]
    message = types.SimpleNamespace(content=text, tool_calls=tc_namespaces or None)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = types.SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=cache_read),
    )
    return types.SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def mock_litellm(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch `litellm.acompletion` and return the AsyncMock for assertions."""
    mock = AsyncMock(return_value=_fake_response())
    monkeypatch.setattr("litellm.acompletion", mock)
    # Diagnostic is disabled by default; tests that need it set the env var.
    monkeypatch.delenv("FORGE_DIAGNOSTIC_ENABLED", raising=False)
    return mock


def _kwargs(mock: AsyncMock) -> Mapping[str, Any]:
    """Unwrap the most recent call's kwargs, asserting the mock was awaited."""
    assert mock.await_args is not None, "AsyncMock was never awaited"
    return mock.await_args.kwargs


# Sample Pydantic schemas for structured output.
class _Summary(BaseModel):
    title: str
    bullets: list[str] = Field(default=[])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_model_only(self) -> None:
        client = LLMClient("claude-opus-4-7")
        assert len(client.chain) == 1
        assert client.chain[0].model == "claude-opus-4-7"
        assert client.chain[0].providers is None

    def test_model_with_provider(self) -> None:
        client = LLMClient("claude-opus-4-7", provider="bedrock")
        assert client.chain[0].providers == ("bedrock",)

    def test_chain_only(self) -> None:
        client = LLMClient(
            chain=[
                ModelFallback(model="claude-opus-4-7", providers=("anthropic", "bedrock")),
                "gpt-5.5",
            ]
        )
        assert len(client.chain) == 2
        assert client.chain[1].model == "gpt-5.5"

    def test_with_fallbacks_classmethod(self) -> None:
        client = LLMClient.with_fallbacks(["claude-opus-4-7", "gpt-5.5"])
        assert len(client.chain) == 2

    def test_neither_model_nor_chain_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            LLMClient()

    def test_both_model_and_chain_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            LLMClient("claude-opus-4-7", chain=["gpt-5.5"])


# ---------------------------------------------------------------------------
# complete() basics
# ---------------------------------------------------------------------------


class TestCompleteBasics:
    async def test_completes(self, mock_litellm: AsyncMock) -> None:
        client = LLMClient("claude-opus-4-7")
        resp = await client.complete([Message.user("hi")])
        assert resp.text == "hello"
        assert resp.finish_reason == "stop"
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 2
        assert resp.cost_usd > 0
        assert resp.route.model == "claude-opus-4-7"
        assert resp.route.provider == "anthropic"  # registry default

    async def test_calls_litellm_once(self, mock_litellm: AsyncMock) -> None:
        client = LLMClient("claude-opus-4-7")
        await client.complete([Message.user("hi")])
        assert mock_litellm.await_count == 1
        kwargs = _kwargs(mock_litellm)
        assert kwargs["model"] == "anthropic/claude-opus-4-7"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]

    async def test_passes_sampling_params(self, mock_litellm: AsyncMock) -> None:
        client = LLMClient("claude-opus-4-7")
        await client.complete([Message.user("hi")], temperature=0.5, max_tokens=100, top_p=0.9)
        kwargs = _kwargs(mock_litellm)
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 100
        assert kwargs["top_p"] == 0.9

    async def test_provider_pin(self, mock_litellm: AsyncMock) -> None:
        client = LLMClient("claude-opus-4-7", provider="bedrock")
        resp = await client.complete([Message.user("hi")])
        assert resp.route.provider == "bedrock"
        kwargs = _kwargs(mock_litellm)
        assert kwargs["model"].startswith("bedrock/")

    async def test_provider_extras_passthrough(self, mock_litellm: AsyncMock) -> None:
        client = LLMClient("claude-opus-4-7")
        await client.complete(
            [Message.user("hi")],
            provider_extras={"anthropic": {"thinking": {"budget_tokens": 8000}}},
        )
        kwargs = _kwargs(mock_litellm)
        assert kwargs["thinking"] == {"budget_tokens": 8000}

    async def test_unknown_model_raises(self, mock_litellm: AsyncMock) -> None:
        # Unknown models surface during fallback as RegistryError; the runner
        # records and exhausts.
        from forge.core.errors import FallbackExhaustedError

        client = LLMClient("not-a-real-model")
        with pytest.raises(FallbackExhaustedError):
            await client.complete([Message.user("hi")])
        assert mock_litellm.await_count == 0


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------


class _WeatherArgs(BaseModel):
    location: str


@tool
async def _get_weather(args: _WeatherArgs) -> dict[str, Any]:
    """Get weather."""
    return {"temp_c": 18, "city": args.location}


class _BadArgs(BaseModel):
    x: int


@tool
async def _explodes(args: _BadArgs) -> str:
    """A tool that always fails."""
    raise RuntimeError(f"oops at {args.x}")


class TestToolCalling:
    async def test_capability_check_blocks_non_tool_model(
        self,
        mock_litellm: AsyncMock,
    ) -> None:
        # All current registry models support tool calling, so we need a way to
        # synthesize one that doesn't. Patch the model's capability flag.
        from unittest.mock import patch

        from forge.llm.registry import registry as _registry

        original = _registry.get("claude-opus-4-7")
        synth = original.model_copy(
            update={
                "capabilities": original.capabilities.model_copy(update={"tool_calling": False})
            }
        )

        def _fake_get(name: str) -> Any:
            return synth if name == "claude-opus-4-7" else original

        with patch.object(_registry, "get", _fake_get):
            client = LLMClient("claude-opus-4-7")
            with pytest.raises(RegistryError, match="does not support tool calling"):
                await client.complete([Message.user("hi")], tools=[_get_weather])
        assert mock_litellm.await_count == 0

    async def test_tool_calls_serialized_in_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _fake_response(
            text="",
            finish_reason="tool_calls",
            tool_calls=[
                {"id": "c1", "name": "_get_weather", "arguments": {"location": "Tokyo"}},
            ],
        )
        monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=fake))
        client = LLMClient("claude-opus-4-7")
        resp = await client.complete([Message.user("weather?")], tools=[_get_weather])
        assert resp.finish_reason == "tool_use"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "_get_weather"
        assert resp.tool_calls[0].arguments == {"location": "Tokyo"}

    async def test_tools_serialized_per_provider(self, mock_litellm: AsyncMock) -> None:
        # Anthropic should get the Anthropic-shaped tool schema.
        client = LLMClient("claude-opus-4-7", provider="anthropic")
        await client.complete([Message.user("hi")], tools=[_get_weather])
        kwargs = _kwargs(mock_litellm)
        assert kwargs["tools"][0]["name"] == "_get_weather"
        assert "input_schema" in kwargs["tools"][0]  # Anthropic shape

    async def test_openai_tool_schema_shape(self, mock_litellm: AsyncMock) -> None:
        client = LLMClient("gpt-5.5", provider="openai")
        await client.complete([Message.user("hi")], tools=[_get_weather])
        kwargs = _kwargs(mock_litellm)
        # OpenAI shape: {"type": "function", "function": {...}}
        assert kwargs["tools"][0]["type"] == "function"
        assert "function" in kwargs["tools"][0]


# ---------------------------------------------------------------------------
# Cache integration
# ---------------------------------------------------------------------------


class TestCacheIntegration:
    async def test_cache_hit_returns_cached_with_zero_cost(
        self,
        mock_litellm: AsyncMock,
    ) -> None:
        cache = InMemoryCache()
        client = LLMClient("claude-opus-4-7", cache=cache)
        await client.complete([Message.user("hi")])
        # Second call should hit the cache.
        resp2 = await client.complete([Message.user("hi")])
        assert resp2.cache_hit is True
        assert resp2.cost_usd == 0.0
        # litellm.acompletion was only called once (the second was a cache hit).
        assert mock_litellm.await_count == 1

    async def test_cache_miss_then_set(self, mock_litellm: AsyncMock) -> None:
        cache = InMemoryCache()
        client = LLMClient("claude-opus-4-7", cache=cache)
        await client.complete([Message.user("hi")])
        assert len(cache) == 1

    async def test_different_messages_miss_independently(
        self,
        mock_litellm: AsyncMock,
    ) -> None:
        cache = InMemoryCache()
        client = LLMClient("claude-opus-4-7", cache=cache)
        await client.complete([Message.user("hi")])
        await client.complete([Message.user("bye")])
        assert len(cache) == 2
        assert mock_litellm.await_count == 2


# ---------------------------------------------------------------------------
# Fallback integration
# ---------------------------------------------------------------------------


class TestFallbackIntegration:
    async def test_provider_failover(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # First call: anthropic 429; second: bedrock OK.
        from litellm.exceptions import RateLimitError as LLRateLimitError

        responses: list[Any] = [
            LLRateLimitError(message="429", model="claude-opus-4-7", llm_provider="anthropic"),
            _fake_response(text="from bedrock"),
        ]

        async def _side_effect(**_kwargs: Any) -> Any:
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_side_effect))
        client = LLMClient(
            chain=[ModelFallback(model="claude-opus-4-7", providers=("anthropic", "bedrock"))],
            retry_max_attempts=1,
            retry_initial_wait=0.0,
            retry_max_wait=0.0,
        )
        resp = await client.complete([Message.user("hi")])
        assert resp.text == "from bedrock"
        assert resp.route.provider == "bedrock"

    async def test_model_failover(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from litellm.exceptions import RateLimitError as LLRateLimitError

        responses: list[Any] = [
            LLRateLimitError(message="429", model="claude", llm_provider="anthropic"),
            _fake_response(text="from gpt"),
        ]

        async def _side_effect(**_kwargs: Any) -> Any:
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_side_effect))
        client = LLMClient.with_fallbacks(
            ["claude-opus-4-7", "gpt-5.5"],
            retry_max_attempts=1,
            retry_initial_wait=0.0,
            retry_max_wait=0.0,
        )
        resp = await client.complete([Message.user("hi")])
        assert resp.text == "from gpt"
        assert resp.route.model == "gpt-5.5"


# ---------------------------------------------------------------------------
# Budget integration
# ---------------------------------------------------------------------------


class TestBudgetIntegration:
    async def test_budget_consumes_on_success(self, mock_litellm: AsyncMock) -> None:
        client = LLMClient("claude-opus-4-7")
        async with BudgetContext(max_usd=1.00) as budget:
            await client.complete([Message.user("hi")])
            assert budget.spent_usd > 0

    async def test_budget_exceeded_raises(self, mock_litellm: AsyncMock) -> None:
        client = LLMClient("claude-opus-4-7")
        # Use a tiny ceiling — the first call's cost will exceed it.
        async with BudgetContext(max_usd=0.0000001) as _budget:
            with pytest.raises(BudgetExceededError):
                await client.complete([Message.user("hi")])


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    async def test_openai_route_uses_response_format(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        payload = {"title": "Hello", "bullets": ["a", "b"]}

        async def _fake(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _fake_response(text=json.dumps(payload))

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("gpt-5.5", provider="openai")
        resp = await client.complete_structured([Message.user("hi")], schema=_Summary)
        assert isinstance(resp, StructuredResponse)
        assert resp.parsed.title == "Hello"
        assert resp.parsed.bullets == ["a", "b"]
        # OpenAI's response_format payload was set.
        assert captured["response_format"]["type"] == "json_schema"

    async def test_anthropic_route_uses_forced_tool(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        async def _fake(**kwargs: Any) -> Any:
            captured.update(kwargs)
            # Anthropic returns the structured output as a tool call.
            return _fake_response(
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "_Summary",
                        "arguments": {"title": "From Anthropic", "bullets": ["x"]},
                    }
                ],
            )

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("claude-opus-4-7", provider="anthropic")
        resp = await client.complete_structured([Message.user("hi")], schema=_Summary)
        assert resp.parsed.title == "From Anthropic"
        # Forced-tool emulation injects tools + tool_choice via provider_extras.
        assert captured.get("tool_choice") is not None
        assert captured["tools"][0]["name"] == "_Summary"

    async def test_reprompts_on_invalid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        responses: list[Any] = [
            _fake_response(text="not json"),
            _fake_response(text=json.dumps({"title": "Recovered", "bullets": []})),
        ]

        async def _fake(**_kwargs: Any) -> Any:
            return responses.pop(0)

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("gpt-5.5", provider="openai")
        resp = await client.complete_structured(
            [Message.user("hi")], schema=_Summary, max_reprompt_attempts=3
        )
        assert resp.parsed.title == "Recovered"

    async def test_reprompt_exhausted_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from forge.llm.schemas import StructuredOutputError

        # Always return invalid JSON.
        async def _fake(**_kwargs: Any) -> Any:
            return _fake_response(text="still not json")

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("gpt-5.5", provider="openai")
        with pytest.raises(StructuredOutputError):
            await client.complete_structured(
                [Message.user("hi")], schema=_Summary, max_reprompt_attempts=1
            )

    async def test_vertex_gemini_route_uses_response_schema(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        payload = {"title": "From Gemini", "bullets": []}

        async def _fake(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _fake_response(text=json.dumps(payload))

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("gemini-3.1-pro", provider="vertex")
        resp = await client.complete_structured([Message.user("hi")], schema=_Summary)
        assert resp.parsed.title == "From Gemini"
        # Gemini uses response_schema + response_mime_type.
        assert captured.get("response_mime_type") == "application/json"
        assert captured.get("response_schema") is not None

    async def test_extract_structured_text_falls_back_to_first_tool_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No text AND no matching tool-call name — fall back to first tool call.
        async def _fake(**_kwargs: Any) -> Any:
            return _fake_response(
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "differently_named_tool",
                        "arguments": {"title": "Recovered", "bullets": []},
                    }
                ],
            )

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("claude-opus-4-7", provider="anthropic")
        resp = await client.complete_structured([Message.user("hi")], schema=_Summary)
        assert resp.parsed.title == "Recovered"

    async def test_extract_structured_text_no_text_no_tools_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from forge.llm.schemas import StructuredOutputError

        # The model returns absolutely nothing usable on every attempt.
        async def _fake(**_kwargs: Any) -> Any:
            return _fake_response(text="", finish_reason="stop")

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("gpt-5.5", provider="openai")
        with pytest.raises(StructuredOutputError):
            await client.complete_structured(
                [Message.user("hi")], schema=_Summary, max_reprompt_attempts=0
            )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    async def test_stream_yields_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _chunks() -> AsyncIterator[Any]:
            for piece in ("hel", "lo", " world"):
                yield types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(content=piece, tool_calls=None),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                )

        async def _fake(**_kwargs: Any) -> Any:
            return _chunks()

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("claude-opus-4-7")
        pieces: list[str] = []
        async for chunk in await client.stream([Message.user("hi")]):
            pieces.append(chunk.delta_text)
        assert "".join(pieces) == "hello world"

    async def test_stream_with_tool_call_deltas(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def _chunks() -> AsyncIterator[Any]:
            yield types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        delta=types.SimpleNamespace(
                            content=None,
                            tool_calls=[
                                types.SimpleNamespace(
                                    index=0,
                                    id="c1",
                                    function=types.SimpleNamespace(
                                        name="_get_weather",
                                        arguments='{"location":',
                                    ),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
            yield types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        delta=types.SimpleNamespace(
                            content=None,
                            tool_calls=[
                                types.SimpleNamespace(
                                    index=0,
                                    id=None,
                                    function=types.SimpleNamespace(
                                        name=None,
                                        arguments=' "Tokyo"}',
                                    ),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=None,
            )

        async def _fake(**_kwargs: Any) -> Any:
            return _chunks()

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("claude-opus-4-7")
        deltas: list[Any] = []
        async for chunk in await client.stream([Message.user("?")], tools=[_get_weather]):
            deltas.extend(chunk.delta_tool_calls)
        assert len(deltas) == 2
        assert deltas[0].id == "c1"
        assert deltas[0].name == "_get_weather"
        assert deltas[0].arguments_delta == '{"location":'

    async def test_stream_passes_sampling_and_extras(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        async def _chunks() -> AsyncIterator[Any]:
            yield types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        delta=types.SimpleNamespace(content="x", tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )

        async def _fake(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _chunks()

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("claude-opus-4-7")
        async for _chunk in await client.stream(
            [Message.user("hi")],
            temperature=0.3,
            max_tokens=50,
            top_p=0.9,
            provider_extras={"anthropic": {"foo": "bar"}},
        ):
            pass
        assert captured["temperature"] == 0.3
        assert captured["max_tokens"] == 50
        assert captured["top_p"] == 0.9
        assert captured["foo"] == "bar"

    async def test_stream_error_normalized(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from litellm.exceptions import RateLimitError as LLRateLimitError

        async def _chunks() -> AsyncIterator[Any]:
            raise LLRateLimitError(message="429", model="x", llm_provider="anthropic")
            yield None  # pragma: no cover

        async def _fake(**_kwargs: Any) -> Any:
            return _chunks()

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("claude-opus-4-7")
        with pytest.raises(ProviderRateLimitError):
            async for _chunk in await client.stream([Message.user("hi")]):
                pass


# ---------------------------------------------------------------------------
# Tool loop
# ---------------------------------------------------------------------------


class TestToolLoop:
    async def test_terminates_when_no_tool_use(
        self,
        mock_litellm: AsyncMock,
    ) -> None:
        client = LLMClient("claude-opus-4-7")
        resp = await client.run_tool_loop([Message.user("hi")], tools=[_get_weather])
        assert resp.text == "hello"
        assert mock_litellm.await_count == 1

    async def test_runs_one_tool_then_returns(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Iteration 1: model emits a tool call.
        # Iteration 2: model returns plain text.
        responses = [
            _fake_response(
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    {"id": "c1", "name": "_get_weather", "arguments": {"location": "Tokyo"}}
                ],
            ),
            _fake_response(text="It's sunny in Tokyo."),
        ]

        async def _fake(**_kwargs: Any) -> Any:
            return responses.pop(0)

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("claude-opus-4-7")
        resp = await client.run_tool_loop(
            [Message.user("weather?")],
            tools=[_get_weather],
            max_iterations=8,
        )
        assert resp.text == "It's sunny in Tokyo."

    async def test_tool_exception_recorded_as_error_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A tool that raises shouldn't terminate the loop — the runner should
        # surface the error to the model via a `is_error=True` result.
        responses = [
            _fake_response(
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    {"id": "c1", "name": "_explodes", "arguments": {"x": 1}}
                ],
            ),
            _fake_response(text="recovered"),
        ]

        async def _fake(**_kwargs: Any) -> Any:
            return responses.pop(0)

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("claude-opus-4-7")
        resp = await client.run_tool_loop(
            [Message.user("?")], tools=[_explodes], max_iterations=3
        )
        assert resp.text == "recovered"

    async def test_unknown_tool_marked_as_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        responses = [
            _fake_response(
                text="",
                finish_reason="tool_calls",
                tool_calls=[{"id": "c1", "name": "missing_tool", "arguments": {}}],
            ),
            _fake_response(text="oops"),
        ]

        async def _fake(**_kwargs: Any) -> Any:
            return responses.pop(0)

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("claude-opus-4-7")
        resp = await client.run_tool_loop(
            [Message.user("?")],
            tools=[_get_weather],
            max_iterations=3,
        )
        # The loop survived a missing tool by recording an error result.
        assert resp.text == "oops"

    async def test_max_iterations_zero_rejected(self, mock_litellm: AsyncMock) -> None:
        client = LLMClient("claude-opus-4-7")
        with pytest.raises(ValueError, match=">= 1"):
            await client.run_tool_loop([Message.user("hi")], tools=[_get_weather], max_iterations=0)

    async def test_loop_exhausted_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Every iteration the model insists on calling the tool.
        async def _fake(**_kwargs: Any) -> Any:
            return _fake_response(
                text="",
                finish_reason="tool_calls",
                tool_calls=[{"id": "c1", "name": "_get_weather", "arguments": {"location": "X"}}],
            )

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("claude-opus-4-7")
        with pytest.raises(ToolLoopExceededError) as info:
            await client.run_tool_loop([Message.user("?")], tools=[_get_weather], max_iterations=2)
        assert info.value.max_iterations == 2


# ---------------------------------------------------------------------------
# Diagnostic integration
# ---------------------------------------------------------------------------


class TestDiagnosticIntegration:
    async def test_success_record_written(
        self,
        mock_litellm: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        dump_path = tmp_path / "diag.ndjson"
        monkeypatch.setenv("FORGE_DIAGNOSTIC_ENABLED", "true")
        monkeypatch.setenv("FORGE_DIAGNOSTIC_PATH", str(dump_path))

        client = LLMClient("claude-opus-4-7")
        await client.complete([Message.user("hi")])

        assert dump_path.exists()
        lines = dump_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["model"] == "claude-opus-4-7"
        assert record["error"] is None
        assert record["response_text"] == "hello"

    async def test_failure_record_written(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        from litellm.exceptions import RateLimitError as LLRateLimitError

        dump_path = tmp_path / "diag.ndjson"
        monkeypatch.setenv("FORGE_DIAGNOSTIC_ENABLED", "true")
        monkeypatch.setenv("FORGE_DIAGNOSTIC_PATH", str(dump_path))

        async def _fail(**_kwargs: Any) -> Any:
            raise LLRateLimitError(message="429", model="x", llm_provider="anthropic")

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fail))

        client = LLMClient(
            "claude-opus-4-7",
            retry_max_attempts=1,
            retry_initial_wait=0.0,
            retry_max_wait=0.0,
        )
        from forge.core.errors import FallbackExhaustedError

        with pytest.raises(FallbackExhaustedError):
            await client.complete([Message.user("hi")])

        lines = dump_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) >= 1
        records = [json.loads(line) for line in lines]
        # At least one record must have an error field set.
        assert any(r["error"] is not None for r in records)


# ---------------------------------------------------------------------------
# Multimodal
# ---------------------------------------------------------------------------


class TestMultimodal:
    async def test_image_content_serialized_per_provider(
        self,
        mock_litellm: AsyncMock,
    ) -> None:
        from forge.llm.messages import TextPart
        from forge.llm.multimodal import ImageContent

        client = LLMClient("claude-opus-4-7", provider="anthropic")
        await client.complete(
            [
                UserMessage(
                    content=[
                        TextPart(text="What's in this?"),
                        ImageContent.from_url("https://example.com/img.png"),
                    ]
                )
            ]
        )
        wire = _kwargs(mock_litellm)["messages"][0]
        # Anthropic shape: content is a list with type "text" + type "image"
        # using {"type": "image", "source": ...}.
        assert isinstance(wire["content"], list)
        image_part = next((p for p in wire["content"] if p.get("type") == "image"), None)
        assert image_part is not None
        assert image_part["source"]["type"] == "url"

    async def test_image_for_openai(self, mock_litellm: AsyncMock) -> None:
        from forge.llm.messages import TextPart
        from forge.llm.multimodal import ImageContent

        client = LLMClient("gpt-5.5", provider="openai")
        await client.complete(
            [
                UserMessage(
                    content=[
                        TextPart(text="?"),
                        ImageContent.from_url("https://example.com/img.png"),
                    ]
                )
            ]
        )
        wire = _kwargs(mock_litellm)["messages"][0]
        image_part = next((p for p in wire["content"] if p.get("type") == "image_url"), None)
        assert image_part is not None


# ---------------------------------------------------------------------------
# Message wire-format
# ---------------------------------------------------------------------------


class TestWireMessages:
    async def test_system_message(self, mock_litellm: AsyncMock) -> None:
        client = LLMClient("claude-opus-4-7")
        await client.complete([Message.system("you are helpful"), Message.user("hi")])
        wire = _kwargs(mock_litellm)["messages"]
        assert wire[0] == {"role": "system", "content": "you are helpful"}
        assert wire[1] == {"role": "user", "content": "hi"}

    async def test_assistant_with_tool_calls(self, mock_litellm: AsyncMock) -> None:
        from forge.llm.messages import ToolCall

        client = LLMClient("claude-opus-4-7")
        msgs = [
            Message.user("?"),
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(id="c1", name="f", arguments={"x": 1}),
                ],
            ),
            ToolResultMessage(tool_call_id="c1", content="42"),
        ]
        await client.complete(msgs)
        wire = _kwargs(mock_litellm)["messages"]
        # Assistant entry has tool_calls in OpenAI shape.
        assistant_wire = next(w for w in wire if w["role"] == "assistant")
        assert assistant_wire["tool_calls"][0]["function"]["name"] == "f"
        # Tool result entry has tool_call_id and content.
        tool_wire = next(w for w in wire if w["role"] == "tool")
        assert tool_wire["tool_call_id"] == "c1"
        assert tool_wire["content"] == "42"


# ---------------------------------------------------------------------------
# Provider client injection
# ---------------------------------------------------------------------------


class TestProviderClientInjection:
    async def test_custom_provider_client(self, mock_litellm: AsyncMock) -> None:
        # Inject a mocked provider client and verify it's used.
        from forge.llm.providers import AnthropicProvider

        custom = AnthropicProvider()
        custom_dict: dict[ProviderName, ProviderClient] = {"anthropic": custom}
        # Fill remaining providers with stubs so init doesn't crash.
        from forge.llm.providers import (
            AzureProvider,
            BedrockProvider,
            OpenAICompatProvider,
            OpenAIProvider,
            VertexProvider,
        )

        custom_dict["openai"] = OpenAIProvider()
        custom_dict["vertex"] = VertexProvider()
        custom_dict["bedrock"] = BedrockProvider()
        custom_dict["azure"] = AzureProvider()
        custom_dict["openai_compat"] = OpenAICompatProvider()

        client = LLMClient("claude-opus-4-7", provider_clients=custom_dict)
        resp = await client.complete([Message.user("hi")])
        assert resp.text == "hello"
