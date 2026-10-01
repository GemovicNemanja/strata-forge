"""Unit tests for `strata_forge.llm.client`.

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

from strata_forge.core.budget import BudgetContext
from strata_forge.core.errors import (
    BudgetExceededError,
    ProviderRateLimitError,
    RegistryError,
    ValidationError,
)
from strata_forge.llm.cache import InMemoryCache
from strata_forge.llm.client import LLMClient, StructuredResponse
from strata_forge.llm.fallback import ModelFallback
from strata_forge.llm.loop_events import (
    Done,
    IterationStart,
    LoopError,
    PendingToolCalls,
    TextDelta,
    ToolCallStarted,
    ToolResult,
)
from strata_forge.llm.messages import (
    AssistantMessage,
    CallRef,
    Message,
    ProviderItems,
    ReasoningItem,
    TextItem,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from strata_forge.llm.tools import ToolDeclaration, ToolLoopExceededError, tool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping

    import httpx

    from strata_forge.llm.providers.base import ProviderClient
    from strata_forge.llm.registry import ProviderName


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
        from strata_forge.core.errors import FallbackExhaustedError

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


# A declaration-only tool — executed by the caller, never by strata_forge.
_LOAD_MODEL = ToolDeclaration(
    name="load_model",
    description="Load a model in the caller's app.",
    parameters={
        "type": "object",
        "properties": {"repo_id": {"type": "string", "description": "Repo id"}},
        "required": ["repo_id"],
        "additionalProperties": False,
    },
)


class TestToolCalling:
    async def test_capability_check_blocks_non_tool_model(
        self,
        mock_litellm: AsyncMock,
    ) -> None:
        # All current registry models support tool calling, so we need a way to
        # synthesize one that doesn't. Patch the model's capability flag.
        from unittest.mock import patch

        from strata_forge.llm.registry import registry as _registry

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
        # Chat Completions shape, which openai_compat endpoints speak.
        client = LLMClient("gpt-5.5", provider="openai_compat")
        await client.complete([Message.user("hi")], tools=[_get_weather])
        kwargs = _kwargs(mock_litellm)
        # OpenAI shape: {"type": "function", "function": {...}}
        assert kwargs["tools"][0]["type"] == "function"
        assert "function" in kwargs["tools"][0]


class TestRequireToolSupport:
    """The opt-in capability gate for unconfirmable (openai_compat) models."""

    async def test_strict_gate_raises_on_unconfirmable_model(
        self,
        mock_litellm: AsyncMock,
    ) -> None:
        # An openai_compat / OpenRouter model is absent from the curated
        # registry, so its tool-calling capability can't be confirmed. With
        # require_tool_support the gate raises a clean pre-flight error
        # instead of letting the provider reject it opaquely mid-call.
        client = LLMClient(
            "some/unknown-model", provider="openai_compat", require_tool_support=True
        )
        with pytest.raises(RegistryError, match="cannot be confirmed") as info:
            await client.complete([Message.user("hi")], tools=[_get_weather])
        assert info.value.reason == "capability_unknown"
        assert mock_litellm.await_count == 0

    async def test_strict_gate_applies_to_stream_tool_loop(self) -> None:
        client = LLMClient(
            "some/unknown-model", provider="openai_compat", require_tool_support=True
        )
        with pytest.raises(RegistryError, match="cannot be confirmed"):
            await _collect(client.stream_tool_loop([Message.user("hi")], tools=[_get_weather]))

    async def test_lenient_by_default_does_not_block_unconfirmable_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Default require_tool_support=False: the gate is permissive — the call proceeds past it
        # and succeeds. compute_cost used to be stubbed out here "because the untracked model has
        # no registry pricing", which was the production defect in disguise: the real client hit
        # the same wall and turned a completed generation into RegistryError. It is left unstubbed
        # so this test exercises the path a caller actually takes.
        monkeypatch.setattr(
            "litellm.acompletion", AsyncMock(return_value=_fake_response(text="ok"))
        )
        client = LLMClient("some/unknown-model", provider="openai_compat")
        resp = await client.complete([Message.user("hi")], tools=[_get_weather])
        assert resp.text == "ok"

    async def test_known_capable_model_passes_strict_gate(
        self,
        mock_litellm: AsyncMock,
    ) -> None:
        # A registry model that supports tools is unaffected by strict mode.
        client = LLMClient("claude-opus-4-7", require_tool_support=True)
        resp = await client.complete([Message.user("hi")], tools=[_get_weather])
        assert resp.text == "hello"


# ---------------------------------------------------------------------------
# Cache integration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Responses API routes (wire_api: responses)
# ---------------------------------------------------------------------------

_RS_USAGE: dict[str, Any] = {
    "input_tokens": 1_000,
    "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 100},
    "output_tokens": 300,
    "output_tokens_details": {"reasoning_tokens": 250},
    "total_tokens": 1_300,
}


def _rs_reasoning(rs_id: str = "rs_1", blob: str = "gAAAA-encrypted-1") -> dict[str, Any]:
    return {"type": "reasoning", "id": rs_id, "summary": [], "encrypted_content": blob}


def _rs_message(
    text: str, msg_id: str = "msg_1", phase: str | None = "final_answer"
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "message",
        "id": msg_id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    if phase is not None:
        item["phase"] = phase
    return item


def _rs_call(call_id: str, name: str, arguments: str, fc_id: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": fc_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _rs_completed(output: list[dict[str, Any]], *, status: str = "completed") -> dict[str, Any]:
    response: dict[str, Any] = {
        "id": "resp_1",
        "object": "response",
        "status": status,
        "output": output,
        "usage": _RS_USAGE,
    }
    return response


def _rs_text_events(text: str) -> list[dict[str, Any]]:
    message = _rs_message(text)
    return [
        {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**message, "content": []},
        },
        *(
            {"type": "response.output_text.delta", "output_index": 0, "delta": piece}
            for piece in (text[: len(text) // 2], text[len(text) // 2 :])
        ),
        {"type": "response.output_item.done", "output_index": 0, "item": message},
        {"type": "response.completed", "response": _rs_completed([message])},
    ]


def _rs_tool_events(
    calls: list[tuple[str, str, str]],
    *,
    blob: str = "gAAAA-encrypted-1",
    preamble: str | None = None,
) -> list[dict[str, Any]]:
    """A tool turn: a reasoning item at output_index 0, an optional preamble, then the calls."""
    reasoning = _rs_reasoning(blob=blob)
    events: list[dict[str, Any]] = [
        {"type": "response.output_item.added", "output_index": 0, "item": {**reasoning}},
        {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
    ]
    output: list[dict[str, Any]] = [reasoning]
    if preamble is not None:
        message = _rs_message(preamble, msg_id="msg_pre", phase="commentary")
        events += [
            {"type": "response.output_text.delta", "output_index": 1, "delta": preamble},
            {"type": "response.output_item.done", "output_index": 1, "item": message},
        ]
        output.append(message)
    for call_id, name, arguments in calls:
        index = len(output)
        item = _rs_call(call_id, name, arguments, fc_id=f"fc_{call_id}")
        events += [
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": {**item, "arguments": "", "status": "in_progress"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": index,
                "item_id": item["id"],
                "delta": arguments[:5],
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": index,
                "item_id": item["id"],
                "delta": arguments[5:],
            },
            {"type": "response.output_item.done", "output_index": index, "item": item},
        ]
        output.append(item)
    events.append({"type": "response.completed", "response": _rs_completed(output)})
    return events


def _aresponses_stream(*turns: list[dict[str, Any]]) -> AsyncMock:
    """An ``litellm.aresponses`` stand-in: each call streams the next turn's events."""
    remaining = list(turns)

    async def _side_effect(**_kwargs: Any) -> Any:
        events = remaining.pop(0)

        async def _gen() -> AsyncIterator[dict[str, Any]]:
            for event in events:
                yield event

        return _gen()

    return AsyncMock(side_effect=_side_effect)


class TestResponsesRoute:
    """Routes with ``wire_api: responses`` call ``litellm.aresponses``, never ``acompletion``."""

    @pytest.fixture(autouse=True)
    def _never_chat_completions(self, mock_litellm: AsyncMock) -> Iterator[None]:
        # A silent fall back to Chat Completions fails here at once, not after real retries.
        yield
        assert mock_litellm.await_count == 0, "a Responses route called litellm.acompletion"

    async def test_complete_calls_aresponses_with_a_stateless_body(
        self,
        mock_litellm: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        aresponses = AsyncMock(return_value=_rs_completed([_rs_message("hi there")]))
        monkeypatch.setattr("litellm.aresponses", aresponses)
        client = LLMClient("gpt-6.1-sol")
        resp = await client.complete(
            [Message.system("be brief"), Message.user("hi")],
            max_tokens=25_000,
        )
        assert mock_litellm.await_count == 0
        kwargs = _kwargs(aresponses)
        assert kwargs["model"] == "openai/gpt-6.1-sol"
        assert kwargs["store"] is False
        assert kwargs["include"] == ["reasoning.encrypted_content"]
        assert kwargs["instructions"] == "be brief"
        assert kwargs["input"] == [{"type": "message", "role": "user", "content": "hi"}]
        assert kwargs["max_output_tokens"] == 25_000
        for absent in ("messages", "max_tokens", "temperature", "previous_response_id", "stream"):
            assert absent not in kwargs
        assert resp.text == "hi there"
        assert resp.finish_reason == "stop"
        assert resp.route.wire_api == "responses"
        assert resp.provider_items is not None
        assert resp.provider_items.items[0] == TextItem(
            id="msg_1", phase="final_answer", text="hi there"
        )

    async def test_the_diagnostic_dump_redacts_encrypted_reasoning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        dump = tmp_path / "diag.ndjson"
        monkeypatch.setenv("FORGE_DIAGNOSTIC_ENABLED", "true")
        monkeypatch.setenv("FORGE_DIAGNOSTIC_PATH", str(dump))
        aresponses = AsyncMock(
            return_value=_rs_completed([_rs_reasoning(blob="gAAAA-fresh-blob"), _rs_message("ok")])
        )
        monkeypatch.setattr("litellm.aresponses", aresponses)
        prior = AssistantMessage(
            content="Let me check.",
            tool_calls=[ToolCall(id="call_1", name="_get_weather", arguments={"location": "Oslo"})],
            provider_items=ProviderItems(
                provider="openai",
                items=(
                    ReasoningItem(id="rs_0", encrypted_content="gAAAA-replayed-blob"),
                    CallRef(id="fc_1", call_id="call_1"),
                ),
            ),
        )
        await LLMClient("gpt-6.1-sol").complete(
            [
                Message.system("be brief"),
                Message.user("weather?"),
                prior,
                ToolResultMessage(tool_call_id="call_1", content="18C"),
            ]
        )
        # The request itself carried the blob; only the dump is redacted.
        sent = json.dumps(_kwargs(aresponses)["input"])
        assert "gAAAA-replayed-blob" in sent
        record = json.loads(dump.read_text(encoding="utf-8").splitlines()[-1])
        text = json.dumps(record)
        assert "gAAAA-replayed-blob" not in text
        assert "gAAAA-fresh-blob" not in text
        assert f"<{len('gAAAA-replayed-blob')} bytes>" in text
        assert record["messages"][0] == {"type": "instructions", "content": "be brief"}

    def test_a_turn_whose_items_name_an_unknown_call_keeps_only_its_text_and_calls(self) -> None:
        from strata_forge.llm.client import _assistant_turn  # pyright: ignore[reportPrivateUsage]

        calls = [ToolCall(id="call_1", name="_get_weather", arguments={})]
        items = ProviderItems(
            provider="openai",
            items=(
                ReasoningItem(id="rs_1", encrypted_content="blob"),
                CallRef(id="fc_9", call_id="call_9"),
            ),
        )
        turn = _assistant_turn("checking", calls, items)
        assert turn.provider_items is None
        assert turn.content == "checking"
        assert [call.id for call in turn.tool_calls] == ["call_1"]
        kept = _assistant_turn(
            "checking",
            calls,
            ProviderItems(provider="openai", items=(CallRef(id="fc_1", call_id="call_1"),)),
        )
        assert kept.provider_items is not None

    async def test_the_cache_never_stores_provider_items(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        aresponses = AsyncMock(
            return_value=_rs_completed([_rs_reasoning(), _rs_message("hi there")])
        )
        monkeypatch.setattr("litellm.aresponses", aresponses)
        cache = InMemoryCache()
        client = LLMClient("gpt-6.1-sol", cache=cache)
        fresh = await client.complete([Message.user("hi")])
        assert fresh.provider_items is not None  # the caller still gets this turn's items
        hit = await client.complete([Message.user("hi")])
        assert hit.cache_hit
        assert hit.text == "hi there"
        assert hit.provider_items is None
        assert aresponses.await_count == 1

    async def test_usage_is_net_of_cache_and_priced_per_bucket(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        aresponses = AsyncMock(return_value=_rs_completed([_rs_message("ok")]))
        monkeypatch.setattr("litellm.aresponses", aresponses)
        resp = await LLMClient("gpt-6.1-sol").complete([Message.user("hi")])
        assert resp.usage.input_tokens == 700
        assert resp.usage.cache_read_tokens == 200
        assert resp.usage.cache_write_tokens == 100
        assert resp.usage.output_tokens == 300
        # gpt-6.1-sol: $2 input, $0.10 cached, $2.50 cache write, $10 output per M.
        expected = (700 * 2.00 + 200 * 0.10 + 100 * 2.50 + 300 * 10.00) / 1_000_000
        assert abs(resp.cost_usd - expected) < 1e-12

    async def test_tools_are_internally_tagged_and_not_strict(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        call = _rs_call("call_1", "_get_weather", '{"location": "Oslo"}', fc_id="fc_1")
        aresponses = AsyncMock(return_value=_rs_completed([_rs_reasoning(), call]))
        monkeypatch.setattr("litellm.aresponses", aresponses)
        resp = await LLMClient("gpt-6.1-sol").complete([Message.user("w?")], tools=[_get_weather])
        tool_schema = _kwargs(aresponses)["tools"][0]
        assert tool_schema["type"] == "function"
        assert tool_schema["name"] == "_get_weather"
        assert tool_schema["strict"] is False
        assert "function" not in tool_schema
        assert resp.finish_reason == "tool_use"
        assert resp.tool_calls == [
            ToolCall(id="call_1", name="_get_weather", arguments={"location": "Oslo"})
        ]

    async def test_luna_tool_calls_pin_no_reasoning_effort(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        aresponses = AsyncMock(return_value=_rs_completed([_rs_message("ok")]))
        monkeypatch.setattr("litellm.aresponses", aresponses)
        await LLMClient("gpt-6-luna").complete([Message.user("hi")], tools=[_get_weather])
        kwargs = _kwargs(aresponses)
        assert "reasoning" not in kwargs
        assert "reasoning_effort" not in kwargs

    async def test_reasoning_effort_extra_becomes_reasoning_effort_object(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        aresponses = AsyncMock(return_value=_rs_completed([_rs_message("ok")]))
        monkeypatch.setattr("litellm.aresponses", aresponses)
        await LLMClient("gpt-6.1-sol").complete(
            [Message.user("hi")],
            provider_extras={"openai": {"reasoning_effort": "high", "service_tier": "flex"}},
        )
        kwargs = _kwargs(aresponses)
        assert kwargs["reasoning"] == {"effort": "high"}
        assert "reasoning_effort" not in kwargs
        assert kwargs["service_tier"] == "flex"

    async def test_structured_output_uses_text_format(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload = json.dumps({"title": "T", "bullets": ["a"]})
        aresponses = AsyncMock(return_value=_rs_completed([_rs_message(payload)]))
        monkeypatch.setattr("litellm.aresponses", aresponses)
        resp = await LLMClient("gpt-6.1-sol").complete_structured(
            [Message.user("hi")], schema=_Summary
        )
        assert resp.parsed.title == "T"
        text_format = _kwargs(aresponses)["text"]["format"]
        assert text_format["type"] == "json_schema"
        assert text_format["name"] == "_Summary"
        assert text_format["strict"] is True
        assert "response_format" not in _kwargs(aresponses)

    async def test_failed_response_raises_a_mapped_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from strata_forge.core.errors import FallbackExhaustedError

        failed: dict[str, Any] = {
            "id": "resp_1",
            "status": "failed",
            "output": [],
            "error": {"code": "rate_limit_exceeded", "message": "slow down"},
        }
        monkeypatch.setattr("litellm.aresponses", AsyncMock(return_value=failed))
        client = LLMClient(
            "gpt-6.1-sol", retry_max_attempts=1, retry_initial_wait=0.0, retry_max_wait=0.0
        )
        with pytest.raises(FallbackExhaustedError) as info:
            await client.complete([Message.user("hi")])
        assert [type(cause) for _, _, cause in info.value.causes] == [ProviderRateLimitError]

    async def test_stream_yields_text_then_a_final_chunk(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("litellm.aresponses", _aresponses_stream(_rs_text_events("hello!")))
        chunks = [c async for c in await LLMClient("gpt-6.1-sol").stream([Message.user("hi")])]
        assert "".join(c.delta_text for c in chunks) == "hello!"
        final = chunks[-1]
        assert final.finish_reason == "stop"
        assert final.usage is not None
        assert final.usage.input_tokens == 700
        assert final.provider_items is not None

    async def test_stream_cut_before_the_terminal_event_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from strata_forge.core.errors import ProviderServerError

        events = _rs_text_events("hello!")[:-1]
        monkeypatch.setattr("litellm.aresponses", _aresponses_stream(events))
        stream = await LLMClient("gpt-6.1-sol").stream([Message.user("hi")])
        with pytest.raises(ProviderServerError, match="ended before the response completed"):
            _ = [c async for c in stream]

    async def test_stream_failed_event_raises_rate_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        failed: dict[str, Any] = {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {"code": "rate_limit_exceeded", "message": "slow down"},
            },
        }
        monkeypatch.setattr("litellm.aresponses", _aresponses_stream([failed]))
        stream = await LLMClient("gpt-6.1-sol").stream([Message.user("hi")])
        with pytest.raises(ProviderRateLimitError, match="slow down"):
            _ = [c async for c in stream]

    async def test_stream_transport_error_is_mapped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from litellm.exceptions import RateLimitError as LLRateLimitError

        async def _fail(**_kwargs: Any) -> Any:
            raise LLRateLimitError(message="429", model="gpt-6.1-sol", llm_provider="openai")

        monkeypatch.setattr("litellm.aresponses", AsyncMock(side_effect=_fail))
        stream = await LLMClient("gpt-6.1-sol").stream([Message.user("hi")])
        with pytest.raises(ProviderRateLimitError):
            _ = [c async for c in stream]

    async def test_incomplete_turn_ends_the_loop_with_length(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        incomplete = {
            "type": "response.incomplete",
            "response": {
                **_rs_completed([_rs_reasoning()], status="incomplete"),
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        }
        monkeypatch.setattr("litellm.aresponses", _aresponses_stream([incomplete]))
        events = await _collect(
            LLMClient("gpt-6.1-sol").stream_tool_loop([Message.user("hi")], tools=[_get_weather])
        )
        assert isinstance(events[-1], Done)
        assert events[-1].finish_reason == "length"

    async def test_stream_tool_loop_replays_reasoning_within_a_leg(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _aresponses_stream(
            _rs_tool_events(
                [("call_1", "_get_weather", '{"location": "Oslo"}')],
                preamble="Checking the weather.",
            ),
            _rs_text_events("It is 18C in Oslo."),
        )
        monkeypatch.setattr("litellm.aresponses", mock)
        events = await _collect(
            LLMClient("gpt-6.1-sol").stream_tool_loop(
                [Message.user("weather in Oslo?")], tools=[_get_weather]
            )
        )
        assert isinstance(events[-1], Done)
        assert events[-1].finish_reason == "stop"
        assert mock.await_count == 2
        second = mock.await_args_list[1].kwargs
        assert second["stream"] is True
        assert second["input"] == [
            {"type": "message", "role": "user", "content": "weather in Oslo?"},
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": "gAAAA-encrypted-1",
            },
            {
                "type": "message",
                "id": "msg_pre",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Checking the weather.",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
                "phase": "commentary",
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "_get_weather",
                "arguments": json.dumps({"location": "Oslo"}),
                "id": "fc_call_1",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": json.dumps({"temp_c": 18, "city": "Oslo"}),
            },
        ]

    async def test_parallel_calls_open_dense_slots_in_output_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _aresponses_stream(
            _rs_tool_events(
                [
                    ("call_a", "_get_weather", '{"location": "Oslo"}'),
                    ("call_b", "_get_weather", '{"location": "Rome"}'),
                ]
            ),
            _rs_text_events("done"),
        )
        monkeypatch.setattr("litellm.aresponses", mock)
        events = await _collect(
            LLMClient("gpt-6.1-sol").stream_tool_loop([Message.user("w?")], tools=[_get_weather])
        )
        started = [e for e in events if isinstance(e, ToolCallStarted)]
        assert [(e.id, e.arguments["location"]) for e in started] == [
            ("call_a", "Oslo"),
            ("call_b", "Rome"),
        ]
        results = [e for e in events if isinstance(e, ToolResult)]
        assert [r.id for r in results] == ["call_a", "call_b"]
        second_input = mock.await_args_list[1].kwargs["input"]
        outputs = [i["call_id"] for i in second_input if i["type"] == "function_call_output"]
        assert outputs == ["call_a", "call_b"]

    async def test_suspension_carries_provider_items_and_resume_replays_them(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _aresponses_stream(
            _rs_tool_events([("call_1", "load_model", '{"repo_id": "gpt2"}')]),
            _rs_text_events("Loaded."),
        )
        monkeypatch.setattr("litellm.aresponses", mock)
        client = LLMClient("gpt-6.1-sol")
        prompt = [Message.user("load gpt2")]
        events = await _collect(client.stream_tool_loop(prompt, tools=[_LOAD_MODEL]))
        pending = events[-1]
        assert isinstance(pending, PendingToolCalls)
        turn = pending.messages[0]
        assert isinstance(turn, AssistantMessage)
        assert turn.provider_items is not None
        assert turn.provider_items.provider == "openai"
        assert turn.provider_items.items == (
            ReasoningItem(id="rs_1", encrypted_content="gAAAA-encrypted-1"),
            CallRef(id="fc_call_1", call_id="call_1"),
        )

        # The caller serializes the delta (e.g. into a continuation token) and resumes.
        restored = [
            AssistantMessage.model_validate_json(m.model_dump_json()) for m in pending.messages
        ]
        resumed = [*prompt, *restored, ToolResultMessage(tool_call_id="call_1", content="ok")]
        events = await _collect(client.stream_tool_loop(resumed, tools=[_LOAD_MODEL]))
        assert isinstance(events[-1], Done)
        replay = mock.await_args_list[1].kwargs["input"]
        assert replay[1]["encrypted_content"] == "gAAAA-encrypted-1"
        assert replay[2]["id"] == "fc_call_1"
        assert replay[3] == {"type": "function_call_output", "call_id": "call_1", "output": "ok"}

    async def test_turn_from_another_provider_is_replayed_without_items(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        aresponses = AsyncMock(return_value=_rs_completed([_rs_message("ok")]))
        monkeypatch.setattr("litellm.aresponses", aresponses)
        azure_turn = AssistantMessage(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="_get_weather", arguments={"location": "X"})],
            provider_items=ProviderItems(
                provider="azure",
                items=(
                    ReasoningItem(id="rs_9", encrypted_content="azure-blob"),
                    CallRef(id="fc_9", call_id="call_1"),
                ),
            ),
        )
        await LLMClient("gpt-6.1-sol").complete(
            [
                Message.user("w?"),
                azure_turn,
                ToolResultMessage(tool_call_id="call_1", content="sunny"),
            ],
            tools=[_get_weather],
        )
        sent = _kwargs(aresponses)["input"]
        assert all(item["type"] != "reasoning" for item in sent)
        assert sent[1] == {
            "type": "function_call",
            "call_id": "call_1",
            "name": "_get_weather",
            "arguments": json.dumps({"location": "X"}),
        }

    async def test_run_tool_loop_keeps_provider_items(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        call = _rs_call("call_1", "_get_weather", '{"location": "Oslo"}', fc_id="fc_1")
        aresponses = AsyncMock(
            side_effect=[
                _rs_completed([_rs_reasoning(blob="blob-x"), call]),
                _rs_completed([_rs_message("18C")]),
            ]
        )
        monkeypatch.setattr("litellm.aresponses", aresponses)
        resp = await LLMClient("gpt-6.1-sol").run_tool_loop(
            [Message.user("w?")], tools=[_get_weather]
        )
        assert resp.text == "18C"
        second_input = aresponses.await_args_list[1].kwargs["input"]
        assert second_input[1]["encrypted_content"] == "blob-x"
        assert second_input[2]["id"] == "fc_1"


class TestChatCompletionsBesideResponses:
    """Chat Completions routes keep their wire when Responses API turns are around them."""

    async def test_openai_compat_pin_of_the_same_id_stays_on_chat_completions(
        self,
        mock_litellm: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        aresponses = AsyncMock()
        monkeypatch.setattr("litellm.aresponses", aresponses)
        await LLMClient("gpt-6.1-sol", provider="openai_compat").complete([Message.user("hi")])
        assert aresponses.await_count == 0
        assert mock_litellm.await_count == 1

    async def test_a_turn_with_provider_items_reaches_chat_completions_as_plain_history(
        self, mock_litellm: AsyncMock
    ) -> None:
        # A conversation resumed on another provider carries the OpenAI turn's items.
        turn = AssistantMessage(
            content="Let me check.",
            tool_calls=[ToolCall(id="call_1", name="_get_weather", arguments={"location": "Oslo"})],
            provider_items=ProviderItems(
                provider="openai",
                items=(
                    ReasoningItem(id="rs_1", encrypted_content="gAAAA-secret-blob"),
                    TextItem(id="msg_1", phase="commentary", text="Let me check."),
                    CallRef(id="fc_1", call_id="call_1"),
                ),
            ),
        )
        messages = [
            Message.user("weather?"),
            turn,
            ToolResultMessage(tool_call_id="call_1", content="18C"),
        ]
        for model in ("claude-opus-4-7", "gpt-6.1-sol"):
            provider = "openai_compat" if model.startswith("gpt") else None
            await LLMClient(model, provider=provider).complete(messages)
            wire = _kwargs(mock_litellm)["messages"]
            assistant = wire[1]
            assert assistant["role"] == "assistant"
            assert set(assistant) <= {"role", "content", "tool_calls"}
            assert assistant["tool_calls"][0]["id"] == "call_1"
            assert "gAAAA-secret-blob" not in json.dumps(wire)
            assert "rs_1" not in json.dumps(wire)

    async def test_fallback_from_chat_completions_to_a_responses_route(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from litellm.exceptions import RateLimitError as LLRateLimitError

        acompletion = AsyncMock(
            side_effect=LLRateLimitError(message="429", model="claude", llm_provider="anthropic")
        )
        aresponses = AsyncMock(return_value=_rs_completed([_rs_message("from sol")]))
        monkeypatch.setattr("litellm.acompletion", acompletion)
        monkeypatch.setattr("litellm.aresponses", aresponses)
        client = LLMClient(
            chain=["claude-opus-4-7", "gpt-6.1-sol"],
            retry_max_attempts=1,
            retry_initial_wait=0.0,
            retry_max_wait=0.0,
        )
        resp = await client.complete([Message.user("hi")], max_tokens=1_000)
        assert resp.text == "from sol"
        assert resp.route.model == "gpt-6.1-sol"
        assert resp.route.wire_api == "responses"
        assert acompletion.await_count == 1
        assert _kwargs(aresponses)["max_output_tokens"] == 1_000
        assert "messages" not in _kwargs(aresponses)


class TestSamplingParamsGuard:
    """``temperature`` / ``top_p`` are refused pre-flight where the model rejects them."""

    @pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-6-astra", "gpt-5.5", "claude-opus-5-5"])
    async def test_refused_at_the_default_effort(
        self,
        model: str,
        mock_litellm: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        aresponses = AsyncMock()
        monkeypatch.setattr("litellm.aresponses", aresponses)
        client = LLMClient(model)
        with pytest.raises(ValidationError, match="does not accept temperature or top_p"):
            await client.complete([Message.user("hi")], temperature=0.2)
        with pytest.raises(ValidationError):
            await client.stream([Message.user("hi")], top_p=0.9)
        with pytest.raises(ValidationError):
            await _collect(client.stream_tool_loop([Message.user("hi")], tools=[], temperature=0))
        assert aresponses.await_count == 0
        assert mock_litellm.await_count == 0

    async def test_allowed_when_the_caller_sets_effort_none(
        self,
        mock_litellm: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        aresponses = AsyncMock(return_value=_rs_completed([_rs_message("ok")]))
        monkeypatch.setattr("litellm.aresponses", aresponses)
        await LLMClient("gpt-6-luna").complete(
            [Message.user("hi")],
            temperature=0.2,
            provider_extras={"openai": {"reasoning": {"effort": "none"}}},
        )
        kwargs = _kwargs(aresponses)
        assert kwargs["temperature"] == 0.2
        assert kwargs["reasoning"] == {"effort": "none"}
        assert mock_litellm.await_count == 0

    async def test_allowed_on_a_model_whose_default_effort_is_none(
        self,
        mock_litellm: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        aresponses = AsyncMock(return_value=_rs_completed([_rs_message("ok")]))
        monkeypatch.setattr("litellm.aresponses", aresponses)
        await LLMClient("gpt-5.5-instant").complete([Message.user("hi")], temperature=0.3)
        assert _kwargs(aresponses)["temperature"] == 0.3
        assert mock_litellm.await_count == 0

    async def test_refused_with_an_explicit_effort_other_than_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("litellm.aresponses", AsyncMock())
        with pytest.raises(ValidationError, match="reasoning effort to 'none'"):
            await LLMClient("gpt-5.5-instant").complete(
                [Message.user("hi")],
                top_p=0.5,
                provider_extras={"openai": {"reasoning_effort": "low"}},
            )

    async def test_openai_compat_pin_is_not_guarded(self, mock_litellm: AsyncMock) -> None:
        await LLMClient("gpt-6.1-sol", provider="openai_compat").complete(
            [Message.user("hi")], temperature=0.2
        )
        assert _kwargs(mock_litellm)["temperature"] == 0.2


class TestRegistryCapabilityGates:
    async def test_structured_output_refused_preflight(self, mock_litellm: AsyncMock) -> None:
        client = LLMClient("claude-opus-5-5")
        with pytest.raises(RegistryError, match="does not support structured output") as info:
            await client.complete_structured([Message.user("hi")], schema=_Summary)
        assert info.value.reason == "capability_missing"
        assert mock_litellm.await_count == 0

    async def test_openai_compat_pin_skips_the_registry_tool_gate(
        self,
        mock_litellm: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # An operator id that happens to equal a registered tool-less name is the
        # operator's model, not the registered route.
        from strata_forge.llm.registry import Capabilities, Model, Pricing, ProviderRoute, Registry

        toolless = Model(
            name="toolless",
            vendor="openai",
            tier="fast",
            context_window=1024,
            max_output_tokens=256,
            modalities=["text"],
            capabilities=Capabilities(tool_calling=False),
            pricing_per_million_tokens=Pricing(input=1.0, output=1.0),
            routes=[ProviderRoute(provider="openai", provider_model_id="t", is_default=True)],
        )
        monkeypatch.setattr("strata_forge.llm.client.registry", Registry([toolless]))
        await LLMClient("toolless", provider="openai_compat").complete(
            [Message.user("hi")], tools=[_get_weather]
        )
        assert mock_litellm.await_count == 1
        strict = LLMClient("toolless", provider="openai_compat", require_tool_support=True)
        with pytest.raises(RegistryError) as info:
            await strict.complete([Message.user("hi")], tools=[_get_weather])
        assert info.value.reason == "capability_unknown"
        with pytest.raises(RegistryError) as native:
            await LLMClient("toolless", provider="openai").complete(
                [Message.user("hi")], tools=[_get_weather]
            )
        assert native.value.reason == "capability_missing"


class TestChatCompletionsUsage:
    async def test_input_is_net_of_cache_reads_and_writes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _fake_response(prompt_tokens=1_000, completion_tokens=10, cache_read=600)
        fake.usage.cache_creation_input_tokens = 300
        monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=fake))
        resp = await LLMClient("claude-opus-4-7").complete([Message.user("hi")])
        assert resp.usage.input_tokens == 100
        assert resp.usage.cache_read_tokens == 600
        assert resp.usage.cache_write_tokens == 300


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
            _fake_response(text="from gemini"),
        ]

        async def _side_effect(**_kwargs: Any) -> Any:
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_side_effect))
        client = LLMClient.with_fallbacks(
            ["claude-opus-4-7", "gemini-3.1-pro"],
            retry_max_attempts=1,
            retry_initial_wait=0.0,
            retry_max_wait=0.0,
        )
        resp = await client.complete([Message.user("hi")])
        assert resp.text == "from gemini"
        assert resp.route.model == "gemini-3.1-pro"


# ---------------------------------------------------------------------------
# Budget integration
# ---------------------------------------------------------------------------


class TestBudgetIntegration:
    async def test_budget_consumes_on_success(self, mock_litellm: AsyncMock) -> None:
        client = LLMClient("claude-opus-4-7")
        async with BudgetContext(max_usd=1.00) as budget:
            await client.complete([Message.user("hi")])
            assert budget.spent_usd > 0

    async def test_budget_tokens_count_cached_input(self, mock_litellm: AsyncMock) -> None:
        # 50 prompt tokens of which 40 were cache reads: the input is reported net (10), but a
        # token ceiling still counts all 50 plus the output.
        mock_litellm.return_value = _fake_response(prompt_tokens=50, cache_read=40)
        client = LLMClient("claude-opus-4-7")
        async with BudgetContext(max_tokens=1_000) as budget:
            resp = await client.complete([Message.user("hi")])
        assert resp.usage.input_tokens == 10
        assert budget.spent_tokens == 52

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
        client = LLMClient("gpt-5.5", provider="openai_compat")
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
        client = LLMClient("gpt-5.5", provider="openai_compat")
        resp = await client.complete_structured(
            [Message.user("hi")], schema=_Summary, max_reprompt_attempts=3
        )
        assert resp.parsed.title == "Recovered"

    async def test_reprompt_exhausted_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from strata_forge.llm.schemas import StructuredOutputError

        # Always return invalid JSON.
        async def _fake(**_kwargs: Any) -> Any:
            return _fake_response(text="still not json")

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("gpt-5.5", provider="openai_compat")
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
        from strata_forge.llm.schemas import StructuredOutputError

        # The model returns absolutely nothing usable on every attempt.
        async def _fake(**_kwargs: Any) -> Any:
            return _fake_response(text="", finish_reason="stop")

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("gpt-5.5", provider="openai_compat")
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
                tool_calls=[{"id": "c1", "name": "_explodes", "arguments": {"x": 1}}],
            ),
            _fake_response(text="recovered"),
        ]

        async def _fake(**_kwargs: Any) -> Any:
            return responses.pop(0)

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("claude-opus-4-7")
        resp = await client.run_tool_loop([Message.user("?")], tools=[_explodes], max_iterations=3)
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
# Streaming tool loop
# ---------------------------------------------------------------------------


def _stream_text(
    text: str,
    *,
    finish_reason: str | None = None,
    usage: Any = None,
) -> types.SimpleNamespace:
    """A streamed text chunk in LiteLLM's delta shape."""
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                delta=types.SimpleNamespace(content=text, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def _stream_tool(
    index: int,
    *,
    id: str | None = None,  # noqa: A002 — mirrors the wire field name
    name: str | None = None,
    args: str = "",
    finish_reason: str | None = None,
) -> types.SimpleNamespace:
    """A streamed tool-call delta chunk in LiteLLM's delta shape."""
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                delta=types.SimpleNamespace(
                    content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            index=index,
                            id=id,
                            function=types.SimpleNamespace(name=name, arguments=args),
                        )
                    ],
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=_usage_ns() if finish_reason is not None else None,
    )


def _usage_ns(prompt: int = 5, completion: int = 3) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=0),
    )


def _streaming_acompletion(*turns: list[Any]) -> AsyncMock:
    """Mock `litellm.acompletion` to yield one canned chunk stream per call.

    Each positional arg is the list of raw chunk namespaces for one
    ``stream()`` turn (i.e. one loop iteration). Calls beyond the supplied
    turns raise ``IndexError`` so an over-iterating loop fails loudly.
    """
    queue = [list(turn) for turn in turns]

    async def _fake(**_kwargs: Any) -> Any:
        chunks = queue.pop(0)

        async def _gen() -> AsyncIterator[Any]:
            for chunk in chunks:
                yield chunk

        return _gen()

    return AsyncMock(side_effect=_fake)


async def _collect(events: AsyncIterator[Any]) -> list[Any]:
    return [event async for event in events]


class TestStreamToolLoop:
    async def test_no_tool_use_emits_text_then_done(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "litellm.acompletion",
            _streaming_acompletion(
                [_stream_text("hel"), _stream_text("lo", finish_reason="stop")],
            ),
        )
        client = LLMClient("claude-opus-4-7")
        events = await _collect(client.stream_tool_loop([Message.user("hi")], tools=[_get_weather]))
        assert isinstance(events[0], IterationStart)
        assert events[0].index == 0
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "hello"
        assert isinstance(events[-1], Done)
        assert events[-1].finish_reason == "stop"
        assert not any(isinstance(e, (ToolCallStarted, ToolResult)) for e in events)

    async def test_single_tool_then_text_full_event_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "litellm.acompletion",
            _streaming_acompletion(
                [
                    _stream_tool(0, id="c1", name="_get_weather", args='{"location":'),
                    _stream_tool(0, args=' "Tokyo"}', finish_reason="tool_calls"),
                ],
                [_stream_text("It's sunny.", finish_reason="stop")],
            ),
        )
        client = LLMClient("claude-opus-4-7")
        events = await _collect(
            client.stream_tool_loop([Message.user("weather?")], tools=[_get_weather])
        )
        assert [type(e).__name__ for e in events] == [
            "IterationStart",
            "ToolCallStarted",
            "ToolResult",
            "IterationStart",
            "TextDelta",
            "Done",
        ]
        call = events[1]
        assert call.id == "c1"
        assert call.name == "_get_weather"
        assert call.arguments == {"location": "Tokyo"}
        assert call.iteration == 0
        result = events[2]
        assert result.id == "c1"
        assert result.is_error is False
        assert result.iteration == 0
        assert "Tokyo" in result.content  # the stringified tool return
        assert events[3].index == 1
        assert events[-1].finish_reason == "stop"

    async def test_streamed_args_assembled_whole_not_partial(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "litellm.acompletion",
            _streaming_acompletion(
                [
                    _stream_tool(0, id="c1", name="_get_weather", args='{"loc'),
                    _stream_tool(0, args='ation": "T'),
                    _stream_tool(0, args='okyo"}', finish_reason="tool_calls"),
                ],
                [_stream_text("done", finish_reason="stop")],
            ),
        )
        client = LLMClient("claude-opus-4-7")
        events = await _collect(client.stream_tool_loop([Message.user("?")], tools=[_get_weather]))
        starts = [e for e in events if isinstance(e, ToolCallStarted)]
        assert len(starts) == 1
        assert starts[0].arguments == {"location": "Tokyo"}

    async def test_tool_exception_becomes_error_tool_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "litellm.acompletion",
            _streaming_acompletion(
                [
                    _stream_tool(
                        0, id="c1", name="_explodes", args='{"x": 1}', finish_reason="tool_calls"
                    )
                ],
                [_stream_text("recovered", finish_reason="stop")],
            ),
        )
        client = LLMClient("claude-opus-4-7")
        events = await _collect(client.stream_tool_loop([Message.user("?")], tools=[_explodes]))
        result = next(e for e in events if isinstance(e, ToolResult))
        assert result.is_error is True
        assert "RuntimeError" in result.content
        assert "oops at 1" in result.content
        assert isinstance(events[-1], Done)

    async def test_unknown_tool_emits_error_tool_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "litellm.acompletion",
            _streaming_acompletion(
                [
                    _stream_tool(
                        0, id="c1", name="missing_tool", args="{}", finish_reason="tool_calls"
                    )
                ],
                [_stream_text("oops", finish_reason="stop")],
            ),
        )
        client = LLMClient("claude-opus-4-7")
        events = await _collect(client.stream_tool_loop([Message.user("?")], tools=[_get_weather]))
        result = next(e for e in events if isinstance(e, ToolResult))
        assert result.is_error is True
        assert "is not registered" in result.content
        assert isinstance(events[-1], Done)

    async def test_max_iterations_zero_raises_value_error(self) -> None:
        client = LLMClient("claude-opus-4-7")
        with pytest.raises(ValueError, match=">= 1"):
            await _collect(
                client.stream_tool_loop(
                    [Message.user("hi")], tools=[_get_weather], max_iterations=0
                )
            )

    async def test_loop_exhausted_emits_terminal_loop_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _tool_turn() -> list[Any]:
            return [
                _stream_tool(
                    0,
                    id="c1",
                    name="_get_weather",
                    args='{"location": "X"}',
                    finish_reason="tool_calls",
                )
            ]

        monkeypatch.setattr(
            "litellm.acompletion", _streaming_acompletion(_tool_turn(), _tool_turn())
        )
        client = LLMClient("claude-opus-4-7")
        events = await _collect(
            client.stream_tool_loop([Message.user("?")], tools=[_get_weather], max_iterations=2)
        )
        assert isinstance(events[-1], LoopError)
        assert events[-1].exceeded_max_iterations is True
        assert events[-1].error_type == "ToolLoopExceededError"
        assert not any(isinstance(e, Done) for e in events)
        # Exactly one IterationStart per allowed iteration, indices 0..N-1.
        starts = [e for e in events if isinstance(e, IterationStart)]
        assert [s.index for s in starts] == [0, 1]

    async def test_provider_error_mid_stream_emits_loop_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from litellm.exceptions import RateLimitError as LLRateLimitError

        async def _fake(**_kwargs: Any) -> Any:
            async def _gen() -> AsyncIterator[Any]:
                raise LLRateLimitError(message="429", model="x", llm_provider="anthropic")
                yield None  # pragma: no cover

            return _gen()

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("claude-opus-4-7")
        events = await _collect(client.stream_tool_loop([Message.user("hi")], tools=[_get_weather]))
        assert isinstance(events[0], IterationStart)
        assert isinstance(events[-1], LoopError)
        assert events[-1].error_type == "ProviderRateLimitError"
        assert events[-1].exceeded_max_iterations is False

    async def test_malformed_streamed_tool_call_emits_loop_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "litellm.acompletion",
            _streaming_acompletion(
                [
                    _stream_tool(
                        0,
                        id="c1",
                        name="_get_weather",
                        args='{"unclosed":',
                        finish_reason="tool_calls",
                    )
                ],
            ),
        )
        client = LLMClient("claude-opus-4-7")
        events = await _collect(client.stream_tool_loop([Message.user("?")], tools=[_get_weather]))
        assert isinstance(events[-1], LoopError)
        assert events[-1].error_type == "ValidationError"
        assert not any(isinstance(e, ToolCallStarted) for e in events)

    async def test_sampling_and_extras_forwarded_each_iteration(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: list[Mapping[str, Any]] = []
        queue = [
            [
                _stream_tool(
                    0,
                    id="c1",
                    name="_get_weather",
                    args='{"location": "X"}',
                    finish_reason="tool_calls",
                )
            ],
            [_stream_text("done", finish_reason="stop")],
        ]

        async def _fake(**kwargs: Any) -> Any:
            captured.append(kwargs)
            chunks = queue.pop(0)

            async def _gen() -> AsyncIterator[Any]:
                for chunk in chunks:
                    yield chunk

            return _gen()

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("claude-opus-4-7")
        await _collect(
            client.stream_tool_loop(
                [Message.user("?")],
                tools=[_get_weather],
                temperature=0.3,
                max_tokens=50,
                top_p=0.9,
                provider_extras={"anthropic": {"foo": "bar"}},
            )
        )
        assert len(captured) == 2
        for kwargs in captured:
            assert kwargs["temperature"] == 0.3
            assert kwargs["max_tokens"] == 50
            assert kwargs["top_p"] == 0.9
            assert kwargs["foo"] == "bar"

    async def test_capability_gate_raises_before_streaming(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from unittest.mock import patch

        from strata_forge.llm.registry import registry as _registry

        original = _registry.get("claude-opus-4-7")
        synth = original.model_copy(
            update={
                "capabilities": original.capabilities.model_copy(update={"tool_calling": False})
            }
        )

        def _fake_get(name: str) -> Any:
            return synth if name == "claude-opus-4-7" else original

        mock = AsyncMock()
        monkeypatch.setattr("litellm.acompletion", mock)
        with patch.object(_registry, "get", _fake_get):
            client = LLMClient("claude-opus-4-7")
            with pytest.raises(RegistryError, match="does not support tool calling"):
                await _collect(client.stream_tool_loop([Message.user("hi")], tools=[_get_weather]))
        assert mock.await_count == 0

    async def test_invalid_conversation_raises_before_streaming(self) -> None:
        # A tool-result message with no matching prior tool call is invalid;
        # it must raise pre-flight, before any event (or any provider call).
        client = LLMClient("claude-opus-4-7")
        bad = [ToolResultMessage(tool_call_id="nope", content="x")]
        with pytest.raises(ValidationError, match="unknown tool_call_id"):
            await _collect(client.stream_tool_loop(bad, tools=[_get_weather]))

    async def test_multiple_tool_calls_in_one_iteration(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Two calls arrive in a single tool-use turn (distinct indices); both
        # are assembled, invoked in ascending-index order, and fed back.
        monkeypatch.setattr(
            "litellm.acompletion",
            _streaming_acompletion(
                [
                    _stream_tool(0, id="c1", name="_get_weather", args='{"location": "Tokyo"}'),
                    _stream_tool(
                        1,
                        id="c2",
                        name="_get_weather",
                        args='{"location": "Paris"}',
                        finish_reason="tool_calls",
                    ),
                ],
                [_stream_text("both done", finish_reason="stop")],
            ),
        )
        client = LLMClient("claude-opus-4-7")
        events = await _collect(client.stream_tool_loop([Message.user("?")], tools=[_get_weather]))
        assert [type(e).__name__ for e in events] == [
            "IterationStart",
            "ToolCallStarted",
            "ToolResult",
            "ToolCallStarted",
            "ToolResult",
            "IterationStart",
            "TextDelta",
            "Done",
        ]
        starts = [e for e in events if isinstance(e, ToolCallStarted)]
        assert [s.id for s in starts] == ["c1", "c2"]  # ascending index order
        assert starts[0].arguments == {"location": "Tokyo"}
        assert starts[1].arguments == {"location": "Paris"}
        results = [e for e in events if isinstance(e, ToolResult)]
        assert "Tokyo" in results[0].content
        assert "Paris" in results[1].content
        assert all(not r.is_error for r in results)

    async def test_empty_tools_degenerates_to_single_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: list[Mapping[str, Any]] = []

        async def _fake(**kwargs: Any) -> Any:
            captured.append(kwargs)

            async def _gen() -> AsyncIterator[Any]:
                yield _stream_text("hi", finish_reason="stop")

            return _gen()

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("claude-opus-4-7")
        events = await _collect(client.stream_tool_loop([Message.user("hi")], tools=[]))
        assert [type(e).__name__ for e in events] == ["IterationStart", "TextDelta", "Done"]
        # One turn only, and the provider was never handed a `tools` payload
        # (empty tools collapse to `None`, not an empty list).
        assert len(captured) == 1
        assert "tools" not in captured[0]


# ---------------------------------------------------------------------------
# Streaming tool loop — suspension on declaration-only tools
# ---------------------------------------------------------------------------


class TestStreamToolLoopSuspension:
    async def test_declaration_call_suspends_with_pending(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "litellm.acompletion",
            _streaming_acompletion(
                [
                    _stream_text("Loading it now."),
                    _stream_tool(
                        0,
                        id="c1",
                        name="load_model",
                        args='{"repo_id": "org/m"}',
                        finish_reason="tool_calls",
                    ),
                ],
            ),
        )
        client = LLMClient("claude-opus-4-7")
        events = await _collect(
            client.stream_tool_loop([Message.user("load org/m")], tools=[_get_weather, _LOAD_MODEL])
        )
        assert [type(e).__name__ for e in events] == [
            "IterationStart",
            "TextDelta",
            "ToolCallStarted",
            "PendingToolCalls",
        ]
        assert not any(isinstance(e, ToolResult) for e in events)
        pending = events[-1]
        assert isinstance(pending, PendingToolCalls)
        assert [c.id for c in pending.calls] == ["c1"]
        assert pending.calls[0].arguments == {"repo_id": "org/m"}
        assert pending.iteration == 0
        assert pending.iterations_used == 1
        assert pending.usage is not None
        # The delta carries exactly the assistant turn (text + the call).
        assert len(pending.messages) == 1
        assistant = pending.messages[0]
        assert isinstance(assistant, AssistantMessage)
        assert assistant.content == "Loading it now."
        assert [c.id for c in assistant.tool_calls] == ["c1"]

    async def test_mixed_turn_server_tool_executes_then_suspends(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "litellm.acompletion",
            _streaming_acompletion(
                [
                    _stream_tool(0, id="c1", name="_get_weather", args='{"location": "Tokyo"}'),
                    _stream_tool(
                        1,
                        id="c2",
                        name="load_model",
                        args='{"repo_id": "org/m"}',
                        finish_reason="tool_calls",
                    ),
                ],
            ),
        )
        client = LLMClient("claude-opus-4-7")
        events = await _collect(
            client.stream_tool_loop([Message.user("?")], tools=[_get_weather, _LOAD_MODEL])
        )
        assert [type(e).__name__ for e in events] == [
            "IterationStart",
            "ToolCallStarted",
            "ToolResult",
            "ToolCallStarted",
            "PendingToolCalls",
        ]
        pending = events[-1]
        assert isinstance(pending, PendingToolCalls)
        assert [c.id for c in pending.calls] == ["c2"]
        # Delta: the assistant turn plus the executed server-tool result.
        assert [type(m).__name__ for m in pending.messages] == [
            "AssistantMessage",
            "ToolResultMessage",
        ]
        executed = pending.messages[1]
        assert isinstance(executed, ToolResultMessage)
        assert executed.tool_call_id == "c1"
        assert "Tokyo" in executed.content

    async def test_mixed_turn_client_call_first_still_executes_server_tool(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "litellm.acompletion",
            _streaming_acompletion(
                [
                    _stream_tool(0, id="c1", name="load_model", args='{"repo_id": "org/m"}'),
                    _stream_tool(
                        1,
                        id="c2",
                        name="_get_weather",
                        args='{"location": "Paris"}',
                        finish_reason="tool_calls",
                    ),
                ],
            ),
        )
        client = LLMClient("claude-opus-4-7")
        events = await _collect(
            client.stream_tool_loop([Message.user("?")], tools=[_get_weather, _LOAD_MODEL])
        )
        # Started events keep model order; the server tool still executes
        # even though it follows the suspending client call.
        assert [type(e).__name__ for e in events] == [
            "IterationStart",
            "ToolCallStarted",
            "ToolCallStarted",
            "ToolResult",
            "PendingToolCalls",
        ]
        pending = events[-1]
        assert isinstance(pending, PendingToolCalls)
        assert [c.id for c in pending.calls] == ["c1"]
        result = next(e for e in events if isinstance(e, ToolResult))
        assert result.id == "c2"
        assert "Paris" in result.content

    async def test_unknown_tool_with_client_call_errors_then_suspends(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "litellm.acompletion",
            _streaming_acompletion(
                [
                    _stream_tool(0, id="c1", name="_nope", args="{}"),
                    _stream_tool(
                        1,
                        id="c2",
                        name="load_model",
                        args='{"repo_id": "org/m"}',
                        finish_reason="tool_calls",
                    ),
                ],
            ),
        )
        client = LLMClient("claude-opus-4-7")
        events = await _collect(
            client.stream_tool_loop([Message.user("?")], tools=[_get_weather, _LOAD_MODEL])
        )
        result = next(e for e in events if isinstance(e, ToolResult))
        assert result.id == "c1"
        assert result.is_error is True
        pending = events[-1]
        assert isinstance(pending, PendingToolCalls)
        assert [c.id for c in pending.calls] == ["c2"]
        # The unknown-tool error result is part of the resumption delta.
        assert any(
            isinstance(m, ToolResultMessage) and m.tool_call_id == "c1" and m.is_error
            for m in pending.messages
        )

    async def test_resumption_round_trip(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "litellm.acompletion",
            _streaming_acompletion(
                [
                    _stream_tool(
                        0,
                        id="c1",
                        name="load_model",
                        args='{"repo_id": "org/m"}',
                        finish_reason="tool_calls",
                    ),
                ],
            ),
        )
        client = LLMClient("claude-opus-4-7")
        first_input = [Message.user("load org/m")]
        events = await _collect(
            client.stream_tool_loop(first_input, tools=[_get_weather, _LOAD_MODEL])
        )
        pending = events[-1]
        assert isinstance(pending, PendingToolCalls)

        # Resume with the grown conversation: input + delta + the caller's
        # result for the pending call. This must pass pre-flight validation
        # (the regression validate_conversation is on the hook for).
        monkeypatch.setattr(
            "litellm.acompletion",
            _streaming_acompletion([_stream_text("Loaded.", finish_reason="stop")]),
        )
        resumed = [
            *first_input,
            *pending.messages,
            Message.tool_result("c1", "Loaded org/m: 24 layers."),
        ]
        events2 = await _collect(
            client.stream_tool_loop(
                resumed,
                tools=[_get_weather, _LOAD_MODEL],
                max_iterations=8 - pending.iterations_used,
            )
        )
        assert [type(e).__name__ for e in events2] == ["IterationStart", "TextDelta", "Done"]
        done = events2[-1]
        assert isinstance(done, Done)
        assert done.finish_reason == "stop"

    async def test_duplicate_tool_name_raises_preflight(self) -> None:
        clash = ToolDeclaration(name="_get_weather", description="x", parameters={})
        client = LLMClient("claude-opus-4-7")
        with pytest.raises(ValidationError, match="Duplicate tool name"):
            await _collect(
                client.stream_tool_loop([Message.user("hi")], tools=[_get_weather, clash])
            )

    async def test_suspends_on_last_allowed_iteration(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A suspension on the final budgeted turn is a suspension, not an
        # exceeded-iterations LoopError — the turn completed.
        monkeypatch.setattr(
            "litellm.acompletion",
            _streaming_acompletion(
                [
                    _stream_tool(
                        0,
                        id="c1",
                        name="load_model",
                        args='{"repo_id": "org/m"}',
                        finish_reason="tool_calls",
                    ),
                ],
            ),
        )
        client = LLMClient("claude-opus-4-7")
        events = await _collect(
            client.stream_tool_loop([Message.user("?")], tools=[_LOAD_MODEL], max_iterations=1)
        )
        assert isinstance(events[-1], PendingToolCalls)
        assert not any(isinstance(e, LoopError) for e in events)

    async def test_complete_accepts_declarations(self, mock_litellm: AsyncMock) -> None:
        # Serialization-only smoke: a declaration rides the same `tools=`
        # surface as executable tools on non-loop calls.
        client = LLMClient("claude-opus-4-7")
        await client.complete([Message.user("hi")], tools=[_LOAD_MODEL])
        sent = _kwargs(mock_litellm)["tools"]
        assert sent[0]["name"] == "load_model"
        assert sent[0]["input_schema"]["required"] == ["repo_id"]


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
        from strata_forge.core.errors import FallbackExhaustedError

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
        from strata_forge.llm.messages import TextPart
        from strata_forge.llm.multimodal import ImageContent

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
        from strata_forge.llm.messages import TextPart
        from strata_forge.llm.multimodal import ImageContent

        client = LLMClient("gpt-5.5", provider="openai_compat")
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
        from strata_forge.llm.messages import ToolCall

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
        from strata_forge.llm.providers import AnthropicProvider

        custom = AnthropicProvider()
        custom_dict: dict[ProviderName, ProviderClient] = {"anthropic": custom}
        # Fill remaining providers with stubs so init doesn't crash.
        from strata_forge.llm.providers import (
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


class TestUnpricedOpenAICompatResponse:
    """A response the provider already produced must not be discarded for lack of a price.

    `resolve_route` exempts `openai_compat` from the registry because its model ids are
    operator-specific — a vLLM / TGI deployment names its own model. Pricing had no matching
    exemption, so `_normalize_response` looked the same id up and raised `unknown_model` AFTER
    the call succeeded. A batch run against a local vLLM lost all 2098 of its rows that way:
    every generation succeeded and every one was thrown away on the way back.
    """

    async def test_an_unregistered_openai_compat_model_completes_at_zero_cost(
        self, mock_litellm: AsyncMock
    ) -> None:
        client = LLMClient("Qwen/Qwen2.5-0.5B-Instruct", provider="openai_compat")
        resp = await client.complete([Message.user("hi")])
        assert resp.text == "hello"  # the generation survives
        # 0.0 is what a cache hit already reports, and it is honest for self-hosted serving:
        # the cost there is the machine, not the token.
        assert resp.cost_usd == 0.0
        assert resp.usage.total_tokens == 7  # usage is still real

    async def test_a_registered_model_is_still_priced(self, mock_litellm: AsyncMock) -> None:
        # The fallback must not quietly zero out real spend on a model that HAS a price.
        client = LLMClient("claude-opus-4-7")
        resp = await client.complete([Message.user("hi")])
        assert resp.cost_usd > 0.0

    def test_a_non_exempt_provider_still_raises_on_an_unknown_model(self) -> None:
        # Routing guarantees a registered model for every other provider, so an unknown one there
        # is a real defect rather than the documented exemption — it must stay loud.
        from strata_forge.llm.client import _cost_for_route  # pyright: ignore[reportPrivateUsage]
        from strata_forge.llm.responses import Usage
        from strata_forge.llm.routing import ModelRoute

        route = ModelRoute(
            model="not/in-the-registry", provider="openai", provider_model_id="not/in-the-registry"
        )
        with pytest.raises(RegistryError) as info:
            _cost_for_route(Usage(input_tokens=1, output_tokens=1), route)
        assert info.value.reason == "unknown_model"


# ---------------------------------------------------------------------------
# Exact request bodies through the real LiteLLM Responses path (no network)
# ---------------------------------------------------------------------------


class TestResponsesHttpBody:
    """Drive the real ``litellm.aresponses`` against a fake HTTP server.

    A monkeypatched ``aresponses`` cannot see what LiteLLM does to the body on the way
    out (it rebuilds reasoning input items, for one); this captures the JSON that would
    reach the provider. The fake answers a streamed request with server-sent events and
    any other with JSON.
    """

    _BLOB = "gAAAAABo-opaque+encrypted/reasoning=="

    @pytest.fixture
    def http(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        import litellm
        import respx

        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        litellm.in_memory_llm_clients_cache.flush_cache()
        with respx.mock(assert_all_called=False) as router:
            yield router
        litellm.in_memory_llm_clients_cache.flush_cache()

    def _turns(self) -> list[list[dict[str, Any]]]:
        call = _rs_call("call_1", "_get_weather", '{"location": "Oslo"}', fc_id="fc_1")
        return [
            [_rs_reasoning(blob=self._BLOB), call],
            [_rs_message("18C in Oslo")],
        ]

    def _handler(self, bodies: list[dict[str, Any]]) -> Any:
        import httpx

        turns = self._turns()

        def _respond(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            bodies.append(body)
            output = turns[len(bodies) - 1]
            response = {
                **_rs_completed(output),
                "object": "response",
                "created_at": 1,
                "model": body["model"],
            }
            if not body.get("stream"):
                return httpx.Response(200, json=response)
            events: list[dict[str, Any]] = [
                {"type": "response.output_item.done", "output_index": i, "item": item}
                for i, item in enumerate(output)
            ]
            events.append({"type": "response.completed", "response": response})
            sse = "".join(
                f"event: {e['type']}\ndata: {json.dumps({**e, 'sequence_number': n})}\n\n"
                for n, e in enumerate(events)
            )
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

        return _respond

    def _client(self) -> LLMClient:
        from pydantic import SecretStr

        from strata_forge.llm.providers import OpenAIProvider
        from strata_forge.llm.providers.config import OpenAIConfig

        provider = OpenAIProvider(OpenAIConfig(api_key=SecretStr("sk-test")))
        return LLMClient("gpt-6.1-sol", provider_clients={"openai": provider})

    def _assert_bodies(self, bodies: list[dict[str, Any]]) -> None:
        assert len(bodies) == 2
        first, second = bodies
        assert first["model"] == "gpt-6.1-sol"
        assert first["store"] is False
        assert first["include"] == ["reasoning.encrypted_content"]
        assert first["max_output_tokens"] == 25_000
        assert first["tools"][0]["name"] == "_get_weather"
        assert first["tools"][0]["strict"] is False
        assert first["input"] == [{"type": "message", "role": "user", "content": "weather?"}]
        for body in bodies:
            for absent in (
                "messages",
                "max_tokens",
                "temperature",
                "top_p",
                "previous_response_id",
            ):
                assert absent not in body
        replay = second["input"]
        reasoning = next(item for item in replay if item["type"] == "reasoning")
        assert reasoning["id"] == "rs_1"
        assert reasoning["encrypted_content"] == self._BLOB
        assert "status" not in reasoning
        call = next(item for item in replay if item["type"] == "function_call")
        assert (call["id"], call["call_id"], call["name"]) == ("fc_1", "call_1", "_get_weather")
        output = next(item for item in replay if item["type"] == "function_call_output")
        assert output["call_id"] == "call_1"

    async def test_run_tool_loop_bodies(self, http: Any) -> None:
        bodies: list[dict[str, Any]] = []
        http.post("https://api.openai.com/v1/responses").mock(side_effect=self._handler(bodies))
        resp = await self._client().run_tool_loop(
            [Message.user("weather?")], tools=[_get_weather], max_tokens=25_000
        )
        assert resp.text == "18C in Oslo"
        self._assert_bodies(bodies)

    async def test_stream_tool_loop_bodies(self, http: Any, litellm_map_without: Any) -> None:
        # LiteLLM's bundled map lacks the model, as it lacks every model newer than its
        # release; LiteLLM would then fake the stream from one blocking call.
        litellm_map_without("gpt-6.1-sol")
        bodies: list[dict[str, Any]] = []
        http.post("https://api.openai.com/v1/responses").mock(side_effect=self._handler(bodies))
        events = await _collect(
            self._client().stream_tool_loop(
                [Message.user("weather?")], tools=[_get_weather], max_tokens=25_000
            )
        )
        assert isinstance(events[-1], Done)
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "18C in Oslo"
        self._assert_bodies(bodies)
        assert [body.get("stream") for body in bodies] == [True, True]

    async def test_the_organization_rides_as_a_header(self, http: Any) -> None:
        from pydantic import SecretStr

        from strata_forge.llm.providers import OpenAIProvider
        from strata_forge.llm.providers.config import OpenAIConfig

        bodies: list[dict[str, Any]] = []
        headers: list[httpx.Headers] = []
        respond = self._handler(bodies)

        def _capture(request: httpx.Request) -> httpx.Response:
            headers.append(request.headers)
            return respond(request)

        http.post("https://api.openai.com/v1/responses").mock(side_effect=_capture)
        provider = OpenAIProvider(OpenAIConfig(api_key=SecretStr("sk-test"), org_id="org-123"))
        client = LLMClient("gpt-6.1-sol", provider_clients={"openai": provider})
        await client.complete([Message.user("weather?")])
        assert headers[0]["openai-organization"] == "org-123"
        assert headers[0]["authorization"] == "Bearer sk-test"
        assert "organization" not in bodies[0]

    async def test_azure_uses_the_v1_responses_endpoint(self, http: Any) -> None:
        from pydantic import SecretStr

        from strata_forge.llm.providers import AzureProvider
        from strata_forge.llm.providers.config import AzureConfig

        bodies: list[dict[str, Any]] = []
        route = http.post("https://example.openai.azure.com/openai/v1/responses").mock(
            side_effect=self._handler(bodies)
        )
        azure = AzureProvider(
            AzureConfig(api_key=SecretStr("k"), endpoint="https://example.openai.azure.com")
        )
        client = LLMClient("gpt-5.5", provider="azure", provider_clients={"azure": azure})
        await client.complete([Message.user("weather?")], tools=[_get_weather])
        assert route.called
        assert bodies[0]["model"] == "gpt-5.5"
        assert bodies[0]["store"] is False
