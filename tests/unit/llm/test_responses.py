"""Unit tests for `strata_forge.llm.responses`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from strata_forge.llm.messages import ToolCall
from strata_forge.llm.responses import (
    LLMResponse,
    ResponseChunk,
    ToolCallDelta,
    Usage,
)
from strata_forge.llm.routing import ModelRoute

_ROUTE = ModelRoute(
    model="claude-opus-4-7",
    provider="anthropic",
    provider_model_id="claude-opus-4-7",
)


class TestUsage:
    def test_construction_with_defaults(self) -> None:
        u = Usage(input_tokens=10, output_tokens=20)
        assert u.input_tokens == 10
        assert u.output_tokens == 20
        assert u.cache_read_tokens == 0
        assert u.cache_write_tokens == 0
        assert u.total_tokens == 30

    def test_total_includes_only_input_and_output(self) -> None:
        # Cache reads and writes are tracked separately, not summed into total.
        u = Usage(
            input_tokens=10,
            output_tokens=20,
            cache_read_tokens=100,
            cache_write_tokens=50,
        )
        assert u.total_tokens == 30

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(PydanticValidationError, match="greater than or equal to 0"):
            Usage(input_tokens=-1, output_tokens=10)

    def test_frozen(self) -> None:
        u = Usage(input_tokens=10, output_tokens=20)
        with pytest.raises(PydanticValidationError):
            u.input_tokens = 99  # type: ignore[misc]


class TestLLMResponse:
    def test_minimal_construction(self) -> None:
        r = LLMResponse(
            text="Hello",
            finish_reason="stop",
            usage=Usage(input_tokens=5, output_tokens=10),
            cost_usd=0.001,
            route=_ROUTE,
        )
        assert r.text == "Hello"
        assert r.tool_calls == []
        assert r.cache_hit is False
        assert r.latency_ms == 0.0

    def test_with_tool_calls(self) -> None:
        call = ToolCall(id="c1", name="f", arguments={"a": 1})
        r = LLMResponse(
            text="",
            tool_calls=[call],
            finish_reason="tool_use",
            usage=Usage(input_tokens=5, output_tokens=10),
            cost_usd=0.001,
            route=_ROUTE,
        )
        assert r.finish_reason == "tool_use"
        assert len(r.tool_calls) == 1

    def test_cache_hit_zero_cost(self) -> None:
        r = LLMResponse(
            text="cached",
            finish_reason="stop",
            usage=Usage(input_tokens=5, output_tokens=10),
            cost_usd=0.0,
            route=_ROUTE,
            cache_hit=True,
        )
        assert r.cache_hit is True
        assert r.cost_usd == 0.0

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(PydanticValidationError, match="greater than or equal to 0"):
            LLMResponse(
                text="",
                finish_reason="stop",
                usage=Usage(input_tokens=5, output_tokens=10),
                cost_usd=-0.01,
                route=_ROUTE,
            )

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(PydanticValidationError, match="greater than or equal to 0"):
            LLMResponse(
                text="",
                finish_reason="stop",
                usage=Usage(input_tokens=5, output_tokens=10),
                cost_usd=0.001,
                route=_ROUTE,
                latency_ms=-1,
            )

    def test_route_field_preserved(self) -> None:
        r = LLMResponse(
            text="",
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1),
            cost_usd=0.0,
            route=_ROUTE,
        )
        assert r.route.model == "claude-opus-4-7"
        assert r.route.provider == "anthropic"

    def test_frozen(self) -> None:
        r = LLMResponse(
            text="x",
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1),
            cost_usd=0.0,
            route=_ROUTE,
        )
        with pytest.raises(PydanticValidationError):
            r.text = "y"  # type: ignore[misc]


class TestToolCallDelta:
    def test_first_delta_has_id_and_name(self) -> None:
        d = ToolCallDelta(index=0, id="c1", name="get_weather", arguments_delta='{"loc')
        assert d.index == 0
        assert d.id == "c1"
        assert d.name == "get_weather"

    def test_subsequent_delta_omits_id_and_name(self) -> None:
        d = ToolCallDelta(index=0, arguments_delta='ation": "Tokyo"}')
        assert d.id is None
        assert d.name is None
        assert d.arguments_delta == 'ation": "Tokyo"}'

    def test_negative_index_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            ToolCallDelta(index=-1)


class TestResponseChunk:
    def test_minimal_chunk(self) -> None:
        c = ResponseChunk(delta_text="hello")
        assert c.delta_text == "hello"
        assert c.delta_tool_calls == []
        assert c.finish_reason is None
        assert c.usage is None

    def test_final_chunk_has_finish_and_usage(self) -> None:
        c = ResponseChunk(
            finish_reason="stop",
            usage=Usage(input_tokens=5, output_tokens=10),
        )
        assert c.finish_reason == "stop"
        assert c.usage is not None
        assert c.usage.total_tokens == 15

    def test_tool_call_delta_chunk(self) -> None:
        c = ResponseChunk(
            delta_tool_calls=[
                ToolCallDelta(index=0, id="c1", name="f", arguments_delta='{"x":'),
            ],
        )
        assert len(c.delta_tool_calls) == 1
        assert c.delta_tool_calls[0].id == "c1"
