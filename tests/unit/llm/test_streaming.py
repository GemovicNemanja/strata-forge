"""Unit tests for `forge.llm.streaming`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from forge.core.errors import ValidationError
from forge.llm.responses import ResponseChunk, ToolCallDelta, Usage
from forge.llm.streaming import (
    JSONAccumulator,
    StreamingToolCallAccumulator,
    accumulate_text,
    accumulate_tool_calls,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


async def _async_iter(items: list[ResponseChunk]) -> AsyncIterator[ResponseChunk]:
    for item in items:
        yield item


class TestAccumulateText:
    async def test_joins_delta_text(self) -> None:
        chunks = [
            ResponseChunk(delta_text="Hello"),
            ResponseChunk(delta_text=", "),
            ResponseChunk(delta_text="world!"),
        ]
        result = await accumulate_text(_async_iter(chunks))
        assert result == "Hello, world!"

    async def test_empty_stream(self) -> None:
        result = await accumulate_text(_async_iter([]))
        assert result == ""

    async def test_skips_chunks_without_text(self) -> None:
        chunks = [
            ResponseChunk(delta_text="A"),
            ResponseChunk(),  # tool-call only chunk
            ResponseChunk(delta_text="B"),
            ResponseChunk(finish_reason="stop", usage=Usage(input_tokens=1, output_tokens=2)),
        ]
        result = await accumulate_text(_async_iter(chunks))
        assert result == "AB"

    async def test_tool_only_stream_yields_empty_text(self) -> None:
        chunks = [
            ResponseChunk(
                delta_tool_calls=[
                    ToolCallDelta(index=0, id="c1", name="f", arguments_delta="{}"),
                ],
            ),
            ResponseChunk(finish_reason="tool_use"),
        ]
        assert await accumulate_text(_async_iter(chunks)) == ""


class TestAccumulateToolCalls:
    async def test_single_tool_call_assembled(self) -> None:
        chunks = [
            ResponseChunk(
                delta_tool_calls=[
                    ToolCallDelta(
                        index=0,
                        id="call_1",
                        name="get_weather",
                        arguments_delta='{"locat',
                    ),
                ],
            ),
            ResponseChunk(
                delta_tool_calls=[
                    ToolCallDelta(index=0, arguments_delta='ion": "Tokyo"}'),
                ],
            ),
            ResponseChunk(finish_reason="tool_use"),
        ]
        calls = await accumulate_tool_calls(_async_iter(chunks))
        assert len(calls) == 1
        assert calls[0].id == "call_1"
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"location": "Tokyo"}

    async def test_multiple_tool_calls_interleaved(self) -> None:
        chunks = [
            ResponseChunk(
                delta_tool_calls=[
                    ToolCallDelta(index=0, id="c1", name="get_weather", arguments_delta="{"),
                    ToolCallDelta(index=1, id="c2", name="get_time", arguments_delta="{"),
                ],
            ),
            ResponseChunk(
                delta_tool_calls=[
                    ToolCallDelta(index=0, arguments_delta='"location": "Paris"}'),
                    ToolCallDelta(index=1, arguments_delta='"tz": "UTC"}'),
                ],
            ),
        ]
        calls = await accumulate_tool_calls(_async_iter(chunks))
        assert len(calls) == 2
        # Returned in index order.
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"location": "Paris"}
        assert calls[1].name == "get_time"
        assert calls[1].arguments == {"tz": "UTC"}

    async def test_empty_arguments_text_becomes_empty_dict(self) -> None:
        chunks = [
            ResponseChunk(
                delta_tool_calls=[
                    ToolCallDelta(index=0, id="c1", name="ping", arguments_delta=""),
                ],
            ),
        ]
        calls = await accumulate_tool_calls(_async_iter(chunks))
        assert len(calls) == 1
        assert calls[0].arguments == {}

    async def test_missing_id_rejected(self) -> None:
        # id never arrives — incomplete tool call.
        chunks = [
            ResponseChunk(
                delta_tool_calls=[
                    ToolCallDelta(index=0, name="f", arguments_delta="{}"),
                ],
            ),
        ]
        with pytest.raises(ValidationError, match="missing its id"):
            await accumulate_tool_calls(_async_iter(chunks))

    async def test_missing_name_rejected(self) -> None:
        chunks = [
            ResponseChunk(
                delta_tool_calls=[
                    ToolCallDelta(index=0, id="c1", arguments_delta="{}"),
                ],
            ),
        ]
        with pytest.raises(ValidationError, match="missing its name"):
            await accumulate_tool_calls(_async_iter(chunks))

    async def test_invalid_json_rejected(self) -> None:
        chunks = [
            ResponseChunk(
                delta_tool_calls=[
                    ToolCallDelta(
                        index=0,
                        id="c1",
                        name="f",
                        arguments_delta='{"unclosed":',
                    ),
                ],
            ),
        ]
        with pytest.raises(ValidationError, match="invalid JSON arguments"):
            await accumulate_tool_calls(_async_iter(chunks))

    async def test_non_object_arguments_rejected(self) -> None:
        # JSON valid but not an object — tools always take object args.
        chunks = [
            ResponseChunk(
                delta_tool_calls=[
                    ToolCallDelta(
                        index=0,
                        id="c1",
                        name="f",
                        arguments_delta='["not", "object"]',
                    ),
                ],
            ),
        ]
        with pytest.raises(ValidationError, match="expected an object"):
            await accumulate_tool_calls(_async_iter(chunks))

    async def test_empty_stream_returns_empty_list(self) -> None:
        result = await accumulate_tool_calls(_async_iter([]))
        assert result == []


class TestStreamingToolCallAccumulator:
    """The incremental accumulator backing ``LLMClient.stream_tool_loop``."""

    def test_empty_accumulator_has_no_calls(self) -> None:
        acc = StreamingToolCallAccumulator()
        assert acc.has_calls is False
        assert acc.finalize() == []

    def test_incremental_add_across_chunks(self) -> None:
        acc = StreamingToolCallAccumulator()
        acc.add([ToolCallDelta(index=0, id="c1", name="get_weather", arguments_delta='{"loc')])
        assert acc.has_calls is True
        acc.add([ToolCallDelta(index=0, arguments_delta='ation": "Tokyo"}')])
        calls = acc.finalize()
        assert len(calls) == 1
        assert calls[0].id == "c1"
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"location": "Tokyo"}

    def test_multiple_indices_finalized_in_ascending_order(self) -> None:
        acc = StreamingToolCallAccumulator()
        # Deltas arrive out of index order; finalize sorts them.
        acc.add(
            [
                ToolCallDelta(index=1, id="c2", name="second", arguments_delta="{}"),
                ToolCallDelta(index=0, id="c1", name="first", arguments_delta="{}"),
            ]
        )
        calls = acc.finalize()
        assert [c.name for c in calls] == ["first", "second"]

    def test_empty_deltas_are_a_noop(self) -> None:
        acc = StreamingToolCallAccumulator()
        acc.add([])
        assert acc.has_calls is False

    def test_finalize_rejects_missing_id(self) -> None:
        acc = StreamingToolCallAccumulator()
        acc.add([ToolCallDelta(index=0, name="f", arguments_delta="{}")])
        with pytest.raises(ValidationError, match="missing its id"):
            acc.finalize()

    def test_finalize_rejects_invalid_json(self) -> None:
        acc = StreamingToolCallAccumulator()
        acc.add([ToolCallDelta(index=0, id="c1", name="f", arguments_delta='{"x":')])
        with pytest.raises(ValidationError, match="invalid JSON arguments"):
            acc.finalize()


class TestJSONAccumulator:
    def test_empty_buffer_is_not_complete(self) -> None:
        acc = JSONAccumulator()
        assert acc.is_complete() is False
        assert acc.text == ""

    def test_feed_appends(self) -> None:
        acc = JSONAccumulator()
        acc.feed("{")
        acc.feed('"a": 1')
        acc.feed("}")
        assert acc.text == '{"a": 1}'

    def test_partial_buffer_is_not_complete(self) -> None:
        acc = JSONAccumulator()
        acc.feed('{"a":')
        assert acc.is_complete() is False

    def test_complete_object_parses(self) -> None:
        acc = JSONAccumulator()
        acc.feed('{"a": 1, "b": [2, 3]}')
        assert acc.is_complete() is True
        assert acc.parse() == {"a": 1, "b": [2, 3]}

    def test_empty_string_feed_is_noop(self) -> None:
        acc = JSONAccumulator()
        acc.feed("")
        acc.feed("")
        assert acc.text == ""
        assert not acc.is_complete()

    def test_reset_clears_buffer(self) -> None:
        acc = JSONAccumulator()
        acc.feed('{"a": 1}')
        acc.reset()
        assert acc.text == ""
        assert acc.is_complete() is False

    def test_parse_empty_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="buffer is empty"):
            JSONAccumulator().parse()

    def test_parse_invalid_raises_validation_error(self) -> None:
        acc = JSONAccumulator()
        acc.feed('{"a":')  # incomplete
        with pytest.raises(ValidationError, match="not valid JSON"):
            acc.parse()

    def test_parses_scalar_top_level_value(self) -> None:
        # JSON values needn't be objects.
        acc = JSONAccumulator()
        acc.feed('"just a string"')
        assert acc.is_complete()
        assert acc.parse() == "just a string"

    def test_parses_array_top_level_value(self) -> None:
        acc = JSONAccumulator()
        acc.feed("[1, 2, 3]")
        assert acc.is_complete()
        assert acc.parse() == [1, 2, 3]
