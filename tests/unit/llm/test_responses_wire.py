"""Unit tests for `strata_forge.llm.responses_wire` (no network)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from strata_forge.core.errors import (
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderServerError,
    ValidationError,
)
from strata_forge.llm.messages import (
    AssistantMessage,
    CallRef,
    Message,
    ProviderItems,
    ReasoningItem,
    TextItem,
    TextPart,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from strata_forge.llm.multimodal import ImageContent
from strata_forge.llm.responses_wire import (
    INCLUDE_ENCRYPTED_REASONING,
    ResponsesStreamParser,
    build_input,
    build_request,
    parse_response,
    parse_usage,
    redact_encrypted_content,
    requested_effort,
)
from strata_forge.llm.schemas import to_openai_response_format
from strata_forge.llm.tools import ToolDeclaration

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

    from strata_forge.llm.responses import ResponseChunk


_ECHO = ToolDeclaration(
    name="echo_value",
    description="Echo a value back.",
    parameters={
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
)


def _tool_turn(provider: str = "openai") -> AssistantMessage:
    return AssistantMessage(
        content="Let me check.",
        tool_calls=[ToolCall(id="call_1", name="echo_value", arguments={"value": "ping"})],
        provider_items=ProviderItems(
            provider=provider,  # type: ignore[arg-type]
            items=(
                ReasoningItem(id="rs_1", encrypted_content="blob-1", summary=("thought",)),
                TextItem(id="msg_1", phase="commentary", text="Let me check."),
                CallRef(id="fc_1", call_id="call_1"),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


class TestBuildInput:
    def test_leading_system_messages_become_instructions(self) -> None:
        instructions, items = build_input(
            [Message.system("one"), Message.system("two"), Message.user("hi")], "openai"
        )
        assert instructions == "one\n\ntwo"
        assert items == [{"type": "message", "role": "user", "content": "hi"}]

    def test_later_system_message_becomes_a_developer_item_in_place(self) -> None:
        instructions, items = build_input(
            [
                Message.system("base"),
                Message.user("hi"),
                Message.assistant("not json"),
                Message.system("reprompt: return JSON"),
            ],
            "openai",
        )
        assert instructions == "base"
        assert items[-1] == {
            "type": "message",
            "role": "developer",
            "content": "reprompt: return JSON",
        }

    def test_no_system_message_means_no_instructions(self) -> None:
        instructions, _ = build_input([Message.user("hi")], "openai")
        assert instructions is None

    def test_user_parts_become_input_text_and_input_image(self) -> None:
        _, items = build_input(
            [
                UserMessage(
                    content=[
                        TextPart(text="what is this?"),
                        ImageContent.from_url("https://example.com/a.png"),
                        ImageContent.from_bytes(b"\x89PNG", mime_type="image/png"),
                    ]
                )
            ],
            "openai",
        )
        content = items[0]["content"]
        assert content[0] == {"type": "input_text", "text": "what is this?"}
        assert content[1] == {
            "type": "input_image",
            "image_url": "https://example.com/a.png",
            "detail": "auto",
        }
        assert content[2]["image_url"].startswith("data:image/png;base64,")

    def test_matching_provider_items_are_replayed_in_order(self) -> None:
        _, items = build_input([Message.user("q"), _tool_turn()], "openai")
        assert items[1:] == [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "thought"}],
                "encrypted_content": "blob-1",
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Let me check.",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
                "phase": "commentary",
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "echo_value",
                "arguments": json.dumps({"value": "ping"}),
                "id": "fc_1",
            },
        ]
        # Status is a required field of the output-message input shape only.
        assert [item.get("status") for item in items[1:]] == [None, "completed", None]

    def test_items_from_another_provider_are_synthesized_without_ids(self) -> None:
        _, items = build_input([Message.user("q"), _tool_turn(provider="azure")], "openai")
        assert items[1:] == [
            {"type": "message", "role": "assistant", "content": "Let me check."},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "echo_value",
                "arguments": json.dumps({"value": "ping"}),
            },
        ]

    def test_turn_without_items_is_synthesized(self) -> None:
        turn = AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(id="a", name="echo_value", arguments={"value": "1"}),
                ToolCall(id="b", name="echo_value", arguments={"value": "2"}),
            ],
        )
        _, items = build_input([Message.user("q"), turn], "openai")
        assert [item["call_id"] for item in items[1:]] == ["a", "b"]
        assert all("id" not in item for item in items[1:])

    def test_items_without_text_keep_the_plain_content_before_the_calls(self) -> None:
        turn = AssistantMessage(
            content="preamble",
            tool_calls=[
                ToolCall(id="a", name="echo_value", arguments={}),
                ToolCall(id="b", name="echo_value", arguments={}),
            ],
            provider_items=ProviderItems(
                provider="openai",
                items=(
                    ReasoningItem(id="rs_1", encrypted_content="blob"),
                    CallRef(id="fc_a", call_id="a"),
                ),
            ),
        )
        _, items = build_input([turn], "openai")
        assert [item["type"] for item in items] == [
            "reasoning",
            "message",
            "function_call",
            "function_call",
        ]
        assert items[1] == {"type": "message", "role": "assistant", "content": "preamble"}
        assert items[2]["id"] == "fc_a"
        # A call the items never referenced is still sent, synthesized.
        assert items[3]["call_id"] == "b"
        assert "id" not in items[3]

    def test_tool_results_become_function_call_outputs(self) -> None:
        _, items = build_input(
            [
                Message.user("q"),
                _tool_turn(),
                ToolResultMessage(tool_call_id="call_1", content="pong", is_error=True),
            ],
            "openai",
        )
        assert items[-1] == {"type": "function_call_output", "call_id": "call_1", "output": "pong"}


class _Answer(BaseModel):
    value: str


class TestBuildRequest:
    def test_always_stateless_with_encrypted_reasoning(self) -> None:
        body = build_request(messages=[Message.user("hi")], provider="openai")
        assert body["store"] is False
        assert body["include"] == [INCLUDE_ENCRYPTED_REASONING]
        for absent in ("previous_response_id", "stream", "tools", "temperature", "text"):
            assert absent not in body

    def test_parameters_map_onto_responses_names(self) -> None:
        body = build_request(
            messages=[Message.user("hi")],
            provider="openai",
            tools=[_ECHO],
            temperature=0.1,
            top_p=0.9,
            max_tokens=25_000,
            response_format=to_openai_response_format(_Answer),
            stream=True,
        )
        assert body["max_output_tokens"] == 25_000
        assert "max_tokens" not in body
        assert body["temperature"] == 0.1
        assert body["top_p"] == 0.9
        assert body["stream"] is True
        assert body["tools"] == [
            {
                "type": "function",
                "name": "echo_value",
                "description": "Echo a value back.",
                "parameters": _ECHO.parameters,
                "strict": False,
            }
        ]
        text_format = body["text"]["format"]
        assert text_format["type"] == "json_schema"
        assert text_format["name"] == "_Answer"
        assert text_format["strict"] is True
        assert "response_format" not in body

    def test_empty_tool_list_sends_no_tools(self) -> None:
        body = build_request(messages=[Message.user("hi")], provider="openai", tools=[])
        assert "tools" not in body

    def test_extras_translate_reasoning_effort_and_response_format(self) -> None:
        body = build_request(
            messages=[Message.user("hi")],
            provider="openai",
            extras={
                "reasoning_effort": "low",
                "reasoning": {"summary": "auto"},
                "response_format": {"type": "json_object"},
                "service_tier": "flex",
            },
        )
        assert body["reasoning"] == {"summary": "auto", "effort": "low"}
        assert body["text"] == {"format": {"type": "json_object"}}
        assert body["service_tier"] == "flex"
        assert "reasoning_effort" not in body
        assert "response_format" not in body

    @pytest.mark.parametrize(
        "field", ["store", "previous_response_id", "conversation", "background"]
    )
    def test_extras_cannot_make_the_request_stateful(self, field: str) -> None:
        with pytest.raises(ValidationError, match=field):
            build_request(messages=[Message.user("hi")], provider="openai", extras={field: True})

    def test_extras_include_adds_to_the_encrypted_reasoning(self) -> None:
        body = build_request(
            messages=[Message.user("hi")],
            provider="openai",
            extras={"include": ["message.output_text.logprobs", INCLUDE_ENCRYPTED_REASONING]},
        )
        assert body["include"] == [INCLUDE_ENCRYPTED_REASONING, "message.output_text.logprobs"]
        assert body["store"] is False

    @pytest.mark.parametrize("include", ["reasoning.encrypted_content", [1], {"a": 1}])
    def test_extras_include_must_be_a_list_of_strings(self, include: Any) -> None:
        with pytest.raises(ValidationError, match="include"):
            build_request(
                messages=[Message.user("hi")], provider="openai", extras={"include": include}
            )

    def test_extras_text_merges_into_the_structured_output_format(self) -> None:
        response_format = to_openai_response_format(_Answer)
        body = build_request(
            messages=[Message.user("hi")],
            provider="openai",
            response_format=response_format,
            extras={"text": {"verbosity": "low"}},
        )
        assert body["text"]["verbosity"] == "low"
        assert body["text"]["format"]["name"] == "_Answer"

    def test_extras_chat_completions_verbosity_becomes_text_verbosity(self) -> None:
        body = build_request(
            messages=[Message.user("hi")], provider="openai", extras={"verbosity": "high"}
        )
        assert body["text"] == {"verbosity": "high"}
        assert "verbosity" not in body

    def test_extras_text_format_replaces_the_computed_one(self) -> None:
        body = build_request(
            messages=[Message.user("hi")],
            provider="openai",
            response_format=to_openai_response_format(_Answer),
            extras={"text": {"format": {"type": "text"}}},
        )
        assert body["text"] == {"format": {"type": "text"}}

    def test_other_extras_are_set_on_the_body(self) -> None:
        body = build_request(
            messages=[Message.user("hi")],
            provider="openai",
            extras={"prompt_cache_key": "k", "extra_body": {"x": 1}},
        )
        assert body["prompt_cache_key"] == "k"
        assert body["extra_body"] == {"x": 1}

    def test_full_body_snapshot(self, snapshot: SnapshotAssertion) -> None:
        body = build_request(
            messages=[
                Message.system("You are a workspace assistant."),
                Message.user("load gpt2"),
                _tool_turn(),
                ToolResultMessage(tool_call_id="call_1", content="loaded"),
            ],
            provider="openai",
            tools=[_ECHO],
            max_tokens=25_000,
            stream=True,
        )
        assert body == snapshot


class TestRequestedEffort:
    @pytest.mark.parametrize(
        ("extras", "effort"),
        [
            (None, None),
            ({}, None),
            ({"reasoning_effort": "none"}, "none"),
            ({"reasoning": {"effort": "high"}}, "high"),
            ({"reasoning": {"effort": "low"}, "reasoning_effort": "high"}, "low"),
            ({"reasoning": "bogus"}, None),
        ],
    )
    def test_reads_either_spelling(self, extras: dict[str, Any] | None, effort: str | None) -> None:
        assert requested_effort(extras) == effort


def test_redact_encrypted_content_replaces_the_blob_with_its_size() -> None:
    items = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "x" * 42}, {"type": "x"}]
    redacted = redact_encrypted_content(items)
    assert redacted[0]["encrypted_content"] == "<42 bytes>"
    assert redacted[1] == {"type": "x"}
    assert items[0]["encrypted_content"] == "x" * 42


# ---------------------------------------------------------------------------
# Usage + non-streamed parsing
# ---------------------------------------------------------------------------


class TestParseUsage:
    def test_input_is_net_of_cached_reads_and_writes(self) -> None:
        usage = parse_usage(
            {
                "input_tokens": 1_000,
                "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 100},
                "output_tokens": 50,
                "output_tokens_details": {"reasoning_tokens": 40},
            }
        )
        assert usage.input_tokens == 300
        assert usage.cache_read_tokens == 600
        assert usage.cache_write_tokens == 100
        assert usage.output_tokens == 50

    def test_missing_usage_is_zero(self) -> None:
        usage = parse_usage(None)
        assert (usage.input_tokens, usage.output_tokens) == (0, 0)

    def test_missing_details_are_zero(self) -> None:
        usage = parse_usage({"input_tokens": 10, "output_tokens": 2})
        assert usage.input_tokens == 10
        assert usage.cache_read_tokens == 0


class TestParseResponse:
    def test_tool_turn(self) -> None:
        parsed = parse_response(
            {
                "status": "completed",
                "output": [
                    {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "e"},
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo_value",
                        "arguments": "{not json",
                    },
                ],
            },
            provider="openai",
        )
        assert parsed.finish_reason == "tool_use"
        # A malformed argument string degrades to {} on the non-streamed path, as on CC.
        assert parsed.tool_calls == [ToolCall(id="call_1", name="echo_value", arguments={})]
        assert parsed.provider_items == ProviderItems(
            provider="openai",
            items=(
                ReasoningItem(id="rs_1", encrypted_content="e"),
                CallRef(id="fc_1", call_id="call_1"),
            ),
        )

    def test_reasoning_without_encrypted_content_is_not_replayable(self) -> None:
        parsed = parse_response(
            {
                "status": "completed",
                "output": [
                    {"type": "reasoning", "id": "rs_1", "summary": []},
                    {
                        "type": "message",
                        "id": "m",
                        "content": [{"type": "output_text", "text": "x"}],
                    },
                ],
            },
            provider="azure",
        )
        assert parsed.provider_items is not None
        assert parsed.provider_items.provider == "azure"
        assert parsed.provider_items.items == (TextItem(id="m", text="x"),)

    @pytest.mark.parametrize(
        ("reason", "finish"),
        [("max_output_tokens", "length"), ("content_filter", "content_filter"), (None, "length")],
    )
    def test_incomplete(self, reason: str | None, finish: str) -> None:
        parsed = parse_response(
            {"status": "incomplete", "incomplete_details": {"reason": reason}, "output": []},
            provider="openai",
        )
        assert parsed.finish_reason == finish

    def test_refusal_is_visible_text_with_content_filter(self) -> None:
        parsed = parse_response(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "m",
                        "content": [{"type": "refusal", "refusal": "no"}],
                    }
                ],
            },
            provider="openai",
        )
        assert parsed.text == "no"
        assert parsed.finish_reason == "content_filter"

    def test_failed_raises(self) -> None:
        with pytest.raises(ProviderServerError, match="boom"):
            parse_response(
                {"status": "failed", "error": {"code": "server_error", "message": "boom"}},
                provider="openai",
            )


# ---------------------------------------------------------------------------
# Stream parsing
# ---------------------------------------------------------------------------


def _feed(parser: ResponsesStreamParser, events: list[dict[str, Any]]) -> list[ResponseChunk]:
    return [chunk for event in events if (chunk := parser.feed(event)) is not None]


_COMPLETED_USAGE = {"input_tokens": 10, "output_tokens": 5}


class TestResponsesStreamParser:
    def test_text_deltas_then_final_chunk(self) -> None:
        parser = ResponsesStreamParser("openai")
        message = {
            "type": "message",
            "id": "msg_1",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "Hello"}],
        }
        chunks = _feed(
            parser,
            [
                {"type": "response.created", "response": {}},
                {"type": "response.output_text.delta", "output_index": 0, "delta": "Hel"},
                {"type": "response.output_text.delta", "output_index": 0, "delta": "lo"},
                {"type": "response.output_text.done", "output_index": 0, "text": "Hello"},
                {"type": "response.output_item.done", "output_index": 0, "item": message},
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "output": [message],
                        "usage": _COMPLETED_USAGE,
                    },
                },
            ],
        )
        assert [c.delta_text for c in chunks] == ["Hel", "lo", ""]
        final = chunks[-1]
        assert final.finish_reason == "stop"
        assert final.usage is not None
        assert final.usage.input_tokens == 10
        assert final.provider_items == ProviderItems(
            provider="openai", items=(TextItem(id="msg_1", phase="final_answer", text="Hello"),)
        )
        assert parser.finished

    def test_parallel_calls_after_reasoning_get_dense_slots(self) -> None:
        parser = ResponsesStreamParser("openai")
        reasoning: dict[str, Any] = {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "e",
        }
        call_a = {
            "type": "function_call",
            "id": "fc_a",
            "call_id": "call_a",
            "name": "echo_value",
            "arguments": '{"value":"a"}',
        }
        call_b = {**call_a, "id": "fc_b", "call_id": "call_b", "arguments": '{"value":"b"}'}
        chunks = _feed(
            parser,
            [
                {"type": "response.output_item.added", "output_index": 0, "item": reasoning},
                {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
                {
                    "type": "response.output_item.added",
                    "output_index": 1,
                    "item": {**call_a, "arguments": ""},
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 2,
                    "item": {**call_b, "arguments": ""},
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 2,
                    "delta": '{"value":"b"}',
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 1,
                    "delta": '{"value":"a"}',
                },
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": 1,
                    "arguments": '{"value":"a"}',
                },
                {"type": "response.output_item.done", "output_index": 1, "item": call_a},
                {"type": "response.output_item.done", "output_index": 2, "item": call_b},
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "output": [reasoning, call_a, call_b],
                        "usage": _COMPLETED_USAGE,
                    },
                },
            ],
        )
        deltas = [d for c in chunks for d in c.delta_tool_calls]
        opened = [
            (d.index, d.id, d.name) for d in deltas if d.id is not None and d.arguments_delta == ""
        ]
        assert opened[:2] == [(0, "call_a", "echo_value"), (1, "call_b", "echo_value")]
        args: dict[int, str] = {}
        for d in deltas:
            args[d.index] = args.get(d.index, "") + d.arguments_delta
        # Arguments arrive once per slot; the done item never repeats them.
        assert args == {0: '{"value":"a"}', 1: '{"value":"b"}'}
        final = chunks[-1]
        assert final.finish_reason == "tool_use"
        assert final.provider_items is not None
        assert [type(i) for i in final.provider_items.items] == [ReasoningItem, CallRef, CallRef]

    def test_fake_stream_delivers_calls_only_on_the_terminal_event(self) -> None:
        # LiteLLM fakes a stream for a model its map does not know: text deltas plus one
        # response.completed carrying the full output, with no item events.
        parser = ResponsesStreamParser("openai")
        call = {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "echo_value",
            "arguments": '{"value":"x"}',
        }
        chunks = _feed(
            parser,
            [
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "output": [
                            {"type": "reasoning", "id": "rs_1", "encrypted_content": "e"},
                            call,
                        ],
                        "usage": _COMPLETED_USAGE,
                    },
                }
            ],
        )
        assert len(chunks) == 1
        final = chunks[0]
        assert final.finish_reason == "tool_use"
        assert [(d.index, d.id, d.name, d.arguments_delta) for d in final.delta_tool_calls] == [
            (0, "call_1", "echo_value", '{"value":"x"}')
        ]

    def test_fake_stream_with_no_text_deltas_emits_the_text_once(self) -> None:
        parser = ResponsesStreamParser("openai")
        message = {"type": "message", "id": "m", "content": [{"type": "output_text", "text": "hi"}]}
        chunks = _feed(
            parser,
            [
                {
                    "type": "response.completed",
                    "response": {"status": "completed", "output": [message]},
                }
            ],
        )
        assert [c.delta_text for c in chunks] == ["hi"]

    def test_fake_stream_text_deltas_are_not_repeated_by_the_terminal_event(self) -> None:
        # LiteLLM's fake stream tags every text delta output_index 0, even when the message
        # sits after a reasoning item.
        parser = ResponsesStreamParser("openai")
        message = {"type": "message", "id": "m", "content": [{"type": "output_text", "text": "hi"}]}
        chunks = _feed(
            parser,
            [
                {"type": "response.output_text.delta", "output_index": 0, "delta": "hi"},
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "output": [
                            {"type": "reasoning", "id": "rs", "encrypted_content": "e"},
                            message,
                        ],
                    },
                },
            ],
        )
        assert "".join(c.delta_text for c in chunks) == "hi"

    def test_message_done_without_deltas_delivers_its_text_once(self) -> None:
        parser = ResponsesStreamParser("openai")
        message = {"type": "message", "id": "m", "content": [{"type": "output_text", "text": "hi"}]}
        chunks = _feed(
            parser,
            [
                {"type": "response.output_item.done", "output_index": 0, "item": message},
                {
                    "type": "response.completed",
                    "response": {"status": "completed", "output": [message]},
                },
            ],
        )
        assert "".join(c.delta_text for c in chunks) == "hi"

    def test_fake_stream_reporting_incomplete_status_is_length(self) -> None:
        parser = ResponsesStreamParser("openai")
        chunks = _feed(
            parser,
            [
                {
                    "type": "response.completed",
                    "response": {
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                        "output": [],
                    },
                }
            ],
        )
        assert chunks[-1].finish_reason == "length"

    def test_arguments_only_in_the_done_item(self) -> None:
        parser = ResponsesStreamParser("openai")
        call = {"type": "function_call", "id": "fc", "call_id": "c", "name": "n", "arguments": "{}"}
        chunks = _feed(
            parser,
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {**call, "arguments": ""},
                },
                {"type": "response.output_item.done", "output_index": 0, "item": call},
            ],
        )
        assert "".join(d.arguments_delta for c in chunks for d in c.delta_tool_calls) == "{}"

    def test_incomplete_content_filter(self) -> None:
        parser = ResponsesStreamParser("openai")
        (chunk,) = _feed(
            parser,
            [
                {
                    "type": "response.incomplete",
                    "response": {
                        "status": "incomplete",
                        "incomplete_details": {"reason": "content_filter"},
                        "output": [],
                    },
                }
            ],
        )
        assert chunk.finish_reason == "content_filter"

    def test_refusal_deltas_are_visible_and_finish_content_filter(self) -> None:
        parser = ResponsesStreamParser("openai")
        chunks = _feed(
            parser,
            [
                {"type": "response.refusal.delta", "output_index": 0, "delta": "I can't"},
                {"type": "response.completed", "response": {"status": "completed", "output": []}},
            ],
        )
        assert chunks[0].delta_text == "I can't"
        assert chunks[-1].finish_reason == "content_filter"

    @pytest.mark.parametrize(
        ("code", "error_type"),
        [
            ("rate_limit_exceeded", ProviderRateLimitError),
            ("server_error", ProviderServerError),
            (None, ProviderServerError),
            ("invalid_prompt", ProviderBadRequestError),
        ],
    )
    def test_failed_event_raises_mapped(
        self, code: str | None, error_type: type[Exception]
    ) -> None:
        parser = ResponsesStreamParser("openai", model="gpt-6.1-sol")
        with pytest.raises(error_type) as info:
            parser.feed(
                {
                    "type": "response.failed",
                    "response": {"status": "failed", "error": {"code": code, "message": "nope"}},
                }
            )
        assert "nope" in str(info.value)
        assert "gpt-6.1-sol" in str(info.value)

    def test_error_event_raises_mapped(self) -> None:
        parser = ResponsesStreamParser("openai")
        with pytest.raises(ProviderRateLimitError, match="slow"):
            parser.feed({"type": "error", "code": "rate_limit_exceeded", "message": "slow"})

    @pytest.mark.parametrize(
        ("code", "error_type"),
        [
            ("rate_limit_exceeded", ProviderRateLimitError),
            ("context_length_exceeded", ProviderBadRequestError),
            ("server_error", ProviderServerError),
        ],
    )
    def test_nested_error_event_keeps_its_code_and_message(
        self, code: str, error_type: type[Exception]
    ) -> None:
        parser = ResponsesStreamParser("openai", model="gpt-6.1-sol")
        event = {
            "type": "error",
            "sequence_number": 3,
            "error": {"type": "invalid_request_error", "code": code, "message": "too much"},
        }
        with pytest.raises(error_type, match=f"{code}: too much"):
            parser.feed(event)

    def test_nested_error_event_from_litellm_types(self) -> None:
        from litellm.types.llms.openai import (  # pyright: ignore[reportMissingTypeStubs]
            ErrorEvent,
        )

        event = ErrorEvent.model_validate(
            {
                "type": "error",
                "sequence_number": 1,
                "error": {
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                    "message": "slow down",
                },
            }
        )
        parser = ResponsesStreamParser("openai")
        with pytest.raises(ProviderRateLimitError, match="rate_limit_exceeded: slow down"):
            parser.feed(event)

    def test_unknown_and_hosted_tool_events_are_ignored(self) -> None:
        parser = ResponsesStreamParser("openai")
        events: list[dict[str, Any]] = [
            {"type": "response.in_progress", "response": {}},
            {"type": "response.reasoning_summary_text.delta", "delta": "x"},
            {"type": "response.web_search_call.searching"},
            {"type": "response.output_text.delta", "delta": ""},
            {"type": "something.new"},
        ]
        for event in events:
            assert parser.feed(event) is None
        assert not parser.finished

    def test_events_as_attribute_objects(self) -> None:
        # LiteLLM yields Pydantic event models rather than dicts.
        class _Event(BaseModel):
            type: str
            output_index: int = 0
            delta: str = ""

        parser = ResponsesStreamParser("openai")
        chunk = parser.feed(_Event(type="response.output_text.delta", delta="hey"))
        assert chunk is not None
        assert chunk.delta_text == "hey"
