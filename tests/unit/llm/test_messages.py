"""Unit tests for `strata_forge.llm.messages`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from strata_forge.core.errors import ValidationError
from strata_forge.llm.messages import (
    AssistantMessage,
    CallRef,
    Message,
    ProviderItems,
    ReasoningItem,
    SystemMessage,
    TextItem,
    TextPart,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    validate_conversation,
)


class TestSystemMessage:
    def test_construction(self) -> None:
        m = SystemMessage(content="You are a helpful assistant.")
        assert m.role == "system"
        assert m.content == "You are a helpful assistant."

    def test_frozen(self) -> None:
        m = SystemMessage(content="hi")
        with pytest.raises(PydanticValidationError):
            m.content = "changed"  # type: ignore[misc]


class TestUserMessage:
    def test_plain_text(self) -> None:
        m = UserMessage(content="What's the weather?")
        assert m.role == "user"
        assert m.content == "What's the weather?"

    def test_multipart_text(self) -> None:
        m = UserMessage(content=[TextPart(text="Hello"), TextPart(text="World")])
        assert m.role == "user"
        assert isinstance(m.content, list)
        assert len(m.content) == 2
        assert all(isinstance(p, TextPart) for p in m.content)


class TestAssistantMessage:
    def test_text_only(self) -> None:
        m = AssistantMessage(content="The weather is sunny.")
        assert m.role == "assistant"
        assert m.content == "The weather is sunny."
        assert m.tool_calls == []

    def test_with_tool_calls(self) -> None:
        call = ToolCall(id="call_1", name="get_weather", arguments={"location": "Tokyo"})
        m = AssistantMessage(content=None, tool_calls=[call])
        assert m.content is None
        assert len(m.tool_calls) == 1
        assert m.tool_calls[0].name == "get_weather"

    def test_content_can_be_none(self) -> None:
        m = AssistantMessage()
        assert m.content is None
        assert m.tool_calls == []


class TestProviderItems:
    def _items(self, call_id: str = "c1") -> ProviderItems:
        return ProviderItems(
            provider="openai",
            items=(
                ReasoningItem(id="rs_1", encrypted_content="blob", summary=("s",)),
                TextItem(id="msg_1", phase="commentary", text="checking"),
                CallRef(id="fc_1", call_id=call_id),
            ),
        )

    def test_round_trips_through_json_in_order(self) -> None:
        msg = AssistantMessage(
            content="checking",
            tool_calls=[ToolCall(id="c1", name="f")],
            provider_items=self._items(),
        )
        restored = AssistantMessage.model_validate_json(msg.model_dump_json())
        assert restored == msg
        assert restored.provider_items is not None
        assert [i.kind for i in restored.provider_items.items] == ["reasoning", "text", "call"]

    def test_defaults_to_none(self) -> None:
        assert AssistantMessage(content="x").provider_items is None

    def test_call_ref_must_name_one_of_the_turns_calls(self) -> None:
        with pytest.raises(PydanticValidationError, match="not a tool call"):
            AssistantMessage(
                tool_calls=[ToolCall(id="c1", name="f")], provider_items=self._items("c2")
            )

    @pytest.mark.parametrize(
        "item",
        [
            {"kind": "message", "id": "x", "role": "user", "content": "hi"},
            {"kind": "text", "id": "x", "text": "t", "role": "system"},
            {"type": "reasoning", "id": "x", "encrypted_content": "e"},
        ],
    )
    def test_items_cannot_express_another_role(self, item: dict[str, str]) -> None:
        # The union is closed and extra="forbid": no item can smuggle a user, system or
        # developer message into the replayed transcript.
        with pytest.raises(PydanticValidationError):
            ProviderItems.model_validate({"provider": "openai", "items": [item]})

    def test_provider_is_a_responses_provider(self) -> None:
        with pytest.raises(PydanticValidationError):
            ProviderItems.model_validate({"provider": "anthropic", "items": []})

    def test_unknown_phase_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            TextItem(id="m", text="t", phase="draft")  # type: ignore[arg-type]


class TestToolResultMessage:
    def test_construction(self) -> None:
        m = ToolResultMessage(tool_call_id="call_1", content="22°C, sunny")
        assert m.role == "tool"
        assert m.tool_call_id == "call_1"
        assert m.content == "22°C, sunny"
        assert m.is_error is False

    def test_error_result(self) -> None:
        m = ToolResultMessage(tool_call_id="call_1", content="API down", is_error=True)
        assert m.is_error is True


class TestToolCall:
    def test_construction(self) -> None:
        c = ToolCall(id="x", name="f", arguments={"k": "v"})
        assert c.id == "x"
        assert c.name == "f"
        assert c.arguments == {"k": "v"}

    def test_arguments_default_empty(self) -> None:
        c = ToolCall(id="x", name="f")
        assert c.arguments == {}

    def test_frozen(self) -> None:
        c = ToolCall(id="x", name="f")
        with pytest.raises(PydanticValidationError):
            c.name = "g"  # type: ignore[misc]


class TestMessageFactory:
    def test_namespace_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="namespace"):
            Message()  # type: ignore[call-arg]

    def test_system(self) -> None:
        m = Message.system("You are kind.")
        assert isinstance(m, SystemMessage)
        assert m.content == "You are kind."

    def test_user_string(self) -> None:
        m = Message.user("hello")
        assert isinstance(m, UserMessage)
        assert m.content == "hello"

    def test_user_multipart(self) -> None:
        m = Message.user([TextPart(text="hi")])
        assert isinstance(m, UserMessage)
        assert isinstance(m.content, list)

    def test_assistant_text_only(self) -> None:
        m = Message.assistant("done")
        assert isinstance(m, AssistantMessage)
        assert m.content == "done"
        assert m.tool_calls == []

    def test_assistant_with_tool_calls(self) -> None:
        call = ToolCall(id="x", name="f", arguments={})
        m = Message.assistant(tool_calls=[call])
        assert m.content is None
        assert m.tool_calls == [call]

    def test_tool_result(self) -> None:
        m = Message.tool_result("call_1", "the answer is 42")
        assert isinstance(m, ToolResultMessage)
        assert m.tool_call_id == "call_1"
        assert m.is_error is False

    def test_tool_result_with_error_flag(self) -> None:
        m = Message.tool_result("call_1", "oops", is_error=True)
        assert m.is_error is True


class TestValidateConversation:
    def test_empty_conversation_ok(self) -> None:
        validate_conversation([])

    def test_no_tool_use_ok(self) -> None:
        validate_conversation(
            [
                Message.system("be kind"),
                Message.user("hi"),
                Message.assistant("hello"),
            ],
        )

    def test_well_formed_tool_use_ok(self) -> None:
        validate_conversation(
            [
                Message.user("weather in Tokyo?"),
                Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="get_weather", arguments={})],
                ),
                Message.tool_result("c1", "sunny"),
                Message.assistant("It's sunny in Tokyo."),
            ],
        )

    def test_two_tool_calls_two_results_ok(self) -> None:
        validate_conversation(
            [
                Message.user("weather + time?"),
                Message.assistant(
                    tool_calls=[
                        ToolCall(id="c1", name="get_weather", arguments={}),
                        ToolCall(id="c2", name="get_time", arguments={}),
                    ],
                ),
                Message.tool_result("c1", "sunny"),
                Message.tool_result("c2", "noon"),
            ],
        )

    def test_unmatched_result_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown tool_call_id"):
            validate_conversation(
                [
                    Message.user("hi"),
                    Message.tool_result("never-emitted", "result"),
                ],
            )

    def test_result_before_call_rejected(self) -> None:
        # A tool result must follow its tool call, never precede it.
        with pytest.raises(ValidationError, match="unknown tool_call_id"):
            validate_conversation(
                [
                    Message.tool_result("c1", "ahead-of-time"),
                    Message.assistant(
                        tool_calls=[ToolCall(id="c1", name="x", arguments={})],
                    ),
                ],
            )

    def test_duplicate_tool_call_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate ToolCall id"):
            validate_conversation(
                [
                    Message.assistant(
                        tool_calls=[
                            ToolCall(id="c1", name="x", arguments={}),
                        ],
                    ),
                    Message.tool_result("c1", "r"),
                    Message.assistant(
                        tool_calls=[
                            ToolCall(id="c1", name="x", arguments={}),  # duplicate id
                        ],
                    ),
                ],
            )

    def test_error_message_names_index_and_id(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            validate_conversation(
                [
                    Message.user("first"),
                    Message.tool_result("orphan", "result"),  # index 1
                ],
            )
        msg = str(excinfo.value)
        assert "index 1" in msg
        assert "orphan" in msg
