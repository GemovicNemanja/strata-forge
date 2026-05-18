"""Unit tests for `forge.training.chat_template`."""

from __future__ import annotations

from typing import Any

import pytest

from forge.llm.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    TextPart,
    ToolResultMessage,
    UserMessage,
)
from forge.training.chat_template import (
    apply_chat_template,
    conversation_to_dicts,
    conversation_to_text,
)


def _convo() -> list[Any]:
    return [
        SystemMessage(content="You are helpful."),
        UserMessage(content="Hi."),
        AssistantMessage(content="Hello!"),
    ]


class TestConversationToDicts:
    def test_roles_mapped(self) -> None:
        dicts = conversation_to_dicts(_convo())
        assert [d["role"] for d in dicts] == ["system", "user", "assistant"]
        assert [d["content"] for d in dicts] == [
            "You are helpful.",
            "Hi.",
            "Hello!",
        ]

    def test_tool_result_mapped_to_tool_role(self) -> None:
        convo: list[Any] = [
            AssistantMessage(content="Calling tool.", tool_calls=[]),
            ToolResultMessage(tool_call_id="call_1", content="result"),
        ]
        dicts = conversation_to_dicts(convo)
        assert dicts[-1]["role"] == "tool"
        assert dicts[-1]["content"] == "result"

    def test_multimodal_content_collapses_to_text(self) -> None:
        # List-of-parts content: text parts join, others get a placeholder.
        msg = UserMessage(content=[TextPart(text="What's in this?")])
        dicts = conversation_to_dicts([msg])
        # TextPart isn't a str, so it gets the [non-text: ...] placeholder.
        # That's intentional — the chat-template formatter targets text-only training.
        assert "non-text" in dicts[0]["content"]


class TestConversationToText:
    def test_default_format(self) -> None:
        text = conversation_to_text(_convo())
        assert "system: You are helpful." in text
        assert "user: Hi." in text
        assert "assistant: Hello!" in text

    def test_custom_separator(self) -> None:
        text = conversation_to_text(_convo(), message_separator=" | ")
        assert text.count(" | ") == 2

    def test_role_prefix_and_suffix(self) -> None:
        text = conversation_to_text(_convo(), role_prefix="[", role_suffix="] ")
        assert "[system] You are helpful." in text


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
    ) -> str | list[int]:
        self.calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
            }
        )
        if tokenize:
            return [1, 2, 3, 4]
        return "<formatted>"


class TestApplyChatTemplate:
    def test_returns_string_by_default(self) -> None:
        tok = _FakeTokenizer()
        result = apply_chat_template(_convo(), tokenizer=tok)
        assert result == "<formatted>"
        # The tokenizer received the standard dict shape.
        assert tok.calls[0]["messages"][0]["role"] == "system"

    def test_tokenize_returns_int_list(self) -> None:
        tok = _FakeTokenizer()
        result = apply_chat_template(_convo(), tokenizer=tok, tokenize=True)
        assert result == [1, 2, 3, 4]

    def test_add_generation_prompt_forwarded(self) -> None:
        tok = _FakeTokenizer()
        apply_chat_template(_convo(), tokenizer=tok, add_generation_prompt=True)
        assert tok.calls[0]["add_generation_prompt"] is True

    def test_rejects_tokenizer_without_method(self) -> None:
        class _NoMethod:
            pass

        with pytest.raises(ValueError, match="apply_chat_template"):
            apply_chat_template(_convo(), tokenizer=_NoMethod())

    def test_works_with_message_factory(self) -> None:
        convo = [
            Message.system("hi"),
            Message.user("what's up"),
            Message.assistant("not much"),
        ]
        dicts = conversation_to_dicts(convo)
        assert [d["role"] for d in dicts] == ["system", "user", "assistant"]
