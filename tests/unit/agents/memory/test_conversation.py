"""Unit tests for `strata_forge.agents.memory.conversation`."""

from __future__ import annotations

import pytest

from strata_forge.agents import AssistantMessage, ConversationMemory, SystemMessage, UserMessage

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_default(self) -> None:
        memory = ConversationMemory()
        assert memory.messages == ()
        assert memory.system_message is None
        assert len(memory) == 0

    def test_system_message_as_instance(self) -> None:
        sys_msg = SystemMessage(content="be helpful")
        memory = ConversationMemory(system_message=sys_msg)
        assert memory.system_message is sys_msg
        assert memory.messages == (sys_msg,)

    def test_system_message_as_string(self) -> None:
        memory = ConversationMemory(system_message="be helpful")
        assert isinstance(memory.system_message, SystemMessage)
        assert memory.system_message.content == "be helpful"


# ---------------------------------------------------------------------------
# Appending
# ---------------------------------------------------------------------------


class TestAppend:
    def test_append_user_message(self) -> None:
        memory = ConversationMemory()
        memory.append(UserMessage(content="hi"))
        assert len(memory) == 1
        assert memory.messages[0].content == "hi"

    def test_append_assistant_message(self) -> None:
        memory = ConversationMemory()
        memory.append(AssistantMessage(content="hello"))
        assert len(memory) == 1

    def test_append_rejects_system_message(self) -> None:
        memory = ConversationMemory()
        with pytest.raises(ValueError, match="system message"):
            memory.append(SystemMessage(content="oops"))

    def test_extend_adds_multiple(self) -> None:
        memory = ConversationMemory()
        memory.extend([UserMessage(content="a"), AssistantMessage(content="b")])
        assert len(memory) == 2

    def test_append_user_convenience(self) -> None:
        memory = ConversationMemory()
        memory.append_user("hello")
        assert isinstance(memory.messages[0], UserMessage)
        assert memory.messages[0].content == "hello"

    def test_append_assistant_convenience(self) -> None:
        memory = ConversationMemory()
        memory.append_assistant("response")
        assert isinstance(memory.messages[0], AssistantMessage)


# ---------------------------------------------------------------------------
# messages / non_system_messages
# ---------------------------------------------------------------------------


class TestMessages:
    def test_messages_includes_system_first(self) -> None:
        memory = ConversationMemory(system_message="sys")
        memory.append_user("u1")
        memory.append_assistant("a1")
        names = [type(m).__name__ for m in memory.messages]
        assert names == ["SystemMessage", "UserMessage", "AssistantMessage"]

    def test_non_system_messages_excludes_system(self) -> None:
        memory = ConversationMemory(system_message="sys")
        memory.append_user("u1")
        names = [type(m).__name__ for m in memory.non_system_messages]
        assert names == ["UserMessage"]

    def test_messages_returns_tuple(self) -> None:
        memory = ConversationMemory()
        memory.append_user("hi")
        assert isinstance(memory.messages, tuple)


# ---------------------------------------------------------------------------
# clear / replace_system_message
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_drops_history_preserves_system(self) -> None:
        memory = ConversationMemory(system_message="sys")
        memory.append_user("u1")
        memory.append_assistant("a1")
        memory.clear()
        assert memory.non_system_messages == ()
        assert memory.system_message is not None


class TestReplaceSystem:
    def test_replace_with_string(self) -> None:
        memory = ConversationMemory(system_message="original")
        memory.replace_system_message("rewritten")
        assert memory.system_message is not None
        assert memory.system_message.content == "rewritten"

    def test_replace_with_instance(self) -> None:
        memory = ConversationMemory()
        new = SystemMessage(content="fresh")
        memory.replace_system_message(new)
        assert memory.system_message is new

    def test_replace_with_none_clears(self) -> None:
        memory = ConversationMemory(system_message="original")
        memory.replace_system_message(None)
        assert memory.system_message is None


# ---------------------------------------------------------------------------
# trim_to_messages
# ---------------------------------------------------------------------------


class TestTrimToMessages:
    def test_keeps_last_n_messages(self) -> None:
        memory = ConversationMemory()
        for i in range(5):
            memory.append_user(f"msg-{i}")
        memory.trim_to_messages(2)
        assert len(memory.non_system_messages) == 2
        assert memory.non_system_messages[0].content == "msg-3"
        assert memory.non_system_messages[1].content == "msg-4"

    def test_no_op_when_under_budget(self) -> None:
        memory = ConversationMemory()
        memory.append_user("only")
        memory.trim_to_messages(5)
        assert len(memory.non_system_messages) == 1

    def test_system_message_preserved(self) -> None:
        memory = ConversationMemory(system_message="sys")
        memory.append_user("u1")
        memory.append_assistant("a1")
        memory.trim_to_messages(0)
        # All non-system gone; system survives.
        assert memory.non_system_messages == ()
        assert memory.system_message is not None

    def test_negative_max_rejected(self) -> None:
        memory = ConversationMemory()
        with pytest.raises(ValueError, match="max_messages"):
            memory.trim_to_messages(-1)

    def test_zero_keeps_only_system(self) -> None:
        memory = ConversationMemory(system_message="sys")
        memory.append_user("u")
        memory.trim_to_messages(0)
        assert len(memory.messages) == 1


# ---------------------------------------------------------------------------
# trim_to_tokens
# ---------------------------------------------------------------------------


class TestTrimToTokens:
    def test_drops_oldest_until_fits(self) -> None:
        # Each message ~1 token per character; 10 messages of "xxx".
        memory = ConversationMemory()
        for _ in range(10):
            memory.append_user("xxx")
        # Tight budget — should drop most.
        memory.trim_to_tokens(5)
        kept = len(memory.non_system_messages)
        assert kept < 10

    def test_no_op_when_under_budget(self) -> None:
        memory = ConversationMemory()
        memory.append_user("hi")
        memory.trim_to_tokens(10_000)
        assert len(memory.non_system_messages) == 1

    def test_system_message_preserved_even_under_pressure(self) -> None:
        memory = ConversationMemory(system_message="x" * 100)
        memory.append_user("y" * 100)
        memory.trim_to_tokens(5)
        # System is never evicted.
        assert memory.system_message is not None
        # Non-system is dropped since it exceeds the budget by itself.
        assert memory.non_system_messages == ()

    def test_negative_max_rejected(self) -> None:
        memory = ConversationMemory()
        with pytest.raises(ValueError, match="max_tokens"):
            memory.trim_to_tokens(-1)

    def test_zero_budget_drops_all_history(self) -> None:
        memory = ConversationMemory()
        memory.append_user("hi")
        memory.trim_to_tokens(0)
        assert memory.non_system_messages == ()

    def test_model_hint_forwarded(self) -> None:
        # No crash when a specific model is given.
        memory = ConversationMemory()
        memory.append_user("hello world")
        memory.trim_to_tokens(2, model="claude-opus-4-7")

    def test_handles_messages_without_text_content(self) -> None:
        # An AssistantMessage with content=None and tool_calls is valid.
        memory = ConversationMemory()
        memory.append(AssistantMessage(content=None, tool_calls=[]))
        # Should not raise.
        memory.trim_to_tokens(100)

    def test_handles_multimodal_user_message_list_content(self) -> None:
        # UserMessage.content can be a list of content parts (string +
        # ImageContent). The text extractor should pull text strings
        # out and ignore non-text parts gracefully.
        memory = ConversationMemory()
        # Simulate a list-content user message; the parts list mixes
        # a string with a fake image-shaped object.
        from dataclasses import dataclass

        @dataclass
        class _FakeImagePart:
            url: str

        # Use model_construct to bypass Pydantic's str-only validator.
        msg = UserMessage.model_construct(
            content=["the text part", _FakeImagePart(url="https://x.test")]
        )
        memory.append(msg)
        # trim_to_tokens walks the content and shouldn't crash; the
        # text part counts toward the token budget.
        memory.trim_to_tokens(10_000)
        memory.trim_to_tokens(0)
        # Aggressive trim drops the message entirely.
        assert memory.non_system_messages == ()
