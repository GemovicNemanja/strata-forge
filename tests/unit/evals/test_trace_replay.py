"""Unit tests for `forge.evals.trace_replay`."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.evals.trace_replay import (
    ReplayOverrides,
    extract_messages_from_trace,
    replay_trace,
)
from forge.llm.messages import AssistantMessage, SystemMessage, UserMessage
from forge.llm.responses import LLMResponse, Usage
from forge.llm.routing import ModelRoute


def _new_response(text: str = "new") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(input_tokens=5, output_tokens=3),
        cost_usd=0.001,
        route=ModelRoute(
            model="claude-opus-4-7",
            provider="anthropic",
            provider_model_id="claude-opus-4-7",
        ),
    )


@dataclass
class FakeTrace:
    """Shape that the Langfuse SDK's get_trace returns."""

    input: Any
    output: Any


def _llm_client(new_text: str = "fresh-response") -> AsyncMock:
    client = AsyncMock()
    client.complete = AsyncMock(return_value=_new_response(new_text))
    return client


def _langfuse_client(trace: FakeTrace) -> MagicMock:
    client = MagicMock()
    client.get_trace = MagicMock(return_value=trace)
    return client


# ---------------------------------------------------------------------------
# extract_messages_from_trace
# ---------------------------------------------------------------------------


class TestExtractMessages:
    def test_list_of_role_content_dicts(self) -> None:
        trace = FakeTrace(
            input=[
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            output="hello",
        )
        messages, original = extract_messages_from_trace(trace)
        assert len(messages) == 3
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], UserMessage)
        assert isinstance(messages[2], AssistantMessage)
        assert original == "hello"

    def test_string_input_becomes_single_user_message(self) -> None:
        trace = FakeTrace(input="just a string", output="response")
        messages, _ = extract_messages_from_trace(trace)
        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "just a string"

    def test_dict_output_with_content_key(self) -> None:
        trace = FakeTrace(input="x", output={"content": "from-content-key"})
        _, original = extract_messages_from_trace(trace)
        assert original == "from-content-key"

    def test_dict_output_with_text_key(self) -> None:
        trace = FakeTrace(input="x", output={"text": "from-text-key"})
        _, original = extract_messages_from_trace(trace)
        assert original == "from-text-key"

    def test_none_output(self) -> None:
        trace = FakeTrace(input="x", output=None)
        _, original = extract_messages_from_trace(trace)
        assert original == ""

    def test_unknown_role_raises(self) -> None:
        trace = FakeTrace(
            input=[{"role": "spaceship", "content": "beep"}],
            output="x",
        )
        with pytest.raises(ValueError, match="Unknown role"):
            extract_messages_from_trace(trace)

    def test_unsupported_input_shape_raises(self) -> None:
        trace = FakeTrace(input=42, output="x")
        with pytest.raises(ValueError, match="Unsupported"):
            extract_messages_from_trace(trace)

    def test_non_dict_entry_in_input_list_raises(self) -> None:
        trace = FakeTrace(input=["not-a-dict"], output="x")
        with pytest.raises(ValueError, match="entry type"):
            extract_messages_from_trace(trace)

    def test_other_output_type_stringified(self) -> None:
        trace = FakeTrace(input="x", output=42)
        _, original = extract_messages_from_trace(trace)
        assert original == "42"


# ---------------------------------------------------------------------------
# replay_trace — happy path
# ---------------------------------------------------------------------------


class TestReplayHappyPath:
    async def test_basic_replay(self) -> None:
        trace = FakeTrace(
            input=[{"role": "user", "content": "what is 2+2?"}],
            output="4",
        )
        lf_client = _langfuse_client(trace)
        llm_client = _llm_client("fresh-4")
        result = await replay_trace(
            trace_id="trace-123",
            client=llm_client,
            langfuse_client=lf_client,
        )
        assert result.original_trace_id == "trace-123"
        assert result.original_response_text == "4"
        assert result.new_response.text == "fresh-4"
        lf_client.get_trace.assert_called_once_with(id="trace-123")

    async def test_messages_forwarded_to_llm_client(self) -> None:
        trace = FakeTrace(
            input=[
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hi"},
            ],
            output="hello",
        )
        llm_client = _llm_client()
        await replay_trace(
            trace_id="t",
            client=llm_client,
            langfuse_client=_langfuse_client(trace),
        )
        messages = llm_client.complete.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


class TestOverrides:
    async def test_system_message_override(self) -> None:
        trace = FakeTrace(
            input=[
                {"role": "system", "content": "original"},
                {"role": "user", "content": "hi"},
            ],
            output="x",
        )
        llm_client = _llm_client()
        await replay_trace(
            trace_id="t",
            client=llm_client,
            langfuse_client=_langfuse_client(trace),
            overrides=ReplayOverrides(system_message="rewritten"),
        )
        messages = llm_client.complete.call_args.kwargs["messages"]
        assert messages[0].content == "rewritten"
        # User message unchanged.
        assert messages[1].content == "hi"

    async def test_user_message_override_targets_last_user(self) -> None:
        trace = FakeTrace(
            input=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "response"},
                {"role": "user", "content": "second"},
            ],
            output="x",
        )
        llm_client = _llm_client()
        await replay_trace(
            trace_id="t",
            client=llm_client,
            langfuse_client=_langfuse_client(trace),
            overrides=ReplayOverrides(user_message="rewritten"),
        )
        messages = llm_client.complete.call_args.kwargs["messages"]
        # First user message unchanged; second was replaced.
        assert messages[0].content == "first"
        assert messages[2].content == "rewritten"

    async def test_sampling_overrides_passed_to_complete(self) -> None:
        trace = FakeTrace(input=[{"role": "user", "content": "hi"}], output="x")
        llm_client = _llm_client()
        await replay_trace(
            trace_id="t",
            client=llm_client,
            langfuse_client=_langfuse_client(trace),
            overrides=ReplayOverrides(temperature=0.5, max_tokens=128, top_p=0.95),
        )
        kwargs = llm_client.complete.call_args.kwargs
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 128
        assert kwargs["top_p"] == 0.95


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


class TestClientConstruction:
    async def test_raises_when_langfuse_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        from forge.config import reset_settings

        reset_settings()
        with pytest.raises(RuntimeError, match="not configured"):
            await replay_trace(
                trace_id="t",
                client=_llm_client(),
                langfuse_client=None,
            )

    async def test_raises_when_extra_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_HOST", "http://x")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setitem(sys.modules, "langfuse", None)
        from forge.config import reset_settings

        reset_settings()
        with pytest.raises(ImportError, match=r"\[langfuse\] extra"):
            await replay_trace(
                trace_id="t",
                client=_llm_client(),
                langfuse_client=None,
            )

    async def test_explicit_client_short_circuits_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Even with no LANGFUSE env vars set, an explicit client works.
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        from forge.config import reset_settings

        reset_settings()
        trace = FakeTrace(input=[{"role": "user", "content": "hi"}], output="x")
        result = await replay_trace(
            trace_id="t",
            client=_llm_client(),
            langfuse_client=_langfuse_client(trace),
        )
        assert result.original_trace_id == "t"

    async def test_builds_from_settings_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGFUSE_HOST", "http://test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        from forge.config import reset_settings

        reset_settings()
        # Inject a fake langfuse module so the lazy import succeeds.
        trace = FakeTrace(input=[{"role": "user", "content": "hi"}], output="recorded")
        lf_client = _langfuse_client(trace)
        fake_module = types.ModuleType("langfuse")
        fake_module.Langfuse = MagicMock(return_value=lf_client)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langfuse", fake_module)

        result = await replay_trace(
            trace_id="t",
            client=_llm_client(),
        )
        assert result.original_response_text == "recorded"
