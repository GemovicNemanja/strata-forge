"""Unit tests for `forge.sync` — sync wrappers around the async LLM API."""

from __future__ import annotations

import json
import types
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from forge import sync
from forge.llm.client import LLMClient, StructuredResponse
from forge.llm.fallback import ModelFallback
from forge.llm.messages import Message
from forge.llm.tools import tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_response(
    *,
    text: str = "hello",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> types.SimpleNamespace:
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
        prompt_tokens=5,
        completion_tokens=2,
        total_tokens=7,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=0),
    )
    return types.SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def mock_litellm(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value=_fake_response())
    monkeypatch.setattr("litellm.acompletion", mock)
    monkeypatch.delenv("FORGE_DIAGNOSTIC_ENABLED", raising=False)
    return mock


class _Summary(BaseModel):
    title: str
    bullets: list[str]


class _WeatherArgs(BaseModel):
    location: str


@tool
async def _get_weather(args: _WeatherArgs) -> str:
    """Get weather."""
    return f"sunny in {args.location}"


# ---------------------------------------------------------------------------
# complete
# ---------------------------------------------------------------------------


class TestComplete:
    def test_runs_from_model(self, mock_litellm: AsyncMock) -> None:
        resp = sync.complete([Message.user("hi")], model="claude-opus-4-7")
        assert resp.text == "hello"
        assert resp.route.model == "claude-opus-4-7"
        assert mock_litellm.await_count == 1

    def test_runs_from_chain(self, mock_litellm: AsyncMock) -> None:
        resp = sync.complete(
            [Message.user("hi")],
            chain=[ModelFallback(model="claude-opus-4-7")],
        )
        assert resp.text == "hello"

    def test_runs_from_existing_client(self, mock_litellm: AsyncMock) -> None:
        client = LLMClient("claude-opus-4-7")
        resp = sync.complete([Message.user("hi")], client=client)
        assert resp.text == "hello"
        # Second call reuses the same client.
        resp2 = sync.complete([Message.user("bye")], client=client)
        assert resp2.text == "hello"

    def test_forwards_kwargs(self, mock_litellm: AsyncMock) -> None:
        sync.complete(
            [Message.user("hi")],
            model="claude-opus-4-7",
            temperature=0.5,
            max_tokens=100,
        )
        assert mock_litellm.await_args is not None
        kwargs = mock_litellm.await_args.kwargs
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 100

    def test_rejects_client_plus_model(self) -> None:
        client = LLMClient("claude-opus-4-7")
        with pytest.raises(ValueError, match="either"):
            sync.complete([Message.user("hi")], client=client, model="gpt-5.5")

    def test_rejects_client_plus_chain(self) -> None:
        client = LLMClient("claude-opus-4-7")
        with pytest.raises(ValueError, match="either"):
            sync.complete(
                [Message.user("hi")],
                client=client,
                chain=[ModelFallback(model="gpt-5.5")],
            )


# ---------------------------------------------------------------------------
# complete_structured
# ---------------------------------------------------------------------------


class TestCompleteStructured:
    def test_returns_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {"title": "Hi", "bullets": ["a"]}

        async def _fake(**_kwargs: Any) -> Any:
            return _fake_response(text=json.dumps(payload))

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        resp = sync.complete_structured(
            [Message.user("hi")],
            schema=_Summary,
            model="gpt-5.5",
            provider="openai",
        )
        assert isinstance(resp, StructuredResponse)
        assert resp.parsed.title == "Hi"
        assert resp.parsed.bullets == ["a"]

    def test_uses_existing_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload: dict[str, Any] = {"title": "X", "bullets": []}

        async def _fake(**_kwargs: Any) -> Any:
            return _fake_response(text=json.dumps(payload))

        monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
        client = LLMClient("gpt-5.5", provider="openai")
        resp = sync.complete_structured(
            [Message.user("hi")],
            schema=_Summary,
            client=client,
        )
        assert resp.parsed.title == "X"


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------


class TestStream:
    def test_drains_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _chunks() -> Any:
            for piece in ("hel", "lo"):
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
        chunks = list(sync.stream([Message.user("hi")], model="claude-opus-4-7"))
        assert len(chunks) == 2
        assert "".join(c.delta_text for c in chunks) == "hello"

    def test_returns_iterator(self, mock_litellm: AsyncMock) -> None:
        # Empty stream still yields an iterator (not a list).
        async def _chunks() -> Any:
            if False:  # pragma: no cover
                yield None
            return

        async def _fake(**_kwargs: Any) -> Any:
            return _chunks()

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("litellm.acompletion", AsyncMock(side_effect=_fake))
            result = sync.stream([Message.user("hi")], model="claude-opus-4-7")
            assert hasattr(result, "__next__")


# ---------------------------------------------------------------------------
# run_tool_loop
# ---------------------------------------------------------------------------


class TestRunToolLoop:
    def test_completes(self, mock_litellm: AsyncMock) -> None:
        resp = sync.run_tool_loop(
            [Message.user("hi")],
            tools=[_get_weather],
            model="claude-opus-4-7",
        )
        assert resp.text == "hello"

    def test_with_existing_client(self, mock_litellm: AsyncMock) -> None:
        client = LLMClient("claude-opus-4-7")
        resp = sync.run_tool_loop(
            [Message.user("hi")],
            tools=[_get_weather],
            client=client,
            max_iterations=3,
        )
        assert resp.text == "hello"

    def test_multi_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        resp = sync.run_tool_loop(
            [Message.user("weather?")],
            tools=[_get_weather],
            model="claude-opus-4-7",
        )
        assert resp.text == "It's sunny in Tokyo."


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_public_api(self) -> None:
        assert set(sync.__all__) == {
            "complete",
            "complete_structured",
            "stream",
            "run_tool_loop",
        }

    def test_functions_are_callable(self) -> None:
        for name in ("complete", "complete_structured", "stream", "run_tool_loop"):
            assert callable(getattr(sync, name))
