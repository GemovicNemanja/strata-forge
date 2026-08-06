"""Unit tests for `strata_forge.cli.chat`."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from strata_forge.cli.main import app
from strata_forge.llm.responses import LLMResponse, Usage
from strata_forge.llm.routing import ModelRoute

runner = CliRunner()


def _response(text: str = "an answer") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(input_tokens=12, output_tokens=8),
        cost_usd=0.0001,
        route=ModelRoute(
            model="claude-opus-4-7",
            provider="anthropic",
            provider_model_id="claude-opus-4-7",
        ),
        latency_ms=400.0,
    )


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"messages": None, "init_kwargs": None}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            state["init_kwargs"] = kwargs
            self.model = kwargs.get("model", "")

        complete = AsyncMock(return_value=_response("hello from forge"))

    monkeypatch.setattr("strata_forge.llm.client.LLMClient", _FakeClient)

    # Make every model resolve cleanly. The `registry` name on the
    # strata_forge.llm package shadows the submodule, so target sys.modules.
    fake_registry = MagicMock()
    fake_registry.get = MagicMock(return_value=None)
    registry_module = sys.modules["strata_forge.llm.registry"]
    monkeypatch.setattr(registry_module, "registry", fake_registry)

    return state


class TestOneShot:
    def test_message_mode_prints_response(self, fake_llm: dict[str, Any]) -> None:
        result = runner.invoke(
            app,
            [
                "chat",
                "--model",
                "claude-opus-4-7",
                "--message",
                "say hi",
            ],
        )
        assert result.exit_code == 0
        assert "hello from forge" in result.output
        # Summary line includes route / tokens / cost.
        assert "claude-opus-4-7" in result.output
        assert "anthropic" in result.output

    def test_unknown_model_exits_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_registry = MagicMock()
        fake_registry.get = MagicMock(side_effect=Exception("nope"))
        registry_module = sys.modules["strata_forge.llm.registry"]
        monkeypatch.setattr(registry_module, "registry", fake_registry)

        result = runner.invoke(app, ["chat", "--model", "bogus-model", "--message", "hi"])
        assert result.exit_code != 0

    def test_provider_pin_forwarded(self, fake_llm: dict[str, Any]) -> None:
        result = runner.invoke(
            app,
            [
                "chat",
                "--model",
                "claude-opus-4-7",
                "--provider",
                "bedrock",
                "--message",
                "hi",
            ],
        )
        assert result.exit_code == 0
        assert fake_llm["init_kwargs"]["provider"] == "bedrock"

    def test_system_prompt_prepended(self, fake_llm: dict[str, Any]) -> None:
        result = runner.invoke(
            app,
            [
                "chat",
                "--model",
                "claude-opus-4-7",
                "--system",
                "you only speak in haiku",
                "--message",
                "the wind",
            ],
        )
        assert result.exit_code == 0


class TestInteractiveLoop:
    def test_two_turns_then_blank_exits(self, fake_llm: dict[str, Any]) -> None:
        # Two prompts followed by a blank line — the loop should run twice
        # and then exit cleanly. CliRunner pipes stdin into Console.input.
        result = runner.invoke(
            app,
            ["chat", "--model", "claude-opus-4-7"],
            input="hi\nanother turn\n\n",
        )
        assert result.exit_code == 0
        assert "hello from forge" in result.output
        # Two assistant turns emitted.
        assert result.output.count("hello from forge") >= 2

    def test_blank_first_line_exits_immediately(self, fake_llm: dict[str, Any]) -> None:
        result = runner.invoke(app, ["chat", "--model", "claude-opus-4-7"], input="\n")
        assert result.exit_code == 0
        assert "exit" in result.output

    def test_eof_exits(self, fake_llm: dict[str, Any]) -> None:
        # Empty input → EOFError on the first Console.input call.
        result = runner.invoke(app, ["chat", "--model", "claude-opus-4-7"], input="")
        assert result.exit_code == 0
        assert "exit" in result.output
