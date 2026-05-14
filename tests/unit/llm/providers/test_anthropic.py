"""Unit tests for `forge.llm.providers.anthropic`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from forge.llm.providers.anthropic import AnthropicProvider, to_anthropic_tool_schema
from forge.llm.providers.config import AnthropicConfig


@pytest.fixture(autouse=True)
def _strip_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


class TestProviderShape:
    def test_class_vars(self) -> None:
        assert AnthropicProvider.name == "anthropic"
        assert AnthropicProvider.litellm_prefix == "anthropic/"

    def test_default_config(self) -> None:
        client = AnthropicProvider()
        assert isinstance(client.config, AnthropicConfig)
        assert client.config.api_key is None

    def test_explicit_config(self) -> None:
        cfg = AnthropicConfig()
        client = AnthropicProvider(cfg)
        assert client.config is cfg

    def test_litellm_model(self) -> None:
        client = AnthropicProvider()
        assert client.litellm_model("claude-opus-4-7") == "anthropic/claude-opus-4-7"
        assert client.litellm_model("claude-sonnet-4-6") == "anthropic/claude-sonnet-4-6"


class TestAuthKwargs:
    def test_empty_when_no_key(self) -> None:
        client = AnthropicProvider()
        assert client.auth_kwargs() == {}

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        client = AnthropicProvider()
        kwargs = client.auth_kwargs()
        assert kwargs == {"api_key": "sk-ant-test"}

    def test_secret_value_extracted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-extract-me")
        client = AnthropicProvider()
        assert client.auth_kwargs()["api_key"] == "sk-ant-extract-me"


class TestAcompletionWiring:
    async def test_calls_litellm_with_anthropic_prefix_and_auth(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = AnthropicProvider()
        result = await client.acompletion(
            provider_model_id="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result == "ok"
        assert captured["model"] == "anthropic/claude-opus-4-7"
        assert captured["api_key"] == "sk-ant-real"


class TestAnthropicToolSchema:
    """Tool schema → Anthropic's tool-use format."""

    def test_basic_shape(self) -> None:
        schema = to_anthropic_tool_schema(
            name="get_weather",
            description="Get the current weather for a location.",
            parameters_schema={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
        # Anthropic's flat shape: no outer "type: function" wrapper.
        assert schema == {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }

    def test_does_not_wrap_in_function_type(self) -> None:
        schema = to_anthropic_tool_schema("x", "desc", {"type": "object"})
        # Distinguishes Anthropic from OpenAI's nested shape.
        assert "type" not in schema
        assert "function" not in schema
        assert "input_schema" in schema

    def test_from_pydantic_model_json_schema(self) -> None:
        class WeatherArgs(BaseModel):
            location: str = Field(..., description="City, country")
            units: str = "celsius"

        schema = to_anthropic_tool_schema(
            name="get_weather",
            description="Get weather for a city.",
            parameters_schema=WeatherArgs.model_json_schema(),
        )
        assert schema["name"] == "get_weather"
        assert schema["description"] == "Get weather for a city."
        params = schema["input_schema"]
        assert params["type"] == "object"
        assert "location" in params["properties"]
        assert params["properties"]["location"]["description"] == "City, country"

    def test_empty_parameters(self) -> None:
        schema = to_anthropic_tool_schema(
            name="ping",
            description="Health check.",
            parameters_schema={"type": "object", "properties": {}},
        )
        assert schema["input_schema"] == {"type": "object", "properties": {}}
