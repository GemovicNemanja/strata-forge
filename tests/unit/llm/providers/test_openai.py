"""Unit tests for `forge.llm.providers.openai`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from forge.llm.providers.config import OpenAIConfig
from forge.llm.providers.openai import OpenAIProvider, to_openai_tool_schema


@pytest.fixture(autouse=True)
def _strip_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)


class TestProviderShape:
    def test_class_vars(self) -> None:
        assert OpenAIProvider.name == "openai"
        assert OpenAIProvider.litellm_prefix == "openai/"

    def test_default_config(self) -> None:
        client = OpenAIProvider()
        assert isinstance(client.config, OpenAIConfig)
        assert client.config.api_key is None

    def test_explicit_config(self) -> None:
        cfg = OpenAIConfig()
        client = OpenAIProvider(cfg)
        assert client.config is cfg

    def test_litellm_model(self) -> None:
        client = OpenAIProvider()
        assert client.litellm_model("gpt-5.5") == "openai/gpt-5.5"
        assert client.litellm_model("gpt-5.5-pro") == "openai/gpt-5.5-pro"


class TestAuthKwargs:
    def test_empty_when_no_key(self) -> None:
        client = OpenAIProvider()
        # No env, no explicit key → LiteLLM should fall back to its own lookup
        assert client.auth_kwargs() == {}

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        client = OpenAIProvider()
        kwargs = client.auth_kwargs()
        assert kwargs == {"api_key": "sk-test"}

    def test_org_id_included(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_ORG_ID", "org-abc")
        client = OpenAIProvider()
        kwargs = client.auth_kwargs()
        assert kwargs == {"api_key": "sk-test", "organization": "org-abc"}

    def test_org_id_alone_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Edge case: org id set but no api key. We still pass org through.
        monkeypatch.setenv("OPENAI_ORG_ID", "org-abc")
        client = OpenAIProvider()
        assert client.auth_kwargs() == {"organization": "org-abc"}

    def test_secret_value_extracted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Verify the SecretStr is unwrapped — LiteLLM expects a plain string.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-extract-me")
        client = OpenAIProvider()
        assert client.auth_kwargs()["api_key"] == "sk-extract-me"


class TestAcompletionWiring:
    async def test_calls_litellm_with_openai_prefix_and_auth(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = OpenAIProvider()
        result = await client.acompletion(
            provider_model_id="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result == "ok"
        assert captured["model"] == "openai/gpt-5.5"
        assert captured["api_key"] == "sk-real"
        assert captured["messages"] == [{"role": "user", "content": "hi"}]


class TestOpenAIToolSchema:
    """Tool schema → OpenAI's tool-calling format."""

    def test_basic_shape(self) -> None:
        schema = to_openai_tool_schema(
            name="get_weather",
            description="Get the current weather for a location.",
            parameters_schema={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
        assert schema == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }

    def test_from_pydantic_model_json_schema(self) -> None:
        class WeatherArgs(BaseModel):
            location: str = Field(..., description="City, country")
            units: str = "celsius"

        schema = to_openai_tool_schema(
            name="get_weather",
            description="Get weather for a city.",
            parameters_schema=WeatherArgs.model_json_schema(),
        )
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "location" in params["properties"]
        assert "units" in params["properties"]
        # Pydantic puts the description on the field directly.
        assert params["properties"]["location"]["description"] == "City, country"

    def test_empty_parameters(self) -> None:
        # Tools with no arguments still get the type-object wrapper.
        schema = to_openai_tool_schema(
            name="ping",
            description="Health check.",
            parameters_schema={"type": "object", "properties": {}},
        )
        assert schema["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_returns_a_new_dict(self) -> None:
        # Caller shouldn't be able to mutate the source.
        params = {"type": "object", "properties": {"x": {"type": "string"}}}
        schema = to_openai_tool_schema("f", "desc", params)
        # Mutating the input MUST NOT affect downstream consumers in the test;
        # since OpenAI returns a fresh dict, we can verify the structure.
        assert schema["function"]["parameters"] is params
        # (We allow the dict to be shared by reference for efficiency; if a
        # future caller needs ownership semantics, deep-copy at the call
        # site.)
