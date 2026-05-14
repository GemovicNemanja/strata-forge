"""Unit tests for `forge.llm.providers.vertex`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from forge.llm.providers.config import VertexConfig
from forge.llm.providers.vertex import VertexProvider, to_gemini_tool_schema


@pytest.fixture(autouse=True)
def _strip_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.delenv("GCP_REGION", raising=False)


class TestProviderShape:
    def test_class_vars(self) -> None:
        assert VertexProvider.name == "vertex"
        assert VertexProvider.litellm_prefix == "vertex_ai/"

    def test_default_config(self) -> None:
        client = VertexProvider()
        assert isinstance(client.config, VertexConfig)
        assert client.config.project is None
        # The config carries a default region even without env.
        assert client.config.region == "us-central1"

    def test_explicit_config(self) -> None:
        cfg = VertexConfig()
        client = VertexProvider(cfg)
        assert client.config is cfg


class TestLitellmModel:
    """Vertex routes Gemini and Claude through different LiteLLM namespaces."""

    def test_gemini_uses_vertex_ai_prefix(self) -> None:
        client = VertexProvider()
        assert client.litellm_model("gemini-3.1-pro") == "vertex_ai/gemini-3.1-pro"
        assert client.litellm_model("gemini-3.1-flash-lite") == "vertex_ai/gemini-3.1-flash-lite"

    def test_claude_on_vertex_uses_anthropic_vertex_prefix(self) -> None:
        client = VertexProvider()
        assert client.litellm_model("claude-opus-4-7") == "anthropic_vertex/claude-opus-4-7"
        assert client.litellm_model("claude-sonnet-4-6") == "anthropic_vertex/claude-sonnet-4-6"
        assert client.litellm_model("claude-haiku-4-5") == "anthropic_vertex/claude-haiku-4-5"

    def test_unknown_id_defaults_to_vertex_ai(self) -> None:
        # Any non-claude id falls into the vertex_ai namespace.
        client = VertexProvider()
        assert client.litellm_model("palm-something") == "vertex_ai/palm-something"


class TestAuthKwargs:
    def test_no_creds_still_returns_default_region(self) -> None:
        # The default region is always present (defaulted in the config).
        client = VertexProvider()
        kwargs = client.auth_kwargs()
        assert kwargs == {"vertex_location": "us-central1"}

    def test_project_and_region_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GCP_PROJECT", "my-project")
        monkeypatch.setenv("GCP_REGION", "europe-west4")
        client = VertexProvider()
        kwargs = client.auth_kwargs()
        assert kwargs == {
            "vertex_project": "my-project",
            "vertex_location": "europe-west4",
        }

    def test_credentials_path_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/sa.json")  # noqa: S108
        monkeypatch.setenv("GCP_PROJECT", "my-project")
        client = VertexProvider()
        kwargs = client.auth_kwargs()
        assert kwargs["vertex_credentials"] == "/tmp/sa.json"  # noqa: S108
        assert kwargs["vertex_project"] == "my-project"


class TestAcompletionWiring:
    async def test_gemini_routes_through_vertex_ai(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GCP_PROJECT", "proj")
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = VertexProvider()
        await client.acompletion(
            provider_model_id="gemini-3.1-pro",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert captured["model"] == "vertex_ai/gemini-3.1-pro"
        assert captured["vertex_project"] == "proj"

    async def test_claude_on_vertex_routes_through_anthropic_vertex(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GCP_PROJECT", "proj")
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = VertexProvider()
        await client.acompletion(
            provider_model_id="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert captured["model"] == "anthropic_vertex/claude-opus-4-7"
        assert captured["vertex_project"] == "proj"


class TestGeminiToolSchema:
    """Tool schema → Gemini's function-declaration format."""

    def test_basic_shape(self) -> None:
        schema = to_gemini_tool_schema(
            name="get_weather",
            description="Get the current weather for a location.",
            parameters_schema={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        )
        # Gemini uses `parameters` (like OpenAI) but no outer "type=function"
        # wrapper (like Anthropic). It's the third distinct shape.
        assert schema == {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        }

    def test_does_not_use_openai_or_anthropic_keys(self) -> None:
        schema = to_gemini_tool_schema("x", "desc", {"type": "object"})
        # Distinguishes Gemini from OpenAI (no outer "function") and from
        # Anthropic (uses "parameters" not "input_schema").
        assert "function" not in schema
        assert "input_schema" not in schema
        assert "parameters" in schema

    def test_from_pydantic_model_json_schema(self) -> None:
        class WeatherArgs(BaseModel):
            location: str = Field(..., description="City, country")

        schema = to_gemini_tool_schema(
            name="get_weather",
            description="Get weather.",
            parameters_schema=WeatherArgs.model_json_schema(),
        )
        params = schema["parameters"]
        assert params["type"] == "object"
        assert params["properties"]["location"]["description"] == "City, country"
