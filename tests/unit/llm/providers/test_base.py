"""Unit tests for `forge.llm.providers.base`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from forge.llm.providers.base import ProviderClient
from forge.llm.providers.config import OpenAIConfig, ProviderConfig

if TYPE_CHECKING:
    from forge.llm.registry import ProviderName


class _FakeProvider(ProviderClient):
    """Concrete test subclass — minimal viable provider client."""

    name: ClassVar[ProviderName] = "openai"
    litellm_prefix: ClassVar[str] = "openai/"

    def auth_kwargs(self) -> dict[str, Any]:
        # Return a recognizable marker so we can assert it gets passed through.
        return {"api_key": "test-key", "extra": "marker"}


class TestAbstractBase:
    def test_base_is_abstract(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            ProviderClient(OpenAIConfig())  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        client = _FakeProvider(OpenAIConfig())
        assert client.name == "openai"
        assert client.litellm_prefix == "openai/"

    def test_config_stored_on_instance(self) -> None:
        cfg = OpenAIConfig()
        client = _FakeProvider(cfg)
        assert client.config is cfg

    def test_accepts_provider_config_subclass(self) -> None:
        # The base accepts any ProviderConfig subclass.
        client = _FakeProvider(ProviderConfig())
        assert isinstance(client.config, ProviderConfig)


class TestLitellmModel:
    def test_prefix_concatenation(self) -> None:
        client = _FakeProvider(OpenAIConfig())
        assert client.litellm_model("gpt-5.5") == "openai/gpt-5.5"

    def test_with_dot_in_id(self) -> None:
        client = _FakeProvider(OpenAIConfig())
        assert client.litellm_model("gpt-5.5-pro") == "openai/gpt-5.5-pro"

    def test_with_namespace_in_id(self) -> None:
        # Bedrock-style IDs contain dots and dashes.
        client = _FakeProvider(OpenAIConfig())
        assert (
            client.litellm_model("anthropic.claude-opus-4-7") == "openai/anthropic.claude-opus-4-7"
        )


class TestAuthKwargs:
    def test_subclass_returns_provider_specific_kwargs(self) -> None:
        client = _FakeProvider(OpenAIConfig())
        assert client.auth_kwargs() == {"api_key": "test-key", "extra": "marker"}


class TestAcompletion:
    async def test_calls_litellm_with_prefixed_model_and_auth_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "fake-response"

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = _FakeProvider(OpenAIConfig())
        result = await client.acompletion(
            provider_model_id="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
        )

        assert result == "fake-response"
        assert captured["model"] == "openai/gpt-5.5"
        assert captured["messages"] == [{"role": "user", "content": "hi"}]
        # auth_kwargs merged into the call
        assert captured["api_key"] == "test-key"
        assert captured["extra"] == "marker"
        # call-site kwargs preserved
        assert captured["temperature"] == 0.7

    async def test_call_site_kwargs_override_auth_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> None:
            captured.update(kwargs)

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = _FakeProvider(OpenAIConfig())
        await client.acompletion(
            provider_model_id="gpt-5.5",
            messages=[],
            api_key="override-key",
        )

        # call-site api_key wins
        assert captured["api_key"] == "override-key"


class TestAstream:
    async def test_passes_stream_true_and_yields_chunks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        async def _async_iter() -> Any:
            yield "chunk-1"
            yield "chunk-2"

        async def _fake_acompletion(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _async_iter()

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = _FakeProvider(OpenAIConfig())
        collected: list[Any] = []
        async for chunk in client.astream(
            provider_model_id="gpt-5.5",
            messages=[],
        ):
            collected.append(chunk)

        assert collected == ["chunk-1", "chunk-2"]
        assert captured["stream"] is True
        assert captured["model"] == "openai/gpt-5.5"
