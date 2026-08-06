"""Unit tests for `strata_forge.llm.providers.openai_compat`."""

from __future__ import annotations

from typing import Any

import pytest

from strata_forge.llm.providers.config import OpenAICompatConfig
from strata_forge.llm.providers.openai_compat import OpenAICompatProvider


@pytest.fixture(autouse=True)
def _strip_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.delenv("FORGE_OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("FORGE_OPENAI_COMPAT_API_KEY", raising=False)


class TestProviderShape:
    def test_class_vars(self) -> None:
        assert OpenAICompatProvider.name == "openai_compat"
        # Wire format identical to native OpenAI — discriminator is api_base.
        assert OpenAICompatProvider.litellm_prefix == "openai/"

    def test_default_config(self) -> None:
        client = OpenAICompatProvider()
        assert isinstance(client.config, OpenAICompatConfig)
        assert client.config.base_url is None
        assert client.config.api_key is None

    def test_explicit_config(self) -> None:
        cfg = OpenAICompatConfig()
        client = OpenAICompatProvider(cfg)
        assert client.config is cfg

    def test_explicit_base_url(self) -> None:
        cfg = OpenAICompatConfig(base_url="http://localhost:8000/v1")
        client = OpenAICompatProvider(cfg)
        assert client.config.base_url == "http://localhost:8000/v1"

    def test_litellm_model(self) -> None:
        # Uses the same prefix as the native OpenAI provider.
        client = OpenAICompatProvider()
        assert client.litellm_model("meta-llama/Llama-3.1-70B") == "openai/meta-llama/Llama-3.1-70B"
        assert client.litellm_model("custom-model") == "openai/custom-model"


class TestAuthKwargs:
    def test_empty_when_no_base_url(self) -> None:
        # Without a base_url, this provider has nothing to add — falls
        # back to native OpenAI behavior, which isn't useful but doesn't
        # break either.
        client = OpenAICompatProvider()
        assert client.auth_kwargs() == {}

    def test_base_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        client = OpenAICompatProvider()
        assert client.auth_kwargs() == {"api_base": "http://localhost:8000/v1"}

    def test_base_url_and_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_API_KEY", "dummy")
        client = OpenAICompatProvider()
        assert client.auth_kwargs() == {
            "api_base": "http://localhost:8000/v1",
            "api_key": "dummy",
        }

    def test_secret_value_extracted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_API_KEY", "extract-me")
        client = OpenAICompatProvider()
        assert client.auth_kwargs()["api_key"] == "extract-me"

    def test_unauthenticated_dev_deployment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # vLLM / TGI / SGLang in dev mode often run without auth.
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        client = OpenAICompatProvider()
        kwargs = client.auth_kwargs()
        assert "api_base" in kwargs
        assert "api_key" not in kwargs


class TestAcompletionWiring:
    async def test_routes_through_custom_base_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://vllm:8000/v1")
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_API_KEY", "dummy")
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = OpenAICompatProvider()
        await client.acompletion(
            provider_model_id="meta-llama/Llama-3.1-70B-Instruct",
            messages=[{"role": "user", "content": "hi"}],
        )
        # Uses the openai/ namespace — LiteLLM uses api_base to route
        # away from api.openai.com.
        assert captured["model"] == "openai/meta-llama/Llama-3.1-70B-Instruct"
        assert captured["api_base"] == "http://vllm:8000/v1"
        assert captured["api_key"] == "dummy"

    async def test_call_site_base_url_overrides_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://default:8000/v1")
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = OpenAICompatProvider()
        await client.acompletion(
            provider_model_id="custom-model",
            messages=[],
            api_base="http://override:9000/v1",  # call-site override
        )
        assert captured["api_base"] == "http://override:9000/v1"
