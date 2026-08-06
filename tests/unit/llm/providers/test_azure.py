"""Unit tests for `strata_forge.llm.providers.azure`."""

from __future__ import annotations

from typing import Any

import pytest

from strata_forge.llm.providers.azure import AzureProvider
from strata_forge.llm.providers.config import AzureConfig


@pytest.fixture(autouse=True)
def _strip_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)


class TestProviderShape:
    def test_class_vars(self) -> None:
        assert AzureProvider.name == "azure"
        assert AzureProvider.litellm_prefix == "azure/"

    def test_default_config(self) -> None:
        client = AzureProvider()
        assert isinstance(client.config, AzureConfig)
        # The config carries a default api_version even without env.
        assert client.config.api_version == "2025-10-01-preview"
        assert client.config.api_key is None
        assert client.config.endpoint is None

    def test_explicit_config(self) -> None:
        cfg = AzureConfig()
        client = AzureProvider(cfg)
        assert client.config is cfg

    def test_litellm_model(self) -> None:
        # Note: in practice the model id is the user's *deployment name* in
        # Azure, which may or may not match the OpenAI model name. The
        # registry stores the OpenAI name as a sensible default.
        client = AzureProvider()
        assert client.litellm_model("gpt-5.5") == "azure/gpt-5.5"
        assert client.litellm_model("gpt-5.5-pro") == "azure/gpt-5.5-pro"


class TestAuthKwargs:
    def test_no_keys_still_returns_default_api_version(self) -> None:
        # api_version always emitted; api_key/api_base dropped when unset
        # so LiteLLM can fall back to its own env lookup.
        client = AzureProvider()
        kwargs = client.auth_kwargs()
        assert kwargs == {"api_version": "2025-10-01-preview"}

    def test_full_credentials_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://ex.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2026-04-01")
        client = AzureProvider()
        kwargs = client.auth_kwargs()
        assert kwargs == {
            "api_key": "az-key",
            "api_base": "https://ex.openai.azure.com",
            "api_version": "2026-04-01",
        }

    def test_secret_value_extracted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-extract-me")
        client = AzureProvider()
        assert client.auth_kwargs()["api_key"] == "az-extract-me"

    def test_partial_creds_passed_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Operators sometimes set only the endpoint and rely on Azure's
        # managed-identity / Entra ID auth handled by LiteLLM separately.
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
        client = AzureProvider()
        kwargs = client.auth_kwargs()
        assert kwargs == {
            "api_base": "https://example.openai.azure.com",
            "api_version": "2025-10-01-preview",
        }


class TestAcompletionWiring:
    async def test_routes_through_azure_namespace(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://ex.openai.azure.com")
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = AzureProvider()
        await client.acompletion(
            provider_model_id="gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert captured["model"] == "azure/gpt-5.5"
        assert captured["api_key"] == "k"
        assert captured["api_base"] == "https://ex.openai.azure.com"
        assert captured["api_version"] == "2025-10-01-preview"

    async def test_call_site_api_version_overrides_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-10-01-preview")
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = AzureProvider()
        await client.acompletion(
            provider_model_id="gpt-5.5",
            messages=[],
            api_version="2026-04-01",  # call-site override
        )
        assert captured["api_version"] == "2026-04-01"
