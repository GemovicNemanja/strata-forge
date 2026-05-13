"""Unit tests for `forge.llm.providers.config`."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from forge.llm.providers.config import (
    AnthropicConfig,
    AzureConfig,
    BedrockConfig,
    OpenAICompatConfig,
    OpenAIConfig,
    VertexConfig,
)


@pytest.fixture(autouse=True)
def _strip_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Each test starts from a clean env so default-construction paths are deterministic."""
    for var in (
        "OPENAI_API_KEY",
        "OPENAI_ORG_ID",
        "ANTHROPIC_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GCP_PROJECT",
        "GCP_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "FORGE_OPENAI_COMPAT_BASE_URL",
        "FORGE_OPENAI_COMPAT_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


class TestOpenAIConfig:
    def test_defaults(self) -> None:
        cfg = OpenAIConfig()
        assert cfg.api_key is None
        assert cfg.org_id is None
        assert cfg.enabled is False

    def test_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_ORG_ID", "org-abc")
        cfg = OpenAIConfig()
        assert isinstance(cfg.api_key, SecretStr)
        assert cfg.api_key.get_secret_value() == "sk-test"
        assert cfg.org_id == "org-abc"
        assert cfg.enabled is True

    def test_secret_redacted_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-DO-NOT-LEAK")
        cfg = OpenAIConfig()
        assert "DO-NOT-LEAK" not in repr(cfg)
        assert "**" in repr(cfg)


class TestAnthropicConfig:
    def test_defaults(self) -> None:
        cfg = AnthropicConfig()
        assert cfg.api_key is None
        assert cfg.enabled is False

    def test_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        cfg = AnthropicConfig()
        assert cfg.api_key is not None
        assert cfg.api_key.get_secret_value() == "sk-ant-test"
        assert cfg.enabled is True


class TestVertexConfig:
    def test_defaults(self) -> None:
        cfg = VertexConfig()
        assert cfg.application_credentials is None
        assert cfg.project is None
        assert cfg.region == "us-central1"
        assert cfg.enabled is False

    def test_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/sa.json")  # noqa: S108
        monkeypatch.setenv("GCP_PROJECT", "my-project")
        monkeypatch.setenv("GCP_REGION", "europe-west4")
        cfg = VertexConfig()
        assert cfg.application_credentials == "/tmp/sa.json"  # noqa: S108
        assert cfg.project == "my-project"
        assert cfg.region == "europe-west4"
        assert cfg.enabled is True

    def test_partial_creds_not_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GCP_PROJECT", "my-project")
        # No GOOGLE_APPLICATION_CREDENTIALS.
        assert VertexConfig().enabled is False


class TestBedrockConfig:
    def test_defaults(self) -> None:
        cfg = BedrockConfig()
        assert cfg.access_key_id is None
        assert cfg.secret_access_key is None
        assert cfg.region == "us-east-1"
        # AWS credential chain may still supply creds; enabled iff region is set.
        assert cfg.enabled is True

    def test_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret/value")
        monkeypatch.setenv("AWS_REGION", "eu-west-2")
        cfg = BedrockConfig()
        assert cfg.access_key_id is not None
        assert cfg.access_key_id.get_secret_value() == "AKIAEXAMPLE"
        assert cfg.secret_access_key is not None
        assert cfg.secret_access_key.get_secret_value() == "secret/value"
        assert cfg.region == "eu-west-2"

    def test_secret_redacted_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "BIG-SECRET-DO-NOT-LEAK")
        cfg = BedrockConfig()
        assert "DO-NOT-LEAK" not in repr(cfg)


class TestAzureConfig:
    def test_defaults(self) -> None:
        cfg = AzureConfig()
        assert cfg.api_key is None
        assert cfg.endpoint is None
        assert cfg.api_version == "2025-10-01-preview"
        assert cfg.enabled is False

    def test_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://ex.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2026-04-01")
        cfg = AzureConfig()
        assert cfg.api_key is not None
        assert cfg.api_key.get_secret_value() == "az-key"
        assert cfg.endpoint == "https://ex.openai.azure.com"
        assert cfg.api_version == "2026-04-01"
        assert cfg.enabled is True

    def test_enabled_requires_both_key_and_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        # No endpoint.
        assert AzureConfig().enabled is False


class TestOpenAICompatConfig:
    def test_defaults(self) -> None:
        cfg = OpenAICompatConfig()
        assert cfg.base_url is None
        assert cfg.api_key is None
        assert cfg.enabled is False

    def test_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_API_KEY", "dummy")
        cfg = OpenAICompatConfig()
        assert cfg.base_url == "http://localhost:8000/v1"
        assert cfg.api_key is not None
        assert cfg.api_key.get_secret_value() == "dummy"
        assert cfg.enabled is True

    def test_enabled_only_requires_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # vLLM / TGI / SGLang often run without auth in development.
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        cfg = OpenAICompatConfig()
        assert cfg.api_key is None
        assert cfg.enabled is True
