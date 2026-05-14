"""Unit tests for `forge.llm.providers.bedrock`."""

from __future__ import annotations

from typing import Any

import pytest

from forge.llm.providers.bedrock import BedrockProvider
from forge.llm.providers.config import BedrockConfig


@pytest.fixture(autouse=True)
def _strip_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)


class TestProviderShape:
    def test_class_vars(self) -> None:
        assert BedrockProvider.name == "bedrock"
        assert BedrockProvider.litellm_prefix == "bedrock/"

    def test_default_config(self) -> None:
        client = BedrockProvider()
        assert isinstance(client.config, BedrockConfig)
        # The config carries a default region even without env.
        assert client.config.region == "us-east-1"

    def test_explicit_config(self) -> None:
        cfg = BedrockConfig()
        client = BedrockProvider(cfg)
        assert client.config is cfg

    def test_litellm_model(self) -> None:
        client = BedrockProvider()
        assert (
            client.litellm_model("anthropic.claude-opus-4-7") == "bedrock/anthropic.claude-opus-4-7"
        )
        assert (
            client.litellm_model("anthropic.claude-sonnet-4-6")
            == "bedrock/anthropic.claude-sonnet-4-6"
        )


class TestAuthKwargs:
    def test_no_keys_still_returns_default_region(self) -> None:
        # The boto3 credential chain may supply creds at call time; we still
        # emit the default region so LiteLLM knows where to look.
        client = BedrockProvider()
        kwargs = client.auth_kwargs()
        assert kwargs == {"aws_region_name": "us-east-1"}

    def test_full_credentials_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret/value")
        monkeypatch.setenv("AWS_REGION", "eu-west-2")
        client = BedrockProvider()
        kwargs = client.auth_kwargs()
        assert kwargs == {
            "aws_access_key_id": "AKIAEXAMPLE",
            "aws_secret_access_key": "secret/value",
            "aws_region_name": "eu-west-2",
        }

    def test_secret_values_extracted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Verify SecretStr fields are unwrapped — LiteLLM expects plain strings.
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-EXTRACT")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET-EXTRACT")
        client = BedrockProvider()
        kwargs = client.auth_kwargs()
        assert kwargs["aws_access_key_id"] == "AKIA-EXTRACT"
        assert kwargs["aws_secret_access_key"] == "SECRET-EXTRACT"

    def test_partial_creds_passed_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Operators sometimes set only AWS_REGION and rely on boto3's chain
        # for the actual credentials.
        monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
        client = BedrockProvider()
        kwargs = client.auth_kwargs()
        assert kwargs == {"aws_region_name": "ap-southeast-2"}


class TestAcompletionWiring:
    async def test_routes_through_bedrock_namespace(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = BedrockProvider()
        await client.acompletion(
            provider_model_id="anthropic.claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert captured["model"] == "bedrock/anthropic.claude-opus-4-7"
        assert captured["aws_region_name"] == "us-east-1"

    async def test_explicit_keys_passed_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AK")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SK")
        captured: dict[str, Any] = {}

        async def _fake_acompletion(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

        client = BedrockProvider()
        await client.acompletion(
            provider_model_id="anthropic.claude-haiku-4-5",
            messages=[],
        )
        assert captured["aws_access_key_id"] == "AK"
        assert captured["aws_secret_access_key"] == "SK"
