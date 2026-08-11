"""Unit tests for `strata_forge.llm.providers.openai_compat`."""

from __future__ import annotations

from typing import Any

import pytest

from strata_forge.llm.providers.config import OpenAICompatConfig
from strata_forge.llm.providers.openai_compat import (
    UNAUTHENTICATED_API_KEY,
    OpenAICompatProvider,
)


@pytest.fixture(autouse=True)
def _strip_env(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.delenv("FORGE_OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("FORGE_OPENAI_COMPAT_API_KEY", raising=False)
    # The OpenAI client reads these itself, so auth_kwargs consults them — a developer who
    # happens to export one must not get a different result from CI.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)


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
        # The placeholder rides along with a configured base_url and no credential — see
        # test_unauthenticated_dev_deployment for why omitting the key is not an option.
        assert client.auth_kwargs() == {
            "api_base": "http://localhost:8000/v1",
            "api_key": UNAUTHENTICATED_API_KEY,
        }

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
        """An unauthenticated server gets a PLACEHOLDER key, not an omitted one.

        vLLM / TGI / SGLang commonly run without auth, but omitting the key does not produce an
        unauthenticated request — the OpenAI client refuses to build a request without one and
        fails before anything reaches the network ("Missing credentials. Please pass an
        `api_key` ..."). Because the failure is identical every time, it takes out an entire
        batch: a run against a local vLLM lost all 2098 rows without one reaching the server.
        """
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        client = OpenAICompatProvider()
        kwargs = client.auth_kwargs()
        assert kwargs["api_base"] == "http://localhost:8000/v1"
        assert kwargs["api_key"] == UNAUTHENTICATED_API_KEY

    @pytest.mark.parametrize("var", ["OPENAI_API_KEY", "OPENAI_ADMIN_KEY"])
    def test_an_ambient_key_is_not_overridden(
        self, monkeypatch: pytest.MonkeyPatch, var: str
    ) -> None:
        # The OpenAI client reads these itself. A caller who exported one means it for an
        # authenticated deployment, and sending a placeholder would replace a real credential
        # with a string that cannot possibly work.
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv(var, "sk-a-real-credential")
        assert "api_key" not in OpenAICompatProvider().auth_kwargs()

    def test_a_configured_key_wins_over_the_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("FORGE_OPENAI_COMPAT_API_KEY", "configured")
        assert OpenAICompatProvider().auth_kwargs()["api_key"] == "configured"

    def test_no_placeholder_without_a_base_url(self) -> None:
        # With no base_url there is no self-hosted server to be unauthenticated against: the call
        # falls through to api.openai.com, where a placeholder would turn a plain "no credentials"
        # into a puzzling rejection of one.
        assert OpenAICompatProvider().auth_kwargs() == {}


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
