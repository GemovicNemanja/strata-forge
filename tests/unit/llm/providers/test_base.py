"""Unit tests for `strata_forge.llm.providers.base`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast
from unittest.mock import AsyncMock

import pytest

from strata_forge.llm.providers.base import ProviderClient
from strata_forge.llm.providers.config import OpenAIConfig, ProviderConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from strata_forge.llm.registry import ProviderName


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


class TestResponsesSeam:
    """``aresponses`` / ``aresponses_stream`` call ``litellm.aresponses`` with the prefixed model."""

    async def test_aresponses_merges_auth_and_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = AsyncMock(return_value={"status": "completed"})
        monkeypatch.setattr("litellm.aresponses", mock)
        client = _FakeProvider(OpenAIConfig())
        result = await client.aresponses(
            provider_model_id="gpt-6.1-sol",
            request={"input": [], "store": False, "api_key": "request-wins"},
        )
        assert result == {"status": "completed"}
        assert mock.await_args is not None
        kwargs = mock.await_args.kwargs
        assert kwargs["model"] == "openai/gpt-6.1-sol"
        assert kwargs["store"] is False
        assert kwargs["extra"] == "marker"
        assert kwargs["api_key"] == "request-wins"
        assert "stream" not in kwargs

    async def test_aresponses_stream_forces_stream_and_yields_events(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _events() -> AsyncIterator[dict[str, str]]:
            yield {"type": "response.created"}
            yield {"type": "response.completed"}

        mock = AsyncMock(return_value=_events())
        monkeypatch.setattr("litellm.aresponses", mock)
        client = _FakeProvider(OpenAIConfig())
        events = [
            e
            async for e in client.aresponses_stream(
                provider_model_id="gpt-6.1-sol", request={"input": [], "stream": False}
            )
        ]
        assert [e["type"] for e in events] == ["response.created", "response.completed"]
        assert mock.await_args is not None
        assert mock.await_args.kwargs["stream"] is True

    def test_responses_auth_defaults_to_auth_kwargs(self) -> None:
        client = _FakeProvider(OpenAIConfig())
        assert client.responses_auth_kwargs() == client.auth_kwargs()

    async def test_extra_headers_from_auth_and_request_are_combined(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _HeaderProvider(_FakeProvider):
            def responses_auth_kwargs(self) -> dict[str, Any]:
                return {"api_key": "k", "extra_headers": {"A": "auth", "B": "auth"}}

        mock = AsyncMock(return_value={})
        monkeypatch.setattr("litellm.aresponses", mock)
        await _HeaderProvider(OpenAIConfig()).aresponses(
            provider_model_id="m", request={"input": [], "extra_headers": {"B": "request"}}
        )
        assert mock.await_args is not None
        assert mock.await_args.kwargs["extra_headers"] == {"A": "auth", "B": "request"}


def _model_map() -> dict[str, dict[str, Any]]:
    """LiteLLM's model map, typed."""
    import litellm

    return cast("dict[str, dict[str, Any]]", litellm.model_cost)  # pyright: ignore[reportUnknownMemberType]


class TestDescribeToLitellm:
    """A Responses route must stream natively even when LiteLLM's model map lacks the model."""

    @staticmethod
    def _responses_routes() -> list[tuple[str, str, str]]:
        from strata_forge.llm.registry import registry

        return [
            (model.name, route.provider, route.provider_model_id)
            for model in registry.list_models()
            for route in model.routes
            if route.wire_api == "responses"
        ]

    @staticmethod
    def _provider(provider: str) -> ProviderClient:
        from pydantic import SecretStr

        from strata_forge.llm.providers import AzureProvider, OpenAIProvider
        from strata_forge.llm.providers.config import AzureConfig

        if provider == "azure":
            return AzureProvider(
                AzureConfig(api_key=SecretStr("k"), endpoint="https://example.openai.azure.com")
            )
        return OpenAIProvider(OpenAIConfig(api_key=SecretStr("sk-test")))

    @staticmethod
    def _fakes_stream(provider: str, provider_model_id: str) -> bool:
        from litellm.llms.azure.responses.transformation import (  # pyright: ignore[reportMissingTypeStubs]
            AzureOpenAIResponsesAPIConfig,
        )
        from litellm.llms.openai.responses.transformation import (  # pyright: ignore[reportMissingTypeStubs]
            OpenAIResponsesAPIConfig,
        )

        config = (
            AzureOpenAIResponsesAPIConfig() if provider == "azure" else OpenAIResponsesAPIConfig()
        )
        return bool(
            config.should_fake_stream(
                model=provider_model_id, stream=True, custom_llm_provider=provider
            )
        )

    def test_there_are_responses_routes_to_check(self) -> None:
        assert len(self._responses_routes()) >= 6

    def test_every_responses_route_streams_natively_once_described(
        self, litellm_map_without: Any
    ) -> None:
        from strata_forge.llm.registry import registry

        routes = self._responses_routes()
        litellm_map_without(*{model_id for _, _, model_id in routes})
        for name, provider, model_id in routes:
            assert self._fakes_stream(provider, model_id), (name, provider)
            self._provider(provider).describe_to_litellm(model_id, registry.get(name))
            assert not self._fakes_stream(provider, model_id), (name, provider)

    def test_an_unmapped_model_is_registered_with_registry_prices(
        self, litellm_map_without: Any
    ) -> None:
        import litellm

        from strata_forge.llm.registry import registry

        litellm_map_without("gpt-6.1-sol")
        model = registry.get("gpt-6.1-sol")
        self._provider("openai").describe_to_litellm("gpt-6.1-sol", model)
        entry = _model_map()["openai/gpt-6.1-sol"]
        pricing = model.pricing_per_million_tokens
        assert entry["litellm_provider"] == "openai"
        assert entry["supports_native_streaming"] is True
        assert entry["input_cost_per_token"] == pricing.input / 1_000_000
        assert entry["output_cost_per_token"] == pricing.output / 1_000_000
        assert pricing.cache_read is not None
        assert entry["cache_read_input_token_cost"] == pricing.cache_read / 1_000_000
        assert entry["max_input_tokens"] == model.context_window
        assert entry["max_output_tokens"] == model.max_output_tokens
        # A `mode: responses` entry would route Chat Completions calls through LiteLLM's bridge.
        assert "mode" not in entry
        # LiteLLM's own cost callbacks now price the model instead of reporting zero.
        cost = float(
            litellm.completion_cost(  # pyright: ignore[reportUnknownArgumentType]
                model="openai/gpt-6.1-sol", prompt="hello", completion="there"
            )
        )
        assert cost > 0

    def test_a_mapped_model_is_left_as_litellm_describes_it(self, litellm_map_without: Any) -> None:
        from strata_forge.llm.registry import registry

        litellm_map_without()
        before = dict(_model_map()["o4-mini"])
        keys = set(_model_map())
        self._provider("openai").describe_to_litellm("o4-mini", registry.get("gpt-5.5-thinking"))
        assert _model_map()["o4-mini"] == before
        assert set(_model_map()) == keys
