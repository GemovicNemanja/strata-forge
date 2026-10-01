"""Unit tests for `strata_forge.llm.routing`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from strata_forge.core.errors import RegistryError
from strata_forge.llm.cost import compute_cost
from strata_forge.llm.registry import (
    REGISTRY_DATA_FILE,
    Capabilities,
    Model,
    Pricing,
    ProviderName,
    ProviderRoute,
    Registry,
)
from strata_forge.llm.registry import registry as global_registry
from strata_forge.llm.responses import Usage
from strata_forge.llm.routing import ModelRoute, resolve


def _make_test_registry() -> Registry:
    """A tiny registry with three models covering single + multi-route shapes."""
    return Registry(
        [
            Model(
                name="single-route",
                vendor="anthropic",
                tier="flagship",
                context_window=1024,
                max_output_tokens=256,
                modalities=["text"],
                capabilities=Capabilities(tool_calling=True),
                pricing_per_million_tokens=Pricing(input=1.0, output=2.0),
                routes=[
                    ProviderRoute(
                        provider="anthropic",
                        provider_model_id="single-native",
                        is_default=True,
                    ),
                ],
                aliases=["lonely"],
            ),
            Model(
                name="multi-route",
                vendor="anthropic",
                tier="flagship",
                context_window=1024,
                max_output_tokens=256,
                modalities=["text"],
                capabilities=Capabilities(tool_calling=True),
                pricing_per_million_tokens=Pricing(input=1.0, output=2.0),
                routes=[
                    ProviderRoute(
                        provider="anthropic",
                        provider_model_id="multi-native",
                        is_default=True,
                    ),
                    ProviderRoute(
                        provider="bedrock",
                        provider_model_id="anthropic.multi-on-bedrock",
                    ),
                    ProviderRoute(
                        provider="vertex",
                        provider_model_id="multi-on-vertex",
                    ),
                ],
                aliases=["multi", "many"],
            ),
            Model(
                name="vertex-only",
                vendor="google",
                tier="balanced",
                context_window=1024,
                max_output_tokens=256,
                modalities=["text"],
                capabilities=Capabilities(tool_calling=True),
                pricing_per_million_tokens=Pricing(input=0.5, output=1.0),
                routes=[
                    ProviderRoute(
                        provider="vertex",
                        provider_model_id="vo-on-vertex",
                        is_default=True,
                    ),
                ],
            ),
        ],
    )


class TestModelRouteDataclass:
    def test_carries_three_fields(self) -> None:
        route = ModelRoute(
            model="some-model",
            provider="anthropic",
            provider_model_id="some-id",
        )
        assert route.model == "some-model"
        assert route.provider == "anthropic"
        assert route.provider_model_id == "some-id"

    def test_is_frozen(self) -> None:
        route = ModelRoute(model="x", provider="anthropic", provider_model_id="x")
        with pytest.raises(FrozenInstanceError):
            route.model = "other"  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        a = ModelRoute(model="x", provider="anthropic", provider_model_id="x")
        b = ModelRoute(model="x", provider="anthropic", provider_model_id="x")
        assert a == b
        assert hash(a) == hash(b)


class TestResolveDefault:
    def test_canonical_name_picks_default_route(self) -> None:
        reg = _make_test_registry()
        route = resolve("multi-route", registry=reg)
        assert route.model == "multi-route"
        assert route.provider == "anthropic"
        assert route.provider_model_id == "multi-native"

    def test_single_route_returns_that_route(self) -> None:
        reg = _make_test_registry()
        route = resolve("single-route", registry=reg)
        assert route.provider == "anthropic"
        assert route.provider_model_id == "single-native"

    def test_alias_resolves_to_canonical_name(self) -> None:
        reg = _make_test_registry()
        route = resolve("multi", registry=reg)
        assert route.model == "multi-route"
        assert route.provider == "anthropic"

    def test_multiple_aliases_for_same_model(self) -> None:
        reg = _make_test_registry()
        assert resolve("multi", registry=reg).model == "multi-route"
        assert resolve("many", registry=reg).model == "multi-route"
        assert resolve("lonely", registry=reg).model == "single-route"


class TestResolveWithProviderPin:
    def test_pinning_existing_route_returns_it(self) -> None:
        reg = _make_test_registry()
        route = resolve("multi-route", provider="bedrock", registry=reg)
        assert route.provider == "bedrock"
        assert route.provider_model_id == "anthropic.multi-on-bedrock"

    def test_pinning_default_explicitly_returns_default(self) -> None:
        reg = _make_test_registry()
        route = resolve("multi-route", provider="anthropic", registry=reg)
        assert route.provider == "anthropic"
        assert route.provider_model_id == "multi-native"

    def test_pinning_with_alias_resolves(self) -> None:
        reg = _make_test_registry()
        route = resolve("multi", provider="vertex", registry=reg)
        assert route.model == "multi-route"
        assert route.provider == "vertex"
        assert route.provider_model_id == "multi-on-vertex"

    def test_unsupported_provider_raises(self) -> None:
        reg = _make_test_registry()
        with pytest.raises(RegistryError) as exc:
            resolve("vertex-only", provider="openai", registry=reg)
        assert exc.value.reason == "unsupported_route"
        assert exc.value.model == "vertex-only"
        assert exc.value.provider == "openai"

    def test_unsupported_provider_for_alias_uses_canonical_name(self) -> None:
        reg = _make_test_registry()
        with pytest.raises(RegistryError) as exc:
            resolve("lonely", provider="bedrock", registry=reg)
        assert exc.value.reason == "unsupported_route"
        # The error reports the canonical model name, not the alias.
        assert exc.value.model == "single-route"


class TestResolveUnknownModel:
    def test_unknown_name_raises(self) -> None:
        reg = _make_test_registry()
        with pytest.raises(RegistryError) as exc:
            resolve("never-heard-of-it", registry=reg)
        assert exc.value.reason == "unknown_model"
        assert exc.value.model == "never-heard-of-it"

    def test_unknown_with_pinned_provider_still_raises_unknown(self) -> None:
        reg = _make_test_registry()
        with pytest.raises(RegistryError) as exc:
            resolve("never-heard-of-it", provider="anthropic", registry=reg)
        assert exc.value.reason == "unknown_model"
        assert exc.value.provider == "anthropic"

    @pytest.mark.parametrize("provider", [None, "anthropic", "openai"])
    def test_unknown_native_id_names_the_registry_file(self, provider: ProviderName | None) -> None:
        # A vendor id the registry does not know fails on every native route; the message
        # must point at the file to change, since it is often all a caller surfaces.
        with pytest.raises(RegistryError) as exc:
            resolve("claude-not-yet-registered", provider=provider)
        assert exc.value.reason == "unknown_model"
        message = str(exc.value)
        assert message.startswith("Unknown model: 'claude-not-yet-registered'")
        assert REGISTRY_DATA_FILE in message
        assert "openai_compat" in message
        assert Path(REGISTRY_DATA_FILE).name == "registry_data.yaml"


# Each vendor's own API, the route its models must resolve on without a pin.
_NATIVE_PROVIDER: dict[str, ProviderName] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "vertex",
}


class TestNativeRouteAllowlist:
    """The registry is the allowlist for vendor-native routes.

    A model a caller reaches on its vendor's own API must resolve there, be priceable
    on the way back, and say truthfully whether it takes tools; otherwise the call fails
    with ``unknown_model`` before (or ``RegistryError`` after) the provider answers.
    """

    @pytest.mark.parametrize("model", global_registry.list_models(), ids=lambda m: m.name)
    def test_every_entry_resolves_on_its_native_provider(self, model: Model) -> None:
        native = _NATIVE_PROVIDER[model.vendor]
        assert model.default_route().provider == native
        pinned = resolve(model.name, provider=native)
        assert pinned.model == model.name
        assert pinned.provider == native
        assert pinned.provider_model_id

    @pytest.mark.parametrize("model", global_registry.list_models(), ids=lambda m: m.name)
    def test_every_entry_is_priced(self, model: Model) -> None:
        usage = Usage(input_tokens=1_000, output_tokens=1_000)
        assert compute_cost(usage, model.name) > 0

    # The vendors' current lineups (Anthropic models overview, OpenAI model pages):
    # (logical name, native provider, provider model id, takes tools on that route).
    CURRENT_LINEUP: tuple[tuple[str, ProviderName, str, bool], ...] = (
        ("claude-fable-5-1", "anthropic", "claude-fable-5-1", True),
        ("claude-opus-5-5", "anthropic", "claude-opus-5-5", True),
        ("claude-sonnet-5-5", "anthropic", "claude-sonnet-5-5", True),
        ("claude-haiku-4-5", "anthropic", "claude-haiku-4-5-20251001", True),
        ("gpt-6-astra", "openai", "gpt-6-astra", True),
        ("gpt-6.1-sol", "openai", "gpt-6.1-sol", False),
        ("gpt-6-luna", "openai", "gpt-6-luna", True),
    )

    @pytest.mark.parametrize(
        ("name", "provider", "provider_model_id", "tools"),
        CURRENT_LINEUP,
        ids=[row[0] for row in CURRENT_LINEUP],
    )
    def test_current_lineup_resolves_natively(
        self,
        name: str,
        provider: ProviderName,
        provider_model_id: str,
        tools: bool,
    ) -> None:
        route = resolve(name, provider=provider)
        assert route.provider_model_id == provider_model_id
        assert global_registry.get(name).capabilities.tool_calling is tools


class TestResolveOpenAICompatBypass:
    """``openai_compat`` routes pass the model id straight through, no registry."""

    def test_unregistered_id_routes_through(self) -> None:
        route = resolve("meta-llama/llama-3.1-8b-instruct:free", provider="openai_compat")
        assert route.model == "meta-llama/llama-3.1-8b-instruct:free"
        assert route.provider == "openai_compat"
        assert route.provider_model_id == "meta-llama/llama-3.1-8b-instruct:free"

    def test_bypass_ignores_registry_entirely(self) -> None:
        # Even with an empty registry (no models), the bypass still resolves.
        reg = Registry([])
        route = resolve("any/model-id", provider="openai_compat", registry=reg)
        assert route.provider == "openai_compat"
        assert route.provider_model_id == "any/model-id"

    def test_preserves_ids_with_slashes_and_tags(self) -> None:
        route = resolve("openai/gpt-oss-20b", provider="openai_compat")
        assert route.provider_model_id == "openai/gpt-oss-20b"

    def test_registered_name_without_pin_is_unaffected(self) -> None:
        # The bypass is keyed on the provider pin only; default resolution of a
        # registered model is unchanged.
        route = resolve("claude-opus-4-7")
        assert route.provider == "anthropic"


class TestResolveAgainstGlobalRegistry:
    """A handful of smoke tests against the real registry singleton."""

    def test_default_route_for_claude_opus(self) -> None:
        route = resolve("claude-opus-4-7")
        assert route.model == "claude-opus-4-7"
        assert route.provider == "anthropic"
        assert route.provider_model_id == "claude-opus-4-7"

    def test_pin_claude_opus_to_bedrock(self) -> None:
        route = resolve("claude-opus-4-7", provider="bedrock")
        assert route.provider == "bedrock"
        assert route.provider_model_id == "anthropic.claude-opus-4-7"

    def test_alias_opus_resolves_to_canonical(self) -> None:
        route = resolve("opus")
        assert route.model == "claude-opus-4-7"

    def test_gemini_pro_default_is_vertex(self) -> None:
        route = resolve("gemini-3.1-pro")
        assert route.provider == "vertex"

    def test_gemini_pinning_to_openai_rejected(self) -> None:
        with pytest.raises(RegistryError) as exc:
            resolve("gemini-3.1-pro", provider="openai")
        assert exc.value.reason == "unsupported_route"
