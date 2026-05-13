"""Unit tests for `forge.llm.registry`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError as PydanticValidationError

from forge.core.errors import RegistryError
from forge.llm.registry import (
    Capabilities,
    Model,
    Pricing,
    ProviderRoute,
    Registry,
    _load_registry,  # pyright: ignore[reportPrivateUsage]
)
from forge.llm.registry import (
    registry as global_registry,
)

if TYPE_CHECKING:
    from pathlib import Path


_UNSET: object = object()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_PRICING = Pricing(input=1.0, output=2.0)
_BASE_CAPS = Capabilities(tool_calling=True, structured_output=True)


def _make_model(
    name: str = "test-model",
    *,
    vendor: str = "anthropic",
    routes: list[ProviderRoute] | object = _UNSET,
    aliases: list[str] | None = None,
) -> Model:
    if routes is _UNSET:
        routes = [
            ProviderRoute(provider="anthropic", provider_model_id=name, is_default=True),
        ]
    return Model(
        name=name,
        vendor=vendor,  # type: ignore[arg-type]
        tier="flagship",
        context_window=100_000,
        max_output_tokens=8_000,
        modalities=["text"],
        capabilities=_BASE_CAPS,
        pricing_per_million_tokens=_BASE_PRICING,
        routes=routes,  # type: ignore[arg-type]
        aliases=aliases or [],
    )


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


class TestModelValidation:
    def test_minimal_valid_model(self) -> None:
        m = _make_model()
        assert m.name == "test-model"
        assert m.default_route().provider == "anthropic"

    def test_no_routes_rejected(self) -> None:
        with pytest.raises(ValueError, match="no provider routes"):
            _make_model(routes=[])

    def test_no_default_route_rejected(self) -> None:
        with pytest.raises(ValueError, match="no default route"):
            _make_model(
                routes=[
                    ProviderRoute(provider="anthropic", provider_model_id="x", is_default=False),
                ],
            )

    def test_multiple_default_routes_rejected(self) -> None:
        with pytest.raises(ValueError, match="multiple default routes"):
            _make_model(
                routes=[
                    ProviderRoute(provider="anthropic", provider_model_id="x", is_default=True),
                    ProviderRoute(provider="bedrock", provider_model_id="y", is_default=True),
                ],
            )

    def test_duplicate_provider_routes_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate provider routes"):
            _make_model(
                routes=[
                    ProviderRoute(provider="anthropic", provider_model_id="x", is_default=True),
                    ProviderRoute(provider="anthropic", provider_model_id="y"),
                ],
            )

    def test_negative_pricing_rejected(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            Pricing(input=-1.0, output=2.0)

    def test_zero_context_window_rejected(self) -> None:
        # model_copy doesn't re-run validators; construct directly to test
        # field-level validation.
        with pytest.raises(PydanticValidationError, match="greater than 0"):
            Model(
                name="x",
                vendor="anthropic",
                tier="flagship",
                context_window=0,
                max_output_tokens=8_000,
                modalities=["text"],
                capabilities=_BASE_CAPS,
                pricing_per_million_tokens=_BASE_PRICING,
                routes=[
                    ProviderRoute(provider="anthropic", provider_model_id="x", is_default=True),
                ],
            )

    def test_frozen_models_reject_mutation(self) -> None:
        m = _make_model()
        with pytest.raises(PydanticValidationError):
            m.name = "different"  # type: ignore[misc]

    def test_route_for_returns_match(self) -> None:
        m = _make_model(
            routes=[
                ProviderRoute(provider="anthropic", provider_model_id="x", is_default=True),
                ProviderRoute(provider="bedrock", provider_model_id="y"),
            ],
        )
        bedrock = m.route_for("bedrock")
        assert bedrock is not None
        assert bedrock.provider_model_id == "y"

    def test_route_for_returns_none_when_absent(self) -> None:
        m = _make_model()
        assert m.route_for("azure") is None


# ---------------------------------------------------------------------------
# Registry assembly + validation
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_empty_registry(self) -> None:
        reg = Registry([])
        assert len(reg) == 0
        assert "claude-opus-4-7" not in reg

    def test_get_by_canonical_name(self) -> None:
        reg = Registry([_make_model("model-a")])
        assert reg.get("model-a").name == "model-a"

    def test_get_by_alias(self) -> None:
        reg = Registry([_make_model("model-a", aliases=["a", "alpha"])])
        assert reg.get("a").name == "model-a"
        assert reg.get("alpha").name == "model-a"

    def test_get_unknown_raises_registry_error(self) -> None:
        reg = Registry([_make_model("model-a")])
        with pytest.raises(RegistryError) as exc:
            reg.get("nope")
        assert exc.value.reason == "unknown_model"
        assert exc.value.model == "nope"

    def test_duplicate_names_rejected(self) -> None:
        with pytest.raises(RegistryError) as exc:
            Registry([_make_model("dup"), _make_model("dup")])
        assert exc.value.reason == "duplicate_name"

    def test_alias_conflicts_with_name(self) -> None:
        with pytest.raises(RegistryError) as exc:
            Registry(
                [
                    _make_model("model-a"),
                    _make_model("model-b", aliases=["model-a"]),
                ],
            )
        assert exc.value.reason == "alias_conflict"

    def test_alias_duplicated_across_models(self) -> None:
        with pytest.raises(RegistryError) as exc:
            Registry(
                [
                    _make_model("model-a", aliases=["shared"]),
                    _make_model("model-b", aliases=["shared"]),
                ],
            )
        assert exc.value.reason == "alias_conflict"

    def test_list_models_returns_all(self) -> None:
        reg = Registry([_make_model("a"), _make_model("b"), _make_model("c")])
        assert {m.name for m in reg.list_models()} == {"a", "b", "c"}

    def test_list_models_filtered_by_vendor(self) -> None:
        reg = Registry(
            [
                _make_model("anth-1", vendor="anthropic"),
                _make_model("oai-1", vendor="openai"),
                _make_model("g-1", vendor="google"),
            ],
        )
        names = {m.name for m in reg.list_models(vendor="anthropic")}
        assert names == {"anth-1"}

    def test_contains_canonical_and_alias(self) -> None:
        reg = Registry([_make_model("m", aliases=["alias-one"])])
        assert "m" in reg
        assert "alias-one" in reg
        assert "missing" not in reg

    def test_contains_returns_false_for_non_string(self) -> None:
        reg = Registry([_make_model("m")])
        assert (None in reg) is False  # type: ignore[operator]


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


class TestLoader:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RegistryError) as exc:
            _load_registry(tmp_path / "does-not-exist.yaml")
        assert exc.value.reason == "missing_data_file"

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("not: [closed\n")
        with pytest.raises(RegistryError) as exc:
            _load_registry(path)
        assert exc.value.reason == "parse_error"

    def test_missing_models_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "no-key.yaml"
        path.write_text("something: else\n")
        with pytest.raises(RegistryError) as exc:
            _load_registry(path)
        assert exc.value.reason == "bad_schema"

    def test_non_list_models_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "not-list.yaml"
        path.write_text("models: not-a-list\n")
        with pytest.raises(RegistryError) as exc:
            _load_registry(path)
        assert exc.value.reason == "bad_schema"

    def test_invalid_entry_raises_validation_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad-entry.yaml"
        path.write_text(
            "models:\n  - name: x\n    vendor: anthropic\n    # missing required fields\n",
        )
        with pytest.raises(RegistryError) as exc:
            _load_registry(path)
        assert exc.value.reason == "validation_error"

    def test_valid_minimal_yaml_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "good.yaml"
        path.write_text(
            "models:\n"
            "  - name: tiny\n"
            "    vendor: anthropic\n"
            "    tier: fast\n"
            "    context_window: 1024\n"
            "    max_output_tokens: 256\n"
            "    modalities: [text]\n"
            "    capabilities:\n"
            "      tool_calling: true\n"
            "    pricing_per_million_tokens:\n"
            "      input: 0.1\n"
            "      output: 0.2\n"
            "    routes:\n"
            "      - provider: anthropic\n"
            "        provider_model_id: tiny\n"
            "        is_default: true\n",
        )
        reg = _load_registry(path)
        assert "tiny" in reg
        assert reg.get("tiny").context_window == 1024


# ---------------------------------------------------------------------------
# Global registry (loaded from registry_data.yaml at import time)
# ---------------------------------------------------------------------------


class TestGlobalRegistry:
    EXPECTED_MODEL_NAMES: frozenset[str] = frozenset(
        {
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "gpt-5.5",
            "gpt-5.5-pro",
            "gpt-5.5-thinking",
            "gpt-5.5-instant",
            "gemini-3.1-pro",
            "gemini-3.1-flash-lite",
        },
    )

    def test_loads_expected_count(self) -> None:
        assert len(global_registry) == len(self.EXPECTED_MODEL_NAMES)

    def test_all_expected_models_present(self) -> None:
        actual = {m.name for m in global_registry.list_models()}
        assert actual == self.EXPECTED_MODEL_NAMES

    def test_anthropic_models_have_three_routes(self) -> None:
        for name in ("claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"):
            model = global_registry.get(name)
            providers = {r.provider for r in model.routes}
            assert providers == {"anthropic", "bedrock", "vertex"}

    def test_openai_models_have_two_routes(self) -> None:
        for name in ("gpt-5.5", "gpt-5.5-pro", "gpt-5.5-thinking", "gpt-5.5-instant"):
            model = global_registry.get(name)
            providers = {r.provider for r in model.routes}
            assert providers == {"openai", "azure"}

    def test_gemini_models_only_on_vertex(self) -> None:
        for name in ("gemini-3.1-pro", "gemini-3.1-flash-lite"):
            model = global_registry.get(name)
            providers = {r.provider for r in model.routes}
            assert providers == {"vertex"}

    def test_every_model_has_exactly_one_default_route(self) -> None:
        for model in global_registry.list_models():
            defaults = [r for r in model.routes if r.is_default]
            assert len(defaults) == 1, f"{model.name} has {len(defaults)} default routes"

    def test_every_model_supports_tool_calling(self) -> None:
        # Initial registry: all 10 foundation models support tool calling.
        for model in global_registry.list_models():
            assert model.capabilities.tool_calling, f"{model.name} missing tool_calling"

    def test_pricing_is_non_negative(self) -> None:
        for model in global_registry.list_models():
            p = model.pricing_per_million_tokens
            assert p.input >= 0
            assert p.output >= 0
            if p.cache_read is not None:
                assert p.cache_read >= 0
            if p.cache_write is not None:
                assert p.cache_write >= 0

    @pytest.mark.parametrize(
        ("alias", "expected_canonical"),
        [
            ("opus", "claude-opus-4-7"),
            ("sonnet", "claude-sonnet-4-6"),
            ("haiku", "claude-haiku-4-5"),
            ("gpt55", "gpt-5.5"),
            ("gemini-pro", "gemini-3.1-pro"),
            ("gemini-flash-lite", "gemini-3.1-flash-lite"),
        ],
    )
    def test_aliases_resolve(self, alias: str, expected_canonical: str) -> None:
        assert global_registry.get(alias).name == expected_canonical
