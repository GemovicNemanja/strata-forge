"""Model registry — the canonical source of truth for what Forge can call.

The registry is loaded once from ``registry_data.yaml`` at import time. The
module-level ``registry`` singleton is the only access point most callers
need; tests construct their own ``Registry`` instances for isolation.

Per `ADR 0004 <../../docs/architecture/adr/0004-model-registry-scope.md>`,
the initial registry is scoped to the latest foundation models from
Anthropic, OpenAI, and Google. New vendors require a superseding ADR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from forge.core.errors import RegistryError

__all__ = [
    "Capabilities",
    "Modality",
    "Model",
    "Pricing",
    "ProviderName",
    "ProviderRoute",
    "Registry",
    "Tier",
    "Vendor",
    "registry",
]


Vendor = Literal["anthropic", "openai", "google"]
Tier = Literal["flagship", "reasoning", "balanced", "fast"]
Modality = Literal["text", "image", "audio"]
ProviderName = Literal[
    "anthropic",
    "openai",
    "vertex",
    "bedrock",
    "azure",
    "openai_compat",
]


class ProviderRoute(BaseModel):
    """A concrete ``(provider, provider_model_id)`` dispatch target for a model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderName
    provider_model_id: str
    is_default: bool = False


class Pricing(BaseModel):
    """USD pricing per million tokens for a model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float | None = Field(default=None, ge=0)
    cache_write: float | None = Field(default=None, ge=0)


class Capabilities(BaseModel):
    """Feature flags for what a model supports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_calling: bool = False
    structured_output: bool = False
    streaming: bool = True
    vision: bool = False
    prompt_caching: bool = False


class Model(BaseModel):
    """A logical Forge model with one or more provider routes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    vendor: Vendor
    tier: Tier
    context_window: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    modalities: list[Modality]
    capabilities: Capabilities
    pricing_per_million_tokens: Pricing
    routes: list[ProviderRoute]
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_routes(self) -> Model:
        if not self.routes:
            msg = f"Model {self.name!r} has no provider routes"
            raise ValueError(msg)
        default_count = sum(1 for r in self.routes if r.is_default)
        if default_count == 0:
            msg = f"Model {self.name!r} has no default route"
            raise ValueError(msg)
        if default_count > 1:
            msg = f"Model {self.name!r} has multiple default routes"
            raise ValueError(msg)
        providers = [r.provider for r in self.routes]
        if len(providers) != len(set(providers)):
            msg = f"Model {self.name!r} has duplicate provider routes"
            raise ValueError(msg)
        return self

    def default_route(self) -> ProviderRoute:
        """Return the default route — guaranteed to exist by the validator."""
        for route in self.routes:
            if route.is_default:
                return route
        msg = "unreachable: validator guarantees exactly one default route"
        raise AssertionError(msg)

    def route_for(self, provider: ProviderName) -> ProviderRoute | None:
        """Return the route for ``provider``, or ``None`` if not supported."""
        for route in self.routes:
            if route.provider == provider:
                return route
        return None


class Registry:
    """In-memory catalog of logical models and their provider routes."""

    def __init__(self, models: list[Model]) -> None:
        names = [m.name for m in models]
        duplicates = [name for name in set(names) if names.count(name) > 1]
        if duplicates:
            raise RegistryError(
                f"Duplicate model names in registry: {duplicates}",
                reason="duplicate_name",
            )

        all_aliases: dict[str, str] = {}
        name_set = set(names)
        for model in models:
            for alias in model.aliases:
                if alias in name_set:
                    raise RegistryError(
                        f"Alias {alias!r} conflicts with a model name",
                        model=model.name,
                        reason="alias_conflict",
                    )
                if alias in all_aliases:
                    raise RegistryError(
                        f"Alias {alias!r} is claimed by multiple models",
                        reason="alias_conflict",
                    )
                all_aliases[alias] = model.name

        self._models: dict[str, Model] = {m.name: m for m in models}
        self._aliases: dict[str, str] = all_aliases

    def get(self, name: str) -> Model:
        """Look up a model by canonical name or alias. Raises ``RegistryError``."""
        canonical = self._aliases.get(name, name)
        model = self._models.get(canonical)
        if model is None:
            raise RegistryError(
                f"Unknown model: {name!r}",
                model=name,
                reason="unknown_model",
            )
        return model

    def list_models(self, *, vendor: Vendor | None = None) -> list[Model]:
        """Return all registered models, optionally filtered by vendor."""
        models = list(self._models.values())
        if vendor is not None:
            models = [m for m in models if m.vendor == vendor]
        return models

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        canonical = self._aliases.get(name, name)
        return canonical in self._models

    def __len__(self) -> int:
        return len(self._models)


# ---------------------------------------------------------------------------
# Module-level singleton loaded from registry_data.yaml at import time.
# ---------------------------------------------------------------------------

_REGISTRY_DATA_PATH = Path(__file__).parent / "registry_data.yaml"


def _load_registry(path: Path = _REGISTRY_DATA_PATH) -> Registry:
    """Load and validate the registry from a YAML file."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise RegistryError(
            f"Registry data file not found: {path}",
            reason="missing_data_file",
        ) from exc
    except yaml.YAMLError as exc:
        raise RegistryError(
            f"Failed to parse registry data: {exc}",
            reason="parse_error",
        ) from exc

    if not isinstance(data, dict) or "models" not in data:
        raise RegistryError(
            "Registry data must be a YAML mapping with a top-level 'models' key",
            reason="bad_schema",
        )

    raw_items = cast("Any", data["models"])
    if not isinstance(raw_items, list):
        raise RegistryError(
            "Registry 'models' field must be a list",
            reason="bad_schema",
        )
    items: list[Any] = cast("list[Any]", raw_items)

    try:
        models = [Model.model_validate(item) for item in items]
    except ValidationError as exc:
        raise RegistryError(
            f"Registry validation failed: {exc}",
            reason="validation_error",
        ) from exc

    return Registry(models)


registry = _load_registry()
