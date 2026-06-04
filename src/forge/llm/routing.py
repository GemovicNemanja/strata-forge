"""Route resolution: logical model + optional provider → concrete dispatch target.

A :class:`ModelRoute` is the immutable triple of (logical model name, provider,
provider_model_id) that an LLM call actually targets. The :func:`resolve`
function turns a user-supplied ``(model, provider=None)`` pair into one of
those routes by consulting the registry — performing alias resolution and
provider pinning along the way.

This module is the only seam that LLM consumers should call to translate
"the user asked for Claude Opus via Bedrock" into "send the request to
``anthropic.claude-opus-4-7`` on the ``bedrock`` provider client."
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.core.errors import RegistryError
from forge.llm.registry import ProviderName, Registry
from forge.llm.registry import registry as _global_registry

__all__ = ["ModelRoute", "resolve"]


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """A resolved dispatch target.

    Attributes:
        model: The *canonical* logical model name (after alias resolution).
        provider: The provider that will serve the call.
        provider_model_id: The model identifier that the provider expects.
    """

    model: str
    provider: ProviderName
    provider_model_id: str


def resolve(
    model: str,
    provider: ProviderName | None = None,
    *,
    registry: Registry | None = None,
) -> ModelRoute:
    """Resolve a ``(model, provider)`` pair into a concrete :class:`ModelRoute`.

    Args:
        model: A logical Forge model name *or* a registered alias for one.
        provider: If supplied, pin the call to this specific provider; the
            ``(model, provider)`` combo must be a registered route. If
            ``None``, the model's default route is used.
        registry: Override the global registry. Tests use this to drive
            resolution against a fixture registry without touching the
            process-wide singleton.

    Returns:
        A :class:`ModelRoute` whose ``model`` field is the canonical name
        (aliases are already resolved) and whose ``provider`` /
        ``provider_model_id`` reflect the chosen route.

    Raises:
        RegistryError: ``model`` (or its alias) is not in the registry, or
            the requested ``provider`` is not a supported route for it. The
            error's ``reason`` tag is ``unknown_model`` or
            ``unsupported_route`` respectively. The ``openai_compat`` provider
            is exempt: its model ids are operator-specific and intentionally
            absent from the curated registry.
    """
    # OpenAI-compatible endpoints (vLLM / TGI / SGLang, but also OpenRouter,
    # Groq, the Gemini OpenAI-compat endpoint, local Ollama, ...) carry
    # operator-specific model ids that the curated registry does not — and by
    # design should not — track. When the caller pins ``openai_compat`` they
    # supply the provider's own model id directly, so route it through as-is
    # rather than consulting the registry (which would raise ``unknown_model``).
    if provider == "openai_compat":
        return ModelRoute(
            model=model,
            provider="openai_compat",
            provider_model_id=model,
        )

    reg = registry if registry is not None else _global_registry
    entry = reg.get(model)  # raises RegistryError(reason="unknown_model")

    if provider is None:
        provider_route = entry.default_route()
    else:
        provider_route = entry.route_for(provider)
        if provider_route is None:
            raise RegistryError(
                f"Model {entry.name!r} has no route for provider {provider!r}",
                model=entry.name,
                provider=provider,
                reason="unsupported_route",
            )

    return ModelRoute(
        model=entry.name,
        provider=provider_route.provider,
        provider_model_id=provider_route.provider_model_id,
    )
