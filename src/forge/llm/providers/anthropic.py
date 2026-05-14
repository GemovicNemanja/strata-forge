"""Anthropic provider client.

Routes through LiteLLM's ``anthropic/*`` namespace. Reads
``ANTHROPIC_API_KEY`` from the environment via :class:`AnthropicConfig`.

The module also exposes :func:`to_anthropic_tool_schema` — Anthropic uses a
flatter shape than OpenAI: ``{"name", "description", "input_schema"}``
without an outer ``type: "function"`` wrapper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from forge.llm.providers.base import ProviderClient
from forge.llm.providers.config import AnthropicConfig

if TYPE_CHECKING:
    from forge.llm.registry import ProviderName

__all__ = ["AnthropicProvider", "to_anthropic_tool_schema"]


class AnthropicProvider(ProviderClient):
    """Provider client for Anthropic's native API."""

    name: ClassVar[ProviderName] = "anthropic"
    litellm_prefix: ClassVar[str] = "anthropic/"

    config: AnthropicConfig

    def __init__(self, config: AnthropicConfig | None = None) -> None:
        super().__init__(config or AnthropicConfig())

    def auth_kwargs(self) -> dict[str, Any]:
        """Return LiteLLM-recognized auth kwargs from the active config.

        Returns an empty dict when no key is configured so LiteLLM can fall
        back to its own ``ANTHROPIC_API_KEY`` env lookup.
        """
        kwargs: dict[str, Any] = {}
        if self.config.api_key is not None:
            kwargs["api_key"] = self.config.api_key.get_secret_value()
        return kwargs


def to_anthropic_tool_schema(
    name: str,
    description: str,
    parameters_schema: dict[str, Any],
) -> dict[str, Any]:
    """Convert a tool definition into Anthropic's tool-use schema.

    Anthropic's shape is flatter than OpenAI's — no outer ``type: function``
    wrapper, and the parameter schema lives under ``input_schema`` rather
    than ``parameters``.

    Args:
        name: Tool name. Must be unique within the call's tool list.
        description: Human-readable description shown to the model.
        parameters_schema: JSON Schema for the tool's argument object.
            Typically the output of ``pydantic_model.model_json_schema()``.

    Returns:
        Anthropic's tool descriptor:

        .. code-block:: python

            {
                "name": ...,
                "description": ...,
                "input_schema": ...,
            }
    """
    return {
        "name": name,
        "description": description,
        "input_schema": parameters_schema,
    }
