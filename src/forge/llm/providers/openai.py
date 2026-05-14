"""OpenAI provider client.

Routes through LiteLLM's ``openai/*`` namespace. Reads
``OPENAI_API_KEY`` and optional ``OPENAI_ORG_ID`` from the environment
via :class:`OpenAIConfig`.

The module also exposes :func:`to_openai_tool_schema` — the canonical
shape for OpenAI's tool-calling API. The full ``Tool`` abstraction lives
in ``forge.llm.tools`` (lands later); this function operates on a raw
JSON-Schema dict so it can be reused for either provider native calls
or LiteLLM-mediated ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from forge.llm.providers.base import ProviderClient
from forge.llm.providers.config import OpenAIConfig

if TYPE_CHECKING:
    from forge.llm.registry import ProviderName

__all__ = ["OpenAIProvider", "to_openai_tool_schema"]


class OpenAIProvider(ProviderClient):
    """Provider client for OpenAI's native API."""

    name: ClassVar[ProviderName] = "openai"
    litellm_prefix: ClassVar[str] = "openai/"

    config: OpenAIConfig

    def __init__(self, config: OpenAIConfig | None = None) -> None:
        super().__init__(config or OpenAIConfig())

    def auth_kwargs(self) -> dict[str, Any]:
        """Return LiteLLM-recognized auth kwargs from the active config.

        If no ``OPENAI_API_KEY`` is configured, returns an empty dict so
        LiteLLM falls back to its own environment lookup.
        """
        kwargs: dict[str, Any] = {}
        if self.config.api_key is not None:
            kwargs["api_key"] = self.config.api_key.get_secret_value()
        if self.config.org_id is not None:
            # LiteLLM forwards "organization" to the OpenAI SDK.
            kwargs["organization"] = self.config.org_id
        return kwargs


def to_openai_tool_schema(
    name: str,
    description: str,
    parameters_schema: dict[str, Any],
) -> dict[str, Any]:
    """Convert a tool definition into OpenAI's tool-call schema.

    Args:
        name: Tool name. Must be unique within the call's tool list.
        description: Human-readable description shown to the model.
        parameters_schema: JSON Schema for the tool's argument object.
            Typically the output of ``pydantic_model.model_json_schema()``.

    Returns:
        OpenAI's standard tool descriptor:

        .. code-block:: python

            {
                "type": "function",
                "function": {
                    "name": ...,
                    "description": ...,
                    "parameters": ...,
                },
            }
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters_schema,
        },
    }
