"""Vertex AI provider client.

Serves both Gemini (Google's own models) and Anthropic-on-Vertex (Claude
hosted on Google's infrastructure). LiteLLM uses different namespace
prefixes for the two cases:

- ``vertex_ai/<gemini-id>`` for Gemini models
- ``anthropic_vertex/<claude-id>`` for Claude models on Vertex

This class overrides :meth:`litellm_model` to pick the right prefix based
on the model id. Anything starting with ``claude`` routes through the
Anthropic-on-Vertex namespace; everything else uses ``vertex_ai``.

Credentials follow Google's Application Default Credentials chain — set
``GOOGLE_APPLICATION_CREDENTIALS`` to a service-account JSON path, and
configure project + region via ``GCP_PROJECT`` / ``GCP_REGION``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from forge.llm.providers.base import ProviderClient
from forge.llm.providers.config import VertexConfig

if TYPE_CHECKING:
    from forge.llm.registry import ProviderName

__all__ = ["VertexProvider", "to_gemini_tool_schema"]


_CLAUDE_ON_VERTEX_PREFIX = "anthropic_vertex/"


class VertexProvider(ProviderClient):
    """Provider client for Google Vertex AI.

    Routes Gemini models through ``vertex_ai/*`` and Anthropic-on-Vertex
    models through ``anthropic_vertex/*``.
    """

    name: ClassVar[ProviderName] = "vertex"
    litellm_prefix: ClassVar[str] = "vertex_ai/"

    config: VertexConfig

    def __init__(self, config: VertexConfig | None = None) -> None:
        super().__init__(config or VertexConfig())

    def litellm_model(self, provider_model_id: str) -> str:
        """Pick the LiteLLM namespace based on the model family.

        Claude models on Vertex use a different LiteLLM namespace
        (``anthropic_vertex/``) than Gemini models (``vertex_ai/``).
        """
        if provider_model_id.startswith("claude"):
            return f"{_CLAUDE_ON_VERTEX_PREFIX}{provider_model_id}"
        return f"{self.litellm_prefix}{provider_model_id}"

    def auth_kwargs(self) -> dict[str, Any]:
        """Return LiteLLM-recognized auth kwargs.

        LiteLLM accepts ``vertex_project`` / ``vertex_location`` /
        ``vertex_credentials`` for Vertex calls. When unset, LiteLLM falls
        back to Google's Application Default Credentials chain.
        """
        kwargs: dict[str, Any] = {}
        if self.config.project is not None:
            kwargs["vertex_project"] = self.config.project
        if self.config.region:
            kwargs["vertex_location"] = self.config.region
        if self.config.application_credentials is not None:
            kwargs["vertex_credentials"] = self.config.application_credentials
        return kwargs


def to_gemini_tool_schema(
    name: str,
    description: str,
    parameters_schema: dict[str, Any],
) -> dict[str, Any]:
    """Convert a tool definition into Gemini's function-declaration shape.

    Gemini wraps tools in ``{"function_declarations": [<decl>, ...]}`` at
    the call site; this function returns a single declaration. Build the
    outer wrapper when assembling the actual API call.

    Args:
        name: Tool name. Must be unique within the call's tool list.
        description: Human-readable description shown to the model.
        parameters_schema: JSON Schema for the tool's argument object.

    Returns:
        Gemini's function declaration:

        .. code-block:: python

            {
                "name": ...,
                "description": ...,
                "parameters": ...,
            }
    """
    return {
        "name": name,
        "description": description,
        "parameters": parameters_schema,
    }
