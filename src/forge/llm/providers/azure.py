"""Azure OpenAI provider client.

Routes through LiteLLM's ``azure/*`` namespace. The model string after the
``azure/`` prefix is the *deployment name* — what the user gave their
Azure deployment in the portal — which may or may not match the
underlying OpenAI model id. Our registry stores the OpenAI model id as
the provider_model_id for Azure routes; users whose deployment names
differ should either rename their deployment or pass an override at the
call site.

Tool calling on Azure uses OpenAI's exact format (it's a wire-compatible
re-host), so callers reuse
:func:`forge.llm.providers.openai.to_openai_tool_schema`. This module
therefore exposes only the provider client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from forge.llm.providers.base import ProviderClient
from forge.llm.providers.config import AzureConfig

if TYPE_CHECKING:
    from forge.llm.registry import ProviderName

__all__ = ["AzureProvider"]


class AzureProvider(ProviderClient):
    """Provider client for Azure OpenAI."""

    name: ClassVar[ProviderName] = "azure"
    litellm_prefix: ClassVar[str] = "azure/"

    config: AzureConfig

    def __init__(self, config: AzureConfig | None = None) -> None:
        super().__init__(config or AzureConfig())

    def auth_kwargs(self) -> dict[str, Any]:
        """Return LiteLLM-recognized Azure kwargs.

        LiteLLM accepts ``api_key`` / ``api_base`` (the Azure endpoint) /
        ``api_version`` for Azure calls. ``api_version`` is always emitted
        because the config defaults it to the latest GA-stable string;
        ``api_key`` and ``api_base`` are dropped when unset so LiteLLM can
        fall back to its own env lookup.
        """
        kwargs: dict[str, Any] = {}
        if self.config.api_key is not None:
            kwargs["api_key"] = self.config.api_key.get_secret_value()
        if self.config.endpoint is not None:
            kwargs["api_base"] = self.config.endpoint
        if self.config.api_version:
            kwargs["api_version"] = self.config.api_version
        return kwargs
