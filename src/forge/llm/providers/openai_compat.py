"""OpenAI-compatible provider — vLLM / TGI / SGLang and similar self-hosted servers.

These servers implement the OpenAI HTTP API surface. LiteLLM routes through
the same ``openai/`` namespace as the native OpenAI provider; the
discriminator is the ``api_base`` kwarg pointing at the user's deployment.

The model id after the ``openai/`` prefix is whatever the self-hosted
server advertises (often the original Hugging Face model name, e.g.
``meta-llama/Llama-3.1-70B-Instruct``). The registry doesn't carry these
because they're operator-specific; callers supply them at the call site
once ``forge.compute`` ships in Phase 5.

Tool calling: OpenAI-compatible servers that support function calling use
OpenAI's tool schema verbatim. Callers reuse
:func:`forge.llm.providers.openai.to_openai_tool_schema`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from forge.llm.providers.base import ProviderClient
from forge.llm.providers.config import OpenAICompatConfig

if TYPE_CHECKING:
    from forge.llm.registry import ProviderName

__all__ = ["OpenAICompatProvider"]


class OpenAICompatProvider(ProviderClient):
    """Provider client for OpenAI-compatible self-hosted inference servers."""

    name: ClassVar[ProviderName] = "openai_compat"
    # Wire format is identical to OpenAI; the discriminator is api_base.
    litellm_prefix: ClassVar[str] = "openai/"

    config: OpenAICompatConfig

    def __init__(self, config: OpenAICompatConfig | None = None) -> None:
        super().__init__(config or OpenAICompatConfig())

    def auth_kwargs(self) -> dict[str, Any]:
        """Return ``api_base`` (required) and optional ``api_key``.

        Self-hosted dev deployments often run unauthenticated, so the
        ``api_key`` is optional. ``api_base`` is the URL of the OpenAI-
        compatible server's ``/v1`` endpoint and is the only thing that
        distinguishes this provider from the native ``OpenAIProvider``.
        """
        kwargs: dict[str, Any] = {}
        if self.config.base_url is not None:
            kwargs["api_base"] = self.config.base_url
        if self.config.api_key is not None:
            kwargs["api_key"] = self.config.api_key.get_secret_value()
        return kwargs
