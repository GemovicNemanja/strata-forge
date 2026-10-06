"""OpenAI-compatible provider — vLLM / TGI / SGLang and similar self-hosted servers.

These servers implement the OpenAI HTTP API surface. LiteLLM routes through
the same ``openai/`` namespace as the native OpenAI provider; the
discriminator is the ``api_base`` kwarg pointing at the user's deployment.

The model id after the ``openai/`` prefix is whatever the self-hosted
server advertises (often the original Hugging Face model name, e.g.
``meta-llama/Llama-3.1-70B-Instruct``). The registry doesn't carry these
because they're operator-specific; callers supply them at the call site.

Tool calling: OpenAI-compatible servers that support function calling use
OpenAI's tool schema verbatim. Callers reuse
:func:`strata_forge.llm.providers.openai.to_openai_tool_schema`.

``base_url`` is caller-trusted. The key and the whole conversation go to whatever URL is
configured, unchecked: no scheme, host, path or address validation, and deliberately no
private-address block, because loopback is the normal case (a model server on the same
machine). A caller that takes ``base_url`` from someone it does not trust validates it first:
exact equality with a URL it allows, never a prefix or host-only match (the OpenAI client
resolves ``..`` segments and sends ``%``-encoded ones raw). A call-site ``api_base`` overrides
``base_url`` and is the same trust surface. Such a caller also passes ``api_key`` explicitly
(:data:`UNAUTHENTICATED_API_KEY` when it has none): left unset, the config reads
``FORGE_OPENAI_COMPAT_API_KEY`` and the client falls back to an ambient ``OPENAI_API_KEY``,
either of which then goes to that URL.

The transport follows redirects. A hop to another host or port drops the ``Authorization``
header, so the configured ``api_key`` stays on the configured host (an ``http`` to ``https``
upgrade on the same host keeps it). Nothing else is dropped: a 307/308 re-sends the request
body, a credential in ``extra_headers`` follows every hop, and the reply (or an error quoting
it) comes back to the caller. Trusting a ``base_url`` therefore includes trusting every
redirect its server issues, on any path the caller lets through.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar

from strata_forge.llm.providers.base import ProviderClient
from strata_forge.llm.providers.config import OpenAICompatConfig

if TYPE_CHECKING:
    from strata_forge.llm.registry import ProviderName

__all__ = ["UNAUTHENTICATED_API_KEY", "OpenAICompatProvider"]

# Stands in for a credential when the server wants none. The OpenAI client refuses to build a
# request without an api_key at all, so "unauthenticated" has to be spelled as a value rather
# than an omission. vLLM's own documentation uses this string for the same reason.
UNAUTHENTICATED_API_KEY = "EMPTY"
# Env vars the OpenAI client reads on its own. If either is set, the caller means it — passing a
# placeholder would override a real credential aimed at an authenticated deployment.
_AMBIENT_KEY_VARS = ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY")


class OpenAICompatProvider(ProviderClient):
    """Provider client for OpenAI-compatible self-hosted inference servers."""

    name: ClassVar[ProviderName] = "openai_compat"
    # Wire format is identical to OpenAI; the discriminator is api_base.
    litellm_prefix: ClassVar[str] = "openai/"

    config: OpenAICompatConfig

    def __init__(self, config: OpenAICompatConfig | None = None) -> None:
        super().__init__(config or OpenAICompatConfig())

    def auth_kwargs(self) -> dict[str, Any]:
        """Return ``api_base`` (required) and an ``api_key``.

        Self-hosted deployments commonly run unauthenticated, so the configured ``api_key`` is
        optional — but OMITTING it does not produce an unauthenticated request. The OpenAI client
        refuses to build a request without a key at all and fails before anything reaches the
        network::

            OpenAIException - Missing credentials. Please pass an `api_key` ... or set the
            OPENAI_API_KEY ... environment variable.

        That is indistinguishable, from the caller's side, from the server rejecting them, and it
        fails EVERY request identically: a batch run against a local vLLM lost all 2098 of its
        rows this way without one of them reaching the server. So "no credential" is sent as a
        placeholder value, which an unauthenticated server ignores.

        ``OPENAI_API_KEY`` / ``OPENAI_ADMIN_KEY`` in the environment are left alone: the client
        reads them itself, and a caller who set one means it for an authenticated deployment.

        The placeholder is scoped to a configured ``base_url``, i.e. to a server this provider is
        actually pointed at. With no base_url there is no self-hosted deployment to be
        unauthenticated against — the call falls through to api.openai.com, where a placeholder
        would turn a plain "no credentials" into a puzzling rejection of one.

        ``api_base`` is the URL of the OpenAI-compatible server's ``/v1`` endpoint and is the only
        thing that distinguishes this provider from the native ``OpenAIProvider``.
        """
        kwargs: dict[str, Any] = {}
        if self.config.base_url is not None:
            kwargs["api_base"] = self.config.base_url
        if self.config.api_key is not None:
            kwargs["api_key"] = self.config.api_key.get_secret_value()
        elif self.config.base_url is not None and not any(
            os.environ.get(var) for var in _AMBIENT_KEY_VARS
        ):
            kwargs["api_key"] = UNAUTHENTICATED_API_KEY
        return kwargs
