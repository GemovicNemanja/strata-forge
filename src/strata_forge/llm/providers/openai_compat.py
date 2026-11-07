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

The ordinary completion transport follows redirects. A hop to another host or port drops the ``Authorization``
header, so the configured ``api_key`` stays on the configured host (an ``http`` to ``https``
upgrade on the same host keeps it). Nothing else is dropped: a 307/308 re-sends the request
body, a credential in ``extra_headers`` follows every hop, and the reply (or an error quoting
it) comes back to the caller. Trusting a ``base_url`` therefore includes trusting every
redirect its server issues, on any path the caller lets through.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from typing import TYPE_CHECKING, Any, ClassVar, cast
from urllib.parse import urlsplit

import httpx

from strata_forge.llm.providers.base import ProviderClient
from strata_forge.llm.providers.config import OpenAICompatConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from openai.types.chat.completion_create_params import ResponseFormat

    from strata_forge.llm.generation import GenerationProfile, GenerationRequest, GenerationResponse
    from strata_forge.llm.registry import ProviderName

__all__ = ["UNAUTHENTICATED_API_KEY", "OpenAICompatProvider"]

# Stands in for a credential when the server wants none. The OpenAI client refuses to build a
# request without an api_key at all, so "unauthenticated" has to be spelled as a value rather
# than an omission. vLLM's own documentation uses this string for the same reason.
UNAUTHENTICATED_API_KEY = "EMPTY"
# Env vars the OpenAI client reads on its own. If either is set, the caller means it — passing a
# placeholder would override a real credential aimed at an authenticated deployment.
_AMBIENT_KEY_VARS = ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY")


@dataclass(frozen=True)
class _ObservedGenerationEndpoint:
    base_url: str
    served_model: str
    model_ref: str
    profile: GenerationProfile
    sdk_version: str
    vllm_version: str


async def _observe_generation_endpoint(
    provider: OpenAICompatProvider, request: GenerationRequest
) -> _ObservedGenerationEndpoint:
    # A configured URL or requested profile cannot prove an owned native process exists.
    from strata_forge.llm.generation import GenerationError

    raise GenerationError("generation_profile_unavailable")


def _matching_generation_endpoint(
    observed: _ObservedGenerationEndpoint, request: GenerationRequest
) -> bool:
    from strata_forge.llm.generation import scalar_text

    if (
        any(
            type(value) is not str
            for value in (
                observed.base_url,
                observed.served_model,
                observed.model_ref,
                observed.profile,
                observed.sdk_version,
                observed.vllm_version,
            )
        )
        or not observed.served_model
        or len(observed.served_model) > 512
        or not scalar_text(observed.served_model)
    ):
        return False
    if len(observed.base_url) > 2048 or any(
        ord(char) <= 32 or ord(char) == 127 for char in observed.base_url
    ):
        return False
    try:
        url = urlsplit(observed.base_url)
        return (
            observed.profile == request.profile
            and observed.model_ref == request.model_ref
            and observed.sdk_version == "2.36.0"
            and observed.vllm_version == "0.24.0"
            and distribution_version("openai") == observed.sdk_version
            and url.scheme == "http"
            and url.hostname is not None
            and ipaddress.ip_address(url.hostname).is_loopback
            and url.username is None
            and url.password is None
            and url.port is not None
            and url.path in ("/v1", "/v1/")
            and not url.query
            and not url.fragment
        )
    except ValueError, TypeError:
        return False


class _GenerationTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.inner = httpx.AsyncHTTPTransport(retries=0, trust_env=False)
        self.refusal_code: str | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        from strata_forge.llm.generation import MAX_GENERATION_RESPONSE_BYTES, GenerationError

        response = await self.inner.handle_async_request(request)
        code: str | None = None
        if 300 <= response.status_code < 400:
            code = "generation_redirect_refused"
        elif not 200 <= response.status_code < 300:
            # The SDK normally drains error bodies before raising. Never hand one to it.
            code = "generation_http_error"
        elif response.headers.get("content-encoding", "identity").lower() != "identity":
            code = "generation_response_encoding_unsupported"
        elif (
            response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/json"
        ):
            code = "generation_response_invalid"
        else:
            advertised = response.headers.get("content-length")
            if advertised is not None and (
                not advertised.isascii()
                or not advertised.isdigit()
                or len(advertised) > 10
                or int(advertised) > MAX_GENERATION_RESPONSE_BYTES
            ):
                code = "generation_response_limit_exceeded"
        if code is not None:
            self.refusal_code = code
            await response.aclose()
            raise GenerationError(code)
        if not isinstance(response.stream, httpx.AsyncByteStream):
            self.refusal_code = "generation_response_invalid"
            await response.aclose()
            raise GenerationError(self.refusal_code)
        response.stream = _GenerationStream(response.stream, self)
        return response

    async def aclose(self) -> None:
        await self.inner.aclose()


class _GenerationStream(httpx.AsyncByteStream):
    def __init__(self, source: httpx.AsyncByteStream, transport: _GenerationTransport) -> None:
        self.source, self.transport = source, transport
        self.count = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        from strata_forge.llm.generation import MAX_GENERATION_RESPONSE_BYTES, GenerationError

        if self.transport.refusal_code is not None:
            raise GenerationError(self.transport.refusal_code)
        async for chunk in self.source:
            if self.transport.refusal_code is not None:
                raise GenerationError(self.transport.refusal_code)
            self.count += len(chunk)
            if self.count > MAX_GENERATION_RESPONSE_BYTES:
                self.transport.refusal_code = "generation_response_limit_exceeded"
                await self.source.aclose()
                raise GenerationError(self.transport.refusal_code)
            yield chunk

    async def aclose(self) -> None:
        await self.source.aclose()


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

    async def agenerate_one(self, user_text: str, request: GenerationRequest) -> GenerationResponse:
        """Use only a fixed observed local endpoint, preserving bounded raw protocol fields."""
        from strata_forge.llm.generation import (
            GenerationError,
            HarmonyChannels,
            JsonSchemaOutput,
            parse_generation_response,
            request_body,
            revalidate_request,
        )

        result: GenerationResponse | None = None
        failure: str | None = None
        transport: _GenerationTransport | None = None
        try:
            current = revalidate_request(request)
            # Freeze text and all controls before an observer may await native-process evidence.
            if type(user_text) is not str:
                raise GenerationError("generation_request_invalid")
            body = request_body(current, user_text=user_text, served_model="pending")
            deadline = asyncio.get_running_loop().time() + current.deadline_seconds
            async with asyncio.timeout_at(deadline):
                observed = await _observe_generation_endpoint(self, current)
            if type(observed) is not _ObservedGenerationEndpoint:
                raise GenerationError("generation_profile_unavailable")
            if not _matching_generation_endpoint(observed, current):
                raise GenerationError("generation_profile_unavailable")
            # SDK2.36 consumes this variable even when default_headers are explicit.
            if any(name == "OPENAI_CUSTOM_HEADERS" for name in os.environ):
                raise GenerationError("generation_sdk_configuration_unsupported")
            body = request_body(current, user_text=user_text, served_model=observed.served_model)
            from openai import AsyncOpenAI, omit

            transport = _GenerationTransport()
            http = httpx.AsyncClient(
                transport=transport,
                trust_env=False,
                follow_redirects=False,
                timeout=httpx.Timeout(current.deadline_seconds),
            )
            try:
                async with asyncio.timeout_at(deadline):
                    async with AsyncOpenAI(
                        api_key=UNAUTHENTICATED_API_KEY,
                        admin_api_key="",
                        organization="",
                        project="",
                        webhook_secret="",
                        base_url=observed.base_url,
                        http_client=http,
                        max_retries=0,
                        timeout=current.deadline_seconds,
                        default_headers={"Accept-Encoding": "identity"},
                    ) as client:
                        response_format = (
                            cast("ResponseFormat", body["response_format"])
                            if isinstance(current.output_format, JsonSchemaOutput)
                            else omit
                        )
                        extras = {
                            key: body[key]
                            for key in ("top_k", "min_p", "chat_template_kwargs")
                            if key in body
                        }
                        async with client.chat.completions.with_streaming_response.create(
                            model=observed.served_model,
                            messages=[{"role": "user", "content": user_text}],
                            temperature=current.temperature,
                            top_p=current.top_p,
                            max_completion_tokens=current.max_tokens,
                            stop=list(current.stop),
                            n=1,
                            stream=False,
                            seed=int(current.effective_seed_decimal)
                            if current.effective_seed_decimal is not None
                            else omit,
                            reasoning_effort=current.channels.effort
                            if isinstance(current.channels, HarmonyChannels)
                            else omit,
                            response_format=response_format,
                            extra_body=extras,
                        ) as response:
                            raw = bytearray()
                            async for chunk in response.iter_bytes():
                                raw.extend(chunk)
                            if transport.refusal_code is not None:
                                raise GenerationError(transport.refusal_code)
                            result = parse_generation_response(
                                current, bytes(raw), served_model=observed.served_model
                            )
            finally:
                await http.aclose()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            safe = {
                "generation_request_invalid",
                "generation_profile_unavailable",
                "generation_sdk_configuration_unsupported",
                "generation_response_invalid",
                "generation_response_limit_exceeded",
                "generation_response_encoding_unsupported",
                "generation_redirect_refused",
                "generation_http_error",
            }
            if transport is not None and transport.refusal_code in safe:
                failure = transport.refusal_code
            elif type(error) is GenerationError and len(error.args) == 1 and error.args[0] in safe:
                failure = error.args[0]
            elif type(error) is TimeoutError:
                failure = "generation_timeout"
            else:
                failure = "generation_transport_failed"
        if failure is not None or result is None:
            raise GenerationError(failure or "generation_transport_failed")
        return result
