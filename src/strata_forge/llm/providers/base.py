"""``ProviderClient`` — abstract base over LiteLLM for every provider.

Concrete subclasses (``openai.py``, ``anthropic.py``, ``vertex.py``,
``bedrock.py``, ``azure.py``, ``openai_compat.py``) declare:

- ``name``: the Forge provider identifier (matches the registry's ``provider`` field)
- ``litellm_prefix``: the model-string prefix LiteLLM expects (e.g. ``"anthropic/"``)
- ``auth_kwargs()``: provider-specific kwargs (api keys, region, ...) merged
  into every LiteLLM call

The base provides concrete ``acompletion`` and ``astream`` implementations
on top of ``litellm.acompletion``, and ``aresponses`` / ``aresponses_stream``
on top of ``litellm.aresponses`` for the routes that speak OpenAI's Responses
API. Provider subclasses rarely need to override those; when they do (e.g. for
unusual streaming protocols), they should call the base via ``super()`` after
fixing up their inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, cast

import litellm

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from strata_forge.llm.providers.config import ProviderConfig
    from strata_forge.llm.registry import ProviderName

__all__ = ["ProviderClient"]


class ProviderClient(ABC):
    """Abstract base for LLM provider clients.

    Subclasses configure how LiteLLM should reach the underlying provider
    (model-string prefix + auth kwargs); the base class owns the actual
    call flow.
    """

    name: ClassVar[ProviderName]
    litellm_prefix: ClassVar[str]

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    def auth_kwargs(self) -> dict[str, Any]:
        """Return the kwargs to merge into every LiteLLM call.

        Most providers return their API key under the LiteLLM-recognized
        keyword (``api_key``, ``aws_region_name``, ``vertex_project``,
        ``api_base``, ...). Return an empty dict to let LiteLLM read its
        creds from the environment.
        """

    def responses_auth_kwargs(self) -> dict[str, Any]:
        """Return the kwargs to merge into every Responses API call.

        Defaults to :meth:`auth_kwargs`; a provider whose Responses endpoint needs different
        connection settings (Azure's API version) overrides it.
        """
        return self.auth_kwargs()

    def litellm_model(self, provider_model_id: str) -> str:
        """Format the model string LiteLLM expects for this provider.

        Example: ``"anthropic/claude-opus-4-7"`` from prefix
        ``"anthropic/"`` and id ``"claude-opus-4-7"``.
        """
        return f"{self.litellm_prefix}{provider_model_id}"

    async def acompletion(
        self,
        *,
        provider_model_id: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Single non-streaming completion via LiteLLM.

        Returns the raw LiteLLM response. Callers normalize it into a
        ``strata_forge.llm.responses.LLMResponse`` upstream of this seam.
        """
        merged = {**self.auth_kwargs(), **kwargs}
        # LiteLLM's acompletion is typed as returning ModelResponse but with
        # streaming the actual return is an async iterator of chunks; the
        # base intentionally returns the raw value so callers can normalize.
        return await litellm.acompletion(  # pyright: ignore[reportUnknownMemberType]
            model=self.litellm_model(provider_model_id),
            messages=messages,
            **merged,
        )

    async def astream(
        self,
        *,
        provider_model_id: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Streaming completion via LiteLLM.

        Yields raw LiteLLM chunks; the streaming utilities upstream of this
        seam convert them into ``ResponseChunk`` instances.
        """
        merged = {**self.auth_kwargs(), **kwargs}
        # See the note in acompletion: with stream=True the return is an
        # async iterator that LiteLLM's stubs don't model precisely. Cast
        # to AsyncIterator[Any] so the `async for` below is well-typed.
        raw = await litellm.acompletion(  # pyright: ignore[reportUnknownMemberType]
            model=self.litellm_model(provider_model_id),
            messages=messages,
            stream=True,
            **merged,
        )
        response = cast("AsyncIterator[Any]", raw)
        async for chunk in response:
            yield chunk

    async def aresponses(
        self,
        *,
        provider_model_id: str,
        request: dict[str, Any],
    ) -> Any:
        """One non-streaming Responses API call via ``litellm.aresponses``.

        ``request`` is the body :func:`strata_forge.llm.responses_wire.build_request` builds;
        LiteLLM forwards it to the provider unchanged. Returns the raw response object.
        """
        params = _merge_request(self.responses_auth_kwargs(), request)
        return await litellm.aresponses(  # pyright: ignore[reportUnknownMemberType]
            model=self.litellm_model(provider_model_id),
            **params,
        )

    async def aresponses_stream(
        self,
        *,
        provider_model_id: str,
        request: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """A streaming Responses API call via ``litellm.aresponses``; yields the typed events."""
        params = _merge_request(self.responses_auth_kwargs(), {**request, "stream": True})
        raw = await litellm.aresponses(  # pyright: ignore[reportUnknownMemberType]
            model=self.litellm_model(provider_model_id),
            **params,
        )
        events = cast("AsyncIterator[Any]", raw)
        async for event in events:
            yield event


def _merge_request(auth: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``request`` on ``auth``; ``extra_headers`` from both are combined, request winning."""
    merged: dict[str, Any] = {**auth, **request}
    auth_headers = auth.get("extra_headers")
    request_headers = request.get("extra_headers")
    if isinstance(auth_headers, dict) and isinstance(request_headers, dict):
        merged["extra_headers"] = {
            **cast("dict[str, Any]", auth_headers),
            **cast("dict[str, Any]", request_headers),
        }
    return merged
