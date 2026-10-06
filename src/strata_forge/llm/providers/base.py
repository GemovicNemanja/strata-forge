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

``describe_to_litellm`` gives LiteLLM a registered model its own model map lacks.
LiteLLM fakes a Responses stream (one blocking call, replayed as deltas) for any
model it cannot look up, and its bundled map trails the vendors' lineups, so
without it whether a new model streamed would depend on LiteLLM's import-time
fetch of its remote map.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, cast

import litellm

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from strata_forge.llm.providers.config import ProviderConfig
    from strata_forge.llm.registry import Model, ProviderName

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

    def describe_to_litellm(self, provider_model_id: str, model: Model) -> None:
        """Register ``model`` with LiteLLM under this provider when LiteLLM's map lacks it.

        The entry carries the registry's streaming capability and prices, so LiteLLM streams
        the model natively and its own cost callbacks (the Langfuse trace) price it instead of
        reporting zero. A model LiteLLM already maps is left as LiteLLM describes it.
        """
        custom_llm_provider = self.litellm_prefix.rstrip("/")
        try:
            litellm.get_model_info(  # pyright: ignore[reportUnknownMemberType]
                model=provider_model_id, custom_llm_provider=custom_llm_provider
            )
        except Exception:  # LiteLLM signals an unmapped model with a bare Exception
            litellm.register_model(  # pyright: ignore[reportUnknownMemberType]
                {
                    self.litellm_model(provider_model_id): _litellm_model_entry(
                        custom_llm_provider, model
                    )
                }
            )

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


def _litellm_model_entry(custom_llm_provider: str, model: Model) -> dict[str, Any]:
    """A LiteLLM model-map entry for ``model``, from the registry's own values.

    It carries no ``mode``: LiteLLM sends a ``mode: responses`` model's Chat Completions calls
    through its Responses bridge, which Forge does not use.
    """
    pricing = model.pricing_per_million_tokens
    entry: dict[str, Any] = {
        "litellm_provider": custom_llm_provider,
        "max_input_tokens": model.context_window,
        "max_output_tokens": model.max_output_tokens,
        "input_cost_per_token": pricing.input / 1_000_000,
        "output_cost_per_token": pricing.output / 1_000_000,
        "supports_native_streaming": model.capabilities.streaming,
    }
    if pricing.cache_read is not None:
        entry["cache_read_input_token_cost"] = pricing.cache_read / 1_000_000
    if pricing.cache_write is not None:
        entry["cache_creation_input_token_cost"] = pricing.cache_write / 1_000_000
    return entry


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
