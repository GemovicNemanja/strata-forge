"""Synchronous wrappers around the async LLM API.

The async :class:`forge.llm.LLMClient` is the canonical interface — every
public method is `async def`. This module wraps the most common surface
(``complete``, ``complete_structured``, ``stream``, ``run_tool_loop``) in
sync facades for CLI and notebook ergonomics, where ``asyncio.run`` would
otherwise be at every call site.

Construction style: pass an existing :class:`~forge.llm.LLMClient`
instance via ``client=`` to reuse it across calls, or pass
``model=``/``provider=``/``chain=`` to build a one-shot client. Building
a fresh client on every call is fine for scripts and notebooks; reuse
matters when the client carries state (cache, mocked provider clients).

.. warning::

    Sync :func:`stream` collects every chunk into a list before
    returning the iterator — :mod:`asyncio.run` is single-shot, so true
    chunk-by-chunk streaming cannot work through a sync interface. Use
    :meth:`forge.llm.LLMClient.stream` from an async context if you
    actually need incremental output.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from forge.llm.client import LLMClient, StructuredResponse

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from forge.llm.fallback import FallbackEntry
    from forge.llm.messages import AnyMessage
    from forge.llm.registry import ProviderName
    from forge.llm.responses import LLMResponse, ResponseChunk
    from forge.llm.tools import Tool

__all__ = [
    "complete",
    "complete_structured",
    "run_tool_loop",
    "stream",
]


def _resolve_client(
    *,
    client: LLMClient | None,
    model: str | None,
    provider: ProviderName | None,
    chain: Sequence[FallbackEntry] | None,
) -> LLMClient:
    """Pick a client: ``client`` if given, otherwise build one from the kwargs.

    Exactly one of ``client`` / (``model`` or ``chain``) must be
    supplied; passing both is a programmer error and surfaces as a
    :class:`ValueError`.
    """
    if client is not None:
        if model is not None or provider is not None or chain is not None:
            msg = "Pass either `client` or (`model`/`provider`/`chain`), not both"
            raise ValueError(msg)
        return client
    return LLMClient(model=model, provider=provider, chain=chain)


def complete(
    messages: Sequence[AnyMessage],
    *,
    model: str | None = None,
    provider: ProviderName | None = None,
    chain: Sequence[FallbackEntry] | None = None,
    client: LLMClient | None = None,
    **kwargs: Any,
) -> LLMResponse:
    """Synchronous :meth:`~forge.llm.LLMClient.complete`.

    Accepts the same keyword arguments as the async method —
    ``temperature``, ``max_tokens``, ``top_p``, ``tools``,
    ``response_format``, ``provider_extras`` — and returns the same
    :class:`~forge.llm.LLMResponse`.
    """
    target = _resolve_client(client=client, model=model, provider=provider, chain=chain)
    return asyncio.run(target.complete(messages, **kwargs))


def complete_structured[M: BaseModel](
    messages: Sequence[AnyMessage],
    *,
    schema: type[M],
    model: str | None = None,
    provider: ProviderName | None = None,
    chain: Sequence[FallbackEntry] | None = None,
    client: LLMClient | None = None,
    **kwargs: Any,
) -> StructuredResponse[M]:
    """Synchronous :meth:`~forge.llm.LLMClient.complete_structured`.

    Returns the same :class:`~forge.llm.StructuredResponse` — call
    ``.parsed`` to get the validated Pydantic instance.
    """
    target = _resolve_client(client=client, model=model, provider=provider, chain=chain)
    return asyncio.run(target.complete_structured(messages, schema=schema, **kwargs))


def stream(
    messages: Sequence[AnyMessage],
    *,
    model: str | None = None,
    provider: ProviderName | None = None,
    chain: Sequence[FallbackEntry] | None = None,
    client: LLMClient | None = None,
    **kwargs: Any,
) -> Iterator[ResponseChunk]:
    """Synchronous wrapper that drains :meth:`~forge.llm.LLMClient.stream`.

    .. note::

        Every chunk is collected into a list before the iterator is
        returned — see the module docstring for why. For incremental
        output use the async API directly.
    """
    target = _resolve_client(client=client, model=model, provider=provider, chain=chain)

    async def _drain() -> list[ResponseChunk]:
        chunks: list[ResponseChunk] = []
        async for chunk in await target.stream(messages, **kwargs):
            chunks.append(chunk)
        return chunks

    return iter(asyncio.run(_drain()))


def run_tool_loop(
    messages: Sequence[AnyMessage],
    *,
    tools: Sequence[Tool],
    max_iterations: int = 8,
    model: str | None = None,
    provider: ProviderName | None = None,
    chain: Sequence[FallbackEntry] | None = None,
    client: LLMClient | None = None,
    **kwargs: Any,
) -> LLMResponse:
    """Synchronous :meth:`~forge.llm.LLMClient.run_tool_loop`."""
    target = _resolve_client(client=client, model=model, provider=provider, chain=chain)
    return asyncio.run(
        target.run_tool_loop(
            messages,
            tools=tools,
            max_iterations=max_iterations,
            **kwargs,
        )
    )
