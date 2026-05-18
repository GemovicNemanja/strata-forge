"""Vector-backed long-term memory for agents.

:class:`EpisodicMemory` wraps a :class:`VectorStore` plus a
caller-supplied ``embed_fn`` (an async function that turns text into a
fixed-length vector). The agent's ``.add(text, ...)`` calls embed the
text and store it; ``.search(query, top_k=...)`` embeds the query and
returns the top-k closest stored items.

The vector-store backend is pluggable through the
:class:`VectorStore` Protocol. The in-process
:class:`InMemoryVectorStore` works out of the box; once
:mod:`forge.rag` lands its Qdrant backend, any
:class:`EpisodicMemory` written against the Protocol will accept it
without code changes.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from forge.agents.memory.vector_store import VectorItem

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from forge.agents.memory.vector_store import VectorSearchResult, VectorStore

__all__ = [
    "EmbedFn",
    "EpisodicMemory",
]


type EmbedFn = Callable[[str], Awaitable[Sequence[float]]]
"""Async embedding function — takes a string, returns a fixed-length vector."""


class EpisodicMemory:
    """A text-keyed vector memory built on a :class:`VectorStore`.

    Args:
        store: Any object satisfying the :class:`VectorStore`
            Protocol. The in-process :class:`InMemoryVectorStore`
            works for tests; production setups will pass a Qdrant
            or other backend.
        embed: Async function that maps a string to its embedding
            vector. The exact embedding model is up to the caller —
            ``EpisodicMemory`` doesn't care, but a single memory
            instance must use the same embed function for both
            stores and queries.
    """

    def __init__(self, *, store: VectorStore, embed: EmbedFn) -> None:
        self._store = store
        self._embed = embed

    async def add(
        self,
        text: str,
        *,
        item_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Embed ``text`` and store it; return the assigned id.

        When ``item_id`` is provided, it's used as-is (and overwrites
        any existing item with the same id). When omitted, a UUIDv4
        hex string is generated.
        """
        if not text:
            err = "EpisodicMemory.add: text must be non-empty"
            raise ValueError(err)
        assigned_id = item_id if item_id is not None else uuid.uuid4().hex
        embedding = await self._embed(text)
        item = VectorItem(
            id=assigned_id,
            embedding=tuple(embedding),
            text=text,
            metadata=dict(metadata or {}),
        )
        await self._store.add([item])
        return assigned_id

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[VectorSearchResult, ...]:
        """Return the ``top_k`` items most similar to ``query``.

        Embeds ``query`` via ``embed_fn`` and forwards to the
        backend's ``search``.
        """
        if not query:
            err = "EpisodicMemory.search: query must be non-empty"
            raise ValueError(err)
        if top_k <= 0:
            err = f"top_k must be >= 1; got {top_k}"
            raise ValueError(err)
        embedding = await self._embed(query)
        return await self._store.search(embedding, top_k=top_k)

    async def delete(self, ids: Sequence[str]) -> None:
        """Forward delete to the backend."""
        await self._store.delete(ids)

    async def clear(self) -> None:
        """Drop every stored item."""
        await self._store.clear()
