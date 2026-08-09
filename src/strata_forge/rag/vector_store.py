"""Pluggable vector-store Protocol plus an in-process implementation.

The :class:`VectorStore` Protocol is the contract any backend must
satisfy. :class:`InMemoryVectorStore` is a dict-backed, no-dep
implementation suitable for tests, prototyping, and small-scale
agent / RAG runs. The Qdrant-backed implementation lives in
:mod:`strata_forge.rag.qdrant` and satisfies the same Protocol.

These primitives originally lived in :mod:`strata_forge.agents.memory`;
ADR 0012 moved them here so the dependency arrow points
the right way (``agents → rag``) and so a single source of truth
serves both modules. :mod:`strata_forge.agents.memory` re-exports the
names for back-compatibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "InMemoryVectorStore",
    "VectorItem",
    "VectorSearchResult",
    "VectorStore",
    "cosine_similarity",
]


@dataclass(frozen=True, slots=True)
class VectorItem:
    """One stored embedding plus the text and metadata it represents."""

    id: str
    embedding: tuple[float, ...]
    text: str
    metadata: Mapping[str, Any] = field(default_factory=lambda: cast("Mapping[str, Any]", {}))


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """A search hit: the stored item and the similarity score."""

    item: VectorItem
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """The contract every vector-store backend satisfies.

    Async by design — the Qdrant and other production backends are
    async-native.
    """

    async def add(self, items: Sequence[VectorItem]) -> None:
        """Insert or replace items by id."""
        ...  # pragma: no cover — Protocol body

    async def search(
        self, embedding: Sequence[float], *, top_k: int = 5
    ) -> tuple[VectorSearchResult, ...]:
        """Return the ``top_k`` highest-scoring items for ``embedding``."""
        ...  # pragma: no cover — Protocol body

    async def delete(self, ids: Sequence[str]) -> None:
        """Remove items by id. Missing ids are silently ignored."""
        ...  # pragma: no cover — Protocol body

    async def clear(self) -> None:
        """Remove every stored item."""
        ...  # pragma: no cover — Protocol body


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns ``0.0`` when either vector is all zeros — defensive
    default for undefined direction. Raises :class:`ValueError`
    when the vectors have different lengths.
    """
    if len(a) != len(b):
        err = f"vector length mismatch: {len(a)} vs {len(b)}"
        raise ValueError(err)
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    """Dict-backed :class:`VectorStore`.

    Suitable for tests and prototyping. Cosine similarity in pure
    Python — no numpy dep. Storage is a single dict keyed by item
    id, so :meth:`add` overwrites existing entries with the same id.
    """

    def __init__(self) -> None:
        self._items: dict[str, VectorItem] = {}

    async def add(self, items: Sequence[VectorItem]) -> None:
        for item in items:
            self._items[item.id] = item

    async def search(
        self, embedding: Sequence[float], *, top_k: int = 5
    ) -> tuple[VectorSearchResult, ...]:
        if top_k <= 0:
            err = f"top_k must be >= 1; got {top_k}"
            raise ValueError(err)
        if not self._items:
            return ()
        scored = [
            VectorSearchResult(
                item=item,
                score=cosine_similarity(embedding, item.embedding),
            )
            for item in self._items.values()
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return tuple(scored[:top_k])

    async def delete(self, ids: Sequence[str]) -> None:
        for item_id in ids:
            self._items.pop(item_id, None)

    async def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
