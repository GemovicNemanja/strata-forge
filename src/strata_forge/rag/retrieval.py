"""Retriever Protocol and RetrievalResult shape.

Concrete retrievers — :class:`DenseRetriever` (Phase 4.2),
:class:`BM25Retriever` and :class:`HybridRetriever` (Phase 4.3) —
all implement this Protocol. The RAG pipeline (Phase 4.4) composes
retrievers and rerankers through it without caring about the
specific backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from strata_forge.rag.chunking import Chunk  # noqa: TC001 — Pydantic needs runtime resolution

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "RetrievalResult",
    "Retriever",
]


class RetrievalResult(BaseModel):
    """One hit returned by a :class:`Retriever`.

    Attributes:
        chunk: The retrieved :class:`Chunk`.
        score: Retriever-specific score. Higher is better. Different
            retrievers normalize differently (cosine in
            ``[-1, 1]`` for dense; BM25 unbounded; RRF fused score
            in ``[0, ~1]``) — callers should treat the score as
            ordinal, not absolute, unless they know the retriever's
            semantics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: Chunk
    score: float


@runtime_checkable
class Retriever(Protocol):
    """The contract every retriever satisfies.

    Async because production retrievers hit a network (vector
    store, search API). In-process retrievers still implement
    ``async def retrieve`` for shape uniformity.
    """

    async def retrieve(self, query: str, *, top_k: int = 5) -> Sequence[RetrievalResult]:
        """Return up to ``top_k`` retrieval hits for ``query``."""
        ...  # pragma: no cover — Protocol body
