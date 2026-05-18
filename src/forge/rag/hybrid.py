"""Hybrid retrieval via Reciprocal Rank Fusion.

:class:`HybridRetriever` runs several retrievers in parallel and
fuses their results using Reciprocal Rank Fusion (RRF). The fused
score for a chunk is the sum of ``1 / (rrf_k + rank)`` across every
retriever that returned it; chunks that show up high in multiple
ranking lists win.

RRF needs no calibration between retrievers — dense cosine scores,
BM25 scores, and reranker logits can all be fused without manual
weight tuning. Optional per-retriever ``weights`` multiply each
retriever's contribution so callers can still bias the fusion
toward (say) dense when they know the corpus.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from forge.rag.retrieval import RetrievalResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from forge.rag.chunking import Chunk
    from forge.rag.retrieval import Retriever

__all__ = [
    "HybridRetriever",
]


class HybridRetriever:
    """Fuse multiple :class:`Retriever`s via Reciprocal Rank Fusion.

    Args:
        retrievers: At least two retrievers to fuse. With a single
            retriever, hybrid retrieval degenerates to the wrapped
            retriever's results — the constructor rejects that case
            so the caller is forced to be explicit.
        weights: Optional per-retriever multiplier on the RRF
            contribution. Same length as ``retrievers``. Default
            ``None`` (uniform weight 1.0).
        rrf_k: RRF constant; default 60 (the value used in the
            original RRF paper). Larger values flatten the score
            distribution.
        per_retriever_top_k: How many results to ask each retriever
            for. Default ``None``, which uses the caller's top_k
            argument verbatim. Set higher to widen the candidate
            pool before fusion.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        weights: Sequence[float] | None = None,
        rrf_k: int = 60,
        per_retriever_top_k: int | None = None,
    ) -> None:
        if len(retrievers) < 2:
            err = "HybridRetriever: pass at least two retrievers to fuse"
            raise ValueError(err)
        if weights is not None and len(weights) != len(retrievers):
            err = (
                f"HybridRetriever: weights length ({len(weights)}) must match "
                f"retrievers length ({len(retrievers)})"
            )
            raise ValueError(err)
        if rrf_k < 1:
            err = f"rrf_k must be >= 1; got {rrf_k}"
            raise ValueError(err)
        if per_retriever_top_k is not None and per_retriever_top_k < 1:
            err = f"per_retriever_top_k must be >= 1; got {per_retriever_top_k}"
            raise ValueError(err)
        self._retrievers = tuple(retrievers)
        self._weights = tuple(weights) if weights is not None else tuple(1.0 for _ in retrievers)
        self._rrf_k = rrf_k
        self._per_retriever_top_k = per_retriever_top_k

    @property
    def retrievers(self) -> tuple[Retriever, ...]:
        return self._retrievers

    @property
    def weights(self) -> tuple[float, ...]:
        return self._weights

    @property
    def rrf_k(self) -> int:
        return self._rrf_k

    async def retrieve(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        """Run every retriever and fuse the results via RRF."""
        if not query:
            err = "HybridRetriever.retrieve: query must be non-empty"
            raise ValueError(err)
        if top_k <= 0:
            err = f"top_k must be >= 1; got {top_k}"
            raise ValueError(err)

        per_retriever = self._per_retriever_top_k or top_k
        # Query all retrievers concurrently.
        results_per_retriever = await asyncio.gather(
            *(retriever.retrieve(query, top_k=per_retriever) for retriever in self._retrievers)
        )

        # Accumulate RRF scores keyed by chunk id; keep the first Chunk we see
        # for each id (they should be identical across retrievers — we don't
        # try to merge metadata).
        fused_scores: dict[str, float] = {}
        seen_chunk: dict[str, Chunk] = {}
        for retriever_idx, results in enumerate(results_per_retriever):
            weight = self._weights[retriever_idx]
            for rank, result in enumerate(results):
                contribution = weight / (self._rrf_k + rank + 1)
                fused_scores[result.chunk.id] = (
                    fused_scores.get(result.chunk.id, 0.0) + contribution
                )
                if result.chunk.id not in seen_chunk:
                    seen_chunk[result.chunk.id] = result.chunk

        ranked = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
        return tuple(
            RetrievalResult(chunk=seen_chunk[chunk_id], score=score)
            for chunk_id, score in ranked[:top_k]
        )
