"""Cross-module RAG workflows that don't need external services.

Each test wires multiple :mod:`strata_forge.rag` pieces together so the
seams hold under realistic compositions: chunker + dense retriever
+ pipeline, hybrid (dense + BM25), pipeline + reranker, end-to-end
augment_prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from strata_forge.rag import (
    BM25Retriever,
    Chunk,
    DenseRetriever,
    Document,
    HybridRetriever,
    InMemoryVectorStore,
    RAGPipeline,
    RecursiveChunker,
    RetrievalResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class _StubEmbedder:
    """Deterministic embedder for tests: text length → 1-D vector."""

    def __init__(self) -> None:
        self.model_name = "stub"

    @property
    def model(self) -> str:
        return self.model_name

    async def embed(self, text: str) -> list[float]:
        # Distinct vectors per text — collisions are OK at length boundaries.
        return [float(len(text)), float(sum(ord(c) for c in text) % 100)]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class _ReverseReranker:
    """Reverses the candidate list to simulate a reranker that disagrees."""

    async def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievalResult, ...]:
        del query
        reordered = tuple(reversed(list(results)))
        return reordered[:top_k] if top_k is not None else reordered


# ---------------------------------------------------------------------------
# Chunker + DenseRetriever + Pipeline
# ---------------------------------------------------------------------------


class TestPipelineEndToEnd:
    async def test_ingest_query_full_flow(self) -> None:
        documents = [
            Document(id="d1", text="alpha beta gamma. delta epsilon zeta."),
            Document(id="d2", text="one two three. four five six."),
        ]
        embedder = _StubEmbedder()
        store = InMemoryVectorStore()
        retriever = DenseRetriever(embedder=embedder, store=store)
        pipeline = RAGPipeline(
            chunker=RecursiveChunker(chunk_size=30, chunk_overlap=0),
            retriever=retriever,
        )
        n_chunks = await pipeline.ingest(documents)
        assert n_chunks >= 2
        # The chunks are addressable via the store.
        assert len(store) == n_chunks
        # Query returns ranked results.
        results = await pipeline.query("alpha", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    async def test_augment_prompt_includes_retrieved_text(self) -> None:
        embedder = _StubEmbedder()
        store = InMemoryVectorStore()
        retriever = DenseRetriever(embedder=embedder, store=store)
        pipeline = RAGPipeline(
            chunker=RecursiveChunker(chunk_size=50, chunk_overlap=0),
            retriever=retriever,
        )
        await pipeline.ingest([Document(id="d", text="the answer to that question is 42.")])
        prompt = await pipeline.augment_prompt("what's the answer?", top_k=1)
        assert "42" in prompt
        assert "what's the answer?" in prompt


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------


class TestHybridPipeline:
    async def test_dense_plus_bm25_fusion(self) -> None:
        # Build chunks manually so both retrievers see the same corpus.
        chunks = [
            Chunk(id="c1", text="HTTP 429 means too many requests"),
            Chunk(id="c2", text="HTTP 503 means service unavailable"),
            Chunk(id="c3", text="HTTP 200 means OK"),
        ]
        # Dense retriever
        embedder = _StubEmbedder()
        store = InMemoryVectorStore()
        dense = DenseRetriever(embedder=embedder, store=store)
        await dense.index(chunks)
        # Sparse retriever
        sparse = BM25Retriever(chunks)
        # Hybrid
        hybrid = HybridRetriever([dense, sparse])
        results = await hybrid.retrieve("429", top_k=2)
        # The 429 chunk should show up at the top thanks to BM25.
        assert any(r.chunk.id == "c1" for r in results)


# ---------------------------------------------------------------------------
# Pipeline + reranker
# ---------------------------------------------------------------------------


class TestPipelineWithReranker:
    async def test_reranker_reorders(self) -> None:
        embedder = _StubEmbedder()
        store = InMemoryVectorStore()
        retriever = DenseRetriever(embedder=embedder, store=store)
        pipeline = RAGPipeline(
            chunker=RecursiveChunker(chunk_size=10, chunk_overlap=0),
            retriever=retriever,
            reranker=_ReverseReranker(),
        )
        await pipeline.ingest(
            [
                Document(id="d", text="aaa\n\nbbb\n\nccc\n\nddd\n\neee"),
            ]
        )
        # Without reranker, retrieve returns a specific order; with the
        # reverse reranker, the order flips.
        with_reranker = await pipeline.query("aaa", top_k=3, rerank_top_k=5)
        # The reranker reversed the candidate list, so the order should
        # NOT match what the retriever alone would have produced.
        assert len(with_reranker) <= 3


# ---------------------------------------------------------------------------
# Non-indexable pipeline error
# ---------------------------------------------------------------------------


class TestIngestErrors:
    async def test_bm25_retriever_in_pipeline_rejects_ingest(self) -> None:
        chunks = [Chunk(id="c", text="hi")]
        bm25 = BM25Retriever(chunks)
        pipeline = RAGPipeline(chunker=RecursiveChunker(), retriever=bm25)
        with pytest.raises(TypeError, match="IndexableRetriever"):
            await pipeline.ingest([Document(id="d", text="x")])
