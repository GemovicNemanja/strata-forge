"""Unit tests for `strata_forge.rag.pipeline`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from strata_forge.rag.chunking import Chunk, Document, RecursiveChunker
from strata_forge.rag.pipeline import (
    DEFAULT_AUGMENT_TEMPLATE,
    IndexableRetriever,
    RAGPipeline,
)
from strata_forge.rag.retrieval import RetrievalResult

if TYPE_CHECKING:
    from collections.abc import Sequence


class _IndexableFakeRetriever:
    """Test retriever that records what it was asked to index and to retrieve."""

    def __init__(self) -> None:
        self.indexed: list[Chunk] = []
        self.calls: list[tuple[str, int]] = []
        self._results: tuple[RetrievalResult, ...] = ()

    def set_results(self, results: tuple[RetrievalResult, ...]) -> None:
        self._results = results

    async def index(self, chunks: Sequence[Chunk]) -> None:
        self.indexed.extend(chunks)

    async def retrieve(self, query: str, *, top_k: int = 5) -> Sequence[RetrievalResult]:
        self.calls.append((query, top_k))
        return self._results[:top_k]


class _NonIndexableFakeRetriever:
    """Test retriever without an .index method (e.g. a BM25 retriever)."""

    def __init__(self, results: tuple[RetrievalResult, ...] = ()) -> None:
        self._results = results

    async def retrieve(self, query: str, *, top_k: int = 5) -> Sequence[RetrievalResult]:
        del query
        return self._results[:top_k]


class _FakeReranker:
    """Reverses the input order to simulate a reranker that disagrees."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievalResult, ...]:
        self.calls.append((query, top_k or 0))
        reordered = tuple(reversed(list(results)))
        if top_k is None:
            return reordered
        return reordered[:top_k]


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, text=text)


def _retrieval(chunk_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(chunk=_chunk(chunk_id, f"text-{chunk_id}"), score=score)


# ---------------------------------------------------------------------------
# IndexableRetriever Protocol
# ---------------------------------------------------------------------------


class TestIndexableProtocol:
    def test_indexable_satisfies(self) -> None:
        assert isinstance(_IndexableFakeRetriever(), IndexableRetriever)

    def test_non_indexable_does_not_satisfy(self) -> None:
        assert not isinstance(_NonIndexableFakeRetriever(), IndexableRetriever)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_properties_exposed(self) -> None:
        chunker = RecursiveChunker(chunk_size=500)
        retriever = _IndexableFakeRetriever()
        reranker = _FakeReranker()
        pipeline = RAGPipeline(chunker=chunker, retriever=retriever, reranker=reranker)
        assert pipeline.chunker is chunker
        assert pipeline.retriever is retriever
        assert pipeline.reranker is reranker

    def test_reranker_optional(self) -> None:
        pipeline = RAGPipeline(
            chunker=RecursiveChunker(),
            retriever=_IndexableFakeRetriever(),
        )
        assert pipeline.reranker is None


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


class TestIngest:
    async def test_chunks_and_indexes(self) -> None:
        retriever = _IndexableFakeRetriever()
        pipeline = RAGPipeline(
            chunker=RecursiveChunker(chunk_size=10, chunk_overlap=0),
            retriever=retriever,
        )
        documents = [
            Document(id="doc-1", text="alpha beta gamma delta epsilon"),
            Document(id="doc-2", text="hello world"),
        ]
        count = await pipeline.ingest(documents)
        # First doc is over chunk_size so it splits; second fits.
        assert count == len(retriever.indexed)
        # Chunks are tagged with their document_id.
        assert all(c.document_id in {"doc-1", "doc-2"} for c in retriever.indexed)

    async def test_empty_documents_returns_zero(self) -> None:
        retriever = _IndexableFakeRetriever()
        pipeline = RAGPipeline(chunker=RecursiveChunker(), retriever=retriever)
        count = await pipeline.ingest([])
        assert count == 0
        assert retriever.indexed == []

    async def test_non_indexable_retriever_rejected(self) -> None:
        pipeline = RAGPipeline(
            chunker=RecursiveChunker(),
            retriever=_NonIndexableFakeRetriever(),
        )
        with pytest.raises(TypeError, match="IndexableRetriever"):
            await pipeline.ingest([Document(id="d", text="hello")])


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


class TestQueryWithoutReranker:
    async def test_passes_query_through_to_retriever(self) -> None:
        retriever = _IndexableFakeRetriever()
        retriever.set_results((_retrieval("a"), _retrieval("b")))
        pipeline = RAGPipeline(chunker=RecursiveChunker(), retriever=retriever)
        results = await pipeline.query("hi", top_k=2)
        assert [r.chunk.id for r in results] == ["a", "b"]
        assert retriever.calls == [("hi", 2)]

    async def test_empty_query_rejected(self) -> None:
        pipeline = RAGPipeline(chunker=RecursiveChunker(), retriever=_IndexableFakeRetriever())
        with pytest.raises(ValueError, match="non-empty"):
            await pipeline.query("")

    async def test_invalid_top_k_rejected(self) -> None:
        pipeline = RAGPipeline(chunker=RecursiveChunker(), retriever=_IndexableFakeRetriever())
        with pytest.raises(ValueError, match="top_k"):
            await pipeline.query("q", top_k=0)


class TestQueryWithReranker:
    async def test_reranks_after_retrieval(self) -> None:
        retriever = _IndexableFakeRetriever()
        retriever.set_results((_retrieval("a"), _retrieval("b"), _retrieval("c")))
        reranker = _FakeReranker()
        pipeline = RAGPipeline(
            chunker=RecursiveChunker(),
            retriever=retriever,
            reranker=reranker,
        )
        results = await pipeline.query("q", top_k=3)
        # Fake reranker reverses; final order is c, b, a.
        assert [r.chunk.id for r in results] == ["c", "b", "a"]
        assert reranker.calls == [("q", 3)]

    async def test_widens_pool_with_rerank_top_k(self) -> None:
        retriever = _IndexableFakeRetriever()
        retriever.set_results(tuple(_retrieval(str(i)) for i in range(10)))
        reranker = _FakeReranker()
        pipeline = RAGPipeline(
            chunker=RecursiveChunker(),
            retriever=retriever,
            reranker=reranker,
        )
        await pipeline.query("q", top_k=3, rerank_top_k=10)
        # Retriever was asked for 10 candidates; reranker was asked for top 3.
        assert retriever.calls == [("q", 10)]
        assert reranker.calls == [("q", 3)]

    async def test_rerank_top_k_smaller_than_top_k_rejected(self) -> None:
        pipeline = RAGPipeline(
            chunker=RecursiveChunker(),
            retriever=_IndexableFakeRetriever(),
            reranker=_FakeReranker(),
        )
        with pytest.raises(ValueError, match="rerank_top_k"):
            await pipeline.query("q", top_k=5, rerank_top_k=2)


# ---------------------------------------------------------------------------
# augment_prompt
# ---------------------------------------------------------------------------


class TestAugmentPrompt:
    async def test_substitutes_query_and_context(self) -> None:
        retriever = _IndexableFakeRetriever()
        retriever.set_results(
            (
                RetrievalResult(chunk=Chunk(id="a", text="first source"), score=1.0),
                RetrievalResult(chunk=Chunk(id="b", text="second source"), score=0.5),
            )
        )
        pipeline = RAGPipeline(chunker=RecursiveChunker(), retriever=retriever)
        prompt = await pipeline.augment_prompt("what is x?", top_k=2)
        assert "what is x?" in prompt
        assert "first source" in prompt
        assert "second source" in prompt
        # Default template's "Sources:" header is present.
        assert "Sources:" in prompt

    async def test_custom_template(self) -> None:
        retriever = _IndexableFakeRetriever()
        retriever.set_results((RetrievalResult(chunk=Chunk(id="a", text="X"), score=1.0),))
        pipeline = RAGPipeline(chunker=RecursiveChunker(), retriever=retriever)
        prompt = await pipeline.augment_prompt(
            "Q",
            top_k=1,
            template="CTX={context}/QRY={query}",
        )
        assert prompt == "CTX=X/QRY=Q"

    async def test_custom_separator(self) -> None:
        retriever = _IndexableFakeRetriever()
        retriever.set_results(
            (
                RetrievalResult(chunk=Chunk(id="a", text="alpha"), score=1.0),
                RetrievalResult(chunk=Chunk(id="b", text="beta"), score=0.5),
            )
        )
        pipeline = RAGPipeline(chunker=RecursiveChunker(), retriever=retriever)
        prompt = await pipeline.augment_prompt(
            "q",
            top_k=2,
            template="{context}",
            source_separator=" || ",
        )
        assert prompt == "alpha || beta"

    async def test_default_template_has_placeholders(self) -> None:
        # The default template is exposed; verify it has both placeholders.
        assert "{context}" in DEFAULT_AUGMENT_TEMPLATE
        assert "{query}" in DEFAULT_AUGMENT_TEMPLATE
