"""Unit tests for `strata_forge.rag.dense`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from strata_forge.rag.chunking import Chunk
from strata_forge.rag.dense import DenseRetriever
from strata_forge.rag.retrieval import Retriever
from strata_forge.rag.vector_store import InMemoryVectorStore

if TYPE_CHECKING:
    from collections.abc import Sequence


class _MockEmbedder:
    """Minimal Embedder implementation for tests."""

    def __init__(
        self,
        mapping: dict[str, list[float]] | None = None,
        *,
        model: str = "mock",
    ) -> None:
        self._mapping = mapping or {}
        self._model_name = model
        self.embed = AsyncMock(side_effect=self._embed)
        self.embed_batch = AsyncMock(side_effect=self._embed_batch)

    @property
    def model(self) -> str:
        return self._model_name

    async def _embed(self, text: str) -> Sequence[float]:
        if text not in self._mapping:
            msg = f"unexpected embed input: {text!r}"
            raise AssertionError(msg)
        return self._mapping[text]

    async def _embed_batch(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [await self._embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_satisfies_retriever_protocol(self) -> None:
        retriever = DenseRetriever(embedder=_MockEmbedder(), store=InMemoryVectorStore())
        assert isinstance(retriever, Retriever)


# ---------------------------------------------------------------------------
# Index tests
# ---------------------------------------------------------------------------


class TestIndex:
    async def test_indexes_chunks(self) -> None:
        store = InMemoryVectorStore()
        embedder = _MockEmbedder({"hello": [1.0, 0.0], "world": [0.0, 1.0]})
        retriever = DenseRetriever(embedder=embedder, store=store)

        await retriever.index(
            [
                Chunk(id="c1", text="hello", document_id="doc-1"),
                Chunk(id="c2", text="world", document_id="doc-2"),
            ]
        )
        assert len(store) == 2

    async def test_embeds_in_batch(self) -> None:
        store = InMemoryVectorStore()
        embedder = _MockEmbedder({"a": [1.0], "b": [0.5], "c": [0.0]})
        retriever = DenseRetriever(embedder=embedder, store=store)

        await retriever.index(
            [
                Chunk(id="1", text="a"),
                Chunk(id="2", text="b"),
                Chunk(id="3", text="c"),
            ]
        )
        # The embedder's batch method gets the full list in one call.
        embedder.embed_batch.assert_called_once_with(["a", "b", "c"])
        # The single-text embed shouldn't have been called.
        embedder.embed.assert_not_called()

    async def test_empty_chunks_noop(self) -> None:
        store = InMemoryVectorStore()
        embedder = _MockEmbedder()
        retriever = DenseRetriever(embedder=embedder, store=store)
        await retriever.index([])
        embedder.embed.assert_not_called()
        embedder.embed_batch.assert_not_called()
        assert len(store) == 0

    async def test_chunk_metadata_propagated_to_store(self) -> None:
        store = InMemoryVectorStore()
        embedder = _MockEmbedder({"hello": [1.0, 0.0]})
        retriever = DenseRetriever(embedder=embedder, store=store)

        chunk = Chunk(
            id="c1",
            text="hello",
            metadata={"source": "wiki"},
            document_id="doc-x",
        )
        await retriever.index([chunk])

        # The stored item carries the chunk's metadata and a document_id hint.
        results = await retriever.retrieve("hello")
        assert results[0].chunk.id == "c1"
        assert results[0].chunk.metadata == {"source": "wiki"}
        assert results[0].chunk.document_id == "doc-x"


# ---------------------------------------------------------------------------
# Retrieve tests
# ---------------------------------------------------------------------------


class TestRetrieve:
    async def test_returns_top_k_chunks(self) -> None:
        store = InMemoryVectorStore()
        embedder = _MockEmbedder(
            {
                "cat": [1.0, 0.0],
                "dog": [0.95, 0.1],
                "fish": [0.0, 1.0],
                "query": [1.0, 0.05],
            }
        )
        retriever = DenseRetriever(embedder=embedder, store=store)
        await retriever.index(
            [
                Chunk(id="cat", text="cat"),
                Chunk(id="dog", text="dog"),
                Chunk(id="fish", text="fish"),
            ]
        )

        results = await retriever.retrieve("query", top_k=2)
        assert len(results) == 2
        assert results[0].chunk.text == "cat"
        assert results[1].chunk.text == "dog"

    async def test_empty_query_rejected(self) -> None:
        retriever = DenseRetriever(embedder=_MockEmbedder(), store=InMemoryVectorStore())
        with pytest.raises(ValueError, match="non-empty"):
            await retriever.retrieve("")

    async def test_invalid_top_k_rejected(self) -> None:
        retriever = DenseRetriever(
            embedder=_MockEmbedder({"q": [1.0]}), store=InMemoryVectorStore()
        )
        with pytest.raises(ValueError, match="top_k"):
            await retriever.retrieve("q", top_k=0)

    async def test_search_on_empty_store(self) -> None:
        retriever = DenseRetriever(
            embedder=_MockEmbedder({"q": [1.0]}), store=InMemoryVectorStore()
        )
        results = await retriever.retrieve("q")
        assert results == ()

    async def test_document_id_recovered_from_metadata(self) -> None:
        store = InMemoryVectorStore()
        embedder = _MockEmbedder({"hello": [1.0]})
        retriever = DenseRetriever(embedder=embedder, store=store)

        chunk = Chunk(id="c", text="hello", document_id="doc-42")
        await retriever.index([chunk])
        results = await retriever.retrieve("hello")
        assert results[0].chunk.document_id == "doc-42"

    async def test_chunk_without_document_id_round_trips(self) -> None:
        store = InMemoryVectorStore()
        embedder = _MockEmbedder({"hello": [1.0]})
        retriever = DenseRetriever(embedder=embedder, store=store)

        await retriever.index([Chunk(id="c", text="hello")])
        results = await retriever.retrieve("hello")
        assert results[0].chunk.document_id is None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_embedder_property(self) -> None:
        embedder = _MockEmbedder()
        store = InMemoryVectorStore()
        retriever = DenseRetriever(embedder=embedder, store=store)
        assert retriever.embedder is embedder
        assert retriever.store is store
