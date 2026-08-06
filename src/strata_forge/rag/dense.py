"""Dense :class:`Retriever` — embed the query, query the vector store.

:class:`DenseRetriever` is the standard "ANN over embeddings" path:
takes an :class:`Embedder` plus a :class:`VectorStore` in the
constructor, exposes a one-shot :meth:`index` helper for ingesting
chunks, and serves queries through :meth:`retrieve`.

Search results are mapped back to :class:`Chunk` instances so
downstream consumers (rerankers, the RAG pipeline) work in the
chunk shape regardless of the underlying storage backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strata_forge.rag.chunking import Chunk
from strata_forge.rag.retrieval import RetrievalResult
from strata_forge.rag.vector_store import VectorItem

if TYPE_CHECKING:
    from collections.abc import Sequence

    from strata_forge.rag.embedding import Embedder
    from strata_forge.rag.vector_store import VectorStore

__all__ = [
    "DenseRetriever",
]

_DOCUMENT_ID_KEY = "__forge_document_id"


class DenseRetriever:
    """Retriever that embeds the query and looks it up in a :class:`VectorStore`.

    Args:
        embedder: Any :class:`Embedder` implementation. The query and
            the indexed chunks pass through the same embedder, so a
            single instance must be used for both directions.
        store: The :class:`VectorStore` holding the chunk embeddings.
    """

    def __init__(self, *, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def store(self) -> VectorStore:
        return self._store

    async def index(self, chunks: Sequence[Chunk]) -> None:
        """Embed ``chunks`` in one batch call and store them.

        Convenience for one-shot indexing — callers running a
        streaming pipeline can drive :meth:`Embedder.embed_batch` and
        :meth:`VectorStore.add` themselves. The chunk's
        ``document_id`` is stashed in the vector metadata under
        ``__forge_document_id`` so :meth:`retrieve` can recover it.
        """
        if not chunks:
            return
        embeddings = await self._embedder.embed_batch([c.text for c in chunks])
        items = [
            VectorItem(
                id=chunk.id,
                embedding=tuple(embedding),
                text=chunk.text,
                metadata=self._encode_metadata(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        await self._store.add(items)

    async def retrieve(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        """Embed ``query`` and return the top-``top_k`` chunks."""
        if not query:
            err = "DenseRetriever.retrieve: query must be non-empty"
            raise ValueError(err)
        if top_k <= 0:
            err = f"top_k must be >= 1; got {top_k}"
            raise ValueError(err)
        embedding = await self._embedder.embed(query)
        hits = await self._store.search(embedding, top_k=top_k)
        return tuple(
            RetrievalResult(chunk=self._decode_item(hit.item), score=hit.score) for hit in hits
        )

    @staticmethod
    def _encode_metadata(chunk: Chunk) -> dict[str, object]:
        metadata: dict[str, object] = dict(chunk.metadata)
        if chunk.document_id is not None:
            metadata[_DOCUMENT_ID_KEY] = chunk.document_id
        return metadata

    @staticmethod
    def _decode_item(item: VectorItem) -> Chunk:
        metadata = dict(item.metadata)
        document_id = metadata.pop(_DOCUMENT_ID_KEY, None)
        return Chunk(
            id=item.id,
            text=item.text,
            metadata=metadata,
            document_id=str(document_id) if document_id is not None else None,
        )
