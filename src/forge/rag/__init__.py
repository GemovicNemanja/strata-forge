"""Retrieval-augmented generation primitives.

The module ships four Protocols (:class:`Embedder`, :class:`Chunker`,
:class:`Retriever`, :class:`VectorStore`) plus concrete in-process
and production-backend implementations.

The :class:`QdrantVectorStore` and any reranker SDKs are
lazy-imported behind the ``[rag]`` extra. The composable RAG
pipeline lands in Phase 4.4.
"""

from forge.rag.chunking import (
    Chunk,
    Chunker,
    Document,
    RecursiveChunker,
)
from forge.rag.dense import DenseRetriever
from forge.rag.embedding import Embedder, LiteLLMEmbedder
from forge.rag.qdrant import QdrantVectorStore
from forge.rag.retrieval import RetrievalResult, Retriever
from forge.rag.vector_store import (
    InMemoryVectorStore,
    VectorItem,
    VectorSearchResult,
    VectorStore,
    cosine_similarity,
)

__all__ = [
    "Chunk",
    "Chunker",
    "DenseRetriever",
    "Document",
    "Embedder",
    "InMemoryVectorStore",
    "LiteLLMEmbedder",
    "QdrantVectorStore",
    "RecursiveChunker",
    "RetrievalResult",
    "Retriever",
    "VectorItem",
    "VectorSearchResult",
    "VectorStore",
    "cosine_similarity",
]
