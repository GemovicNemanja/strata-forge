"""Retrieval-augmented generation primitives.

The 4.1 foundation ships the Protocol set and the in-process
implementations: :class:`Embedder` + :class:`LiteLLMEmbedder`,
:class:`Document` / :class:`Chunk` + :class:`Chunker` +
:class:`RecursiveChunker`, :class:`Retriever` /
:class:`RetrievalResult`, and the :class:`VectorStore` family
(relocated from :mod:`forge.agents.memory` per ADR 0012).

Vector stores beyond the in-process backend (Phase 4.2), sparse +
hybrid retrieval and rerankers (Phase 4.3), and the composable
pipeline (Phase 4.4) land in subsequent sub-phases.
"""

from forge.rag.chunking import (
    Chunk,
    Chunker,
    Document,
    RecursiveChunker,
)
from forge.rag.embedding import Embedder, LiteLLMEmbedder
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
    "Document",
    "Embedder",
    "InMemoryVectorStore",
    "LiteLLMEmbedder",
    "RecursiveChunker",
    "RetrievalResult",
    "Retriever",
    "VectorItem",
    "VectorSearchResult",
    "VectorStore",
    "cosine_similarity",
]
