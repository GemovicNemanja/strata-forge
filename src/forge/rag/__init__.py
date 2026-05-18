"""Retrieval-augmented generation primitives.

The module ships four Protocols (:class:`Embedder`, :class:`Chunker`,
:class:`Retriever`, :class:`VectorStore`) plus concrete in-process
and production-backend implementations.

The :class:`QdrantVectorStore` and :class:`CohereReranker` lazy-import
their SDKs behind the ``[rag]`` extra; :class:`CrossEncoderReranker`
needs ``sentence-transformers`` installed separately (it transitively
brings torch, which is too heavy for the default ``[rag]`` extra).
The composable RAG pipeline lands in Phase 4.4.
"""

from forge.rag.bm25 import BM25Retriever, tokenize
from forge.rag.chunking import (
    Chunk,
    Chunker,
    Document,
    RecursiveChunker,
)
from forge.rag.dense import DenseRetriever
from forge.rag.embedding import Embedder, LiteLLMEmbedder
from forge.rag.hybrid import HybridRetriever
from forge.rag.qdrant import QdrantVectorStore
from forge.rag.rerankers import CohereReranker, CrossEncoderReranker, Reranker
from forge.rag.retrieval import RetrievalResult, Retriever
from forge.rag.vector_store import (
    InMemoryVectorStore,
    VectorItem,
    VectorSearchResult,
    VectorStore,
    cosine_similarity,
)

__all__ = [
    "BM25Retriever",
    "Chunk",
    "Chunker",
    "CohereReranker",
    "CrossEncoderReranker",
    "DenseRetriever",
    "Document",
    "Embedder",
    "HybridRetriever",
    "InMemoryVectorStore",
    "LiteLLMEmbedder",
    "QdrantVectorStore",
    "RecursiveChunker",
    "Reranker",
    "RetrievalResult",
    "Retriever",
    "VectorItem",
    "VectorSearchResult",
    "VectorStore",
    "cosine_similarity",
    "tokenize",
]
