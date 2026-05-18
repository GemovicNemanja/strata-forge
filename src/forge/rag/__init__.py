"""Retrieval-augmented generation primitives.

The module ships four Protocols (:class:`Embedder`, :class:`Chunker`,
:class:`Retriever`, :class:`VectorStore`) plus concrete in-process
and production-backend implementations and a composable
:class:`RAGPipeline` that ties them together.

The :class:`QdrantVectorStore` and :class:`CohereReranker` lazy-import
their SDKs behind the ``[rag]`` extra; :class:`CrossEncoderReranker`
needs ``sentence-transformers`` installed separately (it transitively
brings torch, which is too heavy for the default ``[rag]`` extra).
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
from forge.rag.pipeline import (
    DEFAULT_AUGMENT_TEMPLATE,
    IndexableRetriever,
    RAGPipeline,
)
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
    "DEFAULT_AUGMENT_TEMPLATE",
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
    "IndexableRetriever",
    "LiteLLMEmbedder",
    "QdrantVectorStore",
    "RAGPipeline",
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
