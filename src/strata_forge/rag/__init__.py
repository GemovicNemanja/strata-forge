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

from strata_forge.rag.bm25 import BM25Retriever, tokenize
from strata_forge.rag.chunking import (
    Chunk,
    Chunker,
    Document,
    RecursiveChunker,
)
from strata_forge.rag.dense import DenseRetriever
from strata_forge.rag.embedding import Embedder, LiteLLMEmbedder
from strata_forge.rag.hybrid import HybridRetriever
from strata_forge.rag.pipeline import (
    DEFAULT_AUGMENT_TEMPLATE,
    IndexableRetriever,
    RAGPipeline,
)
from strata_forge.rag.qdrant import QdrantVectorStore
from strata_forge.rag.rerankers import CohereReranker, CrossEncoderReranker, Reranker
from strata_forge.rag.retrieval import RetrievalResult, Retriever
from strata_forge.rag.vector_store import (
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
