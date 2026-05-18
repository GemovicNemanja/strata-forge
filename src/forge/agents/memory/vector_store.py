"""Vector-store re-exports for :mod:`forge.agents.memory`.

The Protocol and the in-process implementation live in
:mod:`forge.rag.vector_store` (per ADR 0012); this module is a thin
re-export so existing agent code keeps working.

New code should import directly from :mod:`forge.rag` —
``from forge.rag import VectorStore, InMemoryVectorStore, …``.
"""

from forge.rag.vector_store import (
    InMemoryVectorStore,
    VectorItem,
    VectorSearchResult,
    VectorStore,
)
from forge.rag.vector_store import cosine_similarity as _cosine_similarity

__all__ = [
    "InMemoryVectorStore",
    "VectorItem",
    "VectorSearchResult",
    "VectorStore",
    "_cosine_similarity",
]
