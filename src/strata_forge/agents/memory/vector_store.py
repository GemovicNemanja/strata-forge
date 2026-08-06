"""Vector-store re-exports for :mod:`strata_forge.agents.memory`.

The Protocol and the in-process implementation live in
:mod:`strata_forge.rag.vector_store` (per ADR 0012); this module is a thin
re-export so existing agent code keeps working.

New code should import directly from :mod:`strata_forge.rag` —
``from strata_forge.rag import VectorStore, InMemoryVectorStore, …``.
"""

from strata_forge.rag.vector_store import (
    InMemoryVectorStore,
    VectorItem,
    VectorSearchResult,
    VectorStore,
)
from strata_forge.rag.vector_store import cosine_similarity as _cosine_similarity

__all__ = [
    "InMemoryVectorStore",
    "VectorItem",
    "VectorSearchResult",
    "VectorStore",
    "_cosine_similarity",
]
