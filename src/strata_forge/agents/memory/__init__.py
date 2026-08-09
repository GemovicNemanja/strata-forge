"""Memory primitives for :class:`strata_forge.agents.Agent`.

Two flavours of memory ship:

- :class:`ConversationMemory` — in-process append-only message
  history with optional token-budget or message-count trimming.
  Preserves the system message across trims so the agent's
  instructions never get evicted.
- :class:`EpisodicMemory` — vector-backed long-term memory against a
  pluggable :class:`VectorStore` Protocol. A concrete
  :class:`InMemoryVectorStore` ships here for tests and prototyping;
  :mod:`strata_forge.rag` provides a Qdrant-backed implementation,
  :class:`QdrantVectorStore`, that satisfies the same Protocol.
"""

from strata_forge.agents.memory.conversation import ConversationMemory
from strata_forge.agents.memory.episodic import EmbedFn, EpisodicMemory
from strata_forge.agents.memory.vector_store import (
    InMemoryVectorStore,
    VectorItem,
    VectorSearchResult,
    VectorStore,
)

__all__ = [
    "ConversationMemory",
    "EmbedFn",
    "EpisodicMemory",
    "InMemoryVectorStore",
    "VectorItem",
    "VectorSearchResult",
    "VectorStore",
]
