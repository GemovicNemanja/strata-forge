"""Memory primitives for :class:`forge.agents.Agent`.

Two flavours of memory ship:

- :class:`ConversationMemory` — in-process append-only message
  history with optional token-budget or message-count trimming.
  Preserves the system message across trims so the agent's
  instructions never get evicted.
- :class:`EpisodicMemory` — vector-backed long-term memory against a
  pluggable :class:`VectorStore` Protocol. A concrete
  :class:`InMemoryVectorStore` ships here for tests and prototyping;
  :mod:`forge.rag` (Phase 4) will provide a Qdrant-backed
  implementation that satisfies the same Protocol.
"""

from forge.agents.memory.conversation import ConversationMemory
from forge.agents.memory.episodic import EmbedFn, EpisodicMemory
from forge.agents.memory.vector_store import (
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
