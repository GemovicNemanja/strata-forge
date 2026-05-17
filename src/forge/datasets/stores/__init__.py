"""Backend implementations for :class:`forge.datasets.DatasetStore`.

Phase 2.3.1 ships :class:`InMemoryDatasetStore`; the Langfuse backend
lands in 2.3.2.
"""

from forge.datasets.stores.memory import InMemoryDatasetStore

__all__ = ["InMemoryDatasetStore"]
