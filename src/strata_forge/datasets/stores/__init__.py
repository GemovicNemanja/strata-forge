"""Backend implementations for :class:`strata_forge.datasets.DatasetStore`.

:class:`InMemoryDatasetStore` is dict-backed and the default for
tests, notebooks, and prototyping. :class:`LangfuseDatasetStore` is
the Langfuse-backed production store, lazy-importing the
``[langfuse]`` extra inside its constructor.
"""

from strata_forge.datasets.stores.langfuse import LangfuseDatasetStore
from strata_forge.datasets.stores.memory import InMemoryDatasetStore

__all__ = [
    "InMemoryDatasetStore",
    "LangfuseDatasetStore",
]
