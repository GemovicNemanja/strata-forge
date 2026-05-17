"""Typed dataset shapes, versioning, and pluggable backends.

The public surface is everything the eval runner (Phase 2.4) and
downstream user code consume. Backends in :mod:`forge.datasets.stores`
and the HF bridge in :mod:`forge.datasets.hf_bridge` lazy-import their
SDKs so :mod:`forge.datasets` is importable without the
``[langfuse]`` or ``[hf]`` extras installed.
"""

from forge.datasets.hf_bridge import from_hf_dataset, to_hf_dataset
from forge.datasets.schema import Dataset, DatasetItem
from forge.datasets.store import DatasetNotFoundError, DatasetStore
from forge.datasets.stores import InMemoryDatasetStore, LangfuseDatasetStore
from forge.datasets.synthetic import distill, self_instruct
from forge.datasets.versioning import DatasetDelta, dataset_version, diff

__all__ = [
    "Dataset",
    "DatasetDelta",
    "DatasetItem",
    "DatasetNotFoundError",
    "DatasetStore",
    "InMemoryDatasetStore",
    "LangfuseDatasetStore",
    "dataset_version",
    "diff",
    "distill",
    "from_hf_dataset",
    "self_instruct",
    "to_hf_dataset",
]
