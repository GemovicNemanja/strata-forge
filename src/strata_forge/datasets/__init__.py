"""Typed dataset shapes, versioning, and pluggable backends.

The public surface is everything the eval runner and
downstream user code consume. Backends in :mod:`strata_forge.datasets.stores`
and the HF bridge in :mod:`strata_forge.datasets.hf_bridge` lazy-import their
SDKs so :mod:`strata_forge.datasets` is importable without the
``[langfuse]`` or ``[hf]`` extras installed.
"""

from strata_forge.datasets.hf_bridge import from_hf_dataset, to_hf_dataset
from strata_forge.datasets.schema import Dataset, DatasetItem
from strata_forge.datasets.store import DatasetNotFoundError, DatasetStore
from strata_forge.datasets.stores import InMemoryDatasetStore, LangfuseDatasetStore
from strata_forge.datasets.synthetic import distill, self_instruct
from strata_forge.datasets.versioning import DatasetDelta, dataset_version, diff

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
