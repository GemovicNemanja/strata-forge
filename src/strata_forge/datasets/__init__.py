"""Typed dataset shapes, versioning, and pluggable backends.

The public surface is everything the eval runner (Phase 2.4) and
downstream user code consume. Backends in :mod:`strata_forge.datasets.stores`
and the HF bridge in :mod:`strata_forge.datasets.hf_bridge` lazy-import their
SDKs so :mod:`strata_forge.datasets` is importable without the
``[langfuse]`` or ``[hf]`` extras installed.
"""

from typing import TYPE_CHECKING

from strata_forge.datasets.hf_bridge import from_hf_dataset, to_hf_dataset
from strata_forge.datasets.schema import Dataset, DatasetItem
from strata_forge.datasets.store import DatasetNotFoundError, DatasetStore
from strata_forge.datasets.stores import InMemoryDatasetStore, LangfuseDatasetStore
from strata_forge.datasets.synthetic import distill, self_instruct
from strata_forge.datasets.versioning import DatasetDelta, dataset_version, diff

if TYPE_CHECKING:
    from strata_forge.datasets.dataset_execution import DatasetResult as DatasetResult
    from strata_forge.datasets.dataset_execution import LocalDatasetExecutor as LocalDatasetExecutor
    from strata_forge.datasets.dataset_execution_types import (
        DatasetConvertSpec as DatasetConvertSpec,
    )
    from strata_forge.datasets.dataset_execution_types import (
        DatasetExecutionError as DatasetExecutionError,
    )
    from strata_forge.datasets.dataset_execution_types import (
        DatasetInputManifest as DatasetInputManifest,
    )
    from strata_forge.datasets.dataset_execution_types import (
        DatasetInstallation as DatasetInstallation,
    )
    from strata_forge.datasets.dataset_execution_types import DatasetQuerySpec as DatasetQuerySpec
    from strata_forge.datasets.dataset_execution_types import NamedParquetFiles as NamedParquetFiles
    from strata_forge.datasets.dataset_execution_types import (
        ProvidedDatasetFiles as ProvidedDatasetFiles,
    )
    from strata_forge.datasets.sql import LocalSQLExecutor as LocalSQLExecutor
    from strata_forge.datasets.sql import SQLResult as SQLResult
    from strata_forge.datasets.sql_types import LocalParquetFiles as LocalParquetFiles
    from strata_forge.datasets.sql_types import LocalSQLInstallation as LocalSQLInstallation
    from strata_forge.datasets.sql_types import SQLExecutionError as SQLExecutionError
    from strata_forge.datasets.sql_types import SQLInputManifest as SQLInputManifest
    from strata_forge.datasets.sql_types import SQLQuerySpec as SQLQuerySpec

__all__ = [
    "Dataset",
    "DatasetConvertSpec",
    "DatasetDelta",
    "DatasetExecutionError",
    "DatasetInputManifest",
    "DatasetInstallation",
    "DatasetItem",
    "DatasetNotFoundError",
    "DatasetQuerySpec",
    "DatasetResult",
    "DatasetStore",
    "InMemoryDatasetStore",
    "LangfuseDatasetStore",
    "LocalDatasetExecutor",
    "LocalParquetFiles",
    "LocalSQLExecutor",
    "LocalSQLInstallation",
    "NamedParquetFiles",
    "ProvidedDatasetFiles",
    "SQLExecutionError",
    "SQLInputManifest",
    "SQLQuerySpec",
    "SQLResult",
    "dataset_version",
    "diff",
    "distill",
    "from_hf_dataset",
    "self_instruct",
    "to_hf_dataset",
]


def __getattr__(name: str) -> object:
    if type(name) is str and name in {"LocalDatasetExecutor", "DatasetResult"}:
        from strata_forge.datasets import dataset_execution

        return getattr(dataset_execution, name)
    if type(name) is str and name in {
        "DatasetConvertSpec",
        "DatasetExecutionError",
        "DatasetInputManifest",
        "DatasetInstallation",
        "DatasetQuerySpec",
        "NamedParquetFiles",
        "ProvidedDatasetFiles",
    }:
        from strata_forge.datasets import dataset_execution_types

        return getattr(dataset_execution_types, name)

    if type(name) is str and name in {"LocalSQLExecutor", "SQLResult"}:
        from strata_forge.datasets import sql

        return getattr(sql, name)
    if type(name) is str and name in {
        "LocalParquetFiles",
        "LocalSQLInstallation",
        "SQLExecutionError",
        "SQLInputManifest",
        "SQLQuerySpec",
    }:
        from strata_forge.datasets import sql_types

        return getattr(sql_types, name)
    raise AttributeError("unknown dataset attribute")
