"""In-process :class:`DatasetStore` — dict-backed, content-hash versioning.

Use this for tests, notebooks, and prototyping. Versions are derived
from the dataset's canonical content via
:func:`strata_forge.datasets.versioning.dataset_version`, so two puts with
identical content return the same version string (natural dedup) and
content-divergent puts always produce distinct versions.

"Latest" is whichever version was put most recently. Versions are
listed newest-first via the insertion order of the underlying dict
(Python 3.7+ guarantee).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strata_forge.datasets.store import DatasetNotFoundError, DatasetStore
from strata_forge.datasets.versioning import dataset_version

if TYPE_CHECKING:
    from strata_forge.datasets.schema import Dataset

__all__ = [
    "InMemoryDatasetStore",
]


class InMemoryDatasetStore(DatasetStore):
    """Dict-backed dataset store.

    State is a ``name -> dict[version, Dataset]`` map. The inner dict
    preserves insertion order, so "latest" is just the last-inserted
    version and ``versions()`` is its keys reversed.

    Operations are atomic per asyncio scheduling quantum — no internal
    awaits during state mutation — so concurrent puts/gets from the
    same loop won't tear state. For cross-process safety, use the
    Langfuse-backed store in :mod:`strata_forge.datasets.stores.langfuse`.
    """

    def __init__(self) -> None:
        self._versions: dict[str, dict[str, Dataset]] = {}

    async def get(self, name: str, version: str | None = None) -> Dataset:
        versions_map = self._versions.get(name)
        if versions_map is None:
            msg = f"Dataset {name!r} not in store"
            raise DatasetNotFoundError(msg, name=name, version=version)
        if version is None:
            # Last-inserted is the latest.
            return next(reversed(versions_map.values()))
        if version not in versions_map:
            msg = f"Version {version!r} of dataset {name!r} not in store"
            raise DatasetNotFoundError(msg, name=name, version=version)
        return versions_map[version]

    async def put(self, dataset: Dataset) -> str:
        version = dataset_version(dataset)
        self._versions.setdefault(dataset.name, {})[version] = dataset
        return version

    async def versions(self, name: str) -> list[str]:
        versions_map = self._versions.get(name)
        if versions_map is None:
            msg = f"Dataset {name!r} not in store"
            raise DatasetNotFoundError(msg, name=name)
        return list(reversed(versions_map.keys()))

    async def list_names(self) -> list[str]:
        return sorted(self._versions)

    async def delete(self, name: str, version: str | None = None) -> None:
        versions_map = self._versions.get(name)
        if versions_map is None:
            # Unknown name: no-op per DatasetStore.delete contract.
            return
        if version is None:
            del self._versions[name]
            return
        if version not in versions_map:
            msg = f"Version {version!r} of dataset {name!r} not in store"
            raise DatasetNotFoundError(msg, name=name, version=version)
        del versions_map[version]
        if not versions_map:
            del self._versions[name]
