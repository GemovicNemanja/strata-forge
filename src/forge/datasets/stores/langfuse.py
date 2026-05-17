"""Langfuse-backed :class:`DatasetStore`.

Forge versions are encoded into the Langfuse dataset name as
``{forge_name}__v{version}`` so each Forge version is a distinct
Langfuse dataset — visible in the Langfuse UI, addressable via the
Langfuse SDK without out-of-band bookkeeping, and naturally
deduplicated (re-putting identical content collides on the composed
name and the existing dataset is reused).

The Langfuse Python SDK is sync; this module wraps every call in
``asyncio.to_thread`` so the public surface stays async-only per the
:mod:`forge` architectural rule.

The ``langfuse`` package is imported lazily inside the client
constructor so importing :mod:`forge.datasets.stores.langfuse` works
without the ``[langfuse]`` extra installed. The import raises only
when a user actually constructs the store without the extra.
"""

from __future__ import annotations

import asyncio
from typing import Any

from forge.datasets.schema import Dataset, DatasetItem
from forge.datasets.store import DatasetNotFoundError, DatasetStore
from forge.datasets.versioning import dataset_version

__all__ = [
    "VERSION_SEPARATOR",
    "LangfuseDatasetStore",
    "compose_langfuse_name",
    "decompose_langfuse_name",
]

VERSION_SEPARATOR = "__v"


def compose_langfuse_name(forge_name: str, version: str) -> str:
    """Compose the Langfuse dataset name for ``(forge_name, version)``."""
    return f"{forge_name}{VERSION_SEPARATOR}{version}"


def decompose_langfuse_name(langfuse_name: str) -> tuple[str, str] | None:
    """Split a Langfuse dataset name into ``(forge_name, version)`` if it matches.

    Returns ``None`` for names that don't carry the ``__v`` separator —
    those are non-Forge datasets and the store ignores them.
    """
    idx = langfuse_name.rfind(VERSION_SEPARATOR)
    if idx < 0:
        return None
    return langfuse_name[:idx], langfuse_name[idx + len(VERSION_SEPARATOR) :]


class LangfuseDatasetStore(DatasetStore):
    """Persist Forge datasets in Langfuse.

    Versions are encoded in the Langfuse dataset name; one Langfuse
    dataset corresponds to exactly one ``(forge_name, version)`` pair.

    ``client`` can be passed explicitly (handy for tests and for
    callers who already have a Langfuse client they want to share);
    when omitted, the store builds one from
    :class:`forge.config.LangfuseConfig` on first use.
    """

    def __init__(self, *, client: Any | None = None) -> None:
        self._explicit_client = client
        self._cached_client: Any | None = None

    def _get_client(self) -> Any:
        if self._explicit_client is not None:
            return self._explicit_client
        if self._cached_client is not None:
            return self._cached_client
        self._cached_client = self._build_client()
        return self._cached_client

    @staticmethod
    def _build_client() -> Any:
        from forge.config import get_settings

        config = get_settings().langfuse
        if not config.enabled:
            msg = (
                "Langfuse is not configured. Set LANGFUSE_HOST, "
                "LANGFUSE_PUBLIC_KEY, and LANGFUSE_SECRET_KEY (or pass "
                "client=... explicitly)."
            )
            raise RuntimeError(msg)
        try:
            from langfuse import (  # pyright: ignore[reportMissingImports]
                Langfuse,  # pyright: ignore[reportUnknownVariableType]
            )
        except ImportError as exc:
            msg = (
                "The [langfuse] extra is required for LangfuseDatasetStore. "
                "Install it with: pip install 'ai-forge[langfuse]'."
            )
            raise ImportError(msg) from exc
        if config.public_key is None or config.secret_key is None:  # pragma: no cover
            msg = "Langfuse keys missing despite enabled=True"
            raise RuntimeError(msg)
        return Langfuse(  # pyright: ignore[reportUnknownVariableType]
            host=config.host,
            public_key=config.public_key.get_secret_value(),
            secret_key=config.secret_key.get_secret_value(),
        )

    async def _list_langfuse_datasets(self) -> list[Any]:
        """Return every Langfuse dataset object (across pagination)."""
        client = self._get_client()

        def _fetch() -> list[Any]:
            collected: list[Any] = []
            page = 1
            while True:
                resp = client.api.datasets.list(page=page, limit=100)
                data = list(getattr(resp, "data", []) or [])
                collected.extend(data)
                meta = getattr(resp, "meta", None)
                total_pages = getattr(meta, "total_pages", None) if meta else None
                if total_pages is None or page >= total_pages:
                    break
                page += 1
            return collected

        return await asyncio.to_thread(_fetch)

    async def _list_versions_newest_first(self, name: str) -> list[str]:
        datasets = await self._list_langfuse_datasets()
        pairs: list[tuple[str, Any]] = []
        for ds in datasets:
            decomposed = decompose_langfuse_name(getattr(ds, "name", ""))
            if decomposed is None:
                continue
            forge_name, version = decomposed
            if forge_name == name:
                pairs.append((version, ds))
        pairs.sort(
            key=lambda pair: getattr(pair[1], "created_at", None) or "",
            reverse=True,
        )
        return [version for version, _ in pairs]

    @staticmethod
    def _to_forge_item(raw_item: Any) -> DatasetItem:
        return DatasetItem(
            id=str(raw_item.id),
            input=dict(raw_item.input or {}),
            expected_output=raw_item.expected_output,
            metadata=dict(getattr(raw_item, "metadata", None) or {}),
        )

    def _to_forge_dataset(self, name: str, raw_dataset: Any) -> Dataset:
        raw_items = list(getattr(raw_dataset, "items", []) or [])
        items = tuple(self._to_forge_item(it) for it in raw_items)
        return Dataset(
            name=name,
            items=items,
            description=getattr(raw_dataset, "description", "") or "",
            metadata=dict(getattr(raw_dataset, "metadata", None) or {}),
        )

    async def get(self, name: str, version: str | None = None) -> Dataset:
        if version is None:
            versions = await self._list_versions_newest_first(name)
            if not versions:
                msg = f"Dataset {name!r} not in Langfuse"
                raise DatasetNotFoundError(msg, name=name)
            version = versions[0]
        client = self._get_client()
        composed = compose_langfuse_name(name, version)

        def _fetch() -> Any:
            return client.get_dataset(name=composed)

        try:
            raw = await asyncio.to_thread(_fetch)
        except Exception as exc:
            msg = f"Version {version!r} of dataset {name!r} not in Langfuse"
            raise DatasetNotFoundError(msg, name=name, version=version) from exc
        return self._to_forge_dataset(name, raw)

    async def put(self, dataset: Dataset) -> str:
        version = dataset_version(dataset)
        client = self._get_client()
        composed = compose_langfuse_name(dataset.name, version)

        # If the composed-name dataset already exists in Langfuse,
        # treat the put as a no-op — content-hash versioning guarantees
        # identical content, so re-uploading is wasted work.
        existing_versions = await self._list_versions_newest_first(dataset.name)
        if version in existing_versions:
            return version

        def _create() -> None:
            client.create_dataset(
                name=composed,
                description=dataset.description or None,
                metadata=dict(dataset.metadata) if dataset.metadata else None,
            )
            for item in dataset.items:
                client.create_dataset_item(
                    dataset_name=composed,
                    id=item.id,
                    input=dict(item.input),
                    expected_output=item.expected_output,
                    metadata=dict(item.metadata) if item.metadata else None,
                )

        await asyncio.to_thread(_create)
        return version

    async def versions(self, name: str) -> list[str]:
        versions = await self._list_versions_newest_first(name)
        if not versions:
            msg = f"Dataset {name!r} not in Langfuse"
            raise DatasetNotFoundError(msg, name=name)
        return versions

    async def list_names(self) -> list[str]:
        datasets = await self._list_langfuse_datasets()
        names: set[str] = set()
        for ds in datasets:
            decomposed = decompose_langfuse_name(getattr(ds, "name", ""))
            if decomposed is not None:
                names.add(decomposed[0])
        return sorted(names)

    async def delete(self, name: str, version: str | None = None) -> None:
        existing = await self._list_versions_newest_first(name)
        if not existing:
            return
        if version is None:
            targets = list(existing)
        else:
            if version not in existing:
                msg = f"Version {version!r} of dataset {name!r} not in Langfuse"
                raise DatasetNotFoundError(msg, name=name, version=version)
            targets = [version]
        client = self._get_client()

        def _delete_all() -> None:
            for v in targets:
                client.api.datasets.delete(dataset_name=compose_langfuse_name(name, v))

        await asyncio.to_thread(_delete_all)
