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
constructor so importing :mod:`strata_forge.datasets.stores.langfuse` works
without the ``[langfuse]`` extra installed. The import raises only
when a user actually constructs the store without the extra.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from strata_forge.datasets.schema import Dataset, DatasetItem
from strata_forge.datasets.store import DatasetNotFoundError, DatasetStore
from strata_forge.datasets.versioning import dataset_version

__all__ = [
    "VERSION_SEPARATOR",
    "LangfuseDatasetStore",
    "compose_langfuse_name",
    "decompose_langfuse_name",
]

VERSION_SEPARATOR = "__v"


@dataclass(frozen=True)
class _DatasetSummary:
    """Lightweight stand-in for the SDK dataset object the REST list
    endpoint replaces. Only the fields this module reads are kept."""

    name: str
    created_at: str


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
    :class:`strata_forge.config.LangfuseConfig` on first use.
    """

    def __init__(self, *, client: Any | None = None) -> None:
        self._explicit_client = client
        self._cached_client: Any | None = None
        # Captured during _build_client so list/delete can hit the
        # Langfuse REST API directly — v4 dropped the `client.api`
        # proxy that earlier SDKs exposed for dataset listing.
        self._host: str | None = None
        self._public_key: str | None = None
        self._secret_key: str | None = None

    def _get_client(self) -> Any:
        if self._explicit_client is not None:
            return self._explicit_client
        if self._cached_client is not None:
            return self._cached_client
        self._cached_client = self._build_client()
        return self._cached_client

    def _build_client(self) -> Any:
        from strata_forge.config import get_settings

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
                "Install it with: pip install 'strata-forge[langfuse]'."
            )
            raise ImportError(msg) from exc
        if config.public_key is None or config.secret_key is None:  # pragma: no cover
            msg = "Langfuse keys missing despite enabled=True"
            raise RuntimeError(msg)
        self._host = config.host
        self._public_key = config.public_key.get_secret_value()
        self._secret_key = config.secret_key.get_secret_value()
        return Langfuse(  # pyright: ignore[reportUnknownVariableType]
            host=self._host,
            public_key=self._public_key,
            secret_key=self._secret_key,
        )

    def _rest_credentials(self) -> tuple[str, str, str]:
        """Resolve (host, public_key, secret_key) for REST calls.

        Falls back to env vars when the store was constructed with an
        explicit client and the credentials weren't captured.
        """
        import os

        # Trigger client build (and credential capture) when needed.
        if self._explicit_client is None and self._cached_client is None:
            self._get_client()
        host = self._host or os.environ.get("LANGFUSE_HOST")
        public_key = self._public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = self._secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        if not host or not public_key or not secret_key:
            msg = (
                "LangfuseDatasetStore needs host + keys for list/delete; "
                "set LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
                "or construct without an explicit client so they are loaded "
                "from forge settings."
            )
            raise RuntimeError(msg)
        return host, public_key, secret_key

    async def _list_langfuse_datasets(self) -> list[Any]:
        """Return every Langfuse dataset name across pagination.

        v4 dropped ``client.api.datasets.list``; we use the REST
        endpoint directly. Returned items are minimal dicts with the
        fields this module reads (``name``, ``createdAt``) — the older
        in-tree code used SDK objects with ``getattr``, so a simple
        dict shape is compatible.
        """
        import httpx

        host, public_key, secret_key = self._rest_credentials()
        collected: list[Any] = []
        page = 1
        async with httpx.AsyncClient(timeout=15.0) as http:
            while True:
                resp = await http.get(
                    f"{host.rstrip('/')}/api/public/v2/datasets",
                    auth=(public_key, secret_key),
                    params={"page": page, "limit": 100},
                )
                resp.raise_for_status()
                payload = resp.json()
                data: Any = payload.get("data") or []
                for item in data:
                    collected.append(
                        _DatasetSummary(
                            name=str(item.get("name", "")),
                            created_at=str(item.get("createdAt") or ""),
                        )
                    )
                meta: Any = payload.get("meta") or {}
                total_pages: Any = meta.get("totalPages", 1)
                if page >= int(total_pages):
                    break
                page += 1
        return collected

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
        import httpx

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
        host, public_key, secret_key = self._rest_credentials()
        async with httpx.AsyncClient(timeout=15.0) as http:
            for v in targets:
                composed = compose_langfuse_name(name, v)
                resp = await http.delete(
                    f"{host.rstrip('/')}/api/public/v2/datasets/{composed}",
                    auth=(public_key, secret_key),
                )
                # Treat 404 as already gone; otherwise raise.
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
