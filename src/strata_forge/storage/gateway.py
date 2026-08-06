"""Async file-storage gateway backed by `fsspec`.

:class:`StorageGateway` is a small async wrapper around the
synchronous ``fsspec`` API. Every method off-loads the actual I/O
to a thread (``asyncio.to_thread``) so the public surface stays
async-uniform with the rest of Forge.

Supported targets are whatever ``fsspec`` understands — the
``[storage]`` extra brings the common cloud filesystems
(``s3fs``, ``gcsfs``, ``adlfs``) and ``huggingface_hub``; ``file``,
``memory``, and ``http`` are built into fsspec itself.

The class auto-detects the protocol from each URL, so callers can
freely mix targets in a single program::

    gw = StorageGateway()
    text = await gw.read_text("./local/file.txt")
    blob = await gw.read_bytes("s3://my-bucket/data.parquet")
    await gw.write_text("gs://staging/manifest.json", json.dumps(...))

Per-protocol options (auth, region, anonymity) flow through the
constructor's ``options=`` mapping. Each top-level key is a
protocol name and each value is forwarded verbatim to
``fsspec.filesystem(protocol, **value)``::

    gw = StorageGateway(options={
        "s3": {"key": "...", "secret": "...", "client_kwargs": {"region_name": "us-east-1"}},
        "gs": {"token": "/path/to/keyfile.json"},
    })

The fsspec module itself is imported lazily inside
:meth:`StorageGateway._filesystem`, so importing
:mod:`strata_forge.storage.gateway` works without the ``[storage]``
extra installed — the :class:`ImportError` surfaces only on first
filesystem use.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["FileInfo", "StorageGateway"]


type FileInfo = dict[str, Any]
"""Per-entry fsspec metadata: at least ``name`` and ``size``; other
keys (``type``, ``mtime``, etc.) are protocol-dependent."""


_DEFAULT_PROTOCOL = "file"


def _split_protocol(url: str) -> tuple[str, str]:
    """Return ``(protocol, path)`` for ``url``.

    Bare paths (``./foo``, ``/tmp/bar``) get the ``file`` protocol.
    Multi-letter ``s3://``-style URLs return the scheme; single-letter
    Windows drive paths (``C:\\foo``) are treated as ``file``.
    """
    parsed = urlparse(url)
    if not parsed.scheme or len(parsed.scheme) == 1:
        return _DEFAULT_PROTOCOL, url
    return parsed.scheme, url


class StorageGateway:
    """Async gateway for read/write/list/copy across any fsspec target.

    Args:
        options: Mapping of ``protocol -> kwargs`` forwarded to
            ``fsspec.filesystem``. Use this to pass credentials,
            regions, or any other protocol-specific knobs.
    """

    def __init__(
        self,
        *,
        options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._options: dict[str, dict[str, Any]] = {
            protocol: dict(opts) for protocol, opts in (options or {}).items()
        }
        # Cache one filesystem per protocol; instances are reusable.
        self._cache: dict[str, Any] = {}

    def _import_fsspec(self) -> Any:
        try:
            return __import__("fsspec")
        except ImportError as exc:
            msg = (
                "The [storage] extra is required for StorageGateway. "
                "Install it with: pip install 'strata-forge[storage]'."
            )
            raise ImportError(msg) from exc

    def _filesystem(self, protocol: str) -> Any:
        if protocol in self._cache:
            return self._cache[protocol]
        fsspec_mod = self._import_fsspec()
        kwargs = self._options.get(protocol, {})
        fs: Any = fsspec_mod.filesystem(protocol, **kwargs)
        self._cache[protocol] = fs
        return fs

    def _fs_for(self, url: str) -> tuple[Any, str]:
        protocol, _ = _split_protocol(url)
        return self._filesystem(protocol), url

    async def read_bytes(self, url: str) -> bytes:
        """Read the entire contents of ``url`` as bytes."""
        fs, target = self._fs_for(url)

        def _read() -> bytes:
            with fs.open(target, "rb") as handle:
                data: Any = handle.read()
                return bytes(data)

        return await asyncio.to_thread(_read)

    async def read_text(self, url: str, *, encoding: str = "utf-8") -> str:
        """Read the entire contents of ``url`` as decoded text."""
        data = await self.read_bytes(url)
        return data.decode(encoding)

    async def write_bytes(self, url: str, data: bytes) -> None:
        """Overwrite ``url`` with ``data``."""
        fs, target = self._fs_for(url)

        def _write() -> None:
            with fs.open(target, "wb") as handle:
                handle.write(data)

        await asyncio.to_thread(_write)

    async def write_text(self, url: str, text: str, *, encoding: str = "utf-8") -> None:
        """Overwrite ``url`` with ``text`` encoded as ``encoding``."""
        await self.write_bytes(url, text.encode(encoding))

    async def exists(self, url: str) -> bool:
        """Return whether ``url`` exists."""
        fs, target = self._fs_for(url)
        return bool(await asyncio.to_thread(fs.exists, target))

    async def info(self, url: str) -> FileInfo:
        """Return per-file metadata (size, type, mtime, …)."""
        fs, target = self._fs_for(url)
        raw: Any = await asyncio.to_thread(fs.info, target)
        if isinstance(raw, dict):
            return dict(raw)  # type: ignore[arg-type]
        err = f"unexpected info() result type {type(raw).__name__!r}"
        raise TypeError(err)

    async def ls(self, url: str, *, detail: bool = False) -> list[str] | list[FileInfo]:
        """List entries under ``url``.

        When ``detail=False`` (default), returns names as strings.
        When ``detail=True``, returns one :data:`FileInfo` per entry.
        """
        fs, target = self._fs_for(url)
        raw: Any = await asyncio.to_thread(fs.ls, target, detail=detail)
        if detail:
            return [dict(item) for item in raw]
        return [str(item) for item in raw]

    async def delete(self, url: str, *, recursive: bool = False) -> None:
        """Delete ``url``.

        Args:
            url: The target to remove.
            recursive: When ``True``, removes a directory and its
                contents. When ``False``, deleting a non-empty
                directory raises a protocol-specific error.
        """
        fs, target = self._fs_for(url)
        await asyncio.to_thread(fs.rm, target, recursive=recursive)

    async def copy(self, src: str, dst: str, *, recursive: bool = False) -> None:
        """Copy ``src`` to ``dst``.

        Same-protocol copies use the filesystem's native ``copy``.
        Cross-protocol copies stream bytes through the gateway —
        OK for small files, but you'll want a smarter transfer
        path (multipart uploads, signed URLs) for large datasets.
        """
        src_protocol, _ = _split_protocol(src)
        dst_protocol, _ = _split_protocol(dst)
        if src_protocol == dst_protocol:
            fs = self._filesystem(src_protocol)
            await asyncio.to_thread(fs.copy, src, dst, recursive=recursive)
            return
        if recursive:
            err = "cross-protocol recursive copy is not supported"
            raise NotImplementedError(err)
        data = await self.read_bytes(src)
        await self.write_bytes(dst, data)

    async def move(self, src: str, dst: str) -> None:
        """Move ``src`` to ``dst``.

        Same-protocol moves use the filesystem's native ``move``.
        Cross-protocol moves copy then delete the source.
        """
        src_protocol, _ = _split_protocol(src)
        dst_protocol, _ = _split_protocol(dst)
        if src_protocol == dst_protocol:
            fs = self._filesystem(src_protocol)
            await asyncio.to_thread(fs.move, src, dst)
            return
        await self.copy(src, dst)
        await self.delete(src)

    async def makedirs(self, url: str, *, exist_ok: bool = True) -> None:
        """Create ``url`` (and any missing parents)."""
        fs, target = self._fs_for(url)
        await asyncio.to_thread(fs.makedirs, target, exist_ok=exist_ok)
