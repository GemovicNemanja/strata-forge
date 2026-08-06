"""Unit tests for `strata_forge.storage.gateway.StorageGateway`."""

from __future__ import annotations

import io
import sys
import types
from typing import Any

import pytest

from strata_forge.storage import FileInfo, StorageGateway
from strata_forge.storage.gateway import _split_protocol  # pyright: ignore[reportPrivateUsage]


class TestSplitProtocol:
    def test_local_path(self) -> None:
        assert _split_protocol("./foo/bar.txt") == ("file", "./foo/bar.txt")

    def test_absolute_path(self) -> None:
        assert _split_protocol("./scratch/x.bin") == ("file", "./scratch/x.bin")

    def test_s3(self) -> None:
        assert _split_protocol("s3://bucket/key") == ("s3", "s3://bucket/key")

    def test_gs(self) -> None:
        assert _split_protocol("gs://bucket/key") == ("gs", "gs://bucket/key")

    def test_az(self) -> None:
        assert _split_protocol("az://container/blob") == ("az", "az://container/blob")

    def test_http(self) -> None:
        assert _split_protocol("https://example.com/x") == (
            "https",
            "https://example.com/x",
        )

    def test_windows_drive_treated_as_file(self) -> None:
        # Single-letter scheme is a Windows drive, not a protocol.
        assert _split_protocol("C:/Users/x.txt") == ("file", "C:/Users/x.txt")


# ---------------------------------------------------------------------------
# Fake fsspec — captures every call
# ---------------------------------------------------------------------------


class _FakeFile:
    def __init__(self, target: str, mode: str, store: dict[str, bytes]) -> None:
        self.target = target
        self.mode = mode
        self.store = store
        self._buffer: io.BytesIO
        if "r" in mode:
            self._buffer = io.BytesIO(store.get(target, b""))
        else:
            self._buffer = io.BytesIO()

    def __enter__(self) -> _FakeFile:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if "w" in self.mode:
            self.store[self.target] = self._buffer.getvalue()

    def read(self) -> bytes:
        return self._buffer.read()

    def write(self, data: bytes) -> int:
        return self._buffer.write(data)


class _FakeFilesystem:
    def __init__(self, protocol: str, **kwargs: Any) -> None:
        self.protocol = protocol
        self.kwargs = kwargs
        self.store: dict[str, bytes] = {}
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.calls.append((name, args, kwargs))

    def open(self, target: str, mode: str = "rb") -> _FakeFile:
        self._record("open", (target,), {"mode": mode})
        return _FakeFile(target, mode, self.store)

    def exists(self, target: str) -> bool:
        self._record("exists", (target,), {})
        return target in self.store

    def info(self, target: str) -> dict[str, Any]:
        self._record("info", (target,), {})
        return {"name": target, "size": len(self.store.get(target, b""))}

    def ls(self, target: str, *, detail: bool = False) -> list[Any]:
        self._record("ls", (target,), {"detail": detail})
        entries = [k for k in self.store if k.startswith(target.rstrip("/"))]
        if detail:
            return [{"name": e, "size": len(self.store[e])} for e in entries]
        return entries

    def rm(self, target: str, *, recursive: bool = False) -> None:
        self._record("rm", (target,), {"recursive": recursive})
        if recursive:
            for k in list(self.store):
                if k.startswith(target):
                    del self.store[k]
        else:
            self.store.pop(target, None)

    def copy(self, src: str, dst: str, *, recursive: bool = False) -> None:
        self._record("copy", (src, dst), {"recursive": recursive})
        if recursive:
            for k in list(self.store):
                if k.startswith(src):
                    self.store[k.replace(src, dst, 1)] = self.store[k]
        elif src in self.store:
            self.store[dst] = self.store[src]

    def move(self, src: str, dst: str) -> None:
        self._record("move", (src, dst), {})
        if src in self.store:
            self.store[dst] = self.store.pop(src)

    def makedirs(self, target: str, *, exist_ok: bool = True) -> None:
        self._record("makedirs", (target,), {"exist_ok": exist_ok})


@pytest.fixture
def fake_fsspec(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeFilesystem]:
    created: dict[str, _FakeFilesystem] = {}

    def _filesystem(protocol: str, **kwargs: Any) -> _FakeFilesystem:
        # Cache so cross-protocol tests share state.
        fs = created.get(protocol)
        if fs is None:
            fs = _FakeFilesystem(protocol, **kwargs)
            created[protocol] = fs
        return fs

    fake = types.ModuleType("fsspec")
    fake.filesystem = _filesystem  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fsspec", fake)
    return created


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


class TestReadWrite:
    async def test_round_trip_bytes(self, fake_fsspec: dict[str, _FakeFilesystem]) -> None:
        gw = StorageGateway()
        await gw.write_bytes("./scratch/x.bin", b"hello")
        assert await gw.read_bytes("./scratch/x.bin") == b"hello"

    async def test_round_trip_text(self, fake_fsspec: dict[str, _FakeFilesystem]) -> None:
        gw = StorageGateway()
        await gw.write_text("./report.txt", "hi there")
        assert await gw.read_text("./report.txt") == "hi there"

    async def test_text_encoding(self, fake_fsspec: dict[str, _FakeFilesystem]) -> None:
        gw = StorageGateway()
        await gw.write_text("./latin.txt", "café", encoding="latin-1")
        # round-trip with the matching encoding
        assert await gw.read_text("./latin.txt", encoding="latin-1") == "café"


# ---------------------------------------------------------------------------
# exists / info / ls
# ---------------------------------------------------------------------------


class TestQuery:
    async def test_exists_false_then_true(self, fake_fsspec: dict[str, _FakeFilesystem]) -> None:
        gw = StorageGateway()
        assert await gw.exists("./scratch/a") is False
        await gw.write_bytes("./scratch/a", b"x")
        assert await gw.exists("./scratch/a") is True

    async def test_info_returns_dict(self, fake_fsspec: dict[str, _FakeFilesystem]) -> None:
        gw = StorageGateway()
        await gw.write_bytes("./scratch/file", b"abcd")
        info: FileInfo = await gw.info("./scratch/file")
        assert info["name"] == "./scratch/file"
        assert info["size"] == 4

    async def test_ls_names(self, fake_fsspec: dict[str, _FakeFilesystem]) -> None:
        gw = StorageGateway()
        await gw.write_bytes("/d/a", b"1")
        await gw.write_bytes("/d/b", b"22")
        names = await gw.ls("/d")
        assert set(names) == {"/d/a", "/d/b"}

    async def test_ls_detail(self, fake_fsspec: dict[str, _FakeFilesystem]) -> None:
        gw = StorageGateway()
        await gw.write_bytes("/dd/a", b"1")
        details = await gw.ls("/dd", detail=True)
        first: Any = details[0]
        assert first["size"] == 1


# ---------------------------------------------------------------------------
# delete / copy / move / makedirs
# ---------------------------------------------------------------------------


class TestMutation:
    async def test_delete(self, fake_fsspec: dict[str, _FakeFilesystem]) -> None:
        gw = StorageGateway()
        await gw.write_bytes("/x", b"data")
        await gw.delete("/x")
        assert await gw.exists("/x") is False

    async def test_delete_recursive(self, fake_fsspec: dict[str, _FakeFilesystem]) -> None:
        gw = StorageGateway()
        await gw.write_bytes("/d/a", b"1")
        await gw.write_bytes("/d/b", b"2")
        await gw.delete("/d", recursive=True)
        assert await gw.exists("/d/a") is False
        assert await gw.exists("/d/b") is False

    async def test_same_protocol_copy(self, fake_fsspec: dict[str, _FakeFilesystem]) -> None:
        gw = StorageGateway()
        await gw.write_bytes("/a", b"copy-me")
        await gw.copy("/a", "/b")
        assert await gw.read_bytes("/b") == b"copy-me"

    async def test_cross_protocol_copy_streams_bytes(
        self, fake_fsspec: dict[str, _FakeFilesystem]
    ) -> None:
        gw = StorageGateway()
        await gw.write_bytes("/local", b"contents")
        await gw.copy("/local", "s3://bucket/key")
        # Routed through the s3 filesystem.
        assert await gw.read_bytes("s3://bucket/key") == b"contents"

    async def test_cross_protocol_recursive_rejected(
        self, fake_fsspec: dict[str, _FakeFilesystem]
    ) -> None:
        gw = StorageGateway()
        with pytest.raises(NotImplementedError, match="cross-protocol"):
            await gw.copy("/a", "s3://b/c", recursive=True)

    async def test_same_protocol_move(self, fake_fsspec: dict[str, _FakeFilesystem]) -> None:
        gw = StorageGateway()
        await gw.write_bytes("/m", b"data")
        await gw.move("/m", "/n")
        assert await gw.read_bytes("/n") == b"data"
        assert await gw.exists("/m") is False

    async def test_cross_protocol_move_copies_then_deletes(
        self, fake_fsspec: dict[str, _FakeFilesystem]
    ) -> None:
        gw = StorageGateway()
        await gw.write_bytes("/m", b"data")
        await gw.move("/m", "s3://b/k")
        assert await gw.read_bytes("s3://b/k") == b"data"
        assert await gw.exists("/m") is False

    async def test_makedirs(self, fake_fsspec: dict[str, _FakeFilesystem]) -> None:
        gw = StorageGateway()
        await gw.makedirs("/some/dir")
        # The recorded call exists on the file-protocol fs.
        assert any(c[0] == "makedirs" for c in fake_fsspec["file"].calls)


# ---------------------------------------------------------------------------
# Options + caching
# ---------------------------------------------------------------------------


class TestOptions:
    async def test_per_protocol_options_forwarded(
        self, fake_fsspec: dict[str, _FakeFilesystem]
    ) -> None:
        gw = StorageGateway(
            options={
                "s3": {"key": "AKIA", "secret": "shh"},
                "gs": {"token": "/keyfile.json"},
            }
        )
        await gw.write_bytes("s3://b/k", b"x")
        await gw.write_bytes("gs://b/k", b"y")
        assert fake_fsspec["s3"].kwargs == {"key": "AKIA", "secret": "shh"}
        assert fake_fsspec["gs"].kwargs == {"token": "/keyfile.json"}

    async def test_filesystem_cached(self, fake_fsspec: dict[str, _FakeFilesystem]) -> None:
        gw = StorageGateway()
        await gw.write_bytes("/a", b"1")
        await gw.write_bytes("/b", b"2")
        # Both writes used the same fs instance from the cache.
        assert gw._cache["file"] is fake_fsspec["file"]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Missing extra
# ---------------------------------------------------------------------------


class TestMissingExtra:
    async def test_extra_missing_raises_at_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "fsspec", None)
        gw = StorageGateway()
        with pytest.raises(ImportError, match=r"\[storage\] extra"):
            await gw.read_bytes("/anywhere")


# ---------------------------------------------------------------------------
# info type guard
# ---------------------------------------------------------------------------


class TestInfoTypeGuard:
    async def test_non_dict_info_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _WeirdFS:
            def info(self, target: str) -> str:
                del target
                return "not-a-dict"

        fake = types.ModuleType("fsspec")
        fake.filesystem = lambda _protocol, **_: _WeirdFS()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fsspec", fake)

        gw = StorageGateway()
        with pytest.raises(TypeError, match="info"):
            await gw.info("/x")
