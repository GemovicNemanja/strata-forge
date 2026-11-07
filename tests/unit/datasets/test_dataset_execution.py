"""Controlled parent I/O/admission faults; not actual worker or cleanup attestation."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# Fault controls inspect fixed private cleanup/worker boundaries without runtime bypasses.
import asyncio
import hashlib
import os
import threading
import time
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from strata_forge.datasets import _dataset_wire as wire
from strata_forge.datasets import _sql_wire as sql
from strata_forge.datasets import dataset_execution as execution
from strata_forge.datasets._dataset_worker import bundle_sha256
from strata_forge.datasets._sql_container import ContainerOutcome
from strata_forge.datasets.dataset_execution_types import (
    DatasetConvertSpec,
    DatasetExecutionError,
    DatasetInputManifest,
    DatasetInstallation,
    ProvidedDatasetFiles,
)


def installation() -> DatasetInstallation:
    return DatasetInstallation.from_wire(
        {
            "version": 1,
            "profile": wire.PROFILE,
            "docker_executable": "/fictional/docker",
            "engine_socket": "/fictional/socket",
            "image_id": "sha256:" + "a" * 64,
            "worker_bundle_sha256": bundle_sha256(),
        }
    )


def provided(path: Path) -> tuple[ProvidedDatasetFiles, DatasetConvertSpec, int]:
    data = path.read_bytes()
    manifest = DatasetInputManifest.from_wire(
        {
            "version": 1,
            "format": "jsonl",
            "parts": [
                {"ordinal": 0, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
            ],
        }
    )
    fd = os.open(path, os.O_RDONLY)
    source = ProvidedDatasetFiles(manifest=manifest, fds=(fd,))
    spec = DatasetConvertSpec.from_wire(
        {
            "version": 1,
            "format": "jsonl",
            "schema": [
                {
                    "name": "id",
                    "type": {"kind": "integer", "bits": 64, "signed": True},
                    "nullable": False,
                }
            ],
            "input_manifest_sha256": manifest.digest(),
            "timeout_ms": 10000,
        }
    )
    return source, spec, fd


class Container:
    instances: ClassVar[list[Container]] = []
    hold: ClassVar[asyncio.Event | None] = None
    entered: ClassVar[asyncio.Event | None] = None
    retired: ClassVar[asyncio.Event | None] = None
    release: ClassVar[asyncio.Event | None] = None
    corrupt: ClassVar[str | None] = None

    def __init__(self, value: dict[str, sql.JSON], **_: object) -> None:
        self.value = value
        self.available = True
        self.calls = 0
        self.instances.append(self)

    async def aclose(self) -> None:
        self.available = False

    async def run(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        request: bytes,
        operation_id: str,
        deadline: float,
    ) -> ContainerOutcome:
        del deadline
        self.calls += 1
        if self.entered is not None:
            self.entered.set()
        if self.hold is not None:
            try:
                await self.hold.wait()
            except asyncio.CancelledError:
                assert self.retired is not None
                assert self.release is not None
                self.retired.set()
                while not self.release.is_set():
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        continue
                raise
        assert (input_dir / "part-000000.bin").read_bytes() == b'{"id":1}\n'
        raw = cast("dict[str, sql.JSON]", sql.parse(request, limit=sql.MAX_REQUEST))
        data = b"PAR1-fictional-controlled-output"
        (output_dir / "data.parquet").write_bytes(data)
        columns: list[sql.JSON] = [
            {"name": "id", "type": {"kind": "integer", "bits": 64, "signed": True}}
        ]
        output: dict[str, sql.JSON] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "schema": columns,
            "schema_sha256": sql.digest(columns),
            "rows": 1,
        }
        if self.corrupt == "hash":
            output["sha256"] = "0" * 64
        if self.corrupt == "extra":
            output["fictional"] = "canary"
        result: dict[str, sql.JSON] = {
            "version": 1,
            "kind": "result",
            "operation_id": operation_id,
            "request_sha256": raw["request_sha256"],
            "bundle_sha256": self.value["worker_bundle_sha256"],
            "table": None,
            "output": output,
        }
        return ContainerOutcome(
            sql.canonical(result, limit=sql.MAX_RESULT),
            "engine-A",
            cast("str", self.value["image_id"]),
            cast("str", self.value["worker_bundle_sha256"]),
            0,
            False,
        )


@pytest.fixture
def controlled(monkeypatch: pytest.MonkeyPatch) -> None:
    Container.instances = []
    Container.hold = Container.entered = Container.retired = Container.release = None
    Container.corrupt = None
    monkeypatch.setattr(execution, "SQLContainerOwner", Container)


@pytest.mark.usefixtures("controlled")
async def test_immutable_output_cleanup_and_fd_offset(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_bytes(b'{"id":1}\n')
    source, spec, fd = provided(path)
    os.lseek(fd, 3, os.SEEK_SET)
    try:
        async with execution.LocalDatasetExecutor(installation=installation()) as executor:
            result = await executor.convert(spec=spec, source=source)
            assert executor._staging is None
            assert os.lseek(fd, 0, os.SEEK_CUR) == 3
            assert result.output_bytes() == b"PAR1-fictional-controlled-output"
            assert result.model_dump() == {}
            assert "fictional" not in repr(result)
            payload = result.wire_payload()
            payload["output"] = None
            assert result.wire_payload()["output"] is not None
    finally:
        os.close(fd)


@pytest.mark.usefixtures("controlled")
@pytest.mark.parametrize("corruption", ["hash", "extra"])
async def test_changed_output_never_adopts(tmp_path: Path, corruption: str) -> None:
    path = tmp_path / "input.jsonl"
    path.write_bytes(b'{"id":1}\n')
    source, spec, fd = provided(path)
    Container.corrupt = corruption
    try:
        async with execution.LocalDatasetExecutor(installation=installation()) as executor:
            with pytest.raises(DatasetExecutionError) as caught:
                await executor.convert(spec=spec, source=source)
            assert caught.value.__context__ is None
            assert "canary" not in repr(caught.value)
            assert executor._staging is None
    finally:
        os.close(fd)


@pytest.mark.usefixtures("controlled")
async def test_repeated_cancel_holds_admission_until_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_bytes(b'{"id":1}\n')
    source, spec, fd = provided(path)
    Container.hold, Container.entered = asyncio.Event(), asyncio.Event()
    Container.retired, Container.release = asyncio.Event(), asyncio.Event()
    try:
        async with execution.LocalDatasetExecutor(installation=installation()) as executor:
            operation = asyncio.create_task(executor.convert(spec=spec, source=source))
            await Container.entered.wait()
            operation.cancel()
            await Container.retired.wait()
            operation.cancel()
            with pytest.raises(DatasetExecutionError) as caught:
                await executor.convert(spec=spec, source=source)
            assert caught.value.code == "busy"
            with pytest.raises(DatasetExecutionError):
                await executor.aclose()
            assert not operation.done()
            assert executor._staging is not None
            Container.release.set()
            with pytest.raises(asyncio.CancelledError):
                await operation
            assert executor._staging is None
    finally:
        os.close(fd)


def test_no_follow_and_exact_output_bytes(tmp_path: Path) -> None:
    content = b"fictional"
    target = tmp_path / "real"
    target.write_bytes(content)
    output = tmp_path / "output"
    output.mkdir()
    (output / "data.parquet").symlink_to(target)
    descriptor: dict[str, sql.JSON] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }
    with pytest.raises(OSError, match=r"(Too many levels|Symbolic link)"):
        execution._output(output, descriptor, time.monotonic() + 1)


def test_stage_mismatch_does_not_move_source_fd(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"fictional")
    directory = tmp_path / "input"
    directory.mkdir()
    fd = os.open(source, os.O_RDONLY)
    os.lseek(fd, 4, os.SEEK_SET)
    try:
        with pytest.raises(sql.WireError):
            execution._stage(
                [{"size_bytes": 9, "sha256": "0" * 64}],
                (fd,),
                directory,
                time.monotonic() + 1,
                threading.Event(),
            )
        assert os.lseek(fd, 0, os.SEEK_CUR) == 4
        assert os.fstat(fd).st_size == 9
    finally:
        os.close(fd)
