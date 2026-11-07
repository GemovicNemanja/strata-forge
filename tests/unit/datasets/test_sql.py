"""Local orchestration fixtures exercise real FD staging with an inert container boundary."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import ClassVar, cast

import pytest

from strata_forge.datasets import _sql_wire as wire
from strata_forge.datasets import sql
from strata_forge.datasets._sql_container import ContainerOutcome
from strata_forge.datasets._sql_worker import bundle_sha256
from strata_forge.datasets.sql_types import (
    LocalParquetFiles,
    LocalSQLInstallation,
    SQLExecutionError,
    SQLInputManifest,
    SQLQuerySpec,
)

_FIXTURE = Path(__file__).parents[2] / "fixtures/headless-sql"


def manifest() -> SQLInputManifest:
    return SQLInputManifest.from_wire(json.loads((_FIXTURE / "typed-manifest.json").read_bytes()))


def installation() -> LocalSQLInstallation:
    return LocalSQLInstallation.from_wire(
        {
            "version": 1,
            "profile": wire.PROFILE,
            "docker_executable": "/fictional/docker",
            "engine_socket": "/fictional/socket",
            "image_id": "sha256:" + "a" * 64,
            "worker_bundle_sha256": bundle_sha256(),
        }
    )


def query(**updates: object) -> SQLQuerySpec:
    return SQLQuerySpec.from_wire(
        {
            "version": 1,
            "sql": "SELECT id FROM dataset",
            "input_manifest_sha256": manifest().digest(),
            **updates,
        }
    )


def table() -> dict[str, wire.JSON]:
    return {
        "version": 1,
        "columns": [{"name": "id", "type": {"kind": "integer", "bits": 64, "signed": True}}],
        "rows": [[{"kind": "integer", "value": "9007199254740993"}]],
        "returned_rows": 1,
        "result_total_rows": 1,
        "result_truncated": False,
        "coverage": {
            "input_manifest_materialized_complete": True,
            "upstream_inventory_verified": False,
            "query_scan_coverage": "not_measured",
            "result_state": "exhausted",
        },
    }


class Container:
    instances: ClassVar[list[Container]] = []
    held: asyncio.Event | None = None
    entered: asyncio.Event | None = None
    refusal: str | None = None
    mismatch = False

    def __init__(self, profile: dict[str, wire.JSON]) -> None:
        self.profile = profile
        self.available = True
        self.calls = 0
        self.directories: list[Path] = []
        self.closed = False
        Container.instances.append(self)

    async def run(
        self, *, input_dir: Path, request: bytes, operation_id: str, deadline: float
    ) -> ContainerOutcome:
        assert deadline > 0
        self.calls += 1
        self.directories.append(input_dir)
        assert (await asyncio.to_thread(input_dir.stat)).st_mode & 0o777 == 0o555
        assert (await asyncio.to_thread(input_dir.parent.stat)).st_mode & 0o777 == 0o700
        data = wire.parse(request, limit=wire.MAX_REQUEST)
        assert type(data) is dict
        content = (input_dir / "part-000000.parquet").read_bytes()
        assert content == (_FIXTURE / "typed.parquet").read_bytes()
        if self.entered is not None:
            self.entered.set()
        if self.held is not None:
            await self.held.wait()
        if self.refusal is not None:
            raise wire.WireError(self.refusal)
        response: dict[str, wire.JSON] = {
            "version": 1,
            "kind": "result",
            "operation_id": "0" * 32 if self.mismatch else operation_id,
            "request_sha256": data["request_sha256"],
            "manifest_sha256": manifest().digest(),
            "table": table(),
            "bundle_sha256": self.profile["worker_bundle_sha256"],
            "source_observed": {
                "parts": 1,
                "rows": 3,
                "bytes": 1461,
                "schema_sha256": manifest().wire_payload()["schema_sha256"],
            },
        }
        return ContainerOutcome(
            wire.canonical(response, limit=wire.MAX_RESULT),
            "fictional-engine",
            cast("str", self.profile["image_id"]),
            cast("str", self.profile["worker_bundle_sha256"]),
            0,
            False,
        )

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def inert_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    Container.instances = []
    Container.held = Container.entered = None
    Container.refusal = None
    Container.mismatch = False
    monkeypatch.setattr(sql, "SQLContainerOwner", Container)


def source(fd: int) -> LocalParquetFiles:
    return LocalParquetFiles.from_open_fds(manifest=manifest(), fds=(fd,))


async def test_real_fd_staging_and_receipt_known_shape_preserve_caller_position() -> None:
    descriptor = os.open(_FIXTURE / "typed.parquet", os.O_RDONLY)
    try:
        os.lseek(descriptor, 7, os.SEEK_SET)
        async with sql.LocalSQLExecutor(installation=installation()) as executor:
            result = await executor.query(spec=query(), source=source(descriptor))
        payload = result.wire_payload()
        actual_table = payload["table"]
        assert actual_table == table()
        receipt = cast("dict[str, wire.JSON]", payload["receipt"])
        assert receipt["source_observed"] == {
            "parts": 1,
            "rows": 3,
            "bytes": 1461,
            "schema_sha256": manifest().wire_payload()["schema_sha256"],
        }
        assert receipt["result_sha256"] == wire.digest(actual_table, limit=wire.MAX_RESULT)
        assert receipt["sql_sha256"] == hashlib.sha256(b"SELECT id FROM dataset").hexdigest()
        assert receipt["cleanup"] == {
            "container_waited": True,
            "stopped_pid_zero": True,
            "container_removed": True,
            "attach_reaped": True,
            "staging_removed": True,
            "exit_code": 0,
            "oom_killed": False,
        }
        assert "SELECT" not in repr(result)
        assert "parquet" not in repr(result)
        assert Container.instances[0].calls == 1
        assert Container.instances[0].closed
        assert all(not path.exists() for path in Container.instances[0].directories)
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 7
        assert os.fstat(descriptor).st_size == 1461
        cast("dict[str, wire.JSON]", payload["table"])["rows"] = []
        assert result.wire_payload()["table"] == table()
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("kind", ["write_fd", "pipe", "wrong_bytes", "wrong_pin"])
async def test_source_failures_refuse_before_container_dispatch(tmp_path: Path, kind: str) -> None:
    if kind == "pipe":
        descriptor, writer = os.pipe()
        os.close(writer)
    elif kind == "write_fd":
        descriptor = os.open(tmp_path / "write", os.O_CREAT | os.O_RDWR, 0o600)
    elif kind == "wrong_bytes":
        (tmp_path / "wrong").write_bytes(b"x" * 1461)
        descriptor = os.open(tmp_path / "wrong", os.O_RDONLY)
    else:
        descriptor = os.open(_FIXTURE / "typed.parquet", os.O_RDONLY)
    try:
        async with sql.LocalSQLExecutor(installation=installation()) as executor:
            with pytest.raises(SQLExecutionError) as caught:
                await executor.query(
                    spec=query(input_manifest_sha256="f" * 64) if kind == "wrong_pin" else query(),
                    source=source(descriptor),
                )
        assert caught.value.code == "source_mismatch"
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None
        assert not Container.instances[0].calls
        assert os.fstat(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("code", ["query_refused", "output_limit", "deadline"])
async def test_fixed_container_failure_has_no_retry_or_raw_chain(code: str) -> None:
    Container.refusal = code
    with (_FIXTURE / "typed.parquet").open("rb") as opened:
        async with sql.LocalSQLExecutor(installation=installation()) as executor:
            with pytest.raises(SQLExecutionError) as caught:
                await executor.query(spec=query(), source=source(opened.fileno()))
    assert caught.value.code == code
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert Container.instances[0].calls == 1
    assert all(not path.exists() for path in Container.instances[0].directories)


async def test_mismatched_response_never_becomes_receipt() -> None:
    Container.mismatch = True
    with (_FIXTURE / "typed.parquet").open("rb") as opened:
        async with sql.LocalSQLExecutor(installation=installation()) as executor:
            with pytest.raises(SQLExecutionError):
                await executor.query(spec=query(), source=source(opened.fileno()))
    assert Container.instances[0].calls == 1


async def test_busy_has_no_queue_and_cancellation_removes_staging() -> None:
    Container.held, Container.entered = asyncio.Event(), asyncio.Event()
    with (_FIXTURE / "typed.parquet").open("rb") as opened:
        async with sql.LocalSQLExecutor(installation=installation()) as executor:
            running = asyncio.create_task(
                executor.query(spec=query(), source=source(opened.fileno()))
            )
            await asyncio.wait_for(Container.entered.wait(), 2)
            with pytest.raises(SQLExecutionError, match="sql_execution_refused") as caught:
                await executor.query(spec=query(), source=source(opened.fileno()))
            assert caught.value.code == "busy"
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            assert Container.instances[0].calls == 1
            assert all(not path.exists() for path in Container.instances[0].directories)


async def test_result_reconstruction_refuses_coherent_boolean_cleanup_tamper() -> None:
    with (_FIXTURE / "typed.parquet").open("rb") as opened:
        async with sql.LocalSQLExecutor(installation=installation()) as executor:
            result = await executor.query(spec=query(), source=source(opened.fileno()))
    payload = copy.deepcopy(result.wire_payload())
    receipt = cast("dict[str, wire.JSON]", payload["receipt"])
    cast("dict[str, wire.JSON]", receipt["cleanup"])["container_waited"] = 1
    receipt["receipt_sha256"] = wire.digest(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"},
        limit=wire.MAX_RESULT,
    )
    forged = sql.SQLResult.model_construct(canonical=wire.canonical(payload, limit=wire.MAX_RESULT))
    with pytest.raises(SQLExecutionError):
        forged.wire_payload()
