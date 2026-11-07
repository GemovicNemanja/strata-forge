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


async def test_concurrent_enter_never_creates_two_owners(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    entering, release = threading.Event(), threading.Event()

    class Held(Container):
        def __init__(self, profile: dict[str, wire.JSON]) -> None:
            super().__init__(profile)
            entering.set()
            assert release.wait(2)

    monkeypatch.setattr(sql, "SQLContainerOwner", Held)
    executor = sql.LocalSQLExecutor(installation=installation())
    first = asyncio.create_task(executor.__aenter__())
    try:
        assert await asyncio.to_thread(entering.wait, 2)
        with pytest.raises(SQLExecutionError) as caught:
            await executor.__aenter__()
        assert caught.value.code == "unavailable"
        with pytest.raises(SQLExecutionError) as caught:
            await executor.aclose()
        assert caught.value.code == "busy"
    finally:
        release.set()
        await first
        await executor.aclose()
    assert len(Container.instances) == 1


async def test_concurrent_close_joins_one_owned_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    calls: list[int] = []

    async def close(self: Container) -> None:
        calls.append(1)
        entered.set()
        await release.wait()
        self.closed = True

    monkeypatch.setattr(Container, "aclose", close)
    executor = sql.LocalSQLExecutor(installation=installation())
    await executor.__aenter__()
    first = asyncio.create_task(executor.aclose())
    await entered.wait()
    second = asyncio.create_task(executor.aclose())
    await asyncio.sleep(0)
    try:
        assert len(calls) == 1
    finally:
        release.set()
        await asyncio.gather(first, second)
    await executor.aclose()
    assert len(calls) == 1


@pytest.mark.parametrize("exit_code", [1, 2, True])
async def test_valid_result_frame_followed_by_failed_exit_is_never_success(
    monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    from dataclasses import replace

    actual = Container.run

    async def failed(
        self: Container, *, input_dir: Path, request: bytes, operation_id: str, deadline: float
    ) -> ContainerOutcome:
        return replace(
            await actual(
                self,
                input_dir=input_dir,
                request=request,
                operation_id=operation_id,
                deadline=deadline,
            ),
            exit_code=exit_code,
        )

    monkeypatch.setattr(Container, "run", failed)
    with (_FIXTURE / "typed.parquet").open("rb") as opened:
        async with sql.LocalSQLExecutor(installation=installation()) as executor:
            with pytest.raises(SQLExecutionError) as caught:
                await executor.query(spec=query(), source=source(opened.fileno()))
    assert caught.value.code == "worker_failed"
    assert caught.value.__context__ is None
    assert Container.instances[0].calls == 1
    assert all(not path.exists() for path in Container.instances[0].directories)


@pytest.mark.parametrize(
    ("exit_code", "expected"), [(0, "worker_failed"), (1, "query_refused"), (2, "worker_failed")]
)
async def test_error_frame_requires_exact_fixed_worker_failure_exit(
    monkeypatch: pytest.MonkeyPatch, exit_code: int, expected: str
) -> None:
    async def refused(
        self: Container, *, input_dir: Path, request: bytes, operation_id: str, deadline: float
    ) -> ContainerOutcome:
        data = cast("dict[str,wire.JSON]", wire.parse(request, limit=wire.MAX_REQUEST))
        frame = {
            "version": 1,
            "kind": "error",
            "operation_id": operation_id,
            "request_sha256": data["request_sha256"],
            "code": "query_refused",
        }
        self.calls += 1
        return ContainerOutcome(
            wire.canonical(frame, limit=wire.MAX_RESULT),
            "fictional-engine",
            cast("str", self.profile["image_id"]),
            cast("str", self.profile["worker_bundle_sha256"]),
            exit_code,
            False,
        )

    monkeypatch.setattr(Container, "run", refused)
    with (_FIXTURE / "typed.parquet").open("rb") as opened:
        async with sql.LocalSQLExecutor(installation=installation()) as executor:
            with pytest.raises(SQLExecutionError) as caught:
                await executor.query(spec=query(), source=source(opened.fileno()))
    assert caught.value.code == expected
    assert Container.instances[0].calls == 1


async def test_captured_loop_refuses_cross_thread_loop_before_staging_or_dispatch() -> None:
    with (_FIXTURE / "typed.parquet").open("rb") as opened:
        async with sql.LocalSQLExecutor(installation=installation()) as executor:

            async def other_loop() -> None:
                for operation in (
                    executor.__aenter__(),
                    executor.query(spec=query(), source=source(opened.fileno())),
                    executor.aclose(),
                ):
                    with pytest.raises(SQLExecutionError) as caught:
                        await operation
                    assert caught.value.code == "unavailable"

            await asyncio.to_thread(lambda: asyncio.run(other_loop()))
            assert Container.instances[0].calls == 0
            result = await executor.query(spec=query(), source=source(opened.fileno()))
            assert result.wire_payload()["table"] == table()


async def test_actual_stdlib_fault_child_result_then_exit_one_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This controlled process is not the attested SQL image. It proves the failed-exit
    # observation rather than pretending an actual SQL worker normally emits this pair.
    import sys
    from dataclasses import replace

    actual = Container.run
    observed: list[int] = []

    async def failed(
        self: Container, *, input_dir: Path, request: bytes, operation_id: str, deadline: float
    ) -> ContainerOutcome:
        outcome = await actual(
            self, input_dir=input_dir, request=request, operation_id=operation_id, deadline=deadline
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys;sys.stdout.buffer.write(sys.stdin.buffer.read(2097153));sys.stdout.buffer.flush();raise SystemExit(1)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        )
        output, _ = await process.communicate(
            wire.frame(wire.parse(outcome.payload, limit=wire.MAX_RESULT), limit=wire.MAX_RESULT)
        )
        assert process.returncode == 1
        observed.append(process.returncode)
        decoder = wire.FrameDecoder(limit=wire.MAX_RESULT)
        decoder.feed(output)
        decoded = decoder.finish()
        return replace(
            outcome,
            payload=wire.canonical(decoded, limit=wire.MAX_RESULT),
            exit_code=process.returncode,
        )

    monkeypatch.setattr(Container, "run", failed)
    with (_FIXTURE / "typed.parquet").open("rb") as opened:
        async with sql.LocalSQLExecutor(installation=installation()) as executor:
            with pytest.raises(SQLExecutionError) as caught:
                await executor.query(spec=query(), source=source(opened.fileno()))
    assert caught.value.code == "worker_failed"
    assert observed == [1]
    assert Container.instances[0].calls == 1


async def test_staging_uses_fixed_unix_tmp_and_canonical_mount_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tempfile

    made: list[Path] = []
    actual = tempfile.mkdtemp

    def create(*, prefix: str, dir: str) -> str:  # noqa: A002 — preserve the stdlib keyword contract.
        assert dir == "/tmp"  # noqa: S108 — exact namespace admission, not an unsafe file path.
        result = actual(prefix=prefix, dir=dir)
        made.append(Path(result))
        return result

    monkeypatch.setattr(sql.tempfile, "mkdtemp", create)
    with (_FIXTURE / "typed.parquet").open("rb") as opened:
        async with sql.LocalSQLExecutor(installation=installation()) as executor:
            await executor.query(spec=query(), source=source(opened.fileno()))
    assert len(made) == 1
    directory = Container.instances[0].directories[0]
    assert directory.parent == made[0].resolve()
    assert directory == directory.resolve()
    assert not made[0].exists()
