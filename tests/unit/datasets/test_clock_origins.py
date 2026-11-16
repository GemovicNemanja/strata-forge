"""Local executor clock conformance; inert containers never attest native execution."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# Fixed sibling test fixtures expose operational boundaries, never runtime selectors.
import asyncio
import importlib
import importlib.util
import os
import socket
import time
from typing import TYPE_CHECKING, cast

import pytest

from strata_forge.datasets import dataset_execution, sql
from strata_forge.datasets._sql_container import SQLContainerOwner, _Profile
from tests.unit.datasets import test_dataset_execution as dataset_fixture
from tests.unit.datasets import test_sql as sql_fixture

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **__: object) -> None:
        pytest.fail("clock conformance performs no network operation")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def loop_factory(kind: str) -> Callable[[], asyncio.AbstractEventLoop]:
    if kind == "stdlib":
        return asyncio.new_event_loop
    if importlib.util.find_spec("uvloop") is None:
        pytest.skip(
            "optional installed uvloop profile absent; local clock conformance is tracked "
            "in docs/modules/native-dataset-execution.md"
        )
    module = importlib.import_module("uvloop")
    factory: object = getattr(module, "new_event_loop", None)
    assert callable(factory)
    return cast("Callable[[], asyncio.AbstractEventLoop]", factory)


def execute(kind: str, action: Callable[[], Awaitable[None]]) -> None:
    async def operation() -> None:
        await action()

    with asyncio.Runner(loop_factory=loop_factory(kind)) as runner:
        runner.run(operation())


@pytest.mark.parametrize("kind", ["stdlib", "uvloop"])
def test_dataset_executor_retains_original_clock_budget(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def operation() -> None:
        dataset_fixture.Container.instances = []
        dataset_fixture.Container.hold = None
        dataset_fixture.Container.entered = None
        dataset_fixture.Container.retired = None
        dataset_fixture.Container.release = None
        dataset_fixture.Container.corrupt = None
        path = tmp_path / "fictional.jsonl"
        path.write_bytes(b'{"id":1}\n')
        source, spec, descriptor = dataset_fixture.provided(path)
        os.lseek(descriptor, 3, os.SEEK_SET)
        monkeypatch.setattr(dataset_execution, "SQLContainerOwner", dataset_fixture.Container)
        try:
            async with dataset_execution.LocalDatasetExecutor(
                installation=dataset_fixture.installation()
            ) as executor:
                result = await executor.convert(spec=spec, source=source)
                assert result.output_bytes() == b"PAR1-fictional-controlled-output"
                assert executor._staging is None
                assert os.lseek(descriptor, 0, os.SEEK_CUR) == 3
        finally:
            os.close(descriptor)

    execute(kind, operation)


@pytest.mark.parametrize("kind", ["stdlib", "uvloop"])
def test_sql_executor_retains_original_clock_budget(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def operation() -> None:
        sql_fixture.Container.instances = []
        sql_fixture.Container.held = None
        sql_fixture.Container.entered = None
        sql_fixture.Container.refusal = None
        sql_fixture.Container.mismatch = False
        descriptor = os.open(sql_fixture._FIXTURE / "typed.parquet", os.O_RDONLY)
        monkeypatch.setattr(sql, "SQLContainerOwner", sql_fixture.Container)
        try:
            async with sql.LocalSQLExecutor(installation=sql_fixture.installation()) as executor:
                result = await executor.query(
                    spec=sql_fixture.query(), source=sql_fixture.source(descriptor)
                )
                assert result.wire_payload()["table"] == sql_fixture.table()
        finally:
            os.close(descriptor)

    execute(kind, operation)


@pytest.mark.parametrize("kind", ["stdlib", "uvloop"])
@pytest.mark.parametrize("profile", [_Profile.SQL, _Profile.DATASET])
def test_shared_owner_stdio_retains_original_clock_budget(
    kind: str, profile: _Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def operation() -> None:
        installation = (
            dataset_fixture.installation().wire_payload()
            if profile is _Profile.DATASET
            else sql_fixture.installation().wire_payload()
        )
        owner = SQLContainerOwner(installation, _profile=profile)

        class Pipe:
            def __init__(self) -> None:
                self.sent = False

            async def read(self, _: int) -> bytes:
                if self.sent:
                    return b""
                self.sent = True
                return b"fictional-stdio"

        class Process:
            def __init__(self) -> None:
                self.stdout = Pipe()
                self.returncode = 0

            async def wait(self) -> int:
                return 0

        async def spawning(
            self: SQLContainerOwner, args: tuple[str, ...], *, pipe: bool = False
        ) -> asyncio.subprocess.Process:
            del self, args, pipe
            await asyncio.sleep(0.005)
            # This controlled stdio protocol has no OS child or native worker.
            return cast("asyncio.subprocess.Process", Process())

        monkeypatch.setattr(SQLContainerOwner, "_spawn", spawning)
        try:
            result = await owner._command("fixture", deadline=time.monotonic() + 1)
            assert result == b"fictional-stdio"
        finally:
            await owner.aclose()

    execute(kind, operation)
