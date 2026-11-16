"""Synthetic parent failure/lifetime controls, not actual container settlement evidence."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# Controlled owner outcomes exercise fixed issuance without pretending to run a daemon.
import asyncio
import copy
import gc
import hashlib
import os
import pickle
import threading
import weakref
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, cast

import pytest

from strata_forge.datasets import _dataset_wire as wire
from strata_forge.datasets import _sql_wire as sql
from strata_forge.datasets import dataset_execution as execution
from strata_forge.datasets._dataset_worker import bundle_sha256
from strata_forge.datasets._sql_container import ContainerOutcome, ContainerSettlement
from strata_forge.datasets.dataset_execution_types import (
    DatasetConvertSpec,
    DatasetExecutionError,
    DatasetInputManifest,
    DatasetInstallation,
    ProvidedDatasetFiles,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class Owner:
    mode: ClassVar[str] = "clean"
    instances: ClassVar[list[Owner]] = []

    def __init__(self, installation: dict[str, sql.JSON], **_: object) -> None:
        self.installation = installation
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
        del input_dir, deadline
        self.calls += 1
        raw = cast("dict[str, sql.JSON]", sql.parse(request, limit=sql.MAX_REQUEST))
        if self.calls == 1:
            if self.mode == "unknown":
                raise sql.WireError("worker_failed")
            if self.mode == "cancel":
                raise asyncio.CancelledError
            payload: dict[str, sql.JSON] = {
                "version": 1,
                "kind": "error",
                "operation_id": "0" * 32 if self.mode == "correlation" else operation_id,
                "request_sha256": raw["request_sha256"],
                "code": "profile_refused" if self.mode == "profile" else "source_mismatch",
            }
            settlement = ContainerSettlement(operation_id, True, True, True, True)
            if self.mode in {"wait", "pid", "remove", "attach", "foreign_operation"}:
                field = {
                    "wait": "container_waited",
                    "pid": "stopped_pid_zero",
                    "remove": "container_removed",
                    "attach": "attach_reaped",
                    "foreign_operation": "operation_id",
                }[self.mode]
                settlement = replace(
                    settlement, **{field: "0" * 32 if self.mode == "foreign_operation" else False}
                )
            return ContainerOutcome(
                sql.canonical(payload, limit=sql.MAX_RESULT),
                "engine-A",
                cast("str", self.installation["image_id"]),
                cast("str", self.installation["worker_bundle_sha256"]),
                1,
                False,
                None if self.mode == "missing" else settlement,
            )
        data = b"PAR1-fictional-output"
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
        payload = {
            "version": 1,
            "kind": "result",
            "operation_id": operation_id,
            "request_sha256": raw["request_sha256"],
            "bundle_sha256": self.installation["worker_bundle_sha256"],
            "table": None,
            "output": output,
        }
        return ContainerOutcome(
            sql.canonical(payload, limit=sql.MAX_RESULT),
            "engine-A",
            cast("str", self.installation["image_id"]),
            cast("str", self.installation["worker_bundle_sha256"]),
            0,
            False,
        )


@pytest.fixture
def source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[execution.LocalDatasetExecutor, DatasetConvertSpec, ProvidedDatasetFiles]]:
    Owner.mode = "clean"
    Owner.instances.clear()
    monkeypatch.setattr(execution, "SQLContainerOwner", Owner)
    installation = DatasetInstallation.from_wire(
        {
            "version": 1,
            "profile": wire.PROFILE,
            "docker_executable": "/fictional/docker",
            "engine_socket": "/fictional/socket",
            "image_id": "sha256:" + "a" * 64,
            "worker_bundle_sha256": bundle_sha256(),
        }
    )
    path = tmp_path / "fictional.jsonl"
    data = b'{"id":1}\n'
    path.write_bytes(data)
    manifest = DatasetInputManifest.from_wire(
        {
            "version": 1,
            "format": "jsonl",
            "parts": [
                {"ordinal": 0, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
            ],
        }
    )
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
    fd = os.open(path, os.O_RDONLY)
    try:
        yield (
            execution.LocalDatasetExecutor(installation=installation),
            spec,
            ProvidedDatasetFiles(manifest=manifest, fds=(fd,)),
        )
    finally:
        os.close(fd)


async def test_exact_failed_call_observation_then_same_instance_success(
    source: tuple[execution.LocalDatasetExecutor, DatasetConvertSpec, ProvidedDatasetFiles],
) -> None:
    executor, spec, files = source
    async with executor:
        with pytest.raises(DatasetExecutionError) as caught:
            await executor.convert(spec=spec, source=files)
        error = caught.value
        observation = executor.failure_observation(error)
        assert observation is not None
        payload = observation.wire_payload()
        assert payload is not None
        assert set(payload) == {
            "version",
            "code",
            "kind",
            "operation_id",
            "request_sha256",
            "profile",
            "engine_id",
            "image_id",
            "worker_bundle_sha256",
            "cleanup",
            "reusable_at_settlement",
        }
        assert payload["code"] == "source_mismatch"
        assert payload["kind"] == "convert"
        assert payload["reusable_at_settlement"] is True
        assert payload["cleanup"] == {
            "container_waited": True,
            "stopped_pid_zero": True,
            "container_removed": True,
            "attach_reaped": True,
            "staging_removed": True,
            "exit_code": 1,
            "oom_killed": False,
        }
        error.code = "changed"
        error.args = ({"canary": "must not enter metadata"},)
        payload["code"] = "changed"
        cast("dict[str, sql.JSON]", payload["cleanup"])["container_waited"] = False
        fresh = observation.wire_payload()
        assert fresh is not None
        assert fresh["code"] == "source_mismatch"
        assert "canary" not in repr(observation)
        assert "/fictional" not in str(observation.wire_payload())
        result = await executor.convert(spec=spec, source=files)
        assert result.output_bytes() == b"PAR1-fictional-output"
        assert len(Owner.instances) == 1
        assert Owner.instances[0].calls == 2
        assert executor.failure_observation(error) is None
        assert observation.wire_payload() is None


@pytest.mark.parametrize(
    "mode",
    ["missing", "wait", "pid", "remove", "attach", "foreign_operation", "correlation", "unknown"],
)
async def test_no_witness_from_available_flag_or_unproved_settlement(
    source: tuple[execution.LocalDatasetExecutor, DatasetConvertSpec, ProvidedDatasetFiles],
    mode: str,
) -> None:
    executor, spec, files = source
    Owner.mode = mode
    async with executor:
        with pytest.raises(DatasetExecutionError) as caught:
            await executor.convert(spec=spec, source=files)
        assert Owner.instances[0].available is True
        assert executor.failure_observation(caught.value) is None


async def test_known_profile_refusal_is_not_a_reusable_witness(
    source: tuple[execution.LocalDatasetExecutor, DatasetConvertSpec, ProvidedDatasetFiles],
) -> None:
    executor, spec, files = source
    Owner.mode = "profile"
    async with executor:
        with pytest.raises(DatasetExecutionError) as caught:
            await executor.convert(spec=spec, source=files)
        observation = executor.failure_observation(caught.value)
        assert observation is not None
        payload = observation.wire_payload()
        assert payload is not None
        assert payload["reusable_at_settlement"] is False


async def test_foreign_error_task_forged_handle_and_close_are_denied(
    source: tuple[execution.LocalDatasetExecutor, DatasetConvertSpec, ProvidedDatasetFiles],
) -> None:
    executor, spec, files = source
    async with executor:
        with pytest.raises(DatasetExecutionError) as caught:
            await executor.convert(spec=spec, source=files)
        observation = executor.failure_observation(caught.value)
        assert observation is not None
        assert executor.failure_observation(DatasetExecutionError("source_mismatch")) is None
        forged = object.__new__(execution.DatasetFailureObservation)
        assert forged.wire_payload() is None

        async def foreign() -> None:
            assert executor.failure_observation(caught.value) is None
            assert observation.wire_payload() is None

        await asyncio.create_task(foreign())
        assert observation.wire_payload() is not None
        for operation in (
            lambda: copy.copy(observation),
            lambda: copy.deepcopy(observation),
            lambda: pickle.dumps(observation),
            lambda: setattr(observation, "_owner", object()),
            lambda: delattr(observation, "_owner"),
        ):
            with pytest.raises(DatasetExecutionError):
                operation()
    assert observation.wire_payload() is None


async def test_staging_failure_cannot_issue_a_clean_worker_witness(
    source: tuple[execution.LocalDatasetExecutor, DatasetConvertSpec, ProvidedDatasetFiles],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, spec, files = source
    actual = execution._remove

    def failed(path: Path) -> None:
        raise OSError("fictional private cleanup diagnostic")

    async with executor:
        with monkeypatch.context() as guard:
            guard.setattr(execution, "_remove", failed)
            with pytest.raises(DatasetExecutionError) as caught:
                await executor.convert(spec=spec, source=files)
            assert caught.value.code == "cleanup_unresolved"
            assert caught.value.__context__ is None
            assert executor.failure_observation(caught.value) is None
        if executor._staging is not None:
            actual(executor._staging)
            executor._staging = None


async def test_staging_drain_precedes_actual_error_issuance_and_in_task_capture(
    source: tuple[execution.LocalDatasetExecutor, DatasetConvertSpec, ProvidedDatasetFiles],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, spec, files = source
    entered, release = threading.Event(), threading.Event()
    actual = execution._remove

    def held(path: Path) -> None:
        entered.set()
        assert release.wait(5)
        actual(path)

    async def failed() -> dict[str, sql.JSON]:
        try:
            await executor.convert(spec=spec, source=files)
        except DatasetExecutionError as error:
            observation = executor.failure_observation(error)
            assert observation is not None
            payload = observation.wire_payload()
            assert payload is not None
            return payload
        raise AssertionError("expected fixed worker refusal")

    async with executor:
        monkeypatch.setattr(execution, "_remove", held)
        task = asyncio.create_task(failed())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert not task.done()
            assert executor.failure_observation(DatasetExecutionError("source_mismatch")) is None
        finally:
            release.set()
        payload = await task
        assert payload["reusable_at_settlement"] is True


async def test_observation_retains_no_exception_or_executor_lifetime(
    source: tuple[execution.LocalDatasetExecutor, DatasetConvertSpec, ProvidedDatasetFiles],
) -> None:
    executor, spec, files = source
    async with executor:
        observation: execution.DatasetFailureObservation | None = None

        async def fail() -> weakref.ReferenceType[DatasetExecutionError]:
            nonlocal observation
            try:
                await executor.convert(spec=spec, source=files)
            except DatasetExecutionError as error:
                observation = executor.failure_observation(error)
                return weakref.ref(error)
            raise AssertionError("expected fixed refusal")

        reference = await fail()
        gc.collect()
        assert reference() is None
        assert observation is not None
        assert observation.wire_payload() is None


async def test_cancel_has_no_new_witness_and_preserves_legacy_same_executor_reuse(
    source: tuple[execution.LocalDatasetExecutor, DatasetConvertSpec, ProvidedDatasetFiles],
) -> None:
    executor, spec, files = source
    Owner.mode = "cancel"
    async with executor:
        with pytest.raises(asyncio.CancelledError):
            await executor.convert(spec=spec, source=files)
        assert executor.failure_observation(DatasetExecutionError("worker_failed")) is None
        assert (
            await executor.convert(spec=spec, source=files)
        ).output_bytes() == b"PAR1-fictional-output"


async def test_hostile_error_subclass_and_forged_weak_owner_execute_no_callbacks(
    source: tuple[execution.LocalDatasetExecutor, DatasetConvertSpec, ProvidedDatasetFiles],
) -> None:
    executor, spec, files = source
    invoked: list[str] = []

    class HostileError(DatasetExecutionError):
        def __getattribute__(self, name: str) -> object:
            invoked.append(name)
            raise AssertionError("foreign attribute must not execute")

    class ForeignOwner:
        def __getattribute__(self, name: str) -> object:
            invoked.append(name)
            raise AssertionError("foreign owner must not execute")

    foreign = ForeignOwner()
    forged = object.__new__(execution.DatasetFailureObservation)
    object.__setattr__(forged, "_owner", weakref.ref(foreign))
    assert forged.wire_payload() is None
    assert executor.failure_observation(BaseException.__new__(HostileError)) is None
    assert invoked == []
    async with executor:
        with pytest.raises(DatasetExecutionError) as caught:
            await executor.convert(spec=spec, source=files)
        object.__setattr__(caught.value, "code", {"canary": "unhashable is not observation input"})
        observation = executor.failure_observation(caught.value)
        assert observation is not None
        payload = observation.wire_payload()
        assert payload is not None
        assert payload["code"] == "source_mismatch"


async def test_foreign_loop_and_no_io_inspection(
    source: tuple[execution.LocalDatasetExecutor, DatasetConvertSpec, ProvidedDatasetFiles],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, spec, files = source
    async with executor:
        with pytest.raises(DatasetExecutionError) as caught:
            await executor.convert(spec=spec, source=files)
        observation = executor.failure_observation(caught.value)
        assert observation is not None

        async def foreign() -> None:
            assert observation.wire_payload() is None

        await asyncio.to_thread(lambda: asyncio.run(foreign()))

        def forbidden(*args: object, **kwargs: object) -> object:
            raise AssertionError("inspection performs no I/O, hash or caller work")

        with monkeypatch.context() as guard:
            guard.setattr(os, "open", forbidden)
            guard.setattr(hashlib, "sha256", forbidden)
            guard.setattr(asyncio, "to_thread", forbidden)
            assert observation.wire_payload() is not None


async def test_retained_observation_does_not_retain_executor_or_error(
    source: tuple[execution.LocalDatasetExecutor, DatasetConvertSpec, ProvidedDatasetFiles],
) -> None:
    original, spec, files = source
    installation = DatasetInstallation.from_wire(original._installation)

    async def run() -> tuple[
        execution.DatasetFailureObservation, weakref.ReferenceType[execution.LocalDatasetExecutor]
    ]:
        selected = execution.LocalDatasetExecutor(installation=installation)
        async with selected:
            with pytest.raises(DatasetExecutionError) as caught:
                await selected.convert(spec=spec, source=files)
            observation = selected.failure_observation(caught.value)
            assert observation is not None
        return observation, weakref.ref(selected)

    observation, reference = await run()
    gc.collect()
    assert reference() is None
    assert observation.wire_payload() is None
