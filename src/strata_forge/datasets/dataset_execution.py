"""Owned local dataset transformations, pinned named inputs and immutable output bytes."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import os
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.datasets import _dataset_wire as wire
from strata_forge.datasets import _sql_wire as sql
from strata_forge.datasets._dataset_worker import bundle_sha256
from strata_forge.datasets._sql_container import (  # pyright: ignore[reportPrivateUsage] -- fixed sibling is the only dataset profile selector.
    SQLContainerOwner,
    _Profile,  # pyright: ignore[reportPrivateUsage] -- closed fixed sibling selector.
    drain,
)
from strata_forge.datasets.dataset_execution_types import (
    DatasetConvertSpec,
    DatasetExecutionError,
    DatasetInstallation,
    DatasetQuerySpec,
    NamedParquetFiles,
    ProvidedDatasetFiles,
)


def _description(value: object) -> dict[str, sql.JSON]:
    data = wire.obj(value, {"sha256", "size_bytes", "schema", "schema_sha256", "rows"})
    schema = sql.columns(data["schema"], source=True)
    wire.need(sql.digest(schema) == wire.sha(data["schema_sha256"]), "source_mismatch")
    return {
        "sha256": wire.sha(data["sha256"]),
        "size_bytes": wire.count(data["size_bytes"], 1, wire.MAX_BYTES),
        "schema": schema,
        "schema_sha256": data["schema_sha256"],
        "rows": wire.count(data["rows"], 0, wire.MAX_ROWS),
    }


def _result(canonical: bytes, parquet: bytes | None) -> dict[str, sql.JSON]:
    data = wire.obj(
        sql.parse(canonical, limit=sql.MAX_RESULT),
        {"version", "kind", "table", "output", "receipt"},
    )
    wire.need(type(data["version"]) is int and data["version"] == 1)
    kind = data["kind"]
    wire.need(type(kind) is str and kind in {"convert", "query", "materialize"})
    receipt = wire.obj(
        data["receipt"],
        {
            "version",
            "operation_id",
            "request_sha256",
            "spec",
            "inputs",
            "profile",
            "engine_id",
            "image_id",
            "worker_bundle_sha256",
            "result_sha256",
            "cleanup",
            "elapsed_ms",
            "receipt_sha256",
        },
    )
    wire.need(type(receipt["version"]) is int and receipt["version"] == 1)
    wire.need(re.fullmatch(r"[0-9a-f]{32}", wire.text(receipt["operation_id"], 32)) is not None)
    wire.need(receipt["profile"] == wire.PROFILE)
    spec = (
        wire.convert_spec(receipt["spec"])
        if kind == "convert"
        else wire.query_spec(receipt["spec"])
    )
    inputs: sql.JSON = (
        wire.input_manifest(receipt["inputs"])
        if kind == "convert"
        else wire.named_inputs(receipt["inputs"])
    )
    wire.need(spec == receipt["spec"] and inputs == receipt["inputs"])
    wire.need(
        spec["input_manifest_sha256" if kind == "convert" else "inputs_sha256"]
        == sql.digest(inputs),
        "source_mismatch",
    )
    if kind == "convert":
        wire.need(
            spec["format"] == cast("dict[str, sql.JSON]", inputs)["format"], "source_mismatch"
        )
    request: dict[str, sql.JSON] = {
        "version": 1,
        "kind": kind,
        "operation_id": receipt["operation_id"],
        "spec": spec,
        "inputs": inputs,
    }
    wire.need(
        wire.sha(receipt["request_sha256"]) == sql.digest(request, limit=sql.MAX_REQUEST),
        "source_mismatch",
    )
    wire.need(
        re.fullmatch(r"[a-zA-Z0-9:-]{1,128}", wire.text(receipt["engine_id"], 128)) is not None
    )
    image = wire.text(receipt["image_id"], 71)
    wire.need(image.startswith("sha256:") and wire.sha(image[7:]) == image[7:])
    wire.sha(receipt["worker_bundle_sha256"])
    wire.count(receipt["elapsed_ms"], 0, 2**53 - 1)
    cleanup = wire.obj(
        receipt["cleanup"],
        {
            "container_waited",
            "stopped_pid_zero",
            "container_removed",
            "attach_reaped",
            "staging_removed",
            "exit_code",
            "oom_killed",
        },
    )
    wire.need(
        all(
            cleanup[k] is True
            for k in (
                "container_waited",
                "stopped_pid_zero",
                "container_removed",
                "attach_reaped",
                "staging_removed",
            )
        )
    )
    wire.need(
        type(cleanup["exit_code"]) is int
        and cleanup["exit_code"] == 0
        and cleanup["oom_killed"] is False
    )
    if kind == "query":
        wire.need(data["output"] is None and parquet is None)
        table = sql.result_table(data["table"])
        wire.need(
            table == data["table"]
            and cast("int", table["returned_rows"]) <= cast("int", spec["max_result_rows"])
        )
        wire.need(
            table["result_truncated"] is False or table["returned_rows"] == spec["max_result_rows"]
        )
    else:
        wire.need(data["table"] is None and type(parquet) is bytes)
        output = _description(data["output"])
        wire.need(output == data["output"] and len(cast("bytes", parquet)) == output["size_bytes"])
        wire.need(
            hashlib.sha256(cast("bytes", parquet)).hexdigest() == output["sha256"],
            "source_mismatch",
        )
    wire.need(
        wire.sha(receipt["result_sha256"])
        == sql.digest({"table": data["table"], "output": data["output"]}, limit=sql.MAX_RESULT)
    )
    wire.need(
        wire.sha(receipt["receipt_sha256"])
        == sql.digest(
            {k: v for k, v in receipt.items() if k != "receipt_sha256"}, limit=sql.MAX_RESULT
        )
    )
    wire.need(sql.canonical(data, limit=sql.MAX_RESULT) == canonical)
    return data


class DatasetResult(BaseModel):
    """Descriptive local execution result; hashes never grant upstream source access."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)
    canonical: bytes = Field(repr=False, exclude=True)
    parquet: bytes | None = Field(default=None, repr=False, exclude=True)

    def __init__(self, **data: object) -> None:
        failed = False
        try:
            super().__init__(**data)
            if type(self) is not DatasetResult:
                raise DatasetExecutionError("invalid_spec")
            _result(self.canonical, self.parquet)
        except Exception:
            failed = True
        if failed:
            raise DatasetExecutionError("invalid_spec")

    def wire_payload(self) -> dict[str, sql.JSON]:
        result: dict[str, sql.JSON] | None = None
        try:
            wire.need(type(self) is DatasetResult)
            result = _result(self.canonical, self.parquet)
        except Exception:
            result = None
        if result is None:
            raise DatasetExecutionError("invalid_spec")
        return result

    def output_bytes(self) -> bytes:
        """Return verified immutable bytes; consumers must budget/chunk them explicitly."""
        DatasetResult.wire_payload(self)
        if type(self.parquet) is not bytes:
            raise DatasetExecutionError("invalid_spec")
        return self.parquet


def _stage(
    parts: list[dict[str, sql.JSON]],
    fds: tuple[int, ...],
    directory: Path,
    deadline: float,
    stop: threading.Event,
) -> None:
    duplicates: list[int] = []
    try:
        for fd in fds:
            duplicate = os.dup(fd)
            duplicates.append(duplicate)
            wire.need(
                fcntl.fcntl(duplicate, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY,
                "source_mismatch",
            )
            wire.need(stat.S_ISREG(os.fstat(duplicate).st_mode), "source_mismatch")
        for index, (fd, part) in enumerate(zip(duplicates, parts, strict=True)):
            wire.need(os.fstat(fd).st_size == part["size_bytes"], "source_mismatch")
            hashed = hashlib.sha256()
            offset = 0
            with (directory / f"part-{index:06d}.bin").open("xb") as target:
                while True:
                    wire.need(not stop.is_set() and time.monotonic() < deadline, "deadline")
                    chunk = os.pread(fd, 65536, offset)
                    if not chunk:
                        break
                    offset += len(chunk)
                    wire.need(offset <= cast("int", part["size_bytes"]), "source_mismatch")
                    hashed.update(chunk)
                    target.write(chunk)
            wire.need(
                offset == part["size_bytes"] and hashed.hexdigest() == part["sha256"],
                "source_mismatch",
            )
            (directory / f"part-{index:06d}.bin").chmod(0o444)
        directory.chmod(0o555)
    finally:
        for fd in duplicates:
            os.close(fd)


def _output(directory: Path, description: dict[str, sql.JSON], deadline: float) -> bytes:
    wire.need({p.name for p in directory.iterdir()} == {"data.parquet"}, "source_mismatch")
    fd = os.open(directory / "data.parquet", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        wire.need(
            stat.S_ISREG(info.st_mode) and info.st_size == description["size_bytes"],
            "source_mismatch",
        )
        result = bytearray()
        hashed = hashlib.sha256()
        while chunk := os.read(fd, 65536):
            wire.need(time.monotonic() < deadline, "deadline")
            wire.need(
                len(result) + len(chunk) <= cast("int", description["size_bytes"]),
                "source_mismatch",
            )
            hashed.update(chunk)
            result.extend(chunk)
        wire.need(
            len(result) == description["size_bytes"]
            and hashed.hexdigest() == description["sha256"],
            "source_mismatch",
        )
        return bytes(result)
    finally:
        os.close(fd)


def _remove(directory: Path) -> None:
    for name in ("input", "output"):
        if (directory / name).exists():
            (directory / name).chmod(0o700)
    shutil.rmtree(directory)


class LocalDatasetExecutor:
    """One bounded local operation per selected owner; no remote authority or auto retry."""

    def __init__(self, *, installation: DatasetInstallation) -> None:
        if type(installation) is not DatasetInstallation:
            raise DatasetExecutionError("invalid_spec")
        self._installation = DatasetInstallation.wire_payload(installation)
        self._owner: SQLContainerOwner | None = None
        self._staging: Path | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_lock = threading.Lock()
        self._busy = False
        self._closed = False
        self._closing: asyncio.Task[bool] | None = None

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        with self._loop_lock:
            if self._loop is None:
                self._loop = loop
            elif self._loop is not loop:
                raise DatasetExecutionError("unavailable")

    async def _make_owner(self) -> None:
        self._owner = await asyncio.to_thread(
            SQLContainerOwner, self._installation, _profile=_Profile.DATASET
        )

    async def __aenter__(self) -> LocalDatasetExecutor:
        self._check_loop()
        if self._busy or self._closed or self._owner is not None:
            raise DatasetExecutionError("unavailable")
        self._busy = True
        making = asyncio.create_task(self._make_owner())
        cancelled = failed = False
        try:
            await asyncio.shield(making)
        except asyncio.CancelledError:
            cancelled = True
            with contextlib.suppress(Exception):
                await drain(cast("asyncio.Task[object]", making))
        except Exception:
            failed = True
        self._busy = False
        if cancelled or failed:
            await self.aclose()
            if cancelled:
                raise asyncio.CancelledError
            raise DatasetExecutionError("unavailable")
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def _clear(self) -> None:
        if self._staging is not None:
            await asyncio.to_thread(_remove, self._staging)
            self._staging = None

    async def _close(self) -> bool:
        failed = False
        try:
            if self._owner is not None:
                await self._owner.aclose()
            await self._clear()
        except Exception:
            failed = True
        return failed

    async def aclose(self) -> None:
        """Refuse active work; cleanup uncertainty keeps the retired owner unavailable."""
        self._check_loop()
        if self._busy:
            raise DatasetExecutionError("busy")
        self._closed = True
        caller = asyncio.current_task()
        cancels = caller.cancelling() if caller is not None else 0
        if self._closing is None or (self._closing.done() and self._closing.result()):
            self._closing = asyncio.create_task(self._close())
        failed = await drain(cast("asyncio.Task[object]", self._closing))
        if caller is not None and caller.cancelling() > cancels:
            raise asyncio.CancelledError
        if failed:
            raise DatasetExecutionError("cleanup_unresolved")

    async def convert(
        self, *, spec: DatasetConvertSpec, source: ProvidedDatasetFiles
    ) -> DatasetResult:
        """Validate complete provided bytes and produce normalized immutable Parquet."""
        if type(spec) is not DatasetConvertSpec or type(source) is not ProvidedDatasetFiles:
            raise DatasetExecutionError("invalid_spec")
        checked = DatasetConvertSpec.wire_payload(spec)
        manifest, fds = ProvidedDatasetFiles.checked(source)
        if checked["format"] != manifest["format"] or checked[
            "input_manifest_sha256"
        ] != sql.digest(manifest):
            raise DatasetExecutionError("source_mismatch")
        return await self._execute("convert", checked, manifest, fds)

    async def query(
        self, *, spec: DatasetQuerySpec, sources: tuple[NamedParquetFiles, ...]
    ) -> DatasetResult:
        """Observe one bounded typed result against complete pinned named inputs."""
        return await self._named("query", spec, sources)

    async def materialize(
        self, *, spec: DatasetQuerySpec, sources: tuple[NamedParquetFiles, ...]
    ) -> DatasetResult:
        """Materialize one SELECT result into bounded immutable Parquet bytes."""
        return await self._named("materialize", spec, sources)

    async def _named(
        self, kind: str, spec: DatasetQuerySpec, sources: tuple[NamedParquetFiles, ...]
    ) -> DatasetResult:
        inputs: list[sql.JSON] = []
        fds: tuple[int, ...] = ()
        checked: dict[str, sql.JSON] | None = None
        try:
            wire.need(
                type(spec) is DatasetQuerySpec and type(sources) is tuple and 1 <= len(sources) <= 4
            )
            for source in sources:
                item, descriptors = NamedParquetFiles.checked(source)
                inputs.append(item)
                fds += descriptors
            inputs = wire.named_inputs(inputs)
            checked = DatasetQuerySpec.wire_payload(spec)
            wire.need(checked["inputs_sha256"] == sql.digest(inputs), "source_mismatch")
        except Exception:
            checked = None
        if checked is None:
            raise DatasetExecutionError("invalid_spec")
        return await self._execute(kind, checked, inputs, fds)

    def _make_staging(self) -> Path:
        path = Path(tempfile.mkdtemp(prefix="strata-dataset-", dir="/tmp")).resolve(strict=True)
        self._staging = path
        (path / "input").mkdir(mode=0o700)
        # The private 0700 parent protects the host path; only this empty child is
        # writable by the fixed unprivileged UID through its exact container mount.
        (path / "output").mkdir(mode=0o777)
        (path / "output").chmod(0o777)
        return path

    async def _execute(
        self, kind: str, spec: dict[str, sql.JSON], inputs: sql.JSON, fds: tuple[int, ...]
    ) -> DatasetResult:
        self._check_loop()
        if self._busy:
            raise DatasetExecutionError("busy")
        if (
            self._closed
            or self._owner is None
            or not self._owner.available
            or self._staging is not None
        ):
            raise DatasetExecutionError("unavailable")
        self._busy = True
        started = time.monotonic()
        deadline = started + cast("int", spec["timeout_ms"]) / 1000
        stop = threading.Event()
        result: DatasetResult | None = None
        cancelled = False
        code = "worker_failed"
        try:
            # Loop scheduling uses its own epoch; eligibility retains this original clock.
            async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
                hashing = asyncio.create_task(asyncio.to_thread(bundle_sha256))
                try:
                    bundle = await asyncio.shield(hashing)
                except asyncio.CancelledError:
                    await drain(cast("asyncio.Task[object]", hashing))
                    raise
                wire.need(bundle == self._installation["worker_bundle_sha256"], "profile_refused")
                making = asyncio.create_task(asyncio.to_thread(self._make_staging))
                try:
                    path = await asyncio.shield(making)
                except asyncio.CancelledError:
                    await drain(cast("asyncio.Task[object]", making))
                    raise
                if kind == "convert":
                    parts = cast(
                        "list[dict[str, sql.JSON]]", cast("dict[str, sql.JSON]", inputs)["parts"]
                    )
                else:
                    parts = [
                        cast("dict[str, sql.JSON]", part)
                        for item in cast("list[dict[str, sql.JSON]]", inputs)
                        for part in cast(
                            "list[sql.JSON]", cast("dict[str, sql.JSON]", item["manifest"])["parts"]
                        )
                    ]
                copying = asyncio.create_task(
                    asyncio.to_thread(_stage, parts, fds, path / "input", deadline, stop)
                )
                try:
                    await asyncio.shield(copying)
                except asyncio.CancelledError:
                    stop.set()
                    with contextlib.suppress(Exception):
                        await drain(cast("asyncio.Task[object]", copying))
                    raise
                operation = uuid.uuid4().hex
                unsigned: dict[str, sql.JSON] = {
                    "version": 1,
                    "kind": kind,
                    "operation_id": operation,
                    "spec": spec,
                    "inputs": inputs,
                }
                request_sha = sql.digest(unsigned, limit=sql.MAX_REQUEST)
                outcome = await self._owner.run(
                    input_dir=path / "input",
                    output_dir=path / "output",
                    request=sql.canonical(
                        {**unsigned, "request_sha256": request_sha}, limit=sql.MAX_REQUEST
                    ),
                    operation_id=operation,
                    deadline=deadline,
                )
                data = wire.obj(
                    sql.parse(outcome.payload, limit=sql.MAX_RESULT),
                    {"version", "kind", "operation_id", "request_sha256", "code"}
                    if outcome.exit_code == 1
                    else {
                        "version",
                        "kind",
                        "operation_id",
                        "request_sha256",
                        "bundle_sha256",
                        "table",
                        "output",
                    },
                )
                wire.need(
                    type(data["version"]) is int
                    and data["version"] == 1
                    and data["operation_id"] == operation
                    and data["request_sha256"] == request_sha,
                    "source_mismatch",
                )
                if outcome.exit_code == 1:
                    wire.need(data["kind"] == "error", "worker_failed")
                    raise sql.WireError(data["code"])
                wire.need(
                    type(outcome.exit_code) is int
                    and outcome.exit_code == 0
                    and outcome.oom_killed is False
                    and data["kind"] == "result"
                    and data["bundle_sha256"] == bundle,
                    "worker_failed",
                )
                parquet: bytes | None = None
                if kind == "query":
                    wire.need(
                        data["output"] is None and not list((path / "output").iterdir()),
                        "source_mismatch",
                    )
                    data["table"] = sql.result_table(data["table"])
                else:
                    wire.need(data["table"] is None)
                    description = _description(data["output"])
                    reading = asyncio.create_task(
                        asyncio.to_thread(_output, path / "output", description, deadline)
                    )
                    try:
                        parquet = await asyncio.shield(reading)
                    except asyncio.CancelledError:
                        await drain(cast("asyncio.Task[object]", reading))
                        raise
                removing = asyncio.create_task(self._clear())
                try:
                    await asyncio.shield(removing)
                except asyncio.CancelledError:
                    await drain(cast("asyncio.Task[object]", removing))
                    raise
                receipt: dict[str, sql.JSON] = {
                    "version": 1,
                    "operation_id": operation,
                    "request_sha256": request_sha,
                    "spec": spec,
                    "inputs": inputs,
                    "profile": wire.PROFILE,
                    "engine_id": outcome.engine_id,
                    "image_id": outcome.image_id,
                    "worker_bundle_sha256": bundle,
                    "result_sha256": sql.digest(
                        {"table": data["table"], "output": data["output"]}, limit=sql.MAX_RESULT
                    ),
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                    "cleanup": {
                        "container_waited": True,
                        "stopped_pid_zero": True,
                        "container_removed": True,
                        "attach_reaped": True,
                        "staging_removed": True,
                        "exit_code": 0,
                        "oom_killed": False,
                    },
                }
                receipt["receipt_sha256"] = sql.digest(receipt, limit=sql.MAX_RESULT)
                result = DatasetResult(
                    canonical=sql.canonical(
                        {
                            "version": 1,
                            "kind": kind,
                            "table": data["table"],
                            "output": data["output"],
                            "receipt": receipt,
                        },
                        limit=sql.MAX_RESULT,
                    ),
                    parquet=parquet,
                )
                wire.need(time.monotonic() < deadline, "deadline")
        except asyncio.CancelledError:
            cancelled = True
        except TimeoutError:
            code = "deadline"
            result = None
        except sql.WireError as refusal:
            code = refusal.code
            result = None
        except DatasetExecutionError as refusal:
            code = refusal.code
            result = None
        except Exception:
            code = "worker_failed"
            result = None
        caller = asyncio.current_task()
        cancels = caller.cancelling() if caller is not None else 0
        if self._staging is not None and self._owner.available:
            try:
                await drain(asyncio.create_task(self._clear()))
            except Exception:
                code = "cleanup_unresolved"
                result = None
        self._busy = False
        cancelled = cancelled or (caller is not None and caller.cancelling() > cancels)
        if cancelled:
            raise asyncio.CancelledError
        if result is None:
            raise DatasetExecutionError(code)
        return result
