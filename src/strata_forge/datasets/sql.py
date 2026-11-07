"""Local-only isolated SQL over a complete explicitly provided pinned Parquet population."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import os
import shutil
import stat
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.datasets import _sql_wire as wire
from strata_forge.datasets._sql_container import SQLContainerOwner, drain
from strata_forge.datasets._sql_worker import bundle_sha256
from strata_forge.datasets.sql_types import (
    LocalParquetFiles,
    LocalSQLInstallation,
    SQLExecutionError,
    SQLInputManifest,
    SQLQuerySpec,
)

__all__ = ["LocalSQLExecutor", "SQLResult"]


def _need(value: bool, code: str = "worker_failed") -> None:
    if not value:
        raise wire.WireError(code)


def _result(value: object) -> dict[str, wire.JSON]:
    _need(type(value) is dict)
    data = cast("dict[str, wire.JSON]", value)
    _need(
        set(data) == {"version", "table", "receipt"}
        and type(data["version"]) is int
        and data["version"] == 1
    )
    table = wire.result_table(data["table"])
    _need(type(data["receipt"]) is dict)
    receipt = cast("dict[str, wire.JSON]", data["receipt"])
    _need(
        set(receipt)
        == {
            "version",
            "operation_id",
            "request_sha256",
            "spec_sha256",
            "sql_sha256",
            "manifest_sha256",
            "manifest",
            "result_sha256",
            "profile",
            "engine_id",
            "image_id",
            "worker_bundle_sha256",
            "source_observed",
            "elapsed_ms",
            "staging_ms",
            "container_ms",
            "cleanup",
            "receipt_sha256",
        }
    )
    _need(type(receipt["version"]) is int and receipt["version"] == 1)
    for key in (
        "request_sha256",
        "spec_sha256",
        "sql_sha256",
        "manifest_sha256",
        "result_sha256",
        "worker_bundle_sha256",
        "receipt_sha256",
    ):
        item = receipt[key]
        _need(type(item) is str and len(item) == 64 and all(c in "0123456789abcdef" for c in item))
    operation = receipt["operation_id"]
    _need(
        type(operation) is str
        and len(operation) == 32
        and all(c in "0123456789abcdef" for c in operation)
    )
    manifest = wire.manifest(receipt["manifest"])
    _need(receipt["manifest_sha256"] == wire.digest(manifest))
    _need(receipt["result_sha256"] == wire.digest(table, limit=wire.MAX_RESULT))
    _need(receipt["profile"] == wire.PROFILE)
    image = receipt["image_id"]
    _need(
        type(image) is str
        and len(image) == 71
        and image.startswith("sha256:")
        and all(c in "0123456789abcdef" for c in image[7:])
    )
    engine = receipt["engine_id"]
    _need(
        type(engine) is str
        and 1 <= len(engine) <= 128
        and all(
            c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:-" for c in engine
        )
    )
    for key in ("elapsed_ms", "staging_ms", "container_ms"):
        _need(type(receipt[key]) is int and 0 <= cast("int", receipt[key]) <= 10000)
    parts = cast("list[dict[str, wire.JSON]]", manifest["parts"])
    expected: dict[str, wire.JSON] = {
        "parts": manifest["part_count"],
        "rows": manifest["expected_rows"],
        "bytes": sum(cast("int", part["size_bytes"]) for part in parts),
        "schema_sha256": manifest["schema_sha256"],
    }
    _need(
        wire.canonical(receipt["source_observed"], limit=8192)
        == wire.canonical(expected, limit=8192)
    )
    cleanup = {
        "container_waited": True,
        "stopped_pid_zero": True,
        "container_removed": True,
        "attach_reaped": True,
        "staging_removed": True,
        "exit_code": 0,
        "oom_killed": False,
    }
    _need(wire.canonical(receipt["cleanup"], limit=8192) == wire.canonical(cleanup, limit=8192))
    unsigned = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    _need(receipt["receipt_sha256"] == wire.digest(unsigned, limit=wire.MAX_RESULT))
    result: dict[str, wire.JSON] = {"version": 1, "table": table, "receipt": receipt.copy()}
    wire.canonical(result, limit=wire.MAX_RESULT)
    return result


class SQLResult(BaseModel):
    """Canonical immutable content and descriptive receipt, never source authorization."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)
    canonical: bytes = Field(repr=False)

    def __init__(self, **data: object) -> None:
        failed = False
        try:
            super().__init__(**data)
            self.wire_payload()
        except Exception:
            failed = True
        if failed:
            raise SQLExecutionError("worker_failed")

    def wire_payload(self) -> dict[str, wire.JSON]:
        value: dict[str, wire.JSON] | None = None
        try:
            _need(type(self) is SQLResult and type(self.canonical) is bytes)
            value = _result(wire.parse(self.canonical, limit=wire.MAX_RESULT))
            _need(wire.canonical(value, limit=wire.MAX_RESULT) == self.canonical)
        except AttributeError, wire.WireError:
            value = None
        if value is None:
            raise SQLExecutionError("worker_failed")
        return value


def _stage(
    source: LocalParquetFiles, directory: Path, deadline: float, stop: threading.Event
) -> None:
    manifest, descriptors = LocalParquetFiles.checked(source)
    parts = cast("list[dict[str, wire.JSON]]", SQLInputManifest.wire_payload(manifest)["parts"])
    copied: list[int] = []
    try:
        for descriptor in descriptors:
            duplicate = os.dup(descriptor)
            copied.append(duplicate)
            _need(
                fcntl.fcntl(duplicate, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY,
                "source_mismatch",
            )
            _need(stat.S_ISREG(os.fstat(duplicate).st_mode), "source_mismatch")
        for ordinal, (descriptor, part) in enumerate(zip(copied, parts, strict=True)):
            info = os.fstat(descriptor)
            _need(info.st_size == part["size_bytes"], "source_mismatch")
            hashed = hashlib.sha256()
            offset = 0
            target = directory / f"part-{ordinal:06d}.parquet"
            with target.open("xb") as output:
                while True:
                    _need(not stop.is_set() and time.monotonic() < deadline, "deadline")
                    chunk = os.pread(descriptor, 65536, offset)
                    if not chunk:
                        break
                    offset += len(chunk)
                    _need(offset <= cast("int", part["size_bytes"]), "source_mismatch")
                    hashed.update(chunk)
                    output.write(chunk)
            _need(
                offset == part["size_bytes"] and hashed.hexdigest() == part["sha256"],
                "source_mismatch",
            )
            target.chmod(0o444)
        directory.chmod(0o555)
    finally:
        for descriptor in copied:
            os.close(descriptor)


def _remove(directory: Path) -> None:
    directory.chmod(0o700)
    if (directory / "input").exists():
        (directory / "input").chmod(0o700)
    shutil.rmtree(directory)


def _response(
    payload: bytes,
    *,
    operation: str,
    request_sha: str,
    manifest: dict[str, wire.JSON],
    bundle: str,
    max_rows: int,
) -> dict[str, wire.JSON]:
    decoded = wire.parse(payload, limit=wire.MAX_RESULT)
    _need(type(decoded) is dict)
    data = cast("dict[str, wire.JSON]", decoded)
    if data.get("kind") == "error":
        _need(set(data) == {"version", "kind", "operation_id", "request_sha256", "code"})
        _need(data["operation_id"] == operation and data["request_sha256"] == request_sha)
        raise wire.WireError(data["code"])
    _need(
        set(data)
        == {
            "version",
            "kind",
            "operation_id",
            "request_sha256",
            "manifest_sha256",
            "table",
            "bundle_sha256",
            "source_observed",
        }
    )
    _need(type(data["version"]) is int and data["version"] == 1 and data["kind"] == "result")
    _need(
        data["operation_id"] == operation
        and data["request_sha256"] == request_sha
        and data["manifest_sha256"] == wire.digest(manifest)
        and data["bundle_sha256"] == bundle
    )
    expected: dict[str, wire.JSON] = {
        "parts": manifest["part_count"],
        "rows": manifest["expected_rows"],
        "bytes": sum(
            cast("int", part["size_bytes"])
            for part in cast("list[dict[str, wire.JSON]]", manifest["parts"])
        ),
        "schema_sha256": manifest["schema_sha256"],
    }
    _need(
        wire.canonical(data["source_observed"], limit=8192) == wire.canonical(expected, limit=8192)
    )
    table = wire.result_table(data["table"])
    _need(
        cast("int", table["returned_rows"]) <= max_rows
        and (table["result_truncated"] is False or table["returned_rows"] == max_rows)
    )
    return table


class LocalSQLExecutor:
    """Trusted local operator operation, one query at a time, no queue or remote authority."""

    def __init__(self, *, installation: LocalSQLInstallation) -> None:
        if type(installation) is not LocalSQLInstallation:
            raise SQLExecutionError("invalid_spec")
        self._installation = LocalSQLInstallation.wire_payload(installation)
        self._owner: SQLContainerOwner | None = None
        self._busy = False
        self._closed = False
        self._staging: Path | None = None
        self._closing: asyncio.Task[bool] | None = None

    async def _make_staging(self) -> None:
        def make() -> None:
            self._staging = Path(tempfile.mkdtemp(prefix="strata-sql-input-", dir="/private/tmp"))
            (self._staging / "input").mkdir(mode=0o700)

        await asyncio.to_thread(make)

    async def _clear_staging(self) -> None:
        if self._staging is not None:
            await asyncio.to_thread(_remove, self._staging)
            self._staging = None

    async def _make_owner(self) -> None:
        self._owner = await asyncio.to_thread(SQLContainerOwner, self._installation)

    async def __aenter__(self) -> LocalSQLExecutor:
        if self._owner is not None or self._closed or self._busy:
            raise SQLExecutionError("unavailable")
        self._busy = True
        starting = asyncio.create_task(self._make_owner())
        failed = False
        try:
            await asyncio.shield(starting)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await drain(cast("asyncio.Task[object]", starting))
                self._busy = False
                await self.aclose()
            raise
        except Exception:
            failed = True
        finally:
            self._busy = False
        if failed:
            raise SQLExecutionError("unavailable")
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def _close(self) -> bool:
        failed = False
        if self._owner is not None:
            try:
                await self._owner.aclose()
            except Exception:
                failed = True
        if self._staging is not None and not failed:
            try:
                await self._clear_staging()
            except Exception:
                failed = True
        return failed

    async def aclose(self) -> None:
        if self._busy:
            raise SQLExecutionError("busy")
        self._closed = True
        caller = asyncio.current_task()
        cancellations = caller.cancelling() if caller is not None else 0
        if self._closing is None or (self._closing.done() and self._closing.result()):
            self._closing = asyncio.create_task(self._close())
        failed = await drain(cast("asyncio.Task[object]", self._closing))
        if caller is not None and caller.cancelling() > cancellations:
            raise asyncio.CancelledError
        if failed:
            raise SQLExecutionError("cleanup_unresolved")

    async def query(self, *, spec: SQLQuerySpec, source: LocalParquetFiles) -> SQLResult:
        started = time.monotonic()
        if self._busy:
            raise SQLExecutionError("busy")
        if (
            self._closed
            or self._owner is None
            or not self._owner.available
            or self._staging is not None
        ):
            raise SQLExecutionError("unavailable")
        if type(spec) is not SQLQuerySpec or type(source) is not LocalParquetFiles:
            raise SQLExecutionError("invalid_spec")
        checked = SQLQuerySpec.wire_payload(spec)
        source_manifest, _ = LocalParquetFiles.checked(source)
        manifest = SQLInputManifest.wire_payload(source_manifest)
        if checked["input_manifest_sha256"] != wire.digest(manifest):
            raise SQLExecutionError("source_mismatch")
        deadline = started + cast("int", checked["timeout_ms"]) / 1000
        self._busy = True
        stop = threading.Event()
        cancelled = False
        code = "worker_failed"
        result: SQLResult | None = None
        try:
            async with asyncio.timeout_at(deadline):
                hashing = asyncio.create_task(asyncio.to_thread(bundle_sha256))
                try:
                    bundle = await asyncio.shield(hashing)
                except asyncio.CancelledError:
                    await drain(cast("asyncio.Task[object]", hashing))
                    raise
                _need(bundle == self._installation["worker_bundle_sha256"], "profile_refused")
                making = asyncio.create_task(self._make_staging())
                try:
                    await asyncio.shield(making)
                except asyncio.CancelledError:
                    await drain(cast("asyncio.Task[object]", making))
                    raise
                _need(self._staging is not None)
                staging = cast("Path", self._staging) / "input"
                copying = asyncio.create_task(
                    asyncio.to_thread(_stage, source, staging, deadline, stop)
                )
                try:
                    await asyncio.shield(copying)
                except asyncio.CancelledError:
                    stop.set()
                    with contextlib.suppress(Exception):
                        await drain(cast("asyncio.Task[object]", copying))
                    raise
                staged = time.monotonic()
                operation = uuid.uuid4().hex
                unsigned: dict[str, wire.JSON] = {
                    "version": 1,
                    "operation_id": operation,
                    "query": checked,
                    "manifest": manifest,
                }
                request_sha = wire.digest(unsigned, limit=wire.MAX_REQUEST)
                request = wire.canonical(
                    {**unsigned, "request_sha256": request_sha}, limit=wire.MAX_REQUEST
                )
                outcome = await self._owner.run(
                    input_dir=staging,
                    request=request,
                    operation_id=operation,
                    deadline=deadline,
                )
                finished = time.monotonic()
                _need(outcome.exit_code in (0, 1) and outcome.oom_killed is False)
                table = _response(
                    outcome.payload,
                    operation=operation,
                    request_sha=request_sha,
                    manifest=manifest,
                    bundle=bundle,
                    max_rows=cast("int", checked["max_result_rows"]),
                )
                removing = asyncio.create_task(self._clear_staging())
                try:
                    await asyncio.shield(removing)
                except asyncio.CancelledError:
                    await drain(cast("asyncio.Task[object]", removing))
                    raise
                receipt: dict[str, wire.JSON] = {
                    "version": 1,
                    "operation_id": operation,
                    "request_sha256": request_sha,
                    "spec_sha256": SQLQuerySpec.digest(spec),
                    "sql_sha256": hashlib.sha256(
                        cast("str", checked["sql"]).encode("utf-8")
                    ).hexdigest(),
                    "manifest_sha256": wire.digest(manifest),
                    "manifest": manifest,
                    "result_sha256": wire.digest(table, limit=wire.MAX_RESULT),
                    "profile": wire.PROFILE,
                    "engine_id": outcome.engine_id,
                    "image_id": outcome.image_id,
                    "worker_bundle_sha256": bundle,
                    "source_observed": {
                        "parts": manifest["part_count"],
                        "rows": manifest["expected_rows"],
                        "bytes": sum(
                            cast("int", part["size_bytes"])
                            for part in cast("list[dict[str, wire.JSON]]", manifest["parts"])
                        ),
                        "schema_sha256": manifest["schema_sha256"],
                    },
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                    "staging_ms": int((staged - started) * 1000),
                    "container_ms": int((finished - staged) * 1000),
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
                receipt["receipt_sha256"] = wire.digest(receipt, limit=wire.MAX_RESULT)
                result = SQLResult(
                    canonical=wire.canonical(
                        {"version": 1, "table": table, "receipt": receipt}, limit=wire.MAX_RESULT
                    )
                )
                _need(time.monotonic() < deadline, "deadline")
        except asyncio.CancelledError:
            cancelled = True
        except TimeoutError:
            code = "deadline"
        except wire.WireError as refusal:
            code = refusal.code
        except SQLExecutionError as refusal:
            code = refusal.code
        except Exception:
            code = "worker_failed"
        caller = asyncio.current_task()
        cancellations = caller.cancelling() if caller is not None else 0
        if self._staging is not None and self._owner.available:
            try:
                await drain(asyncio.create_task(self._clear_staging()))
            except Exception:
                code = "cleanup_unresolved"
                result = None
        self._busy = False
        cancelled = cancelled or (caller is not None and caller.cancelling() > cancellations)
        if cancelled:
            raise asyncio.CancelledError
        if result is None:
            raise SQLExecutionError(code)
        return result
