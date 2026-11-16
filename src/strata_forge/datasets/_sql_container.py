"""Owned fixed-profile Docker lifecycle; native diagnostics never leave this leaf."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import cast
from urllib.parse import quote

from strata_forge.datasets import _sql_wire as wire
from strata_forge.datasets._sql_worker import expected_ready

_ID = re.compile(r"[0-9a-f]{64}")
_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin"}
_FLAGS = (
    "--pull=never",
    "--network",
    "none",
    "--read-only",
    "--user",
    "65532:65532",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--memory",
    "256m",
    "--memory-swap",
    "256m",
    "--cpus",
    "1",
    "--pids-limit",
    "32",
    "--log-driver",
    "none",
    "--ulimit",
    "nofile=64:64",
    "--ulimit",
    "fsize=33554432:33554432",
    "--ulimit",
    "core=0:0",
    "--tmpfs",
    "/tmp:rw,nosuid,nodev,noexec,size=67108864,mode=0700,uid=65532,gid=65532",  # noqa: S108 — private bounded container tmpfs, not a host temporary path.
)


class _Profile(Enum):
    SQL = "sql_local_arm64_v1"
    DATASET = "dataset_transform_arm64_v1"


_DATASET_FLAGS = tuple(
    "512m"
    if value == "256m"
    else "fsize=67108864:67108864"
    if value == "fsize=33554432:33554432"
    else value.replace("size=67108864", "size=134217728")
    if value == _FLAGS[-1]
    else value
    for value in _FLAGS
)


async def drain(task: asyncio.Task[object]) -> object:
    """Finish owned cleanup despite repeated caller cancellation; never detach it."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def _reap_cli(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    if process.stdout is not None:
        while await process.stdout.read(65536):
            continue
    await process.wait()


async def _read_all(reader: asyncio.StreamReader, limit: int) -> bytes:
    result = bytearray()
    while chunk := await reader.read(65536):
        if len(result) + len(chunk) > limit:
            raise wire.WireError("output_limit")
        result.extend(chunk)
    return bytes(result)


@dataclass(frozen=True)
class ContainerSettlement:
    """Observed fixed-owner cleanup facts without process or source references."""

    operation_id: str
    container_waited: bool
    stopped_pid_zero: bool
    container_removed: bool
    attach_reaped: bool


@dataclass(frozen=True)
class ContainerOutcome:
    payload: bytes
    engine_id: str
    image_id: str
    bundle_sha256: str
    exit_code: int
    oom_killed: bool
    settlement: ContainerSettlement | None = None


class SQLContainerOwner:
    """One local owner; uncertainty retains its lease rather than adopting new work."""

    def __init__(
        self, installation: dict[str, wire.JSON], *, _profile: _Profile = _Profile.SQL
    ) -> None:
        if type(_profile) is not _Profile:
            raise wire.WireError("profile_refused")
        self._profile = _profile
        if _profile is _Profile.DATASET:
            from strata_forge.datasets import _dataset_wire

            self._installation = _dataset_wire.installation(installation)
        else:
            self._installation = wire.installation(installation)
        self._flags = _DATASET_FLAGS if _profile is _Profile.DATASET else _FLAGS
        self._output_dir: Path | None = None
        self._directory = Path(tempfile.mkdtemp(prefix="strata-sql-cli-"))
        self._directory.chmod(0o700)
        self._command_prefix = (
            cast("str", self._installation["docker_executable"]),
            "--host",
            "unix://" + quote(cast("str", self._installation["engine_socket"]), safe="/"),
            "--config",
            str(self._directory),
        )
        self._owner = uuid.uuid4().hex
        self._name = ""
        self._cid = ""
        self._operation = ""
        self._engine = ""
        self._identity_mismatch = False
        self._create_issued = False
        self._start_issued = False
        self._waited = False
        self._clean = True
        self._closed = False
        self._attach: asyncio.subprocess.Process | None = None
        self._stopped_pid_zero = self._removed = self._attach_reaped = False

    @property
    def available(self) -> bool:
        return self._clean and not self._closed and not self._identity_mismatch

    async def _spawn(
        self, args: tuple[str, ...], *, pipe: bool = False
    ) -> asyncio.subprocess.Process:
        spawning = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *self._command_prefix,
                *args,
                env=_ENV,
                start_new_session=True,
                stdin=asyncio.subprocess.PIPE if pipe else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        )
        try:
            return await asyncio.shield(spawning)
        except asyncio.CancelledError:
            process = cast(
                "asyncio.subprocess.Process", await drain(cast("asyncio.Task[object]", spawning))
            )
            await drain(asyncio.create_task(_reap_cli(process)))
            raise

    async def _command(self, *args: str, deadline: float) -> bytes:
        process: asyncio.subprocess.Process | None = None
        result: bytes | None = None
        failed = False
        try:
            async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
                process = await self._spawn(args)
                if process.stdout is None:
                    raise wire.WireError("unavailable")
                result = await _read_all(process.stdout, 131072)
                failed = await process.wait() != 0
        except asyncio.CancelledError:
            if process is not None:
                await drain(asyncio.create_task(_reap_cli(process)))
            raise
        except Exception:
            failed = True
        finally:
            if process is not None and process.returncode is None:
                await drain(asyncio.create_task(_reap_cli(process)))
        if failed or result is None:
            raise wire.WireError("unavailable")
        return result

    async def _engine_id(self, deadline: float) -> str:
        raw = await self._command("info", "--format", "{{.ID}}", deadline=deadline)
        value = raw.rstrip(b"\n")
        if not 1 <= len(value) <= 128 or any(
            c not in b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:-"
            for c in value
        ):
            raise wire.WireError("profile_refused")
        return value.decode("ascii")

    async def _same_engine(self, deadline: float) -> None:
        if await self._engine_id(deadline) != self._engine:
            self._identity_mismatch = True
            raise wire.WireError("cleanup_unresolved")

    async def _inspect(self, name: str, deadline: float) -> dict[str, object]:
        raw = await self._command("container", "inspect", name, deadline=deadline)
        value: object = json.loads(raw)
        if (
            type(value) is not list
            or len(cast("list[object]", value)) != 1
            or type(cast("list[object]", value)[0]) is not dict
        ):
            raise wire.WireError("cleanup_unresolved")
        return cast("dict[str, object]", value[0])

    def _owned(self, value: dict[str, object]) -> bool:
        config = value.get("Config")
        if type(config) is not dict:
            return False
        labels = cast("dict[str, object]", config).get("Labels")
        return (
            type(labels) is dict
            and cast("dict[str, object]", labels).get("org.strata.sql.owner") == self._owner
            and cast("dict[str, object]", labels).get("org.strata.sql.operation") == self._operation
            and value.get("Image") == self._installation["image_id"]
        )

    def _admitted(self, value: dict[str, object], input_dir: Path) -> bool:
        if not self._owned(value):
            return False
        host, config, mounts = value.get("HostConfig"), value.get("Config"), value.get("Mounts")
        if type(host) is not dict or type(config) is not dict or type(mounts) is not list:
            return False
        host = cast("dict[str, object]", host)
        config = cast("dict[str, object]", config)
        mounts = cast("list[object]", mounts)
        expected: dict[str, object] = {
            "NetworkMode": "none",
            "ReadonlyRootfs": True,
            "Memory": 536870912 if self._profile is _Profile.DATASET else 268435456,
            "MemorySwap": 536870912 if self._profile is _Profile.DATASET else 268435456,
            "NanoCpus": 1000000000,
            "PidsLimit": 32,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            "LogConfig": {"Type": "none", "Config": {}},
            "Tmpfs": {"/tmp": self._flags[-1].split(":", 1)[1]},  # noqa: S108 — exact container-private tmpfs readback.
        }
        if any(
            type(host.get(key)) is not type(item) or host.get(key) != item
            for key, item in expected.items()
        ):
            return False
        if (
            config.get("User") != "65532:65532"
            or config.get("Entrypoint")
            != [
                "python",
                "-I",
                "-B",
                "/app/_dataset_worker.py"
                if self._profile is _Profile.DATASET
                else "/app/_sql_worker.py",
            ]
            or config.get("Cmd") != ["--operation", self._operation]
        ):
            return False
        if (
            len(mounts) != (2 if self._profile is _Profile.DATASET else 1)
            or type(mounts[0]) is not dict
        ):
            return False
        if self._profile is _Profile.DATASET:
            expected_output = [
                cast("dict[str, object]", m)
                for m in mounts
                if type(m) is dict and cast("dict[str, object]", m).get("Destination") == "/output"
            ]
            if len(expected_output) != 1 or self._output_dir is None:
                return False
            output = expected_output[0]
            if (
                output.get("Type") != "bind"
                or output.get("Source") != str(self._output_dir)
                or output.get("RW") is not True
            ):
                return False
            mounts = [
                m
                for m in mounts
                if type(m) is dict and cast("dict[str, object]", m).get("Destination") == "/input"
            ]
            if len(mounts) != 1:
                return False
        mount = cast("dict[str, object]", mounts[0])
        return (
            mount.get("Type") == "bind"
            and mount.get("Source") == str(input_dir)
            and mount.get("Destination") == "/input"
            and mount.get("RW") is False
        )

    async def _cleanup(self) -> bool:
        # Cleanup may outlast the request budget. A timeout cannot prove process death.
        deadline = time.monotonic() + 5
        okay = False
        try:
            if not self._create_issued:
                if self._identity_mismatch:
                    self._clean = False
                    return False
                self._clean = True
                return True
            await self._same_engine(deadline)
            if not self._cid and self._name:
                raw = await self._command(
                    "container",
                    "ls",
                    "--all",
                    "--filter",
                    "name=^/" + self._name + "$",
                    "--no-trunc",
                    "--format",
                    "{{.ID}}",
                    deadline=deadline,
                )
                if raw == b"":
                    # No start was issued, but daemon create may still be pending after
                    # a lost acknowledgement. Empty lookup is not completed cleanup.
                    okay = False
                elif _ID.fullmatch(raw.decode("ascii").strip()):
                    candidate = raw.decode("ascii").strip()
                    if not self._owned(await self._inspect(candidate, deadline)):
                        raise wire.WireError("cleanup_unresolved")
                    self._cid = candidate
            if self._cid:
                state = await self._inspect(self._cid, deadline)
                if not self._owned(state):
                    raise wire.WireError("cleanup_unresolved")
                details = cast("dict[str, object]", state["State"])
                if details.get("Running") is True:
                    await self._command("kill", self._cid, deadline=deadline)
                if self._start_issued and not self._waited:
                    waited = await self._command("wait", self._cid, deadline=deadline)
                    if not re.fullmatch(rb"[0-9]{1,3}\n", waited):
                        raise wire.WireError("cleanup_unresolved")
                    self._waited = True
                state = await self._inspect(self._cid, deadline)
                details = cast("dict[str, object]", state["State"])
                if (
                    details.get("Running") is not False
                    or type(details.get("Pid")) is not int
                    or details.get("Pid") != 0
                ):
                    raise wire.WireError("cleanup_unresolved")
                self._stopped_pid_zero = True
                await self._same_engine(deadline)
                await self._command("rm", self._cid, deadline=deadline)
                await self._same_engine(deadline)
                absent = await self._command(
                    "container",
                    "ls",
                    "--all",
                    "--filter",
                    "id=" + self._cid,
                    "--no-trunc",
                    "--format",
                    "{{.ID}}",
                    deadline=deadline,
                )
                await self._same_engine(deadline)
                okay = absent == b"" and (not self._start_issued or self._waited)
                self._removed = okay
        except Exception:
            okay = False
        finally:
            if self._attach is not None:
                try:
                    await _reap_cli(self._attach)
                    self._attach_reaped = self._attach.returncode is not None
                except Exception:
                    okay = False
        self._clean = okay
        return okay

    async def _frame(self, reader: asyncio.StreamReader, limit: int) -> bytes:
        import struct

        prefix = await reader.readexactly(4)
        size = cast("int", struct.unpack("!I", prefix)[0])
        if not 0 < size <= limit:
            raise wire.WireError("output_limit")
        return await reader.readexactly(size)

    async def run(
        self,
        *,
        input_dir: Path,
        request: bytes,
        operation_id: str,
        deadline: float,
        output_dir: Path | None = None,
    ) -> ContainerOutcome:
        if not self.available:
            raise wire.WireError("unavailable")
        if (self._profile is _Profile.DATASET) != (type(output_dir) is type(input_dir)):
            raise wire.WireError("profile_refused")
        if self._profile is _Profile.SQL and output_dir is not None:
            raise wire.WireError("profile_refused")
        self._output_dir = output_dir
        self._clean = False
        self._cid = ""
        self._operation = operation_id
        self._name = "strata-sql-" + self._owner + "-" + operation_id
        self._create_issued = self._start_issued = self._waited = False
        self._stopped_pid_zero = self._removed = self._attach_reaped = False
        self._attach = None
        outcome: ContainerOutcome | None = None
        code = "unavailable"
        cancelled = False
        try:
            async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
                self._engine = await self._engine_id(deadline)
                image_raw = await self._command(
                    "image",
                    "inspect",
                    cast("str", self._installation["image_id"]),
                    deadline=deadline,
                )
                images: object = json.loads(image_raw)
                if (
                    type(images) is not list
                    or len(cast("list[object]", images)) != 1
                    or type(cast("list[object]", images)[0]) is not dict
                ):
                    raise wire.WireError("profile_refused")
                image = cast("dict[str, object]", images[0])
                labels = cast(
                    "dict[str, object]", cast("dict[str, object]", image["Config"])["Labels"]
                )
                if (
                    image.get("Id") != self._installation["image_id"]
                    or image.get("Os") != "linux"
                    or image.get("Architecture") != "arm64"
                    or labels.get("org.strata.sql.profile") != self._profile.value
                    or labels.get("org.strata.sql.bundle")
                    != self._installation["worker_bundle_sha256"]
                ):
                    raise wire.WireError("profile_refused")
                await self._same_engine(deadline)
                self._create_issued = True
                raw_id = await self._command(
                    "create",
                    "--interactive",
                    "--name",
                    self._name,
                    "--label",
                    "org.strata.sql.owner=" + self._owner,
                    "--label",
                    "org.strata.sql.operation=" + operation_id,
                    *self._flags,
                    *(
                        ("--mount", "type=bind,source=" + str(output_dir) + ",target=/output")
                        if self._profile is _Profile.DATASET
                        else ()
                    ),
                    "--mount",
                    "type=bind,source=" + str(input_dir) + ",target=/input,readonly",
                    cast("str", self._installation["image_id"]),
                    "--operation",
                    operation_id,
                    deadline=deadline,
                )
                identifier = raw_id.decode("ascii").strip()
                if _ID.fullmatch(identifier) is None:
                    raise wire.WireError("unavailable")
                self._cid = identifier
                await self._same_engine(deadline)
                if not self._admitted(await self._inspect(self._cid, deadline), input_dir):
                    raise wire.WireError("profile_refused")
                self._start_issued = True
                process = await self._spawn(
                    ("start", "--attach", "--interactive", self._cid), pipe=True
                )
                self._attach = process
                if process.stdout is None or process.stdin is None:
                    raise wire.WireError("unavailable")
                ready = wire.parse(await self._frame(process.stdout, 8192), limit=8192)
                bundle = cast("str", self._installation["worker_bundle_sha256"])
                if self._profile is _Profile.DATASET:
                    from strata_forge.datasets._dataset_worker import (
                        expected_ready as dataset_ready,
                    )

                    expected = dataset_ready(operation_id, bundle)
                else:
                    expected = expected_ready(operation_id, bundle)
                if wire.canonical(ready, limit=8192) != wire.canonical(expected, limit=8192):
                    raise wire.WireError("profile_refused")
                process.stdin.write(
                    wire.frame(wire.parse(request, limit=wire.MAX_REQUEST), limit=wire.MAX_REQUEST)
                )
                await process.stdin.drain()
                process.stdin.close()
                await process.stdin.wait_closed()
                payload = await self._frame(process.stdout, wire.MAX_RESULT)
                if await process.stdout.read(1) != b"":
                    raise wire.WireError("worker_failed")
                await process.wait()
                await self._same_engine(deadline)
                waited = await self._command("wait", self._cid, deadline=deadline)
                if re.fullmatch(rb"[0-9]{1,3}\n", waited) is None:
                    raise wire.WireError("worker_failed")
                exit_code = int(waited)
                self._waited = True
                state = await self._inspect(self._cid, deadline)
                details = cast("dict[str, object]", state["State"])
                if (
                    not self._owned(state)
                    or details.get("Running") is not False
                    or type(details.get("Pid")) is not int
                    or details.get("Pid") != 0
                    or type(details.get("ExitCode")) is not int
                    or details.get("ExitCode") != exit_code
                    or details.get("OOMKilled") is not False
                ):
                    raise wire.WireError(
                        "resource_limit" if details.get("OOMKilled") is True else "worker_failed"
                    )
                outcome = ContainerOutcome(
                    payload,
                    self._engine,
                    cast("str", self._installation["image_id"]),
                    cast("str", self._installation["worker_bundle_sha256"]),
                    exit_code,
                    False,
                )
        except asyncio.CancelledError:
            cancelled = True
        except TimeoutError:
            code = "deadline"
        except wire.WireError as refusal:
            code = refusal.code
        except Exception:
            code = "worker_failed"
        caller = asyncio.current_task()
        cancellations = caller.cancelling() if caller is not None else 0
        clean = await drain(asyncio.create_task(self._cleanup()))
        cancelled = cancelled or (caller is not None and caller.cancelling() > cancellations)
        if cancelled:
            raise asyncio.CancelledError
        if not clean:
            raise wire.WireError("cleanup_unresolved")
        if outcome is None:
            raise wire.WireError(code)
        if time.monotonic() >= deadline:
            raise wire.WireError("deadline")
        if self._profile is _Profile.DATASET:
            return replace(
                outcome,
                settlement=ContainerSettlement(
                    operation_id,
                    self._waited,
                    self._stopped_pid_zero,
                    self._removed,
                    self._attach_reaped,
                ),
            )
        return outcome

    async def aclose(self) -> None:
        self._closed = True
        if not self._clean and not await drain(asyncio.create_task(self._cleanup())):
            raise wire.WireError("cleanup_unresolved")
        if self._directory.exists():
            self._directory.rmdir()
