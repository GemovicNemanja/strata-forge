"""Controlled lifecycle failures; these are not actual daemon/native-engine evidence."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, cast

import pytest

from strata_forge.datasets import _sql_wire as wire
from strata_forge.datasets._sql_container import SQLContainerOwner, drain

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class Owner(SQLContainerOwner):
    calls: list[tuple[str, ...]]
    engine = "engine-A"
    missing = False
    stopped = False
    bad_wait = False
    wrong_owner = False

    def __init__(self) -> None:
        super().__init__(
            {
                "version": 1,
                "profile": wire.PROFILE,
                "docker_executable": "/fictional/docker",
                "engine_socket": "/fictional/socket",
                "image_id": "sha256:" + "a" * 64,
                "worker_bundle_sha256": "b" * 64,
            }
        )
        vars(self).update(
            _engine="engine-A",
            _create_issued=True,
            _start_issued=True,
            _cid="c" * 64,
            _name="fictional",
            _operation="d" * 32,
            _clean=False,
        )
        self.calls = []

    async def _command(self, *args: str, deadline: float) -> bytes:
        assert deadline > 0
        self.calls.append(args)
        if args[0] == "info":
            return self.engine.encode() + b"\n"
        if args[:2] == ("container", "ls"):
            return b""
        if args[:2] == ("container", "inspect"):
            if self.missing:
                raise wire.WireError("unavailable")
            return json.dumps(
                [
                    {
                        "Image": "sha256:" + "a" * 64,
                        "Config": {
                            "Labels": {
                                "org.strata.sql.owner": "foreign"
                                if self.wrong_owner
                                else vars(self)["_owner"],
                                "org.strata.sql.operation": "d" * 32,
                            }
                        },
                        "State": {"Running": not self.stopped, "Pid": 0 if self.stopped else 12},
                    }
                ]
            ).encode()
        if args[0] == "kill":
            self.stopped = True
            return b"c\n"
        if args[0] == "wait":
            return b"garbage\n" if self.bad_wait else b"137\n"
        if args[0] == "rm":
            return b"c\n"
        raise AssertionError("unexpected fixed command")

    async def cleanup(self) -> bool:
        fn = cast("object", vars(SQLContainerOwner)["_cleanup"])
        return await cast("Callable[[SQLContainerOwner], Awaitable[bool]]", fn)(self)


async def test_confirmed_wait_pid_zero_removal_and_same_engine_allow_release() -> None:
    owner = Owner()
    try:
        assert await owner.cleanup() is True
        assert owner.available
        names = [call[0] for call in owner.calls]
        assert names.index("kill") < names.index("wait") < names.index("rm")
        assert owner.calls[-1] == ("info", "--format", "{{.ID}}")
    finally:
        await owner.aclose()


@pytest.mark.parametrize("kind", ["changed_engine", "missing_started", "bad_wait", "foreign_owner"])
async def test_uncertain_execution_never_releases_or_removes(kind: str) -> None:
    owner = Owner()
    owner.engine = "engine-B" if kind == "changed_engine" else "engine-A"
    owner.missing = kind == "missing_started"
    owner.bad_wait = kind == "bad_wait"
    owner.wrong_owner = kind == "foreign_owner"
    try:
        assert await owner.cleanup() is False
        assert not owner.available
        assert not any(call[0] == "rm" for call in owner.calls)
        with pytest.raises(wire.WireError) as caught:
            await owner.aclose()
        assert caught.value.code == "cleanup_unresolved"
    finally:
        # No actual subprocess exists in this controlled fixture.
        owner.engine, owner.missing, owner.bad_wait, owner.wrong_owner = (
            "engine-A",
            False,
            False,
            False,
        )
        await owner.aclose()


async def test_lost_create_empty_lookup_does_not_prove_daemon_request_settled() -> None:
    owner = Owner()
    vars(owner).update(_cid="", _start_issued=False)
    try:
        assert await owner.cleanup() is False
        assert not any(call[0] in ("start", "kill", "wait", "rm") for call in owner.calls)
    finally:
        vars(owner).update(_create_issued=False)
        await owner.aclose()


async def test_same_engine_absence_after_lost_start_is_not_reap() -> None:
    owner = Owner()
    vars(owner).update(_cid="", _start_issued=True)
    try:
        assert await owner.cleanup() is False
        assert not owner.available
    finally:
        vars(owner).update(_create_issued=False, _start_issued=False)
        await owner.aclose()


async def test_repeated_cancellation_does_not_detach_owned_drain() -> None:
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def cleanup() -> None:
        entered.set()
        await release.wait()
        finished.set()

    owned = asyncio.create_task(cleanup())
    waiting = asyncio.create_task(drain(owned))
    await entered.wait()
    waiting.cancel()
    await asyncio.sleep(0)
    waiting.cancel()
    await asyncio.sleep(0)
    assert not waiting.done()
    assert not owned.cancelled()
    release.set()
    await waiting
    assert finished.is_set()


async def test_bounded_helper_output_is_drained_and_actual_process_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    owner = Owner()
    vars(owner)["_command_prefix"] = (
        sys.executable,
        "-I",
        "-B",
        "-c",
        "import os; os.write(1,b'x'*1048576)",
    )
    # Call the real implementation, not this fixture's controlled command map.
    real = cast("Callable[..., Awaitable[bytes]]", vars(SQLContainerOwner)["_command"])
    observed: list[asyncio.subprocess.Process] = []
    original = cast(
        "Callable[..., Awaitable[asyncio.subprocess.Process]]", vars(SQLContainerOwner)["_spawn"]
    )

    async def spawn(
        self: SQLContainerOwner, args: tuple[str, ...], *, pipe: bool = False
    ) -> asyncio.subprocess.Process:
        process = await original(self, args, pipe=pipe)
        observed.append(process)
        return process

    monkeypatch.setattr(SQLContainerOwner, "_spawn", spawn)
    import time

    try:
        with pytest.raises(wire.WireError) as caught:
            await real(owner, deadline=time.monotonic() + 2)
        assert caught.value.code == "unavailable"
        assert observed[0].returncode is not None
        assert caught.value.__context__ is None
        assert not any(p.returncode is None for p in observed)
    finally:
        await owner.aclose()
