"""Starting the plugin's process, relaying what it reports, and making sure it is gone.

:func:`run_plugin_process` starts the sandboxed host (:mod:`strata_forge.plugins.sandbox`) in a
session of its own and:

- reads its progress pipe (standard output) in bounded chunks, and relays only what validates as
  a progress event of the plugin's: a ``step`` (counts, an epoch, metrics keyed
  ``plugin.<name>``) or a ``phase`` caption, labelled as the plugin's. Anything else (another
  kind of event, a forge-shaped metric, a line too long, text that is not an event) is dropped,
  and events are relayed at a bounded rate;
- keeps only the last :data:`STDERR_TAIL_BYTES` of its standard error;
- enforces the operator's wall-clock limit;
- after the process exits, whether it finished, failed, ran out of time or was cancelled, signals
  its whole process group, waits for it, and confirms that no process of the run is left (by its
  process group, and by the job file's path on any process's command line). Only then does the
  caller go on to check and push the output.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError as PydanticValidationError

from strata_forge.plugins.host import MAX_PHASE_CHARS, PLUGIN_METRIC_PREFIX
from strata_forge.plugins.sandbox import Sandbox, kill_command
from strata_forge.training.progress import ProgressEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "MAX_LINE_BYTES",
    "STDERR_TAIL_BYTES",
    "ChildResult",
    "PluginStillRunningError",
    "relayable",
    "run_plugin_process",
]

#: The longest progress line relayed; a longer one is dropped whole.
MAX_LINE_BYTES = 16 * 1024
#: How much of the plugin's standard error is kept.
STDERR_TAIL_BYTES = 8 * 1024
#: The shortest gap between two relayed step events.
MIN_STEP_INTERVAL_S = 0.5
#: Seconds between SIGTERM and SIGKILL for the process group.
TERMINATE_GRACE_SECONDS = 10.0
#: How long the group is given to disappear after SIGKILL before the run fails.
GONE_TIMEOUT_SECONDS = 10.0
_READ_CHUNK = 64 * 1024


class PluginStillRunningError(Exception):
    """A plugin process outlived the kill: nothing may be pushed."""


@dataclass(frozen=True, slots=True)
class ChildResult:
    code: int
    stderr_tail: str
    timed_out: bool


def relayable(line: bytes) -> ProgressEvent | None:
    """The event to relay for one progress line from the plugin, or ``None`` to drop it."""
    try:
        event = ProgressEvent.model_validate_json(line)
    except PydanticValidationError, ValueError:
        return None
    if event.kind == "step":
        metrics = {
            k: v
            for k, v in event.metrics.items()
            if k.startswith(PLUGIN_METRIC_PREFIX) and math.isfinite(v)
        }
        epoch = event.epoch if event.epoch is not None and math.isfinite(event.epoch) else None
        step = event.step if event.step is not None and event.step >= 0 else None
        total = (
            event.total_steps if event.total_steps is not None and event.total_steps >= 0 else None
        )
        return ProgressEvent(
            kind="step", stage="run", step=step, total_steps=total, epoch=epoch, metrics=metrics
        )
    if event.kind == "phase" and event.message:
        return ProgressEvent(
            kind="phase", stage="run", message=f"plugin: {event.message[:MAX_PHASE_CHARS]}"
        )
    return None


async def _relay(stream: asyncio.StreamReader, sink: Callable[[ProgressEvent], None]) -> None:
    pending = b""
    dropping = False
    last_step = 0.0
    held: ProgressEvent | None = None
    while chunk := await stream.read(_READ_CHUNK):
        pending += chunk
        *lines, pending = pending.split(b"\n")
        for line in lines:
            if dropping:  # the tail of an over-long line
                dropping = False
                continue
            event = relayable(line) if len(line) <= MAX_LINE_BYTES else None
            if event is None:
                continue
            now = time.monotonic()
            if event.kind == "step" and now - last_step < MIN_STEP_INTERVAL_S:
                held = event
                continue
            if event.kind == "step":
                last_step, held = now, None
            sink(event)
        if len(pending) > MAX_LINE_BYTES:
            pending, dropping = b"", True
    if held is not None:
        sink(held)


async def _tail(stream: asyncio.StreamReader, keep: bytearray) -> None:
    while chunk := await stream.read(_READ_CHUNK):
        keep.extend(chunk)
        if len(keep) > STDERR_TAIL_BYTES:
            del keep[: len(keep) - STDERR_TAIL_BYTES]


def _quiet(args: Sequence[str]) -> int:
    try:
        done = subprocess.run(  # noqa: S603 -- a fixed sudo kill command, no shell
            list(args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return -1
    return done.returncode


def _signal_group(sandbox: Sandbox, pgid: int, sig: signal.Signals) -> None:
    if sandbox.isolation == "uid":
        # Root's and nobody's processes: the runner's uid cannot signal them itself.
        _quiet(kill_command(sandbox, pgid, "KILL" if sig == signal.SIGKILL else "TERM"))
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def _group_alive(sandbox: Sandbox, pgid: int) -> bool:
    if sandbox.isolation == "uid":
        args = kill_command(sandbox, pgid, "TERM")
        at = args.index("-s")
        return _quiet([*args[:at], "-0", *args[at + 2 :]]) == 0
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _marked(marker: str) -> list[int]:
    """Processes whose command line names ``marker`` (the run's job file), where ``/proc`` is."""
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    found: list[int] = []
    needle = marker.encode()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if needle in (entry / "cmdline").read_bytes():
                found.append(int(entry.name))
        except OSError:
            continue
    return found


async def _ensure_gone(sandbox: Sandbox, pgid: int, marker: str) -> None:
    _signal_group(sandbox, pgid, signal.SIGTERM)
    deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
    while (_group_alive(sandbox, pgid) or _marked(marker)) and time.monotonic() < deadline:  # noqa: ASYNC110 -- polling processes, not an event
        await asyncio.sleep(0.2)
    _signal_group(sandbox, pgid, signal.SIGKILL)
    for pid in _marked(marker):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + GONE_TIMEOUT_SECONDS
    while _group_alive(sandbox, pgid) or _marked(marker):
        if time.monotonic() >= deadline:
            msg = "a plugin process survived being killed; nothing is pushed"
            raise PluginStillRunningError(msg)
        await asyncio.sleep(0.2)


async def run_plugin_process(
    command: Sequence[str],
    *,
    sandbox: Sandbox,
    env: Mapping[str, str],
    cwd: Path,
    marker: str,
    timeout_s: float,
    sink: Callable[[ProgressEvent], None],
) -> ChildResult:
    """Run the plugin's process to its end and make sure nothing of it is left."""
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        env=dict(env),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid
    stderr = bytearray()
    if proc.stdout is None or proc.stderr is None:  # PIPE above; a typed guard
        msg = "the plugin process has no pipes"
        raise RuntimeError(msg)
    readers = [
        asyncio.create_task(_relay(proc.stdout, sink)),
        asyncio.create_task(_tail(proc.stderr, stderr)),
    ]
    timed_out = False
    try:
        try:
            async with asyncio.timeout(timeout_s):
                # The exit itself, not ``wait()``: that also waits for the pipes to close, which a
                # process the plugin left behind can hold open until the time limit.
                while proc.returncode is None:  # noqa: ASYNC110 -- the exit has no event of its own
                    await asyncio.sleep(0.2)
        except TimeoutError:
            timed_out = True
    finally:
        # Every way out (done, failed, out of time, cancelled) ends with the group gone.
        await asyncio.shield(_ensure_gone(sandbox, pgid, marker))
        with contextlib.suppress(ProcessLookupError):
            if proc.returncode is None:
                proc.kill()
        await proc.wait()
        # Every writer of the pipes is gone, so the readers reach end of file; the bound is
        # for a pipe some unrelated process still holds open.
        _, unfinished = await asyncio.wait(readers, timeout=5.0)
        for task in unfinished:
            task.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
    return ChildResult(
        code=proc.returncode if proc.returncode is not None else -1,
        stderr_tail=bytes(stderr).decode("utf-8", "replace"),
        timed_out=timed_out,
    )
