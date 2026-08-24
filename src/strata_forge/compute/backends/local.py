"""Local :class:`Backend` — runs tasks in a child process on this machine.

Primarily for tests and ad-hoc local runs; production workloads
should use the SSH or SkyPilot backends. The local backend has no
optional deps and ignores ``Task.resources`` (everything runs on
whatever hardware the parent process has).

Each ``submit`` spawns a child via ``asyncio.create_subprocess_exec``
with ``bash -lc <run>``; both pipes are drained continuously into an
in-memory buffer keyed by job id so ``logs`` works without touching
the filesystem.

The drain is a stream, not a read-to-EOF: a process that comes up
wrong and then hangs (a model server that never binds its port) has
already written the output explaining why, and a reader that only
yields once the pipes close hands back nothing for exactly the
failure worth diagnosing. ``logs`` therefore returns partial output
while a job is still running.

Pass ``log_dir=`` to also tee both streams to
``serve.stdout.log`` / ``serve.stderr.log`` as the bytes arrive.
Those files outlive ``cleanup`` and the backend object, which is the
point: when the process that owned the buffers is gone, the file is
the only remaining evidence. It is opt-in so that a library user who
never asks for it never finds log files appearing beside their code.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from strata_forge.compute.backends.base import (
    MAX_CONSOLE_CHUNK_BYTES,
    MAX_READ_FILE_BYTES,
    safe_workdir_relpath,
)
from strata_forge.compute.job import ConsoleChunk, Job, JobStatus

if TYPE_CHECKING:
    from strata_forge.compute.task import Task

__all__ = ["LocalBackend"]

# Filenames used under ``log_dir``. Deliberately NOT ``stdout.log`` / ``stderr.log``: a job
# workdir prepared by another backend (the SSH launcher) already owns those names.
SERVE_STDOUT_LOG = "serve.stdout.log"
SERVE_STDERR_LOG = "serve.stderr.log"

_READ_CHUNK_BYTES = 8192
# Kept in memory per stream. The file (when one is configured) keeps everything; a job that
# runs for hours must not grow the parent's heap without bound just because it is chatty.
_MAX_BUFFER_BYTES = 1 << 20
# How often the child's exit is checked, and how long the drain may keep reading afterwards.
# A process the child backgrounded holds the write end of the pipe open, so EOF is not
# guaranteed and neither the drain nor `Process.wait()` can be what ends the job.
_EXIT_POLL_S = 0.05
_DRAIN_GRACE_S = 5.0


async def _drain(
    reader: asyncio.StreamReader,
    buffer: bytearray,
    path: Path | None,
) -> None:
    """Copy one pipe into ``buffer`` (capped) and, when given, append it to ``path``.

    Writes are unbuffered so a tailing reader sees output as it happens — the file exists to
    be read while the process is still misbehaving, not after it exits.
    """
    fh = path.open("ab", buffering=0) if path is not None else None
    try:
        while chunk := await reader.read(_READ_CHUNK_BYTES):
            buffer.extend(chunk)
            if len(buffer) > _MAX_BUFFER_BYTES:
                del buffer[: len(buffer) - _MAX_BUFFER_BYTES]
            if fh is not None:
                fh.write(chunk)
    finally:
        if fh is not None:
            fh.close()


# How long the group gets to honour SIGTERM before SIGKILL, and how often liveness is re-asked.
_TERM_GRACE_S = 5.0
_GROUP_POLL_S = 0.05


def _group_is_alive(pgid: int) -> bool:
    """Is any process still in ``pgid``?

    Signal 0 performs the permission and existence checks without delivering anything, which is
    the actual question — and unlike waiting on the process, it cannot block. A group we are not
    allowed to signal counts as alive: it exists, and reporting it dead would be a lie that ends
    the retry loop early.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _reap_group(process: asyncio.subprocess.Process) -> None:
    """Stop everything in ``process``'s group: SIGTERM, a grace period, then SIGKILL.

    Deliberately NOT ``await process.wait()``. asyncio resolves that only once every pipe has hit
    EOF, and any descendant inherits those pipes — so waiting on a process whose child is holding
    stdout open waits for the child, which is exactly the thing being killed. That turned cancel
    into a hang, and the serving teardown's ``contextlib.suppress`` could not break it because a
    hang raises nothing. Liveness is asked of the group instead.

    The group is signalled even when the direct child has already exited: it may have left the
    real work running, which is the leak this exists to close.
    """
    pgid = process.pid  # start_new_session makes the child its own group leader
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break  # nothing left in the group
        except PermissionError:
            break  # not ours to signal; escalating would not change that
        if sig is signal.SIGKILL:
            break
        deadline = loop.time() + _TERM_GRACE_S
        # Polled rather than awaited: the OS offers no "this process group is now empty" event.
        # Only the direct child is waitable, and the descendants that actually hold the GPU are
        # not ours to wait on — signal 0 is the only question that can be asked about a group.
        while loop.time() < deadline and _group_is_alive(pgid):  # noqa: ASYNC110
            await asyncio.sleep(_GROUP_POLL_S)
        if not _group_is_alive(pgid):
            break

    # Reap the direct child so it does not linger as a zombie holding its pid. It has been
    # signalled, so this resolves promptly; the timeout covers the pipe-EOF case above.
    if process.returncode is None:
        with contextlib.suppress(TimeoutError, ProcessLookupError):
            await asyncio.wait_for(process.wait(), timeout=_TERM_GRACE_S)


class _JobState:
    """Bookkeeping for one in-flight or completed local job."""

    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None
        self.stdout_buffer = bytearray()
        self.stderr_buffer = bytearray()
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.exit_code: int | None = None
        self.cancelled = False
        # The directory the subprocess ran in — read_file resolves paths under it.
        self.workdir: str | None = None
        # Where this job's streams were teed, when a log_dir was configured.
        self.log_paths: tuple[Path, Path] | None = None


class LocalBackend:
    """In-process subprocess backend.

    Args:
        name: Optional override for the backend's reported name.
            Default ``"local"``.
        env_inherit: When ``True`` (default), child processes
            inherit the parent's ``os.environ``. When ``False``,
            the child sees only ``Task.env``.
        log_dir: When set, every job also tees its stdout/stderr
            to ``serve.stdout.log`` / ``serve.stderr.log`` under
            this directory, written as the bytes arrive and left
            in place by ``cleanup``. ``None`` (default) keeps the
            backend filesystem-free. The filenames are fixed so an
            orchestrator can find them without knowing the job id,
            so give concurrent jobs their own directories.
    """

    def __init__(
        self,
        *,
        name: str = "local",
        env_inherit: bool = True,
        log_dir: str | Path | None = None,
    ) -> None:
        if not name:
            err = "LocalBackend: name must be non-empty"
            raise ValueError(err)
        self._name = name
        self._env_inherit = env_inherit
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._jobs: dict[str, _JobState] = {}

    @property
    def name(self) -> str:
        return self._name

    async def submit(self, task: Task) -> Job:
        if task.num_nodes != 1:
            err = f"LocalBackend can only run single-node tasks; got num_nodes={task.num_nodes}"
            raise ValueError(err)

        job_id = uuid.uuid4().hex
        state = _JobState()
        state.workdir = task.workdir
        state.log_paths = self._prepare_log_paths(state)
        self._jobs[job_id] = state

        env = self._build_env(task)
        script = self._build_script(task)

        async def _runner() -> None:
            try:
                state.started_at = datetime.now(UTC)
                process = await asyncio.create_subprocess_exec(
                    "bash",
                    "-lc",
                    script,
                    cwd=task.workdir or None,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    # The child leads its own process group, so cancel can signal the whole tree
                    # with one killpg. The tree is the point: a served model is not one process.
                    # vLLM runs its engine in SEPARATE worker processes, and those are what hold
                    # the GPU — terminating only the shell child leaves them resident, so the next
                    # run finds the device occupied by a job the user already cancelled.
                    #
                    # Without a new session the group would be the ORCHESTRATOR's, and killpg
                    # would signal the caller that asked for the cancel.
                    start_new_session=True,
                )
                state.process = process
                stdout_log, stderr_log = state.log_paths or (None, None)
                drains = [
                    asyncio.create_task(_drain(reader, buffer, path))
                    for reader, buffer, path in (
                        (process.stdout, state.stdout_buffer, stdout_log),
                        (process.stderr, state.stderr_buffer, stderr_log),
                    )
                    if reader is not None
                ]
                # NOT `await process.wait()`: asyncio only resolves that once every pipe has
                # reached EOF, and a process the child backgrounded inherits those descriptors
                # and can hold them open long after the child is reaped. `returncode` is set
                # when the child itself exits, so the job's lifecycle hangs off that instead —
                # otherwise a job whose server orphaned itself never reaches a terminal state.
                while process.returncode is None:  # noqa: ASYNC110 — asyncio exposes no event
                    await asyncio.sleep(_EXIT_POLL_S)  # for "child reaped", only this flag
                state.exit_code = process.returncode
                if drains:
                    _, pending = await asyncio.wait(drains, timeout=_DRAIN_GRACE_S)
                    for drain in pending:
                        drain.cancel()
                    if pending:
                        await asyncio.wait(pending)
            except Exception as exc:
                state.exit_code = -1
                state.stderr_buffer.extend(f"local backend exception: {exc!r}".encode())
            finally:
                state.finished_at = datetime.now(UTC)

        # Fire-and-forget; status / logs / cancel observe state directly.
        asyncio.create_task(_runner())  # noqa: RUF006 — lifetime managed via _jobs

        return Job(
            id=job_id,
            backend=self._name,
            task_name=task.name,
            metadata={"script_preview": script[:120]},
        )

    def _prepare_log_paths(self, state: _JobState) -> tuple[Path, Path] | None:
        """Resolve (and create) this job's log files, or ``None`` when teeing is off.

        An unusable log directory degrades to in-memory-only logging with a note on the job's
        stderr — losing the artifact is bad, but failing the job over a log path is worse.
        """
        if self._log_dir is None:
            return None
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            state.stderr_buffer.extend(f"local backend: log_dir unusable: {exc!r}\n".encode())
            return None
        return (self._log_dir / SERVE_STDOUT_LOG, self._log_dir / SERVE_STDERR_LOG)

    def _build_env(self, task: Task) -> dict[str, str]:
        import os

        env: dict[str, str] = {}
        if self._env_inherit:
            env.update(os.environ)
        env.update(task.env)
        return env

    @staticmethod
    def _build_script(task: Task) -> str:
        parts: list[str] = []
        if task.setup:
            parts.append(task.setup)
        parts.append(task.run)
        return " && ".join(parts) if len(parts) > 1 else parts[0]

    def _require_job(self, job: Job) -> _JobState:
        if job.backend != self._name:
            err = (
                f"LocalBackend.{self._call_label()}: job belongs to backend "
                f"{job.backend!r}, not {self._name!r}"
            )
            raise ValueError(err)
        state = self._jobs.get(job.id)
        if state is None:
            err = f"LocalBackend: unknown job id {job.id!r}"
            raise ValueError(err)
        return state

    @staticmethod
    def _call_label() -> str:
        # Best-effort caller label for error messages — we don't try
        # to be precise via the stack.
        return "request"

    async def status(self, job: Job) -> JobStatus:
        state = self._require_job(job)
        if state.cancelled:
            return JobStatus(
                state="cancelled",
                exit_code=state.exit_code,
                started_at=state.started_at,
                finished_at=state.finished_at,
                message="cancelled by user",
            )
        if state.finished_at is None:
            if state.started_at is None:
                return JobStatus(state="pending")
            return JobStatus(state="running", started_at=state.started_at)
        # Terminated.
        if state.exit_code == 0:
            return JobStatus(
                state="succeeded",
                exit_code=0,
                started_at=state.started_at,
                finished_at=state.finished_at,
            )
        return JobStatus(
            state="failed",
            exit_code=state.exit_code,
            started_at=state.started_at,
            finished_at=state.finished_at,
            message=(
                f"non-zero exit ({state.exit_code})"
                if state.exit_code is not None
                else "subprocess crashed before exit"
            ),
        )

    async def logs(self, job: Job, *, tail: int | None = None) -> str:
        """Return what the job has printed so far — including while it is still running."""
        state = self._require_job(job)
        combined = state.stdout_buffer.decode(
            "utf-8", errors="replace"
        ) + state.stderr_buffer.decode("utf-8", errors="replace")
        if tail is None:
            return combined
        if tail <= 0:
            err = f"tail must be >= 1 when set; got {tail}"
            raise ValueError(err)
        lines = combined.splitlines()
        return "\n".join(lines[-tail:])

    async def console(
        self,
        job: Job,
        *,
        stdout_offset: int = 0,
        stderr_offset: int = 0,
        max_bytes: int = MAX_CONSOLE_CHUNK_BYTES,
    ) -> ConsoleChunk:
        state = self._require_job(job)
        if max_bytes <= 0:
            err = f"max_bytes must be >= 1; got {max_bytes}"
            raise ValueError(err)
        out, out_dropped = _slice_stream(state.stdout_buffer, stdout_offset, max_bytes)
        err_text, err_dropped = _slice_stream(state.stderr_buffer, stderr_offset, max_bytes)
        return ConsoleChunk(
            stdout=out,
            stderr=err_text,
            stdout_offset=len(state.stdout_buffer),
            stderr_offset=len(state.stderr_buffer),
            dropped_bytes=out_dropped + err_dropped,
        )

    async def read_file(self, job: Job, path: str, *, tail: int | None = None) -> str:
        state = self._require_job(job)
        rel = safe_workdir_relpath(path)
        if tail is not None:
            tail = int(tail)
            if tail <= 0:
                err = f"tail must be >= 1 when set; got {tail}"
                raise ValueError(err)
        root = (Path(state.workdir) if state.workdir else Path.cwd()).resolve()
        target = root / rel

        def _read() -> str:
            # Defense-in-depth beyond the lexical guard: reject a workdir symlink that
            # resolves OUTSIDE the workdir. Bounded read so a huge file can't OOM us.
            if not target.resolve().is_relative_to(root):
                err = f"read_file: path resolves outside the job workdir; got {path!r}"
                raise ValueError(err)
            try:
                with target.open(encoding="utf-8", errors="replace") as fh:
                    return fh.read(MAX_READ_FILE_BYTES)
            except FileNotFoundError, NotADirectoryError, IsADirectoryError:
                return ""

        content = await asyncio.to_thread(_read)
        if tail is None:
            return content
        return "\n".join(content.splitlines()[-tail:])

    async def cancel(self, job: Job) -> None:
        state = self._require_job(job)
        process = state.process
        if process is None:
            # Process hasn't been spawned yet; mark cancelled so the
            # next status() reports it.
            state.cancelled = True
            return
        # The SWEEP is not gated on `finished_at`: the job's own process exiting does not mean its
        # work stopped. A launcher that starts a server and returns reports success while the
        # server keeps running, and the teardown path calls cancel precisely to collect that.
        # Sweeping an empty group is a no-op, so the old early return bought nothing but a live
        # server left behind.
        finished = state.finished_at is not None
        await _reap_group(process)
        # The VERDICT is gated: a job that already reached a terminal state keeps it. Cancelling
        # a job that succeeded is a tidy-up, not a retroactive re-run of how it ended.
        if not finished:
            state.cancelled = True

    async def cleanup(self, job: Job) -> None:
        """Drop the job's in-memory state and make sure its process is gone.

        Any ``log_dir`` files are left on disk on purpose: they exist to be read AFTER the
        buffers they mirror have been discarded.
        """
        state = self._jobs.pop(job.id, None)
        if state is None:
            return  # already cleaned up — idempotent
        # Make sure any lingering process is gone before we drop the state. Once this returns,
        # the state is dropped and nothing can name the group again, so it sweeps unconditionally
        # for the same reason cancel does — a job whose own process exited may still have left
        # the work running.
        process = state.process
        if process is not None:
            await _reap_group(process)


def _slice_stream(buffer: bytes | bytearray, offset: int, cap: int) -> tuple[str, int]:
    """Return (text after ``offset``, bytes dropped), keeping the NEWEST ``cap`` bytes."""
    pending = buffer[max(0, int(offset)) :]
    if len(pending) <= cap:
        return pending.decode("utf-8", errors="replace"), 0
    # Same choice the SSH backend makes: a watcher wants where the job is now, and the shortfall
    # is reported rather than hidden.
    return pending[-cap:].decode("utf-8", errors="replace"), len(pending) - cap
