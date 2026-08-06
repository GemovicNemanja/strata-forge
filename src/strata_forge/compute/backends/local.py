"""Local :class:`Backend` — runs tasks in a child process on this machine.

Primarily for tests and ad-hoc local runs; production workloads
should use the SSH or SkyPilot backends. The local backend has no
optional deps and ignores ``Task.resources`` (everything runs on
whatever hardware the parent process has).

Each ``submit`` spawns a child via ``asyncio.create_subprocess_exec``
with ``bash -lc <run>``; the captured stdout/stderr is appended to
an in-memory buffer keyed by job id so ``logs`` and ``cleanup`` work
without touching the filesystem.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from strata_forge.compute.backends.base import MAX_READ_FILE_BYTES, safe_workdir_relpath
from strata_forge.compute.job import Job, JobStatus

if TYPE_CHECKING:
    from strata_forge.compute.task import Task

__all__ = ["LocalBackend"]


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


class LocalBackend:
    """In-process subprocess backend.

    Args:
        name: Optional override for the backend's reported name.
            Default ``"local"``.
        env_inherit: When ``True`` (default), child processes
            inherit the parent's ``os.environ``. When ``False``,
            the child sees only ``Task.env``.
    """

    def __init__(self, *, name: str = "local", env_inherit: bool = True) -> None:
        if not name:
            err = "LocalBackend: name must be non-empty"
            raise ValueError(err)
        self._name = name
        self._env_inherit = env_inherit
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
                )
                state.process = process
                stdout_bytes, stderr_bytes = await process.communicate()
                state.stdout_buffer.extend(stdout_bytes or b"")
                state.stderr_buffer.extend(stderr_bytes or b"")
                state.exit_code = process.returncode
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

    async def read_file(self, job: Job, path: str, *, tail: int | None = None) -> str:
        from pathlib import Path

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
        if state.finished_at is not None:
            return
        process = state.process
        if process is None:
            # Process hasn't been spawned yet; mark cancelled so the
            # next status() reports it.
            state.cancelled = True
            return
        process.terminate()
        # Wait briefly for graceful shutdown; SIGKILL if it doesn't comply.
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except TimeoutError:
            process.kill()
            await process.wait()
        state.cancelled = True

    async def cleanup(self, job: Job) -> None:
        state = self._jobs.pop(job.id, None)
        if state is None:
            return  # already cleaned up — idempotent
        # Make sure any lingering process is gone before we drop the state.
        process = state.process
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except TimeoutError:
                process.kill()
                await process.wait()
