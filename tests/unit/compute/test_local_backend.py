"""Unit tests for `forge.compute.backends.local.LocalBackend`."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import pytest

from forge.compute import Backend, LocalBackend, Task

if TYPE_CHECKING:
    from pathlib import Path

    from forge.compute.job import Job


async def _wait_until_terminal(
    backend: LocalBackend,
    job: Job,
    *,
    timeout: float = 10.0,  # noqa: ASYNC109 — explicit wall-clock budget for test polling
) -> None:
    """Poll until the job reaches a terminal state."""

    async def _poll() -> None:
        while True:
            status = await backend.status(job)
            if status.is_terminal:
                return
            await asyncio.sleep(0.05)

    await asyncio.wait_for(_poll(), timeout=timeout)


class TestProtocolCompliance:
    def test_satisfies_backend_protocol(self) -> None:
        backend = LocalBackend()
        assert isinstance(backend, Backend)


class TestConstruction:
    def test_default_name(self) -> None:
        backend = LocalBackend()
        assert backend.name == "local"

    def test_custom_name(self) -> None:
        backend = LocalBackend(name="my-local")
        assert backend.name == "my-local"

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            LocalBackend(name="")


class TestSubmitHappyPath:
    async def test_runs_and_succeeds(self) -> None:
        backend = LocalBackend()
        task = Task(name="echo", run="echo hello")
        job = await backend.submit(task)
        assert job.backend == "local"
        assert job.task_name == "echo"

        await _wait_until_terminal(backend, job)
        status = await backend.status(job)
        assert status.state == "succeeded"
        assert status.exit_code == 0
        assert status.started_at is not None
        assert status.finished_at is not None

    async def test_logs_capture_stdout(self) -> None:
        backend = LocalBackend()
        job = await backend.submit(Task(name="hi", run="echo HELLO-FROM-TASK"))
        await _wait_until_terminal(backend, job)
        logs = await backend.logs(job)
        assert "HELLO-FROM-TASK" in logs

    async def test_setup_runs_before_run(self) -> None:
        backend = LocalBackend()
        job = await backend.submit(
            Task(
                name="composed",
                setup="echo FROM-SETUP",
                run="echo FROM-RUN",
            )
        )
        await _wait_until_terminal(backend, job)
        logs = await backend.logs(job)
        assert "FROM-SETUP" in logs
        assert "FROM-RUN" in logs

    async def test_env_vars_visible_to_run(self) -> None:
        backend = LocalBackend()
        job = await backend.submit(
            Task(
                name="env",
                run="echo MY_VAR=$MY_VAR",
                env={"MY_VAR": "the-value"},
            )
        )
        await _wait_until_terminal(backend, job)
        logs = await backend.logs(job)
        assert "MY_VAR=the-value" in logs

    async def test_workdir_used(self, tmp_path: Path) -> None:
        backend = LocalBackend()
        job = await backend.submit(Task(name="cwd", run="pwd", workdir=str(tmp_path)))
        await _wait_until_terminal(backend, job)
        logs = await backend.logs(job)
        # The working directory tail should match (Linux + macOS may
        # prefix /private on macOS for /var/folders tempdirs).
        assert str(tmp_path) in logs or logs.strip().endswith(tmp_path.name)


class TestSubmitFailure:
    async def test_non_zero_exit_marks_failed(self) -> None:
        backend = LocalBackend()
        job = await backend.submit(Task(name="exit42", run="exit 42"))
        await _wait_until_terminal(backend, job)
        status = await backend.status(job)
        assert status.state == "failed"
        assert status.exit_code == 42

    async def test_command_not_found_marks_failed(self) -> None:
        backend = LocalBackend()
        job = await backend.submit(Task(name="nope", run="this-command-does-not-exist-1234567"))
        await _wait_until_terminal(backend, job)
        status = await backend.status(job)
        assert status.state == "failed"


class TestSubmitValidation:
    async def test_multi_node_rejected(self) -> None:
        backend = LocalBackend()
        with pytest.raises(ValueError, match="single-node"):
            await backend.submit(Task(name="x", run="hi", num_nodes=2))


class TestLogs:
    async def test_logs_tail(self) -> None:
        backend = LocalBackend()
        # Three lines of output; tail=2 returns just the last two.
        job = await backend.submit(Task(name="t", run="printf 'one\\ntwo\\nthree\\n'"))
        await _wait_until_terminal(backend, job)
        tail = await backend.logs(job, tail=2)
        lines = tail.splitlines()
        assert lines[-1] == "three"
        assert "one" not in tail

    async def test_logs_tail_invalid(self) -> None:
        backend = LocalBackend()
        job = await backend.submit(Task(name="t", run="echo x"))
        await _wait_until_terminal(backend, job)
        with pytest.raises(ValueError, match="tail"):
            await backend.logs(job, tail=0)


class TestCancel:
    async def test_cancel_running_job(self) -> None:
        backend = LocalBackend()
        job = await backend.submit(Task(name="sleep", run="sleep 30"))
        # Wait for the process to actually start.
        for _ in range(50):
            status = await backend.status(job)
            if status.state == "running":
                break
            await asyncio.sleep(0.05)
        await backend.cancel(job)
        await _wait_until_terminal(backend, job)
        status = await backend.status(job)
        assert status.state == "cancelled"

    async def test_cancel_terminal_job_is_noop(self) -> None:
        backend = LocalBackend()
        job = await backend.submit(Task(name="quick", run="echo done"))
        await _wait_until_terminal(backend, job)
        await backend.cancel(job)  # should not raise
        status = await backend.status(job)
        # Succeeded jobs stay succeeded after a no-op cancel.
        assert status.state == "succeeded"

    async def test_cancel_before_process_spawned(self) -> None:
        # Race window: the runner coroutine is scheduled but hasn't yet
        # called create_subprocess_exec. The cancel path should mark the
        # job cancelled without crashing.
        from forge.compute.backends.local import _JobState  # pyright: ignore[reportPrivateUsage]
        from forge.compute.job import Job

        backend = LocalBackend()
        job_id = "synth-cancel-1"
        # Inject a job state with no process — simulates the race.
        state = _JobState()
        backend._jobs[job_id] = state  # pyright: ignore[reportPrivateUsage]
        job = Job(id=job_id, backend="local", task_name="t")
        await backend.cancel(job)
        assert state.cancelled is True

    async def test_cancel_sigkill_after_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force wait_for to time out so the SIGKILL branch runs.
        backend = LocalBackend()
        job = await backend.submit(Task(name="sleep", run="sleep 30"))
        for _ in range(50):
            if (await backend.status(job)).state == "running":
                break
            await asyncio.sleep(0.05)

        import asyncio as _asyncio

        original_wait_for = _asyncio.wait_for

        async def _fake_wait_for(awaitable: object, timeout: float) -> object:  # noqa: ASYNC109
            del timeout
            # First call (from cancel) times out; the inner cancel of the
            # awaitable + kill path then runs.
            raise TimeoutError

        monkeypatch.setattr(_asyncio, "wait_for", _fake_wait_for)
        await backend.cancel(job)
        monkeypatch.setattr(_asyncio, "wait_for", original_wait_for)
        await _wait_until_terminal(backend, job)
        assert (await backend.status(job)).state == "cancelled"


class TestCleanup:
    async def test_cleanup_removes_job_state(self) -> None:
        backend = LocalBackend()
        job = await backend.submit(Task(name="t", run="echo bye"))
        await _wait_until_terminal(backend, job)
        await backend.cleanup(job)
        # After cleanup the job is no longer known to the backend.
        with pytest.raises(ValueError, match="unknown job"):
            await backend.status(job)

    async def test_cleanup_is_idempotent(self) -> None:
        backend = LocalBackend()
        job = await backend.submit(Task(name="t", run="echo x"))
        await _wait_until_terminal(backend, job)
        await backend.cleanup(job)
        await backend.cleanup(job)  # second call must not raise

    async def test_cleanup_kills_lingering_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Cleanup mid-run with a process that doesn't honor SIGTERM in
        # time — exercises the wait_for/timeout → kill fallback.
        backend = LocalBackend()
        job = await backend.submit(Task(name="sleep", run="sleep 30"))
        for _ in range(50):
            if (await backend.status(job)).state == "running":
                break
            await asyncio.sleep(0.05)

        import asyncio as _asyncio

        async def _fake_wait_for(awaitable: object, timeout: float) -> object:  # noqa: ASYNC109
            del timeout
            raise TimeoutError

        monkeypatch.setattr(_asyncio, "wait_for", _fake_wait_for)
        await backend.cleanup(job)
        # Job is gone from the state dict.
        with pytest.raises(ValueError, match="unknown"):
            await backend.status(job)

    async def test_runner_handles_spawn_exception(self) -> None:
        # An unreadable workdir makes create_subprocess_exec raise;
        # the runner must capture the exception and mark the job failed.
        backend = LocalBackend()
        job = await backend.submit(
            Task(name="bad-wd", run="echo hi", workdir="/proc/self/nonexistent/x")
        )
        await _wait_until_terminal(backend, job)
        status = await backend.status(job)
        assert status.state == "failed"
        logs = await backend.logs(job)
        assert "local backend exception" in logs

    async def test_status_unknown_job_raises(self) -> None:
        backend = LocalBackend()
        from forge.compute.job import Job

        bogus = Job(id="not-real", backend="local", task_name="t")
        with pytest.raises(ValueError, match="unknown job"):
            await backend.status(bogus)

    async def test_status_wrong_backend_rejected(self) -> None:
        backend = LocalBackend()
        from forge.compute.job import Job

        job = Job(id="x", backend="some-other-backend", task_name="t")
        with pytest.raises(ValueError, match="backend"):
            await backend.status(job)


class TestEnvInherit:
    async def test_inherits_parent_env_by_default(self) -> None:
        # Set a marker env var in the parent before submit.
        os.environ["FORGE_TEST_MARKER"] = "inherited-ok"
        try:
            backend = LocalBackend()
            job = await backend.submit(Task(name="env", run="echo MARK=$FORGE_TEST_MARKER"))
            await _wait_until_terminal(backend, job)
            logs = await backend.logs(job)
            assert "MARK=inherited-ok" in logs
        finally:
            os.environ.pop("FORGE_TEST_MARKER", None)

    async def test_env_inherit_false_isolates_env(self) -> None:
        os.environ["FORGE_TEST_MARKER"] = "should-not-leak"
        try:
            backend = LocalBackend(env_inherit=False)
            job = await backend.submit(
                Task(name="env", run="echo MARK=${FORGE_TEST_MARKER:-MISSING}")
            )
            await _wait_until_terminal(backend, job)
            logs = await backend.logs(job)
            assert "MARK=MISSING" in logs
        finally:
            os.environ.pop("FORGE_TEST_MARKER", None)


class TestReadFile:
    async def _job_in(self, backend: LocalBackend, workdir: Path) -> Job:
        job = await backend.submit(Task(name="t", run="true", workdir=str(workdir)))
        await _wait_until_terminal(backend, job)
        return job

    async def test_reads_workdir_file(self, tmp_path: Path) -> None:
        (tmp_path / "progress.jsonl").write_text('{"step": 1}\n')
        backend = LocalBackend()
        job = await self._job_in(backend, tmp_path)
        assert await backend.read_file(job, "progress.jsonl") == '{"step": 1}\n'

    async def test_tail(self, tmp_path: Path) -> None:
        (tmp_path / "m.txt").write_text("a\nb\nc\nd\n")
        backend = LocalBackend()
        job = await self._job_in(backend, tmp_path)
        assert await backend.read_file(job, "m.txt", tail=2) == "c\nd"

    async def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        backend = LocalBackend()
        job = await self._job_in(backend, tmp_path)
        assert await backend.read_file(job, "missing.jsonl") == ""

    async def test_invalid_tail(self, tmp_path: Path) -> None:
        (tmp_path / "m.txt").write_text("x\n")
        backend = LocalBackend()
        job = await self._job_in(backend, tmp_path)
        with pytest.raises(ValueError, match="tail"):
            await backend.read_file(job, "m.txt", tail=0)

    async def test_rejects_traversal(self, tmp_path: Path) -> None:
        # A secret one level above the workdir must be unreachable.
        (tmp_path / "secret.txt").write_text("nope")
        sub = tmp_path / "run"
        sub.mkdir()
        backend = LocalBackend()
        job = await self._job_in(backend, sub)
        with pytest.raises(ValueError, match="within the job workdir"):
            await backend.read_file(job, "../secret.txt")

    async def test_rejects_absolute(self, tmp_path: Path) -> None:
        backend = LocalBackend()
        job = await self._job_in(backend, tmp_path)
        with pytest.raises(ValueError, match="workdir-relative"):
            await backend.read_file(job, "/etc/passwd")
