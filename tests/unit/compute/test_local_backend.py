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
