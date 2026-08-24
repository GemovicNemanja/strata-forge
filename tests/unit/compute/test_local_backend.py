"""Unit tests for `strata_forge.compute.backends.local.LocalBackend`."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import pytest

from strata_forge.compute import Backend, LocalBackend, Task
from strata_forge.compute.backends import local as local_backend

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from strata_forge.compute.job import Job


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


async def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 10.0,  # noqa: ASYNC109 — explicit wall-clock budget for test polling
) -> None:
    """Poll ``predicate`` until it holds; raises :class:`TimeoutError` if it never does."""

    async def _poll() -> None:
        while True:
            if predicate():
                return
            await asyncio.sleep(0.05)

    await asyncio.wait_for(_poll(), timeout=timeout)


def _pid_alive(pid: int) -> bool:
    """Does ``pid`` still exist? Signal 0 checks without delivering anything."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours to signal
    return True


def _serve_log_names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob("serve.*.log"))


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


class TestConsole:
    """Incremental reads: the same contract the SSH backend implements, over in-memory buffers."""

    async def test_resumes_from_the_offset_it_returned(self) -> None:
        backend = LocalBackend()
        job = await backend.submit(Task(name="t", run="printf 'one\\ntwo\\n'"))
        await _wait_until_terminal(backend, job)

        first = await backend.console(job)
        assert first.stdout == "one\ntwo\n"
        # Nothing more was written, so a second read from that offset is empty rather than a
        # repeat of the same lines -- which is the whole point of reading incrementally.
        second = await backend.console(
            job, stdout_offset=first.stdout_offset, stderr_offset=first.stderr_offset
        )
        assert second.stdout == ""
        assert second.stdout_offset == first.stdout_offset

    async def test_separates_the_two_streams(self) -> None:
        backend = LocalBackend()
        job = await backend.submit(Task(name="t", run="echo out; echo err 1>&2"))
        await _wait_until_terminal(backend, job)
        chunk = await backend.console(job)
        assert chunk.stdout.strip() == "out"
        assert chunk.stderr.strip() == "err"

    async def test_an_overflowing_window_keeps_the_newest_and_reports_the_gap(self) -> None:
        backend = LocalBackend()
        job = await backend.submit(Task(name="t", run="printf '0123456789'"))
        await _wait_until_terminal(backend, job)
        # `max_bytes` is the TOTAL for the call, split evenly between the two streams.
        chunk = await backend.console(job, max_bytes=8)
        assert chunk.stdout == "6789"
        assert chunk.dropped_bytes == 6

    async def test_a_nonpositive_cap_is_rejected(self) -> None:
        backend = LocalBackend()
        job = await backend.submit(Task(name="t", run="echo x"))
        await _wait_until_terminal(backend, job)
        with pytest.raises(ValueError, match="max_bytes"):
            await backend.console(job, max_bytes=0)

    async def test_offsets_stay_absolute_once_the_window_starts_sliding(self) -> None:
        """The buffer is a sliding 1 MiB window, so its indices are NOT stream offsets.

        Reading `len(buffer)` as an offset pins the cursor at the cap forever: every later poll
        asks for bytes past the end of a buffer that has stopped growing, gets nothing, and
        reports `dropped_bytes=0` while the rest of the run's output disappears -- including the
        final line, which is the one a watcher is waiting for.
        """
        backend = LocalBackend()
        marker = "THE-LAST-LINE"
        # Deliberately past _MAX_BUFFER_BYTES: below the cap nothing slides and the bug hides.
        job = await backend.submit(
            Task(name="t", run=f"python3 -c \"print('x'*(1<<21)); print('{marker}')\"")
        )
        await _wait_until_terminal(backend, job)

        seen: list[str] = []
        chunk = await backend.console(job, max_bytes=8192)
        seen.append(chunk.stdout)
        for _ in range(60):
            chunk = await backend.console(
                job, stdout_offset=chunk.stdout_offset, stderr_offset=chunk.stderr_offset
            )
            seen.append(chunk.stdout)
        assert marker in "".join(seen), "the end of the output must eventually be delivered"


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
        from strata_forge.compute.backends.local import (
            _JobState,  # pyright: ignore[reportPrivateUsage]
        )
        from strata_forge.compute.job import Job

        backend = LocalBackend()
        job_id = "synth-cancel-1"
        # Inject a job state with no process — simulates the race.
        state = _JobState()
        backend._jobs[job_id] = state  # pyright: ignore[reportPrivateUsage]
        job = Job(id=job_id, backend="local", task_name="t")
        await backend.cancel(job)
        assert state.cancelled is True

    async def test_cancel_kills_the_whole_tree_not_just_the_child(self, tmp_path: Path) -> None:
        """A cancelled job must leave nothing of its own running.

        This is the leak that made "Cancelling" meaningless in production: the recorded process
        is a shell, and the work — a served model and its engine workers — lives BELOW it. A
        served model is never one process, so signalling only the direct child reparents the
        rest onto init, where it keeps holding the GPU that the next run then cannot get.

        Watching real pids is the point. The test this replaces faked an `asyncio.wait_for`
        timeout and asserted only the reported STATE, so it passed against a cancel that killed
        nothing at all — which is how the leak shipped.
        """
        backend = LocalBackend()
        pidfile = tmp_path / "grandchild.pid"
        # A grandchild that outlives its parent shell unless the GROUP is signalled: the shell
        # backgrounds it and exits immediately, so `terminate()` on the child is a no-op for it.
        job = await backend.submit(
            Task(
                name="tree",
                run=f"sleep 300 & echo $! > {pidfile}; sleep 300",
            )
        )
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text().strip().isdigit():
                break
            await asyncio.sleep(0.05)
        grandchild = int(pidfile.read_text().strip())
        assert _pid_alive(grandchild), "grandchild never started; the test proves nothing"

        await backend.cancel(job)

        for _ in range(100):  # reaping is a signal, not an instant
            if not _pid_alive(grandchild):
                break
            await asyncio.sleep(0.05)
        assert not _pid_alive(grandchild), (
            f"pid {grandchild} survived the cancel — the work outlived the job"
        )
        await _wait_until_terminal(backend, job)
        assert (await backend.status(job)).state == "cancelled"

    async def test_cancel_does_not_signal_the_orchestrator(self) -> None:
        # The child leads its own group precisely so killpg cannot reach back at us. If the
        # child shared our group, this cancel would deliver SIGTERM to the test process.
        backend = LocalBackend()
        job = await backend.submit(Task(name="sleep", run="sleep 30"))
        for _ in range(100):
            if (await backend.status(job)).state == "running":
                break
            await asyncio.sleep(0.05)

        await backend.cancel(job)  # a signal to our own group would kill the test run

        assert os.getpid() == os.getpid()  # reached at all == we were not signalled
        assert (await backend.status(job)).state == "cancelled"

    async def test_cancel_returns_even_when_a_child_holds_the_pipes(self, tmp_path: Path) -> None:
        """Cancel must not wait on pipe EOF, which a surviving descendant can hold open forever.

        `Process.wait()` resolves on EOF, not on exit, so cancelling a job whose grandchild
        inherited stdout used to block until that grandchild closed it — waiting on the very
        thing being killed. The serving teardown wrapped the call in `contextlib.suppress`,
        which cannot interrupt a hang.
        """
        backend = LocalBackend()
        marker = tmp_path / "started"
        job = await backend.submit(
            Task(name="holder", run=f"sleep 300 & touch {marker}; sleep 300")
        )
        for _ in range(100):
            if marker.exists():
                break
            await asyncio.sleep(0.05)

        # The assertion is the timeout: a cancel that waits on pipes never returns.
        await asyncio.wait_for(backend.cancel(job), timeout=20)


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
        from strata_forge.compute.job import Job

        bogus = Job(id="not-real", backend="local", task_name="t")
        with pytest.raises(ValueError, match="unknown job"):
            await backend.status(bogus)

    async def test_status_wrong_backend_rejected(self) -> None:
        backend = LocalBackend()
        from strata_forge.compute.job import Job

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


class TestStreamedOutput:
    """A hung process is the case worth diagnosing, so its output must be reachable early.

    The failure these guard: reading both pipes to EOF yields nothing at all until the
    process exits, so a model server that comes up wrong and then sits there hands back an
    empty log for as long as it hangs — and the buffers die with the runner afterwards.
    """

    async def test_logs_readable_while_the_process_is_still_running(self) -> None:
        backend = LocalBackend()
        # `exec` so bash becomes the sleeper: one process, so cancel closes the pipe at once.
        job = await backend.submit(Task(name="hang", run="echo EARLY-STDOUT; exec sleep 10"))
        try:
            captured: list[str] = []

            def _seen() -> bool:
                captured.append(backend._jobs[job.id].stdout_buffer.decode())  # pyright: ignore[reportPrivateUsage]
                return "EARLY-STDOUT" in captured[-1]

            await _wait_until(_seen, timeout=5.0)
            assert "EARLY-STDOUT" in await backend.logs(job)
            assert (await backend.status(job)).state == "running"
        finally:
            await backend.cancel(job)

    async def test_terminal_state_does_not_wait_on_an_orphan_holding_the_pipe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The child exits but a process it backgrounded keeps the write end of the pipe, so
        # EOF never comes. Gating termination on EOF would pin the job at "running" forever.
        monkeypatch.setattr(local_backend, "_DRAIN_GRACE_S", 0.2)
        backend = LocalBackend()
        job = await backend.submit(Task(name="orphan", run="sleep 10 & echo PARENT-DONE"))
        await _wait_until_terminal(backend, job, timeout=5.0)
        assert (await backend.status(job)).state == "succeeded"
        assert "PARENT-DONE" in await backend.logs(job)


class TestServeLogFiles:
    async def test_no_files_are_written_without_log_dir(self, tmp_path: Path) -> None:
        # Opt-in: a library user who never asks for logs never finds files beside their code.
        backend = LocalBackend()
        job = await backend.submit(Task(name="t", run="echo hi", workdir=str(tmp_path)))
        await _wait_until_terminal(backend, job)
        assert _serve_log_names(tmp_path) == []

    async def test_streams_to_disk_before_the_process_exits(self, tmp_path: Path) -> None:
        backend = LocalBackend(log_dir=tmp_path)
        job = await backend.submit(Task(name="hang", run="echo EARLY-STDOUT; exec sleep 10"))
        try:
            out = tmp_path / "serve.stdout.log"
            await _wait_until(
                lambda: out.exists() and "EARLY-STDOUT" in out.read_text(), timeout=5.0
            )
            assert (await backend.status(job)).state == "running"
        finally:
            await backend.cancel(job)

    async def test_files_outlive_cleanup(self, tmp_path: Path) -> None:
        backend = LocalBackend(log_dir=tmp_path)
        job = await backend.submit(Task(name="t", run="echo OUT; echo ERR 1>&2"))
        await _wait_until_terminal(backend, job)
        await backend.cleanup(job)
        # The in-memory state is gone; the artifact is the whole point of writing it down.
        with pytest.raises(ValueError, match="unknown job"):
            await backend.status(job)
        assert "OUT" in (tmp_path / "serve.stdout.log").read_text()
        assert "ERR" in (tmp_path / "serve.stderr.log").read_text()

    async def test_unusable_log_dir_degrades_instead_of_failing_the_job(
        self, tmp_path: Path
    ) -> None:
        # A log path is never worth failing a job over — the job runs, the reason is recorded.
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        backend = LocalBackend(log_dir=blocker / "logs")
        job = await backend.submit(Task(name="t", run="echo STILL-RAN"))
        await _wait_until_terminal(backend, job)
        assert (await backend.status(job)).state == "succeeded"
        logs = await backend.logs(job)
        assert "STILL-RAN" in logs
        assert "log_dir unusable" in logs


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

    async def test_rejects_symlink_escape(self, tmp_path: Path) -> None:
        # A symlink INSIDE the workdir pointing outside is rejected by the realpath
        # check — the lexical guard alone cannot catch this.
        (tmp_path / "secret.txt").write_text("top-secret")
        run = tmp_path / "run"
        run.mkdir()
        (run / "link.txt").symlink_to(tmp_path / "secret.txt")
        backend = LocalBackend()
        job = await self._job_in(backend, run)
        with pytest.raises(ValueError, match="resolves outside the job workdir"):
            await backend.read_file(job, "link.txt")
