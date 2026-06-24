"""Unit tests for `forge.compute.backends.ssh.SSHBackend`."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import AsyncMock

import pytest

from forge.compute import Backend, SSHBackend, Task

# ---------------------------------------------------------------------------
# Fake asyncssh connection
# ---------------------------------------------------------------------------


class _FakeProcessResult:
    """Mimics asyncssh's ProcessResult."""

    def __init__(self, exit_status: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_status = exit_status
        self.stdout = stdout
        self.stderr = stderr


class _FakeSSHConnection:
    """Records commands; returns stubbed results from a script."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.script: list[_FakeProcessResult] = []
        self.closed = False
        self.last_timeout: float | None = None

    def queue(self, *results: _FakeProcessResult) -> None:
        self.script.extend(results)

    # This double mirrors asyncssh's `run(*, timeout=...)` API, so it deliberately accepts a
    # `timeout` parameter (ASYNC109 targets real async code that should use `asyncio.timeout`).
    async def run(
        self, command: str, *, check: bool = False, timeout: float | None = None  # noqa: ASYNC109
    ) -> _FakeProcessResult:
        self.commands.append(command)
        self.last_timeout = timeout
        result = self.script.pop(0) if self.script else _FakeProcessResult()
        if check and result.exit_status != 0:
            err = f"command failed (exit {result.exit_status}): {command!r}"
            raise RuntimeError(err)
        return result

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


@pytest.fixture
def fake_connection() -> _FakeSSHConnection:
    return _FakeSSHConnection()


@pytest.fixture
def backend(fake_connection: _FakeSSHConnection) -> SSHBackend:
    return SSHBackend(connection=fake_connection)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_explicit_connection_short_circuits_creds(self) -> None:
        backend = SSHBackend(connection=_FakeSSHConnection())
        assert backend.name == "ssh"

    def test_missing_host_username_rejected(self) -> None:
        with pytest.raises(ValueError, match="connection"):
            SSHBackend()

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            SSHBackend(connection=_FakeSSHConnection(), name="")

    def test_extra_missing_raises_at_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio

        monkeypatch.setitem(sys.modules, "asyncssh", None)
        backend = SSHBackend(host="example.com", username="me")
        with pytest.raises(ImportError, match=r"\[compute\] extra"):
            asyncio.run(backend.submit(Task(name="t", run="echo")))

    async def test_satisfies_backend_protocol(self) -> None:
        backend = SSHBackend(connection=_FakeSSHConnection())
        assert isinstance(backend, Backend)


# ---------------------------------------------------------------------------
# Timeouts — an unresponsive host must not stall a caller indefinitely
# ---------------------------------------------------------------------------


class TestTimeouts:
    async def test_connect_applies_handshake_timeouts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        async def fake_connect(**kwargs: Any) -> _FakeSSHConnection:
            captured.update(kwargs)
            return _FakeSSHConnection()

        monkeypatch.setitem(sys.modules, "asyncssh", types.SimpleNamespace(connect=fake_connect))
        backend = SSHBackend(host="example.com", username="me")
        await backend._get_connection()  # pyright: ignore[reportPrivateUsage]
        assert captured["connect_timeout"] > 0
        assert captured["login_timeout"] > 0

    async def test_remote_commands_carry_a_timeout(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        await backend.submit(Task(name="t", run="echo"))
        assert fake_connection.last_timeout is not None
        assert fake_connection.last_timeout > 0


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    async def test_creates_workdir_writes_wrapper_launches(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        # Queue results: mkdir → ok; cat wrapper → ok; launch → echoes PID.
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="12345\n"),
        )
        task = Task(name="my-task", run="echo hi", env={"FOO": "bar"})
        job = await backend.submit(task)
        assert job.backend == "ssh"
        assert job.task_name == "my-task"
        assert job.metadata["pid"] == "12345"
        assert "remote_workdir" in job.metadata

        # First command makes the workdir.
        assert "mkdir -p" in fake_connection.commands[0]
        # Second writes the wrapper script (heredoc with the run command).
        assert "wrapper.sh" in fake_connection.commands[1]
        assert "echo hi" in fake_connection.commands[1]
        assert "export FOO=" in fake_connection.commands[1]
        # Third launches under nohup.
        assert "nohup" in fake_connection.commands[2]

    async def test_setup_runs_before_run(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="999\n"),
        )
        await backend.submit(Task(name="t", setup="pip install x", run="python x.py"))
        wrapper_cmd = fake_connection.commands[1]
        # setup && run inside the wrapper.
        assert "pip install x && python x.py" in wrapper_cmd

    async def test_multi_node_rejected(self, backend: SSHBackend) -> None:
        with pytest.raises(ValueError, match="single-node"):
            await backend.submit(Task(name="t", run="hi", num_nodes=2))

    async def test_unparseable_pid_raises(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="garbage-output\n"),
        )
        with pytest.raises(RuntimeError, match="pid output"):
            await backend.submit(Task(name="t", run="echo"))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    async def test_running(self, backend: SSHBackend, fake_connection: _FakeSSHConnection) -> None:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="sleep 30"))
        # Probe: "RUNNING".
        fake_connection.queue(_FakeProcessResult(stdout="RUNNING\n"))
        status = await backend.status(job)
        assert status.state == "running"

    async def test_succeeded(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="echo"))
        # Probe: exit-file contents 0.
        fake_connection.queue(_FakeProcessResult(stdout="0\n"))
        status = await backend.status(job)
        assert status.state == "succeeded"
        assert status.exit_code == 0

    async def test_failed_with_exit_code(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="false"))
        fake_connection.queue(_FakeProcessResult(stdout="1\n"))
        status = await backend.status(job)
        assert status.state == "failed"
        assert status.exit_code == 1

    async def test_cancelled_when_pid_gone_no_exit_file(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="sleep 30"))
        fake_connection.queue(_FakeProcessResult(stdout="MISSING\n"))
        status = await backend.status(job)
        assert status.state == "cancelled"

    async def test_unparseable_exit_file(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="echo"))
        fake_connection.queue(_FakeProcessResult(stdout="not-a-number\n"))
        status = await backend.status(job)
        assert status.state == "failed"
        assert "unparseable" in status.message


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


class TestLogs:
    async def test_full_logs(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="echo"))
        fake_connection.queue(_FakeProcessResult(stdout="line1\nline2\n"))
        logs = await backend.logs(job)
        assert "line1" in logs

    async def test_tail(self, backend: SSHBackend, fake_connection: _FakeSSHConnection) -> None:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="echo"))
        fake_connection.queue(_FakeProcessResult(stdout="last-2-lines\n"))
        logs = await backend.logs(job, tail=2)
        # The command sent to the host used `tail -n 2`.
        assert "tail -n 2" in fake_connection.commands[-1]
        assert "last-2-lines" in logs

    async def test_invalid_tail(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="echo"))
        with pytest.raises(ValueError, match="tail"):
            await backend.logs(job, tail=0)


# ---------------------------------------------------------------------------
# read_file (workdir-confined side-channel reads, e.g. progress.jsonl)
# ---------------------------------------------------------------------------


class TestReadFile:
    async def _submit(self, backend: SSHBackend, fake_connection: _FakeSSHConnection) -> Any:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        return await backend.submit(Task(name="t", run="echo"))

    async def test_reads_workdir_file(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        job = await self._submit(backend, fake_connection)
        fake_connection.queue(_FakeProcessResult(stdout='{"step": 1}\n'))
        out = await backend.read_file(job, "progress.jsonl")
        assert out == '{"step": 1}\n'
        cmd = fake_connection.commands[-1]
        assert cmd.startswith("head -c ")  # byte-capped read
        assert f"{job.metadata['remote_workdir']}/progress.jsonl" in cmd

    async def test_tail(self, backend: SSHBackend, fake_connection: _FakeSSHConnection) -> None:
        job = await self._submit(backend, fake_connection)
        fake_connection.queue(_FakeProcessResult(stdout="last\n"))
        await backend.read_file(job, "progress.jsonl", tail=3)
        assert "tail -n 3" in fake_connection.commands[-1]

    async def test_invalid_tail(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        job = await self._submit(backend, fake_connection)
        with pytest.raises(ValueError, match="tail"):
            await backend.read_file(job, "progress.jsonl", tail=0)

    async def test_missing_file_returns_empty(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        job = await self._submit(backend, fake_connection)
        fake_connection.queue(_FakeProcessResult(stdout=""))  # cat ... 2>/dev/null -> empty
        assert await backend.read_file(job, "missing.jsonl") == ""

    async def test_rejects_traversal(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        job = await self._submit(backend, fake_connection)
        before = len(fake_connection.commands)
        with pytest.raises(ValueError, match="within the job workdir"):
            await backend.read_file(job, "../../etc/passwd")
        # The unsafe path is rejected before any remote command runs.
        assert len(fake_connection.commands) == before

    async def test_rejects_absolute(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        job = await self._submit(backend, fake_connection)
        before = len(fake_connection.commands)
        with pytest.raises(ValueError, match="workdir-relative"):
            await backend.read_file(job, "/etc/passwd")
        assert len(fake_connection.commands) == before


# ---------------------------------------------------------------------------
# cancel / cleanup
# ---------------------------------------------------------------------------


class TestCancelCleanup:
    async def test_cancel_sends_term_then_kill(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="sleep 30"))
        fake_connection.queue(_FakeProcessResult())
        await backend.cancel(job)
        # The cancel command should reference both signals.
        cancel_cmd = fake_connection.commands[-1]
        assert "kill -TERM" in cancel_cmd
        assert "kill -KILL" in cancel_cmd

    async def test_cleanup_removes_remote_workdir(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="echo"))
        fake_connection.queue(_FakeProcessResult())
        await backend.cleanup(job)
        # Cleanup `rm -rf`s the workdir.
        cleanup_cmd = fake_connection.commands[-1]
        assert "rm -rf" in cleanup_cmd
        assert job.metadata["remote_workdir"] in cleanup_cmd

    async def test_cleanup_refuses_outside_remote_root(
        self, fake_connection: _FakeSSHConnection
    ) -> None:
        backend = SSHBackend(connection=fake_connection, remote_root=".forge-test")
        # Build a job whose metadata pretends a workdir outside the root.
        from forge.compute.job import Job

        bad_job = Job(
            id="bogus",
            backend="ssh",
            task_name="x",
            metadata={"remote_workdir": "/etc/passwd", "pid": "1"},
        )
        with pytest.raises(ValueError, match="not under remote_root"):
            await backend.cleanup(bad_job)


# ---------------------------------------------------------------------------
# Wrong-backend rejection
# ---------------------------------------------------------------------------


class TestJobOwnership:
    async def test_status_wrong_backend(self, backend: SSHBackend) -> None:
        from forge.compute.job import Job

        bogus = Job(
            id="x",
            backend="not-ssh",
            task_name="t",
            metadata={"remote_workdir": ".forge-compute/x", "pid": "1"},
        )
        with pytest.raises(ValueError, match="not"):
            await backend.status(bogus)

    async def test_status_missing_metadata(self, backend: SSHBackend) -> None:
        from forge.compute.job import Job

        bogus = Job(id="x", backend="ssh", task_name="t")
        with pytest.raises(ValueError, match="remote_workdir"):
            await backend.status(bogus)


# ---------------------------------------------------------------------------
# Settings-based connection construction
# ---------------------------------------------------------------------------


class TestSettingsBackedConnection:
    async def test_lazy_imports_asyncssh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Inject a fake `asyncssh` whose `connect` returns our FakeSSHConnection.
        fake_module = types.ModuleType("asyncssh")
        fake_module.connect = AsyncMock(  # type: ignore[attr-defined]
            return_value=_FakeSSHConnection()
        )
        monkeypatch.setitem(sys.modules, "asyncssh", fake_module)

        backend = SSHBackend(host="my-host", username="me", port=2222)
        # First call triggers connect with our settings.
        conn = await backend._get_connection()  # type: ignore[attr-defined]
        assert isinstance(conn, _FakeSSHConnection)
        # Subsequent calls return the cached connection.
        conn2 = await backend._get_connection()  # type: ignore[attr-defined]
        assert conn2 is conn

    async def test_connect_forwards_client_keys_and_passphrase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Exercise the `client_keys` and `passphrase` branches in _get_connection.
        captured: dict[str, Any] = {}

        async def _fake_connect(**kwargs: Any) -> _FakeSSHConnection:
            captured.update(kwargs)
            return _FakeSSHConnection()

        fake_module = types.ModuleType("asyncssh")
        fake_module.connect = _fake_connect  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "asyncssh", fake_module)

        backend = SSHBackend(
            host="h",
            username="u",
            client_keys=["/k1", "/k2"],
            passphrase="hunter2",
        )
        await backend._get_connection()  # type: ignore[attr-defined]
        assert captured["client_keys"] == ["/k1", "/k2"]
        assert captured["passphrase"] == "hunter2"


class TestJobMetadataErrors:
    async def test_status_rejects_job_without_pid(self, backend: SSHBackend) -> None:
        from forge.compute.job import Job

        bogus = Job(
            id="x",
            backend="ssh",
            task_name="t",
            metadata={"remote_workdir": ".forge-compute/x"},  # no pid
        )
        with pytest.raises(ValueError, match="missing pid"):
            await backend.status(bogus)

    async def test_status_handles_unparseable_submitted_at(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        from forge.compute.job import Job

        # Manually build a job with a broken submitted_at; status should still work.
        fake_connection.queue(_FakeProcessResult(stdout="RUNNING\n"))
        job = Job(
            id="abc",
            backend="ssh",
            task_name="t",
            metadata={
                "remote_workdir": ".forge-compute/abc",
                "pid": "100",
                "submitted_at": "not-an-isoformat",
            },
        )
        status = await backend.status(job)
        assert status.state == "running"
        # Unparseable submitted_at falls back to None — line 226-227 path.
        assert status.started_at is None


class TestClose:
    async def test_close_with_no_explicit_connection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Backend constructed with host/username (so it owns the connection)
        # — close() must invoke close + wait_closed on the cached connection.
        fake_conn = _FakeSSHConnection()
        fake_module = types.ModuleType("asyncssh")
        fake_module.connect = AsyncMock(return_value=fake_conn)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "asyncssh", fake_module)

        backend = SSHBackend(host="h", username="u")
        await backend._get_connection()  # type: ignore[attr-defined]
        await backend.close()
        assert fake_conn.closed is True

    async def test_close_with_explicit_connection_is_noop(self) -> None:
        # Connection passed by the caller — close() must not touch it.
        conn = _FakeSSHConnection()
        backend = SSHBackend(connection=conn)
        await backend.close()
        assert conn.closed is False
