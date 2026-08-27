"""Unit tests for `strata_forge.compute.backends.ssh.SSHBackend`."""

from __future__ import annotations

import asyncio
import base64
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import types
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from strata_forge.compute import MAX_CONSOLE_CHUNK_BYTES, Backend, Job, SSHBackend, Task
from strata_forge.compute.backends.ssh import (
    _REPORT_MARKER,  # pyright: ignore[reportPrivateUsage]
)

# ---------------------------------------------------------------------------
# Fake asyncssh connection
# ---------------------------------------------------------------------------


class _FakeProcessResult:
    """Mimics asyncssh's ProcessResult."""

    def __init__(self, exit_status: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_status = exit_status
        self.stdout = stdout
        self.stderr = stderr


#: How much of a reply this double hands back per `read` call. Small enough that every console
#: reply in these tests spans several reads, which is the case the real reader always produces and
#: a single-shot double never does.
_FAKE_READ_CHUNK = 48


class _FakeReader:
    """Mimics asyncssh's `SSHReader`, INCLUDING that `read(n)` is `read up to n`.

    asyncssh returns the moment any data is buffered rather than waiting for `n` characters, so a
    reply written by the remote in several pieces takes several reads to collect. A double that
    hands back the whole string on the first call cannot fail for a caller that only reads once —
    which is exactly the bug this shape exists to catch.
    """

    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0
        #: The bounds this reader was asked for, in order — the caller's own cap is the only thing
        #: standing between the control plane and a reply sized by the remote.
        self.requested: list[int] = []

    async def read(self, n: int = -1) -> str:
        self.requested.append(n)
        if n < 0:
            piece, self._pos = self._text[self._pos :], len(self._text)
            return piece
        take = min(n, _FAKE_READ_CHUNK)
        piece = self._text[self._pos : self._pos + take]
        self._pos += len(piece)
        return piece


class _NeverEndingReader(_FakeReader):
    """A remote that finishes its reply and then holds the channel open forever.

    Not exotic: any login shell that backgrounds a process inheriting stdout does this, and so
    does a deliberately hostile one. The reply arrives in full; EOF never does.
    """

    async def read(self, n: int = -1) -> str:
        piece = await super().read(n)
        if piece:
            return piece
        # Exhausted, and no EOF is coming. A reader waiting for one waits for the deadline.
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")  # pragma: no cover


class _FakeProcess:
    def __init__(self, stdout: str, *, never_ends: bool = False) -> None:
        self.stdout = _NeverEndingReader(stdout) if never_ends else _FakeReader(stdout)
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


class _FakeSSHConnection:
    """Records commands; returns stubbed results from a script."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.script: list[_FakeProcessResult] = []
        self.closed = False
        self.last_timeout: float | None = None
        self.last_process: _FakeProcess | None = None
        #: Make every process hold its channel open after the reply, like a shell that backgrounds
        #: something inheriting stdout.
        self.never_ends = False

    def queue(self, *results: _FakeProcessResult) -> None:
        self.script.extend(results)

    # This double mirrors asyncssh's `run(*, timeout=...)` API, so it deliberately accepts a
    # `timeout` parameter (ASYNC109 targets real async code that should use `asyncio.timeout`).
    async def run(
        self,
        command: str,
        *,
        check: bool = False,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> _FakeProcessResult:
        self.commands.append(command)
        self.last_timeout = timeout
        result = self.script.pop(0) if self.script else _FakeProcessResult()
        # The real launcher prints a structured report, not a bare pid. A test that queues a
        # number is naming the PID, not describing the wire format, so render it the way the
        # remote shell would — including the job landing in a group of its own, which is the
        # normal case. A test about the legacy (no-group) path queues the report itself.
        if _REPORT_MARKER in command and result.stdout.strip().isdigit():
            pid = result.stdout.strip()
            result = _FakeProcessResult(
                result.exit_status,
                f"{_REPORT_MARKER} pid={pid} m=1 g={pid}\n",
                result.stderr,
            )
        if check and result.exit_status != 0:
            err = f"command failed (exit {result.exit_status}): {command!r}"
            raise RuntimeError(err)
        return result

    async def create_process(self, command: str) -> _FakeProcess:
        """Mimics asyncssh's `create_process` — the BOUNDED read path.

        `run()` collects the whole reply before returning it; the console read deliberately does
        not use it, so the double has to offer the streaming shape too.
        """
        result = await self.run(command)
        self.last_process = _FakeProcess(result.stdout, never_ends=self.never_ends)
        return self.last_process

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
    async def test_connect_applies_handshake_timeouts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
# Launcher detachment
#
# sshd holds the session channel open until the command's stdout and stderr reach EOF, and every
# process in a backgrounded tree inherits them. A launcher that leaves them open therefore keeps
# submit() blocked for the whole lifetime of the job it just launched.
#
# `subprocess.run` reads its pipes to EOF exactly as sshd waits on the channel, so running the
# generated command under a local shell reproduces the failure without needing an SSH server: an
# undetached launcher makes the call block until the fake job finishes.
# ---------------------------------------------------------------------------


async def _launch_command(backend: SSHBackend, connection: _FakeSSHConnection) -> tuple[str, str]:
    """Submit once and return (launch command, remote workdir)."""
    connection.queue(
        _FakeProcessResult(),
        _FakeProcessResult(),
        _FakeProcessResult(stdout="4242\n"),
    )
    job = await backend.submit(Task(name="t", run="echo hi"))
    return connection.commands[2], str(job.metadata["remote_workdir"])


def _pid_alive(pid: int) -> bool:
    """Does `pid` still exist? Signal 0 checks without delivering anything."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _inner_script(command: str) -> str:
    """The script the launcher hands to its explicit `bash -c`."""
    parts = shlex.split(command)
    assert parts[:2] == ["bash", "-c"], command
    return parts[2]


def _report_line(stdout: str) -> dict[str, str]:
    """Parse the launcher's `key=value` report out of a real shell's output."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(_REPORT_MARKER):
            fields: dict[str, str] = {}
            for token in line[len(_REPORT_MARKER) :].split():
                key, _, value = token.partition("=")
                fields[key] = value
            return fields
    raise AssertionError(f"no launch report in output: {stdout!r}")


class TestLauncherDetachment:
    async def test_backgrounded_subshell_redirects_its_own_descriptors(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        command, _ = await _launch_command(backend, fake_connection)
        # The redirection must sit on the subshell itself, before `&`. Redirecting only the
        # commands inside it leaves the subshell holding the channel.
        assert re.search(r"\)\s*<\s*/dev/null\s*>\s*/dev/null\s*2>&1\s*&", command), command

    async def test_launch_is_confined_to_a_brace_group(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        command, _ = await _launch_command(backend, fake_connection)
        script = _inner_script(command)
        # `&` binds looser than `&&`. Without the brace group the shell backgrounds the entire
        # `cd … && ( … )` and-list, forking an outer subshell that inherits the channel — the
        # redirection above is then not enough on its own.
        assert "&& { " in script, script
        assert script.rstrip().endswith("; }"), script

    async def test_bookkeeping_subshell_ignores_hangup(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        command, _ = await _launch_command(backend, fake_connection)
        # nohup protects only the process it execs, so a hangup between launch and completion
        # would otherwise kill the subshell before it records the exit code.
        assert 'trap "" HUP' in _inner_script(command), command

    async def test_launcher_asserts_job_control_under_an_explicit_bash(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        """Job control is what gives the job a process group cancel can signal.

        It has to be requested explicitly AND under a named bash: sshd hands the command to the
        user's LOGIN shell, and that lottery decides whether the job gets its own group at all —
        dash and sh accept `set -m` but still leave background jobs in the session's group, and
        zsh rejects the option outright and would launch nothing.
        """
        command, _ = await _launch_command(backend, fake_connection)
        assert command.startswith("bash -c "), command
        assert _inner_script(command).startswith("set -m; "), command

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell semantics")
    def test_launcher_returns_before_the_job_finishes(self, tmp_path: Path) -> None:
        """The regression itself: the launcher must not block for the job's lifetime."""
        command, workdir = asyncio.run(self._submit_into(tmp_path, sleep_seconds=3, exit_code=7))

        started = time.monotonic()
        completed = subprocess.run(  # noqa: S603 — the command under test IS the input
            ["/bin/bash", "-c", command],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        elapsed = time.monotonic() - started

        assert completed.returncode == 0, completed.stderr
        # The fake job sleeps for 3s; an attached launcher returns only after it exits.
        assert elapsed < 1.5, f"launcher blocked for {elapsed:.2f}s — it is still attached"
        report = _report_line(completed.stdout)
        assert report["pid"].isdigit(), completed.stdout
        # Run under a real bash, the launcher must actually get job control and land the job in
        # a group of its own — the property the whole cancel path depends on.
        assert report["m"] == "1", completed.stdout
        assert report["g"] == report["pid"], completed.stdout

        # The handle belongs in the workdir. With `cd` inside the backgrounded list it lands in
        # the login shell's cwd instead, where concurrent submits overwrite each other's pid.
        assert (tmp_path / workdir / "forge.pid").is_file()
        assert not (tmp_path / "forge.pid").exists()

        # Detaching must not cost the bookkeeping: the job still records its exit code and logs.
        exit_file = tmp_path / workdir / "forge.exit"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not exit_file.is_file():
            time.sleep(0.1)
        assert exit_file.is_file(), "the job never recorded an exit code"
        assert exit_file.read_text().strip() == "7"
        assert (tmp_path / workdir / "stdout.log").read_text().strip() == "OUT"
        assert (tmp_path / workdir / "stderr.log").read_text().strip() == "ERR"

    @staticmethod
    async def _submit_into(root: Path, *, sleep_seconds: int, exit_code: int) -> tuple[str, str]:
        """Generate a real launch command and stage a fake job for it to run."""
        connection = _FakeSSHConnection()
        command, workdir = await _launch_command(SSHBackend(connection=connection), connection)
        staged = root / workdir
        staged.mkdir(parents=True)
        (staged / "wrapper.sh").write_text(
            f"echo OUT\necho ERR >&2\nsleep {sleep_seconds}\nexit {exit_code}\n"
        )
        return command, workdir


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

    async def test_a_process_that_vanished_uncancelled_is_a_failure(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        """Nothing may claim a death was asked for unless someone asked.

        Pid gone with no exit file and no cancellation marker means the MACHINE took it — an OOM
        kill, a reboot, a segfault in the interpreter. Reporting that as `cancelled` was not just
        wrong: an orchestrator that captures diagnostics only for FAILED jobs then discards the
        logs of exactly the deaths nobody can otherwise explain.
        """
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="sleep 30"))
        fake_connection.queue(_FakeProcessResult(stdout="MISSING\n"))
        status = await backend.status(job)
        assert status.state == "failed"
        assert "without recording an exit code" in (status.message or "")

    async def test_a_cancelled_job_reports_cancelled(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        # `cancel` leaves a marker before it signals, which is the only evidence that separates a
        # cancellation from a kill — afterwards the two are identical: pid gone, no exit file.
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="sleep 30"))
        fake_connection.queue(_FakeProcessResult(stdout="CANCELLED\n"))
        status = await backend.status(job)
        assert status.state == "cancelled"

    async def test_cancel_marks_before_it_signals(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        # Between the marker and the signal the process is already dying; a status landing in that
        # window would otherwise read the death as one nobody asked for. So the order is load-bearing.
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="sleep 30"))
        fake_connection.queue(_FakeProcessResult())
        await backend.cancel(job)
        cmd = fake_connection.commands[-1]
        assert "forge.cancelled" in cmd
        assert cmd.index("forge.cancelled") < cmd.index("kill -TERM")

    async def test_an_empty_exit_file_is_not_a_verdict(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        """The wrapper creates the exit file and writes to it as two steps.

        A poll landing between them sees an empty file, which is not an outcome — it is a race.
        `splitlines()[-1]` raised IndexError on it, which `except ValueError` never caught, so the
        exception escaped `status` entirely and reached the caller's poll loop.
        """
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="echo"))
        fake_connection.queue(_FakeProcessResult(stdout=""))
        status = await backend.status(job)
        # Non-terminal: keep the job alive for one more poll rather than invent an outcome.
        assert status.state == "running"
        assert status.exit_code is None

    async def test_a_whitespace_only_exit_file_is_not_a_verdict(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        # Same race, one flush later: the newline landed and the digits did not.
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        job = await backend.submit(Task(name="t", run="echo"))
        fake_connection.queue(_FakeProcessResult(stdout="\n  \n"))
        status = await backend.status(job)
        assert status.state == "running"

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
        from strata_forge.compute.job import Job

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
        from strata_forge.compute.job import Job

        bogus = Job(
            id="x",
            backend="not-ssh",
            task_name="t",
            metadata={"remote_workdir": ".forge-compute/x", "pid": "1"},
        )
        with pytest.raises(ValueError, match="not"):
            await backend.status(bogus)

    async def test_status_missing_metadata(self, backend: SSHBackend) -> None:
        from strata_forge.compute.job import Job

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
        from strata_forge.compute.job import Job

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
        from strata_forge.compute.job import Job

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


class TestCancelReachesTheWholeTree:
    """Cancel must stop the WORK, not the bookkeeper that launched it.

    The pid the launcher records belongs to a bookkeeping subshell; the run — the wrapper, the
    Python runner, the inference engine holding the GPU — lives below it. Signalling that pid
    alone reparents all of it onto init, where it keeps the device busy for the next run. These
    drive real bash against real process trees, because the property is about process groups and
    signal delivery, which no assertion on a command string can establish.
    """

    async def test_the_signal_target_is_the_group(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        fake_connection.queue(
            _FakeProcessResult(), _FakeProcessResult(), _FakeProcessResult(stdout="4242\n")
        )
        job = await backend.submit(Task(name="t", run="sleep 1"))
        assert job.metadata["pgroup"] is True

        await backend.cancel(job)
        cancel_cmd = fake_connection.commands[-1]
        # A leading `-` is what makes kill address the GROUP. `--` keeps it an argument rather
        # than an option, which is why it is spelled this way and not `kill -TERM -4242`.
        assert "kill -TERM -- -4242" in cancel_cmd, cancel_cmd
        assert "kill -KILL -- -4242" in cancel_cmd, cancel_cmd

    async def test_a_job_without_its_own_group_keeps_the_single_pid_path(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        """A shell that could not give the job its own group must not be group-signalled.

        Its pid names no group, so `kill -- -PID` fails with ESRCH and stops nothing at all —
        and the same probe would report a live job as gone. Jobs submitted before this existed
        are in exactly that position, which is why the marker is recorded rather than assumed.
        """
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout=f"{_REPORT_MARKER} pid=77 m=0 g=1\n"),
        )
        job = await backend.submit(Task(name="t", run="sleep 1"))
        assert job.metadata["pgroup"] is False

        await backend.cancel(job)
        assert "kill -TERM -- 77" in fake_connection.commands[-1]
        assert "-- -77" not in fake_connection.commands[-1]

    async def test_a_disagreeing_pgid_is_not_trusted(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        # The shell claimed job control, but ps measured the job in a DIFFERENT group. Believing
        # the claim would group-signal a pid that leads no group.
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout=f"{_REPORT_MARKER} pid=77 m=1 g=1234\n"),
        )
        job = await backend.submit(Task(name="t", run="sleep 1"))
        assert job.metadata["pgroup"] is False

    async def test_liveness_is_asked_of_the_group(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        # The bookkeeper can die while the runner it launched still holds the GPU. A pid-only
        # probe calls that finished, so a half-killed job would read terminal while it runs on.
        fake_connection.queue(
            _FakeProcessResult(), _FakeProcessResult(), _FakeProcessResult(stdout="4242\n")
        )
        job = await backend.submit(Task(name="t", run="sleep 1"))
        fake_connection.queue(_FakeProcessResult(stdout="RUNNING\n"))
        await backend.status(job)
        assert "kill -0 -4242" in fake_connection.commands[-1], fake_connection.commands[-1]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
    def test_the_launched_group_really_contains_the_whole_tree(self, tmp_path: Path) -> None:
        """End to end against real bash: signal the group, and the descendants die.

        This is the regression. The old launcher recorded a pid that named no group, so the
        only thing a cancel could reach was the subshell — every process doing the actual work
        survived it.
        """
        connection = _FakeSSHConnection()
        command, workdir = asyncio.run(
            _launch_command(SSHBackend(connection=connection), connection)
        )
        staged = tmp_path / workdir
        staged.mkdir(parents=True)
        # A wrapper whose real work is a GRANDCHILD, like the runner's model server.
        (staged / "wrapper.sh").write_text(
            "sleep 300 &\necho $! > deep.pid\nsleep 300\n",
        )

        completed = subprocess.run(  # noqa: S603 — the command under test IS the input
            ["/bin/bash", "-c", command],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        report = _report_line(completed.stdout)
        pgid = int(report["pid"])

        deep_pid_file = staged / "deep.pid"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not deep_pid_file.is_file():
            time.sleep(0.1)
        assert deep_pid_file.is_file(), "the grandchild never started; the test proves nothing"
        deep_pid = int(deep_pid_file.read_text().strip())
        assert _pid_alive(deep_pid)

        # Exactly what cancel does: signal the GROUP.
        os.killpg(pgid, signal.SIGKILL)

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and _pid_alive(deep_pid):
            time.sleep(0.1)
        assert not _pid_alive(deep_pid), (
            f"pid {deep_pid} survived the group kill — the work outlived the cancel"
        )

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
    def test_the_group_holds_the_job_and_not_the_launching_shell(self, tmp_path: Path) -> None:
        # The group must be the JOB's. If the job merely joined the caller's group, cancelling
        # it would signal the shell that launched it — on a real host, sshd's session.
        connection = _FakeSSHConnection()
        command, workdir = asyncio.run(
            _launch_command(SSHBackend(connection=connection), connection)
        )
        staged = tmp_path / workdir
        staged.mkdir(parents=True)
        (staged / "wrapper.sh").write_text("sleep 5\n")

        completed = subprocess.run(  # noqa: S603 — the command under test IS the input
            ["/bin/bash", "-c", f"echo launcher_pgid=$(ps -o pgid= -p $$ | tr -dc 0-9); {command}"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        launcher_pgid = next(
            line.split("=", 1)[1].strip()
            for line in completed.stdout.splitlines()
            if line.startswith("launcher_pgid=")
        )
        report = _report_line(completed.stdout)
        assert report["g"] != launcher_pgid, (
            f"the job joined the launcher's group {launcher_pgid} — cancelling it would "
            "signal the shell that started it"
        )


class TestConsole:
    """The incremental console read.

    Two halves, tested two ways: the WIRE protocol (framing, offsets, gap accounting) against the
    fake connection, and the remote SHELL SCRIPT against a real `bash` over real files — because
    the fake connection never executes the command, and the command is where this feature can
    actually be wrong.
    """

    @staticmethod
    async def _job(backend: SSHBackend, fake_connection: _FakeSSHConnection) -> Any:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        return await backend.submit(Task(name="t", run="echo"))

    @staticmethod
    def _reply(out: bytes, err: bytes, *, total_out: int, total_err: int) -> _FakeProcessResult:
        return _FakeProcessResult(
            stdout=(
                f"__forge_console__ {total_out} {total_err} {len(out)} {len(err)}\n"
                f"{base64.b64encode(out).decode()}\n{base64.b64encode(err).decode()}\n"
            )
        )

    async def test_reads_both_streams_and_reports_the_resume_offsets(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        job = await self._job(backend, fake_connection)
        fake_connection.queue(self._reply(b"hello\n", b"oops\n", total_out=6, total_err=5))
        chunk = await backend.console(job)
        assert chunk.stdout == "hello\n"
        assert chunk.stderr == "oops\n"
        assert (chunk.stdout_offset, chunk.stderr_offset) == (6, 5)
        assert chunk.dropped_bytes == 0

    async def test_a_reply_too_large_for_one_read_is_still_assembled(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        """The case a real run is made of, and the one a single-shot read gets wrong.

        `SSHReader.read(n)` returns as soon as anything is buffered, and the remote composes this
        reply in several writes -- so all but the smallest console read arrives in pieces. A reader
        that took the first piece for the whole reply would see payloads shorter than the header
        promised, correctly refuse to advance its offsets, and then do the same on every later
        poll: a console that stays empty for the life of the run while progress streams normally.
        """
        job = await self._job(backend, fake_connection)
        printed = b"".join(b"row %d / 10570 . ok . 402 ms\n" % i for i in range(200))
        fake_connection.queue(self._reply(printed, b"", total_out=len(printed), total_err=0))
        chunk = await backend.console(job, max_bytes=64 * 1024)
        assert chunk.stdout == printed.decode()
        assert chunk.stdout_offset == len(printed)
        assert chunk.dropped_bytes == 0
        # Several reads, each bounded: the loop is the fix, the cap is what makes it safe.
        reader = fake_connection.last_process.stdout
        assert len(reader.requested) > 1
        assert all(n > 0 for n in reader.requested)

    async def test_a_remote_that_never_hangs_up_does_not_stall_the_read(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        """The reply is self-delimiting, so waiting for EOF is waiting for nothing.

        EOF arrives when the remote closes the channel, and it does not do that while any process
        still holds stdout -- a login shell that backgrounds a daemon is enough. A read that waited
        for it would spend the whole command deadline on every poll of that target, and the
        orchestrator polls runs sequentially in one process shared by every account: one such
        target would stop other people's runs being reconciled at all.
        """
        job = await self._job(backend, fake_connection)
        fake_connection.never_ends = True
        fake_connection.queue(self._reply(b"hello\n", b"", total_out=6, total_err=0))
        # No deadline of its own: if the read waits for EOF this never returns and the suite hangs,
        # which is a louder failure than a wrong assertion.
        chunk = await asyncio.wait_for(backend.console(job), timeout=5)
        assert chunk.stdout == "hello\n"
        assert chunk.stdout_offset == 6

    async def test_passes_the_offsets_it_was_given_to_the_remote_side(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        job = await self._job(backend, fake_connection)
        fake_connection.queue(self._reply(b"", b"", total_out=6, total_err=0))
        await backend.console(job, stdout_offset=6, stderr_offset=2)
        command = fake_connection.commands[-1]
        assert "o-6" in command
        assert "e-2" in command

    async def test_an_overflowing_window_keeps_the_newest_and_says_what_it_dropped(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        """A silent jump-cut would read as a whole transcript. The hole is reported."""
        job = await self._job(backend, fake_connection)
        # The remote side wrote 1000 bytes; the cap allowed only the last 10. `max_bytes` is the
        # TOTAL for the call and is split evenly between the streams, so 20 buys 10 of stdout.
        fake_connection.queue(self._reply(b"0123456789", b"", total_out=1000, total_err=0))
        chunk = await backend.console(job, max_bytes=20)
        assert chunk.stdout == "0123456789"
        assert chunk.dropped_bytes == 990
        # Advanced to the file's real size: re-offering the skipped middle would leave a busy
        # job's reader permanently behind, dropping the same bytes forever.
        assert chunk.stdout_offset == 1000

    async def test_a_banner_before_the_marker_is_ignored(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        # A login shell's motd/rc chatter lands ahead of anything the command printed.
        job = await self._job(backend, fake_connection)
        reply = self._reply(b"hi\n", b"", total_out=3, total_err=0)
        fake_connection.queue(_FakeProcessResult(stdout="Welcome to Ubuntu\n" + reply.stdout))
        assert (await backend.console(job)).stdout == "hi\n"

    async def test_a_reply_with_no_marker_yields_nothing_and_keeps_the_offsets(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        """Never advance past bytes that were not read: that would lose them permanently."""
        job = await self._job(backend, fake_connection)
        fake_connection.queue(_FakeProcessResult(stdout="garbage\n"))
        chunk = await backend.console(job, stdout_offset=17, stderr_offset=4)
        assert chunk.stdout == ""
        assert (chunk.stdout_offset, chunk.stderr_offset) == (17, 4)

    async def test_a_slice_that_cut_a_character_in_half_does_not_raise(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        # Offsets are byte offsets, so a boundary can land mid-sequence. Cosmetic, not fatal.
        job = await self._job(backend, fake_connection)
        half = "é".encode()[:1]
        fake_connection.queue(self._reply(half + b"ok", b"", total_out=3, total_err=0))
        assert (await backend.console(job)).stdout.endswith("ok")

    async def test_a_nonpositive_cap_is_rejected(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        job = await self._job(backend, fake_connection)
        with pytest.raises(ValueError, match="max_bytes"):
            await backend.console(job, max_bytes=0)

    # -- the remote shell script, executed for real ---------------------------------

    @staticmethod
    def _run_script(
        workdir: Path, out_off: int, err_off: int, cap: int
    ) -> tuple[Any, tuple[str, str]]:
        from strata_forge.compute.backends.ssh import (
            _console_command,  # pyright: ignore[reportPrivateUsage]
            _console_frames,  # pyright: ignore[reportPrivateUsage]
        )

        proc = subprocess.run(  # noqa: S602 — the command under test IS a shell command
            _console_command(str(workdir), out_off, err_off, cap),
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        return _console_frames(proc.stdout)

    def test_the_script_reports_sizes_and_slices_for_real(self, tmp_path: Path) -> None:
        (tmp_path / "stdout.log").write_bytes(b"abcdefghij")
        (tmp_path / "stderr.log").write_bytes(b"XYZ")
        sizes, payloads = self._run_script(tmp_path, 4, 0, 1024)
        assert sizes == (10, 3, 6, 3)
        assert base64.b64decode(payloads[0]) == b"efghij"
        assert base64.b64decode(payloads[1]) == b"XYZ"

    def test_the_script_treats_missing_files_as_empty(self, tmp_path: Path) -> None:
        # A job that has not written anything yet is not an error.
        sizes, payloads = self._run_script(tmp_path, 0, 0, 1024)
        assert sizes == (0, 0, 0, 0)
        assert payloads == ("", "")

    def test_the_script_caps_to_the_newest_bytes(self, tmp_path: Path) -> None:
        (tmp_path / "stdout.log").write_bytes(b"0123456789")
        sizes, payloads = self._run_script(tmp_path, 0, 0, 4)
        assert sizes == (10, 0, 4, 0)
        assert base64.b64decode(payloads[0]) == b"6789"

    def test_the_script_treats_a_shrunken_file_as_all_unread(self, tmp_path: Path) -> None:
        """A truncation (a restart opening with `>`, logrotate) leaves the reader past the end.

        Returning nothing would strand it there while the file filled up again underneath, and
        reporting the negative pending count as zero would claim a continuous transcript.
        """
        (tmp_path / "stdout.log").write_bytes(b"abc")
        sizes, payloads = self._run_script(tmp_path, 99, 0, 1024)
        assert sizes == (3, 0, 3, 0)
        assert base64.b64decode(payloads[0]) == b"abc"

    def test_the_script_anchors_the_slice_to_the_offset_not_the_end(self, tmp_path: Path) -> None:
        """The race the header/payload agreement depends on.

        `tail -c N` counts back from the END of the file, and the job is still writing. Measuring
        at one instant and slicing at another handed back a window shifted off the one the header
        described -- losing bytes at the front and re-delivering bytes at the back, which are
        precisely the two failures byte offsets exist to eliminate.
        """
        from strata_forge.compute.backends.ssh import (
            _console_command,  # pyright: ignore[reportPrivateUsage]
            _console_frames,  # pyright: ignore[reportPrivateUsage]
        )

        target = tmp_path / "stdout.log"
        target.write_bytes(b"A" * 900 + b"B" * 100)
        # Grow the file BETWEEN the measurement and the slice, the way a live job would.
        command = _console_command(str(tmp_path), 900, 0, 1024).replace(
            "so=$((o-no+1));",
            f"so=$((o-no+1)); printf 'CCCCC' >> {target};",
            1,
        )
        proc = subprocess.run(  # noqa: S602 — the command under test IS a shell command
            command, shell=True, capture_output=True, text=True, check=False
        )
        sizes, payloads = _console_frames(proc.stdout)
        assert sizes == (1000, 0, 100, 0)
        # Exactly the 100 bytes the header promised — not the 100 ending at the new EOF.
        assert base64.b64decode(payloads[0]) == b"B" * 100

    def test_the_script_round_trips_binary_output(self, tmp_path: Path) -> None:
        """Console output is not guaranteed to be text — ANSI, NULs and CRs all appear."""
        payload = bytes(range(256))
        (tmp_path / "stdout.log").write_bytes(payload)
        _sizes, payloads = self._run_script(tmp_path, 0, 0, 4096)
        assert base64.b64decode(payloads[0]) == payload


class TestConsoleDistrustsTheRemote:
    """The reply is composed by a shell on a machine its owner controls.

    It is INPUT, not instruction. Every field is a claim, the payload length is a claim, and the
    process doing the reading is shared by every account -- so a reply that is hostile, or merely
    from a box whose userland is missing a tool, must cost this caller a poll and nothing more.
    """

    @staticmethod
    async def _job(backend: SSHBackend, fake_connection: _FakeSSHConnection) -> Any:
        fake_connection.queue(
            _FakeProcessResult(),
            _FakeProcessResult(),
            _FakeProcessResult(stdout="42\n"),
        )
        return await backend.submit(Task(name="t", run="echo"))

    async def test_the_reply_is_read_under_a_bound_the_remote_cannot_widen(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        """The remote-side cap lives in the remote's own shell; this one does not.

        A `wc`/`tail`/`base64` shim on the box's PATH makes the cap whatever its owner wants, and
        `connection.run()` would buffer the whole answer before this code saw a byte of it.
        """
        job = await self._job(backend, fake_connection)
        huge = base64.b64encode(b"x" * 5_000_000).decode()
        fake_connection.queue(
            _FakeProcessResult(stdout=f"__forge_console__ 5000000 0 5000000 0\n{huge}\n\n")
        )
        chunk = await backend.console(job, max_bytes=1024)
        assert chunk.stdout == ""  # refused, not ingested
        assert len(chunk.stdout) < 5_000_000

    async def test_a_negative_size_does_not_escape_as_an_exception(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        """`ConsoleChunk` pins its offsets at `ge=0`, so an unchecked -1 would raise a
        ValidationError out of a backend method from inside the shared polling loop."""
        job = await self._job(backend, fake_connection)
        fake_connection.queue(_FakeProcessResult(stdout="__forge_console__ -1 -1 0 0\n\n\n"))
        chunk = await backend.console(job, stdout_offset=7)
        assert chunk.stdout_offset == 7

    async def test_an_absurd_size_is_refused_rather_than_stored_as_a_cursor(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        """A 2**64 offset would be persisted and then interpolated back into remote arithmetic
        that WRAPS at 64 bits instead of failing -- turning the next read into a replay."""
        job = await self._job(backend, fake_connection)
        fake_connection.queue(
            _FakeProcessResult(stdout=f"__forge_console__ {2**64 + 1} 0 0 0\n\n\n")
        )
        assert (await backend.console(job)).stdout_offset == 0

    async def test_a_claimed_slice_larger_than_the_budget_is_refused(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        job = await self._job(backend, fake_connection)
        payload = base64.b64encode(b"y" * 4096).decode()
        fake_connection.queue(
            _FakeProcessResult(stdout=f"__forge_console__ 4096 0 4096 0\n{payload}\n\n")
        )
        chunk = await backend.console(job, max_bytes=100)
        assert chunk.stdout == ""
        assert chunk.stdout_offset == 0

    async def test_a_payload_that_did_not_arrive_intact_leaves_the_offset_alone(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        """A box with no `base64`, or a reply cut short by the command timeout, must not be read
        as "nothing was printed" -- that advances the cursor past bytes nobody ever saw."""
        job = await self._job(backend, fake_connection)
        # The header promises 10 bytes; the payload frame is empty.
        fake_connection.queue(_FakeProcessResult(stdout="__forge_console__ 4096 0 10 0\n\n\n"))
        chunk = await backend.console(job, stdout_offset=4086)
        assert chunk.stdout_offset == 4086
        assert chunk.dropped_bytes == 0

    async def test_a_corrupt_payload_is_treated_the_same(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        job = await self._job(backend, fake_connection)
        fake_connection.queue(
            _FakeProcessResult(stdout="__forge_console__ 8 0 8 0\n!!not-b64!!\n\n")
        )
        assert (await backend.console(job, stdout_offset=0)).stdout_offset == 0

    async def test_a_shrunken_file_reports_what_it_could_not_show(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        # The reader was at 5000; the file now holds 100 bytes and only 10 came back.
        job = await self._job(backend, fake_connection)
        fake_connection.queue(
            _FakeProcessResult(
                stdout=(
                    f"__forge_console__ 100 0 10 0\n{base64.b64encode(b'0123456789').decode()}\n\n"
                )
            )
        )
        chunk = await backend.console(job, stdout_offset=5000, max_bytes=20)
        assert chunk.dropped_bytes == 90
        assert chunk.stdout_offset == 100

    async def test_a_float_budget_does_not_silently_uncap_the_remote(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        """`[ "$n" -gt 65536.0 ]` is not a comparison the remote shell can make: it errors, the
        slice length keeps its uncapped value, and the cap quietly ceases to exist. A value that
        came through JSON is a float."""
        job = await self._job(backend, fake_connection)
        fake_connection.queue(self._null_reply())
        await backend.console(job, max_bytes=1024.0)  # pyright: ignore[reportArgumentType]
        assert "-gt 512 " in fake_connection.commands[-1] + " "

    async def test_a_budget_beyond_the_module_ceiling_is_clamped(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        job = await self._job(backend, fake_connection)
        fake_connection.queue(self._null_reply())
        await backend.console(job, max_bytes=1 << 30)
        assert f"-gt {MAX_CONSOLE_CHUNK_BYTES // 2} " in fake_connection.commands[-1] + " "

    @staticmethod
    def _null_reply() -> _FakeProcessResult:
        return _FakeProcessResult(stdout="__forge_console__ 0 0 0 0\n\n\n")


class TestWorkdirConfinement:
    async def test_a_workdir_outside_the_remote_root_is_refused(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        """Confined once, for every method that composes a remote path from it.

        Quoting stops a workdir from breaking OUT of the command; it does nothing to stop one
        from pointing somewhere else. `cleanup` guarded its own `rm -rf`; `logs`, `read_file` and
        `console` inherited nothing, so a record saying "../../../../etc" read /etc.
        """
        job = Job(
            id="j",
            backend="ssh",
            task_name="t",
            metadata={"remote_workdir": "/etc", "pid": "1"},
        )
        for call in (backend.console(job), backend.logs(job), backend.cleanup(job)):
            with pytest.raises(ValueError, match="remote_root"):
                await call

    async def test_a_traversal_inside_the_root_is_refused(
        self, backend: SSHBackend, fake_connection: _FakeSSHConnection
    ) -> None:
        # `startswith(root)` alone accepts this: the prefix matches and the `..` then leaves.
        job = Job(
            id="j",
            backend="ssh",
            task_name="t",
            metadata={"remote_workdir": ".forge-compute/../../../etc", "pid": "1"},
        )
        with pytest.raises(ValueError, match="remote_root"):
            await backend.console(job)
