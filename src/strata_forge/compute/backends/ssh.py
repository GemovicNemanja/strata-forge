"""SSH-backed compute backend.

Submits :class:`Task` instances to a remote host via ``asyncssh``.
The remote side runs the task as a background process under
``nohup``, in a process GROUP of its own, and Forge polls that group
with ``kill -0``. The group is the unit because a job is not one
process: the recorded pid is a bookkeeping shell, and the work below
it is what holds the hardware — so ``cancel`` signals the group, not
the pid. Stdout / stderr land in files inside a per-job remote
working directory.

The ``asyncssh`` SDK is imported lazily inside the constructor
(behind the ``[compute]`` extra) so ``import strata_forge.compute`` works
without it. The :class:`ImportError` surfaces only when a caller
actually constructs the backend.

This backend ignores :attr:`Task.resources` — the remote host's
hardware is fixed; if you need to provision GPUs on demand, use
the SkyPilot backend.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import shlex
import uuid
from datetime import UTC, datetime
from typing import Any

from strata_forge.compute.backends.base import (
    MAX_CONSOLE_CHUNK_BYTES,
    MAX_READ_FILE_BYTES,
    safe_workdir_relpath,
)
from strata_forge.compute.job import ConsoleChunk, Job, JobStatus
from strata_forge.compute.task import Task  # noqa: TC001 — runtime use in submit

__all__ = ["SSHBackend"]


_PID_FILE = "forge.pid"
_EXIT_FILE = "forge.exit"
# Written by `cancel` BEFORE it signals, so `status` can tell a cancellation from a process the
# machine killed. Both look identical afterwards — pid gone, no exit file — and guessing
# "cancelled" for every such job reported an OOM kill, a reboot, or a segfault as something the
# user asked for. The lie is not free downstream: an orchestrator that captures diagnostics only
# for FAILED jobs discards the logs of exactly the deaths nobody chose.
_CANCELLED_FILE = "forge.cancelled"
# Prefixes the console read's size header for the same reason `_REPORT_MARKER` prefixes the
# launch report: a login shell may print a banner or an rc file's chatter first.
_CONSOLE_MARKER = "__forge_console__"
# Room for the header line, base64 padding and the two trailing newlines when sizing how much
# of the reply is worth reading at all.
_CONSOLE_FRAME_SLACK = 256
# Beyond this a reported stream size is not a measurement, it is a fabrication or a bug. Well
# under 2**53 so the value stays exact through JSON and inside the remote shell's 64-bit
# arithmetic, which wraps silently rather than failing.
_MAX_PLAUSIBLE_BYTES = 1 << 48
_STDOUT_FILE = "stdout.log"
_STDERR_FILE = "stderr.log"
_FORGE_REMOTE_ROOT = ".forge-compute"
# Prefixes the launcher's structured report so it can be picked out of whatever the login shell
# printed first — a banner, an rc file's chatter, a motd. Parsing the last line was fragile for
# the same reason.
_REPORT_MARKER = "__forge_launch__"
# How long a cancelled job's group may take to honour SIGTERM before SIGKILL. The runner spends
# it shutting the model server down; the server runs in its own session, so a SIGKILL that
# preempted that teardown would leave the GPU held by the very process cancel exists to stop.
_CANCEL_GRACE_S = 10


def _report_fields(stdout: str) -> dict[str, str]:
    """Parse the launcher's ``key=value`` report out of the remote's chatter.

    A login shell may print a banner or motd before anything of ours runs, so the report is
    found by its marker rather than by position.
    """
    for line in reversed(stdout.splitlines()):
        if not line.startswith(_REPORT_MARKER):
            continue
        fields: dict[str, str] = {}
        for token in line[len(_REPORT_MARKER) :].split():
            key, _, value = token.partition("=")
            fields[key] = value
        return fields
    return {}


# Bound every SSH op so an unresponsive / black-holed host can't stall a caller (e.g. a polling
# loop) indefinitely. Connect/login cap the handshake; the command timeout caps each remote command
# — read_file's bounded ``head -c`` (<= MAX_READ_FILE_BYTES) is the longest and sits well under it.
_CONNECT_TIMEOUT_S = 30.0
_LOGIN_TIMEOUT_S = 30.0
_COMMAND_TIMEOUT_S = 120.0


class SSHBackend:
    """Submit Forge tasks to an SSH-accessible host.

    Args:
        host: Hostname or IP of the remote machine.
        username: SSH login user.
        port: SSH port. Default 22.
        client_keys: Optional list of private-key paths. ``asyncssh``
            picks them up in order; defaults to the user's
            ``~/.ssh/id_*`` files.
        known_hosts: Path to a ``known_hosts`` file. Pass
            ``None`` to disable host-key checking (only safe in
            tightly controlled networks).
        passphrase: Optional passphrase for encrypted keys.
        remote_root: Directory under ``~`` where Forge creates
            per-job workdirs. Default ``".forge-compute"``.
        name: Backend identifier reported via :attr:`name`. Default
            ``"ssh"``.
        connection: Optional pre-built ``asyncssh.SSHClientConnection``
            (handy for tests and for callers who want to share a
            connection across multiple submits). When set, the other
            connection-related args are ignored.
    """

    def __init__(
        self,
        *,
        host: str = "",
        username: str = "",
        port: int = 22,
        client_keys: list[str] | None = None,
        known_hosts: str | None = None,
        passphrase: str | None = None,
        remote_root: str = _FORGE_REMOTE_ROOT,
        name: str = "ssh",
        connection: Any | None = None,
    ) -> None:
        if not name:
            err = "SSHBackend: name must be non-empty"
            raise ValueError(err)
        if connection is None and (not host or not username):
            err = "SSHBackend: pass either an explicit `connection` or both `host` and `username`."
            raise ValueError(err)
        self._host = host
        self._username = username
        self._port = port
        self._client_keys = list(client_keys or [])
        self._known_hosts = known_hosts
        self._passphrase = passphrase
        self._remote_root = remote_root
        self._name = name
        self._explicit_connection = connection
        self._connection: Any | None = None

    @property
    def name(self) -> str:
        return self._name

    async def _get_connection(self) -> Any:
        if self._explicit_connection is not None:
            return self._explicit_connection
        if self._connection is not None:
            return self._connection
        try:
            asyncssh_mod: Any = __import__("asyncssh")
        except ImportError as exc:
            msg = (
                "The [compute] extra is required for SSHBackend. "
                "Install it with: pip install 'strata-forge[compute]'."
            )
            raise ImportError(msg) from exc
        kwargs: dict[str, Any] = {
            "host": self._host,
            "username": self._username,
            "port": self._port,
            "known_hosts": self._known_hosts,
            "connect_timeout": _CONNECT_TIMEOUT_S,
            "login_timeout": _LOGIN_TIMEOUT_S,
        }
        if self._client_keys:
            kwargs["client_keys"] = self._client_keys
        if self._passphrase is not None:
            kwargs["passphrase"] = self._passphrase
        self._connection = await asyncssh_mod.connect(**kwargs)
        return self._connection

    async def _run_remote_bounded(self, command: str, limit: int) -> str:
        """Run ``command`` and read AT MOST ``limit`` characters of its stdout.

        ``connection.run()`` collects the entire reply into memory before returning it, and the
        reply is composed by a shell on a machine the user controls -- so a cap expressed only in
        that shell is a cap the remote side is free to ignore. This is the control plane's own
        bound, and it matters because this command is polled for the whole life of every run from
        one process shared by every account.
        """
        connection = await self._get_connection()
        async with asyncio.timeout(_COMMAND_TIMEOUT_S):
            process = await connection.create_process(command)
            try:
                return str(await process.stdout.read(limit) or "")
            finally:
                process.close()
                with contextlib.suppress(Exception):
                    await process.wait_closed()

    async def _run_remote(self, command: str, *, check: bool = False) -> tuple[int, str, str]:
        """Run ``command`` on the remote host; return (exit, stdout, stderr)."""
        connection = await self._get_connection()
        result = await connection.run(command, check=check, timeout=_COMMAND_TIMEOUT_S)
        return (
            int(result.exit_status or 0),
            str(result.stdout or ""),
            str(result.stderr or ""),
        )

    async def submit(self, task: Task) -> Job:
        if task.num_nodes != 1:
            err = f"SSHBackend can only run single-node tasks; got num_nodes={task.num_nodes}"
            raise ValueError(err)

        job_id = uuid.uuid4().hex
        remote_workdir = f"{self._remote_root}/{job_id}"

        # Create the workdir.
        await self._run_remote(f"mkdir -p {shlex.quote(remote_workdir)}", check=True)

        # Build the env preamble and the wrapper script.
        env_lines = [f"export {k}={shlex.quote(v)}" for k, v in task.env.items()]
        body_lines: list[str] = []
        if task.setup:
            body_lines.append(task.setup)
        body_lines.append(task.run)
        body = " && ".join(body_lines) if len(body_lines) > 1 else body_lines[0]

        wrapper_script = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -e",
                *env_lines,
                body,
            ]
        )
        # Write the wrapper.
        await self._run_remote(
            f"cat > {shlex.quote(remote_workdir)}/wrapper.sh <<'__FORGE_EOF__'\n"
            f"{wrapper_script}\n__FORGE_EOF__\n"
            f"chmod +x {shlex.quote(remote_workdir)}/wrapper.sh",
            check=True,
        )

        # Launch detached, capture the PID, write the exit code on exit.
        #
        # sshd keeps the session channel open until it sees EOF on the command's stdout AND
        # stderr, not when the command exits — and every process in a backgrounded tree inherits
        # those descriptors. asyncssh's run() in turn resolves on channel close rather than on
        # exit-status, so a launcher that leaves them open makes submit() block for the entire
        # lifetime of the job it just launched. Redirecting the backgrounded subshell's OWN
        # descriptors is what releases the channel; the inner per-command redirections still win,
        # so the job's stdout/stderr keep landing in their log files.
        #
        # The brace group is load-bearing, not cosmetic: `&` binds looser than `&&`, so without it
        # the shell backgrounds the whole `cd … && ( … )` and-list and forks an OUTER subshell that
        # inherits the channel anyway — the redirection alone does not fix the hang. It also keeps
        # `cd` in the foreground shell, so the pid file is written inside the workdir instead of
        # $HOME, where concurrent submits would otherwise overwrite each other's handle.
        #
        # `trap '' HUP` covers the bookkeeping subshell at session teardown, which nohup does not:
        # nohup protects only the process it execs, so without the trap a hangup between launch and
        # completion loses the exit code and leaves a live job indistinguishable from a cancelled
        # one. `< /dev/null` detaches stdin so a long bootstrap that reads it sees EOF rather than
        # EIO once the channel is gone.
        # `set -m` (job control) is what gives the job a process group of its OWN, whose id equals
        # the pid recorded here — so cancel can stop the entire tree with a single kill. Without
        # it the subshell just joins the sshd session's group and the recorded pid names no group
        # at all, so `kill -- -PID` fails with ESRCH and stops nothing. That is the whole bug:
        # the recorded pid is a bookkeeper, and the work — the wrapper, the runner, the inference
        # engine holding the GPU — lives below it.
        #
        # It runs under an explicit `bash -c` because sshd hands the command to the user's LOGIN
        # shell, and job control does not survive that lottery: dash and sh accept `set -m` but
        # still leave background jobs in the session's group, and zsh rejects the option outright
        # and exits without launching anything. bash is already required here (the job runs as
        # `bash wrapper.sh`), so naming it adds no new dependency.
        launch_script = (
            "set -m; "
            f"cd {shlex.quote(remote_workdir)} && {{ "
            f'( trap "" HUP; nohup bash wrapper.sh > {_STDOUT_FILE} 2> {_STDERR_FILE}; '
            f"echo $? > {_EXIT_FILE} ) < /dev/null > /dev/null 2>&1 & "
            f"p=$!; echo $p > {_PID_FILE}; "
            # Job control is REPORTED, not assumed. `$-` carries `m` only if the shell really
            # enabled it, and `ps` corroborates that the job actually landed in a group of its
            # own. A shell built without job control therefore marks the job legacy and keeps the
            # single-pid probe instead of group-signalling into nothing. `ps` may find the job
            # already gone (a task that outran the probe), which says nothing either way, so an
            # empty reading defers to `$-`.
            "case $- in *m*) m=1;; *) m=0;; esac; "
            'g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -dc 0-9); '
            f'printf "{_REPORT_MARKER} pid=%s m=%s g=%s\\n" "$p" "$m" "$g"; }}'
        )
        _exit, stdout, _stderr = await self._run_remote(
            f"bash -c {shlex.quote(launch_script)}", check=True
        )
        fields = _report_fields(stdout)
        pid = fields.get("pid", "")
        if not pid.isdigit():
            err = f"SSHBackend: unexpected pid output from remote: {stdout!r}"
            raise RuntimeError(err)
        measured_pgid = fields.get("g", "")
        own_group = fields.get("m") == "1" and measured_pgid in ("", pid)

        return Job(
            id=job_id,
            backend=self._name,
            task_name=task.name,
            metadata={
                "pid": pid,
                # Whether `pid` also names the job's process GROUP. Jobs submitted before this
                # existed carry no flag, and must keep the single-pid probe: their pid is not a
                # pgid, so a group probe would report a live job as gone and a group signal would
                # stop nothing. Drop the flag and the legacy branches once such jobs have drained.
                "pgroup": own_group,
                "remote_workdir": remote_workdir,
                "host": self._host,
                "submitted_at": datetime.now(UTC).isoformat(),
            },
        )

    def _job_workdir(self, job: Job) -> str:
        if job.backend != self._name:
            err = f"SSHBackend: job belongs to backend {job.backend!r}, not {self._name!r}"
            raise ValueError(err)
        workdir = job.metadata.get("remote_workdir")
        if not isinstance(workdir, str) or not workdir:
            err = f"SSHBackend: job {job.id!r} missing remote_workdir metadata"
            raise ValueError(err)
        # Confined here rather than in each caller, so every method that composes a remote path
        # from it inherits the guard `cleanup` already asks for. A record whose metadata said
        # "../../../../etc" would otherwise have `logs`/`console`/`read_file` reading /etc.
        if ".." in workdir.split("/") or not workdir.startswith(self._remote_root + "/"):
            err = (
                f"SSHBackend: job {job.id!r} workdir {workdir!r} is not under "
                f"remote_root {self._remote_root!r}"
            )
            raise ValueError(err)
        return workdir

    def _job_pid(self, job: Job) -> str:
        pid = job.metadata.get("pid")
        if not isinstance(pid, str) or not pid:
            err = f"SSHBackend: job {job.id!r} missing pid metadata"
            raise ValueError(err)
        return pid

    def _signal_target(self, job: Job) -> str:
        """What `kill` should address: the job's whole process group, or just its pid.

        A leading `-` makes `kill` treat the number as a process GROUP, which is what reaches the
        wrapper, the runner and the engine below it. Only jobs whose launcher CONFIRMED it got
        its own group are addressed that way — see the `pgroup` marker in submit.
        """
        pid = self._job_pid(job)
        return f"-{pid}" if job.metadata.get("pgroup") is True else pid

    async def status(self, job: Job) -> JobStatus:
        workdir = self._job_workdir(job)
        # Liveness is asked of the GROUP where there is one. The bookkeeping subshell can die
        # while the runner it launched keeps holding the GPU, and a pid-only probe calls that
        # job finished — so a partially-killed job would read terminal while its work continues.
        pid = self._signal_target(job)
        # Probe liveness, then check for the exit-code file the wrapper
        # writes when it terminates.
        # `cat` of an EMPTY exit file succeeds with no output, so MISSING must not be reached by
        # `||` alone — the marker is echoed only when the file is absent. Without that, an empty
        # file produced empty output that matched neither branch below and crashed the parse.
        probe_cmd = (
            f"kill -0 {shlex.quote(pid)} 2>/dev/null && echo RUNNING || "
            f"{{ test -f {shlex.quote(workdir)}/{_CANCELLED_FILE} && echo CANCELLED; }} || "
            f"{{ test -f {shlex.quote(workdir)}/{_EXIT_FILE} "
            f"&& cat {shlex.quote(workdir)}/{_EXIT_FILE}; }} || echo MISSING"
        )
        _exit, stdout, _stderr = await self._run_remote(probe_cmd)
        text = stdout.strip()
        submitted_at_raw = job.metadata.get("submitted_at")
        started_at: datetime | None = None
        if isinstance(submitted_at_raw, str):
            try:
                started_at = datetime.fromisoformat(submitted_at_raw)
            except ValueError:
                started_at = None

        if text == "RUNNING":
            return JobStatus(state="running", started_at=started_at)
        if text == "CANCELLED":
            # `cancel` left its marker, so this death was ASKED FOR. Nothing else may claim that.
            return JobStatus(
                state="cancelled",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                message="cancelled",
            )
        if text == "MISSING":
            # Process gone, no exit file, and nobody cancelled it: the machine took it — an OOM
            # kill, a reboot, a segfault in the interpreter itself. That is a FAILURE, and calling
            # it "cancelled" both misreported it and (because diagnostics are captured only for
            # failures) threw away the logs of the one kind of death nobody can otherwise explain.
            return JobStatus(
                state="failed",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                message="process died without recording an exit code (killed, or the host went away)",
            )
        # text should be a numeric exit code. An EMPTY read is the write-in-progress race, not a
        # verdict: the wrapper creates the file and writes to it as two steps, so a poll landing
        # between them sees an empty file. Reporting non-terminal keeps the job alive for one more
        # poll rather than inventing an outcome — and rather than crashing, which is what
        # `splitlines()[-1]` did on empty input (IndexError, which `except ValueError` never caught).
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return JobStatus(
                state="running",
                started_at=started_at,
                message="exit file not written yet",
            )
        try:
            exit_code = int(lines[-1].strip())
        except ValueError:
            return JobStatus(
                state="failed",
                started_at=started_at,
                message=f"unparseable exit-file content: {text!r}",
            )
        return JobStatus(
            state="succeeded" if exit_code == 0 else "failed",
            exit_code=exit_code,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            message="" if exit_code == 0 else f"non-zero exit ({exit_code})",
        )

    async def logs(self, job: Job, *, tail: int | None = None) -> str:
        workdir = self._job_workdir(job)
        if tail is not None and tail <= 0:
            err = f"tail must be >= 1 when set; got {tail}"
            raise ValueError(err)
        if tail is None:
            cmd = (
                f"cat {shlex.quote(workdir)}/{_STDOUT_FILE} 2>/dev/null; "
                f"cat {shlex.quote(workdir)}/{_STDERR_FILE} 2>/dev/null"
            )
        else:
            cmd = (
                f"tail -n {tail} {shlex.quote(workdir)}/{_STDOUT_FILE} "
                f"2>/dev/null; "
                f"tail -n {tail} {shlex.quote(workdir)}/{_STDERR_FILE} "
                f"2>/dev/null"
            )
        _exit, stdout, _stderr = await self._run_remote(cmd)
        return stdout

    async def read_file(self, job: Job, path: str, *, tail: int | None = None) -> str:
        workdir = self._job_workdir(job)
        rel = safe_workdir_relpath(path)
        if tail is not None:
            tail = int(tail)
            if tail <= 0:
                err = f"tail must be >= 1 when set; got {tail}"
                raise ValueError(err)
        # `rel` is validated workdir-relative and the full target is shlex-quoted, so the
        # path can't break the command. `head -c` byte-caps the result so a huge file
        # can't exhaust the polling process. Missing file -> errors swallowed -> "".
        target = shlex.quote(f"{workdir}/{rel}")
        if tail is None:
            cmd = f"head -c {MAX_READ_FILE_BYTES} {target} 2>/dev/null"
        else:
            cmd = f"tail -n {tail} {target} 2>/dev/null | head -c {MAX_READ_FILE_BYTES}"
        _exit, stdout, _stderr = await self._run_remote(cmd)
        return stdout

    async def console(
        self,
        job: Job,
        *,
        stdout_offset: int = 0,
        stderr_offset: int = 0,
        max_bytes: int = MAX_CONSOLE_CHUNK_BYTES,
    ) -> ConsoleChunk:
        workdir = self._job_workdir(job)
        budget = _console_budget(max_bytes)
        out_off, err_off = max(0, int(stdout_offset)), max(0, int(stderr_offset))
        # Every early return hands back the offsets UNCHANGED. Advancing past bytes that did not
        # arrive intact would lose them permanently; repeating a poll costs one round trip.
        unread = ConsoleChunk(stdout_offset=out_off, stderr_offset=err_off)

        raw = await self._run_remote_bounded(
            _console_command(workdir, out_off, err_off, budget), _reply_limit(budget)
        )
        sizes, payloads = _console_frames(raw)
        if sizes is None or not _plausible(sizes, budget):
            return unread
        total_out, total_err, take_out, take_err = sizes
        out, err = _b64_text(payloads[0], take_out), _b64_text(payloads[1], take_err)
        if out is None or err is None:
            return unread

        dropped = (_unread_bytes(total_out, out_off) - take_out) + (
            _unread_bytes(total_err, err_off) - take_err
        )
        return ConsoleChunk(
            stdout=out,
            stderr=err,
            # The offsets advance to each file's CURRENT size, not to what was read: the skipped
            # middle is reported as dropped and must not be re-offered, or a busy job would keep
            # the reader permanently behind, forever re-dropping the same bytes.
            stdout_offset=total_out,
            stderr_offset=total_err,
            dropped_bytes=max(0, dropped),
        )

    async def cancel(self, job: Job) -> None:
        target = self._signal_target(job)
        workdir = self._job_workdir(job)
        # The marker is written BEFORE the signal, and deliberately not after: between the two the
        # process is already dying, and a `status` landing in that window would otherwise read the
        # death as one nobody asked for. Signalling the group also takes the bookkeeping subshell
        # with it, so no exit code is ever written on this path — the marker is the ONLY evidence
        # that this death was asked for.
        #
        # SIGTERM first, then SIGKILL, and the grace period is not merely politeness: the runner
        # uses it to tear down the model server it started, which lives in a session of its own
        # and is therefore not in this group. Killing outright would strand exactly the process
        # that holds the GPU.
        quoted = shlex.quote(target)
        cmd = (
            f"touch {shlex.quote(workdir)}/{_CANCELLED_FILE} 2>/dev/null; "
            f"kill -TERM -- {quoted} 2>/dev/null; "
            # Poll rather than sleeping the full grace: a job that goes down promptly should not
            # hold the caller open, and one that needs the time still gets it.
            f"for _ in $(seq 1 {_CANCEL_GRACE_S * 2}); do "
            f"kill -0 -- {quoted} 2>/dev/null || break; sleep 0.5; done; "
            f"kill -KILL -- {quoted} 2>/dev/null; "
            "true"
        )
        await self._run_remote(cmd)

    async def cleanup(self, job: Job) -> None:
        # `_job_workdir` confines to the remote_root, which is what makes this `rm -rf` safe.
        workdir = self._job_workdir(job)
        await self._run_remote(f"rm -rf {shlex.quote(workdir)}")

    async def close(self) -> None:
        """Close the underlying SSH connection if we own it."""
        if self._connection is not None and self._explicit_connection is None:
            self._connection.close()
            await self._connection.wait_closed()
            self._connection = None


def _console_budget(max_bytes: int) -> int:
    """The per-STREAM slice budget. ``max_bytes`` is the total for the call, split evenly.

    Split rather than applied to each stream separately, so the ceiling a caller sizes a frame or
    a database column against is the one it actually gets. Evenly rather than first-come, because
    stderr is where a failure announces itself and a chatty stdout must not be able to starve it.

    Coerced and clamped rather than trusted: a value that arrived through JSON is a float, and
    ``[ "$n" -gt 65536.0 ]`` is not a comparison the remote shell can make -- it errors, leaves
    the slice length uncapped, and the cap silently ceases to exist.
    """
    try:
        cap = int(max_bytes)
    except TypeError, ValueError:
        cap = 0
    if cap < 2:
        err = f"max_bytes must be >= 2; got {max_bytes!r}"
        raise ValueError(err)
    return min(cap, MAX_CONSOLE_CHUNK_BYTES) // 2


def _reply_limit(budget: int) -> int:
    """How much of the remote's reply is worth reading: two base64 frames plus the header."""
    return 2 * (4 * ((budget + 2) // 3) + _CONSOLE_FRAME_SLACK) + _CONSOLE_FRAME_SLACK


def _unread_bytes(total: int, offset: int) -> int:
    """How much of a stream the caller has not seen.

    ``total < offset`` means the file SHRANK -- truncated by a restart that opened it with `>`,
    by logrotate, or by the user. Everything now in it is unread, and reporting the subtraction's
    negative result as zero would claim a continuous transcript across a hole.
    """
    return total - offset if total >= offset else total


def _plausible(sizes: tuple[int, int, int, int], budget: int) -> bool:
    """Is this header one a working remote could have produced?

    The reply is composed on the user's own machine, so it is input rather than instruction. A
    negative size reaches ``ConsoleChunk``'s ``ge=0`` and raises a ``ValidationError`` out of a
    backend method no caller is catching; a huge one is persisted as a resume cursor and then
    interpolated back into remote arithmetic that WRAPS at 64 bits rather than failing, which
    turns the next read into a replay.
    """
    if any(value < 0 or value > _MAX_PLAUSIBLE_BYTES for value in sizes):
        return False
    _total_out, _total_err, take_out, take_err = sizes
    return take_out <= budget and take_err <= budget


def _console_command(workdir: str, out_off: int, err_off: int, cap: int) -> str:
    """Compose the one-round-trip incremental console read.

    Sizes are measured and the slice lengths computed REMOTELY, in the same shell, so the two
    always describe the same instant. Measuring here and slicing there would race a job that is
    still writing, and the frame lengths would no longer match the payloads.

    The payloads are base64 so that byte counts survive the transport: the SSH channel hands back
    decoded text, and a raw slice that cut a multi-byte character mid-sequence would arrive with a
    length no longer equal to the number of bytes it represents — which is the one invariant the
    offsets depend on.
    """
    out = shlex.quote(f"{workdir}/{_STDOUT_FILE}")
    err = shlex.quote(f"{workdir}/{_STDERR_FILE}")
    return (
        # `tr -dc 0-9` because `wc -c` pads its output with spaces on BSD userlands; the default
        # covers a file the job has not created yet, which is not an error.
        f"o=$(wc -c < {out} 2>/dev/null | tr -dc 0-9); o=${{o:-0}}; "
        f"e=$(wc -c < {err} 2>/dev/null | tr -dc 0-9); e=${{e:-0}}; "
        # A negative pending count means the file SHRANK below where the reader was -- a
        # truncation or a rotation. Everything in it now is unread, so read from the start.
        f'no=$((o-{out_off})); [ "$no" -lt 0 ] && no=$o; [ "$no" -gt {cap} ] && no={cap}; '
        f'ne=$((e-{err_off})); [ "$ne" -lt 0 ] && ne=$e; [ "$ne" -gt {cap} ] && ne={cap}; '
        # Anchor each slice to its START. `tail -c N` counts back from the END OF THE FILE, and
        # the file is still being written -- so a job that appended between `wc` and `tail` would
        # hand back a window shifted off the one the header describes, losing bytes at the front
        # and re-delivering bytes at the back. Both are exactly what the offsets exist to prevent.
        # When the slice was capped this start is still the newest `cap` bytes.
        f"so=$((o-no+1)); se=$((e-ne+1)); "
        f'printf "{_CONSOLE_MARKER} %s %s %s %s\\n" "$o" "$e" "$no" "$ne"; '
        f'if [ "$no" -gt 0 ]; then tail -c +"$so" {out} 2>/dev/null | head -c "$no" '
        f'| base64 | tr -d "\\n"; fi; echo; '
        f'if [ "$ne" -gt 0 ]; then tail -c +"$se" {err} 2>/dev/null | head -c "$ne" '
        f'| base64 | tr -d "\\n"; fi; echo'
    )


def _console_frames(raw: str) -> tuple[tuple[int, int, int, int] | None, tuple[str, str]]:
    """Split the reply into (sizes, payloads), or (None, ...) if the marker never arrived.

    Located by marker for the same reason `submit` does it: a login shell may print a banner, a
    motd or an rc file's chatter before anything this command wrote.
    """
    lines = raw.splitlines()
    index = next(
        (i for i in range(len(lines) - 1, -1, -1) if lines[i].startswith(_CONSOLE_MARKER)), None
    )
    if index is None:
        return None, ("", "")
    fields = lines[index].split()
    if len(fields) != 5:
        return None, ("", "")
    try:
        sizes = (int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4]))
    except ValueError:
        return None, ("", "")
    tail = lines[index + 1 :]
    return sizes, (tail[0] if tail else "", tail[1] if len(tail) > 1 else "")


def _b64_text(payload: str, expected: int) -> str | None:
    """Decode one base64 frame, or None if it is not the ``expected`` number of bytes.

    None is not the same as empty: it means the payload did not arrive intact -- a remote with no
    `base64` binary, a reply cut short by the command timeout, a stray CR the alphabet forbids --
    and the caller must NOT advance its offset past bytes it never actually read.
    """
    if expected == 0:
        return ""
    try:
        data = base64.b64decode(payload, validate=True)
    except ValueError:  # binascii.Error subclasses it
        return None
    if len(data) != expected:
        return None
    # A byte slice can cut a multi-byte character in half; that boundary is cosmetic, not fatal.
    return data.decode("utf-8", errors="replace")
