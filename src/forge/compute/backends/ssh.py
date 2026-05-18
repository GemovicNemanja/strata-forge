"""SSH-backed compute backend.

Submits :class:`Task` instances to a remote host via ``asyncssh``.
The remote side runs the task as a background process under
``nohup``; Forge captures the PID and uses ``kill -0`` to poll
liveness. Stdout / stderr land in files inside a per-job remote
working directory.

The ``asyncssh`` SDK is imported lazily inside the constructor
(behind the ``[compute]`` extra) so ``import forge.compute`` works
without it. The :class:`ImportError` surfaces only when a caller
actually constructs the backend.

This backend ignores :attr:`Task.resources` — the remote host's
hardware is fixed; if you need to provision GPUs on demand, use
the SkyPilot backend.
"""

from __future__ import annotations

import shlex
import uuid
from datetime import UTC, datetime
from typing import Any

from forge.compute.job import Job, JobStatus
from forge.compute.task import Task  # noqa: TC001 — runtime use in submit

__all__ = ["SSHBackend"]


_PID_FILE = "forge.pid"
_EXIT_FILE = "forge.exit"
_STDOUT_FILE = "stdout.log"
_STDERR_FILE = "stderr.log"
_FORGE_REMOTE_ROOT = ".forge-compute"


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
                "Install it with: pip install 'ai-forge[compute]'."
            )
            raise ImportError(msg) from exc
        kwargs: dict[str, Any] = {
            "host": self._host,
            "username": self._username,
            "port": self._port,
            "known_hosts": self._known_hosts,
        }
        if self._client_keys:
            kwargs["client_keys"] = self._client_keys
        if self._passphrase is not None:
            kwargs["passphrase"] = self._passphrase
        self._connection = await asyncssh_mod.connect(**kwargs)
        return self._connection

    async def _run_remote(self, command: str, *, check: bool = False) -> tuple[int, str, str]:
        """Run ``command`` on the remote host; return (exit, stdout, stderr)."""
        connection = await self._get_connection()
        result = await connection.run(command, check=check)
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

        # Launch under nohup, capture the PID, write the exit code on exit.
        launch_cmd = (
            f"cd {shlex.quote(remote_workdir)} && "
            f"(nohup bash wrapper.sh > {_STDOUT_FILE} 2> {_STDERR_FILE}; "
            f"echo $? > {_EXIT_FILE}) & "
            f"echo $! > {_PID_FILE}; cat {_PID_FILE}"
        )
        _exit, stdout, _stderr = await self._run_remote(launch_cmd, check=True)
        pid = stdout.strip().splitlines()[-1]
        if not pid.isdigit():
            err = f"SSHBackend: unexpected pid output from remote: {stdout!r}"
            raise RuntimeError(err)

        return Job(
            id=job_id,
            backend=self._name,
            task_name=task.name,
            metadata={
                "pid": pid,
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
        return workdir

    def _job_pid(self, job: Job) -> str:
        pid = job.metadata.get("pid")
        if not isinstance(pid, str) or not pid:
            err = f"SSHBackend: job {job.id!r} missing pid metadata"
            raise ValueError(err)
        return pid

    async def status(self, job: Job) -> JobStatus:
        workdir = self._job_workdir(job)
        pid = self._job_pid(job)
        # Probe liveness, then check for the exit-code file the wrapper
        # writes when it terminates.
        probe_cmd = (
            f"kill -0 {shlex.quote(pid)} 2>/dev/null && echo RUNNING || "
            f"cat {shlex.quote(workdir)}/{_EXIT_FILE} 2>/dev/null || echo MISSING"
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
        if text == "MISSING":
            # Process not alive and no exit file — usually means cancelled
            # before the wrapper got a chance to write the exit code.
            return JobStatus(
                state="cancelled",
                started_at=started_at,
                message="process gone, no exit file",
            )
        # text should be a numeric exit code.
        try:
            exit_code = int(text.splitlines()[-1].strip())
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

    async def cancel(self, job: Job) -> None:
        pid = self._job_pid(job)
        # Best-effort SIGTERM, brief wait, SIGKILL.
        cmd = (
            f"kill -TERM {shlex.quote(pid)} 2>/dev/null; "
            "sleep 2; "
            f"kill -KILL {shlex.quote(pid)} 2>/dev/null; "
            "true"
        )
        await self._run_remote(cmd)

    async def cleanup(self, job: Job) -> None:
        workdir = self._job_workdir(job)
        # Refuse to wipe anything that doesn't live under the remote_root.
        if not workdir.startswith(self._remote_root + "/"):
            err = (
                f"SSHBackend.cleanup: refusing to remove {workdir!r} — "
                f"not under remote_root {self._remote_root!r}"
            )
            raise ValueError(err)
        await self._run_remote(f"rm -rf {shlex.quote(workdir)}")

    async def close(self) -> None:
        """Close the underlying SSH connection if we own it."""
        if self._connection is not None and self._explicit_connection is None:
            self._connection.close()
            await self._connection.wait_closed()
            self._connection = None
