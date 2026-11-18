"""How the plugin's process is kept apart from the runner that holds the run's credential.

The runner holds the Hub token in memory for the whole run (it downloads the inputs and pushes the
output). The plugin's process must not be able to reach it, nor anything else of the runner's:
its files, its ``/proc`` entry, its memory. The strongest sandbox the machine offers is used,
in this order:

- ``uid``: bubblewrap started through passwordless ``sudo``, the plugin dropped to the
  unprivileged ``nobody`` uid inside it (``setpriv``, no new privileges, no supplementary
  groups). Another uid cannot read the runner's files or ptrace it, and the sandbox adds the
  namespaces below.
- ``bwrap``: the same bubblewrap sandbox as the runner's own uid. New PID, network, IPC, UTS and
  cgroup namespaces: the runner's processes are not visible (no ``/proc`` entry to read, nothing
  to ptrace), there is no network at all, and the filesystem is only what is bound in: the system
  directories read-only, the plugin's environment and its interpreter read-only, the job file and
  the inputs read-only, one writable output directory, a private ``/tmp``. No home directory is
  bound (``HOME`` is the private ``/tmp``), so no SSH key, cached login or token file is there to
  read. The sandbox dies with the runner (``--die-with-parent``).
- ``none``: a plain child process in its own session, as the runner's uid, with network. Only
  when the operator's allow-list entry sets ``allow_unisolated``, never when the kernel lets a
  process ptrace any other of its uid (``kernel.yama.ptrace_scope`` 0, or no Yama on Linux), and
  never when the runner's environment holds a credential-shaped name (``/proc/<runner>/environ``
  is readable to the same uid). It is recorded in the run's manifest.

Which one ran, and the ptrace scope, are recorded with the run.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from strata_forge.plugins.pins import Isolation

__all__ = [
    "NOBODY_UID",
    "Sandbox",
    "SandboxPaths",
    "SandboxUnavailableError",
    "chmod_for_uid",
    "choose_sandbox",
    "kill_command",
    "ptrace_scope",
    "sandbox_command",
]

#: The unprivileged uid and gid the ``uid`` sandbox runs the plugin as.
NOBODY_UID = 65534
_YAMA = Path("/proc/sys/kernel/yama/ptrace_scope")
# The sandbox's own private, empty /tmp (a tmpfs inside the mount namespace), never the host's.
_SANDBOX_TMP = "/tmp"  # noqa: S108
# System directories bound read-only into the sandbox when they exist (``--ro-bind-try``).
_SYSTEM_DIRS = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/etc", "/opt")
_CREDENTIAL_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
_PROBE_TIMEOUT_S = 20.0


class SandboxUnavailableError(Exception):
    """No acceptable way to run the plugin apart from the runner on this machine."""


@dataclass(frozen=True, slots=True)
class Sandbox:
    """The isolation chosen for one run, and what decided it."""

    isolation: Isolation
    #: ``kernel.yama.ptrace_scope`` (0-3), or ``None`` where it is not available.
    ptrace_scope: int | None

    def record(self) -> dict[str, object]:
        return {"isolation": self.isolation, "ptrace_scope": self.ptrace_scope}


@dataclass(frozen=True, slots=True)
class SandboxPaths:
    """What a sandboxed plugin may see."""

    #: The plugin environment's root and its base interpreter's installation (read-only).
    environment: Path
    interpreter_home: Path | None
    #: The job file and the inputs directory (read-only).
    job: Path
    inputs: Path
    #: The one writable directory.
    output: Path


def ptrace_scope() -> int | None:
    """``kernel.yama.ptrace_scope``, or ``None`` when the machine has no such setting."""
    try:
        return int(_YAMA.read_text(encoding="ascii").strip())
    except OSError, ValueError:
        return None


async def _succeeds(args: Sequence[str]) -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
    except OSError:
        return False
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT_S):
            return await proc.wait() == 0
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return False


_BWRAP_PROBE = (
    "--ro-bind",
    "/",
    "/",
    "--unshare-pid",
    "--unshare-net",
    "--die-with-parent",
    "true",
)


async def choose_sandbox(
    *, allow_unisolated: bool, environ: Mapping[str, str] | None = None
) -> Sandbox:
    """The strongest isolation that works here, or :class:`SandboxUnavailableError`.

    Each candidate is probed by running it, not by looking for its binary: a ``bwrap`` the
    kernel's user-namespace policy blocks, or a ``sudo`` that would ask for a password, fails its
    probe and is passed over.
    """
    scope = ptrace_scope()
    if sys.platform.startswith("linux"):
        bwrap, sudo, setpriv = shutil.which("bwrap"), shutil.which("sudo"), shutil.which("setpriv")
        if bwrap and sudo and setpriv and await _succeeds([sudo, "-n", bwrap, *_BWRAP_PROBE]):
            return Sandbox("uid", scope)
        if bwrap and await _succeeds([bwrap, *_BWRAP_PROBE]):
            return Sandbox("bwrap", scope)
    if not allow_unisolated:
        msg = (
            "no sandbox is available on this machine to run the plugin apart from the run's "
            "credential: install bubblewrap (bwrap), or the operator must allow this plugin to "
            "run unisolated"
        )
        raise SandboxUnavailableError(msg)
    if sys.platform.startswith("linux") and (scope is None or scope == 0):
        msg = (
            "this machine lets any process ptrace another of its user "
            f"(kernel.yama.ptrace_scope {'unset' if scope is None else scope}), so an unisolated "
            "plugin could read the run's credential: install bubblewrap (bwrap) or set "
            "ptrace_scope to 1 or more"
        )
        raise SandboxUnavailableError(msg)
    source = os.environ if environ is None else environ
    exposed = sorted(n for n in source if any(w in n.upper() for w in _CREDENTIAL_WORDS))
    if exposed:
        msg = (
            "the runner's environment holds credential-shaped variables "
            f"({', '.join(exposed)}), which an unisolated plugin could read: unset them or "
            "install bubblewrap (bwrap)"
        )
        raise SandboxUnavailableError(msg)
    return Sandbox("none", scope)


def _gpu_devices() -> list[str]:
    dev = Path("/dev")
    return sorted(str(p) for p in [*dev.glob("nvidia*"), *dev.glob("dri/*")] if p.exists())


def sandbox_command(
    sandbox: Sandbox,
    inner: Sequence[str],
    *,
    paths: SandboxPaths,
    env: Mapping[str, str],
    gpu: bool,
) -> list[str]:
    """The full command that runs ``inner`` (the plugin host) under ``sandbox``.

    For ``none`` it is ``inner`` itself (the caller passes ``env`` to the process). For the
    bubblewrap sandboxes the environment is cleared and set inside, so nothing of the caller's
    (or of ``sudo``'s) environment crosses over.
    """
    if sandbox.isolation == "none":
        return list(inner)
    bwrap = shutil.which("bwrap") or "bwrap"
    args: list[str] = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--clearenv",
    ]
    for directory in _SYSTEM_DIRS:
        args += ["--ro-bind-try", directory, directory]
    args += ["--proc", "/proc", "--dev", "/dev", "--perms", "1777", "--tmpfs", _SANDBOX_TMP]
    readonly = [paths.environment, paths.job, paths.inputs]
    if paths.interpreter_home is not None:
        readonly.append(paths.interpreter_home)
    for path in readonly:
        args += ["--ro-bind", str(path), str(path)]
    args += ["--bind", str(paths.output), str(paths.output)]
    if gpu:
        # The driver's device nodes, and /sys read-only for the device queries CUDA and NVML make.
        args += ["--ro-bind-try", "/sys", "/sys"]
        for device in _gpu_devices():
            args += ["--dev-bind", device, device]
    for name, value in sorted(
        {
            **env,
            "HOME": _SANDBOX_TMP,
            "TMPDIR": _SANDBOX_TMP,
            "HF_HOME": f"{_SANDBOX_TMP}/hf",
        }.items()
    ):
        args += ["--setenv", name, value]
    args += ["--chdir", _SANDBOX_TMP, "--"]
    if sandbox.isolation == "uid":
        setpriv = shutil.which("setpriv") or "setpriv"
        sudo = shutil.which("sudo") or "sudo"
        args = [
            sudo,
            "-n",
            *args,
            setpriv,
            f"--reuid={NOBODY_UID}",
            f"--regid={NOBODY_UID}",
            "--clear-groups",
            "--no-new-privs",
        ]
    return [*args, *inner]


def kill_command(sandbox: Sandbox, pgid: int, signal_name: Literal["TERM", "KILL"]) -> list[str]:
    """For a ``uid`` sandbox the processes are root's and nobody's, so ``sudo`` signals them."""
    sudo = shutil.which("sudo") or "sudo"
    return [sudo, "-n", "kill", "-s", signal_name, "--", f"-{pgid}"]


def chmod_for_uid(paths: SandboxPaths) -> None:
    """Let the ``nobody`` uid read the inputs and write the output (inside the runner's own,
    private work directory, which no other user can reach on the host)."""
    # Readable and writable by nobody's uid, reachable only inside the private workdir.
    for root, _, files in paths.inputs.walk():
        root.chmod(0o755)
        for name in files:
            (root / name).chmod(0o644)
    paths.job.chmod(0o644)
    paths.output.chmod(0o777)
