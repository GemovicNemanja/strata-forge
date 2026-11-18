"""How plugin code is kept apart from the runner that holds the run's credential.

The runner holds the Hub token in memory for the whole run (it downloads the inputs and pushes the
output). Plugin code must not be able to reach it, nor anything else of the runner's: its files,
its ``/proc`` entry, its memory, and the engine environment the runner imports from. Plugin code
runs in two places, and both are sandboxed: the plugin's process itself, and the build of a git
source (its build backend). The strongest sandbox the machine offers is used, in this order:

- ``uid``: bubblewrap started through passwordless ``sudo``, the plugin dropped to the
  unprivileged ``nobody`` uid inside it (``setpriv``, no new privileges, no supplementary
  groups). Another uid cannot read the runner's files or ptrace it, and the sandbox adds the
  namespaces below. Bubblewrap's parent is then ``sudo``, which outlives a killed runner, so the
  sandbox's first process is :mod:`strata_forge.plugins.supervisor` (root, before the drop): it
  watches a pipe only the runner holds, and when the runner dies it ends, and the kernel ends
  everything else in the sandbox with it.
- ``bwrap``: the same bubblewrap sandbox as the runner's own uid. New PID, network, IPC, UTS and
  cgroup namespaces: the runner's processes are not visible (no ``/proc`` entry to read, nothing
  to ptrace), there is no network at all, and the filesystem is only what is bound in: the system
  directories read-only, what the process needs read-only (the interpreter, the engine, the
  plugin's environment, the job file, the inputs) and only its own output read-write, a private
  ``/tmp``. No home directory is bound (``HOME`` is the private ``/tmp``), so no SSH key, cached
  login or token file is there to read, and the engine is read-only, so nothing the plugin does
  can change what the runner imports. The sandbox dies with the runner (``--die-with-parent``).

There is no third, unsandboxed mode. A process of the runner's own uid outside a sandbox can write
the engine environment the runner imports from, and read ``/proc/<runner>/environ`` whatever the
ptrace scope; a machine with no working sandbox does not run plugins.

Every directory bubblewrap creates to hold a bind is made ``0755`` (``--perms``): as root, it
would make them ``0700``, and the ``nobody`` uid could not reach anything bound beneath them.

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
    "engine_paths",
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
    """What sandboxed plugin code may see beyond the system directories: paths bound read-only,
    and the few it may write. Nothing else of the host's filesystem is there."""

    readonly: tuple[Path, ...]
    writable: tuple[Path, ...]


def engine_paths() -> tuple[Path, ...]:
    """What the engine's interpreter needs to run inside a sandbox, read-only: its environment,
    the base installation it runs on, and any import path outside both (an editable install).
    Read-only binds, so the engine itself is never writable from inside."""
    found = {Path(sys.prefix), Path(sys.base_prefix)}
    found.update(Path(p) for p in sys.path if p and Path(p).is_absolute() and Path(p).exists())
    return tuple(sorted(p.resolve() for p in found))


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


async def choose_sandbox() -> Sandbox:
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
    msg = (
        "no sandbox is available on this machine to run plugin code apart from the run's "
        "credential: install bubblewrap (bwrap) and allow it user namespaces, or give the "
        "runner passwordless sudo for bwrap"
    )
    raise SandboxUnavailableError(msg)


def _gpu_devices() -> list[str]:
    dev = Path("/dev")
    return sorted(str(p) for p in [*dev.glob("nvidia*"), *dev.glob("dri/*")] if p.exists())


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _binds(paths: SandboxPaths) -> list[str]:
    """The bind arguments, each preceded by the directories that hold it, made ``0755``.

    A path already visible through a system directory or through another read-only bind is not
    bound again. Shallow paths come first, so a parent's bind never hides a child's."""
    system = [Path(d) for d in _SYSTEM_DIRS]
    readonly = sorted(set(paths.readonly), key=lambda p: (len(p.parts), str(p)))
    kept: list[Path] = []
    for path in readonly:
        if any(_within(path, root) for root in [*system, *kept]):
            continue
        kept.append(path)
    binds = [(p, "--ro-bind") for p in kept] + [(p, "--bind") for p in paths.writable]
    binds.sort(key=lambda b: len(b[0].parts))
    made = {Path("/"), Path(_SANDBOX_TMP)}
    bound: list[Path] = []
    args: list[str] = []
    for path, flag in binds:
        for parent in reversed(path.parents):
            if parent in made or any(_within(parent, root) for root in [*system, *bound]):
                continue
            args += ["--perms", "0755", "--dir", str(parent)]
            made.add(parent)
        args += [flag, str(path), str(path)]
        bound.append(path)
    return args


def sandbox_command(
    sandbox: Sandbox,
    inner: Sequence[str],
    *,
    paths: SandboxPaths,
    env: Mapping[str, str],
    gpu: bool,
) -> list[str]:
    """The full command that runs ``inner`` (plugin code: the host, or a git source's build)
    under ``sandbox``. The environment is cleared and set inside, so nothing of the caller's (or
    of ``sudo``'s) environment crosses over; there is never any network."""
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
    args += _binds(paths)
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
        # Still root inside the sandbox, the supervisor ties the sandbox's life to the runner's
        # (bubblewrap's parent is ``sudo``, which outlives a killed runner), then drops to nobody.
        args = [
            sudo,
            "-n",
            *args,
            sys.executable,
            "-I",
            "-m",
            "strata_forge.plugins.supervisor",
            "--",
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


def chmod_for_uid(*, readable: Sequence[Path], writable: Sequence[Path]) -> None:
    """Let the ``nobody`` uid read ``readable`` and write ``writable``, recursively. Only for
    paths inside the runner's own private work directory: on the host no other user can reach
    them, and inside the sandbox only what is bound is there."""
    for top, mode_dir, mode_file in [
        *((p, 0o755, 0o644) for p in readable),
        *((p, 0o777, 0o666) for p in writable),
    ]:
        if top.is_file():
            top.chmod(mode_file)
            continue
        for root, _, files in top.walk():
            root.chmod(mode_dir)
            for name in files:
                path = root / name
                if not path.is_symlink():
                    executable = path.stat().st_mode & 0o111
                    path.chmod(mode_file | (0o111 if executable else 0))
