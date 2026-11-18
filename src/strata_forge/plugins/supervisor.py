"""Tie a ``uid`` sandbox's life to the runner's.

In a ``uid`` sandbox the runner starts ``sudo -n bwrap ...``: bubblewrap's parent is ``sudo``,
not the runner, so ``--die-with-parent`` follows ``sudo``, which outlives a runner that is killed.
Without more, a plugin would go on running (and writing) after the run that started it is gone.

So inside the sandbox, still as root and before the drop to ``nobody``, this process starts the
rest of the command and watches its own standard input: a pipe whose write end only the runner
holds (:func:`strata_forge.plugins.child.run_plugin_process`). Nothing is ever written to it; it
reaches end of file only when the runner closes it, which the kernel does when the runner dies,
however it dies. Then this process kills its child and exits, and since it is the sandbox's first
process, bubblewrap's init exits with it and the kernel kills everything left in the sandbox's
PID namespace. Otherwise it waits for its child and exits with its status.

The child's standard input is ``/dev/null``, so the plugin never holds the pipe, and the ``nobody``
uid it runs as cannot signal this root process. Started without such a pipe (standard input
already at end of file), it ends its child at once: there is no unsupervised mode.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading

__all__ = ["LIFELINE_CLOSED_EXIT", "exit_status", "main"]

#: The exit status when the runner's end of the pipe closed before the child finished.
LIFELINE_CLOSED_EXIT = 125


def exit_status(returncode: int) -> int:
    """A child's return code as a shell would report it (``128 + n`` for signal ``n``)."""
    return 128 - returncode if returncode < 0 else returncode


def _watch(child: subprocess.Popen[bytes], fd: int) -> None:
    with contextlib.suppress(OSError):
        while os.read(fd, 4096):
            pass
    # The runner is gone (or never handed over a pipe): nothing of the sandbox may outlive it.
    with contextlib.suppress(ProcessLookupError):
        child.send_signal(signal.SIGKILL)
    os._exit(LIFELINE_CLOSED_EXIT)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["--"]:
        args = args[1:]
    if not args:
        print("usage: python -m strata_forge.plugins.supervisor -- <command...>", file=sys.stderr)
        return 2
    child = subprocess.Popen(args, stdin=subprocess.DEVNULL)  # noqa: S603 -- the runner's own command
    threading.Thread(target=_watch, args=(child, sys.stdin.fileno()), daemon=True).start()
    return exit_status(child.wait())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
