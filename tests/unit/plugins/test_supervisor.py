"""The ``uid`` sandbox's supervisor: its child lives only as long as the runner's pipe is open."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest

from strata_forge.plugins.supervisor import LIFELINE_CLOSED_EXIT, exit_status

if TYPE_CHECKING:
    from pathlib import Path

SUPERVISOR = [sys.executable, "-I", "-m", "strata_forge.plugins.supervisor", "--"]


def _wait_for(path: Path) -> int:
    deadline = time.monotonic() + 20
    while not path.exists() or not path.read_text():
        if time.monotonic() > deadline:
            pytest.fail("the child never started")
        time.sleep(0.05)
    return int(path.read_text())


def _gone(pid: int) -> bool:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        # A zombie of a dead parent is reaped by init; on macOS ``kill`` sees it until then.
        time.sleep(0.05)
    return False


def test_the_child_is_killed_when_the_runner_s_end_closes(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    code = f"import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(300)"
    read, write = os.pipe()
    proc = subprocess.Popen([*SUPERVISOR, sys.executable, "-c", code], stdin=read)  # noqa: S603
    os.close(read)
    child = _wait_for(pidfile)
    assert proc.poll() is None
    os.close(write)  # what the kernel does when the runner dies
    assert proc.wait(timeout=20) == LIFELINE_CLOSED_EXIT
    assert _gone(child)


def test_without_a_pipe_the_child_is_ended_at_once(tmp_path: Path) -> None:
    code = "import time; time.sleep(300)"
    proc = subprocess.run(  # noqa: S603
        [*SUPERVISOR, sys.executable, "-c", code], stdin=subprocess.DEVNULL, timeout=30, check=False
    )
    assert proc.returncode == LIFELINE_CLOSED_EXIT


def test_the_child_s_status_passes_through_while_the_pipe_is_open() -> None:
    read, write = os.pipe()
    try:
        done = subprocess.run(  # noqa: S603
            [*SUPERVISOR, sys.executable, "-c", "raise SystemExit(3)"],
            stdin=read,
            timeout=30,
            check=False,
        )
    finally:
        os.close(read)
        os.close(write)
    assert done.returncode == 3


def test_a_signalled_child_reports_as_a_shell_would() -> None:
    assert exit_status(-9) == 137
    assert exit_status(0) == 0
    assert exit_status(2) == 2
