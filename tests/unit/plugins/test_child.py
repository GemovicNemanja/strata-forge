"""The plugin's process: only valid plugin events relayed, a bounded stderr tail, a time limit,
and nothing of it left when the call returns, however it ended."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from typing import TYPE_CHECKING

import pytest

from strata_forge.plugins.child import (
    MAX_LINE_BYTES,
    STDERR_TAIL_BYTES,
    relayable,
    run_plugin_process,
)
from strata_forge.plugins.sandbox import Sandbox
from strata_forge.training.progress import ProgressEvent

if TYPE_CHECKING:
    from pathlib import Path

NONE = Sandbox("none", None)


@pytest.fixture(autouse=True)
def short_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    from strata_forge.plugins import child

    monkeypatch.setattr(child, "TERMINATE_GRACE_SECONDS", 1.0)


def _run(
    tmp_path: Path, code: str, *, timeout_s: float = 30.0
) -> tuple[list[ProgressEvent], int, str, bool]:
    events: list[ProgressEvent] = []
    marker = str(tmp_path / "job-marker.json")
    result = asyncio.run(
        run_plugin_process(
            [sys.executable, "-c", code, marker],
            sandbox=NONE,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            cwd=tmp_path,
            marker=marker,
            timeout_s=timeout_s,
            sink=events.append,
        )
    )
    return events, result.code, result.stderr_tail, result.timed_out


def test_only_plugin_events_are_relayed_and_labelled() -> None:
    step = ProgressEvent(
        kind="step", step=3, total_steps=9, metrics={"plugin.acc": 0.5, "gpu_util": 99.0}
    )
    relayed = relayable(step.model_dump_json().encode())
    assert relayed is not None
    assert relayed.metrics == {"plugin.acc": 0.5}  # a forge-shaped gauge is never the plugin's
    assert relayed.stage == "run"
    phase = relayable(ProgressEvent(kind="phase", message="warming up").model_dump_json().encode())
    assert phase is not None
    assert phase.message == "plugin: warming up"
    for kind in ("end", "error", "start"):
        assert relayable(ProgressEvent(kind=kind, message="x").model_dump_json().encode()) is None  # type: ignore[arg-type]
    assert relayable(b"hello from print") is None
    assert relayable(b'{"kind": "step", "unknown": 1}') is None


def test_progress_comes_through_the_pipe_and_stderr_is_capped(tmp_path: Path) -> None:
    code = (
        "import json, sys\n"
        f"sys.stdout.write({json.dumps(ProgressEvent(kind='phase', message='hi').model_dump_json())} + '\\n')\n"
        "sys.stdout.write('x' * " + str(MAX_LINE_BYTES * 2) + " + '\\n')\n"
        "sys.stdout.write('not json\\n')\n"
        "sys.stdout.flush()\n"
        "sys.stderr.write('e' * " + str(STDERR_TAIL_BYTES * 3) + " + 'LAST')\n"
    )
    events, rc, stderr, timed_out = _run(tmp_path, code)
    assert rc == 0
    assert not timed_out
    assert [e.message for e in events] == ["plugin: hi"]
    assert len(stderr) == STDERR_TAIL_BYTES
    assert stderr.endswith("LAST")


def test_a_process_left_behind_after_a_normal_exit_is_killed(tmp_path: Path) -> None:
    pidfile = tmp_path / "orphan.pid"
    code = (
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
    )
    _, rc, _, _ = _run(tmp_path, code)
    assert rc == 0
    orphan = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(orphan, 0)


def test_the_time_limit_stops_the_whole_group(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    code = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(300)\n"
    )
    _, _, _, timed_out = _run(tmp_path, code, timeout_s=1.5)
    assert timed_out
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_a_process_ignoring_sigterm_is_killed(tmp_path: Path) -> None:
    code = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(300)\n"
    _, rc, _, timed_out = _run(tmp_path, code, timeout_s=1.0)
    assert timed_out
    assert rc == -signal.SIGKILL
