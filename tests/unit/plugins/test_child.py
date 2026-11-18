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

NONE = Sandbox("bwrap", None)  # signalled as the runner's own group, like bubblewrap


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


def test_a_process_that_writes_nothing_for_the_idle_limit_is_stalled(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    code = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "print('working', flush=True)\n"
        "time.sleep(300)\n"
    )
    marker = str(tmp_path / "job-marker.json")
    result = asyncio.run(
        run_plugin_process(
            [sys.executable, "-c", code, marker],
            sandbox=NONE,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            cwd=tmp_path,
            marker=marker,
            timeout_s=60.0,
            sink=lambda _event: None,
            idle_timeout_s=1.5,
        )
    )
    assert result.stalled
    assert not result.timed_out
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_output_keeps_a_process_under_its_idle_limit(tmp_path: Path) -> None:
    code = "import sys, time\nfor _ in range(6):\n    print('x', file=sys.stderr, flush=True)\n    time.sleep(0.4)\n"
    marker = str(tmp_path / "job-marker.json")
    result = asyncio.run(
        run_plugin_process(
            [sys.executable, "-c", code, marker],
            sandbox=NONE,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            cwd=tmp_path,
            marker=marker,
            timeout_s=60.0,
            sink=lambda _event: None,
            idle_timeout_s=1.5,
        )
    )
    assert (result.code, result.stalled) == (0, False)


def test_a_process_ignoring_sigterm_is_killed(tmp_path: Path) -> None:
    code = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(300)\n"
    _, rc, _, timed_out = _run(tmp_path, code, timeout_s=1.0)
    assert timed_out
    assert rc == -signal.SIGKILL


def _fake_proc(root: Path, table: dict[int, tuple[int, str]]) -> Path:
    for pid, (parent, state) in table.items():
        (root / str(pid)).mkdir(parents=True)
        # A comm with spaces and parentheses, as a hostile process may name itself.
        (root / str(pid) / "stat").write_text(f"{pid} (evil ) name) {state} {parent} 1 1 0\n")
    return root


def test_a_stopped_process_anywhere_under_the_plugin_is_found(tmp_path: Path) -> None:
    from strata_forge.plugins.child import stopped_in_tree

    # 10 is the runner's child; 11 its sandbox init in a session of its own; 12 the plugin.
    proc = _fake_proc(tmp_path, {10: (1, "S"), 11: (10, "t"), 12: (11, "S"), 20: (1, "T")})
    assert stopped_in_tree(10, proc)
    assert not stopped_in_tree(12, proc)  # 20 is stopped, but not the plugin's
    assert not stopped_in_tree(10, tmp_path / "no-proc")


def test_a_plugin_that_stays_stopped_ends_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from strata_forge.plugins import child

    monkeypatch.setattr(child, "STOPPED_LIMIT_SECONDS", 0.5)
    monkeypatch.setattr(child, "_STOPPED_CHECK_SECONDS", 0.1)

    def always_stopped(_pid: int) -> bool:
        return True

    monkeypatch.setattr(child, "stopped_in_tree", always_stopped)
    marker = str(tmp_path / "job-marker.json")
    result = asyncio.run(
        run_plugin_process(
            [sys.executable, "-c", "import time; time.sleep(300)", marker],
            sandbox=NONE,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            cwd=tmp_path,
            marker=marker,
            timeout_s=60.0,
            sink=lambda _e: None,
        )
    )
    assert result.stopped
    assert not result.timed_out


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_a_sigstopped_child_is_seen_on_linux(tmp_path: Path) -> None:
    import subprocess
    import time

    from strata_forge.plugins.child import stopped_in_tree

    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time; "
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "print(p.pid, flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert parent.stdout is not None
    grandchild = int(parent.stdout.readline())
    try:
        assert not stopped_in_tree(parent.pid)
        os.kill(grandchild, signal.SIGSTOP)
        time.sleep(0.3)
        assert stopped_in_tree(parent.pid)
    finally:
        os.kill(grandchild, signal.SIGKILL)
        parent.kill()
        parent.wait()
