"""GPU counter sampling — aggregation, caching, and the many ways a box declines to answer.

The failure paths carry most of the weight here. This code runs on someone else's machine, in the
middle of a run that may already be hours long, and the one behaviour that matters more than any
number it reports is that it never raises.
"""

from __future__ import annotations

import subprocess
import threading
from typing import TYPE_CHECKING, Any

from strata_forge.training import hardware
from strata_forge.training.hardware import GpuSampler, sample_gpu

if TYPE_CHECKING:
    import pytest


def _fake_smi(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    *,
    returncode: int = 0,
    binary: str | None = "/usr/bin/nvidia-smi",
    raises: Exception | None = None,
) -> list[list[str]]:
    """Stub `nvidia-smi`, returning the list that records each argv it was called with."""
    calls: list[list[str]] = []

    def _which(_name: str) -> str | None:
        return binary

    class _Proc:
        """A Popen stand-in: `communicate` is what the sampler actually drives."""

        def __init__(self) -> None:
            self.returncode = returncode
            self.killed = False

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            del timeout
            if raises is not None:
                raise raises
            return stdout, ""

        def kill(self) -> None:
            self.killed = True

    def _popen(argv: list[str], **_kwargs: Any) -> Any:
        calls.append(argv)
        return _Proc()

    monkeypatch.setattr(hardware.shutil, "which", _which)
    monkeypatch.setattr(hardware.subprocess, "Popen", _popen)
    return calls


class TestSampleGpu:
    def test_reads_one_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_smi(monkeypatch, "94, 61440, 81920, 74\n")
        assert sample_gpu() == {
            "gpu_count": 1.0,
            "gpu_util_pct": 94.0,
            "gpu_mem_used_mb": 61440.0,
            "gpu_mem_total_mb": 81920.0,
            "gpu_temp_c": 74.0,
        }

    def test_aggregates_a_multi_gpu_box(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mean utilization, summed memory — and MAX temperature, not mean.

        One card cooking while its neighbours idle is the fact worth surfacing; an average would
        report a comfortable 60C for a box with a throttling GPU in it.
        """
        _fake_smi(monkeypatch, "90, 1000, 4000, 60\n50, 2000, 4000, 88\n")
        got = sample_gpu()
        assert got["gpu_count"] == 2.0
        assert got["gpu_util_pct"] == 70.0
        assert got["gpu_mem_used_mb"] == 3000.0
        assert got["gpu_mem_total_mb"] == 8000.0
        assert got["gpu_temp_c"] == 88.0

    def test_queries_the_documented_fields_without_a_shell(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _fake_smi(monkeypatch, "1, 1, 1, 1\n")
        sample_gpu()
        argv = calls[0]
        assert argv[0] == "/usr/bin/nvidia-smi"  # the resolved binary, never a bare name
        assert argv[1] == ("--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu")
        assert argv[2] == "--format=csv,noheader,nounits"

    def test_no_binary_is_not_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A CPU-only box is a normal box, not a broken one."""
        _fake_smi(monkeypatch, "", binary=None)
        assert sample_gpu() == {}

    def test_a_nonzero_exit_reports_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_smi(monkeypatch, "", returncode=9)
        assert sample_gpu() == {}

    def test_a_timeout_reports_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A driver wedged in an uninterruptible wait must not take the run with it."""
        _fake_smi(monkeypatch, "", raises=subprocess.TimeoutExpired("nvidia-smi", 2.0))
        assert sample_gpu() == {}

    def test_an_os_error_reports_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_smi(monkeypatch, "", raises=OSError("exec format error"))
        assert sample_gpu() == {}

    def test_a_card_missing_one_field_still_contributes_the_others(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`[N/A]` for a single field is routine on MIG instances and vGPU passthrough.

        Handled per FIELD, not per row: dropping the whole card would silently under-report the
        box's memory, which is worse than omitting the one gauge it cannot answer.
        """
        _fake_smi(monkeypatch, "90, 1000, 4000, 60\n50, 2000, 4000, [N/A]\n")
        got = sample_gpu()
        assert got["gpu_count"] == 2.0
        assert got["gpu_util_pct"] == 70.0  # both cards reported utilization
        assert got["gpu_mem_total_mb"] == 8000.0  # and both reported memory
        assert got["gpu_temp_c"] == 60.0  # only the card that could answer

    def test_a_card_reporting_nothing_still_counts_as_a_card(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # nvidia-smi listed it, so the box has it. The aggregates describe what answered.
        _fake_smi(monkeypatch, "90, 1000, 4000, 60\n[N/A], [N/A], [N/A], [N/A]\n\n")
        got = sample_gpu()
        assert got["gpu_count"] == 2.0
        assert got["gpu_util_pct"] == 90.0

    def test_a_changed_column_count_reports_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Guards against a driver whose CSV shape moved: better silent than mislabelled."""
        _fake_smi(monkeypatch, "90, 1000\n")
        assert sample_gpu() == {}


class TestGpuSampler:
    def test_caches_between_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _fake_smi(monkeypatch, "94, 61440, 81920, 74\n")
        sampler = GpuSampler(min_interval_s=3600.0)
        first = sampler.refresh_now()
        sampler.sample()
        second = sampler.sample()
        assert second == first
        assert len(calls) == 1, "a cached read must not shell out again"

    def test_re_reads_once_the_interval_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _fake_smi(monkeypatch, "94, 61440, 81920, 74\n")
        sampler = GpuSampler(min_interval_s=0.0)
        sampler.refresh_now()
        sampler.refresh_now()
        assert len(calls) == 2

    def test_hands_out_copies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reading rides into a ProgressEvent's metrics; a caller mutating it there must
        not corrupt what the next event reports."""
        _fake_smi(monkeypatch, "94, 61440, 81920, 74\n")
        sampler = GpuSampler(min_interval_s=3600.0)
        got = sampler.refresh_now()
        got["gpu_util_pct"] = 0.0
        assert sampler.sample()["gpu_util_pct"] == 94.0

    def test_a_box_with_no_gpu_stays_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_smi(monkeypatch, "", binary=None)
        sampler = GpuSampler()
        assert sampler.refresh_now() == {}
        assert sampler.sample() == {}


class TestAnUnkillableProbe:
    """The failure this module is shaped around: `nvidia-smi` wedged in uninterruptible D state.

    `subprocess.run(timeout=...)` does not bound that case — its POSIX timeout path kills the child
    and then calls `process.wait()` with no timeout, which never returns for a D-state process. The
    caller would block forever inside a metrics call, and because the stall is in a blocking C call
    the run could not even be cancelled.
    """

    def test_a_timeout_kills_then_gives_up_rather_than_waiting_forever(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        killed: list[bool] = []

        class _Wedged:
            returncode = 0

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                del timeout
                # Times out both before AND after the kill — exactly what D state looks like.
                raise subprocess.TimeoutExpired("nvidia-smi", 2.0)

            def kill(self) -> None:
                killed.append(True)

        def _which(_name: str) -> str:
            return "/usr/bin/nvidia-smi"

        monkeypatch.setattr(hardware.shutil, "which", _which)

        def _popen(*_args: Any, **_kwargs: Any) -> Any:
            return _Wedged()

        monkeypatch.setattr(hardware.subprocess, "Popen", _popen)

        assert sample_gpu() == {}
        assert killed == [True], "the child must be killed, not merely awaited"

    def test_a_failed_refresh_keeps_the_last_good_reading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A gap would make the chart look like the GPU stopped; a slightly stale gauge is honest.
        _fake_smi(monkeypatch, "94, 61440, 81920, 74\n")
        sampler = GpuSampler(min_interval_s=0.0)
        assert sampler.refresh_now()["gpu_util_pct"] == 94.0

        _fake_smi(monkeypatch, "", returncode=9)
        assert sampler.refresh_now()["gpu_util_pct"] == 94.0

    def test_sample_never_blocks_on_the_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`sample()` is called from coroutines; it must return without touching the subprocess."""
        started = threading.Event()
        release = threading.Event()

        class _Slow:
            returncode = 0

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                del timeout
                started.set()
                release.wait(5.0)
                return "94, 61440, 81920, 74\n", ""

            def kill(self) -> None: ...

        def _which(_name: str) -> str:
            return "/usr/bin/nvidia-smi"

        monkeypatch.setattr(hardware.shutil, "which", _which)

        def _popen(*_args: Any, **_kwargs: Any) -> Any:
            return _Slow()

        monkeypatch.setattr(hardware.subprocess, "Popen", _popen)

        sampler = GpuSampler(min_interval_s=0.0)
        assert sampler.sample() == {}  # returns at once, with nothing read yet
        assert started.wait(5.0), "the refresh should have been handed to a background thread"
        release.set()
