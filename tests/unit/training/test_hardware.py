"""GPU counter sampling — aggregation, caching, and the many ways a box declines to answer.

The failure paths carry most of the weight here. This code runs on someone else's machine, in the
middle of a run that may already be hours long, and the one behaviour that matters more than any
number it reports is that it never raises.
"""

from __future__ import annotations

import subprocess
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

    def _run(argv: list[str], **_kwargs: Any) -> Any:
        calls.append(argv)
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(hardware.shutil, "which", _which)
    monkeypatch.setattr(hardware.subprocess, "run", _run)
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

    def test_skips_rows_it_cannot_parse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`[N/A]` is what a card that does not report a field actually prints."""
        _fake_smi(monkeypatch, "90, 1000, 4000, 60\n[N/A], [N/A], [N/A], [N/A]\n\n")
        got = sample_gpu()
        assert got["gpu_count"] == 1.0
        assert got["gpu_util_pct"] == 90.0

    def test_a_changed_column_count_reports_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Guards against a driver whose CSV shape moved: better silent than mislabelled."""
        _fake_smi(monkeypatch, "90, 1000\n")
        assert sample_gpu() == {}


class TestGpuSampler:
    def test_caches_between_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _fake_smi(monkeypatch, "94, 61440, 81920, 74\n")
        sampler = GpuSampler(min_interval_s=3600.0)
        first = sampler.sample()
        second = sampler.sample()
        assert first == second
        assert len(calls) == 1, "a cached read must not shell out again"

    def test_re_reads_once_the_interval_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _fake_smi(monkeypatch, "94, 61440, 81920, 74\n")
        sampler = GpuSampler(min_interval_s=0.0)
        sampler.sample()
        sampler.sample()
        assert len(calls) == 2

    def test_hands_out_copies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reading rides into a ProgressEvent's metrics; a caller mutating it there must
        not corrupt what the next event reports."""
        _fake_smi(monkeypatch, "94, 61440, 81920, 74\n")
        sampler = GpuSampler(min_interval_s=3600.0)
        got = sampler.sample()
        got["gpu_util_pct"] = 0.0
        assert sampler.sample()["gpu_util_pct"] == 94.0

    def test_a_box_with_no_gpu_stays_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_smi(monkeypatch, "", binary=None)
        sampler = GpuSampler()
        assert sampler.sample() == {}
        assert sampler.sample() == {}
