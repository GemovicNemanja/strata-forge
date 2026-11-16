"""POSIX privacy for both local audit writers, including reused legacy metric serialization."""

from __future__ import annotations

import os
import stat
from typing import TYPE_CHECKING, Any

import pytest

from strata_forge.pipelines import _population_report as report
from strata_forge.pipelines._population import PopulationError
from strata_forge.pipelines.eval_runner import run_population_evaluation
from strata_forge.pipelines.inference_runner import execute_population_scoring
from tests.unit.pipelines.test_population_workers import Policy, Reader, job
from tests.unit.pipelines.test_population_workers import fake_hf as _fake_hf
from tests.unit.pipelines.test_population_workers import hf_fixture as _hf_fixture

fake_hf = _fake_hf
hf_fixture = _hf_fixture

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("name", ["inference", "native_evaluation"])
async def test_public_existing_leaf_refuses_without_mutating_caller_directory(
    name: str, tmp_path: Path
) -> None:
    parent = tmp_path / "public"
    parent.mkdir()
    parent.chmod(0o755)
    directory = parent / "existing"
    directory.mkdir()
    directory.chmod(0o755)
    reader = Reader()
    call = execute_population_scoring if name == "inference" else run_population_evaluation
    with pytest.raises(PopulationError, match="artifact_directory_not_private"):
        await call(job(name), reader=reader, policy=Policy(), output_dir=directory)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o755
    assert list(directory.iterdir()) == []
    assert reader.calls == []
    assert stat.S_IMODE(parent.stat().st_mode) == 0o755


@pytest.mark.parametrize("name", ["inference", "native_evaluation"])
@pytest.mark.parametrize("umask", [0, 0o777])
async def test_owned_attempt_all_files_private_independent_of_umask(
    name: str,
    umask: int,
    fake_hf: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "public"
    parent.mkdir()
    parent.chmod(0o755)
    directory = parent / "created" / "nested" / "attempt"
    original_metric_writer = report._write_items  # pyright: ignore[reportPrivateUsage] — observe the actual reused metric file writer.
    metrics_modes: list[int] = []

    def metric_writer(path: Path, *args: Any, **kwargs: Any) -> None:
        metrics_modes.append(stat.S_IMODE(path.stat().st_mode))
        original_metric_writer(path, *args, **kwargs)
        metrics_modes.append(stat.S_IMODE(path.stat().st_mode))

    monkeypatch.setattr(report, "_write_items", metric_writer)
    original_umask = os.umask(umask)
    try:
        call = execute_population_scoring if name == "inference" else run_population_evaluation
        result = await call(job(name), reader=Reader(), policy=Policy(), output_dir=directory)
    finally:
        os.umask(original_umask)
    assert result.report_path.is_file()
    assert result.counts.rows_in == 5
    assert stat.S_IMODE(parent.stat().st_mode) == 0o755
    assert stat.S_IMODE((parent / "created").stat().st_mode) == 0o700
    for path in (directory, *directory.rglob("*")):
        metadata = path.stat()
        assert metadata.st_uid == os.getuid()
        assert stat.S_IMODE(metadata.st_mode) == (0o700 if path.is_dir() else 0o600)
    if name == "native_evaluation":
        assert metrics_modes == [0o600, 0o600]
    else:
        assert metrics_modes == []


def test_existing_wrong_owner_refuses_without_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    actual_uid = os.getuid()
    monkeypatch.setattr(report.os, "getuid", lambda: actual_uid + 1)
    with pytest.raises(PopulationError, match="artifact_directory_not_private"):
        report.reserve_attempt(directory, run_id="fictional", report_name="scoring.json")
    assert list(directory.iterdir()) == []
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_symlink_leaf_refuses_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    target.chmod(0o700)
    directory = tmp_path / "link"
    directory.symlink_to(target, target_is_directory=True)
    with pytest.raises(PopulationError, match="artifact_directory_not_private"):
        report.reserve_attempt(directory, run_id="fictional", report_name="metrics.json")
    assert directory.is_symlink()
    assert list(target.iterdir()) == []
