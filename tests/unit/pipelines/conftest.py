"""Shared fixtures for the runner tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _job_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Run every runner test in a scratch working directory, as a job runs in its own workdir:
    a runner leaves its ``receipt.json`` there, never in the repository."""
    monkeypatch.chdir(tmp_path)
