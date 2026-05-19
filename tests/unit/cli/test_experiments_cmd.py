"""Unit tests for `forge.cli.experiments`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from forge.cli.main import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def isolated_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    report_dir = tmp_path / "experiments"
    monkeypatch.setattr("forge.cli.experiments._REPORT_DIR", report_dir)
    return report_dir


class TestList:
    def test_empty(self, isolated_reports: Path) -> None:
        result = runner.invoke(app, ["experiments", "list"])
        assert result.exit_code == 0
        assert "no saved experiments" in result.output

    def test_directory_exists_but_no_files(self, isolated_reports: Path) -> None:
        # Directory present, no markdown files — exercises the second
        # "empty" branch separately from the directory-missing branch.
        isolated_reports.mkdir(parents=True, exist_ok=True)
        result = runner.invoke(app, ["experiments", "list"])
        assert result.exit_code == 0
        assert "no saved experiments" in result.output

    def test_lists_reports(self, isolated_reports: Path) -> None:
        isolated_reports.mkdir(parents=True, exist_ok=True)
        (isolated_reports / "alpha.md").write_text("# alpha\n")
        (isolated_reports / "beta.md").write_text("# beta\n")
        result = runner.invoke(app, ["experiments", "list"])
        assert result.exit_code == 0
        assert "alpha" in result.output
        assert "beta" in result.output


class TestShow:
    def test_show_existing(self, isolated_reports: Path) -> None:
        isolated_reports.mkdir(parents=True, exist_ok=True)
        (isolated_reports / "alpha.md").write_text("# alpha\nbody-line\n")
        result = runner.invoke(app, ["experiments", "show", "alpha"])
        assert result.exit_code == 0
        assert "body-line" in result.output

    def test_show_missing(self, isolated_reports: Path) -> None:
        result = runner.invoke(app, ["experiments", "show", "ghost"])
        assert result.exit_code != 0


class TestDelete:
    def test_deletes_existing(self, isolated_reports: Path) -> None:
        isolated_reports.mkdir(parents=True, exist_ok=True)
        (isolated_reports / "gone.md").write_text("# gone\n")
        result = runner.invoke(app, ["experiments", "delete", "gone"])
        assert result.exit_code == 0
        assert "deleted" in result.output
        assert not (isolated_reports / "gone.md").exists()

    def test_missing(self, isolated_reports: Path) -> None:
        result = runner.invoke(app, ["experiments", "delete", "ghost"])
        assert result.exit_code != 0
