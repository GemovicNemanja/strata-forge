"""Unit tests for `forge.cli.eval_cmd`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from forge.cli.main import app
from forge.datasets.schema import Dataset, DatasetItem
from forge.datasets.stores.memory import InMemoryDatasetStore
from forge.evals.experiment import GraderResult, Outcome, Trial
from forge.llm.responses import Usage

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _make_dataset() -> Dataset:
    items = tuple(
        DatasetItem(
            id=f"item-{i}",
            input={"q": f"q{i}"},
            expected_output=f"a{i}",
        )
        for i in range(2)
    )
    return Dataset(name="eval-pack", items=items)


def _make_outcomes() -> tuple[Outcome, ...]:
    trial = Trial(
        experiment_name="eval-pack-claude-T",
        model="claude-opus-4-7",
        provider="anthropic",
        prompt_name=None,
        item_id="item-0",
        response_text="a0",
        usage=Usage(input_tokens=1, output_tokens=1),
        cost_usd=0.0,
        latency_ms=10.0,
    )
    grader = GraderResult(grader_name="exact_match", score=1.0, passed=True)
    return (Outcome(trial=trial, grader_results=(grader,)),)


@pytest.fixture
def fake_eval(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"runs": []}

    async def _fake_run(*args: Any, **kwargs: Any) -> Any:
        state["runs"].append({"args": args, "kwargs": kwargs})
        return _make_outcomes()

    monkeypatch.setattr("forge.evals.runner.run_experiment", _fake_run)

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            state["client_kwargs"] = kwargs

        complete = AsyncMock()

    monkeypatch.setattr("forge.llm.client.LLMClient", _FakeClient)
    return state


@pytest.fixture
def store_with_dataset(monkeypatch: pytest.MonkeyPatch) -> InMemoryDatasetStore:
    import asyncio

    store = InMemoryDatasetStore()
    asyncio.run(store.put(_make_dataset()))
    monkeypatch.setattr("forge.cli.eval_cmd.dataset_store_from_settings", lambda: store)
    return store


@pytest.fixture
def empty_store(monkeypatch: pytest.MonkeyPatch) -> InMemoryDatasetStore:
    store = InMemoryDatasetStore()
    monkeypatch.setattr("forge.cli.eval_cmd.dataset_store_from_settings", lambda: store)
    return store


@pytest.fixture
def isolated_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect both modules' report dirs to a temp path."""
    report_dir = tmp_path / "experiments"
    monkeypatch.setattr("forge.cli.eval_cmd._REPORT_DIR", report_dir)
    monkeypatch.setattr("forge.cli.experiments._REPORT_DIR", report_dir)
    return report_dir


class TestRun:
    def test_runs_and_writes_report(
        self,
        store_with_dataset: InMemoryDatasetStore,
        fake_eval: dict[str, Any],
        isolated_reports: Path,
    ) -> None:
        result = runner.invoke(
            app,
            [
                "eval",
                "run",
                "--model",
                "claude-opus-4-7",
                "--dataset",
                "eval-pack",
                "--grader",
                "exact_match",
                "--name",
                "my-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "my-run" in result.output
        report_path = isolated_reports / "my-run.md"
        assert report_path.exists()

    def test_unknown_grader_rejected(
        self,
        store_with_dataset: InMemoryDatasetStore,
        fake_eval: dict[str, Any],
        isolated_reports: Path,
    ) -> None:
        result = runner.invoke(
            app,
            [
                "eval",
                "run",
                "--model",
                "claude-opus-4-7",
                "--dataset",
                "eval-pack",
                "--grader",
                "no-such-grader",
            ],
        )
        assert result.exit_code != 0
        assert "unknown grader" in result.output

    def test_missing_dataset_rejected(
        self,
        empty_store: InMemoryDatasetStore,
        fake_eval: dict[str, Any],
        isolated_reports: Path,
    ) -> None:
        result = runner.invoke(
            app,
            [
                "eval",
                "run",
                "--model",
                "claude-opus-4-7",
                "--dataset",
                "missing",
                "--grader",
                "exact_match",
            ],
        )
        assert result.exit_code != 0


class TestListShow:
    def test_list_empty(self, isolated_reports: Path) -> None:
        result = runner.invoke(app, ["eval", "list"])
        assert result.exit_code == 0
        assert "no saved experiments" in result.output

    def test_list_directory_exists_but_no_files(self, isolated_reports: Path) -> None:
        isolated_reports.mkdir(parents=True, exist_ok=True)
        result = runner.invoke(app, ["eval", "list"])
        assert result.exit_code == 0
        assert "no saved experiments" in result.output

    def test_list_after_write(self, isolated_reports: Path) -> None:
        isolated_reports.mkdir(parents=True, exist_ok=True)
        (isolated_reports / "alpha.md").write_text("# alpha\n")
        result = runner.invoke(app, ["eval", "list"])
        assert result.exit_code == 0
        assert "alpha" in result.output

    def test_show_missing(self, isolated_reports: Path) -> None:
        result = runner.invoke(app, ["eval", "show", "ghost"])
        assert result.exit_code != 0

    def test_show_existing(self, isolated_reports: Path) -> None:
        isolated_reports.mkdir(parents=True, exist_ok=True)
        (isolated_reports / "alpha.md").write_text("# alpha\nbody\n")
        result = runner.invoke(app, ["eval", "show", "alpha"])
        assert result.exit_code == 0
        assert "body" in result.output
