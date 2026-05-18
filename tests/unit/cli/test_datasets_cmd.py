"""Unit tests for `forge.cli.datasets`."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from forge.cli.main import app
from forge.datasets.schema import Dataset, DatasetItem
from forge.datasets.stores.memory import InMemoryDatasetStore

runner = CliRunner()


def _make_dataset(
    name: str = "eval-pack",
    n_items: int = 3,
) -> Dataset:
    items = tuple(
        DatasetItem(
            id=f"item-{i}",
            input={"prompt": f"q{i}"},
            expected_output=f"a{i}",
        )
        for i in range(n_items)
    )
    return Dataset(name=name, description="tiny test set", items=items)


@pytest.fixture
def store_with_one(monkeypatch: pytest.MonkeyPatch) -> InMemoryDatasetStore:
    import asyncio

    store = InMemoryDatasetStore()
    asyncio.run(store.put(_make_dataset()))
    monkeypatch.setattr("forge.cli.datasets.dataset_store_from_settings", lambda: store)
    return store


@pytest.fixture
def empty_store(monkeypatch: pytest.MonkeyPatch) -> InMemoryDatasetStore:
    store = InMemoryDatasetStore()
    monkeypatch.setattr("forge.cli.datasets.dataset_store_from_settings", lambda: store)
    return store


class TestList:
    def test_lists_datasets(self, store_with_one: InMemoryDatasetStore) -> None:
        result = runner.invoke(app, ["datasets", "list"])
        assert result.exit_code == 0
        assert "eval-pack" in result.output

    def test_empty_message(self, empty_store: InMemoryDatasetStore) -> None:
        result = runner.invoke(app, ["datasets", "list"])
        assert result.exit_code == 0
        assert "no datasets registered" in result.output


class TestShow:
    def test_show_dataset(self, store_with_one: InMemoryDatasetStore) -> None:
        result = runner.invoke(app, ["datasets", "show", "eval-pack"])
        assert result.exit_code == 0
        assert "eval-pack" in result.output
        assert "items" in result.output
        assert "3" in result.output  # n_items default

    def test_show_missing(self, empty_store: InMemoryDatasetStore) -> None:
        result = runner.invoke(app, ["datasets", "show", "ghost"])
        assert result.exit_code != 0


class TestHead:
    def test_head_default_n(self, store_with_one: InMemoryDatasetStore) -> None:
        result = runner.invoke(app, ["datasets", "head", "eval-pack"])
        assert result.exit_code == 0
        # Output should contain all 3 items (default n=5, n_items=3).
        assert "q0" in result.output
        assert "q1" in result.output
        assert "q2" in result.output

    def test_head_custom_n(self, store_with_one: InMemoryDatasetStore) -> None:
        result = runner.invoke(app, ["datasets", "head", "eval-pack", "-n", "1"])
        assert result.exit_code == 0
        assert "q0" in result.output
        assert "q1" not in result.output
        assert "more items omitted" in result.output

    def test_head_rejects_zero_n(self, store_with_one: InMemoryDatasetStore) -> None:
        result = runner.invoke(app, ["datasets", "head", "eval-pack", "-n", "0"])
        assert result.exit_code != 0

    def test_head_missing_dataset(self, empty_store: InMemoryDatasetStore) -> None:
        result = runner.invoke(app, ["datasets", "head", "ghost"])
        assert result.exit_code != 0
