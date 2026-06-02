"""Unit tests for `forge.cli.train`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from forge.cli.main import app
from forge.datasets.schema import Dataset, DatasetItem
from forge.datasets.stores.memory import InMemoryDatasetStore

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _make_dataset() -> Dataset:
    items = tuple(
        DatasetItem(id=f"item-{i}", input={"q": f"q{i}"}, expected_output=f"a{i}") for i in range(2)
    )
    return Dataset(name="sft-pack", items=items)


@pytest.fixture
def store_with_dataset(monkeypatch: pytest.MonkeyPatch) -> InMemoryDatasetStore:
    import asyncio

    store = InMemoryDatasetStore()
    asyncio.run(store.put(_make_dataset()))
    monkeypatch.setattr("forge.cli.train.dataset_store_from_settings", lambda: store)
    return store


@pytest.fixture
def empty_store(monkeypatch: pytest.MonkeyPatch) -> InMemoryDatasetStore:
    store = InMemoryDatasetStore()
    monkeypatch.setattr("forge.cli.train.dataset_store_from_settings", lambda: store)
    return store


@pytest.fixture
def fake_sft_runner(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"init_args": None, "trained_with": None}

    class _FakeRunner:
        def __init__(self, config: Any, *, peft_config: Any = None) -> None:
            state["init_args"] = {"config": config, "peft_config": peft_config}

        def train(self, *, train_dataset: Any, **_kwargs: Any) -> Any:
            state["trained_with"] = train_dataset

            class _Result:
                output_dir = "./out"
                train_loss = 0.4

            return _Result()

    monkeypatch.setattr("forge.training.sft.SFTRunner", _FakeRunner)
    return state


@pytest.fixture
def fake_pref_runner(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"init_args": None, "trained_with": None}

    class _FakeRunner:
        def __init__(self, config: Any, *, peft_config: Any = None) -> None:
            state["init_args"] = {"config": config, "peft_config": peft_config}

        def train(self, *, train_dataset: Any, **_kwargs: Any) -> Any:
            state["trained_with"] = train_dataset

            class _Result:
                output_dir = "./out"
                train_loss = 0.3

            return _Result()

    monkeypatch.setattr("forge.training.preference.PreferenceRunner", _FakeRunner)
    return state


class TestSFT:
    def test_runs_with_defaults(
        self,
        store_with_dataset: InMemoryDatasetStore,
        fake_sft_runner: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        result = runner.invoke(
            app,
            [
                "train",
                "sft",
                "--model",
                "gpt2",
                "--dataset",
                "sft-pack",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "done" in result.output
        cfg = fake_sft_runner["init_args"]["config"]
        assert cfg.model_id == "gpt2"
        assert cfg.precision == "bf16"
        assert fake_sft_runner["init_args"]["peft_config"] is None

    def test_lora_adapter(
        self,
        store_with_dataset: InMemoryDatasetStore,
        fake_sft_runner: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        runner.invoke(
            app,
            [
                "train",
                "sft",
                "--model",
                "gpt2",
                "--dataset",
                "sft-pack",
                "--output-dir",
                str(tmp_path / "out"),
                "--adapter",
                "lora",
                "--adapter-rank",
                "32",
            ],
        )
        from forge.training.peft import LoRAConfig

        peft = fake_sft_runner["init_args"]["peft_config"]
        assert isinstance(peft, LoRAConfig)
        assert peft.r == 32

    def test_qlora_adapter(
        self,
        store_with_dataset: InMemoryDatasetStore,
        fake_sft_runner: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        runner.invoke(
            app,
            [
                "train",
                "sft",
                "--model",
                "gpt2",
                "--dataset",
                "sft-pack",
                "--output-dir",
                str(tmp_path / "out"),
                "--adapter",
                "qlora",
            ],
        )
        from forge.training.peft import QLoRAConfig

        peft = fake_sft_runner["init_args"]["peft_config"]
        assert isinstance(peft, QLoRAConfig)

    def test_missing_dataset_rejected(
        self,
        empty_store: InMemoryDatasetStore,
        fake_sft_runner: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        result = runner.invoke(
            app,
            [
                "train",
                "sft",
                "--model",
                "gpt2",
                "--dataset",
                "missing",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0

    def test_dataset_file_bypasses_the_store(
        self,
        empty_store: InMemoryDatasetStore,
        fake_sft_runner: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        # The store is empty, so this only succeeds by loading the shipped file.
        ds_file = tmp_path / "dataset.json"
        ds_file.write_text(_make_dataset().model_dump_json(), encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "train",
                "sft",
                "--model",
                "gpt2",
                "--dataset",
                "sft-pack",
                "--dataset-file",
                str(ds_file),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 0, result.output
        trained = fake_sft_runner["trained_with"]
        assert [item.id for item in trained.items] == ["item-0", "item-1"]


class TestDPO:
    def test_runs_with_defaults(
        self,
        store_with_dataset: InMemoryDatasetStore,
        fake_pref_runner: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        result = runner.invoke(
            app,
            [
                "train",
                "dpo",
                "--model",
                "./sft-out",
                "--dataset",
                "sft-pack",
                "--output-dir",
                str(tmp_path / "out"),
                "--beta",
                "0.2",
            ],
        )
        assert result.exit_code == 0, result.output
        cfg = fake_pref_runner["init_args"]["config"]
        assert cfg.method == "dpo"
        assert cfg.beta == 0.2

    def test_missing_dataset_rejected(
        self,
        empty_store: InMemoryDatasetStore,
        fake_pref_runner: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        result = runner.invoke(
            app,
            [
                "train",
                "dpo",
                "--model",
                "./sft-out",
                "--dataset",
                "missing",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
