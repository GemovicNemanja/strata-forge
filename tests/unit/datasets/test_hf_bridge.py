"""Unit tests for `strata_forge.datasets.hf_bridge`."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from strata_forge.datasets.hf_bridge import from_hf_dataset, to_hf_dataset
from strata_forge.datasets.schema import Dataset, DatasetItem

# ---------------------------------------------------------------------------
# Minimal fake `datasets` module + Dataset class
# ---------------------------------------------------------------------------


@dataclass
class FakeDatasetInfo:
    dataset_name: str | None = None
    description: str | None = None


class FakeHFDataset:
    """A minimal subset of `datasets.Dataset` covering what the bridge uses."""

    def __init__(self, columns: dict[str, list[Any]]) -> None:
        # Validate same length across columns.
        if columns:
            lengths = {len(v) for v in columns.values()}
            if len(lengths) > 1:
                raise ValueError("All columns must have the same length")
        self._columns = dict(columns)
        self.info = FakeDatasetInfo()

    @classmethod
    def from_dict(cls, mapping: dict[str, list[Any]]) -> FakeHFDataset:
        return cls(mapping)

    @property
    def column_names(self) -> list[str]:
        return list(self._columns.keys())

    def __iter__(self):
        rowcount = len(next(iter(self._columns.values()), []))
        for i in range(rowcount):
            yield {name: self._columns[name][i] for name in self._columns}

    def __len__(self) -> int:
        return len(next(iter(self._columns.values()), []))


@pytest.fixture
def fake_datasets_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake `datasets` module exposing only what we use."""
    module = types.ModuleType("datasets")
    module.Dataset = FakeHFDataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)
    return module


# ---------------------------------------------------------------------------
# Lazy import contract
# ---------------------------------------------------------------------------


class TestLazyImport:
    def test_module_importable_without_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "datasets", None)
        import importlib

        from strata_forge.datasets import hf_bridge

        importlib.reload(hf_bridge)
        assert hasattr(hf_bridge, "to_hf_dataset")
        assert hasattr(hf_bridge, "from_hf_dataset")

    def test_to_hf_dataset_raises_without_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "datasets", None)
        ds = Dataset(name="x", items=(DatasetItem(id="a", input={"q": "y"}),))
        with pytest.raises(ImportError, match=r"\[hf\] extra"):
            to_hf_dataset(ds)

    def test_from_hf_dataset_does_not_need_extra(
        self, fake_datasets_module: types.ModuleType
    ) -> None:
        # `from_hf_dataset` reads from an already-instantiated HF
        # Dataset; the extra is only needed at the boundary that
        # constructs one. Since the caller supplies the object, we
        # don't need to import anything inside this function.
        hf = FakeHFDataset({"input": [{"q": "a"}], "id": ["item-a"]})
        result = from_hf_dataset(hf, name="x")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# to_hf_dataset
# ---------------------------------------------------------------------------


class TestToHFDataset:
    def test_basic_conversion(self, fake_datasets_module: types.ModuleType) -> None:
        ds = Dataset(
            name="set",
            items=(
                DatasetItem(id="a", input={"q": "1"}, expected_output="one"),
                DatasetItem(id="b", input={"q": "2"}, expected_output="two"),
            ),
        )
        hf = to_hf_dataset(ds)
        assert hf.column_names == ["id", "input", "expected_output", "metadata"]
        rows = list(hf)
        assert rows[0]["id"] == "a"
        assert rows[0]["input"] == {"q": "1"}
        assert rows[0]["expected_output"] == "one"
        assert rows[0]["metadata"] == {}

    def test_propagates_name_and_description_to_info(
        self, fake_datasets_module: types.ModuleType
    ) -> None:
        ds = Dataset(
            name="my-name",
            description="hello world",
            items=(DatasetItem(id="a", input={"q": "y"}),),
        )
        hf = to_hf_dataset(ds)
        assert hf.info.dataset_name == "my-name"
        assert hf.info.description == "hello world"

    def test_empty_dataset(self, fake_datasets_module: types.ModuleType) -> None:
        ds = Dataset(name="empty")
        hf = to_hf_dataset(ds)
        assert len(hf) == 0
        assert "id" in hf.column_names

    def test_metadata_per_item(self, fake_datasets_module: types.ModuleType) -> None:
        ds = Dataset(
            name="x",
            items=(
                DatasetItem(id="a", input={"q": "1"}, metadata={"tier": "easy"}),
                DatasetItem(id="b", input={"q": "2"}, metadata={"tier": "hard"}),
            ),
        )
        hf = to_hf_dataset(ds)
        rows = list(hf)
        assert rows[0]["metadata"] == {"tier": "easy"}
        assert rows[1]["metadata"] == {"tier": "hard"}


# ---------------------------------------------------------------------------
# from_hf_dataset
# ---------------------------------------------------------------------------


class TestFromHFDataset:
    def test_basic_ingest(self) -> None:
        hf = FakeHFDataset(
            {
                "id": ["a", "b"],
                "input": [{"q": "1"}, {"q": "2"}],
                "expected_output": ["one", "two"],
                "metadata": [{}, {"tier": "hard"}],
            }
        )
        ds = from_hf_dataset(hf, name="myset")
        assert ds.name == "myset"
        assert len(ds) == 2
        assert ds.items[0].id == "a"
        assert ds.items[0].input == {"q": "1"}
        assert ds.items[0].expected_output == "one"
        assert ds.items[1].metadata == {"tier": "hard"}

    def test_missing_id_column_uses_content_hash(self) -> None:
        hf = FakeHFDataset(
            {
                "input": [{"q": "a"}],
                "expected_output": ["A"],
            }
        )
        ds = from_hf_dataset(hf, name="x")
        item = ds.items[0]
        # The derived ID is a SHA-256 hex digest (64 chars).
        assert len(item.id) == 64

    def test_per_row_id_none_falls_back_to_content_hash(self) -> None:
        hf = FakeHFDataset(
            {
                "id": [None, "b"],
                "input": [{"q": "1"}, {"q": "2"}],
                "expected_output": ["one", "two"],
            }
        )
        ds = from_hf_dataset(hf, name="x")
        assert ds.items[0].id != ds.items[1].id
        assert len(ds.items[0].id) == 64  # derived hash
        assert ds.items[1].id == "b"  # explicit

    def test_custom_column_names(self) -> None:
        hf = FakeHFDataset(
            {
                "uuid": ["a"],
                "prompt": [{"text": "hello"}],
                "answer": ["world"],
            }
        )
        ds = from_hf_dataset(
            hf,
            name="x",
            id_column="uuid",
            input_column="prompt",
            expected_output_column="answer",
        )
        assert ds.items[0].id == "a"
        assert ds.items[0].input == {"text": "hello"}
        assert ds.items[0].expected_output == "world"

    def test_no_expected_output_column(self) -> None:
        hf = FakeHFDataset({"id": ["a"], "input": [{"q": "x"}]})
        ds = from_hf_dataset(hf, name="x", expected_output_column=None, metadata_column=None)
        assert ds.items[0].expected_output is None
        assert ds.items[0].metadata == {}

    def test_missing_input_column_raises(self) -> None:
        hf = FakeHFDataset({"id": ["a"], "wrong": [{"q": "x"}]})
        with pytest.raises(KeyError, match="Input column"):
            from_hf_dataset(hf, name="x")

    def test_non_dict_input_raises(self) -> None:
        hf = FakeHFDataset({"id": ["a"], "input": ["just-a-string"]})
        with pytest.raises(TypeError, match="must hold dicts"):
            from_hf_dataset(hf, name="x")

    def test_propagates_top_level_metadata(self) -> None:
        hf = FakeHFDataset({"input": [{"q": "x"}]})
        ds = from_hf_dataset(hf, name="x", description="hi", metadata={"src": "synth"})
        assert ds.description == "hi"
        assert ds.metadata == {"src": "synth"}


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_to_then_from_preserves_items(self, fake_datasets_module: types.ModuleType) -> None:
        original = Dataset(
            name="myset",
            description="round-trip",
            metadata={"src": "test"},
            items=(
                DatasetItem(
                    id="a",
                    input={"q": "1", "tags": ["x"]},
                    expected_output="one",
                    metadata={"tier": "easy"},
                ),
                DatasetItem(
                    id="b",
                    input={"q": "2"},
                    expected_output={"role": "assistant", "content": "two"},
                    metadata={},
                ),
            ),
        )
        hf = to_hf_dataset(original)
        recovered = from_hf_dataset(
            hf,
            name=original.name,
            description=original.description,
            metadata=dict(original.metadata),
        )
        assert recovered == original
