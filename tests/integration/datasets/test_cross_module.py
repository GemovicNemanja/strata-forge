"""Cross-module dataset workflows that don't need external services.

Each test exercises two or more pieces of :mod:`forge.datasets`
working together — the unit-test suite covers each piece in
isolation; these tests confirm the seams hold.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import AsyncMock

import pytest

from forge.datasets import (
    Dataset,
    DatasetItem,
    InMemoryDatasetStore,
    dataset_version,
    diff,
    distill,
    from_hf_dataset,
    self_instruct,
    to_hf_dataset,
)
from forge.datasets.synthetic.self_instruct import SelfInstructBatch, SelfInstructItem

# ---------------------------------------------------------------------------
# Fake HF Dataset module (mirrors the fixtures in tests/unit/datasets/test_hf_bridge.py)
# ---------------------------------------------------------------------------


class _FakeHFDataset:
    def __init__(self, columns: dict[str, list[Any]]) -> None:
        self._columns = dict(columns)

        class _Info:
            dataset_name: str | None = None
            description: str | None = None

        self.info = _Info()

    @classmethod
    def from_dict(cls, mapping: dict[str, list[Any]]) -> _FakeHFDataset:
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
    module = types.ModuleType("datasets")
    module.Dataset = _FakeHFDataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)
    return module


# ---------------------------------------------------------------------------
# Synthetic -> store -> retrieve -> diff
# ---------------------------------------------------------------------------


class TestSyntheticThroughStore:
    async def test_self_instruct_output_is_storable_and_diffable(self) -> None:
        # 1) Generate items from seeds via self_instruct (mocked LLM).
        parsed = SelfInstructBatch(
            items=[
                SelfInstructItem(input={"q": "x"}, expected_output="X"),
                SelfInstructItem(input={"q": "y"}, expected_output="Y"),
            ]
        )
        response = AsyncMock()
        response.parsed = parsed
        client = AsyncMock()
        client.complete_structured = AsyncMock(return_value=response)
        v1 = await self_instruct(
            seeds=[DatasetItem.from_input({"q": "seed"})],
            instructions="x",
            n=2,
            client=client,
            name="set",
            batch_size=2,
        )

        # 2) Store v1; the version is the content hash.
        store = InMemoryDatasetStore()
        v1_version = await store.put(v1)
        assert v1_version == dataset_version(v1)

        # 3) Generate a second batch with different content; store as v2.
        parsed2 = SelfInstructBatch(
            items=[
                SelfInstructItem(input={"q": "z"}, expected_output="Z"),
            ]
        )
        response2 = AsyncMock()
        response2.parsed = parsed2
        client.complete_structured = AsyncMock(return_value=response2)
        v2 = await self_instruct(
            seeds=[DatasetItem.from_input({"q": "seed"})],
            instructions="x",
            n=1,
            client=client,
            name="set",
            batch_size=1,
        )
        v2_version = await store.put(v2)
        assert v1_version != v2_version

        # 4) diff() partitions cleanly (no items match across the two batches).
        delta = diff(v1, v2)
        assert len(delta.added) == len(v2)
        assert len(delta.removed) == len(v1)
        assert len(delta.unchanged) == 0

        # 5) Round-trip: store -> get -> equality.
        recovered = await store.get("set", version=v1_version)
        assert recovered == v1


# ---------------------------------------------------------------------------
# distill -> store -> versioning
# ---------------------------------------------------------------------------


class TestDistillThroughStore:
    async def test_distill_output_roundtrips_through_store(self) -> None:
        # Unlabeled items in; labeled items out via teacher.
        unlabeled = (
            DatasetItem.from_input({"q": "a"}),
            DatasetItem.from_input({"q": "b"}),
        )
        client = AsyncMock()
        responses = [AsyncMock(text="ans-a"), AsyncMock(text="ans-b")]
        client.complete = AsyncMock(side_effect=responses)
        labeled = await distill(inputs=unlabeled, teacher=client, name="labeled")

        store = InMemoryDatasetStore()
        version = await store.put(labeled)
        # Putting the labeled dataset and immediately re-putting it must be a no-op.
        again = await store.put(labeled)
        assert version == again

        recovered = await store.get("labeled")
        assert recovered == labeled


# ---------------------------------------------------------------------------
# HF bridge <-> store
# ---------------------------------------------------------------------------


class TestHFBridgeThroughStore:
    def test_hf_roundtrip_preserves_version(self, fake_datasets_module: types.ModuleType) -> None:
        original = Dataset(
            name="bridge",
            description="round-trip",
            items=(
                DatasetItem.from_input({"q": "x"}, expected_output="X"),
                DatasetItem.from_input({"q": "y"}, expected_output="Y"),
            ),
        )
        hf_ds = to_hf_dataset(original)
        recovered = from_hf_dataset(
            hf_ds,
            name=original.name,
            description=original.description,
        )
        # The two datasets are structurally equal: same name, items, description.
        assert recovered == original
        # Therefore the content-hash version matches too.
        assert dataset_version(recovered) == dataset_version(original)

    async def test_hf_roundtrip_then_store_dedupes(
        self, fake_datasets_module: types.ModuleType
    ) -> None:
        original = Dataset(
            name="bridge",
            items=(DatasetItem.from_input({"q": "x"}, expected_output="X"),),
        )
        store = InMemoryDatasetStore()
        v1 = await store.put(original)

        hf_ds = to_hf_dataset(original)
        recovered = from_hf_dataset(hf_ds, name=original.name)
        v2 = await store.put(recovered)

        # The roundtripped dataset hashes identically to the original.
        assert v1 == v2
        assert (await store.versions("bridge")) == [v1]


# ---------------------------------------------------------------------------
# Empty dataset edge cases across modules
# ---------------------------------------------------------------------------


class TestEmptyDatasetAcrossModules:
    def test_empty_dataset_roundtrips_through_hf(
        self, fake_datasets_module: types.ModuleType
    ) -> None:
        original = Dataset(name="empty")
        hf_ds = to_hf_dataset(original)
        recovered = from_hf_dataset(hf_ds, name="empty")
        assert recovered == original

    async def test_empty_dataset_storable(self) -> None:
        store = InMemoryDatasetStore()
        empty = Dataset(name="empty")
        version = await store.put(empty)
        assert (await store.get("empty")) == empty
        assert (await store.versions("empty")) == [version]
