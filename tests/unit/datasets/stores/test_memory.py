"""Unit tests for `strata_forge.datasets.stores.memory`."""

from __future__ import annotations

import pytest

from strata_forge.datasets.schema import Dataset, DatasetItem
from strata_forge.datasets.store import DatasetNotFoundError, DatasetStore
from strata_forge.datasets.stores.memory import InMemoryDatasetStore
from strata_forge.datasets.versioning import dataset_version


def _ds(name: str, *ids: str, metadata: dict[str, object] | None = None) -> Dataset:
    items = tuple(DatasetItem(id=i, input={"q": i}) for i in ids)
    return Dataset(name=name, items=items, metadata=metadata or {})


# ---------------------------------------------------------------------------
# Interface conformance
# ---------------------------------------------------------------------------


class TestInterface:
    def test_is_datasetstore(self) -> None:
        assert isinstance(InMemoryDatasetStore(), DatasetStore)


# ---------------------------------------------------------------------------
# put
# ---------------------------------------------------------------------------


class TestPut:
    async def test_put_returns_content_hash_version(self) -> None:
        store = InMemoryDatasetStore()
        ds = _ds("train", "a", "b")
        version = await store.put(ds)
        assert version == dataset_version(ds)
        assert len(version) == 64

    async def test_identical_content_dedupes(self) -> None:
        store = InMemoryDatasetStore()
        v1 = await store.put(_ds("train", "a", "b"))
        v2 = await store.put(_ds("train", "a", "b"))
        # Same content → same version → only one entry stored.
        assert v1 == v2
        assert await store.versions("train") == [v1]

    async def test_different_content_distinct_versions(self) -> None:
        store = InMemoryDatasetStore()
        v1 = await store.put(_ds("train", "a"))
        v2 = await store.put(_ds("train", "a", "b"))
        assert v1 != v2
        # Both versions stored, newest-first.
        assert await store.versions("train") == [v2, v1]

    async def test_different_names_separate_lineages(self) -> None:
        store = InMemoryDatasetStore()
        await store.put(_ds("train", "a"))
        await store.put(_ds("eval", "a"))
        assert set(await store.list_names()) == {"train", "eval"}


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    async def test_get_latest_returns_most_recent(self) -> None:
        store = InMemoryDatasetStore()
        await store.put(_ds("x", "first"))
        await store.put(_ds("x", "first", "second"))
        latest = await store.get("x")
        assert {item.id for item in latest.items} == {"first", "second"}

    async def test_get_specific_version(self) -> None:
        store = InMemoryDatasetStore()
        v1 = await store.put(_ds("x", "first"))
        v2 = await store.put(_ds("x", "second"))
        ds1 = await store.get("x", v1)
        ds2 = await store.get("x", v2)
        assert {item.id for item in ds1.items} == {"first"}
        assert {item.id for item in ds2.items} == {"second"}

    async def test_unknown_name_raises(self) -> None:
        store = InMemoryDatasetStore()
        with pytest.raises(DatasetNotFoundError) as info:
            await store.get("missing")
        assert info.value.name == "missing"

    async def test_unknown_version_raises(self) -> None:
        store = InMemoryDatasetStore()
        await store.put(_ds("x", "a"))
        with pytest.raises(DatasetNotFoundError) as info:
            await store.get("x", "deadbeef" * 8)
        assert info.value.name == "x"
        assert info.value.version is not None


# ---------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------


class TestVersions:
    async def test_newest_first(self) -> None:
        store = InMemoryDatasetStore()
        v1 = await store.put(_ds("x", "a"))
        v2 = await store.put(_ds("x", "a", "b"))
        v3 = await store.put(_ds("x", "a", "b", "c"))
        assert await store.versions("x") == [v3, v2, v1]

    async def test_single_version(self) -> None:
        store = InMemoryDatasetStore()
        v = await store.put(_ds("x", "a"))
        assert await store.versions("x") == [v]

    async def test_unknown_name_raises(self) -> None:
        store = InMemoryDatasetStore()
        with pytest.raises(DatasetNotFoundError) as info:
            await store.versions("missing")
        assert info.value.name == "missing"


# ---------------------------------------------------------------------------
# list_names
# ---------------------------------------------------------------------------


class TestListNames:
    async def test_empty_store(self) -> None:
        assert await InMemoryDatasetStore().list_names() == []

    async def test_sorted_distinct(self) -> None:
        store = InMemoryDatasetStore()
        await store.put(_ds("z", "a"))
        await store.put(_ds("a", "a"))
        await store.put(_ds("m", "a"))
        await store.put(_ds("a", "a", "b"))  # second version of "a"
        assert await store.list_names() == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_delete_all_versions(self) -> None:
        store = InMemoryDatasetStore()
        await store.put(_ds("x", "a"))
        await store.put(_ds("x", "a", "b"))
        await store.delete("x")
        with pytest.raises(DatasetNotFoundError):
            await store.get("x")

    async def test_delete_specific_version(self) -> None:
        store = InMemoryDatasetStore()
        v1 = await store.put(_ds("x", "a"))
        v2 = await store.put(_ds("x", "a", "b"))
        await store.delete("x", v1)
        # v1 gone; v2 still resolves.
        with pytest.raises(DatasetNotFoundError):
            await store.get("x", v1)
        retrieved = await store.get("x", v2)
        assert retrieved is not None

    async def test_delete_only_remaining_version_removes_name(self) -> None:
        store = InMemoryDatasetStore()
        v = await store.put(_ds("x", "a"))
        await store.delete("x", v)
        assert await store.list_names() == []

    async def test_delete_unknown_name_is_noop(self) -> None:
        store = InMemoryDatasetStore()
        await store.delete("missing")
        await store.delete("missing", "some-version")

    async def test_delete_unknown_version_on_known_name_raises(self) -> None:
        store = InMemoryDatasetStore()
        await store.put(_ds("x", "a"))
        with pytest.raises(DatasetNotFoundError) as info:
            await store.delete("x", "deadbeef" * 8)
        assert info.value.name == "x"

    async def test_dedup_preserved_after_partial_delete(self) -> None:
        # Put v1, put v2 (different content), delete v1, re-put v1 content.
        # The re-put should yield the same content hash as the original
        # v1 even though v1 was deleted.
        store = InMemoryDatasetStore()
        v1 = await store.put(_ds("x", "a"))
        await store.put(_ds("x", "a", "b"))
        await store.delete("x", v1)
        v1_again = await store.put(_ds("x", "a"))
        assert v1_again == v1


# ---------------------------------------------------------------------------
# End-to-end round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    async def test_full_identity_preserved(self) -> None:
        store = InMemoryDatasetStore()
        original = Dataset(
            name="train",
            items=(
                DatasetItem.from_input({"q": "1"}, expected_output="one"),
                DatasetItem.from_input({"q": "2"}, expected_output="two"),
            ),
            description="A small set.",
            metadata={"source": "synthetic"},
        )
        version = await store.put(original)
        retrieved = await store.get("train", version)
        # Pydantic frozen models compare by value.
        assert retrieved == original
