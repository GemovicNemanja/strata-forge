"""Unit tests for `forge.datasets.versioning`."""

from __future__ import annotations

import pytest

from forge.datasets.schema import Dataset, DatasetItem
from forge.datasets.versioning import DatasetDelta, dataset_version, diff


def _items(*ids: str) -> tuple[DatasetItem, ...]:
    return tuple(DatasetItem(id=i, input={"q": i}) for i in ids)


# ---------------------------------------------------------------------------
# dataset_version
# ---------------------------------------------------------------------------


class TestDatasetVersion:
    def test_returns_64_char_hex(self) -> None:
        ds = Dataset(name="x", items=_items("a"))
        v = dataset_version(ds)
        assert len(v) == 64
        assert all(c in "0123456789abcdef" for c in v)

    def test_same_content_same_version(self) -> None:
        a = Dataset(name="x", items=_items("a", "b"))
        b = Dataset(name="x", items=_items("a", "b"))
        assert dataset_version(a) == dataset_version(b)

    def test_item_order_does_not_affect_version(self) -> None:
        # The version sorts item IDs internally so insertion order is
        # irrelevant — same set of items → same version.
        a = Dataset(name="x", items=_items("a", "b", "c"))
        b = Dataset(name="x", items=_items("c", "a", "b"))
        assert dataset_version(a) == dataset_version(b)

    def test_different_name_different_version(self) -> None:
        a = Dataset(name="train", items=_items("a"))
        b = Dataset(name="eval", items=_items("a"))
        assert dataset_version(a) != dataset_version(b)

    def test_added_item_changes_version(self) -> None:
        a = Dataset(name="x", items=_items("a"))
        b = Dataset(name="x", items=_items("a", "b"))
        assert dataset_version(a) != dataset_version(b)

    def test_removed_item_changes_version(self) -> None:
        a = Dataset(name="x", items=_items("a", "b"))
        b = Dataset(name="x", items=_items("a"))
        assert dataset_version(a) != dataset_version(b)

    def test_metadata_change_changes_version(self) -> None:
        a = Dataset(name="x", items=_items("a"), metadata={"tier": "easy"})
        b = Dataset(name="x", items=_items("a"), metadata={"tier": "hard"})
        assert dataset_version(a) != dataset_version(b)

    def test_description_does_not_affect_version(self) -> None:
        # Description is free-form prose; bumping it shouldn't cascade.
        a = Dataset(name="x", items=_items("a"), description="v1 notes")
        b = Dataset(name="x", items=_items("a"), description="updated")
        assert dataset_version(a) == dataset_version(b)

    def test_item_metadata_does_not_affect_version(self) -> None:
        # Re-annotating items (changing their .metadata but not their .id)
        # shouldn't change the dataset version.
        item_a = DatasetItem(id="a", input={"q": "1"}, metadata={"tier": "easy"})
        item_b = DatasetItem(id="a", input={"q": "1"}, metadata={"tier": "hard"})
        a = Dataset(name="x", items=(item_a,))
        b = Dataset(name="x", items=(item_b,))
        assert dataset_version(a) == dataset_version(b)

    def test_empty_dataset_has_stable_version(self) -> None:
        a = Dataset(name="empty")
        b = Dataset(name="empty")
        assert dataset_version(a) == dataset_version(b)


# ---------------------------------------------------------------------------
# DatasetDelta
# ---------------------------------------------------------------------------


class TestDatasetDelta:
    def test_is_frozen(self) -> None:
        delta = DatasetDelta(added=(), removed=(), unchanged=())
        with pytest.raises((AttributeError, TypeError)):
            delta.added = (DatasetItem(id="a", input={"q": "1"}),)  # type: ignore[misc]

    def test_is_empty_true_when_no_changes(self) -> None:
        item = DatasetItem(id="a", input={"q": "1"})
        delta = DatasetDelta(added=(), removed=(), unchanged=(item,))
        assert delta.is_empty is True

    def test_is_empty_false_with_additions(self) -> None:
        item = DatasetItem(id="a", input={"q": "1"})
        delta = DatasetDelta(added=(item,), removed=(), unchanged=())
        assert delta.is_empty is False

    def test_is_empty_false_with_removals(self) -> None:
        item = DatasetItem(id="a", input={"q": "1"})
        delta = DatasetDelta(added=(), removed=(item,), unchanged=())
        assert delta.is_empty is False


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_identical_datasets_no_changes(self) -> None:
        ds = Dataset(name="x", items=_items("a", "b", "c"))
        delta = diff(ds, ds)
        assert delta.added == ()
        assert delta.removed == ()
        assert {item.id for item in delta.unchanged} == {"a", "b", "c"}
        assert delta.is_empty

    def test_added_items_detected(self) -> None:
        old = Dataset(name="x", items=_items("a"))
        new = Dataset(name="x", items=_items("a", "b", "c"))
        delta = diff(old, new)
        assert {item.id for item in delta.added} == {"b", "c"}
        assert delta.removed == ()
        assert {item.id for item in delta.unchanged} == {"a"}

    def test_removed_items_detected(self) -> None:
        old = Dataset(name="x", items=_items("a", "b", "c"))
        new = Dataset(name="x", items=_items("a"))
        delta = diff(old, new)
        assert delta.added == ()
        assert {item.id for item in delta.removed} == {"b", "c"}
        assert {item.id for item in delta.unchanged} == {"a"}

    def test_mixed_changes(self) -> None:
        old = Dataset(name="x", items=_items("a", "b", "c"))
        new = Dataset(name="x", items=_items("b", "c", "d"))
        delta = diff(old, new)
        assert {item.id for item in delta.added} == {"d"}
        assert {item.id for item in delta.removed} == {"a"}
        assert {item.id for item in delta.unchanged} == {"b", "c"}

    def test_unchanged_returns_items_from_new(self) -> None:
        # Re-annotated items: same ID, different metadata.
        item_old = DatasetItem(id="a", input={"q": "1"}, metadata={"tier": "easy"})
        item_new = DatasetItem(id="a", input={"q": "1"}, metadata={"tier": "hard"})
        old = Dataset(name="x", items=(item_old,))
        new = Dataset(name="x", items=(item_new,))
        delta = diff(old, new)
        # The unchanged item comes from `new` so metadata updates are visible.
        assert delta.unchanged == (item_new,)
        assert delta.unchanged[0].metadata == {"tier": "hard"}

    def test_results_sorted_by_id(self) -> None:
        old = Dataset(name="x", items=_items("z"))
        new = Dataset(name="x", items=_items("z", "b", "c", "a"))
        delta = diff(old, new)
        added_ids = [item.id for item in delta.added]
        assert added_ids == sorted(added_ids)

    def test_empty_old_dataset(self) -> None:
        old = Dataset(name="x")
        new = Dataset(name="x", items=_items("a", "b"))
        delta = diff(old, new)
        assert len(delta.added) == 2
        assert delta.removed == ()
        assert delta.unchanged == ()

    def test_empty_new_dataset(self) -> None:
        old = Dataset(name="x", items=_items("a", "b"))
        new = Dataset(name="x")
        delta = diff(old, new)
        assert delta.added == ()
        assert len(delta.removed) == 2
        assert delta.unchanged == ()
