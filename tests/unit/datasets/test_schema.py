"""Unit tests for `forge.datasets.schema`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from forge.core.errors import ValidationError
from forge.core.repro import content_hash
from forge.datasets.schema import Dataset, DatasetItem

# ---------------------------------------------------------------------------
# DatasetItem
# ---------------------------------------------------------------------------


class TestDatasetItem:
    def test_direct_construction(self) -> None:
        item = DatasetItem(
            id="custom-id",
            input={"query": "what is x?"},
            expected_output="x is x",
            metadata={"tier": "easy"},
        )
        assert item.id == "custom-id"
        assert item.input == {"query": "what is x?"}
        assert item.expected_output == "x is x"
        assert item.metadata == {"tier": "easy"}

    def test_from_input_derives_content_hash_id(self) -> None:
        item = DatasetItem.from_input(
            {"query": "hi"}, expected_output="hello"
        )
        expected = content_hash(
            {"input": {"query": "hi"}, "expected_output": "hello"}
        )
        assert item.id == expected
        assert len(item.id) == 64

    def test_from_input_same_content_same_id(self) -> None:
        a = DatasetItem.from_input({"q": "x"}, expected_output="y")
        b = DatasetItem.from_input({"q": "x"}, expected_output="y")
        assert a.id == b.id

    def test_from_input_different_content_different_id(self) -> None:
        a = DatasetItem.from_input({"q": "x"}, expected_output="y")
        b = DatasetItem.from_input({"q": "x"}, expected_output="z")
        assert a.id != b.id

    def test_from_input_metadata_does_not_affect_id(self) -> None:
        # Re-tagging an item with new metadata shouldn't invalidate its ID.
        a = DatasetItem.from_input({"q": "x"}, metadata={"tier": "easy"})
        b = DatasetItem.from_input({"q": "x"}, metadata={"tier": "hard"})
        assert a.id == b.id

    def test_from_input_omits_expected_output(self) -> None:
        item = DatasetItem.from_input({"q": "x"})
        assert item.expected_output is None
        assert item.metadata == {}

    def test_is_frozen(self) -> None:
        item = DatasetItem(id="x", input={"q": "y"})
        with pytest.raises(PydanticValidationError, match="frozen"):
            item.id = "other"  # type: ignore[misc]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            DatasetItem(
                id="x",
                input={"q": "y"},
                unknown="oops",  # type: ignore[call-arg]
            )

    def test_metadata_default_independent_across_instances(self) -> None:
        a = DatasetItem(id="x", input={"q": "y"})
        b = DatasetItem(id="z", input={"q": "y"})
        # Even though both default to {}, they're separate objects so a
        # mutation can't bleed.
        assert a.metadata is not b.metadata


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TestDataset:
    def test_construction_with_items(self) -> None:
        items = (
            DatasetItem.from_input({"q": "1"}),
            DatasetItem.from_input({"q": "2"}),
        )
        ds = Dataset(name="train", items=items, description="A small set.")
        assert ds.name == "train"
        assert ds.items == items
        assert ds.description == "A small set."

    def test_empty_dataset_is_valid(self) -> None:
        # Empty datasets are useful for "I'll fill this in" scenarios.
        ds = Dataset(name="empty")
        assert len(ds) == 0
        assert ds.items == ()

    def test_len(self) -> None:
        items = tuple(DatasetItem.from_input({"q": str(i)}) for i in range(5))
        assert len(Dataset(name="five", items=items)) == 5

    def test_by_id_finds_match(self) -> None:
        items = (
            DatasetItem(id="a", input={"q": "1"}),
            DatasetItem(id="b", input={"q": "2"}),
        )
        ds = Dataset(name="x", items=items)
        assert ds.by_id("a") is items[0]
        assert ds.by_id("b") is items[1]

    def test_by_id_returns_none_for_missing(self) -> None:
        ds = Dataset(name="x", items=(DatasetItem(id="a", input={"q": "1"}),))
        assert ds.by_id("missing") is None

    def test_name_must_be_non_empty(self) -> None:
        with pytest.raises(PydanticValidationError):
            Dataset(name="")

    def test_is_frozen(self) -> None:
        ds = Dataset(name="x")
        with pytest.raises(PydanticValidationError, match="frozen"):
            ds.name = "other"  # type: ignore[misc]

    def test_items_are_tuple_not_list(self) -> None:
        ds = Dataset(
            name="x", items=(DatasetItem(id="a", input={"q": "1"}),)
        )
        # The tuple type prevents accidental mutation via .append etc.
        assert isinstance(ds.items, tuple)

    def test_duplicate_item_ids_rejected(self) -> None:
        items = (
            DatasetItem(id="dup", input={"q": "1"}),
            DatasetItem(id="dup", input={"q": "2"}),
        )
        with pytest.raises(ValidationError, match="duplicate item id"):
            Dataset(name="bad", items=items)

    def test_duplicate_error_names_dataset_and_position(self) -> None:
        items = (
            DatasetItem(id="a", input={"q": "1"}),
            DatasetItem(id="b", input={"q": "2"}),
            DatasetItem(id="a", input={"q": "3"}),  # collision at position 2
        )
        with pytest.raises(ValidationError) as info:
            Dataset(name="specific-name", items=items)
        msg = str(info.value)
        assert "specific-name" in msg
        assert "'a'" in msg
        assert "position 2" in msg

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Dataset(name="x", unknown="oops")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Integration shape — round-trip through model_dump
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_model_dump_and_back(self) -> None:
        original = Dataset(
            name="x",
            items=(
                DatasetItem.from_input({"q": "1"}, expected_output="one"),
                DatasetItem.from_input({"q": "2"}, expected_output="two"),
            ),
            description="round-trip test",
            metadata={"source": "synthetic"},
        )
        payload = original.model_dump()
        rebuilt = Dataset.model_validate(payload)
        assert rebuilt == original

    def test_complex_input_values(self) -> None:
        item = DatasetItem.from_input(
            {
                "messages": [
                    {"role": "system", "content": "you are helpful"},
                    {"role": "user", "content": "hi"},
                ],
                "max_tokens": 100,
            },
            expected_output={"role": "assistant", "content": "hello"},
        )
        # JSON-shaped nested values round-trip cleanly.
        assert item.input["messages"][0]["role"] == "system"
        assert isinstance(item.expected_output, dict)


# Silence the type-checker on the `Any` import (used implicitly by callers).
_: Any = None
