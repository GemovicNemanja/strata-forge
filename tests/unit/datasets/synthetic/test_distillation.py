"""Unit tests for `forge.datasets.synthetic.distillation`."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from forge.datasets.schema import Dataset, DatasetItem
from forge.datasets.synthetic.distillation import distill


def _item(
    query: str, *, expected: Any = None, metadata: dict[str, Any] | None = None
) -> DatasetItem:
    return DatasetItem.from_input({"q": query}, expected_output=expected, metadata=metadata)


def _fake_response(text: str) -> Any:
    response = AsyncMock()
    response.text = text
    return response


def _fake_client(*texts: str) -> AsyncMock:
    """Build a mock LLMClient whose `complete` returns each text in order."""
    client = AsyncMock()
    client.complete = AsyncMock(side_effect=[_fake_response(t) for t in texts])
    return client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    async def test_labels_items_with_teacher_output(self) -> None:
        client = _fake_client("answer-a", "answer-b")
        result = await distill(
            inputs=[_item("a"), _item("b")],
            teacher=client,
            name="labeled",
        )
        assert result.name == "labeled"
        assert len(result) == 2
        assert result.items[0].expected_output == "answer-a"
        assert result.items[1].expected_output == "answer-b"

    async def test_accepts_dataset_as_input(self) -> None:
        ds = Dataset(name="src", items=(_item("a"), _item("b")))
        client = _fake_client("A", "B")
        result = await distill(inputs=ds, teacher=client, name="labeled")
        assert len(result) == 2
        assert result.items[0].expected_output == "A"

    async def test_preserves_input_dict(self) -> None:
        client = _fake_client("answer")
        result = await distill(inputs=[_item("hello")], teacher=client, name="x")
        assert result.items[0].input == {"q": "hello"}

    async def test_preserves_item_id(self) -> None:
        item = _item("hello")
        client = _fake_client("answer")
        result = await distill(inputs=[item], teacher=client, name="x")
        assert result.items[0].id == item.id

    async def test_per_item_metadata_merged(self) -> None:
        item = _item("hello", metadata={"tier": "easy"})
        client = _fake_client("answer")
        result = await distill(inputs=[item], teacher=client, name="x")
        # Original metadata preserved, synthetic marker added.
        assert result.items[0].metadata["tier"] == "easy"
        assert result.items[0].metadata["synthetic"] == "distillation"

    async def test_custom_metadata_merged(self) -> None:
        client = _fake_client("answer")
        result = await distill(
            inputs=[_item("hello")],
            teacher=client,
            name="x",
            metadata={"source": "test"},
        )
        assert result.items[0].metadata["synthetic"] == "distillation"
        assert result.items[0].metadata["source"] == "test"


# ---------------------------------------------------------------------------
# skip_existing
# ---------------------------------------------------------------------------


class TestSkipExisting:
    async def test_default_skips_items_with_expected_output(self) -> None:
        client = _fake_client("teacher-answer-for-b")
        items = [
            _item("a", expected="pre-existing"),
            _item("b"),
        ]
        result = await distill(inputs=items, teacher=client, name="x")
        # Only ONE teacher call — the labeled item passed through.
        assert client.complete.call_count == 1
        # The pre-existing label is preserved on the passed-through item.
        recovered_a = next(item for item in result.items if item.input == {"q": "a"})
        assert recovered_a.expected_output == "pre-existing"
        recovered_b = next(item for item in result.items if item.input == {"q": "b"})
        assert recovered_b.expected_output == "teacher-answer-for-b"

    async def test_skip_existing_false_relabels_everything(self) -> None:
        client = _fake_client("fresh-a", "fresh-b")
        items = [
            _item("a", expected="stale"),
            _item("b", expected="also-stale"),
        ]
        result = await distill(inputs=items, teacher=client, name="x", skip_existing=False)
        assert client.complete.call_count == 2
        outputs = sorted(str(item.expected_output) for item in result.items)
        assert outputs == ["fresh-a", "fresh-b"]

    async def test_pre_labeled_item_passes_through_unchanged(self) -> None:
        # When skip_existing=True and an item has a label, the original
        # item is preserved verbatim (no synthetic-marker added).
        item = _item("a", expected="pre", metadata={"tier": "easy"})
        client = _fake_client()  # no responses needed
        result = await distill(inputs=[item], teacher=client, name="x")
        assert result.items[0] is item


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    async def test_respects_concurrency_limit(self) -> None:
        # Track the peak number of in-flight teacher calls.
        peak = 0
        in_flight = 0

        async def slow_complete(**_kwargs: Any) -> Any:
            nonlocal peak, in_flight
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return _fake_response("answer")

        client = AsyncMock()
        client.complete = AsyncMock(side_effect=slow_complete)
        items = [_item(str(i)) for i in range(10)]
        await distill(inputs=items, teacher=client, name="x", concurrency=3)
        assert peak <= 3

    async def test_zero_concurrency_rejected(self) -> None:
        with pytest.raises(ValueError, match="concurrency must be positive"):
            await distill(
                inputs=[_item("a")],
                teacher=_fake_client("x"),
                name="x",
                concurrency=0,
            )


# ---------------------------------------------------------------------------
# Order preservation
# ---------------------------------------------------------------------------


class TestOrder:
    async def test_output_preserves_input_order(self) -> None:
        # Even with concurrency, the output order matches input order.
        client = AsyncMock()
        client.complete = AsyncMock(
            side_effect=[
                _fake_response("ans-0"),
                _fake_response("ans-1"),
                _fake_response("ans-2"),
            ]
        )
        items = [_item("zero"), _item("one"), _item("two")]
        result = await distill(inputs=items, teacher=client, name="x", concurrency=10)
        # Order in result matches order in input.
        recovered = [item.input["q"] for item in result.items]
        assert recovered == ["zero", "one", "two"]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    async def test_input_dict_serialized_to_user_message(self) -> None:
        client = _fake_client("answer")
        await distill(inputs=[_item("specific-query")], teacher=client, name="x")
        call_args = client.complete.call_args
        messages = call_args.kwargs["messages"]
        # System + user.
        assert len(messages) == 2
        user_msg = messages[1]
        assert "specific-query" in user_msg.content

    async def test_custom_system_prompt(self) -> None:
        client = _fake_client("answer")
        await distill(
            inputs=[_item("a")],
            teacher=client,
            name="x",
            system_prompt="my-custom-prompt",
        )
        messages = client.complete.call_args.kwargs["messages"]
        assert messages[0].content == "my-custom-prompt"

    async def test_temperature_passed_through(self) -> None:
        client = _fake_client("answer")
        await distill(
            inputs=[_item("a")],
            teacher=client,
            name="x",
            temperature=0.1,
        )
        call_kwargs = client.complete.call_args.kwargs
        assert call_kwargs["temperature"] == 0.1


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    async def test_empty_input_returns_empty_dataset(self) -> None:
        client = _fake_client()
        result = await distill(inputs=[], teacher=client, name="x")
        assert len(result) == 0
        assert result.name == "x"


# ---------------------------------------------------------------------------
# Output metadata
# ---------------------------------------------------------------------------


class TestOutputMetadata:
    async def test_output_dataset_metadata(self) -> None:
        client = _fake_client("a", "b", "c")
        result = await distill(
            inputs=[_item("1"), _item("2"), _item("3")],
            teacher=client,
            name="x",
            description="distilled set",
        )
        assert result.metadata["synthetic"] == "distillation"
        assert result.metadata["source_count"] == 3
        assert result.description == "distilled set"
