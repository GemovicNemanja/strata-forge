"""Unit tests for `forge.datasets.synthetic.self_instruct`."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from forge.datasets.schema import Dataset, DatasetItem
from forge.datasets.synthetic.self_instruct import (
    SelfInstructBatch,
    SelfInstructItem,
    self_instruct,
)


def _seed(query: str, *, expected: Any = None) -> DatasetItem:
    return DatasetItem.from_input({"q": query}, expected_output=expected)


def _fake_response(items: list[dict[str, Any]]) -> Any:
    """Build a mock object that matches `StructuredResponse[SelfInstructBatch]`."""
    parsed = SelfInstructBatch(
        items=[SelfInstructItem(**raw) for raw in items],
    )
    response = AsyncMock()
    response.parsed = parsed
    return response


def _fake_client(*responses: Any) -> AsyncMock:
    """Build a mock LLMClient whose `complete_structured` returns each response in order."""
    client = AsyncMock()
    client.complete_structured = AsyncMock(side_effect=responses)
    return client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    async def test_generates_requested_number_of_items(self) -> None:
        client = _fake_client(
            _fake_response(
                [
                    {"input": {"q": "x"}, "expected_output": "X"},
                    {"input": {"q": "y"}, "expected_output": "Y"},
                    {"input": {"q": "z"}, "expected_output": "Z"},
                ]
            )
        )
        ds = await self_instruct(
            seeds=[_seed("a", expected="A")],
            instructions="generate uppercase variants",
            n=3,
            client=client,
            name="generated",
            batch_size=3,
        )
        assert ds.name == "generated"
        assert len(ds) == 3
        assert {item.input["q"] for item in ds.items} == {"x", "y", "z"}

    async def test_generated_items_have_content_hash_ids(self) -> None:
        client = _fake_client(_fake_response([{"input": {"q": "x"}, "expected_output": "X"}]))
        ds = await self_instruct(
            seeds=[_seed("a")],
            instructions="x",
            n=1,
            client=client,
            name="generated",
        )
        # SHA-256 hex is 64 chars.
        assert len(ds.items[0].id) == 64

    async def test_generated_items_have_synthetic_metadata(self) -> None:
        client = _fake_client(_fake_response([{"input": {"q": "x"}, "expected_output": "X"}]))
        ds = await self_instruct(
            seeds=[_seed("a")],
            instructions="x",
            n=1,
            client=client,
            name="generated",
        )
        assert ds.items[0].metadata == {"synthetic": "self_instruct"}

    async def test_custom_metadata_overrides_default(self) -> None:
        client = _fake_client(_fake_response([{"input": {"q": "x"}, "expected_output": "X"}]))
        ds = await self_instruct(
            seeds=[_seed("a")],
            instructions="x",
            n=1,
            client=client,
            name="generated",
            metadata={"source": "test", "tier": "easy"},
        )
        assert ds.items[0].metadata == {"source": "test", "tier": "easy"}

    async def test_dataset_seeds_accepted_directly(self) -> None:
        seeds = Dataset(name="seeds", items=(_seed("a", expected="A"),))
        client = _fake_client(_fake_response([{"input": {"q": "x"}, "expected_output": "X"}]))
        ds = await self_instruct(
            seeds=seeds,
            instructions="x",
            n=1,
            client=client,
            name="generated",
        )
        assert len(ds) == 1


# ---------------------------------------------------------------------------
# Batching across multiple calls
# ---------------------------------------------------------------------------


class TestBatching:
    async def test_makes_multiple_calls_to_reach_n(self) -> None:
        # batch_size=2 with n=4 must issue 2 LLM calls.
        client = _fake_client(
            _fake_response(
                [
                    {"input": {"q": "x1"}, "expected_output": "X1"},
                    {"input": {"q": "x2"}, "expected_output": "X2"},
                ]
            ),
            _fake_response(
                [
                    {"input": {"q": "x3"}, "expected_output": "X3"},
                    {"input": {"q": "x4"}, "expected_output": "X4"},
                ]
            ),
        )
        ds = await self_instruct(
            seeds=[_seed("a")],
            instructions="x",
            n=4,
            client=client,
            name="generated",
            batch_size=2,
        )
        assert len(ds) == 4
        assert client.complete_structured.call_count == 2

    async def test_batch_size_asks_for_remaining_only(self) -> None:
        # n=3 with batch_size=10 still only needs one call asking for 3.
        client = _fake_client(
            _fake_response(
                [
                    {"input": {"q": "x"}, "expected_output": "X"},
                    {"input": {"q": "y"}, "expected_output": "Y"},
                    {"input": {"q": "z"}, "expected_output": "Z"},
                ]
            )
        )
        ds = await self_instruct(
            seeds=[_seed("a")],
            instructions="x",
            n=3,
            client=client,
            name="generated",
            batch_size=10,
        )
        assert len(ds) == 3
        assert client.complete_structured.call_count == 1


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDedup:
    async def test_duplicate_items_within_batch_dropped(self) -> None:
        # The LLM returns the same item twice; we keep only one.
        client = _fake_client(
            _fake_response(
                [
                    {"input": {"q": "x"}, "expected_output": "X"},
                    {"input": {"q": "x"}, "expected_output": "X"},
                    {"input": {"q": "y"}, "expected_output": "Y"},
                ]
            )
        )
        ds = await self_instruct(
            seeds=[_seed("a")],
            instructions="x",
            n=2,
            client=client,
            name="generated",
            batch_size=3,
        )
        assert len(ds) == 2
        assert {item.input["q"] for item in ds.items} == {"x", "y"}

    async def test_duplicates_across_batches_dropped(self) -> None:
        client = _fake_client(
            _fake_response([{"input": {"q": "x"}, "expected_output": "X"}]),
            _fake_response(
                [
                    {"input": {"q": "x"}, "expected_output": "X"},
                    {"input": {"q": "y"}, "expected_output": "Y"},
                ]
            ),
        )
        ds = await self_instruct(
            seeds=[_seed("a")],
            instructions="x",
            n=2,
            client=client,
            name="generated",
            batch_size=2,
        )
        assert len(ds) == 2

    async def test_items_matching_seed_ids_dropped(self) -> None:
        # An item the LLM emits that matches a seed ID is filtered.
        client = _fake_client(
            _fake_response(
                [
                    {"input": {"q": "a"}, "expected_output": None},  # same as seed
                    {"input": {"q": "y"}, "expected_output": "Y"},
                ]
            )
        )
        ds = await self_instruct(
            seeds=[_seed("a")],
            instructions="x",
            n=1,
            client=client,
            name="generated",
            batch_size=2,
        )
        # The seed-matching item is dropped, leaving 'y'.
        assert len(ds) == 1
        assert ds.items[0].input["q"] == "y"


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


class TestTermination:
    async def test_terminates_early_when_no_new_items_in_batch(self) -> None:
        # All items in this batch are dupes of seed → batch yields 0 → stop.
        client = _fake_client(
            _fake_response(
                [
                    {"input": {"q": "a"}, "expected_output": None},
                    {"input": {"q": "a"}, "expected_output": None},
                ]
            )
        )
        ds = await self_instruct(
            seeds=[_seed("a")],
            instructions="x",
            n=5,
            client=client,
            name="generated",
            batch_size=2,
            max_attempts=10,
        )
        # No novel items were produced; the dataset comes back empty.
        assert len(ds) == 0
        # Only ONE call was made before bailing.
        assert client.complete_structured.call_count == 1

    async def test_terminates_at_max_attempts(self) -> None:
        # Each batch produces one new item. With n=5 and max_attempts=2 we
        # only get 2 items.
        client = _fake_client(
            _fake_response([{"input": {"q": "x"}, "expected_output": "X"}]),
            _fake_response([{"input": {"q": "y"}, "expected_output": "Y"}]),
        )
        ds = await self_instruct(
            seeds=[_seed("a")],
            instructions="x",
            n=5,
            client=client,
            name="generated",
            batch_size=1,
            max_attempts=2,
        )
        assert len(ds) == 2

    async def test_partial_result_returned_when_truncated(self) -> None:
        # n=10 but only 3 unique items ever emitted before max_attempts kicks in.
        client = _fake_client(
            _fake_response(
                [
                    {"input": {"q": "x"}, "expected_output": "X"},
                    {"input": {"q": "y"}, "expected_output": "Y"},
                    {"input": {"q": "z"}, "expected_output": "Z"},
                ]
            ),
        )
        ds = await self_instruct(
            seeds=[_seed("a")],
            instructions="x",
            n=10,
            client=client,
            name="generated",
            batch_size=3,
            max_attempts=1,
        )
        assert len(ds) == 3


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    async def test_zero_n_rejected(self) -> None:
        with pytest.raises(ValueError, match="n must be positive"):
            await self_instruct(
                seeds=[_seed("a")],
                instructions="x",
                n=0,
                client=_fake_client(),
                name="x",
            )

    async def test_negative_n_rejected(self) -> None:
        with pytest.raises(ValueError, match="n must be positive"):
            await self_instruct(
                seeds=[_seed("a")],
                instructions="x",
                n=-1,
                client=_fake_client(),
                name="x",
            )

    async def test_zero_batch_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            await self_instruct(
                seeds=[_seed("a")],
                instructions="x",
                n=5,
                batch_size=0,
                client=_fake_client(),
                name="x",
            )

    async def test_empty_seeds_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            await self_instruct(
                seeds=[],
                instructions="x",
                n=1,
                client=_fake_client(),
                name="x",
            )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    async def test_seeds_appear_in_user_message(self) -> None:
        client = _fake_client(_fake_response([{"input": {"q": "x"}, "expected_output": "X"}]))
        await self_instruct(
            seeds=[_seed("seed-query", expected="seed-answer")],
            instructions="my-special-instructions",
            n=1,
            client=client,
            name="x",
        )
        call_args = client.complete_structured.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]
        assert "seed-query" in user_msg.content
        assert "seed-answer" in user_msg.content
        assert "my-special-instructions" in user_msg.content

    async def test_temperature_passed_to_client(self) -> None:
        client = _fake_client(_fake_response([{"input": {"q": "x"}, "expected_output": "X"}]))
        await self_instruct(
            seeds=[_seed("a")],
            instructions="x",
            n=1,
            client=client,
            name="x",
            temperature=0.42,
        )
        call_kwargs = client.complete_structured.call_args.kwargs
        assert call_kwargs["temperature"] == 0.42

    async def test_schema_is_self_instruct_batch(self) -> None:
        client = _fake_client(_fake_response([{"input": {"q": "x"}, "expected_output": "X"}]))
        await self_instruct(
            seeds=[_seed("a")],
            instructions="x",
            n=1,
            client=client,
            name="x",
        )
        call_kwargs = client.complete_structured.call_args.kwargs
        assert call_kwargs["schema"] is SelfInstructBatch


# ---------------------------------------------------------------------------
# Output dataset shape
# ---------------------------------------------------------------------------


class TestOutputShape:
    async def test_output_dataset_has_synthetic_marker_metadata(self) -> None:
        client = _fake_client(_fake_response([{"input": {"q": "x"}, "expected_output": "X"}]))
        ds = await self_instruct(
            seeds=[_seed("a")],
            instructions="x",
            n=1,
            client=client,
            name="x",
        )
        assert ds.metadata["synthetic"] == "self_instruct"
        assert ds.metadata["seed_count"] == 1
