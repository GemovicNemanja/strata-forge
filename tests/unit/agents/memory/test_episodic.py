"""Unit tests for `strata_forge.agents.memory.episodic`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from strata_forge.agents.memory.episodic import EmbedFn, EpisodicMemory
from strata_forge.agents.memory.vector_store import InMemoryVectorStore

if TYPE_CHECKING:
    from collections.abc import Sequence


def _embed_factory(
    mapping: dict[str, list[float]],
) -> EmbedFn:
    """Build an async embed function that returns mapping[text]."""

    async def _embed(text: str) -> Sequence[float]:
        if text in mapping:
            return mapping[text]
        # Default fallback for tests that don't pre-register a vector.
        return [float(len(text)), 0.0]

    return _embed


# ---------------------------------------------------------------------------
# Construction + add
# ---------------------------------------------------------------------------


class TestAdd:
    async def test_add_returns_assigned_id(self) -> None:
        store = InMemoryVectorStore()
        memory = EpisodicMemory(store=store, embed=_embed_factory({}))
        item_id = await memory.add("hello world")
        assert isinstance(item_id, str)
        assert len(item_id) > 0

    async def test_add_with_explicit_id(self) -> None:
        store = InMemoryVectorStore()
        memory = EpisodicMemory(store=store, embed=_embed_factory({}))
        result_id = await memory.add("hello", item_id="my-id")
        assert result_id == "my-id"

    async def test_add_stores_in_backend(self) -> None:
        store = InMemoryVectorStore()
        memory = EpisodicMemory(
            store=store,
            embed=_embed_factory({"hello": [1.0, 0.0]}),
        )
        await memory.add("hello", item_id="x")
        # Search the backing store directly to confirm the entry.
        results = await store.search([1.0, 0.0])
        assert len(results) == 1
        assert results[0].item.id == "x"
        assert results[0].item.text == "hello"

    async def test_add_with_metadata(self) -> None:
        store = InMemoryVectorStore()
        memory = EpisodicMemory(store=store, embed=_embed_factory({}))
        await memory.add("note", item_id="x", metadata={"source": "user"})
        results = await store.search([4.0, 0.0])  # "note" = 4 chars
        assert results[0].item.metadata == {"source": "user"}

    async def test_empty_text_rejected(self) -> None:
        memory = EpisodicMemory(
            store=InMemoryVectorStore(),
            embed=_embed_factory({}),
        )
        with pytest.raises(ValueError, match="non-empty"):
            await memory.add("")

    async def test_explicit_id_overwrites(self) -> None:
        store = InMemoryVectorStore()
        memory = EpisodicMemory(
            store=store,
            embed=_embed_factory({"v1": [1.0], "v2": [0.0]}),
        )
        await memory.add("v1", item_id="x")
        await memory.add("v2", item_id="x")
        results = await store.search([0.0])
        assert results[0].item.text == "v2"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    async def test_returns_top_k_results(self) -> None:
        store = InMemoryVectorStore()
        embed = _embed_factory(
            {
                "cats": [1.0, 0.0],
                "dogs": [0.9, 0.1],
                "fish": [0.0, 1.0],
                "query-cats": [1.0, 0.0],
            }
        )
        memory = EpisodicMemory(store=store, embed=embed)
        await memory.add("cats")
        await memory.add("dogs")
        await memory.add("fish")
        results = await memory.search("query-cats", top_k=2)
        # cats first (identical embedding), dogs second.
        assert len(results) == 2
        assert results[0].item.text == "cats"
        assert results[1].item.text == "dogs"

    async def test_empty_query_rejected(self) -> None:
        memory = EpisodicMemory(
            store=InMemoryVectorStore(),
            embed=_embed_factory({}),
        )
        with pytest.raises(ValueError, match="non-empty"):
            await memory.search("")

    async def test_invalid_top_k_rejected(self) -> None:
        memory = EpisodicMemory(
            store=InMemoryVectorStore(),
            embed=_embed_factory({}),
        )
        with pytest.raises(ValueError, match="top_k"):
            await memory.search("query", top_k=0)

    async def test_search_on_empty_store(self) -> None:
        memory = EpisodicMemory(
            store=InMemoryVectorStore(),
            embed=_embed_factory({}),
        )
        results = await memory.search("anything")
        assert results == ()


# ---------------------------------------------------------------------------
# delete + clear
# ---------------------------------------------------------------------------


class TestDeleteClear:
    async def test_delete_removes_items(self) -> None:
        store = InMemoryVectorStore()
        memory = EpisodicMemory(store=store, embed=_embed_factory({}))
        await memory.add("a", item_id="a")
        await memory.add("b", item_id="b")
        await memory.delete(["a"])
        assert len(store) == 1

    async def test_clear_empties_store(self) -> None:
        store = InMemoryVectorStore()
        memory = EpisodicMemory(store=store, embed=_embed_factory({}))
        for i in range(5):
            await memory.add(f"item-{i}")
        await memory.clear()
        assert len(store) == 0
