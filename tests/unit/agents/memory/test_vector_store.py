"""Unit tests for `strata_forge.agents.memory.vector_store`."""

from __future__ import annotations

import pytest

from strata_forge.agents.memory.vector_store import (
    InMemoryVectorStore,
    VectorItem,
    VectorStore,
    _cosine_similarity,  # pyright: ignore[reportPrivateUsage]
)

# ---------------------------------------------------------------------------
# VectorItem
# ---------------------------------------------------------------------------


class TestVectorItem:
    def test_basic_construction(self) -> None:
        item = VectorItem(id="x", embedding=(1.0, 0.0), text="hi")
        assert item.id == "x"
        assert item.embedding == (1.0, 0.0)
        assert item.text == "hi"
        assert item.metadata == {}

    def test_metadata_optional(self) -> None:
        item = VectorItem(id="x", embedding=(1.0,), text="hi", metadata={"k": "v"})
        assert item.metadata == {"k": "v"}

    def test_is_frozen(self) -> None:
        item = VectorItem(id="x", embedding=(1.0,), text="hi")
        with pytest.raises(AttributeError):
            item.id = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert abs(_cosine_similarity([1.0, 2.0], [1.0, 2.0]) - 1.0) < 1e-9

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            _cosine_similarity([1.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# InMemoryVectorStore — Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_satisfies_protocol_at_runtime(self) -> None:
        # runtime_checkable Protocol — isinstance checks should succeed.
        store = InMemoryVectorStore()
        assert isinstance(store, VectorStore)


# ---------------------------------------------------------------------------
# InMemoryVectorStore — behaviour
# ---------------------------------------------------------------------------


class TestInMemoryVectorStore:
    async def test_empty_store_returns_empty_search(self) -> None:
        store = InMemoryVectorStore()
        results = await store.search([1.0, 0.0])
        assert results == ()

    async def test_add_and_search(self) -> None:
        store = InMemoryVectorStore()
        await store.add(
            [
                VectorItem(id="a", embedding=(1.0, 0.0), text="aligned-x"),
                VectorItem(id="b", embedding=(0.0, 1.0), text="aligned-y"),
            ]
        )
        # Query close to (1, 0) returns "a" first.
        results = await store.search([1.0, 0.1])
        assert len(results) == 2
        assert results[0].item.id == "a"
        assert results[0].score > results[1].score

    async def test_top_k_limits_results(self) -> None:
        store = InMemoryVectorStore()
        await store.add(
            [VectorItem(id=str(i), embedding=(float(i), 0.0), text=str(i)) for i in range(10)]
        )
        results = await store.search([5.0, 0.0], top_k=3)
        assert len(results) == 3

    async def test_overwrites_existing_id(self) -> None:
        store = InMemoryVectorStore()
        await store.add([VectorItem(id="x", embedding=(1.0, 0.0), text="v1")])
        await store.add([VectorItem(id="x", embedding=(0.0, 1.0), text="v2")])
        assert len(store) == 1
        results = await store.search([0.0, 1.0])
        assert results[0].item.text == "v2"

    async def test_delete_removes_items(self) -> None:
        store = InMemoryVectorStore()
        await store.add(
            [
                VectorItem(id="a", embedding=(1.0,), text="x"),
                VectorItem(id="b", embedding=(0.0,), text="y"),
            ]
        )
        await store.delete(["a"])
        assert len(store) == 1
        results = await store.search([1.0])
        assert all(r.item.id != "a" for r in results)

    async def test_delete_missing_id_silent(self) -> None:
        store = InMemoryVectorStore()
        await store.add([VectorItem(id="a", embedding=(1.0,), text="x")])
        await store.delete(["nonexistent"])  # must not raise
        assert len(store) == 1

    async def test_clear_empties_store(self) -> None:
        store = InMemoryVectorStore()
        await store.add(
            [VectorItem(id=str(i), embedding=(float(i),), text=str(i)) for i in range(5)]
        )
        await store.clear()
        assert len(store) == 0

    async def test_invalid_top_k_rejected(self) -> None:
        store = InMemoryVectorStore()
        await store.add([VectorItem(id="x", embedding=(1.0,), text="x")])
        with pytest.raises(ValueError, match="top_k"):
            await store.search([1.0], top_k=0)
        with pytest.raises(ValueError, match="top_k"):
            await store.search([1.0], top_k=-1)
