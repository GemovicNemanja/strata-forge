"""Unit tests for `forge.rag.vector_store`."""

from __future__ import annotations

import pytest

from forge.rag.vector_store import (
    InMemoryVectorStore,
    VectorItem,
    VectorStore,
    cosine_similarity,
)


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert abs(cosine_similarity([1.0, 2.0], [1.0, 2.0]) - 1.0) < 1e-9

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            cosine_similarity([1.0], [1.0, 2.0])


class TestProtocol:
    def test_in_memory_store_satisfies_protocol(self) -> None:
        store = InMemoryVectorStore()
        assert isinstance(store, VectorStore)


class TestInMemoryVectorStore:
    async def test_empty_store_returns_empty_search(self) -> None:
        store = InMemoryVectorStore()
        assert await store.search([1.0, 0.0]) == ()

    async def test_add_and_search(self) -> None:
        store = InMemoryVectorStore()
        await store.add(
            [
                VectorItem(id="a", embedding=(1.0, 0.0), text="x"),
                VectorItem(id="b", embedding=(0.0, 1.0), text="y"),
            ]
        )
        results = await store.search([1.0, 0.05])
        assert len(results) == 2
        assert results[0].item.id == "a"

    async def test_top_k_limits_results(self) -> None:
        store = InMemoryVectorStore()
        for i in range(10):
            await store.add([VectorItem(id=str(i), embedding=(float(i),), text=str(i))])
        assert len(await store.search([5.0], top_k=3)) == 3

    async def test_overwrites_existing_id(self) -> None:
        store = InMemoryVectorStore()
        await store.add([VectorItem(id="x", embedding=(1.0,), text="v1")])
        await store.add([VectorItem(id="x", embedding=(1.0,), text="v2")])
        assert len(store) == 1
        results = await store.search([1.0])
        assert results[0].item.text == "v2"

    async def test_delete(self) -> None:
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
        await store.delete(["nonexistent"])  # no error
        assert len(store) == 0

    async def test_clear(self) -> None:
        store = InMemoryVectorStore()
        await store.add([VectorItem(id="x", embedding=(1.0,), text="t")])
        await store.clear()
        assert len(store) == 0

    async def test_invalid_top_k(self) -> None:
        store = InMemoryVectorStore()
        await store.add([VectorItem(id="x", embedding=(1.0,), text="t")])
        with pytest.raises(ValueError, match="top_k"):
            await store.search([1.0], top_k=0)


class TestAgentsReExport:
    """Confirm the relocation didn't break the agents-side import path."""

    def test_agents_re_exports_same_objects(self) -> None:
        from forge.agents.memory.vector_store import InMemoryVectorStore as AgentsIMVS
        from forge.agents.memory.vector_store import VectorStore as AgentsVS

        assert AgentsVS is VectorStore
        assert AgentsIMVS is InMemoryVectorStore
