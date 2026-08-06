"""Unit tests for `strata_forge.rag.hybrid`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from strata_forge.rag.chunking import Chunk
from strata_forge.rag.hybrid import HybridRetriever
from strata_forge.rag.retrieval import RetrievalResult, Retriever

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FixedResultsRetriever:
    """Test retriever that returns a fixed list of results."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, query: str, *, top_k: int = 5) -> Sequence[RetrievalResult]:
        self.calls.append((query, top_k))
        return tuple(self._results[:top_k])


def _result(chunk_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(chunk=Chunk(id=chunk_id, text=f"text for {chunk_id}"), score=score)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_single_retriever_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            HybridRetriever([_FixedResultsRetriever([])])

    def test_empty_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            HybridRetriever([])

    def test_mismatched_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="weights length"):
            HybridRetriever(
                [_FixedResultsRetriever([]), _FixedResultsRetriever([])],
                weights=[1.0, 2.0, 3.0],
            )

    def test_invalid_rrf_k(self) -> None:
        with pytest.raises(ValueError, match="rrf_k"):
            HybridRetriever(
                [_FixedResultsRetriever([]), _FixedResultsRetriever([])],
                rrf_k=0,
            )

    def test_invalid_per_retriever_top_k(self) -> None:
        with pytest.raises(ValueError, match="per_retriever_top_k"):
            HybridRetriever(
                [_FixedResultsRetriever([]), _FixedResultsRetriever([])],
                per_retriever_top_k=0,
            )

    def test_satisfies_retriever_protocol(self) -> None:
        retriever = HybridRetriever([_FixedResultsRetriever([]), _FixedResultsRetriever([])])
        assert isinstance(retriever, Retriever)

    def test_default_weights_uniform(self) -> None:
        retriever = HybridRetriever([_FixedResultsRetriever([]), _FixedResultsRetriever([])])
        assert retriever.weights == (1.0, 1.0)


# ---------------------------------------------------------------------------
# Retrieve behaviour
# ---------------------------------------------------------------------------


class TestRetrieve:
    async def test_fuses_results_from_both_retrievers(self) -> None:
        # Retriever A puts "a" first; B puts "b" first; both also return
        # "shared" lower in their list. Fusion should rank "shared" above
        # "c" and "d" because it appears in both lists.
        retriever_a = _FixedResultsRetriever([_result("a"), _result("shared"), _result("c")])
        retriever_b = _FixedResultsRetriever([_result("b"), _result("shared"), _result("d")])
        hybrid = HybridRetriever([retriever_a, retriever_b])
        results = await hybrid.retrieve("query", top_k=10)
        # All five distinct chunks present.
        ids = [r.chunk.id for r in results]
        assert set(ids) == {"a", "b", "c", "d", "shared"}
        # "shared" should be in the top three (appears in both lists).
        assert "shared" in ids[:3]

    async def test_top_k_limits_output(self) -> None:
        retriever_a = _FixedResultsRetriever([_result(f"a{i}") for i in range(10)])
        retriever_b = _FixedResultsRetriever([_result(f"b{i}") for i in range(10)])
        hybrid = HybridRetriever([retriever_a, retriever_b])
        results = await hybrid.retrieve("q", top_k=3)
        assert len(results) == 3

    async def test_runs_retrievers_concurrently_with_same_query(self) -> None:
        # The hybrid retriever forwards (query, top_k) to each child;
        # we confirm both children saw the same call.
        retriever_a = _FixedResultsRetriever([_result("a")])
        retriever_b = _FixedResultsRetriever([_result("b")])
        hybrid = HybridRetriever([retriever_a, retriever_b])
        await hybrid.retrieve("my query", top_k=5)
        assert retriever_a.calls == [("my query", 5)]
        assert retriever_b.calls == [("my query", 5)]

    async def test_per_retriever_top_k_overrides_caller_top_k(self) -> None:
        retriever_a = _FixedResultsRetriever([_result("a")])
        retriever_b = _FixedResultsRetriever([_result("b")])
        hybrid = HybridRetriever([retriever_a, retriever_b], per_retriever_top_k=20)
        await hybrid.retrieve("q", top_k=3)
        # Each retriever was asked for 20 candidates; fusion truncates to 3.
        assert retriever_a.calls == [("q", 20)]
        assert retriever_b.calls == [("q", 20)]

    async def test_weights_bias_fusion(self) -> None:
        # Retriever A is weighted 10x; "a" should win over "b" even though
        # both appear at rank 0 in their respective lists.
        retriever_a = _FixedResultsRetriever([_result("a")])
        retriever_b = _FixedResultsRetriever([_result("b")])
        hybrid = HybridRetriever([retriever_a, retriever_b], weights=[10.0, 1.0])
        results = await hybrid.retrieve("q")
        assert results[0].chunk.id == "a"

    async def test_empty_query_rejected(self) -> None:
        hybrid = HybridRetriever([_FixedResultsRetriever([]), _FixedResultsRetriever([])])
        with pytest.raises(ValueError, match="non-empty"):
            await hybrid.retrieve("")

    async def test_invalid_top_k_rejected(self) -> None:
        hybrid = HybridRetriever([_FixedResultsRetriever([]), _FixedResultsRetriever([])])
        with pytest.raises(ValueError, match="top_k"):
            await hybrid.retrieve("q", top_k=0)

    async def test_all_retrievers_return_empty(self) -> None:
        hybrid = HybridRetriever([_FixedResultsRetriever([]), _FixedResultsRetriever([])])
        results = await hybrid.retrieve("q")
        assert results == ()

    async def test_first_chunk_metadata_kept_when_id_duplicates(self) -> None:
        # When both retrievers return the same chunk id, the fused result
        # uses the first one's Chunk (we don't try to merge metadata).
        first = Chunk(id="x", text="first text", metadata={"source": "a"})
        second = Chunk(id="x", text="second text", metadata={"source": "b"})
        retriever_a = _FixedResultsRetriever([RetrievalResult(chunk=first, score=1.0)])
        retriever_b = _FixedResultsRetriever([RetrievalResult(chunk=second, score=1.0)])
        hybrid = HybridRetriever([retriever_a, retriever_b])
        results = await hybrid.retrieve("q")
        assert len(results) == 1
        # The result carries the first chunk (from retriever_a).
        assert results[0].chunk.text == "first text"
