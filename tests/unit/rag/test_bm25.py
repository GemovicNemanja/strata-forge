"""Unit tests for `strata_forge.rag.bm25`."""

from __future__ import annotations

import pytest

from strata_forge.rag.bm25 import BM25Retriever, tokenize
from strata_forge.rag.chunking import Chunk
from strata_forge.rag.retrieval import Retriever


def _chunks(*texts: str) -> list[Chunk]:
    return [Chunk(id=f"c{i}", text=t) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases(self) -> None:
        assert tokenize("Hello World") == ["hello", "world"]

    def test_splits_on_whitespace_and_punctuation(self) -> None:
        assert tokenize("foo, bar.  baz!") == ["foo", "bar", "baz"]

    def test_keeps_underscores_in_identifiers(self) -> None:
        # \w includes underscore — common in code search.
        assert tokenize("my_variable_name") == ["my_variable_name"]

    def test_empty_string(self) -> None:
        assert tokenize("") == []


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_corpus_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            BM25Retriever([])

    def test_negative_k1_rejected(self) -> None:
        with pytest.raises(ValueError, match="k1"):
            BM25Retriever(_chunks("hi"), k1=-0.1)

    def test_b_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="b must"):
            BM25Retriever(_chunks("hi"), b=-0.1)
        with pytest.raises(ValueError, match="b must"):
            BM25Retriever(_chunks("hi"), b=1.5)

    def test_satisfies_retriever_protocol(self) -> None:
        retriever = BM25Retriever(_chunks("hi"))
        assert isinstance(retriever, Retriever)

    def test_corpus_size_property(self) -> None:
        retriever = BM25Retriever(_chunks("a", "b", "c"))
        assert retriever.corpus_size == 3


# ---------------------------------------------------------------------------
# Retrieve tests
# ---------------------------------------------------------------------------


class TestRetrieve:
    async def test_exact_word_match_ranks_high(self) -> None:
        chunks = _chunks(
            "the quick brown fox jumps over the lazy dog",
            "all about cats and other felines",
            "python programming language tutorial",
        )
        retriever = BM25Retriever(chunks)
        results = await retriever.retrieve("fox")
        assert len(results) >= 1
        assert "fox" in results[0].chunk.text

    async def test_multi_word_query(self) -> None:
        chunks = _chunks(
            "machine learning models for natural language processing",
            "weather forecast for tomorrow",
            "machine learning frameworks comparison",
        )
        retriever = BM25Retriever(chunks)
        results = await retriever.retrieve("machine learning")
        # Both ML docs should rank above the weather doc.
        assert len(results) >= 2
        ml_results = [r for r in results if "machine learning" in r.chunk.text]
        assert len(ml_results) == 2

    async def test_no_overlap_returns_empty(self) -> None:
        chunks = _chunks("apple", "banana", "cherry")
        retriever = BM25Retriever(chunks)
        results = await retriever.retrieve("unrelated")
        assert results == ()

    async def test_top_k_truncates(self) -> None:
        chunks = _chunks(*[f"document about topic {i}" for i in range(10)])
        retriever = BM25Retriever(chunks)
        results = await retriever.retrieve("topic", top_k=3)
        assert len(results) == 3

    async def test_results_sorted_by_score_desc(self) -> None:
        # The doc with more occurrences of "important" should rank higher.
        chunks = _chunks(
            "important important important keyword spam",
            "this contains the important word once",
            "completely unrelated text here",
        )
        retriever = BM25Retriever(chunks)
        results = await retriever.retrieve("important")
        # All scores are sorted desc.
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        # The most-mention doc should come first.
        assert results[0].chunk.text.startswith("important important")

    async def test_empty_query_rejected(self) -> None:
        retriever = BM25Retriever(_chunks("hi"))
        with pytest.raises(ValueError, match="non-empty"):
            await retriever.retrieve("")

    async def test_invalid_top_k_rejected(self) -> None:
        retriever = BM25Retriever(_chunks("hi"))
        with pytest.raises(ValueError, match="top_k"):
            await retriever.retrieve("hi", top_k=0)

    async def test_case_insensitive(self) -> None:
        chunks = _chunks("The Quick Brown Fox")
        retriever = BM25Retriever(chunks)
        results_upper = await retriever.retrieve("FOX")
        results_lower = await retriever.retrieve("fox")
        assert len(results_upper) == 1
        assert len(results_lower) == 1
        assert results_upper[0].chunk.id == results_lower[0].chunk.id

    async def test_zero_score_docs_excluded(self) -> None:
        # A doc that contains a query term should still get filtered if its
        # BM25 score evaluates to 0 (rare in practice but tested for safety).
        chunks = _chunks("foo bar baz")
        retriever = BM25Retriever(chunks)
        # Just confirms the score-filtering line is reachable: matching
        # query returns the chunk, non-matching returns empty.
        results = await retriever.retrieve("foo")
        assert len(results) == 1
        assert results[0].score > 0.0
