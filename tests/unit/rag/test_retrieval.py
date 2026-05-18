"""Unit tests for `forge.rag.retrieval`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from forge.rag.chunking import Chunk
from forge.rag.retrieval import RetrievalResult, Retriever

if TYPE_CHECKING:
    from collections.abc import Sequence


class TestRetrievalResult:
    def test_basic_construction(self) -> None:
        chunk = Chunk(id="c", text="t")
        result = RetrievalResult(chunk=chunk, score=0.95)
        assert result.chunk is chunk
        assert result.score == 0.95

    def test_is_frozen(self) -> None:
        result = RetrievalResult(chunk=Chunk(id="c", text="t"), score=0.5)
        with pytest.raises(ValidationError, match="frozen"):
            result.score = 0.0  # type: ignore[misc]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalResult(
                chunk=Chunk(id="c", text="t"),
                score=0.5,
                unknown="x",  # type: ignore[call-arg]
            )


class TestRetrieverProtocol:
    def test_duck_typed_retriever_satisfies_protocol(self) -> None:
        class _MyRetriever:
            async def retrieve(self, query: str, *, top_k: int = 5) -> Sequence[RetrievalResult]:
                del query, top_k
                return ()

        assert isinstance(_MyRetriever(), Retriever)

    def test_missing_method_does_not_satisfy(self) -> None:
        class _BadRetriever:
            async def search(self, q: str) -> None:  # wrong method name
                del q

        assert not isinstance(_BadRetriever(), Retriever)
