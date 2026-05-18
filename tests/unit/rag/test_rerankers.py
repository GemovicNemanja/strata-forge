"""Unit tests for `forge.rag.rerankers`."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.rag.chunking import Chunk
from forge.rag.rerankers import (
    CohereReranker,
    CrossEncoderReranker,
    Reranker,
)
from forge.rag.retrieval import RetrievalResult


def _result(chunk_id: str, text: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(chunk=Chunk(id=chunk_id, text=text), score=score)


# ---------------------------------------------------------------------------
# Reranker Protocol
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_cohere_reranker_satisfies_protocol(self) -> None:
        reranker = CohereReranker(client=MagicMock())
        assert isinstance(reranker, Reranker)

    def test_cross_encoder_reranker_satisfies_protocol(self) -> None:
        reranker = CrossEncoderReranker(encoder=MagicMock())
        assert isinstance(reranker, Reranker)


# ---------------------------------------------------------------------------
# CohereReranker — construction
# ---------------------------------------------------------------------------


class TestCohereConstruction:
    def test_empty_model_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CohereReranker(model="")

    def test_model_property_exposed(self) -> None:
        reranker = CohereReranker(model="rerank-multilingual-v3.0", client=MagicMock())
        assert reranker.model == "rerank-multilingual-v3.0"

    def test_extra_missing_raises_at_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio

        monkeypatch.setitem(sys.modules, "cohere", None)
        reranker = CohereReranker()
        with pytest.raises(ImportError, match=r"\[rag\] extra"):
            asyncio.run(reranker.rerank("q", [_result("a", "txt")]))


# ---------------------------------------------------------------------------
# CohereReranker — rerank with mock client
# ---------------------------------------------------------------------------


@dataclass
class _CohereResultEntry:
    index: int
    relevance_score: float


@dataclass
class _CohereResponse:
    results: list[_CohereResultEntry]


@pytest.fixture
def mock_cohere_client() -> AsyncMock:
    client = AsyncMock()
    return client


class TestCohereRerank:
    async def test_reorders_results(self, mock_cohere_client: AsyncMock) -> None:
        # Cohere reorders inputs by relevance: index 2 first, then 0, then 1.
        mock_cohere_client.rerank = AsyncMock(
            return_value=_CohereResponse(
                results=[
                    _CohereResultEntry(index=2, relevance_score=0.9),
                    _CohereResultEntry(index=0, relevance_score=0.6),
                    _CohereResultEntry(index=1, relevance_score=0.3),
                ]
            )
        )
        reranker = CohereReranker(client=mock_cohere_client)
        inputs = [
            _result("a", "first text"),
            _result("b", "second text"),
            _result("c", "third text"),
        ]
        results = await reranker.rerank("query", inputs)
        ids = [r.chunk.id for r in results]
        assert ids == ["c", "a", "b"]
        # Scores come from Cohere's relevance_score.
        assert results[0].score == 0.9

    async def test_top_k_truncates(self, mock_cohere_client: AsyncMock) -> None:
        mock_cohere_client.rerank = AsyncMock(
            return_value=_CohereResponse(
                results=[
                    _CohereResultEntry(index=0, relevance_score=0.5),
                    _CohereResultEntry(index=1, relevance_score=0.3),
                ]
            )
        )
        reranker = CohereReranker(client=mock_cohere_client)
        results = await reranker.rerank(
            "q",
            [_result("a", "x"), _result("b", "y"), _result("c", "z")],
            top_k=2,
        )
        assert len(results) == 2
        # Cohere was asked for top_n=2.
        kwargs = mock_cohere_client.rerank.call_args.kwargs
        assert kwargs["top_n"] == 2

    async def test_empty_results_returns_empty(self, mock_cohere_client: AsyncMock) -> None:
        reranker = CohereReranker(client=mock_cohere_client)
        assert await reranker.rerank("q", []) == ()
        mock_cohere_client.rerank.assert_not_called()

    async def test_empty_query_rejected(self, mock_cohere_client: AsyncMock) -> None:
        reranker = CohereReranker(client=mock_cohere_client)
        with pytest.raises(ValueError, match="non-empty"):
            await reranker.rerank("", [_result("a", "x")])

    async def test_invalid_top_k_rejected(self, mock_cohere_client: AsyncMock) -> None:
        reranker = CohereReranker(client=mock_cohere_client)
        mock_cohere_client.rerank = AsyncMock(return_value=_CohereResponse(results=[]))
        with pytest.raises(ValueError, match="top_k"):
            await reranker.rerank("q", [_result("a", "x")], top_k=0)

    async def test_model_forwarded_to_client(self, mock_cohere_client: AsyncMock) -> None:
        mock_cohere_client.rerank = AsyncMock(return_value=_CohereResponse(results=[]))
        reranker = CohereReranker(model="rerank-multilingual-v3.0", client=mock_cohere_client)
        await reranker.rerank("q", [_result("a", "x")])
        kwargs = mock_cohere_client.rerank.call_args.kwargs
        assert kwargs["model"] == "rerank-multilingual-v3.0"


class TestCohereClientConstruction:
    def test_builds_client_on_first_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_client = AsyncMock()
        fake_module = types.ModuleType("cohere")
        fake_module.AsyncClientV2 = MagicMock(return_value=fake_client)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "cohere", fake_module)

        reranker = CohereReranker(api_key="my-key")
        client = reranker._get_client()  # type: ignore[attr-defined]
        assert client is fake_client
        # The constructor was called with the api_key.
        fake_module.AsyncClientV2.assert_called_once_with(api_key="my-key")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------


class TestCrossEncoderConstruction:
    def test_empty_model_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CrossEncoderReranker(model="")

    def test_extra_missing_raises_at_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio

        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        reranker = CrossEncoderReranker()
        with pytest.raises(ImportError, match="sentence-transformers"):
            asyncio.run(reranker.rerank("q", [_result("a", "x")]))


class TestCrossEncoderRerank:
    async def test_scores_and_reorders(self) -> None:
        # Fake encoder: predict returns scores for each (query, doc) pair.
        encoder = MagicMock()
        encoder.predict = MagicMock(return_value=[0.2, 0.9, 0.5])
        reranker = CrossEncoderReranker(encoder=encoder)
        results = await reranker.rerank(
            "q",
            [_result("a", "x"), _result("b", "y"), _result("c", "z")],
        )
        # Highest-scoring first.
        ids = [r.chunk.id for r in results]
        assert ids == ["b", "c", "a"]
        assert results[0].score == 0.9

    async def test_top_k_truncates(self) -> None:
        encoder = MagicMock()
        encoder.predict = MagicMock(return_value=[0.5, 0.1, 0.9])
        reranker = CrossEncoderReranker(encoder=encoder)
        results = await reranker.rerank(
            "q",
            [_result("a", "x"), _result("b", "y"), _result("c", "z")],
            top_k=2,
        )
        assert len(results) == 2
        # Top two by score.
        assert [r.chunk.id for r in results] == ["c", "a"]

    async def test_empty_results(self) -> None:
        reranker = CrossEncoderReranker(encoder=MagicMock())
        assert await reranker.rerank("q", []) == ()

    async def test_empty_query_rejected(self) -> None:
        reranker = CrossEncoderReranker(encoder=MagicMock())
        with pytest.raises(ValueError, match="non-empty"):
            await reranker.rerank("", [_result("a", "x")])

    async def test_invalid_top_k_rejected(self) -> None:
        reranker = CrossEncoderReranker(encoder=MagicMock())
        with pytest.raises(ValueError, match="top_k"):
            await reranker.rerank("q", [_result("a", "x")], top_k=0)

    async def test_encoder_predict_receives_pairs(self) -> None:
        encoder = MagicMock()
        encoder.predict = MagicMock(return_value=[0.5, 0.3])
        reranker = CrossEncoderReranker(encoder=encoder)
        await reranker.rerank(
            "my query",
            [_result("a", "first"), _result("b", "second")],
        )
        # The encoder.predict call args contain the (query, doc) pairs.
        call_args = encoder.predict.call_args.args[0]
        assert list(call_args) == [("my query", "first"), ("my query", "second")]


class TestCrossEncoderClientConstruction:
    def test_builds_encoder_on_first_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_encoder = MagicMock()
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.CrossEncoder = MagicMock(return_value=fake_encoder)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        reranker = CrossEncoderReranker(model="my-model", device="cpu")
        encoder = reranker._get_encoder()  # type: ignore[attr-defined]
        assert encoder is fake_encoder
        fake_module.CrossEncoder.assert_called_once_with("my-model", device="cpu")  # type: ignore[attr-defined]

    def test_model_property_exposed(self) -> None:
        reranker = CrossEncoderReranker(model="my-model", encoder=MagicMock())
        assert reranker.model == "my-model"
