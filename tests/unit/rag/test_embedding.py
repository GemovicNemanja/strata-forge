"""Unit tests for `strata_forge.rag.embedding`."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from strata_forge.rag.embedding import Embedder, LiteLLMEmbedder

# ---------------------------------------------------------------------------
# Embedder Protocol
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_litellm_embedder_satisfies_protocol(self) -> None:
        embedder = LiteLLMEmbedder()
        assert isinstance(embedder, Embedder)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_model(self) -> None:
        embedder = LiteLLMEmbedder()
        assert embedder.model == "text-embedding-3-small"
        assert embedder.dimensions is None

    def test_explicit_model_and_dimensions(self) -> None:
        embedder = LiteLLMEmbedder(model="cohere/embed-english-v3.0", dimensions=512)
        assert embedder.model == "cohere/embed-english-v3.0"
        assert embedder.dimensions == 512

    def test_empty_model_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            LiteLLMEmbedder(model="")


# ---------------------------------------------------------------------------
# embed / embed_batch behaviour with mocked litellm
# ---------------------------------------------------------------------------


def _mock_litellm_response(vectors: list[list[float]]) -> Any:
    """Build an object that mimics LiteLLM's embedding response shape."""
    response = AsyncMock()
    response.data = [{"embedding": vec, "index": i} for i, vec in enumerate(vectors)]
    return response


@pytest.fixture
def fake_aembedding(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Replace litellm.aembedding with a mock that records every call."""
    import litellm  # type: ignore[import-untyped]

    mock = AsyncMock()
    monkeypatch.setattr(litellm, "aembedding", mock)
    return mock


class TestEmbed:
    async def test_returns_single_vector(self, fake_aembedding: AsyncMock) -> None:
        fake_aembedding.return_value = _mock_litellm_response([[0.1, 0.2, 0.3]])
        embedder = LiteLLMEmbedder()
        vec = await embedder.embed("hello")
        assert list(vec) == [0.1, 0.2, 0.3]

    async def test_forwards_model_and_input(self, fake_aembedding: AsyncMock) -> None:
        fake_aembedding.return_value = _mock_litellm_response([[0.0]])
        embedder = LiteLLMEmbedder(model="my-model")
        await embedder.embed("the text")
        kwargs = fake_aembedding.call_args.kwargs
        assert kwargs["model"] == "my-model"
        assert kwargs["input"] == ["the text"]

    async def test_dimensions_forwarded_when_set(self, fake_aembedding: AsyncMock) -> None:
        fake_aembedding.return_value = _mock_litellm_response([[0.0]])
        embedder = LiteLLMEmbedder(dimensions=512)
        await embedder.embed("x")
        assert fake_aembedding.call_args.kwargs["dimensions"] == 512

    async def test_dimensions_omitted_when_none(self, fake_aembedding: AsyncMock) -> None:
        fake_aembedding.return_value = _mock_litellm_response([[0.0]])
        embedder = LiteLLMEmbedder()
        await embedder.embed("x")
        assert "dimensions" not in fake_aembedding.call_args.kwargs

    async def test_provider_extras_forwarded(self, fake_aembedding: AsyncMock) -> None:
        fake_aembedding.return_value = _mock_litellm_response([[0.0]])
        embedder = LiteLLMEmbedder(provider_extras={"input_type": "search_query"})
        await embedder.embed("x")
        assert fake_aembedding.call_args.kwargs["input_type"] == "search_query"

    async def test_empty_text_rejected(self) -> None:
        embedder = LiteLLMEmbedder()
        with pytest.raises(ValueError, match="non-empty"):
            await embedder.embed("")


class TestEmbedBatch:
    async def test_returns_vectors_in_input_order(self, fake_aembedding: AsyncMock) -> None:
        fake_aembedding.return_value = _mock_litellm_response([[1.0], [2.0], [3.0]])
        embedder = LiteLLMEmbedder()
        result = await embedder.embed_batch(["a", "b", "c"])
        assert [list(v) for v in result] == [[1.0], [2.0], [3.0]]

    async def test_handles_out_of_order_response(self, fake_aembedding: AsyncMock) -> None:
        # Some providers return data in arbitrary order; our code sorts by index.
        response = AsyncMock()
        response.data = [
            {"embedding": [3.0], "index": 2},
            {"embedding": [1.0], "index": 0},
            {"embedding": [2.0], "index": 1},
        ]
        fake_aembedding.return_value = response
        embedder = LiteLLMEmbedder()
        result = await embedder.embed_batch(["a", "b", "c"])
        assert [list(v) for v in result] == [[1.0], [2.0], [3.0]]

    async def test_empty_input_returns_empty(self) -> None:
        embedder = LiteLLMEmbedder()
        result = await embedder.embed_batch([])
        assert list(result) == []

    async def test_rejects_empty_text_in_batch(self) -> None:
        embedder = LiteLLMEmbedder()
        with pytest.raises(ValueError, match="non-empty"):
            await embedder.embed_batch(["a", "", "c"])

    async def test_handles_attribute_style_response(self, fake_aembedding: AsyncMock) -> None:
        # LiteLLM sometimes returns objects with .embedding and .index attributes
        # instead of dicts. Our code handles both shapes.
        from dataclasses import dataclass

        @dataclass
        class _Entry:
            embedding: list[float]
            index: int

        response = AsyncMock()
        response.data = [_Entry(embedding=[0.1], index=0), _Entry(embedding=[0.2], index=1)]
        fake_aembedding.return_value = response

        embedder = LiteLLMEmbedder()
        result = await embedder.embed_batch(["a", "b"])
        assert [list(v) for v in result] == [[0.1], [0.2]]
