"""Reranker Protocol plus two concrete implementations.

Rerankers re-score a candidate list from a retriever using a more
expensive but more accurate model. They take a query + ranked
results, return a new ranked list (typically the same chunks with
new scores, optionally truncated).

Two implementations ship:

- :class:`CohereReranker` — uses Cohere's Rerank API via the
  ``cohere`` SDK (behind the ``[rag]`` extra).
- :class:`CrossEncoderReranker` — runs a local cross-encoder model
  via ``sentence_transformers`` (the user installs that separately;
  it isn't pulled into ``[rag]`` because it brings ``torch``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from strata_forge.rag.retrieval import RetrievalResult

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CohereReranker",
    "CrossEncoderReranker",
    "Reranker",
]


@runtime_checkable
class Reranker(Protocol):
    """The contract every reranker satisfies.

    Async so production rerankers that hit a network (Cohere,
    Voyage, …) and async-native in-process ones share the same
    surface.
    """

    async def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Re-score ``results`` for ``query`` and return the new ranking.

        When ``top_k`` is ``None``, returns every input in the new
        order. When set, truncates to the top-k.
        """
        ...  # pragma: no cover — Protocol body


# ---------------------------------------------------------------------------
# CohereReranker
# ---------------------------------------------------------------------------


class CohereReranker:
    """Reranker backed by Cohere's Rerank API.

    Args:
        model: Cohere rerank model id. Default
            ``"rerank-english-v3.0"``.
        api_key: Cohere API key. When ``None``, the SDK reads
            ``COHERE_API_KEY`` from the environment.
        client: Optional pre-built ``cohere.AsyncClientV2`` (handy
            for tests). When omitted, the reranker builds one from
            ``api_key`` on first use.
    """

    def __init__(
        self,
        *,
        model: str = "rerank-english-v3.0",
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not model:
            err = "CohereReranker: model must be non-empty"
            raise ValueError(err)
        self._model = model
        self._api_key = api_key
        self._explicit_client = client
        self._client: Any | None = None

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._explicit_client is not None:
            return self._explicit_client
        if self._client is None:
            try:
                cohere_mod: Any = __import__("cohere", fromlist=["AsyncClientV2"])
            except ImportError as exc:
                msg = (
                    "The [rag] extra is required for CohereReranker. "
                    "Install it with: pip install 'strata-forge[rag]'."
                )
                raise ImportError(msg) from exc
            kwargs: dict[str, Any] = {}
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            self._client = cohere_mod.AsyncClientV2(**kwargs)
        return self._client

    async def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievalResult, ...]:
        if not query:
            err = "CohereReranker.rerank: query must be non-empty"
            raise ValueError(err)
        if not results:
            return ()
        if top_k is not None and top_k < 1:
            err = f"top_k must be >= 1 when set; got {top_k}"
            raise ValueError(err)
        effective_top_k = top_k if top_k is not None else len(results)

        client = self._get_client()
        documents = [r.chunk.text for r in results]
        response = await client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=min(effective_top_k, len(results)),
        )
        # Cohere returns a list of {index, relevance_score} entries
        # sorted by relevance_score desc.
        reranked: list[RetrievalResult] = []
        for entry in response.results:
            idx = int(entry.index)
            score = float(entry.relevance_score)
            original = results[idx]
            reranked.append(RetrievalResult(chunk=original.chunk, score=score))
        return tuple(reranked)


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """Reranker backed by a local ``sentence_transformers`` cross-encoder.

    Cross-encoders score ``(query, document)`` pairs directly,
    typically out-performing dense / sparse retrieval on relevance
    but at noticeably higher per-query latency. Common choices:
    ``"cross-encoder/ms-marco-MiniLM-L-6-v2"`` (fast),
    ``"BAAI/bge-reranker-base"``.

    ``sentence-transformers`` isn't pulled in by the ``[rag]`` extra
    because it transitively brings ``torch`` (~2 GB on disk). Install
    it separately::

        pip install sentence-transformers

    Args:
        model: Hugging Face cross-encoder model id.
        device: Torch device — ``"cpu"``, ``"cuda"``, ``"mps"``, …
            When ``None``, sentence-transformers picks a default.
        encoder: Optional pre-built ``CrossEncoder`` (handy for
            tests). When omitted, the reranker builds one from
            ``model``/``device`` on first use.
    """

    def __init__(
        self,
        *,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        encoder: Any | None = None,
    ) -> None:
        if not model:
            err = "CrossEncoderReranker: model must be non-empty"
            raise ValueError(err)
        self._model_name = model
        self._device = device
        self._explicit_encoder = encoder
        self._encoder: Any | None = None

    @property
    def model(self) -> str:
        return self._model_name

    def _get_encoder(self) -> Any:
        if self._explicit_encoder is not None:
            return self._explicit_encoder
        if self._encoder is None:
            try:
                st_mod: Any = __import__("sentence_transformers", fromlist=["CrossEncoder"])
            except ImportError as exc:
                msg = (
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install it with: pip install sentence-transformers"
                )
                raise ImportError(msg) from exc
            kwargs: dict[str, Any] = {}
            if self._device is not None:
                kwargs["device"] = self._device
            self._encoder = st_mod.CrossEncoder(self._model_name, **kwargs)
        return self._encoder

    async def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievalResult, ...]:
        if not query:
            err = "CrossEncoderReranker.rerank: query must be non-empty"
            raise ValueError(err)
        if not results:
            return ()
        if top_k is not None and top_k < 1:
            err = f"top_k must be >= 1 when set; got {top_k}"
            raise ValueError(err)

        encoder = self._get_encoder()
        pairs = [(query, r.chunk.text) for r in results]
        # CrossEncoder.predict is sync — run in a thread to avoid blocking.
        import asyncio

        scores = await asyncio.to_thread(encoder.predict, pairs)
        scored = list(zip(results, scores, strict=True))
        scored.sort(key=lambda pair: float(pair[1]), reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return tuple(
            RetrievalResult(chunk=original.chunk, score=float(score)) for original, score in scored
        )
