"""Embedder Protocol plus a LiteLLM-backed concrete implementation.

The :class:`Embedder` Protocol is the contract any concrete embedder
must satisfy. :class:`LiteLLMEmbedder` calls ``litellm.aembedding``
under the hood, giving Forge a provider-agnostic surface (OpenAI's
``text-embedding-3-*``, Cohere's ``embed-*``, Voyage, etc.) with one
import. See [ADR 0012](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/adr/0012-rag-protocols-and-vector-store-relocation.md)
for why embeddings live on their own seam rather than as a method
on :class:`LLMClient`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "Embedder",
    "LiteLLMEmbedder",
]


@runtime_checkable
class Embedder(Protocol):
    """Async function that turns text into a fixed-length embedding vector.

    Implementations are expected to be **deterministic** for a given
    ``(model, text)`` pair — calling :meth:`embed` twice with the same
    text returns the same vector. ``embed_batch`` provides a way to
    amortize per-request overhead by sending multiple texts together
    when the underlying API supports it.
    """

    @property
    def model(self) -> str:
        """The embedding model identifier."""
        ...  # pragma: no cover — Protocol body

    async def embed(self, text: str) -> Sequence[float]:
        """Return the embedding vector for ``text``."""
        ...  # pragma: no cover — Protocol body

    async def embed_batch(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return embedding vectors for ``texts`` in order."""
        ...  # pragma: no cover — Protocol body


class LiteLLMEmbedder:
    """Embedder backed by :func:`litellm.aembedding`.

    Args:
        model: An embedding model identifier LiteLLM recognizes
            (``text-embedding-3-small``, ``cohere/embed-english-v3.0``,
            ``voyage/voyage-3``, …). Defaults to OpenAI's smaller
            model — cheap, broadly compatible, 1536-dim.
        dimensions: Optional override for models that support a
            dimensions parameter (e.g. ``text-embedding-3-small``
            supports 512, 1024, or 1536). Forwarded as
            ``dimensions=`` to LiteLLM.
        provider_extras: Mapping of extra kwargs forwarded verbatim
            to ``litellm.aembedding`` — handy for provider-specific
            knobs (``input_type=`` for Cohere, custom ``api_base``,
            etc.).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        dimensions: int | None = None,
        provider_extras: dict[str, Any] | None = None,
    ) -> None:
        if not model:
            err = "LiteLLMEmbedder: model must be non-empty"
            raise ValueError(err)
        self._model = model
        self._dimensions = dimensions
        self._provider_extras: dict[str, Any] = dict(provider_extras or {})

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    async def embed(self, text: str) -> Sequence[float]:
        if not text:
            err = "LiteLLMEmbedder.embed: text must be non-empty"
            raise ValueError(err)
        vectors = await self._call([text])
        return vectors[0]

    async def embed_batch(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        if any(not t for t in texts):
            err = "LiteLLMEmbedder.embed_batch: every text must be non-empty"
            raise ValueError(err)
        return await self._call(list(texts))

    async def _call(self, texts: list[str]) -> list[list[float]]:
        import litellm  # type: ignore[import-untyped]

        kwargs: dict[str, Any] = {"model": self._model, "input": texts}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        kwargs.update(self._provider_extras)
        aembedding: Any = litellm.aembedding  # type: ignore[reportUnknownMemberType]
        response: Any = await aembedding(**kwargs)
        # LiteLLM normalizes responses to OpenAI's shape: data is a list
        # of {"embedding": [...], "index": i}. We sort by index defensively
        # in case the provider returns out-of-order results.
        data: list[Any] = list(response.data)

        def _index_of(entry: Any) -> int:
            if isinstance(entry, dict):
                return int(cast("dict[str, Any]", entry)["index"])
            return int(entry.index)

        def _embedding_of(entry: Any) -> list[float]:
            if isinstance(entry, dict):
                return list(cast("dict[str, Any]", entry)["embedding"])
            return list(entry.embedding)

        data.sort(key=_index_of)
        return [_embedding_of(entry) for entry in data]
