"""Composable RAG pipeline.

:class:`RAGPipeline` ties the module's primitives together:
chunker → retriever → (optional) reranker. Two helpers wrap the
common workflows: :meth:`ingest` chunks documents and writes them
into the retriever (when the retriever supports it); :meth:`query`
retrieves + reranks; :meth:`augment_prompt` formats a query plus
its retrieved context as a single string ready to drop into an LLM
call.

The pipeline is intentionally a thin composition — every piece is
swappable through the constructor, and the user is expected to wire
the retriever to the right vector store / embedder beforehand. This
keeps the type surface narrow (one class) without burying the
configuration of the underlying primitives behind it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from strata_forge.rag.chunking import Chunk, Chunker, Document
    from strata_forge.rag.rerankers import Reranker
    from strata_forge.rag.retrieval import RetrievalResult, Retriever

__all__ = [
    "DEFAULT_AUGMENT_TEMPLATE",
    "IndexableRetriever",
    "RAGPipeline",
]


@runtime_checkable
class IndexableRetriever(Protocol):
    """A :class:`Retriever` that also accepts chunks to index.

    :class:`DenseRetriever` satisfies this Protocol; the in-process
    BM25 retriever does not (it's constructed with the corpus and
    immutable thereafter). :meth:`RAGPipeline.ingest` checks for
    Protocol satisfaction and raises a clear error when the
    retriever isn't indexable.
    """

    async def index(self, chunks: Sequence[Chunk]) -> None:
        """Chunk and store ``chunks``."""
        ...  # pragma: no cover — Protocol body

    async def retrieve(self, query: str, *, top_k: int = 5) -> Sequence[RetrievalResult]:
        """Return up to ``top_k`` retrieval hits for ``query``."""
        ...  # pragma: no cover — Protocol body


DEFAULT_AUGMENT_TEMPLATE = (
    "Use the following sources to answer the question. If the sources don't "
    "contain the answer, say so — don't speculate.\n\n"
    "Sources:\n{context}\n\n"
    "Question: {query}"
)


class RAGPipeline:
    """Glue between a chunker, a retriever, and (optionally) a reranker.

    Args:
        chunker: How documents are split before indexing.
            :meth:`ingest` calls ``chunker.chunk`` on every document.
        retriever: The :class:`Retriever` that serves :meth:`query`.
            When this retriever also satisfies
            :class:`IndexableRetriever`, :meth:`ingest` writes chunks
            into it; otherwise :meth:`ingest` raises.
        reranker: Optional :class:`Reranker` applied after retrieval.
            When set, :meth:`query` first retrieves
            ``rerank_top_k or top_k`` candidates, then reranks them
            down to ``top_k``.
    """

    def __init__(
        self,
        *,
        chunker: Chunker,
        retriever: Retriever,
        reranker: Reranker | None = None,
    ) -> None:
        self._chunker = chunker
        self._retriever = retriever
        self._reranker = reranker

    @property
    def chunker(self) -> Chunker:
        return self._chunker

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def reranker(self) -> Reranker | None:
        return self._reranker

    async def ingest(self, documents: Sequence[Document]) -> int:
        """Chunk ``documents`` and index them via the retriever.

        Returns the total number of chunks written. Raises
        :class:`TypeError` when the configured retriever doesn't
        satisfy :class:`IndexableRetriever` — the caller needs to
        pass an indexable retriever (e.g. :class:`DenseRetriever`)
        or pre-build / ingest into a non-indexable retriever
        (e.g. :class:`BM25Retriever`) before constructing the
        pipeline.
        """
        if not isinstance(self._retriever, IndexableRetriever):
            err = (
                f"RAGPipeline.ingest: retriever of type "
                f"{type(self._retriever).__name__} doesn't satisfy "
                "IndexableRetriever. Construct the retriever with its "
                "corpus directly, or use an indexable retriever like "
                "DenseRetriever."
            )
            raise TypeError(err)
        all_chunks: list[Chunk] = []
        for document in documents:
            all_chunks.extend(self._chunker.chunk(document))
        if not all_chunks:
            return 0
        await self._retriever.index(all_chunks)
        return len(all_chunks)

    async def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        rerank_top_k: int | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Retrieve and optionally rerank.

        When a reranker is configured, the pipeline first retrieves
        ``rerank_top_k or top_k`` candidates, then reranks them down
        to ``top_k``. A larger ``rerank_top_k`` gives the reranker
        more candidates to work with at the cost of an extra
        retrieval-side roundtrip.

        Args:
            query: The user's query.
            top_k: Number of final results to return.
            rerank_top_k: Number of candidates to pull from the
                retriever before reranking. Defaults to ``top_k``
                when omitted — i.e. no widening.
        """
        if not query:
            err = "RAGPipeline.query: query must be non-empty"
            raise ValueError(err)
        if top_k <= 0:
            err = f"top_k must be >= 1; got {top_k}"
            raise ValueError(err)

        if self._reranker is None:
            candidates = await self._retriever.retrieve(query, top_k=top_k)
            return tuple(candidates)

        widen = rerank_top_k if rerank_top_k is not None else top_k
        if widen < top_k:
            err = (
                f"rerank_top_k ({widen}) must be >= top_k ({top_k}) — "
                "reranking a smaller pool than the final size doesn't help."
            )
            raise ValueError(err)
        candidates = await self._retriever.retrieve(query, top_k=widen)
        reranked = await self._reranker.rerank(query, candidates, top_k=top_k)
        return tuple(reranked)

    async def augment_prompt(
        self,
        query: str,
        *,
        top_k: int = 5,
        rerank_top_k: int | None = None,
        template: str = DEFAULT_AUGMENT_TEMPLATE,
        source_separator: str = "\n---\n",
    ) -> str:
        """Run :meth:`query` and format the result as an augmented prompt.

        The returned string substitutes ``{query}`` and ``{context}``
        into ``template``; ``{context}`` is the retrieved chunks'
        text joined by ``source_separator``. The default template
        instructs the LLM to abstain when the sources don't cover
        the question — adjust it for your prompt style.
        """
        results = await self.query(query, top_k=top_k, rerank_top_k=rerank_top_k)
        context = source_separator.join(result.chunk.text for result in results)
        return template.format(query=query, context=context)
