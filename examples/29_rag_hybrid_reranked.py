"""Hybrid retrieval (dense + BM25) with optional Cohere reranking.

Requires a configured embedder provider (defaults to OpenAI). The
reranker is optional — set ``COHERE_API_KEY`` to enable it.

The pipeline combines two complementary signals:

- :class:`DenseRetriever` (semantic / paraphrase recall)
- :class:`BM25Retriever` (keyword / exact-match recall)

… fuses them via :class:`HybridRetriever` (Reciprocal Rank Fusion),
then optionally reranks the fused list with :class:`CohereReranker`
when an API key is configured.

Usage::

    uv run python examples/29_rag_hybrid_reranked.py
    COHERE_API_KEY=... uv run python examples/29_rag_hybrid_reranked.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from _common import parse_args, require_env  # type: ignore[import-not-found]

from strata_forge.rag import (
    BM25Retriever,
    Chunk,
    DenseRetriever,
    Document,
    HybridRetriever,
    InMemoryVectorStore,
    LiteLLMEmbedder,
    RAGPipeline,
    RecursiveChunker,
)


def _documents() -> list[Document]:
    return [
        Document(
            id="rfc-2616",
            text=(
                "HTTP/1.1 status code 429 indicates that the user has sent "
                "too many requests in a given amount of time. Servers should "
                "include a Retry-After header indicating how long to wait."
            ),
        ),
        Document(
            id="rfc-7231",
            text=(
                "The 503 Service Unavailable status code indicates that the "
                "server is currently unable to handle the request due to a "
                "temporary overload or scheduled maintenance."
            ),
        ),
        Document(
            id="rfc-8174",
            text=(
                "Capitalized 'MUST', 'SHOULD', and 'MAY' in RFC documents are "
                "defined by RFC 2119; they have specific normative meaning "
                "and should not be used otherwise."
            ),
        ),
    ]


async def _main() -> None:
    args = parse_args(
        description="Hybrid (dense + BM25) RAG with optional Cohere reranker.",
        default_provider="openai",
    )
    require_env(args.provider)

    # Build chunks once — feeds both retrievers.
    documents = _documents()
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunker.chunk(doc))

    # Dense path: embed + index into the in-process store.
    embedder = LiteLLMEmbedder(model="text-embedding-3-small")
    store = InMemoryVectorStore()
    dense = DenseRetriever(embedder=embedder, store=store)
    await dense.index(all_chunks)

    # Sparse path: BM25 over the same chunks.
    sparse = BM25Retriever(all_chunks)

    # Fuse.
    hybrid = HybridRetriever([dense, sparse], rrf_k=60)

    # Optional reranker.
    reranker = None
    if os.environ.get("COHERE_API_KEY"):
        try:
            import cohere  # noqa: F401

            from strata_forge.rag import CohereReranker

            reranker = CohereReranker(model="rerank-english-v3.0")
            print("--- using Cohere reranker")
        except ImportError:
            print(
                "[note] COHERE_API_KEY set but cohere SDK missing; skipping reranker.",
                file=sys.stderr,
            )
    else:
        print("--- no COHERE_API_KEY; running without reranker")

    pipeline = RAGPipeline(chunker=chunker, retriever=hybrid, reranker=reranker)

    for query in (
        "What does 429 mean in HTTP?",
        "Rate limiting response header",
        "When can I use SHOULD in spec documents?",
    ):
        print(f"\n--- query: {query}")
        results = await pipeline.query(query, top_k=2, rerank_top_k=4 if reranker else None)
        for r in results:
            print(f"  [score={r.score:.4f}] {r.chunk.text[:100]}...")


if __name__ == "__main__":
    asyncio.run(_main())
