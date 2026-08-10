"""RAG pipeline over Qdrant.

Requires the ``[rag]`` extra (``qdrant-client``) plus a running
Qdrant instance (``make stack-up`` boots one locally). Also needs
a configured embedder provider (defaults to OpenAI). Skips cleanly
when either is missing.

Same pipeline shape as ``27_rag_basic.py`` but with
:class:`QdrantVectorStore` instead of the in-process store —
demonstrating that the :class:`VectorStore` Protocol lets you swap
backends without touching the rest of the pipeline.

Usage::

    make stack-up      # boot Qdrant
    QDRANT_HOST=localhost QDRANT_PORT=6333 \\
        uv run python examples/28_rag_qdrant.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from _common import parse_args, require_env  # type: ignore[import-not-found]

from strata_forge.rag import (
    DenseRetriever,
    Document,
    LiteLLMEmbedder,
    RAGPipeline,
    RecursiveChunker,
)


def _documents() -> list[Document]:
    return [
        Document(
            id="kubernetes",
            text=(
                "Kubernetes is an open-source container orchestration system "
                "for automating software deployment, scaling, and management. "
                "Originally developed by Google, it was later donated to the "
                "Cloud Native Computing Foundation."
            ),
        ),
        Document(
            id="postgres",
            text=(
                "PostgreSQL is a powerful, open source object-relational "
                "database system with over 35 years of active development. "
                "It supports SQL standards and offers many modern features "
                "such as complex queries, foreign keys, and triggers."
            ),
        ),
    ]


async def _main() -> None:
    args = parse_args(
        description="RAG pipeline backed by Qdrant.",
        default_provider="openai",
    )
    require_env(args.provider)

    try:
        import qdrant_client  # noqa: F401
    except ImportError:
        print("[skip] [rag] extra not installed; pip install 'ai-forge[rag]'", file=sys.stderr)
        sys.exit(0)

    from strata_forge.rag import QdrantVectorStore

    embedder = LiteLLMEmbedder(model="text-embedding-3-small")
    store = QdrantVectorStore(
        collection_name="forge-rag-demo",
        embedding_dimensions=1536,
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
    )
    retriever = DenseRetriever(embedder=embedder, store=store)
    pipeline = RAGPipeline(
        chunker=RecursiveChunker(chunk_size=250, chunk_overlap=30),
        retriever=retriever,
    )

    print(f"--- using Qdrant collection {store.collection_name!r}")
    print("--- ingesting documents")
    n_chunks = await pipeline.ingest(_documents())
    print(f"indexed {n_chunks} chunks\n")

    for query in (
        "Who originally developed Kubernetes?",
        "What standards does PostgreSQL support?",
    ):
        print(f"--- query: {query}")
        results = await pipeline.query(query, top_k=2)
        for r in results:
            print(f"  [score={r.score:.3f}] {r.chunk.text[:120]}...")
        print()

    # Tidy up the collection so re-runs start clean.
    await store.clear()
    print("--- collection cleared")


if __name__ == "__main__":
    asyncio.run(_main())
