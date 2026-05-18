"""Basic RAG pipeline: ingest a few documents, query them.

Requires a configured provider (defaults to Anthropic) for the
embedder. Skips cleanly when keys aren't set.

The pipeline:
1. Chunks the source documents with :class:`RecursiveChunker`.
2. Embeds each chunk via :class:`LiteLLMEmbedder`.
3. Stores vectors in an :class:`InMemoryVectorStore`.
4. Serves queries through a :class:`DenseRetriever`, wrapped in a
   :class:`RAGPipeline` for ergonomics.

Usage::

    uv run python examples/26_rag_basic.py
    uv run python examples/26_rag_basic.py --model gpt-5.5 --provider openai
"""

from __future__ import annotations

import asyncio

from _common import parse_args, require_env  # type: ignore[import-not-found]

from forge.rag import (
    DenseRetriever,
    Document,
    InMemoryVectorStore,
    LiteLLMEmbedder,
    RAGPipeline,
    RecursiveChunker,
)


def _documents() -> list[Document]:
    return [
        Document(
            id="solar",
            text=(
                "The Sun is the star at the centre of the Solar System. "
                "Its core fuses hydrogen into helium, releasing energy "
                "that radiates outward through the photosphere. The Sun "
                "accounts for about 99.8 percent of the Solar System's mass."
            ),
        ),
        Document(
            id="mars",
            text=(
                "Mars is the fourth planet from the Sun. Often called the "
                "Red Planet, its surface iron oxide gives it a distinctive "
                "rust colour. Mars has two small moons, Phobos and Deimos, "
                "and was the destination of NASA's Curiosity and Perseverance rovers."
            ),
        ),
        Document(
            id="moon",
            text=(
                "Earth's Moon is the fifth-largest natural satellite in the "
                "Solar System and the only one of a relative size comparable "
                "to the planet it orbits. The Moon's gravitational pull drives "
                "the ocean tides on Earth."
            ),
        ),
    ]


async def _main() -> None:
    # Reuse the provider helpers — the embedder uses a provider-routed model.
    # text-embedding-3-small is OpenAI, but we use --provider only to gate
    # on the relevant env var presence (Anthropic doesn't ship embeddings).
    args = parse_args(
        description="Basic RAG pipeline.",
        default_provider="openai",
    )
    require_env(args.provider)

    embedder = LiteLLMEmbedder(model="text-embedding-3-small")
    store = InMemoryVectorStore()
    retriever = DenseRetriever(embedder=embedder, store=store)
    pipeline = RAGPipeline(
        chunker=RecursiveChunker(chunk_size=200, chunk_overlap=20),
        retriever=retriever,
    )

    print("--- ingesting documents ---")
    n_chunks = await pipeline.ingest(_documents())
    print(f"indexed {n_chunks} chunks across 3 documents\n")

    queries = [
        "What is the Sun mostly made of?",
        "Which moons does Mars have?",
        "How does the Moon affect Earth?",
    ]
    for query in queries:
        print(f"--- query: {query}")
        results = await pipeline.query(query, top_k=2)
        for result in results:
            print(
                f"  [score={result.score:.3f}] "
                f"(doc={result.chunk.document_id}) {result.chunk.text[:120]}..."
            )
        print()


if __name__ == "__main__":
    asyncio.run(_main())
