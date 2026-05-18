"""RAG prototype — chunk, embed, retrieve in-process.

Indexes a handful of inline documents into the in-process vector
store, lets you type a query, and shows the ranked results. Use
this as a scratchpad before wiring up Qdrant or hybrid retrieval.

Set ``OPENAI_API_KEY`` (or another embedder provider) before
launching with::

    uv run marimo edit notebooks/02_rag_prototype.py
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        "# RAG prototype\n\n"
        "Chunk inline docs, embed them, query with a typed string. "
        "Swap pieces (chunker, retriever, reranker) as you iterate."
    )
    return


@app.cell
def _():
    documents = {
        "sun.md": (
            "The Sun is mostly hydrogen and helium. Its energy comes from "
            "nuclear fusion in the core; photons emitted there take "
            "hundreds of thousands of years to reach the surface."
        ),
        "mars.md": (
            "Mars has two small moons, Phobos and Deimos. The planet's "
            "surface appears red because of iron oxide dust."
        ),
        "moon.md": (
            "Earth's Moon causes tides and gradually slows Earth's "
            "rotation. It formed when a Mars-sized body struck the "
            "young Earth."
        ),
    }
    return (documents,)


@app.cell
def _(mo):
    query_input = mo.ui.text(
        value="Why does Mars look red?", label="query"
    )
    top_k_input = mo.ui.slider(1, 5, step=1, value=3, label="top_k")
    return query_input, top_k_input


@app.cell
def _(mo, query_input, top_k_input):
    mo.hstack([query_input, top_k_input])
    return


@app.cell
async def _(documents, query_input, top_k_input):
    from forge.rag.chunking import RecursiveChunker
    from forge.rag.dense import DenseRetriever
    from forge.rag.embedding import LiteLLMEmbedder
    from forge.rag.pipeline import RAGPipeline
    from forge.rag.vector_store import InMemoryVectorStore

    pipeline = RAGPipeline(
        chunker=RecursiveChunker(chunk_size=200, chunk_overlap=20),
        retriever=DenseRetriever(
            embedder=LiteLLMEmbedder(model="text-embedding-3-small"),
            store=InMemoryVectorStore(),
        ),
    )

    docs = [
        {"id": doc_id, "text": text} for doc_id, text in documents.items()
    ]
    n_chunks = await pipeline.ingest(docs)
    results = await pipeline.query(query_input.value, top_k=top_k_input.value)
    return n_chunks, results


@app.cell
def _(mo, n_chunks, results):
    rows = [
        {
            "rank": i,
            "score": f"{r.score:.3f}",
            "doc": r.chunk.document_id,
            "text": r.chunk.text[:120],
        }
        for i, r in enumerate(results, start=1)
    ]
    mo.vstack(
        [
            mo.md(f"_indexed {n_chunks} chunks_"),
            mo.ui.table(rows),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
