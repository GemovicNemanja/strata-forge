# forge.rag

Retrieval-augmented generation primitives: embedder built on the `forge.llm` provider abstraction (any provider's embedding models), Qdrant vector store, chunkers (recursive / token-based / semantic), retrieval (dense, BM25 via `bm25s`, hybrid via reciprocal rank fusion), rerankers (Cohere, cross-encoders), loaders (text + URL), and a composable retrieve → rerank → format pipeline.

> Implementation pending. See `docs/roadmap.md` for current status.
