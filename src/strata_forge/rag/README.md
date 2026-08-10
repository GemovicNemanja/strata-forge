# strata_forge.rag

Retrieval-augmented generation primitives, each behind a runtime-checkable Protocol so any piece
can be swapped: `Embedder`, `Chunker`, `Retriever`, `VectorStore`, `Reranker`. The shipped
implementations are `LiteLLMEmbedder` (calls LiteLLM's embedding API directly — see
[ADR 0012](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/adr/0012-rag-protocols-and-vector-store-relocation.md)),
`RecursiveChunker`, `InMemoryVectorStore` and `QdrantVectorStore`, dense / pure-Python BM25 /
hybrid RRF retrievers, and Cohere or cross-encoder rerankers.

`RAGPipeline` composes them into ingest → retrieve → rerank → augment, and can format retrieved
chunks into a grounded prompt for `strata_forge.llm`.

Qdrant and Cohere need the `[rag]` extra; `CrossEncoderReranker` additionally needs
`sentence-transformers`, which `[rag]` deliberately leaves out because it pulls in torch.

Reference:
[docs/modules/rag.md](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/rag.md).
