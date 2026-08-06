# Agent rules — strata_forge.rag

`strata_forge.rag` is the retrieval-augmented generation layer. It owns
four Protocols — :class:`Embedder`, :class:`VectorStore`,
:class:`Chunker`, :class:`Retriever` — plus concrete in-process
implementations and (later) production backends. See
[ADR 0012](../../../docs/architecture/adr/0012-rag-protocols-and-vector-store-relocation.md)
for the design rationale.

## Purpose

- :class:`Embedder` Protocol + :class:`LiteLLMEmbedder` —
  provider-agnostic embedding via ``litellm.aembedding``.
- :class:`Document` + :class:`Chunk` shapes + :class:`Chunker`
  Protocol + :class:`RecursiveChunker` — chunking primitives.
- :class:`Retriever` Protocol + :class:`RetrievalResult` —
  retrieval contract.
- :class:`VectorStore` Protocol + :class:`InMemoryVectorStore` +
  :class:`VectorItem` / :class:`VectorSearchResult` — vector
  storage primitives. Relocated from :mod:`strata_forge.agents.memory`
  per ADR 0012; the original module re-exports them for
  back-compatibility.
- Phase 4.2: :class:`QdrantVectorStore`, :class:`DenseRetriever`.
- Phase 4.3: :class:`BM25Retriever`, :class:`HybridRetriever`,
  :class:`Reranker` Protocol + Cohere / cross-encoder rerankers.
- Phase 4.4: :class:`RAGPipeline` — composable chunk → embed →
  store → retrieve → rerank flow.

## Boundaries

- **Owns:** ``embedding.py``, ``chunking.py``, ``retrieval.py``,
  ``vector_store.py``, ``stores/`` (Phase 4.2), ``rerankers/``
  (Phase 4.3), ``pipeline.py`` (Phase 4.4).
- **Imports from inside ``forge``:** :mod:`strata_forge.core` (errors,
  ids), :mod:`strata_forge.config` (settings for production backends
  when they need them). May import :mod:`strata_forge.llm` for token
  counting in chunkers, but **does NOT depend on**
  :mod:`strata_forge.llm.LLMClient` — embeddings go through ``litellm``
  directly (ADR 0012).
- **Does NOT import** :mod:`strata_forge.tracing` (ADR 0008),
  :mod:`strata_forge.agents`, :mod:`strata_forge.evals`,
  :mod:`strata_forge.datasets`. The dependency arrow points downward:
  ``agents → rag``, not the reverse.
- **External deps:** Pydantic at module load; ``litellm`` (in
  core) for :class:`LiteLLMEmbedder`. Phase 4.2 adds
  ``qdrant-client`` behind the ``[rag]`` extra. Phase 4.3 adds
  ``rank-bm25`` (or pure-Python BM25), ``cohere``,
  ``sentence-transformers`` — each lazy-imported behind ``[rag]``.

## Public API

The module's ``__init__.py`` re-exports:

- Protocols: :class:`Embedder`, :class:`Chunker`,
  :class:`Retriever`, :class:`VectorStore`.
- Shapes: :class:`Document`, :class:`Chunk`,
  :class:`RetrievalResult`, :class:`VectorItem`,
  :class:`VectorSearchResult`.
- Implementations: :class:`LiteLLMEmbedder`,
  :class:`RecursiveChunker`, :class:`InMemoryVectorStore`.
- Helpers: :func:`cosine_similarity`.

Errors raised from this module are :class:`ForgeError` subclasses
or :class:`ValueError` for input-validation failures.

## Internal patterns

- **Composition by Protocol.** Concrete retrievers, stores,
  chunkers, and embedders all satisfy their respective
  Protocols. Users supply their own implementations without
  inheriting from anything.
- **Frozen Pydantic shapes.** :class:`Document`, :class:`Chunk`,
  :class:`RetrievalResult` use ``frozen=True`` + ``extra="forbid"``
  for stable wire shapes. :class:`VectorItem` and
  :class:`VectorSearchResult` use frozen dataclasses for the same
  reason.
- **Async retrievers + stores.** Production backends hit a
  network; in-process backends still implement ``async`` for
  shape uniformity.
- **Sync chunkers.** Chunking is CPU-bound and benefits from the
  GIL-free path. Async variants land if LLM-based chunkers are
  added later.
- **No new LLM client.** Embeddings call ``litellm.aembedding``
  directly through :class:`LiteLLMEmbedder`; the chat-completion
  path on :class:`strata_forge.llm.LLMClient` is untouched.

## Test expectations

- Unit tests under ``tests/unit/rag/``, one file per source
  module.
- Coverage target: ≥ 90 % line.
- :class:`LiteLLMEmbedder` is tested with a mocked
  ``litellm.aembedding`` via :func:`monkeypatch.setattr`. No live
  network in unit tests.
- Phase 4.2's :class:`QdrantVectorStore` is tested with a mocked
  Qdrant client; one optional ``@pytest.mark.integration`` test
  exercises a live Qdrant from ``docker compose``.

## Gotchas

- **Don't grow :class:`LLMClient` to include embeddings.** The
  seam stays narrow on purpose (ADR 0012). New embedding
  features land on :class:`LiteLLMEmbedder` or a sibling embedder.
- **Don't duplicate :class:`VectorStore` in another module.** It
  used to live in :mod:`strata_forge.agents.memory`; ADR 0012 moved it
  here. Future contributors who reach for "let me define a
  vector store interface in my module" should reuse this one.
- **Chunk overlap counts against chunk_size.** ``chunk_overlap``
  trims the *start* of subsequent chunks by ``chunk_overlap``
  characters from the prior boundary — but each chunk's
  *length* still respects ``chunk_size``. Test against a long
  document to confirm the behaviour matches expectations.

## When to update this file

- Adding a new public class/function to ``__init__.py``.
- Adding a new Protocol or relocating an existing one.
- Adding a new vector-store backend, retriever, or reranker.
- Adding a new dependency to the ``[rag]`` extra.
