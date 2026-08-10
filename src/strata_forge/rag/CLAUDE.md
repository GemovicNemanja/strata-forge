# Agent rules — strata_forge.rag

`strata_forge.rag` is the retrieval-augmented generation layer. It owns
five Protocols — :class:`Embedder`, :class:`VectorStore`,
:class:`Chunker`, :class:`Retriever`, :class:`Reranker` — plus the
:class:`IndexableRetriever` refinement in ``pipeline.py`` and concrete
in-process and production implementations of each. See
[ADR 0012](../../../docs/architecture/adr/0012-rag-protocols-and-vector-store-relocation.md)
for the design rationale.

## Purpose

- :class:`Embedder` Protocol + :class:`LiteLLMEmbedder` —
  provider-agnostic embedding via ``litellm.aembedding``.
- :class:`Document` + :class:`Chunk` shapes + :class:`Chunker`
  Protocol + :class:`RecursiveChunker` — chunking primitives.
- :class:`Retriever` Protocol + :class:`RetrievalResult` — the
  retrieval contract, implemented by :class:`DenseRetriever`
  (embedding similarity over a vector store), :class:`BM25Retriever`
  (dep-free Okapi BM25 over an in-process corpus), and
  :class:`HybridRetriever` (reciprocal-rank fusion of the two).
- :class:`VectorStore` Protocol + :class:`InMemoryVectorStore` +
  :class:`QdrantVectorStore` + :class:`VectorItem` /
  :class:`VectorSearchResult` — vector storage primitives. Relocated
  from :mod:`strata_forge.agents.memory` per ADR 0012; the original
  module re-exports them for back-compatibility.
- :class:`Reranker` Protocol + :class:`CohereReranker` and
  :class:`CrossEncoderReranker` — second-stage reordering.
- :class:`RAGPipeline` — composable chunk → embed → store →
  retrieve → rerank → augment flow, plus
  :data:`DEFAULT_AUGMENT_TEMPLATE`.

## Boundaries

- **Owns:** a flat file layout — ``embedding.py``, ``chunking.py``,
  ``retrieval.py``, ``vector_store.py``, ``dense.py``, ``bm25.py``,
  ``hybrid.py``, ``qdrant.py``, ``rerankers.py``, ``pipeline.py``.
  There is no ``stores/`` or ``rerankers/`` package — a new
  retriever, store, or reranker is a new top-level
  ``rag/<name>.py``.
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
- **External deps:** Pydantic at module load; ``litellm`` (in core)
  for :class:`LiteLLMEmbedder`. The ``[rag]`` extra carries
  ``qdrant-client`` (:class:`QdrantVectorStore`) and ``cohere``
  (:class:`CohereReranker`), both lazy-imported inside the methods
  that use them. :class:`CrossEncoderReranker` lazy-imports
  ``sentence-transformers``, which is *not* in ``[rag]`` — it
  transitively pulls torch, so it is installed separately.
  :class:`BM25Retriever` is pure Python with no dependency at all.

## Public API

The module's ``__init__.py`` re-exports exactly these symbols
(mirror any change here into ``__all__``):

- Protocols: :class:`Embedder`, :class:`Chunker`,
  :class:`Retriever`, :class:`VectorStore`, :class:`Reranker`,
  :class:`IndexableRetriever`.
- Shapes: :class:`Document`, :class:`Chunk`,
  :class:`RetrievalResult`, :class:`VectorItem`,
  :class:`VectorSearchResult`.
- Embedders: :class:`LiteLLMEmbedder`.
- Chunkers: :class:`RecursiveChunker`.
- Stores: :class:`InMemoryVectorStore`, :class:`QdrantVectorStore`.
- Retrievers: :class:`DenseRetriever`, :class:`BM25Retriever`,
  :class:`HybridRetriever`.
- Rerankers: :class:`CohereReranker`, :class:`CrossEncoderReranker`.
- Pipeline: :class:`RAGPipeline`, :data:`DEFAULT_AUGMENT_TEMPLATE`.
- Helpers: :func:`cosine_similarity`, :func:`tokenize`.

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
  GIL-free path; the :class:`Chunker` Protocol is deliberately
  synchronous.
- **No new LLM client.** Embeddings call ``litellm.aembedding``
  directly through :class:`LiteLLMEmbedder`; the chat-completion
  path on :class:`strata_forge.llm.LLMClient` is untouched.

## Test expectations

- Unit tests under ``tests/unit/rag/``, one file per source
  module.
- Coverage: the enforced gate is the repo-wide 85 % line floor
  (``fail_under`` in ``pyproject.toml``); treat a drop in this module
  as a regression.
- :class:`LiteLLMEmbedder` is tested with a mocked
  ``litellm.aembedding`` via :func:`monkeypatch.setattr`. No live
  network in unit tests.
- :class:`QdrantVectorStore` is tested with a mocked Qdrant client;
  one optional ``@pytest.mark.integration`` test exercises a live
  Qdrant from ``docker compose``.

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
