# ADR 0012 — RAG module owns the vector-store / embedder / chunker / retriever Protocols

**Status:** Accepted
**Date:** Initial scaffolding for `strata_forge.rag`
**Supersedes:** —
**Superseded by:** —

## Context

Phase 3.3 shipped a :class:`VectorStore` Protocol inside
:mod:`strata_forge.agents.memory` so that :class:`EpisodicMemory` had
something to bind against. That was the right shape for the agent
module, but the wrong **location** — the same Protocol is needed by
the RAG retriever in Phase 4, and "rag depends on agents for its
core abstraction" is a backwards dependency arrow.

The CLAUDE.md project map (when it was written for Phase 3.3)
already anticipated this: it described `strata_forge.agents` as "imports
from `strata_forge.rag` when its vector-store Protocol lands." Phase 4.1
is the moment to make that real.

In addition to the relocation, Phase 4 needs three more typed
Protocols to anchor the rest of the module: an :class:`Embedder`, a
:class:`Chunker`, and a :class:`Retriever`. Each one has the same
shape question as `VectorStore` had — does the abstraction live in
`strata_forge.rag` (with concrete impls in `strata_forge.rag` and elsewhere), or
does it live in some other module?

## Decision

### 1. The vector-store primitives relocate to `strata_forge.rag`

`VectorStore`, `VectorItem`, `VectorSearchResult`, the
in-process `InMemoryVectorStore`, and `cosine_similarity` all
move from :mod:`strata_forge.agents.memory.vector_store` to
:mod:`strata_forge.rag.vector_store`.

`strata_forge.agents.memory` re-exports the relocated names from
:mod:`strata_forge.rag` so the agent-facing public API (`from
strata_forge.agents import VectorStore, InMemoryVectorStore, …`) keeps
working. `strata_forge.agents.memory.episodic` imports them from
:mod:`strata_forge.rag`. The dependency arrow points one way:
`agents → rag`.

### 2. Four Protocols anchor the RAG module

```python
@runtime_checkable
class Embedder(Protocol):
    @property
    def model(self) -> str: ...
    async def embed(self, text: str) -> Sequence[float]: ...
    async def embed_batch(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

@runtime_checkable
class Chunker(Protocol):
    def chunk(self, document: Document) -> Sequence[Chunk]: ...

@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, query: str, *, top_k: int = 5) -> Sequence[RetrievalResult]: ...

# VectorStore Protocol relocated from strata_forge.agents.memory; same shape.
```

Each Protocol is `runtime_checkable` so duck-typed implementations
satisfy `isinstance` checks without inheritance. The concrete
implementations shipped here all happen to be classes (for
config-via-constructor ergonomics), but the runtime contract is
the Protocol.

### 3. `Embedder` is its own seam, not a method on `LLMClient`

`strata_forge.llm.LLMClient` doesn't grow an `aembedding`-style method.
Embeddings are a separate axis from chat completions — different
models, different pricing, different cache keys, different
batching semantics — and routing them through `LLMClient` would
either bloat the client or pretend they're symmetric when they
aren't.

Instead, `strata_forge.rag.LiteLLMEmbedder` calls `litellm.aembedding`
directly. LiteLLM already provides the provider-agnostic surface
we want; the embedder is a thin async wrapper that keeps the
Protocol surface clean.

If we later need provider routing, fallback, or caching for
embeddings, those land as features of `LiteLLMEmbedder` (or
sibling embedders) — not as `LLMClient.aembedding`.

### 4. `Chunk` and `Document` are frozen Pydantic

```python
class Document(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = {}

class Chunk(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = {}
    document_id: str | None = None
```

Same shape rules as :mod:`strata_forge.datasets` and :mod:`strata_forge.evals`:
`extra="forbid"`, `frozen=True`, dict metadata for free-form
annotations. Chunks carry a back-reference to their source
`Document.id` when chunked from one; standalone chunks (e.g.
external snippets injected into the index) leave it `None`.

### 5. Composition is by Protocol, not by inheritance

A `DenseRetriever` (Phase 4.2) holds an `Embedder` and a
`VectorStore`. A `HybridRetriever` (Phase 4.3) holds two
`Retriever`s and a fusion strategy. The `RAGPipeline`
(Phase 4.4) holds a `Chunker`, an `Embedder`, a `VectorStore`,
and a `Retriever` plus an optional `Reranker`. None of them
extend a shared base class.

This means users can drop in their own implementations of any
Protocol — a custom BM25 retriever, a non-Qdrant store, a
proprietary embedder — without subclassing anything.

## Consequences

**Positive**

- **Dependency arrow points the right way.** `agents → rag`
  matches the architecture document. `EpisodicMemory` uses RAG's
  vector store; the RAG pipeline doesn't know agents exist.
- **One source of truth.** Adding a new VectorStore backend
  (Weaviate, pgvector) means one place: `strata_forge.rag.stores.*`. The
  agent's episodic memory benefits for free.
- **Embedder seam stays narrow.** `LLMClient` remains focused on
  chat completions, tool use, and structured output. Embedding is
  a separate concern with separate cost / routing / batching
  semantics.
- **Protocol-based composition.** Users can swap any piece without
  inheriting from Forge classes. The shipped concrete impls are
  conveniences, not requirements.

**Negative**

- **One-time migration.** Existing code that imports
  `VectorStore` etc. from `strata_forge.agents.memory` still works
  (re-exports), but the canonical path is `strata_forge.rag`. The
  re-exports are documented in :mod:`strata_forge.agents.memory` and we
  may deprecate them in a later phase.
- **`LiteLLMEmbedder` duplicates litellm-wrapping work** that
  `LLMClient` already does for chat completions (retry, error
  mapping, optional caching). Phase 4.1 ships the minimal
  embedder; richer features (`@retry`, provider fallback, the
  cache backend from `strata_forge.llm.cache`) can be added in later
  RAG sub-phases if the demand surfaces.

## Alternatives considered

1. **Keep `VectorStore` in `strata_forge.agents.memory`; have `strata_forge.rag`
   import from `strata_forge.agents`.** Inverts the dependency arrow
   from the CLAUDE.md spec. `strata_forge.rag` would become a leaf module
   that depends on `strata_forge.agents` — wrong direction for the
   layered architecture. Rejected.

2. **Put `VectorStore` in `strata_forge.core`.** Maximally neutral but
   `core` is supposed to be lightweight + dep-free; pulling a
   storage abstraction into it pollutes the foundation. Rejected.

3. **Add `LLMClient.aembedding`.** Symmetric with `acompletion`
   but the cost-model / batching / routing semantics differ
   enough that bolting it onto the same client would obscure the
   differences. Rejected — `Embedder` is its own seam.

4. **Drop the `VectorStore` Protocol entirely; ship one concrete
   `QdrantVectorStore`.** Loses pluggability for the test /
   prototyping path and for future backends. Rejected.
