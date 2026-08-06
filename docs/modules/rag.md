# `strata_forge.rag` — embeddings, vector stores, chunking, retrieval, reranking, pipeline

`strata_forge.rag` is the retrieval-augmented generation layer. It ships
four Protocols, concrete in-process and production-backend
implementations, and a composable :class:`RAGPipeline` that ties
them together. See
[ADR 0012](../architecture/adr/0012-rag-protocols-and-vector-store-relocation.md)
for the design rationale (Protocol-based composition, embedder
seam, vector-store relocation).

Integration points:

- **Protocols:** `Embedder`, `Chunker`, `Retriever`, `VectorStore`,
  `Reranker`. Each is `runtime_checkable`.
- **Embedders:** `LiteLLMEmbedder` (provider-agnostic via
  `litellm.aembedding`).
- **Chunkers:** `RecursiveChunker` (paragraph → newline → sentence
  → word fallback with character-offset metadata).
- **Vector stores:** `InMemoryVectorStore` (in-process,
  dep-free), `QdrantVectorStore` (lazy `[rag]` extra).
- **Retrievers:** `DenseRetriever` (Embedder + VectorStore),
  `BM25Retriever` (pure-Python Okapi BM25), `HybridRetriever`
  (RRF fusion of multiple retrievers).
- **Rerankers:** `CohereReranker` (lazy `[rag]` extra),
  `CrossEncoderReranker` (lazy `sentence-transformers`).
- **Pipeline:** `RAGPipeline` (chunk → retrieve → optional rerank →
  optional prompt augmentation).

Module rules: [`src/strata_forge/rag/CLAUDE.md`](../../src/strata_forge/rag/CLAUDE.md).
Source: [`src/strata_forge/rag/`](../../src/strata_forge/rag/).

---

## Contents

- [Quickstart](#quickstart)
- [Protocols and shapes](#protocols-and-shapes)
- [Embedders](#embedders)
- [Chunkers](#chunkers)
- [Vector stores](#vector-stores)
- [Retrievers](#retrievers)
- [Rerankers](#rerankers)
- [The RAG pipeline](#the-rag-pipeline)
- [Lazy-import contract](#lazy-import-contract)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```python
import asyncio
from strata_forge.rag import (
    DenseRetriever,
    Document,
    InMemoryVectorStore,
    LiteLLMEmbedder,
    RAGPipeline,
    RecursiveChunker,
)

async def main() -> None:
    embedder = LiteLLMEmbedder(model="text-embedding-3-small")
    store = InMemoryVectorStore()
    retriever = DenseRetriever(embedder=embedder, store=store)
    pipeline = RAGPipeline(
        chunker=RecursiveChunker(chunk_size=500, chunk_overlap=50),
        retriever=retriever,
    )
    await pipeline.ingest([
        Document(id="d1", text="The Sun is the star at the centre of the Solar System..."),
    ])
    results = await pipeline.query("What is the Sun?", top_k=3)
    for r in results:
        print(r.score, r.chunk.text)

asyncio.run(main())
```

End-to-end demos:

- [`examples/26_rag_basic.py`](../../examples/26_rag_basic.py)
  — in-process pipeline.
- [`examples/27_rag_qdrant.py`](../../examples/27_rag_qdrant.py)
  — same pipeline backed by Qdrant.
- [`examples/28_rag_hybrid_reranked.py`](../../examples/28_rag_hybrid_reranked.py)
  — dense + BM25 fusion with optional Cohere reranking.

---

## Protocols and shapes

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

class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float

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

@runtime_checkable
class VectorStore(Protocol):
    async def add(self, items: Sequence[VectorItem]) -> None: ...
    async def search(self, embedding: Sequence[float], *, top_k: int = 5) -> tuple[VectorSearchResult, ...]: ...
    async def delete(self, ids: Sequence[str]) -> None: ...
    async def clear(self) -> None: ...

@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self, query: str, results: Sequence[RetrievalResult], *, top_k: int | None = None,
    ) -> tuple[RetrievalResult, ...]: ...
```

`Document` and `Chunk` are frozen Pydantic with `extra="forbid"`.
`RetrievalResult` likewise. `VectorItem` / `VectorSearchResult` are
frozen dataclasses (lower overhead at construction). Anything
satisfying a Protocol — including user-supplied custom
implementations — drops into the pipeline without inheriting from
Forge classes.

---

## Embedders

`LiteLLMEmbedder` is the default. It calls `litellm.aembedding`
directly so the same code targets OpenAI, Cohere, Voyage, …
without per-provider wiring.

```python
from strata_forge.rag import LiteLLMEmbedder

embedder = LiteLLMEmbedder(
    model="text-embedding-3-small",
    dimensions=512,                      # OpenAI dim override
    provider_extras={"input_type": "search_query"},  # Cohere knob
)
vec = await embedder.embed("hello")
batch = await embedder.embed_batch(["a", "b", "c"])
```

The embedder doesn't go through `LLMClient` (ADR 0012). Build
your own `Embedder` implementation if you need provider routing,
caching, or fallback at the embedding layer.

---

## Chunkers

`RecursiveChunker` is the default. Paragraph-aware, with a
fallback chain (`"\n\n"` → `"\n"` → `". "` → `" "` → hard split)
and configurable character-based overlap.

```python
from strata_forge.rag import RecursiveChunker, Document

chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=100)
chunks = chunker.chunk(Document(id="doc-1", text="..."))
# Each chunk's metadata carries:
#   character_start, character_end, chunk_index
# plus any metadata the source Document had.
```

The default separator chain works well for prose; pass
`separators=(...)` to tune for code, transcripts, or markup.

---

## Vector stores

Two backends ship; both satisfy the `VectorStore` Protocol.

```python
from strata_forge.rag import InMemoryVectorStore, QdrantVectorStore

# Tests, prototyping, small-scale agent runs.
store = InMemoryVectorStore()

# Production. Lazy [rag] extra (qdrant-client).
store = QdrantVectorStore(
    collection_name="my-collection",
    embedding_dimensions=1536,
    host="localhost", port=6333,
    # or: url="https://my-cluster.example.com", api_key="..."
)
```

`QdrantVectorStore` auto-creates the collection on first use;
`clear()` drops and recreates it. The store maps `VectorItem.id`
(free-form string) to a deterministic UUIDv5 for Qdrant's point id
and stashes the original on the payload, so search results
round-trip the chunk text + metadata cleanly.

---

## Retrievers

Three retrievers ship; all satisfy the `Retriever` Protocol.

### `DenseRetriever`

Embeds the query and looks it up in a `VectorStore`. Provides an
`index(chunks)` helper for one-shot batch ingestion.

```python
from strata_forge.rag import DenseRetriever

retriever = DenseRetriever(embedder=embedder, store=store)
await retriever.index(chunks)
results = await retriever.retrieve("query", top_k=5)
```

### `BM25Retriever`

Pure-Python Okapi BM25. Indexes the corpus at construction time;
not re-indexable.

```python
from strata_forge.rag import BM25Retriever

retriever = BM25Retriever(chunks, k1=1.5, b=0.75)
results = await retriever.retrieve("exact keyword query", top_k=5)
```

Best for: keyword overlap, rare proper nouns, identifiers,
numeric tokens that semantic embeddings tend to drown out.

### `HybridRetriever`

Fuses two or more retrievers via Reciprocal Rank Fusion (RRF).

```python
from strata_forge.rag import HybridRetriever

hybrid = HybridRetriever(
    [dense, sparse],
    weights=[1.0, 1.0],            # equal weight
    rrf_k=60,                      # standard RRF constant
    per_retriever_top_k=20,        # widen each retriever's candidate pool
)
results = await hybrid.retrieve("query", top_k=5)
```

Children are queried concurrently via `asyncio.gather`. Set
per-retriever weights to bias the fusion when one signal is known
to be stronger.

---

## Rerankers

Re-score a retrieval list with a more expensive model.

### `CohereReranker`

Cohere's hosted Rerank API. Lazy `cohere` SDK behind the `[rag]`
extra; reads `COHERE_API_KEY` from the environment by default.

```python
from strata_forge.rag import CohereReranker

reranker = CohereReranker(model="rerank-english-v3.0")
top_3 = await reranker.rerank("query", candidates, top_k=3)
```

### `CrossEncoderReranker`

Local cross-encoder via `sentence-transformers`. Heavy (~2 GB
disk via torch) — not in the default `[rag]` extra; install
separately:

```
pip install sentence-transformers
```

```python
from strata_forge.rag import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu",
)
top_3 = await reranker.rerank("query", candidates, top_k=3)
```

The sync `predict` call is wrapped in `asyncio.to_thread` so it
doesn't block the event loop.

---

## The RAG pipeline

`RAGPipeline` glues the pieces together.

```python
from strata_forge.rag import RAGPipeline

pipeline = RAGPipeline(
    chunker=RecursiveChunker(chunk_size=500),
    retriever=DenseRetriever(embedder=embedder, store=store),
    reranker=CohereReranker(),    # optional
)

# Chunk + index a batch of documents.
n_chunks = await pipeline.ingest([Document(id=..., text=...), ...])

# Query → retrieve → (optional) rerank.
results = await pipeline.query(
    "What is X?",
    top_k=3,
    rerank_top_k=20,        # widen the candidate pool when reranking
)

# Get a prompt-ready string with retrieved context interpolated.
prompt = await pipeline.augment_prompt("What is X?", top_k=3)
```

`pipeline.ingest()` requires an `IndexableRetriever` — any
retriever with an `index(chunks)` coroutine. `DenseRetriever`
satisfies that. Non-indexable retrievers (`BM25Retriever`,
`HybridRetriever`) need to be constructed with their corpus
upfront; calling `ingest` on a non-indexable pipeline raises
`TypeError` with a clear remediation message.

---

## Lazy-import contract

`import strata_forge.rag` works without any optional extras installed.
SDK imports happen lazily inside the functions that need them:

- `QdrantVectorStore` → `[rag]` extra (`qdrant-client`).
- `CohereReranker` → `[rag]` extra (`cohere`).
- `CrossEncoderReranker` → `sentence-transformers` installed
  separately.
- `LiteLLMEmbedder` uses `litellm` (already in core deps via the
  LLM client).

`ImportError` surfaces only when a caller actually invokes the
relevant function without the extra installed; the message names
the install command explicitly.

---

## Troubleshooting

- **"`The [rag] extra is required for ...`"**: install with
  `pip install 'ai-forge[rag]'`.
- **"`RAGPipeline.ingest: retriever of type X doesn't satisfy IndexableRetriever`"**:
  you passed a non-indexable retriever (BM25, hybrid). Either pre-build the
  retriever with its corpus and skip `pipeline.ingest()`, or wire a
  `DenseRetriever` instead.
- **Qdrant collection dimensionality mismatch**: when you reuse a
  collection name, its existing schema's vector size must match the
  embedder's output. Either pass `embedding_dimensions=` matching
  the existing collection or call `store.clear()` to drop + recreate.
- **Hybrid retrieval surfaces strange documents**: try
  `per_retriever_top_k=20-50` to widen the candidate pool, then
  rerank with a Cohere or cross-encoder reranker to get a precise
  top-3.
- **Cross-encoder very slow**: `predict` runs on a single thread.
  Pass `device="cuda"` (or `"mps"` on Apple Silicon) and batch your
  retrieval calls so the reranker amortizes the model load.
