# Build a RAG pipeline over Qdrant

`strata_forge.rag` is five Protocols — `Chunker`, `Embedder`, `VectorStore`, `Retriever`,
`Reranker` — plus concrete implementations of each and a `RAGPipeline` that composes them. The
Protocols are the point: you start with everything in process, swap the store for Qdrant without
touching the rest, then add a second retrieval signal and a reranker the same way.

This recipe walks that progression: chunk, embed, store, retrieve densely, add BM25 and fuse,
then rerank.

**Prerequisites.** `pip install 'strata-forge[rag]'` for the Qdrant client and the Cohere
reranker, plus credentials for whichever embedding model you use. A local Qdrant comes from
`make stack-up` (Postgres, Qdrant, Redis via `docker/compose.yaml`); Qdrant listens on
`localhost:6333`. `BM25Retriever` and `RecursiveChunker` are pure Python and need no extra at
all.

**Running the snippets.** A block that ends in `asyncio.run(main())` is a complete program. The
rest are fragments: put them inside an `async def main()` and invoke it the same way, or paste them
into a notebook cell, where top-level `await` is legal.

---

## 1. Chunk

`RecursiveChunker` splits on progressively finer separators — paragraph break, newline, sentence
break, space — and falls back to a fixed-size hard split when none of them match. Every `Chunk`
carries its `document_id` and its offsets in `metadata`, so a retrieved chunk always traces back
to its source.

```python
from strata_forge.rag import Document, RecursiveChunker

documents = [
    Document(id="rfc-6585", text="HTTP status code 429 means too many requests. Servers "
                                 "should include a Retry-After header."),
    Document(id="rfc-7231", text="503 Service Unavailable indicates a temporary overload "
                                 "or scheduled maintenance."),
]

chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
chunks = [chunk for doc in documents for chunk in chunker.chunk(doc)]
```

Note that `chunk_overlap` extends each chunk's start backwards without moving its end, so the
effective maximum chunk length is `chunk_size + chunk_overlap` characters. Size your budget
against that number, not against `chunk_size` alone.

## 2. Embed

`LiteLLMEmbedder` calls `litellm.aembedding` directly rather than routing through `LLMClient`.
That is deliberate — embeddings have a different cost model, a different response shape, and no
tool or streaming surface, so
[ADR 0012](../architecture/adr/0012-rag-protocols-and-vector-store-relocation.md) keeps them off
the chat client. Any model id LiteLLM recognises works, and credentials come from that provider's
usual environment variable.

```python
from strata_forge.rag import LiteLLMEmbedder

embedder = LiteLLMEmbedder("text-embedding-3-small")  # 1536-dim, OpenAI
```

Pass `dimensions=` for models that support truncation, and `provider_extras=` for
provider-specific knobs (`input_type=` on Cohere, a custom `api_base`, …) — they are forwarded
verbatim.

## 3. Store and retrieve, in process first

`InMemoryVectorStore` plus `DenseRetriever` gets a pipeline working before any infrastructure
exists. `RAGPipeline.ingest` chunks each document and writes the chunks into the retriever;
`query` embeds the question and returns scored `RetrievalResult`s.

```python
import asyncio

from strata_forge.rag import (
    DenseRetriever,
    InMemoryVectorStore,
    RAGPipeline,
)

pipeline = RAGPipeline(
    chunker=chunker,
    retriever=DenseRetriever(embedder=embedder, store=InMemoryVectorStore()),
)

async def main() -> None:
    indexed = await pipeline.ingest(documents)
    print(f"indexed {indexed} chunks")
    for hit in await pipeline.query("what does 429 mean?", top_k=3):
        print(round(hit.score, 4), hit.chunk.document_id, hit.chunk.text[:80])

asyncio.run(main())
```

Full script:
[`examples/27_rag_basic.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/27_rag_basic.py).

## 4. Swap in Qdrant

`QdrantVectorStore` satisfies the same `VectorStore` Protocol, so only the store constructor
changes. The collection is created on first use; `embedding_dimensions` must match your
embedder's output or the writes are rejected by Qdrant.

```python
from strata_forge.rag import QdrantVectorStore

store = QdrantVectorStore(
    collection_name="docs",
    embedding_dimensions=1536,   # must match the embedder
    host="localhost",            # or url=... + api_key=... for Qdrant Cloud
    port=6333,
    distance="cosine",           # "cosine" | "dot" | "euclid"
)

pipeline = RAGPipeline(
    chunker=chunker,
    retriever=DenseRetriever(embedder=embedder, store=store),
)
```

The `qdrant_client` import is lazy — it happens on the first call that needs a connection, not
at construction, and raises `ImportError` with `pip install 'strata-forge[rag]'` when the extra
is missing. Constructing the store is therefore safe in code paths that may never use it.

`await store.clear()` drops every point in the collection, which is what you want between demo
runs and never what you want in production.

Full script:
[`examples/28_rag_qdrant.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/28_rag_qdrant.py).

## 5. Add BM25 and fuse

Dense retrieval finds paraphrases; BM25 finds exact tokens. Error codes, identifiers, product
SKUs, and rare proper nouns are precisely where embeddings blur and keyword matching does not.
`HybridRetriever` fuses any two or more retrievers with Reciprocal Rank Fusion, which combines
ranks rather than scores and so needs no score normalisation between the two systems.

`BM25Retriever` takes its corpus at construction time and is therefore not an
`IndexableRetriever` — a hybrid pipeline chunks once by hand and feeds both sides:

```python
from strata_forge.rag import BM25Retriever, HybridRetriever

chunks = [chunk for doc in documents for chunk in chunker.chunk(doc)]

dense = DenseRetriever(embedder=embedder, store=store)
await dense.index(chunks)          # dense side indexes explicitly
sparse = BM25Retriever(chunks)     # sparse side takes the corpus directly

hybrid = HybridRetriever([dense, sparse], rrf_k=60, per_retriever_top_k=20)
```

`rrf_k` is the RRF constant (60 in the original paper; larger flattens the score distribution).
`per_retriever_top_k` widens each retriever's candidate pool before fusion — worth setting well
above your final `top_k`, since a document ranked 12th by one retriever and 3rd by the other is
exactly the case RRF exists to rescue. `weights=` biases the fusion toward one side.

Calling `RAGPipeline.ingest` on a pipeline whose retriever is not indexable raises `TypeError`
naming the offending type. That is the reminder to index the two sides yourself.

## 6. Rerank

Retrieval is a recall problem; reranking is a precision problem. Pull a wide candidate list
cheaply, then have a cross-encoder or a hosted rerank API score `(query, chunk)` pairs directly.

```python
from strata_forge.rag import CohereReranker

pipeline = RAGPipeline(
    chunker=chunker,
    retriever=hybrid,
    reranker=CohereReranker(model="rerank-english-v3.0"),  # reads COHERE_API_KEY
)

results = await pipeline.query("rate limiting response header", top_k=3, rerank_top_k=20)
```

With a reranker configured, `query` retrieves `rerank_top_k` candidates and reranks them down to
`top_k`. `rerank_top_k` must be greater than or equal to `top_k` — reranking a pool smaller than
the answer set cannot improve anything, so the pipeline rejects it with `ValueError`.

`CrossEncoderReranker` is the local alternative, running a `sentence_transformers` cross-encoder
such as `cross-encoder/ms-marco-MiniLM-L-6-v2`. It is deliberately **not** in the `[rag]` extra
because it pulls `torch`; install it yourself with `pip install sentence-transformers`.

Full script:
[`examples/29_rag_hybrid_reranked.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/29_rag_hybrid_reranked.py).

## 7. Hand the context to a model

`augment_prompt` runs a query and formats the retrieved chunks into a prompt string. The default
template instructs the model to abstain when the sources do not cover the question, which is the
single highest-value line in a RAG prompt:

```python
from strata_forge.llm import LLMClient, Message

prompt = await pipeline.augment_prompt("what does 429 mean?", top_k=4, rerank_top_k=20)
response = await LLMClient("claude-opus-4-8", provider="anthropic").complete(
    [Message.user(prompt)]
)
print(response.text)
```

Override `template=` (any string with `{query}` and `{context}` placeholders) and
`source_separator=` to match your own prompt style; `DEFAULT_AUGMENT_TEMPLATE` is exported if
you want to extend rather than replace it. Nothing forces you through `augment_prompt` at all —
`query` returns the chunks and you are free to build the prompt yourself.

---

## Gotchas

- **Dimension mismatch is silent until it isn't.** A Qdrant collection is created with the
  dimensions you declare. Changing embedding models later means a new collection, not a new
  constructor argument.
- **The chunker counts characters, not tokens.** There is no token-based chunker; size
  `chunk_size` for your model's context with characters-per-token in mind.
- **`BM25Retriever` holds its corpus in memory** and tokenizes it once at construction. It is
  right for a document set you can fit in RAM and re-derive on startup, not for a corpus you
  expect to grow incrementally.
- **Embedding costs are not tracked by `BudgetContext`.** The budget hooks live on `LLMClient`,
  and the embedder bypasses it by design. Meter ingestion separately.

## Where to go next

- [`docs/modules/rag.md`](../modules/rag.md) — every Protocol body, constructor, and lazy-import
  contract.
- [ADR 0012](../architecture/adr/0012-rag-protocols-and-vector-store-relocation.md) — why the
  embedder bypasses `strata_forge.llm` and where the vector-store primitives live.
- [Run a tool-calling agent](run-a-tool-calling-agent.md) — `EpisodicMemory` accepts any
  `VectorStore`, including the Qdrant one you just built.
