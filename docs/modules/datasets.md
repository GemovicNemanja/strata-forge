# `forge.datasets` — typed dataset shapes, versioning, and pluggable backends

`forge.datasets` is the dataset layer the eval runner consumes. It
ships a small core of typed Pydantic models, content-hash versioning,
an abstract `DatasetStore` plus two concrete backends, and a
bidirectional bridge to Hugging Face Datasets. Two synthetic-data
helpers built on `forge.llm` round out the surface. See
[ADR 0009](../architecture/adr/0009-datasets-langfuse-canonical-hf-exchange.md)
for the canonical-store + exchange-format decision.

Five integration points ship:

- `Dataset` / `DatasetItem` — frozen Pydantic models with
  content-hash IDs and tuple-typed item collections (structural
  immutability).
- `dataset_version` / `diff` / `DatasetDelta` — versioning by
  canonical content hash plus a partition into
  added / removed / unchanged.
- `DatasetStore` (abstract) + `InMemoryDatasetStore` /
  `LangfuseDatasetStore` — pluggable persistence with content-hash
  versioning on both backends.
- `to_hf_dataset` / `from_hf_dataset` — bidirectional Hugging Face
  Datasets bridge (lazy import behind the `[hf]` extra).
- `self_instruct` / `distill` — async synthetic-data helpers built
  on `LLMClient`.

Module rules: [`src/forge/datasets/CLAUDE.md`](../../src/forge/datasets/CLAUDE.md).
Source: [`src/forge/datasets/`](../../src/forge/datasets/).

---

## Contents

- [Quickstart](#quickstart)
- [Data shapes](#data-shapes)
- [Versioning and diffing](#versioning-and-diffing)
- [Stores](#stores)
- [The Hugging Face bridge](#the-hugging-face-bridge)
- [Synthetic-data primitives](#synthetic-data-primitives)
- [Lazy-import contract](#lazy-import-contract)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```python
import asyncio
from forge.datasets import (
    Dataset,
    DatasetItem,
    InMemoryDatasetStore,
    dataset_version,
    diff,
)

async def main() -> None:
    items = (
        DatasetItem.from_input(
            {"question": "What is the capital of France?"},
            expected_output="Paris",
        ),
        DatasetItem.from_input(
            {"question": "What is 2 + 2?"},
            expected_output="4",
        ),
    )
    ds = Dataset(name="trivia", items=items)

    store = InMemoryDatasetStore()
    version = await store.put(ds)          # content-hash
    retrieved = await store.get("trivia")   # latest
    assert retrieved == ds

asyncio.run(main())
```

End-to-end demos:

- [`examples/17_dataset_basics.py`](../../examples/17_dataset_basics.py)
  — schema, store, versioning, diff.
- [`examples/18_dataset_hf_bridge.py`](../../examples/18_dataset_hf_bridge.py)
  — Forge ↔ Hugging Face Datasets round-trip.
- [`examples/19_dataset_synthetic.py`](../../examples/19_dataset_synthetic.py)
  — `self_instruct` + `distill` against a real LLM.

---

## Data shapes

Both models are Pydantic v2 with `frozen=True` and
`extra="forbid"` — wire shapes are stable, attribute reassignment
raises `pydantic.ValidationError`.

### `DatasetItem`

```python
class DatasetItem(BaseModel):
    id: str
    input: dict[str, Any]
    expected_output: Any | None = None
    metadata: dict[str, Any] = Field(default={})
```

Construct directly with an explicit `id`, or via
`DatasetItem.from_input(...)` which derives a SHA-256 content hash
from `input + expected_output`. The hash **excludes** `metadata` so
re-annotating an item (tier tags, source provenance, …) doesn't
invalidate its ID.

```python
a = DatasetItem.from_input({"q": "hi"}, expected_output="hello")
b = DatasetItem.from_input({"q": "hi"}, expected_output="hello",
                            metadata={"tier": "easy"})
assert a.id == b.id  # metadata doesn't change the ID
```

### `Dataset`

```python
class Dataset(BaseModel):
    name: str = Field(min_length=1)
    items: tuple[DatasetItem, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = Field(default={})
```

Item collections are `tuple[...]`, not `list[...]` — Pydantic's
`frozen=True` only prevents attribute reassignment; a tuple field
also prevents in-place `.append` mutation through the attribute.
Construction fails with `forge.core.errors.ValidationError` when two
items share an `id`.

---

## Versioning and diffing

A dataset's version is the SHA-256 of its canonical content:

```python
from forge.datasets import dataset_version

version = dataset_version(ds)  # hashes name + sorted item IDs + metadata
```

The hash **excludes** `description` (free-form prose) and per-item
`metadata` (re-tagging items shouldn't invalidate cached eval
results). Two structurally-identical datasets produce the same
version regardless of how they were assembled — natural
deduplication.

`diff(old, new)` returns a `DatasetDelta` partitioning items by ID:

```python
from forge.datasets import diff

delta = diff(v1, v2)
print(f"added: {len(delta.added)}, removed: {len(delta.removed)}")
print(f"unchanged: {len(delta.unchanged)}")  # carries the NEW item (any metadata updates visible)
if delta.is_empty:
    ...  # no IDs added or removed; metadata may still have changed
```

---

## Stores

Both backends implement the same async ABC:

```python
class DatasetStore(ABC):
    async def get(self, name: str, version: str | None = None) -> Dataset: ...
    async def put(self, dataset: Dataset) -> str: ...
    async def versions(self, name: str) -> list[str]: ...           # newest first
    async def list_names(self) -> list[str]: ...                     # sorted
    async def delete(self, name: str, version: str | None = None) -> None: ...
```

`get(version=None)` returns the latest version. `delete(version=None)`
removes every version of the name. Deleting an unknown name is a
no-op; deleting a known name with an unknown version raises
`DatasetNotFoundError`.

### `InMemoryDatasetStore`

Dict-backed, content-hash versioning, natural dedup. Good for
tests, notebooks, and prototyping.

```python
from forge.datasets import InMemoryDatasetStore

store = InMemoryDatasetStore()
v = await store.put(ds)
again = await store.put(ds)
assert v == again  # identical content -> same version
```

### `LangfuseDatasetStore`

Persists into Langfuse Datasets. Each Forge version is a distinct
Langfuse dataset named `{forge_name}__v{version}` — visible in the
Langfuse UI and addressable via the Langfuse SDK without
out-of-band bookkeeping.

```python
from forge.datasets import LangfuseDatasetStore

# Uses LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY by default.
store = LangfuseDatasetStore()

# Or inject an already-built client (useful for tests, or when sharing
# the Langfuse client with forge.tracing).
store = LangfuseDatasetStore(client=my_langfuse_client)
```

The Langfuse SDK is sync; the store wraps every call in
`asyncio.to_thread` to keep the public surface async-only. Requires
the `[langfuse]` extra:

```
pip install 'ai-forge[langfuse]'
```

Importing `forge.datasets.stores.langfuse` works without the extra —
the `ImportError` surfaces only when a caller actually constructs
the store and triggers the lazy SDK import.

---

## The Hugging Face bridge

```python
from forge.datasets import to_hf_dataset, from_hf_dataset

hf_ds = to_hf_dataset(forge_dataset)
# ... ship to HF Hub, write to parquet, hand to the training pipeline ...

# Ingest any HF Dataset back:
forge_ds = from_hf_dataset(
    hf_ds,
    name="my-set",
    description="optional",
    metadata={"source": "hf-hub"},
    # Custom column names for non-Forge HF datasets:
    id_column="uuid",
    input_column="prompt",
    expected_output_column="answer",
    metadata_column=None,  # ignore even if a 'metadata' column exists
)
```

`to_hf_dataset` produces an HF `Dataset` with four columns —
`id`, `input`, `expected_output`, `metadata` — and stashes the
Forge-level `name` / `description` on `Dataset.info`.

`from_hf_dataset` does the inverse. Rows without an `id_column`
value (missing column OR per-row `None`) get content-hash IDs via
`DatasetItem.from_input`.

Requires the `[hf]` extra: `pip install 'ai-forge[hf]'`. Importing
`forge.datasets.hf_bridge` works without it; the `ImportError`
surfaces only when a caller invokes `to_hf_dataset`.

---

## Synthetic-data primitives

Both helpers are pure async functions that return a fresh `Dataset` —
no mutation of inputs.

### `self_instruct`

Generates new items from seed examples by prompting the LLM with
the seeds and an instruction. Batches generation via
`LLMClient.complete_structured` against a `SelfInstructBatch`
wrapper schema, dedupes by content-hash ID against seeds + prior
batches, and bails out early when a whole batch yields no novel
items. Capped by `max_attempts`.

```python
from forge.datasets import self_instruct
from forge.llm.client import LLMClient

client = LLMClient(model="claude-opus-4-7", provider="anthropic")
new_items = await self_instruct(
    seeds=existing_dataset,
    instructions="produce one-fact trivia Q&A items in the same shape",
    n=50,
    client=client,
    name="trivia-synth",
    batch_size=10,
    temperature=0.9,
)
```

### `distill`

Runs a teacher LLM over each item's input to populate
`expected_output`. Concurrency-bounded via an `asyncio.Semaphore`;
preserves input order. `skip_existing=True` (default) means items
that already carry a non-`None` `expected_output` pass through
unchanged.

```python
from forge.datasets import distill

labeled = await distill(
    inputs=unlabeled_dataset,
    teacher=client,
    name="labeled-set",
    concurrency=5,
    skip_existing=True,
)
```

---

## Lazy-import contract

`import forge.datasets` works without `[langfuse]` or `[hf]`
installed. Both SDKs are imported lazily inside the functions that
need them; the `ImportError` surfaces only when a caller actually
constructs a `LangfuseDatasetStore` or invokes `to_hf_dataset`.

The error message names the missing extra so the fix is obvious:

```
ImportError: The [hf] extra is required for hf_bridge.
Install it with: pip install 'ai-forge[hf]'.
```

---

## Troubleshooting

- **"`Langfuse is not configured`"**: the `LangfuseDatasetStore`
  needs `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, and
  `LANGFUSE_SECRET_KEY`. Either set them in `.env` / process env, or
  pass an explicit `client=...` to the constructor.
- **Same dataset content, different versions**: check that you
  aren't perturbing `metadata` at the dataset level (per-item
  metadata is fine — only dataset-level `metadata` enters the hash).
- **`DatasetNotFoundError` on `delete`**: deleting an unknown
  *name* is a no-op; deleting a known name with an unknown
  *version* raises. Drop the `version=` argument to remove every
  version of the name.
- **`self_instruct` returns fewer items than `n`**: the LLM ran out
  of novel patterns within `max_attempts`. Increase `max_attempts`,
  raise `temperature`, or supply more diverse seeds.
- **`distill` raises**: the teacher's `LLMClient.complete` exception
  propagates verbatim. Inspect via `forge.llm.errors` —
  `ProviderError` subclasses tell you exactly what failed.
