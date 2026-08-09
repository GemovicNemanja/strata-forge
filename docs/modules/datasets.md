# `strata_forge.datasets` — typed dataset shapes, versioning, and pluggable backends

`strata_forge.datasets` is the dataset layer the eval runner consumes. It
ships a small core of typed Pydantic models, content-hash versioning,
an abstract `DatasetStore` plus two concrete backends, and a
bidirectional bridge to Hugging Face Datasets. Two synthetic-data
helpers built on `strata_forge.llm` round out the surface. See
[ADR 0009](../architecture/adr/0009-datasets-langfuse-canonical-hf-exchange.md)
for the canonical-store + exchange-format decision.

Five integration points make up the public surface:

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

Two smaller surfaces are public but live on their submodules rather than
the package root: `SelfInstructBatch` / `SelfInstructItem` (the wrapper
schemas `self_instruct` asks the model to fill) in
`strata_forge.datasets.synthetic`, and `VERSION_SEPARATOR` /
`compose_langfuse_name` / `decompose_langfuse_name` (the
`{name}__v{version}` encoding) in
`strata_forge.datasets.stores.langfuse`.

Module rules: [`src/strata_forge/datasets/CLAUDE.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/datasets/CLAUDE.md).
Source: [`src/strata_forge/datasets/`](https://github.com/GemovicNemanja/strata-forge/tree/main/src/strata_forge/datasets/).

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
from strata_forge.datasets import (
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

- [`examples/18_dataset_basics.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/18_dataset_basics.py)
  — schema, store, versioning, diff.
- [`examples/19_dataset_hf_bridge.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/19_dataset_hf_bridge.py)
  — strata-forge ↔ Hugging Face Datasets round-trip.
- [`examples/20_dataset_synthetic.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/20_dataset_synthetic.py)
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
Construction fails with `strata_forge.core.errors.ValidationError` when two
items share an `id`.

---

## Versioning and diffing

A dataset's version is the SHA-256 of its canonical content:

```python
from strata_forge.datasets import dataset_version

version = dataset_version(ds)  # hashes name + sorted item IDs + metadata
```

The hash **excludes** `description` (free-form prose) and per-item
`metadata` (re-tagging items shouldn't invalidate cached eval
results). Two structurally-identical datasets produce the same
version regardless of how they were assembled — natural
deduplication.

`diff(old, new)` returns a `DatasetDelta` partitioning items by ID:

```python
from strata_forge.datasets import diff

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
from strata_forge.datasets import InMemoryDatasetStore

store = InMemoryDatasetStore()
v = await store.put(ds)
again = await store.put(ds)
assert v == again  # identical content -> same version
```

### `LangfuseDatasetStore`

Persists into Langfuse Datasets. Each strata-forge version is a distinct
Langfuse dataset named `{name}__v{version}` — visible in the
Langfuse UI and addressable via the Langfuse SDK without
out-of-band bookkeeping.

```python
from strata_forge.datasets import LangfuseDatasetStore

# Uses LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY by default.
store = LangfuseDatasetStore()

# Or inject an already-built client (useful for tests, or when sharing
# the Langfuse client with strata_forge.tracing).
store = LangfuseDatasetStore(client=my_langfuse_client)
```

The store uses two transports. `get` and `put` go through the (sync)
Langfuse SDK wrapped in `asyncio.to_thread`, so the public surface stays
async-only. `versions`, `list_names`, `delete`, and the latest-version
lookup inside `get` bypass the SDK entirely and call
`{host}/api/public/v2/datasets` over `httpx.AsyncClient`, because SDK v4
dropped the dataset-listing method.

That split has a consequence for `client=`: injecting a client covers the
SDK half only. The REST half resolves its own credentials from
`LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` and raises
`RuntimeError` when they're absent — so an injected client still needs the
env vars set unless you only ever call `put` and `get(name, version)` with
an explicit version.

Requires the `[langfuse]` extra:

```
pip install 'strata-forge[langfuse]'
```

Importing `strata_forge.datasets.stores.langfuse` works without the extra —
the `ImportError` surfaces only when a caller actually constructs
the store and triggers the lazy SDK import.

---

## The Hugging Face bridge

```python
from strata_forge.datasets import to_hf_dataset, from_hf_dataset

hf_ds = to_hf_dataset(forge_dataset)
# ... ship to HF Hub, write to parquet, hand to the training pipeline ...

# Ingest any HF Dataset back:
forge_ds = from_hf_dataset(
    hf_ds,
    name="my-set",
    description="optional",
    metadata={"source": "hf-hub"},
    # Custom column names for HF datasets not written by this bridge:
    id_column="uuid",
    input_column="prompt",
    expected_output_column="answer",
    metadata_column=None,  # ignore even if a 'metadata' column exists
)
```

`to_hf_dataset` produces an HF `Dataset` with four columns —
`id`, `input`, `expected_output`, `metadata` — and stashes the
dataset-level `name` / `description` on `Dataset.info`.

`from_hf_dataset` does the inverse. Rows without an `id_column`
value (missing column OR per-row `None`) get content-hash IDs via
`DatasetItem.from_input`.

**The round trip is lossy in two places, and both are on you to
compensate for.** Dataset-level `metadata` is dropped by `to_hf_dataset`
— only per-item metadata survives, in the `metadata` column. And
`from_hf_dataset` never reads `info.dataset_name`, so you must pass
`name=` back in yourself; omitting it produces a differently-named
dataset and therefore a different `dataset_version`. Round-trip like
this:

```python
hf_ds = to_hf_dataset(ds)
back = from_hf_dataset(hf_ds, name=ds.name, description=ds.description,
                       metadata=ds.metadata)
assert dataset_version(back) == dataset_version(ds)   # holds
assert back == ds                                     # may not
```

The version survives; strict equality may not. Arrow unifies the
`metadata` column into a single struct schema across every row, so an item
that carried no metadata comes back with the union's keys set to `None`.
That's invisible to `dataset_version` (which hashes name, sorted item IDs,
and dataset-level metadata only) and to `diff` (which keys on item ID),
but it will show up if you compare items directly.

Requires the `[hf]` extra: `pip install 'strata-forge[hf]'`. Importing
`strata_forge.datasets.hf_bridge` works without it; the `ImportError`
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
from strata_forge.datasets import self_instruct
from strata_forge.llm import LLMClient

client = LLMClient(model="claude-haiku-4-5", provider="anthropic")
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
from strata_forge.datasets import distill

labeled = await distill(
    inputs=unlabeled_dataset,
    teacher=client,
    name="labeled-set",
    concurrency=5,
    skip_existing=True,
)
```

### Metadata scope on both helpers

The optional `metadata=` argument on `self_instruct` and `distill` sets
**per-item** metadata on the items they produce. The returned dataset's
own `metadata` is hardcoded and cannot be overridden:
`{"synthetic": "self_instruct", "seed_count": N}` and
`{"synthetic": "distillation", "source_count": N}` respectively. Since
dataset-level metadata feeds `dataset_version`, a synthetic dataset's
version moves whenever the seed or source count moves, and it will never
collide with a hand-built dataset of the same name and items.

---

## Lazy-import contract

`import strata_forge.datasets` works without `[langfuse]` or `[hf]`
installed. Both SDKs are imported lazily inside the functions that
need them; the `ImportError` surfaces only when a caller actually
constructs a `LangfuseDatasetStore` or invokes `to_hf_dataset`.

The error message names the missing extra so the fix is obvious:

```
ImportError: The [hf] extra is required for hf_bridge.
Install it with: pip install 'strata-forge[hf]'.
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
  propagates verbatim. Inspect via `strata_forge.llm.errors` —
  `ProviderError` subclasses tell you exactly what failed.
- **`RuntimeError` about missing Langfuse credentials even though you
  passed `client=`**: `versions`, `list_names`, `delete`, and a versionless
  `get` go over REST rather than the SDK and resolve credentials from the
  environment. Set `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` /
  `LANGFUSE_SECRET_KEY` alongside the injected client.

---

## See also

- [`strata_forge.evals`](evals.md) — `run_experiment` takes a `Dataset`
  directly; this is where datasets get consumed.
- [`strata_forge.llm`](llm.md) — the `LLMClient` behind `self_instruct` and
  `distill`.
- [`strata_forge.storage`](storage.md) — pushing the HF form of a dataset to
  the Hub.
- [`strata_forge.training`](training.md) — the HF `Dataset` the trainers
  expect on the other side of `to_hf_dataset`.
- [`strata_forge.cli`](cli.md) — `strata-forge datasets` for listing,
  inspecting, and diffing from a shell.
- [ADR 0009](../architecture/adr/0009-datasets-langfuse-canonical-hf-exchange.md)
  — why Langfuse is the canonical store and HF the exchange format.
