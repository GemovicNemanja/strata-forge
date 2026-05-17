# Agent rules — forge.datasets

`forge.datasets` is the typed dataset layer Phase 2.4's eval runner
consumes. See
[ADR 0009](../../../docs/architecture/adr/0009-datasets-langfuse-canonical-hf-exchange.md)
for the canonical-store + exchange-format design: Langfuse is the
default persistent store, Hugging Face Datasets is the exchange
format, and Forge-owned Pydantic models are the canonical in-memory
shape every other piece converts into and out of.

## Purpose

- Typed dataset shapes (`DatasetItem`, `Dataset`) with content-hash
  IDs and frozen-tuple item collections.
- Content-hash versioning + dataset diffing
  (`dataset_version`, `diff`, `DatasetDelta`).
- A pluggable async `DatasetStore` interface with an
  `InMemoryDatasetStore` baseline and (Phase 2.3.2) a Langfuse-backed
  store.
- A bidirectional Hugging Face Datasets bridge (Phase 2.3.2) behind
  the `[hf]` extra.
- Synthetic-data primitives (Phase 2.3.3): self-instruct + teacher-
  student distillation built on `forge.llm`.

## Boundaries

- **Owns:** `schema.py`, `store.py`, `versioning.py`, `stores/`,
  `hf_bridge.py` (2.3.2), `synthetic/` (2.3.3).
- **Imports from inside `forge`:** `forge.core` (errors, content_hash)
  and `forge.llm` (for the synthetic primitives). Does NOT import
  `forge.tracing`, `forge.prompts`, or any higher-level module.
- **External deps:** Pydantic at module load. `langfuse` (lazy, behind
  `[langfuse]`) and `datasets` from HF (lazy, behind `[hf]`) load only
  inside the functions that use them.

## Public API

The module's `__init__.py` re-exports the supported surface:

- Data shapes: `DatasetItem`, `Dataset`.
- Versioning: `dataset_version`, `diff`, `DatasetDelta`.
- Store interface + error: `DatasetStore`, `DatasetNotFoundError`.
- Backends: `InMemoryDatasetStore` (Phase 2.3.1); `LangfuseDatasetStore`
  (Phase 2.3.2).
- Bridge: `to_hf_dataset` / `from_hf_dataset` (Phase 2.3.2).
- Synthetic: `self_instruct`, `distill` (Phase 2.3.3).

Anything raised from this module is a `ForgeError` subclass.

## Internal patterns

- **Pydantic v2 frozen models** with `extra="forbid"` so the wire
  shape is stable. Item collections are `tuple[DatasetItem, ...]`
  not `list` — the frozen-on-the-model flag only protects against
  attribute reassignment; a tuple field protects against `.append`
  mutation through the attribute.
- **Content-hash IDs and versions** via `forge.core.repro.content_hash`.
  `DatasetItem.from_input(...)` derives an ID from
  `input + expected_output` (but not metadata, so re-annotating doesn't
  invalidate). `dataset_version()` hashes `name + sorted(item.id) +
  metadata` — description is excluded as free-form prose.
- **Backends own their version scheme.** The in-memory store uses
  content-hash versions and dedupes on identical content (re-putting
  the same dataset returns the same version). Langfuse uses whatever
  Langfuse assigns. The abstract interface treats versions as opaque
  strings.
- **`delete` semantics**: unknown name → no-op; known name + unknown
  version → raise `DatasetNotFoundError`. Mirrors the prompts module's
  rule so typos in version strings surface.
- **Lazy extras imports** for `langfuse` and `datasets` (HF). Importing
  `forge.datasets` works without either extra installed.

## Test expectations

- Unit tests under `tests/unit/datasets/`, one file per source module.
- Coverage target: ≥ 90 % line.
- Mocked Langfuse client + mocked HF `datasets` module for unit tests;
  no live network.
- Hypothesis property tests on `dataset_version` (sort-independence,
  metadata sensitivity, item-metadata invariance) and `diff` (added/
  removed/unchanged is a strict partition).

## Gotchas

- **Pydantic `frozen=True` + list field is a footgun.** A frozen model
  still lets you mutate a list-typed attribute's *contents*. Use
  `tuple[...]` for collections of items so structural immutability
  matches the model's frozen flag.
- **Item metadata shouldn't change the dataset version.** Re-tagging
  items between dataset versions is a common workflow; it should
  produce the same `dataset_version()` so cached eval results aren't
  invalidated. The `dataset_version()` implementation explicitly
  excludes per-item metadata from the hash input.
- **The `input` field shadows the Python builtin.** It's named `input`
  to match Langfuse's wire-format field name; ruff's A002/A003 are
  silenced with per-line `noqa` markers where needed.

## When to update this file

- Adding a new store backend or bridge format.
- Changing the version-hash inputs (would invalidate every existing
  version — needs an ADR).
- Changing what's content-addressed in `DatasetItem.from_input`.
- Adding a new synthetic-data primitive.
