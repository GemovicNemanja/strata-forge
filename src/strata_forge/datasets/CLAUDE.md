# Agent rules — strata_forge.datasets

`strata_forge.datasets` is the typed dataset layer the
`strata_forge.evals` runner consumes. See
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
  `InMemoryDatasetStore` baseline and a Langfuse-backed
  `LangfuseDatasetStore`.
- A bidirectional Hugging Face Datasets bridge behind the `[hf]`
  extra.
- Synthetic-data primitives: `self_instruct` and `distill`
  (teacher-student distillation), both built on `strata_forge.llm`.

## Boundaries

- **Owns:** `schema.py`, `store.py`, `versioning.py`,
  `stores/` (`memory.py`, `langfuse.py`), `hf_bridge.py`,
  `synthetic/` (`self_instruct.py`, `distillation.py`).
- **Imports from inside `forge`:** `strata_forge.core` (errors, content_hash)
  and `strata_forge.llm` (for the synthetic primitives). Does NOT import
  `strata_forge.tracing`, `strata_forge.prompts`, or any higher-level module.
- **External deps:** Pydantic at module load. `langfuse` (lazy, behind
  `[langfuse]`) and `datasets` from HF (lazy, behind `[hf]`) load only
  inside the functions that use them.

## Public API

The module's `__init__.py` re-exports the supported surface:

- Data shapes: `DatasetItem`, `Dataset`.
- Versioning: `dataset_version`, `diff`, `DatasetDelta`.
- Store interface + error: `DatasetStore`, `DatasetNotFoundError`.
- Backends: `InMemoryDatasetStore`, `LangfuseDatasetStore`.
- HF bridge: `to_hf_dataset`, `from_hf_dataset`.
- Synthetic-data primitives: `self_instruct`, `distill`.

Lower-level helpers (e.g. `compose_langfuse_name` /
`decompose_langfuse_name` in the Langfuse store) are exported from
their own submodule rather than the package root.

Anything raised from this module is a `ForgeError` subclass.

## Internal patterns

- **Pydantic v2 frozen models** with `extra="forbid"` so the wire
  shape is stable. Item collections are `tuple[DatasetItem, ...]`
  not `list` — the frozen-on-the-model flag only protects against
  attribute reassignment; a tuple field protects against `.append`
  mutation through the attribute.
- **Content-hash IDs and versions** via `strata_forge.core.repro.content_hash`.
  `DatasetItem.from_input(...)` derives an ID from
  `input + expected_output` (but not metadata, so re-annotating doesn't
  invalidate). `dataset_version()` hashes `name + sorted(item.id) +
  metadata` — description is excluded as free-form prose.
- **Backends own their version scheme.** The in-memory store uses
  content-hash versions and dedupes on identical content. The Langfuse
  store also uses content-hash versions, encoded into the Langfuse
  dataset name as `{forge_name}__v{version}` — one Langfuse dataset
  per `(forge_name, version)` pair. Re-putting identical content
  collides on the composed name and short-circuits as a no-op.
  The abstract interface treats versions as opaque strings.
- **`delete` semantics**: unknown name → no-op; known name + unknown
  version → raise `DatasetNotFoundError`. Mirrors the prompts module's
  rule so typos in version strings surface.
- **Lazy extras imports** for `langfuse` and `datasets` (HF). Importing
  `strata_forge.datasets` works without either extra installed.

## Test expectations

- Unit tests under `tests/unit/datasets/`, one file per source module.
- Coverage: the enforced gate is the repo-wide 85 % line floor
  (`fail_under` in `pyproject.toml`); treat a drop in this module as
  a regression.
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
