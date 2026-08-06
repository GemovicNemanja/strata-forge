# strata_forge.datasets

Typed dataset shapes plus pluggable storage and synthetic-data
primitives. The canonical persistent store is Langfuse (so eval datasets
live alongside the traces they grade); Hugging Face Datasets is the
exchange format (training + external consumption). Forge-owned Pydantic
models — `DatasetItem` and `Dataset` — are the canonical in-memory shape
that every other piece converts into and out of. See
[ADR 0009](../../../docs/architecture/adr/0009-datasets-langfuse-canonical-hf-exchange.md)
for the rationale.

Phase 2.3.1 ships the data model, content-hash versioning + diffing,
and an `InMemoryDatasetStore` backend. Subsequent sub-phases add the
Langfuse store + HF bridge (2.3.2) and synthetic-data primitives
(2.3.3).

> **Status.** Implementation in progress as part of Phase 2.3. See
> [`docs/roadmap.md`](../../../docs/roadmap.md) for the per-sub-phase
> deliverable list.

Module rules: [`CLAUDE.md`](CLAUDE.md). Full reference once shipped:
`docs/modules/datasets.md`.
