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

## Local SQL value contracts

Closed query, pinned input-manifest and trusted operator/file-description values are available
without DuckDB or Docker. They do not execute SQL or grant source access. The standalone shared
codec preserves supported values and refuses lossy/unknown types. See
[the reference](../../../docs/modules/headless-sql.md) for the implemented boundary.


## Provided native dataset execution

`LocalDatasetExecutor` uses a separately configured fixed image for explicit CSV/JSONL or
Parquet conversion, pinned named-table SQL and immutable Parquet materialization. It supplies
no platform grants or durable jobs. See [the contract](../../../docs/modules/native-dataset-execution.md)
for byte/result limits, scalar grammar and owned cleanup.
