# ADR 0009 — Langfuse is canonical for Forge datasets; HF Datasets is the exchange format

**Status:** Accepted
**Date:** Initial scaffolding for `strata_forge.datasets`
**Supersedes:** —
**Superseded by:** —

## Context

`strata_forge.datasets` ships data shapes (typed dataset items + datasets) plus
two backends and a bridge to a third format. Three questions had to be
answered before any code lands:

1. **Where do datasets live by default?** Langfuse, Hugging Face Hub,
   the local filesystem, an in-process store, all of the above?
2. **What's the exchange format?** When a user wants to share a dataset
   externally — or pull in a public one — what does it round-trip
   through?
3. **What's the canonical "Forge dataset" object?** Pydantic models we
   own, an HF `Dataset`, a Langfuse `DatasetItem` list, or something
   else?

Phase 2.4 (the eval runner) will consume the answers. Picking wrong
here means the eval module reaches around `strata_forge.datasets` for the
shape it needs, which defeats the point.

## Decision

**Canonical persistent store: Langfuse.** Forge datasets are intended
to live in Langfuse Datasets by default. The reason is workflow
locality: traces, annotations, and grader output already live in
Langfuse; promoting an annotated trace into an eval dataset is then a
one-system operation rather than an export-import dance.

**Exchange format: Hugging Face Datasets.** The HF Datasets library is
the de-facto exchange format for ML data: it handles
parquet/arrow/JSON serialization, Hub upload/download, streaming, and
the long tail of public datasets you might want to pull in. Forge
provides a bidirectional bridge (`strata_forge.datasets.hf_bridge`) so any HF
`Dataset` flows in or out of a Forge dataset cleanly.

**Canonical in-memory shape: Forge-owned Pydantic models.**
`DatasetItem` and `Dataset` are Pydantic v2 models with frozen
config, content-hash IDs, and explicit `input` / `expected_output` /
`metadata` fields. They're the canonical type the eval module
consumes; the Langfuse store and HF bridge are converters into and
out of them.

**Versioning by content hash.** A `Dataset`'s version is the SHA-256
of its canonical content (name + sorted item IDs + metadata). Two
puts with identical content produce the same version — natural
deduplication. Item IDs are likewise content-derived by default so
two items with the same input + expected output are the same item
regardless of who constructed them.

```
                     strata_forge.evals  (Phase 2.4)
                          │
                          ▼ consumes
              DatasetItem / Dataset  (strata_forge.datasets.schema)
                  ▲              ▲                  ▲
                  │              │                  │
                  │              │                  │
           InMemory-      LangfuseDataset-   hf_bridge: ↔ HF Datasets
           DatasetStore   Store              (read: import; write: export)
                          (canonical)
```

## Consequences

**Positive**

- **Single source of truth.** Datasets that drive evals live in
  Langfuse alongside the traces they grade. No "is this the latest
  version of the dataset?" hunt across S3 + filesystem + repo.
- **Standard exchange.** Anyone with an HF `Dataset` can ingest it
  into Forge in one call. Anyone with a Forge dataset can publish it
  to the Hub in one call.
- **Typed locally.** The eval runner consumes
  `Dataset = Pydantic model with frozen items: tuple[DatasetItem, ...]`.
  No raw dicts; no maybe-this-is-a-list-of-dicts-or-maybe-an-HF-object
  ambiguity.
- **Content-hash IDs and versions** make dedup, cache-key composition,
  and diff computation natural — and let Phase 2.4's CI eval-gate
  detect "the dataset changed between runs" without separate
  bookkeeping.
- **Backends are pluggable.** The same shape moves through the in-memory
  store (tests), Langfuse (production), or the HF bridge (training and
  external consumption). Adding a new backend (a sqlite store, an S3
  parquet store, …) is one file.

**Negative**

- **Langfuse becomes a hard dependency for "production datasets".**
  Users who want versioned dataset storage but don't want to run
  Langfuse have to either install + run the local Langfuse stack or
  use the HF bridge as their primary store. We're betting Langfuse is
  already in the loop for anyone running Forge evals.
- **Two extras to manage.** `[langfuse]` (already used by tracing) and
  a new `[hf]` extra. Both lazy-imported, but two extras is two more
  things for new users to discover.

**Mitigations**

- The in-memory store is a first-class implementation, not just a test
  fixture. Users prototyping locally can stand up evals against an
  `InMemoryDatasetStore` and migrate later. `strata-forge doctor` will
  surface "no production store configured" so it's visible.
- The HF bridge is bidirectional. A user who refuses Langfuse can run
  the whole eval workflow on HF Datasets — the dataset just doesn't
  live alongside the traces.
- The Pydantic schema is *the* canonical shape, not Langfuse's schema
  or HF's schema. Replacing Langfuse with another canonical store
  (e.g. a future Forge-native one) is one new `DatasetStore` subclass
  away.

## Alternatives considered

1. **Files-as-canonical (parquet on S3 or HF Hub).** Loses the
   workflow locality with traces and annotations. The eval module
   would need to reach into Langfuse anyway to pull grader results;
   keeping the dataset elsewhere fragments the source of truth.
   Rejected.

2. **In-memory only.** Fine for the eval module's hot path but doesn't
   solve "where do my datasets live between runs?" Rejected as a
   default; kept as a first-class store option for prototyping.

3. **HF Datasets as canonical, Langfuse as the exchange.** Inverts
   the relationship. HF is great for storage and exchange but doesn't
   carry the annotation/score history that the eval runner needs. Two
   sources of truth would result. Rejected.

4. **Custom storage in a Forge-owned database.** Maximizes control,
   but builds-not-buys observability and ops on something that
   Langfuse already provides for free. Rejected.
