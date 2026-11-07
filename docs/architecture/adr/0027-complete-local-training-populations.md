# Complete local training populations

## Context

The primary classifier SDK and its inert worker intentionally bound materialized rows and
serialized bytes. Raising those constants would not prove complete split coverage, bounded
row materialization, an exact epoch, or a total overlength count before optimizer construction.
The auxiliary verifier SDK resolves individual batches but does not own population traversal.

## Decision

Use a separate local schema-2 population profile. Trusted Python sources provide bounded
NDJSON chunks under a declared immutable named-source identity. Preparation scans the entire
source, including outside an explicit pilot range, before a usable cache exists. SQLite indexes
check full-source stable-ID uniqueness; only selected, valid, unpadded inputs and explicit
supervision enter the training cache. Overflow anywhere aborts with a private complete count
and first encountered ID. Source, canonical population, tokenizer, plan and closed-cache hashes
bind the in-process issued handle.

Public preparation and consumption are async. Fixed canonicalization, validation and receipt
inspection are synchronous pure leaves. SQLite, tokenizer, native optimizer and filesystem work
run in drained background threads; these internal native leaves do not provide sync public I/O.
Cancellation prevents a usable result and drains owned work before returning. Native calls
cannot be forcibly stopped by a Python thread cancellation.

The first consumer is single-process FP32, one epoch, batch size 32 with its final partial batch,
no accumulation or secondary sampler. A recorded CPU Torch permutation determines exactly one
visit to each selected ordinal. The effective supported native AdamW fused defaults and the
explicit learning-rate schedule are recorded. The ordinary primary objective and the explicit
fixed verifier objective remain separate codecs. Unsupported training features refuse.

## Consequences

The profile admits at most 600,000 source rows, 8 GiB serialized source and 64 GiB private scratch;
these are ceilings, never preallocations or memory/VRAM guarantees. Callers declare actual
budgets. The SQLite page cache is bounded; fixed indexed statements need no temporary sorting
B-tree, and no public SQL facility is exposed. Database, rollback journal, private receipts and
claim files share the disk budget. The fixed database page cap leaves space for rollback and
bounded metadata. A native tokenizer may allocate beyond the application spool; process-level
resource isolation remains a worker concern.

Required model and tokenizer refs are separately declared immutable metadata. The fixed
`ResolvedPopulationTokenizer` wrapper must match the explicit tokenizer pin before the native
object is used. Its actual canonical backend/special-token SHA binds content independently.
The declared pin does not prove origin or custody, and the SHA does not prove the pin.
Source identities and hashes are metadata, not authority. The injected Python policy seam is
not a credential issuer, custody attestation or remote adapter. No v1 cap, digest, default,
worker, loader, registry, CLI, run wire or publication path changes. Model aliases remain
caller-owned and must be serialized; no concurrent train/score or mode-restoration guarantee
is made. Cache handles cannot be adopted after a restart, and interrupted/unknown attempts are
not reused or cleaned up automatically.
