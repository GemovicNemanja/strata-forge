# Bounded local classifier SDK measurements

## Context

One latency value cannot describe checkpoint loading, input preparation, native forward,
postprocessing, a local executor queue and a remote serving request at once. Existing scoring
SDKs also own length grouping and background threads; cancelling their await is not a hard
native stop. A benchmark must preserve complete row accounting and specify its denominator.

## Decision

Expose a separate local CPU/FP32/concurrency-one SDK over caller-resolved accepted objects.
A closed plan declares warmups, measured calls, batching and a soft elapsed budget. Fixed
adapters measure a whole scorer SDK call, a preencoded verifier SDK call, or verifier encoding
plus scoring. Supported native families are BERT, GPT-2 and ModernBERT decoder with their
fixed Forge wrappers. Other layouts/devices/precision/compile/quantization refuse; no loader,
worker, artifact writer, serialized callback/import/factory or new authority is introduced.

Public execution is async. Canonical data identities and bounded statistical arithmetic are
synchronous pure leaves; they perform no I/O or native forward. Native inspection and input
fingerprinting run off-loop, and existing SDK calls are shielded and drained on cancellation.
Prevent another benchmark sharing the same known native aliases until cleanup finishes.
Caller ownership remains required for arbitrary outside aliases and shared storage views.

Use integer perf-counter differences, explicitly excluded warmups, named nearest-rank
p50/p95/p99 and complete raw call counts. The whole measured loop includes output validation;
SDK-active throughput uses only summed SDK call durations. These are different rate fields.
Failures, dropped/truncated/missing rows, changed inputs, deadlines and invalid clocks cannot
produce complete headline aggregates. A single native call may exceed the budget; no hard
thread termination or model rollback is promised.

Fast-tokenizer backend JSON contains operational padding/truncation state that SDK calls
legitimately replace. The versioned benchmark tokenizer fingerprint excludes exactly those
two overridden fields, while retaining vocabulary, processing rules and explicit special IDs
and padding side. This avoids an implicit hidden warmup and still refuses semantic changes.

## Consequences

Results are private frozen in-memory values containing counts/digests and allowlisted runtime
metadata, without text, IDs, predictions or native diagnostic strings. Declared source refs
and content fingerprints do not attest origin or grant access. A consistent result can still
be fabricated outside the benchmark; future trusted consumers need their own binding.

Cold start, forward-only, serving, accelerator synchronization, concurrent throughput, remote
job/artifact integration and actual research performance claims remain separate capabilities.
Admission byte/call limits are not a guarantee that model activations fit memory. No actual
research/GPU work or latency threshold is inferred from tiny CPU/fake-clock conformance.
