# ADR 0031 — paired local cluster percentiles

Status: accepted for the bounded local SDK.

## Context

Repeated examples sharing a base problem do not satisfy an independent-item uncertainty
assumption. Separate model conditions can be compared only on the same target and exact M
metric-eligible IDs, gold and cluster memberships. Invalid hard outputs stay wrong; suite
exclusions and verified overlength drops have separate retained counts outside M.

## Decision

A pure metrics leaf supplies one explicit profile: row-pooled paired cluster percentile
intervals, NumPy Generator/PCG64 seed 20261104, 10000 replicates and one C-length int64 draw per
replicate. Both arms and all metrics share the same cluster multiplicities. Unequal cluster
sizes produce a variable row denominator. Public count-based classification reports and
existing Brier/NLL helpers own the point definitions; no alternative macro-F1, probability
repair or NLL clipping policy is introduced.

All declared classes remain in macro-F1, including absent categories. Probability comparisons
require complete compatible vectors for both arms on all M. Invalid hard labels can retain
valid independent vectors. Delta always means candidate minus baseline. Independent sorted
replicate distributions select one-based ranks 250/9750 without interpolation. This produces
no p-value, BCa interval, simultaneous coverage, independence attestation or scientific gate.
M0 has undefined points/intervals; C1 retains points but no interval or RNG draws.

The pure synchronous metrics API follows ADR0020. Immutable validation/canonical codecs in
the accompanying pipeline schema are also synchronous pure values under this narrow
exception to the async execution rule. Async orchestration owns I/O, current policy and
numerical scheduling. The profile uses a private Generator rather than changing global
random state through set_seed: this is a narrow exception to the global reproducibility
convention, needed to preserve the exact paired draw stream without affecting callers.

The local pipeline requires explicit trusted Python sources and a current policy. Typed pins,
source hashes or a serialized registration do not self-grant access. Metadata must match
before row iteration; current unsealed read and exact paired-population registration remain
valid through final return. The fixed policy schema has no issuer/default adapter. No remote
policy-purpose expansion, evaluation worker registration, artifact writer or model call follows.

Admission limits are M 100000, K 2..16, C 10000, combined source bytes64MiB, one row 16 KiB,
metadata/receipt 64 KiB and planned numerical arrays 256 MiB. B*C<=100M bounds cluster draws,
not CPU instructions. Only one draw/multiplicity and B*metrics*3 values are held; no B*M or
B*C draws array exists. Python/library allocation overhead is outside the numerical-payload
estimate. All work must fit an explicit soft elapsed budget; one native call may overrun.

Numerical chunks, source/policy awaits and cleanup are shielded/drained on repeated
cancellation. A canceled, denied, changed, incomplete or over-budget operation cannot return
a partial interval. Cancellation does not kill native threads or undo earlier caller reads.

## Consequences

Receipts retain separate source conditions/calibration declarations, N/E/D/S/F/M evidence,
observed M/counts/bytes, policy snapshots, checksums, actual library versions and explicit
unattested implementation status. N/E/D and membership outside M remain retained declared
source evidence; M/S/F alignment is checked directly. Calibration declarations are not
verified component application. Receipt hashes are descriptive consistency, not authority.

The fixed draw codec uses signed little-endian int64; replicate digests use little-endian
float64 in requested-metric baseline/candidate/delta order. Conformance includes a real
seeded10000-run, independent draw checksum, exact Fraction controls, expanded public point
checks, rank boundaries, resource refusal and drained-cancellation tests. Different NumPy
versions must pass that sequence; arbitrary future numerical compatibility is not claimed.
