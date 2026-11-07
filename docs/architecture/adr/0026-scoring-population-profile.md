# Explicit native no-truncation populations

## Context

Legacy classifier scoring may report a first-sequence truncation. Evaluation v1 deliberately
retains every selected missing/invalid/refused prediction as wrong. An overflow flag alone
therefore cannot silently remove a v1 row from its denominator. A research profile requires
native per-item token overflow exclusions and a separate suite eligibility filter, without
changing generic defaults or claiming source execution from an imported flag.

The content-free CC-20261106-04 request defines the distinction: scoring/evaluation drop an
overlength item; training instead aborts after complete source preflight before optimization.
The latter is a separate worker contract. Logical labels and scientific acceptance remain
owned by their declared task protocols.

## Decision

Use distinct strict population-v2 operational envelopes around the existing RunSpec and sole
EvalSpec task schema. Overflow is required and drop-only. Preserve legacy task field presence,
wire defaults/digests, result models and entrypoints. Wrapper model/dict/JSON serialization is
canonical and roundtrips without introducing unused generation fields. The full envelope,
not an inner task digest, binds any future read decision and report.

Partition all selected memberships into suite exclusions E, native computed overflow D,
valid scored outputs S and ordinary invalid/refused/missing outputs F. N=E+D+S+F and the
metric population M=S+F. Invalid outputs remain wrong; a valid independent probability vector
survives a malformed hard label. Suite-excluded rows are not rendered/tokenized/forwarded.
Only the selected native scorer's complete token count can establish verified overflow.
Tier metadata has a64-value/128-byte-value budget; these are prototype resource bounds.

Retain ordered selected identity, metric ID identity and evaluation target identity separately
from condition provenance. Comparisons require the same gold/source population, exact M
IDs/gold and compatible target metric semantics. Model/checkpoint/tokenizer/source projection/
rendering conditions may differ and remain explicit provenance. Different drop sets refuse;
no implicit common subset or silently altered Holm family is permitted.

External predictions remain usable under honest caller-selected population semantics. A raw
imported overflow field or self-described manifest cannot grant a verified exclusion. Native
verified-drop reuse from stored predictions needs separately reviewed receipt evidence or
exact recomputation; this decision introduces no issuer/credential/attestation framework.

## Consequences

The specification and pure metadata types do not grant access. Trusted native Python
worker functions require outer-digest-bound read policy and produce private local receipts.
Installed CLI/registry and legacy workers reject population envelopes; their activation is
separate from tiny CPU safe-load/tokenization/forward and Parquet/metric source acceptance. Unsupported architectures, publication, calibration/statistical acceptance,
process teardown and research admission remain separate gates. No new model or task support
can be inferred from a JSON type or a completed population audit.
