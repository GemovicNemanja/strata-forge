# ADR 0021: pinned evaluation workers require trusted access decisions

## Context

Classification evaluation needs exact task populations, stable prediction joins, probability
provenance and immutable artifacts. A public Hub URL, cached checkpoint or caller's unsealed
flag proves neither authorization nor absence of protected lineage. Existing generic experiment
execution consumes resolved objects and does not provide this pinned worker contract.

## Decision

Keep the experiment API intact. Add a strict inert `EvalSpec` and one async compute-target
worker for native classification and prediction-table scoring. Each config/task has a separate
declared population, ordered labels, projection, selection, slices and groups. Reuse the shared
scorer/load guard and pure metric/comparison definitions; never add an eval-to-training runner
dependency or accept executable imports/callables in JSON.

Before model or row reads, a trusted installed adapter supplies exact-pin read decisions from
authenticated authorization and fresh custody facts. Unknown, sealed, locked, denied or
mismatched decisions refuse. Canonical/historical aliases, portable markers and lineage must
combine monotonically. There is no permissive standalone fallback or spec-controlled policy
loader. Serialized flags never grant access.

Preserve all selected gold rows; invalid/missing outputs are wrong. Missing probability vectors
have explicit exclusions/coverage; malformed vectors fail without repair. Full-class and
restricted-softmax forecasts remain distinct. Record unavailable methods/heads as not-produced,
and withhold independent inference when group assumptions do not apply.

Write explicitly typed Parquet and a versioned per-population local report/card. Reserve one
output directory per attempt, preserve bounded local failure evidence, and emit no success on
cancellation. Refuse every non-null publication destination until a separate trusted output
fence authorizes it. Read approval and private visibility alone are insufficient; publication
requires canonical identity, fresh protection/private checks and an enforced expected parent.
An actual CommitRef must come from the completed authorized mutation, never an invented SHA.

## Consequences

The worker is usable by trusted Python orchestration and synthetic fixtures while direct
standalone launch denies without its policy adapter. Production integration must install and
verify that adapter; an injected test resolver is not authorization infrastructure. Existing
run ownership, read-plan, attempt reconciliation and process-group teardown remain explicit
consumer responsibilities. The local-only result carries no output commit claim. Any
publication consumer must reconcile actual mutation/receipt outcomes across cancellation
and process/network death rather than claim distributed rollback.

Native Parquet and complete safetensors checkpoints provide a bounded supported path. Dataset
scripts, implicit external/base reads, arbitrary graders, other rendering tasks and calibrated/
clustered methods require their own explicit contracts rather than permissive fallbacks.
