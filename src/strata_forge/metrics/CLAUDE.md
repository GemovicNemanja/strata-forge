# Agent rules — strata_forge.metrics

This is a pure deterministic numerical leaf. Read
[`docs/modules/metrics.md`](../../../docs/modules/metrics.md) and
[ADR 0020](../../../docs/architecture/adr/0020-pure-metric-functions.md).

- Import only `core` and this module from inside Forge. Never import training, evaluation,
  datasets, pipelines, clients, storage, or configuration.
- Functions are synchronous, have no I/O or model state, and return frozen models or immutable
  tuples. Runners own off-loop scheduling and artifact persistence.
- NumPy lives behind `[scoring]` and is imported only inside numerical operations. Point
  metrics and module import work without it.
- Require explicit class order and integer mappings. Unknown is a caller-supplied logical
  label; an invalid output is never silently converted to that label or dropped.
- Preserve invalid counts in gold support and the original denominator. Record probability
  exclusions, coverage, raw/restricted provenance, tolerance, and clipping policy.
- No censored top-k probabilities as exact metrics; no silent vector renormalization.
- Keep point definitions and serialized fields consistent with the synthetic conformance
  fixture in `tests/unit/metrics/fixtures/classification_point_v1.json`.
- `TemperatureCalibration` is a draft component, not the full training calibration-file
  schema. Do not introduce fitting, thresholds, other interval definitions, or pooling implicitly.
- `comparison.py` owns Wilson binomial intervals, exact conditional McNemar, and explicit
  named-family Holm adjustment. Keep all-pair effect denominators distinct from discordant
  test denominators, preserve finite log-p/underflow provenance, and never hide an alpha or
  discard invalid predictions. Holm operates on represented p-values and cannot recover
  underflowed inputs. The synthetic comparison fixture is the numerical conformance reference.
- `cluster_bootstrap` owns only ADR0031's fixed PCG64/10000/seed 20261104 row-pooled paired
  profile. Preserve canonical ordering, variable row denominators, shared point formulas and
  exact percentile ranks. No independence, p-value, BCa or gate is inferred.
- Update this file, module README, reference docs, and tests with any public API change.
- `agreement` supports only two complete caller-declared human raters with distinct identities,
  identical unique ordered item IDs and an explicit complete nominal category set. Refuse missing
  labels/alignment; never silently filter, pool model raters or infer adjudication.
- Keep reference-only coefficient selection separate from rater-marginal chance correction.
  Gold takes precedence over supplied adjudication; absent both refuses. Registered item-set,
  reference version and per-item source refs are caller provenance, not trusted attestation.
- Use integer `5 * max_reference_count >= 4 * n` for the fixed selector and exact rational
  coefficient comparison against the supplied threshold's decimal spelling. The named 0.70
  helper default is not a global annotation/research policy or release gate.
- Preserve undefined coefficient reasons and `confidence_method="not_produced"`. No implicit
  multi-rater, missingness, CI/bootstrap or annotation-resource implementation. Keep the synthetic
  `agreement_point_v1.json` conformance fixture synchronized; no NumPy or ML dependency is needed.

- `temperature_fit` owns explicit bounded raw-logit scalar-temperature math only. No source
  registration, fit eligibility or model application is inferred. Zero derivative may reflect
  rounding/underflow; identification is never claimed, and flat_unidentified requires actual
  equal logits. Preserve unclipped objective versus existing clipped point NLL, bounds and
  deterministic candidate/tie rules. Async registered-source orchestration stays in pipelines.

- `classification_report_from_confusion` requires the exact matrix class and native immutable
  field shapes, then reconstructs all original fields through matrix validation before shared
  report math. Never trust model_construct, manual mutation, subclasses or coercion hooks.
  Preserve integer multiplicities, invalid false negatives, absent/empty policies and existing
  row behavior. Validated counts are not population/independence or bootstrap attestation.

- Cluster math consumes complete supplied observations; policy/registry belongs in pipelines.
  Reconstruct exact immutable values, retain invalid-as-wrong/absent-class definitions, bound
  one-draw/distribution arrays and never allocate B*M replicas or silently omit a replicate.
  Synchronous math cannot enforce current policy or hard cancellation; async ownership is separate.
