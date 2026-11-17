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
- Bootstrap code requires an approved RNG/seed/order/quantile/population contract. The only
  approved one is `agreement_bootstrap` (`item-percentile-bootstrap-v1`, approved for the
  ANNOT-001 report on 2026-11-17): item-level nonparametric percentile bootstrap,
  `Generator(PCG64(seed))` with one C-order `(resamples, n)` index draw over the declared item
  order, caller-supplied count (default 10,000), coefficient fixed by the full-sample reference
  selection, undefined resamples counted and reported, Hyndman-Fan type 7 quantiles at exact
  rational positions, confidence fixed at 0.95. Changing any of these is a new version and needs
  a new approval. No other bootstrap, and no inference of independence from cluster-repeated
  rows, is exported.
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
- Preserve undefined coefficient reasons and `confidence_method="not_produced"` on point reports.
  No implicit multi-rater, missingness, other CI or annotation-resource implementation. Keep the
  synthetic `agreement_point_v1.json` conformance fixture synchronized. Import and point functions
  need no NumPy or ML dependency; only `agreement_bootstrap` imports NumPy, lazily, from
  `[scoring]`. Its per-resample values must stay equal to the exact `Fraction` point arithmetic.
