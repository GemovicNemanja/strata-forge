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
  schema. Fitting lives only in `calibration_fit.py`: `fit_temperature`
  (`temperature-nll-bisection-v1`: NLL bisected on beta = 1/T within [0.05, 20], the bound taken
  and reported when the optimum lies outside), `cross_fit_temperature` (`cluster-cross-fit-v1`:
  sorted cluster keys permuted by `Generator(PCG64(seed))`, dealt into contiguous folds, each
  fold scored with the temperature fitted on the others) and `expected_calibration_error`
  (`top-label-ece-v1`, 15 bins, equal-mass by stable confidence sort with ties by item id and
  `array_split` sizes, or equal-width), computed from `reliability_bins` so the reported bins and
  the ECE never disagree. Two folds put the first ceil(C/2) permuted clusters in
  fold 0. A fitted
  temperature is never used to select a model, threshold or reference. Do not introduce
  thresholds, other interval definitions, or pooling implicitly.
- `comparison.py` owns Wilson binomial intervals, exact conditional McNemar, and explicit
  named-family Holm adjustment. Keep all-pair effect denominators distinct from discordant
  test denominators, preserve finite log-p/underflow provenance, and never hide an alpha or
  discard invalid predictions. Holm operates on represented p-values and cannot recover
  underflowed inputs. The synthetic comparison fixture is the numerical conformance reference.
- Bootstrap code requires an approved RNG/seed/order/quantile/population contract. The only approved
  one is `agreement_bootstrap` (`item-percentile-bootstrap-v1`, approved for the ANNOT-001 report on
  2026-11-17): item-level nonparametric percentile bootstrap, `Generator(PCG64(seed))` with the
  stream of one C-order `(resamples, n)` index draw over the declared item order (drawn in
  consecutive row blocks that reproduce it bit for bit), caller-supplied count (default 10,000),
  coefficient fixed by the full-sample reference selection, undefined resamples counted and
  reported, Hyndman-Fan type 7 quantiles at exact rational positions, confidence fixed at 0.95.
  Changing any of these is a new version and needs a new approval. The second contract is
  `cluster_bootstrap` / `paired_cluster_bootstrap` (`cluster-percentile-bootstrap-v1`, from the
  research pre-registration's metric engine, relayed by the platform coordinator on 2026-11-18):
  whole caller-declared clusters resampled with replacement, clusters ordered by sorted string key,
  `Generator(PCG64(seed))` with the stream of one C-order `(resamples, C)` draw in consecutive row
  blocks, caller-supplied count (default 10,000), classification_report definitions per resample
  (a resample with a declared class absent from gold and predictions has no macro-F1: counted in
  `undefined_resamples`, never dropped silently or zeroed),
  paired systems scored on the same resample and differenced before the quantiles, Hyndman-Fan
  type 7 quantiles at exact rational positions, confidence fixed at 0.95.
  `cluster_mean_bootstrap` / `paired_cluster_mean_bootstrap` apply the same contract to the mean
  of a finite per-row score (a Brier term): cluster sums over cluster sizes per resample. The same versioning
  rule applies. The third is `latency_bootstrap` (`latency-iteration-percentile-bootstrap-v1`), for
  the latency benchmark (research ruling of 2026-11-18: opt-in, off by default, diagnostic only,
  never the headline and never a pass/fail input): it reuses the RNG, stream order, type 7
  quantiles and 0.95 confidence above and resamples measured latency iterations pooled across
  passes, and records its population in its result.
  No other bootstrap, and no inference of independence from cluster-repeated rows, is exported.
- `latency.py` point statistics (`latency-pass-percentiles-v1`) are the latency protocol's: type 7
  percentiles equal to `numpy.percentile`'s default without NumPy, per pass, then their median
  across passes with min and max. Keep the float operations equal to NumPy's `linear` method.
- Update this file, module README, reference docs, and tests with any public API change.
- The two-human statistic (`nominal-two-human-raters-v1`) supports only two complete
  caller-declared human raters with distinct identities, identical unique ordered item IDs and an
  explicit complete nominal category set. Refuse missing labels/alignment; never silently filter,
  pool model raters or infer adjudication. Do not loosen this rule.
- The separate human-vs-model statistic (`nominal-human-vs-model-v1`, `human_model_agreement`) was
  confirmed by the research owner on 2026-11-17 (r10-thresholds T6 as amended by CR-9 and
  CC-20261106-01 §3, relayed by the platform coordinator). Its fixed rules: exactly one
  `HumanAnnotations` and one `ModelRatings` (prompt SHA-256 and run ref provenance), never
  interchangeable with `HumanRatings`; the function excludes and counts human skips and null model
  labels separately, removing an item excluded for either reason from both series; skips `>` the
  skip fraction (default 0.10) of the slice items FAIL, nulls `>` the null fraction (default 0.10)
  REFUSE; the signed skip FAIL (`skip_fraction_exceeded`) outranks every refusal reason and lists
  each applicable one beside it in the ordered `reasons` tuple (empty iff pass); a value, kappa and
  CI are reported only when a coefficient is defined and the slice is not refused; status and value
  are decided before the interval, whose size never refuses a slice; the existing selector runs once
  on the reference label over ALL slice items (gold before adjudication) and a slice with any item
  lacking a reference refuses unless skips also fail it; the threshold is bounded to [-1, 1] and
  both fractions to [0, 1); kappa is reported beside a selected AC1; the point value `>=`
  threshold (default 0.70) gates and the interval is report-only, using the approved bootstrap on
  the pair after exclusions; user-vs-solver kappa on solver-gold items is report-only and never
  gates; the result carries no item IDs or labels. Changing any of these needs a new version and
  approval.
- Keep reference-only coefficient selection separate from rater-marginal chance correction.
  Gold takes precedence over supplied adjudication; absent both refuses. Registered item-set,
  reference version and per-item source refs are caller provenance, not trusted attestation.
- Use integer `5 * max_reference_count >= 4 * n` for the fixed selector and exact rational
  coefficient comparison against the supplied threshold's decimal spelling. The named 0.70
  helper default is not a global annotation/research policy or release gate.
- Preserve undefined coefficient reasons and `confidence_method="not_produced"` on point reports. No
  implicit multi-rater, missingness, other CI or annotation-resource implementation. Keep the
  synthetic `agreement_point_v1.json` conformance fixture synchronized. Import and point functions
  need no NumPy or ML dependency; only `agreement_bootstrap` and `human_model_agreement` import
  NumPy, lazily, from `[scoring]`. Its per-resample values must stay equal to the exact `Fraction`
  point arithmetic.
