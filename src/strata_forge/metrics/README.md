# strata_forge.metrics

Pure classification metrics and comparisons shared by training and evaluation. Functions are synchronous
and accept an explicit ordered label space. Integer labels require an explicit index map.

`metrics.agreement` provides direct-import nominal percent agreement, Cohen kappa and Gwet
AC1 for exactly two complete caller-declared human raters with aligned item IDs. Its separate
registered-reference helper selects AC1 at inclusive four-fifths reference prevalence,
otherwise kappa, and compares a caller-configurable point threshold. Rater marginals still
define the coefficients' chance terms. Empty/undefined results and absent confidence intervals
are explicit; supplied human identity/registration/provenance is not independently attested.
`agreement_bootstrap` adds a seeded item-level percentile interval for the coefficient the
full-sample reference selects, counting resamples where it is undefined; it needs NumPy from
`[scoring]`.
`human_model_agreement` is a separate statistic for one human against one model rater: it
excludes and counts human skips and null model labels, fails a slice with too many skips,
refuses one with too many nulls or a missing reference, gates the reference-selected point value
and reports the bootstrap interval and a non-gating user-versus-solver kappa beside it.
No annotation resource, rater pooling, model opinion, dataset reading or research acceptance
is supplied. See the direct-import agreement surface in the reference docs.

`metrics.latency` summarizes latency measurements: `percentile` (type 7, equal to
`numpy.percentile` with no NumPy), `latency_summary` (per-pass p50/p95/p99, their median across
passes with min and max, and the pooled percentiles) and the opt-in `latency_bootstrap` interval
for the pooled percentiles, which needs NumPy from `[scoring]`.

`classification_report` returns accuracy, macro-F1, invalid-output rate, class scores, and a
confusion matrix. Invalid outputs remain wrong predictions in the denominator and in gold
support. Every declared class contributes to macro-F1, including absent classes with F1 zero.

`brier` uses the multiclass sum of squared class errors. `nll` records its clipping epsilon
and clipped-row count. Both require probability provenance, expose missing-row coverage, and
refuse censored top-k values or malformed vectors. They never infer normalization.

`temperature_scale` applies a supplied temperature using NumPy from `[scoring]`, imported only
when needed. `TemperatureCalibration` is a draft provenance component for application and
JSON round trips; it neither fits a temperature nor implements a complete training artifact.

`wilson_interval` reports binomial proportion bounds for explicit counts. `mcnemar_exact`
compares paired correctness, records the candidate-minus-baseline effect over all pairs,
and preserves a finite log probability when the floating p-value underflows. `holm_correction`
adjusts an explicitly named hypothesis family, retaining caller order without choosing alpha.
These comparisons need no numerical extra; resampling and research thresholds are not inferred.

`cluster_bootstrap` and `paired_cluster_bootstrap` (`metrics.cluster_bootstrap`, NumPy from
`[scoring]`) give seeded percentile intervals for accuracy, macro-F1 and invalid rate by
resampling whole caller-declared clusters, and a candidate-minus-baseline delta whose two systems
share every resample. `cluster_mean_bootstrap` and `paired_cluster_mean_bootstrap` do the same
for the mean of a per-row score such as a Brier term.

`metrics.calibration_fit` fits one softmax temperature by minimising NLL (`fit_temperature`),
cross-fits it by cluster so every row is scored with a temperature its own cluster did not
influence (`cross_fit_temperature`), and computes top-label expected calibration error over
equal-mass or equal-width bins (`expected_calibration_error`). NumPy from `[scoring]`.

`metrics.judge_agreement` reads an LLM judge's scores against gold scores of the same outputs:
the contingency table, exact agreement, Cohen's kappa, and for an integer scale the mean absolute
error and the linearly and quadratically weighted kappas, each null with a reason when undefined.

Reference: [metrics](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/metrics.md).
