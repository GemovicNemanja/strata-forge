# strata_forge.metrics

Pure classification metrics and comparisons shared by training and evaluation. Functions are synchronous
and accept an explicit ordered label space. Integer labels require an explicit index map.

`metrics.agreement` provides direct-import nominal percent agreement, Cohen kappa and Gwet
AC1 for exactly two complete caller-declared human raters with aligned item IDs. Its separate
registered-reference helper selects AC1 at inclusive four-fifths reference prevalence,
otherwise kappa, and compares a caller-configurable point threshold. Rater marginals still
define the coefficients' chance terms. Empty/undefined results and absent confidence intervals
are explicit; supplied human identity/registration/provenance is not independently attested.
No annotation resource, rater pooling, model opinion, dataset reading or research acceptance
is supplied. See the direct-import agreement surface in the reference docs.

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

Reference: [metrics](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/metrics.md).
