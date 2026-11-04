# `strata_forge.metrics` — explicit classification metrics and comparisons

This pure leaf is shared by training and evaluation. It performs no I/O, reads no datasets,
calls no models, and imports neither consumer. Functions are synchronous under
[ADR 0020](../architecture/adr/0020-pure-metric-functions.md). Execution workers own scheduling
and persistence. NumPy is required only by nonempty temperature application, through `[scoring]`.

## Public surface

```python
from strata_forge.metrics import classification_report, brier

labels = ("entailment", "contradiction", "unknown")
report = classification_report(
    ["entailment", "unknown"], ["entailment", None], labels=labels,
)
assert report.accuracy.value == 0.5
assert report.confusion.invalid_by_gold == (0, 0, 1)

score = brier(
    ["entailment", "unknown"], [[0.8, 0.1, 0.1], None], labels=labels,
    probability_kind="class_probabilities",
)
assert score.n == score.excluded == 1
assert score.coverage == 0.5
```

Other exports are `confusion_matrix`, `accuracy`, `macro_f1`, `invalid_rate`, `nll`,
`temperature_scale`, and the frozen types listed below. `strata_forge.evals.classification`
re-exports the same implemented functions and types. Existing outcome aggregates in
`evals.metrics` remain separate from per-row classification.

## Labels and invalid outputs

Every call requires an ordered collection of at least two unique nonempty class names. Class
names are exact strings; case and whitespace are not normalized. Integer gold or predicted
labels require an explicit `index_map={source_integer: declared_label}`. Boolean values are
not integers at this boundary. Many integers may map to one label for a caller-declared
collapse; the function never chooses a collapse or interprets class indexes by position.

Invalid gold labels or unequal population lengths raise Forge `ValidationError`. A missing,
unknown, or otherwise invalid prediction counts as wrong. It is retained in `invalid_by_gold`
and in gold support, rather than silently dropped or mapped to a logical unknown label.
An integer prediction without an index map is a configuration error, not a guessed label.

The confusion matrix has gold rows and prediction columns. `sum(counts) +
sum(invalid_by_gold) == n`. Per-class support includes the invalid count for that gold class;
predicted counts include only valid outputs. Precision is TP / predicted; recall is TP /
support; F1 is 2 TP / (support + predicted). A zero class denominator gives zero. Macro-F1
averages every declared class, so absent classes contribute zero and expose support zero.
Accuracy and invalid rate use all eligible rows, including invalid outputs.

## Probability metrics

Each nonmissing probability row must match the supplied label order, contain finite real
values in [0, 1], and sum to one within `normalization_tolerance` (default `1e-6`). Vectors are
never silently normalized, including vectors inside the allowed tolerance. `None` is the only
missing-vector marker; malformed vectors raise an error. NumPy arrays/scalars are accepted
without importing NumPy into the point-metric implementation.

- `brier`: mean over used rows of **the sum** of squared class errors, not their class mean
  and not the single-component binary form. Its range for normalized vectors is [0, 2].
- `nll`: mean `-log(max(p_gold, clip_epsilon))`; natural logarithms, default epsilon `1e-15`.
  The result records `clip_epsilon` and the number of rows with `p_gold < clip_epsilon`.
- `probability_kind` is mandatory. `class_probabilities` and `restricted_softmax` are supported
  and preserved. `censored_topk` is refused even if the supplied numbers sum to one.
- `calibration_ref` optionally identifies the exact caller-resolved calibration artifact.
  This pure module does not dereference or verify artifact identities.

Raw and restricted values remain distinct populations. The caller preserves label order,
model/scoring identity, config, split, raw/calibrated identity, and slice context alongside the
returned values. A two-class collapse and a three-class report are separate calls with
separate contexts; this API does not pool them, infer headline slices, select examples, or
claim independence between rows that share a `base_problem_id`.

## Denominators and serialization

`MetricValue` is frozen, strict, and rejects extra fields. It records `value`, `n`, `excluded`,
`coverage`, `reason`, and applicable probability/calibration/clipping metadata. `n` is rows
actually used in that metric's denominator, `excluded` is rows omitted from its original
eligible population, and `coverage = n / (n + excluded)` (null for an empty population).

Invalid predictions are used and wrong in accuracy/F1, so they do not reduce coverage. Missing
probability vectors are excluded only from probability metrics, with the exclusion count
visible. An empty population has `value=null`, `n=0`, `excluded=0`, `coverage=null`, and
`reason="empty"`. An eligible population with no vectors has `value=null`, `n=0`, its original
population in `excluded`, `coverage=0`, and `reason="no_probabilities"`. No NaN is serialized.

`ClassificationReport` includes `metric_version="classification-point-v1"`, the confusion
matrix, all declared `ClassScore` entries, accuracy, macro-F1, and invalid rate. `LabelSpace`
records class order. Frozen models support `model_dump_json()` / `model_validate_json()` round
trips. Pydantic model-validation errors describe inconsistent supplied artifacts; function-level
input errors use Forge `ValidationError`.

The synthetic [conformance fixture](../../tests/unit/metrics/fixtures/classification_point_v1.json)
freezes three-way and two-way reference populations and expected values. It is not a research
dataset or a benchmark result.

## Temperature application and draft component

`temperature_scale(logits, temperature=T)` accepts a rectangular finite matrix with at least
two class columns and finite T > 0, then returns immutable probability tuples. Centered
softmax avoids overflow even for extreme finite logits. Empty input returns an empty tuple
without importing NumPy. A missing NumPy installation raises an actionable `[scoring]` hint.

`TemperatureCalibration` is a **draft component**, not the complete training calibration-file
schema. It records schema version, method, T, ordered labels, fit reference/config/split/count,
probability kind, NLL clipping epsilon, and optional before/after Brier/NLL measurements.
Omitted `CalibrationScores` measurements are null. `.apply(logits, labels=...)` requires the
exact recorded column order and class count. The fit reference is supplied provenance; the
caller resolves an immutable source and enforces fit/evaluation split separation.

This surface applies a supplied T. It does not fit T, choose thresholds, write artifacts,
compute ECE, bootstrap intervals, or impose research acceptance gates.

## Explicit-population comparisons

`wilson_interval`, `mcnemar_exact`, and `holm_correction` are synchronous pure functions
implemented in `metrics.comparison` and re-exported by `evals.classification`. They require
no numerical extra, choose no acceptance threshold, and perform no dataset selection.
Frozen results carry `comparison_version="classification-comparison-v1"` and support strict
JSON round trips. Counts must be nonnegative integers, excluding booleans, at most `2**53-1`.
Probabilities must be finite real values; malformed inputs raise Forge `ValidationError`.
Inconsistent supplied result models raise Pydantic validation errors.

### Wilson proportion interval

```python
from strata_forge.metrics import wilson_interval

interval = wilson_interval(80, 100, confidence_level=0.95)
```

The input is a binomial success count and its complete eligible population, not an F1 score
or two unrelated accuracy aggregates. The two-sided Wilson score interval uses
`z = -NormalDist().inv_cdf((1-confidence_level)/2)`; the lower-tail form stays finite even for
the representable confidence immediately below one. It reports estimate, bounds, confidence,
successes, and `n`. Confidence must be strictly between zero and one. Zero successes has
lower bound zero; all successes has upper bound one. Empty `n=0` returns null estimate/bounds
and `reason="empty"`. The method follows the [NIST Wilson score formulation](https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm).

Invalid predictions remain failures in the eligible accuracy population; they are not
exclusions from its interval. The caller declares the sampling unit and independence
assumptions. Multiple reasoning samples or rows sharing `base_problem_id` do not become
independent observations merely because the helper accepts their counts.

### Exact conditional McNemar test

```python
from strata_forge.metrics import mcnemar_exact

paired = mcnemar_exact(
    baseline_only_correct=9, candidate_only_correct=1, n_pairs=100,
)
assert paired.accuracy_delta == -0.08
```

`n_pairs` is mandatory and includes concordant pairs. For baseline-only count `b`,
candidate-only count `c`, discordant `d=b+c`, and `k=min(b,c)`, the p-value is
`min(1, 2*P[Binomial(d,0.5) <= k])`. This is the conditional two-sided exact binomial
test, not a chi-square approximation, continuity correction, or mid-p test. The reported
effect is `(c-b)/n_pairs`, candidate minus baseline over **all** eligible pairs. The binomial
interpretation agrees with [statsmodels' exact McNemar definition](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).

The caller must align both models on the same eligible units and gold outcomes before
forming counts. This API does not infer pairing, silently exclude invalid predictions, pool
two-/three-class configurations, or account for repeated/clustered observations.
`d` must not exceed `n_pairs` or `MAX_EXACT_DISCORDANT` (10,000,000). Empty pairs return null
test/effect values with `reason="empty"`. Nonempty populations with no discordance return
p=1, log-p=0, effect=0. Swapping models preserves p and reverses the effect.

The full two-sided natural logarithm is retained before exponentiation. A nonzero subnormal
p remains nonzero: `(b,c)=(1075,0)` produces the smallest positive float. If exponentiation
rounds to zero, `underflowed=true` and finite `log_p_value` preserve the distinction from a
mathematically zero probability. This is an exact statistical test evaluated in floating
arithmetic, not an arbitrary-precision p-value representation.

Small discordant populations use integer binomial sums, complement/log1p near one, and
binary scaling in the tails. Large populations use the central-binomial specialization of
[Stirling's log-gamma expansion](https://dlmf.nist.gov/5.11.E1), stable log1p ratios near the mode,
and a scaled decreasing tail whose omitted remainder is geometrically bounded. No large
lgamma values are subtracted. The central expansion is used only for mode at least 2048;
its first omitted term is below `3e-33`, separate from floating rounding error. Work is
bounded by the discordant limit and can be linear in discordance; callers keep large jobs
off the event loop.

### Named-family Holm adjustment

```python
from strata_forge.metrics import holm_correction

family = holm_correction(
    {"candidate-a": 0.01, "candidate-b": 0.04, "candidate-c": 0.03},
    family_id="validation/classification:v1",
)
```

The entire family is supplied explicitly, with at most 10,000 hypotheses. Family and
hypothesis IDs are unique bounded ASCII identifiers (1–128 characters; letters, digits,
period, underscore, colon, slash, hyphen). The function preserves mapping iteration order
in its returned hypotheses. For ascending raw p-values, adjusted values are the running
maximum of `(family_size-rank+1)*p` for one-based rank, capped at one; tied inputs have equal adjustments.
It performs the [Holm step-down adjustment](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html),
without selecting alpha, adding unreported hypotheses, or returning rejection decisions.
An empty explicit family has `n=0`, empty hypotheses, and `reason="empty"`.

Holm consumes **represented floating p-values**. It cannot recover a probability that already
underflowed to zero, rank different true values both represented as zero, or reconstruct a
log value not supplied to this API. Preserve upstream McNemar log/underflow metadata beside
the corrected family. This API accepts no log-p inputs and makes no exact-arithmetic claim
about lost underflow information.

The synthetic [comparison fixture](../../tests/unit/metrics/fixtures/classification_comparison_v1.json)
records independently calculated Wilson values, integer-sum/Fraction McNemar values with
high-precision log references, and Holm examples. Tests also use exact integer complements
at large near-central counts and exhaustive small paired tables.

### Resampling boundary

No bootstrap execution is exported. A resampling contract must specify its eligible unit
IDs, exact paired population, declared cluster/base-problem mapping, preserved config/slice
context, and behavior for absent classes or undefined metrics. Reproducibility additionally
requires an explicit RNG algorithm and seed, canonical unit/cluster ordering, replicate
count, quantile/interpolation definition, and failure/coverage policy. Those choices are not
silently made by these three helpers. They supply neither bootstrap intervals nor research
thresholds, calibration fits, or an evaluation runner.

Module rules: [`src/strata_forge/metrics/CLAUDE.md`](../../src/strata_forge/metrics/CLAUDE.md).
