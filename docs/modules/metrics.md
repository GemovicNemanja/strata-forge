# `strata_forge.metrics` — explicit classification metrics and comparisons

This pure leaf is shared by training and evaluation. It performs no I/O, reads no datasets,
calls no models, and imports neither consumer. Functions are synchronous under
[ADR 0020](../architecture/adr/0020-pure-metric-functions.md). Execution workers own scheduling
and persistence. NumPy is required only by nonempty temperature application and the agreement and
cluster bootstraps, through `[scoring]`.

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

This surface applies a supplied T. Fitting and ECE live in
[Temperature fitting, cross-fitting and ECE](#temperature-fitting-cross-fitting-and-ece); nothing
here chooses thresholds, writes artifacts or imposes research acceptance gates.

## Temperature fitting, cross-fitting and ECE

```python
from strata_forge.metrics.calibration_fit import (
    cross_fit_temperature, expected_calibration_error, fit_temperature,
)

fit = fit_temperature(logits, gold, labels=labels)            # TemperatureFit
held_out = cross_fit_temperature(logits, gold, labels=labels, clusters=base_ids, seed=20261104)
ece = expected_calibration_error(gold, held_out.probabilities, labels=labels)  # 15 equal-mass bins
```

Direct-import only; NumPy from `[scoring]`.

- **`fit_temperature`** (`temperature-nll-bisection-v1`) minimises mean NLL of
  `softmax(logits / T)`. NLL is convex in beta = 1/T and its gradient `mean(E_p[z] - z_gold)` is
  nondecreasing, so the gradient is bisected on `[1/20, 1/0.05]` to float precision (at most 200
  steps). An optimum outside `[MIN_TEMPERATURE, MAX_TEMPERATURE]` = `[0.05, 20]` takes the bound
  and sets `at_bound` (`"min"` for a population the logits already separate, `"max"` for one
  they mislead). It records `n`, `nll_before` (T = 1) and `nll_after`. Every row needs finite
  logits in label order and a declared gold label; a constant added to a row changes nothing.
- **`cross_fit_temperature`** (`cluster-cross-fit-v1`) sorts the cluster keys, permutes them with
  `Generator(PCG64(seed))` and deals them into `folds` (default 2) contiguous groups by position
  (the cluster at permuted position `j` goes to fold `j * folds // C`, so with two folds the
  first `ceil(C/2)` are fold 0), so a cluster's rows are never on both sides of a fit. Fold k's rows get `softmax(z / T_k)`
  where `T_k` was fitted on every OTHER fold. The result holds the per-fold fits, the fold row
  counts and every row's held-out probabilities in input order. Fewer clusters than folds
  refuses (`calibration_too_few_clusters`).
- **`expected_calibration_error`** (`top-label-ece-v1`): confidence is the largest probability and
  the prediction its first argmax. `equal_mass` (default) sorts by confidence, ties broken by the
  optional `item_ids` (else by input order; stable either way), and splits into `bins` (default
  15) contiguous groups with `array_split` semantics (the first `n mod bins` hold one row more);
  `equal_width` uses `[k/bins, (k+1)/bins)` with 1.0 in the last bin. ECE is the
  bin-count-weighted mean of `|accuracy - mean confidence|`; empty bins contribute nothing.
  Rows must sum to one within 1e-6.

A fitted temperature is provenance for the calibrated metrics it produced. Nothing here uses one
to select a model, a reference or a threshold.

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

Two bootstrap contracts exist: the agreement interval described under
[Item-level agreement bootstrap](#item-level-agreement-bootstrap), and the classification
interval described under [Cluster percentile bootstrap](#cluster-percentile-bootstrap). Any other
resampling contract
must specify its eligible unit IDs, exact paired population, declared cluster/base-problem
mapping, preserved config/slice context, and behavior for absent classes or undefined metrics.
Reproducibility additionally requires an explicit RNG algorithm and seed, canonical unit/cluster
ordering, replicate count, quantile/interpolation definition, and failure/coverage policy. Those
choices are not silently made by these three helpers. They supply neither bootstrap intervals
nor research thresholds or calibration fits.

### Cluster percentile bootstrap

```python
from strata_forge.metrics.cluster_bootstrap import cluster_bootstrap, paired_cluster_bootstrap

single = cluster_bootstrap(gold, predictions, labels=labels, clusters=base_ids, seed=20261104)
paired = paired_cluster_bootstrap(
    gold, baseline, candidate, labels=labels, clusters=base_ids, seed=20261104
)
assert paired.delta.macro_f1.point == (
    paired.candidate.macro_f1.point - paired.baseline.macro_f1.point
)
```

A nonparametric percentile bootstrap over caller-declared **clusters**
(`bootstrap_version="cluster-percentile-bootstrap-v1"`) for accuracy, macro-F1 and invalid rate.
It needs NumPy from `[scoring]`, imported only when a function runs.

- **Population.** `gold`, every system's predictions and `clusters` are aligned row lists of one
  length. A cluster key is a nonempty string per row (for example a base problem id); a row-level
  bootstrap is a cluster per row. Invalid and missing predictions follow `classification_report`:
  wrong, and in the gold class's support. An empty population refuses with
  `cluster_bootstrap_empty`, a length mismatch with `cluster_bootstrap_population_mismatch`.
- **Order.** Clusters are ordered by their sorted string keys, so neither row order nor the
  order in which clusters first appear changes a result.
- **Draws.** The stream is that of one C-order draw
  `numpy.random.Generator(PCG64(seed)).integers(0, C, size=(resamples, C), dtype=int64)` over the
  `C` canonical clusters, drawn in consecutive row blocks of about 2**20 indices from the same
  generator. Each drawn cluster contributes all its rows. `seed` is an integer in `[0, 2**53-1]`
  (bool refused); `resamples` is at least one and at most 1,000,000, default 10,000.
- **Arithmetic.** Each cluster's confusion cells are counted once; a block of resamples multiplies
  its cluster multiplicities by that table in float64, exact for integer counts below 2**53 in any
  summation order. Each resample's metrics use `classification_report`'s definitions in float64,
  equal to the point arithmetic within `1e-12`, with one exception: a resample whose GOLD
  has no row of a declared class has NO macro-F1, whatever the predictions. It is counted in the
  interval's `undefined_resamples`, never dropped silently and never scored as zero, and the
  quantiles are taken over the defined resamples; when none is defined the bounds are null with
  `reason="all_resamples_undefined"`. A paired delta is undefined where either side is. Accuracy
  and invalid rate are always defined (every resample has rows). The point value keeps
  `classification_report`'s convention; the evaluation runner lists a slice's absent classes.
- **Absent classes.** When the population's gold has no row of a declared class, macro-F1 has
  no point value for ANY system, nor does the paired delta: `point`, `lower` and `upper` are null
  with `reason="absent_gold_class:<labels>"` and `absent_classes` listed. Absence depends on the
  gold only, never on predictions. `macro_f1_present_classes` is the diagnostic over
  `present_classes` (the classes the gold has, one set for every system), under the same
  undefined-resample rule; it is never a gate read.
- **One-sided bounds.** A one-sided 97.5% bound is the matching end of the two-sided 95% interval.
- **Pairing.** `paired_cluster_bootstrap` scores both systems on every resample and differences
  them per resample (candidate minus baseline) before taking quantiles, so the delta interval
  reflects the pairing. Its per-system intervals equal `cluster_bootstrap`'s for the same seed.
- **Quantile rule.** Hyndman-Fan type 7 at exact rational positions `p = 1/40` and `p = 39/40`,
  as for the agreement bootstrap; the confidence level is fixed at 0.95.
- **Golden values.** A unit test pins exact bounds for a fixed seed and input.

`ClusterBootstrap` holds a `settings` block (the contract fields, `seed`, `resamples`, `n`,
`clusters`, `numpy_version`) and `intervals` (`accuracy`, `macro_f1`, `invalid_rate`, each a
`MetricInterval` of `point`, `lower`, `upper`; `point` is the full-sample value).
`PairedClusterBootstrap` holds `settings`, `baseline`, `candidate` and `delta`.
`cluster_mean_bootstrap(values, clusters=..., seed=...)` and `paired_cluster_mean_bootstrap`
apply the same contract to the mean of a finite per-row score (for example each row's Brier
term): each resample's mean is its cluster sums over its cluster sizes, so 0/1 correctness gives
exactly the accuracy interval above. The paired form also reports `delta_sd`, the sample SD
(n - 1) of the per-row differences with its interval over the same resamples (from weighted sums
of x and x squared); it needs at least two rows (`cluster_bootstrap_too_few_rows`). One call covers
one slice: a slice's interval resamples whole clusters within that slice. Both are direct-import
only. 10,000 paired resamples over 12,000 clusters take a few seconds.

Module rules: [`src/strata_forge/metrics/CLAUDE.md`](../../src/strata_forge/metrics/CLAUDE.md).

## Two human raters and registered-reference selection

The direct-import `strata_forge.metrics.agreement` module supplies synchronous nominal
point agreement. It is separate from classifier accuracy and from annotation persistence,
blinding, authorization and research acceptance. Its functions require only Pydantic, core
errors and the Python standard library. They read no files, rows, credentials or models.
No additional package-level or evaluation re-export is registered.

```python
from strata_forge.metrics.agreement import (
    HumanRatings, RegisteredReference, pairwise_agreement,
    reference_prevalence, research_annotation_gate,
)

labels = ("E", "C", "U")
a = HumanRatings(
    rater_id="human-a", rater_kind="human",
    item_ids=("item-1", "item-2"), ratings=("E", "C"),
)
b = HumanRatings(
    rater_id="human-b", rater_kind="human",
    item_ids=("item-1", "item-2"), ratings=("E", "U"),
)
agreement = pairwise_agreement(a, b, labels=labels)
references = reference_prevalence(
    (
        RegisteredReference(
            item_id="item-1", solver_gold="E", adjudicated=None,
            source_ref="declared-solver", version_ref="solver-version-1",
        ),
        RegisteredReference(
            item_id="item-2", solver_gold=None, adjudicated="C",
            source_ref="declared-adjudication", version_ref="adjudication-version-1",
        ),
    ),
    labels=labels, item_set_ref="registered-item-set-1", reference_version="references-1",
)
gate = research_annotation_gate(agreement, references, threshold=0.70)
assert agreement.percent_agreement.value == 0.5
assert gate.selected_coefficient == "cohen_kappa"
assert gate.confidence_method == "not_produced"
```

`HumanRatings` is frozen, strict and closed. The two rater IDs must be distinct and nonempty.
The required `rater_kind="human"` is a caller assertion, not evidence of human identity or
independence. This API refuses model/extra rater metadata instead of pooling it. Each rating
vector has exactly one declared nonmissing label per unique, nonempty item ID. Both item-ID
tuples must be identical in order; no joining, reordering or complete-case deletion occurs.
The explicit class tuple has at least two unique nonempty exact strings, including categories
that neither rater used. There is no integer-index inference or weighted/ordinal agreement.

`PairwiseAgreement` records ordered labels/IDs, rater IDs, `n`, agreement count and the full
contingency matrix (rater A rows, rater B columns). `percent_agreement` is a proportion in
[0, 1], not a number in [0, 100]. All three estimates are retained together:

- `p_o = sum(diagonal_counts) / n`.
- Kappa chance is `sum(a_c * b_c) / n²`, using the **separate** rater count marginals.
- AC1 chance is `sum(pbar_c * (1 - pbar_c)) / (K - 1)`, with
  `pbar_c = (a_c + b_c) / (2*n)` and the complete declared category count K.
- Each coefficient is `(p_o - p_chance) / (1 - p_chance)`.

These definitions follow Gwet's two-rater formulation and generalized nominal AC1 equations
37–38 in the [author's 2008 paper](https://agreestat.com/assets/paper-variance-high-agreement-2008.pdf).
The [author-maintained reference](https://www.stat.ethz.ch/CRAN/web/packages/irrCAC/refman/irrCAC.html#gwet.ac1.raw)
also explains why available-but-unused categories must remain explicit. The implementation
uses independently written integer/Fraction arithmetic, with no irrCAC/R code or dependency.

Counts, chance terms and coefficients are exact rational values during computation. Display
estimates serialize as finite floats. They are not clipped or repaired: negative agreement
is meaningful. Empty input has `n=0`, null values/chance terms and `reason="empty"` for all
three estimates. A kappa chance denominator of zero has `value=null`, chance 1 and
`reason="chance_agreement_one"`; no perfect/zero value is invented. AC1 can remain defined
beside undefined kappa. Report constructors/JSON validation check the matrix, population and
derived estimates for consistency. Outputs are frozen and support strict JSON round trips.

### Reference basis and point comparison

Each `RegisteredReference` must provide an item ID, optional `solver_gold`, optional
`adjudicated`, and nonempty content-free `source_ref`/`version_ref`. Both supplied labels
must belong to the declared categories. A valid gold label wins even when adjudication
differs; only a null gold falls back to adjudication. Missing both or malformed gold refuses;
the helper never obtains a reference from either compared rater. No source body, rationale,
model opinion or automatic adjudication is accepted.

`reference_prevalence` additionally requires explicit `item_set_ref` and `reference_version`.
It preserves each chosen label, its `basis`, source/version and exact ordered IDs, with one
count per complete declared category and the full item denominator. These are supplied
provenance assertions. The primitive cannot establish registration, solver correctness,
immutability, access permission or whether independent humans produced the records.

`research_annotation_gate` requires reference and agreement label/ID tuples to match exactly.
It selects AC1 iff any reference category satisfies integer `5*count >= 4*n`, otherwise
kappa: 16/20 selects AC1, while 15/20 selects kappa. This selector uses reference counts
only. It does not alter either coefficient's rater-derived chance term.

`RESEARCH_ANNOTATION_THRESHOLD=0.70` names this helper's configurable default. A supplied
threshold must be a finite built-in int or float, excluding bool. Comparison is inclusive,
using the exact rational coefficient and the serialized float threshold's decimal spelling;
no rounded displayed coefficient or hidden epsilon chooses pass/fail. No threshold changes
the selector or sets policy for another consumer. A finite threshold outside the coefficient
range is preserved as caller policy, without clamping.

The returned `AgreementGate` retains both full reports, selected coefficient/value, threshold,
`passed`, undefined reason and `confidence_method="not_produced"`. A selected undefined kappa has
`passed=null`, `reason="selected_coefficient_undefined"` and its coefficient reason. An empty slice
selects nothing and has `passed=null`, `reason="empty"`. These point comparisons do not implement a
confidence interval, multi-rater method or completed G0/annotation gate; the separate bootstrap
below supplies the only interval. Callers must enforce the frozen reference source, role separation
and applicable study protocol before interpreting them. Function-level refusals use fixed Forge
`ValidationError` codes without echoing supplied item/rater/source values; input-model validation
hides values.

The [synthetic agreement fixture](../../tests/unit/metrics/fixtures/agreement_point_v1.json)
retains hand-counted rational results, including unequal marginals, negative coefficients
and a declared unused category. It contains no research annotations or model outputs.

### Item-level agreement bootstrap

```python
from strata_forge.metrics.agreement import agreement_bootstrap

interval = agreement_bootstrap(a, b, references, labels=labels, seed=20261104)
assert interval.coefficient == "cohen_kappa"
assert interval.interval.resamples == 10_000
```

```python
def agreement_bootstrap(
    rater_a: HumanRatings,
    rater_b: HumanRatings,
    prevalence: ReferencePrevalence,
    *,
    labels: Sequence[str],
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,  # 10_000
) -> AgreementBootstrap: ...
```

This is a nonparametric percentile bootstrap over **items**, approved as a contract
(`bootstrap_version="item-percentile-bootstrap-v1"`). It requires NumPy from `[scoring]`,
imported only when the function runs; a missing installation raises an actionable hint.
Importing `metrics.agreement` and the point functions still need no NumPy.

- **Population.** The two rater declarations and the reference prevalence must cover the same
  ordered item IDs and declared labels, exactly as `research_annotation_gate` requires. The
  caller removes skipped items before building all three, so the bootstrap population and the
  full-sample selection share one population. An empty population refuses with
  `agreement_bootstrap_empty`, since no interval exists over zero items.
- **Coefficient.** The coefficient is chosen once, by the same reference-only selector as the
  gate on the **full-sample** reference prevalence (AC1 iff some category has integer
  `5*count >= 4*n`, else kappa), and every resample computes that coefficient. A resample is
  not reselected by its own resampled reference prevalence: the interval then describes the
  sampling variability of the reported point estimate, and the reported coefficient never
  changes between the estimate and its interval. References are not resampled.
- **Draws.** The stream is that of one C-order draw
  `numpy.random.Generator(PCG64(seed)).integers(0, n, size=(resamples, n), dtype=int64)`;
  row `r` is resample `r` and column `j` its `j`-th item, indexing the caller's declared item
  order. The rows are drawn in consecutive blocks of about 2**20 indices from the same generator,
  which reproduces that stream bit for bit while bounding memory. Each resample keeps every
  item's (rater A, rater B) pair together. `seed` is a required integer in `[0, 2**53-1]`; bool
  is refused. `resamples` is a caller-supplied integer, at least one, default 10,000. The same
  seed, count, population and NumPy version give an identical result; the NumPy version is
  recorded in `numpy_version` because the generator's stream belongs to NumPy.
- **Arithmetic.** Each block forms the agreement count and both raters' class marginals for
  all its resamples at once (`bincount`), never a K-by-K table per resample. Each coefficient
  reduces to one integer ratio per resample: kappa is `(n*d - S)/(n*n - S)` with
  `S = sum(a_c*b_c)`, and AC1 is `(4n(K-1)d - P)/(4n²(K-1) - P)` with
  `P = sum(pooled_c*(2n - pooled_c))`. While `4*n*n*(K-1) <= 2**53`, a single correctly
  rounded division equals the float of the exact `Fraction` the point estimate uses. Above that,
  each ratio is converted through `Fraction`, so the equality still holds. Only a population
  with `4*n*n*(K-1) >= 2**62`, beyond any slice that fits in memory, refuses with
  `agreement_bootstrap_too_large`. `resamples` is at most `MAX_BOOTSTRAP_RESAMPLES` (1,000,000).
  10,000 resamples of 120 items take about 15 ms.
- **Undefined resamples.** A kappa resample whose chance agreement is one (both raters put
  every drawn item in the same single category) has no value. It is counted in
  `undefined_resamples`, not silently dropped, and is excluded from the quantiles, which are
  taken over the `resamples - undefined_resamples` defined values. AC1 chance never reaches one
  for two or more categories. If every resample is undefined (as when the full-sample selected
  kappa is itself undefined) `lower`/`upper` are null with `reason="all_resamples_undefined"`.
  Callers decide whether a nonzero undefined count is acceptable for their report.
- **Quantile rule.** `quantile_rule="hyndman-fan-7"`, the linear interpolation that is
  NumPy's and R's default. Over the `m` defined values sorted ascending `x[0..m-1]`, the
  probability `p` has the exact rational position `h = (m-1)*p`, `j = floor(h)`, and the bound
  is `x[j] + (h-j)*(x[j+1]-x[j])`, clamped to `[x[j], x[j+1]]` against rounding (`x[j]` when
  `j = m-1`). The 95% interval uses `p = 1/40` and `p = 39/40`; the confidence level is fixed
  at 0.95 (`BOOTSTRAP_CONFIDENCE`) by the approved contract and recorded on the result. The
  bounds follow this exact rule, not `np.quantile`. They agree with
  `np.quantile(..., method="linear")` to within about `2e-15`, but are not bit-identical to it.
- **Golden values.** A unit test pins exact bounds for a fixed seed and input, so a NumPy
  release that changes the PCG64 or `integers` stream fails CI rather than drifting silently.

The frozen `AgreementBootstrap` holds both full point reports, the selected `coefficient`, its
full-sample `estimate` and an `interval` block. `BootstrapInterval` records the contract fields
(`bootstrap_version`, `rng`, `quantile_rule`, `coefficient_selection="full_sample_reference"`,
`confidence_level`), `seed`, `resamples`, `undefined_resamples`, `lower`, `upper`, `reason` and
`numpy_version`, and no item identity, so the human-vs-model result below reuses it. Validation
refuses a coefficient that disagrees with the selector, an estimate that is not the selected
coefficient's, an undefined count above the resample count, misordered or out-of-range bounds,
and a missing reason. Both support strict JSON round trips. Like the point helpers, they are
direct-import only and are not re-exported by `strata_forge.metrics` or
`evals.classification`. One call covers one slice; the caller supplies each slice's population.
The rater contract is unchanged: both rater declarations are `rater_kind="human"`.

## One human against one model rater

`human_model_agreement` is a separate statistic, `statistic_version="nominal-human-vs-model-v1"`,
beside the unchanged two-human one. The research owner confirmed its rules on 2026-11-17. It
imports NumPy from `[scoring]` only for its interval.

```python
from strata_forge.metrics.agreement import (
    HumanAnnotations, ModelRatings, human_model_agreement,
)

result = human_model_agreement(
    HumanAnnotations(rater_id="user-1", rater_kind="human", item_ids=ids, ratings=user_labels),
    ModelRatings(
        rater_id="model-1", rater_kind="model", item_ids=ids, ratings=model_labels,
        model_prompt_sha256=prompt_digest, model_run_ref="run-7",
    ),
    references,                      # one RegisteredReference per slice item, same order
    labels=labels, seed=20261104,    # resamples=10_000, threshold=0.70,
)                                    # max_human_skip_fraction=0.10, max_model_null_fraction=0.10
```

**Raters.** `HumanAnnotations` (`rater_kind="human"`) allows `None` as an explicit skip.
`ModelRatings` (`rater_kind="model"`) allows `None` as a null model label and carries
`model_prompt_sha256` (64 lowercase hex characters) and `model_run_ref` (a nonempty string or
null). The types are distinct and closed. `pairwise_agreement` and `agreement_bootstrap` refuse a
`ModelRatings`, and `human_model_agreement` refuses a `HumanRatings`, two humans or two models.
The rater IDs must differ. The item-ID tuples must be identical in order, and every non-null
label must be a declared category. An unknown model label refuses rather than counting as null.

**Exclusions.** The function, not the caller, excludes per slice. `human_skipped` counts the human's
skips and `model_null` the model's nulls, each over all `items`. An item with both counts in both.
An item excluded for either reason leaves both series, so `used` is the number of items with both
labels. The fractions are parameters (default `0.10`, each in `[0, 1)`), compared strictly and
exactly against their decimal spelling: `human_skipped > fraction * items` fails the slice and
`model_null > fraction * items` refuses it. Twelve of 120 is within either limit; thirteen is not.

**Selection.** The existing selector runs once, on the reference label of **all** slice items
(solver gold when supplied, otherwise adjudication) in exact integers: AC1 iff some category has
`5*count >= 4*items`, otherwise kappa. The coefficient is computed on the used pairs, and the
bootstrap holds it fixed across resamples. When AC1 is selected, `cohen_kappa` reports kappa on
the same pairs beside it, or `cohen_kappa_reason="chance_agreement_one"` when kappa is undefined.

**Status.** The status is `pass`, `fail` or `refused`. `reasons` is an ordered tuple, empty
exactly when the slice passes. Its order is `skip_fraction_exceeded`, `empty`,
`reference_unavailable`, `model_null_fraction_exceeded`, `no_used_items`,
`coefficient_undefined`, `below_threshold`.

- **Exceeded skips fail, always.** The signed skip rule outranks every refusal reason: the
  status is `fail`, `skip_fraction_exceeded` comes first, and every other applicable reason
  follows it. These are `reference_unavailable`, `model_null_fraction_exceeded`,
  `no_used_items`, `coefficient_undefined` and `below_threshold`; listing the null-label reason
  makes a re-pilot also re-run the model labels. An empty slice cannot exceed a skip fraction,
  so it is always refused.
- **Otherwise the first refusal alone refuses.** In order, these are: `empty` (no items),
  `reference_unavailable` (some item has neither gold nor adjudication),
  `model_null_fraction_exceeded`, `no_used_items` and `coefficient_undefined`.
- **Otherwise the threshold decides.** The slice passes when the exact value is `>=` the
  threshold (default 0.70, inclusive; a kappa of exactly 7/10 passes), and fails with
  `below_threshold` when it is lower.

A value, with kappa beside AC1 and the interval, is reported only when a coefficient is defined
and the slice is not refused. So `("skip_fraction_exceeded", "model_null_fraction_exceeded")`
reports one, while a fail with `reference_unavailable`, `no_used_items` or
`coefficient_undefined` does not. A refused slice never reports a value, kappa or interval.
The status and value are decided before the interval is drawn, and no slice size makes the
interval refuse. The interval never gates.

**Interval.** `interval` is the approved `BootstrapInterval` over the used pairs: the same seed,
resample count, single `(resamples, used)` draw in the declared item order with excluded items
removed, type-7 rule and undefined-resample count as `agreement_bootstrap`. It is identical to
`agreement_bootstrap` over those pairs whenever both select the same coefficient.

**User-versus-solver kappa.** `user_solver_kappa` (`role="report_only"`) is Cohen kappa between
the human and the solver gold, over the items that have solver gold and a human label. Model
nulls do not affect it, and adjudicated items are not included. It never changes the status.
`n=0` gives `reason="empty"`.

**Result.** `HumanModelAgreement` is frozen and closed. It records:

- the statistic version and the counts `items`, `human_skipped`, `model_null` and `used`;
- both exclusion fractions, `status`, `reasons`, `coefficient` and `threshold`;
- `value` and `cohen_kappa` as `ExactValue` (`exact`, the reduced rational such as `"7/10"`,
  and `value`, its nearest float);
- `cohen_kappa_reason`, `interval` and `user_solver_kappa`;
- the model provenance.

It holds no item IDs, labels or rater-supplied text. Validation re-derives the exclusion
decisions from the counts through the same decision rule, refuses inconsistent counts,
status/reasons pairs, values, kappa presence and threshold decisions. It also refuses:

- a value without used items;
- a `user_solver_kappa.n` above `items - human_skipped`;
- undefined resamples under AC1, whose chance agreement cannot reach one.

It supports strict JSON round trips. Input errors use fixed Forge `ValidationError` codes:

- `agreement_threshold_invalid` for a threshold outside `[-1, 1]`;
- `agreement_exclusion_limit_invalid` for a fraction outside `[0, 1)`, a non-finite one or a
  non-number;
- `agreement_reference_invalid` for a blank `source_ref` or `version_ref`, as
  `reference_prevalence` refuses.
