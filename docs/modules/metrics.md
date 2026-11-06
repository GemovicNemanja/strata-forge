# `strata_forge.metrics` — explicit classification point metrics

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
`passed`, undefined reason and `confidence_method="not_produced"`. A selected undefined kappa
has `passed=null`, `reason="selected_coefficient_undefined"` and its coefficient reason.
An empty slice selects nothing and has `passed=null`, `reason="empty"`. These point comparisons
do not implement a confidence interval, bootstrap, multi-rater method or completed G0/annotation
gate. Callers must enforce the frozen reference source, role separation and applicable study
protocol before interpreting them. Function-level refusals use fixed Forge `ValidationError`
codes without echoing supplied item/rater/source values; input-model validation hides values.

The [synthetic agreement fixture](../../tests/unit/metrics/fixtures/agreement_point_v1.json)
retains hand-counted rational results, including unequal marginals, negative coefficients
and a declared unused category. It contains no research annotations or model outputs.
