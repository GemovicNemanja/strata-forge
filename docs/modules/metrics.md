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
