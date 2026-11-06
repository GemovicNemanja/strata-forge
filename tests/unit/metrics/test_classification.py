"""Fixed reference populations, including invalid predictions and probability omissions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import pytest
from pydantic import ValidationError as ModelValidationError

from strata_forge.core.errors import ValidationError
from strata_forge.metrics import (
    METRIC_VERSION,
    ClassificationReport,
    ClassScore,
    ConfusionMatrix,
    LabelSpace,
    MetricValue,
    accuracy,
    brier,
    classification_report,
    confusion_matrix,
    invalid_rate,
    macro_f1,
    nll,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

LABELS = ("entailment", "contradiction", "unknown")


class Population(TypedDict):
    labels: list[str]
    gold: list[str]
    predictions: list[str | None]
    probabilities: list[list[float]]
    counts: list[list[int]]
    invalid_by_gold: list[int]
    support: list[int]
    accuracy: float
    macro_f1: float
    invalid_rate: float
    brier_sum_mean: float
    nll_gold_probability_product: float


class Fixture(TypedDict):
    metric_version: str
    three_way: Population
    two_way: Population


@pytest.fixture
def reference() -> Fixture:
    path = Path(__file__).parent / "fixtures" / "classification_point_v1.json"
    return cast("Fixture", json.loads(path.read_text()))


def test_fixed_three_way_population(reference: Fixture) -> None:
    data = reference["three_way"]
    result = classification_report(data["gold"], data["predictions"], labels=data["labels"])
    assert result.metric_version == reference["metric_version"] == METRIC_VERSION
    assert result.confusion.counts == tuple(tuple(row) for row in data["counts"])
    assert result.confusion.invalid_by_gold == tuple(data["invalid_by_gold"])
    assert [score.support for score in result.per_class] == data["support"]
    assert result.accuracy.value == data["accuracy"]
    assert result.macro_f1.value == pytest.approx(data["macro_f1"])
    assert result.invalid_rate.value == data["invalid_rate"]
    assert result.accuracy.n == result.macro_f1.n == result.invalid_rate.n == 4
    assert result.accuracy.excluded == 0
    assert result.accuracy.coverage == 1
    assert result.per_class[0].precision == result.per_class[0].recall == 0.5
    assert ClassificationReport.model_validate_json(result.model_dump_json()) == result


def test_invalid_outputs_reduce_recall_and_never_disappear() -> None:
    report = classification_report(["entailment"] * 2, ["entailment", None], labels=LABELS)
    assert report.per_class[0].support == 2
    assert report.per_class[0].recall == 0.5
    assert report.per_class[0].f1 == pytest.approx(2 / 3)
    assert report.accuracy.value == report.invalid_rate.value == 0.5
    assert sum(map(sum, report.confusion.counts)) + sum(report.confusion.invalid_by_gold) == 2


def test_absent_declared_classes_contribute_zero() -> None:
    report = classification_report(["entailment"], ["entailment"], labels=LABELS)
    assert report.macro_f1.value == pytest.approx(1 / 3)
    assert report.per_class[1].support == report.per_class[1].f1 == 0
    assert report.per_class[2].precision == report.per_class[2].recall == 0


def test_empty_population_is_undefined() -> None:
    report = classification_report([], [], labels=LABELS)
    for metric in (report.accuracy, report.macro_f1, report.invalid_rate):
        assert metric.value is None
        assert metric.n == metric.excluded == 0
        assert metric.coverage is None
        assert metric.reason == "empty"
    assert all(score.support == 0 for score in report.per_class)


def test_public_convenience_functions_have_the_same_denominators() -> None:
    assert accuracy(["entailment"], [None], labels=LABELS).value == 0
    assert macro_f1(["entailment"], [None], labels=LABELS).value == 0
    assert invalid_rate(["entailment"], [None], labels=LABELS).value == 1


def test_integer_mapping_is_explicit_and_can_collapse_classes() -> None:
    result = classification_report(
        [7, 9, 11], [7, 11, 9], labels=("yes", "no"), index_map={7: "yes", 9: "no", 11: "no"}
    )
    assert result.accuracy.value == 1
    assert result.confusion.counts == ((1, 0), (0, 2))


@pytest.mark.parametrize(("gold", "predictions"), [([0], ["entailment"]), (["entailment"], [0])])
def test_integer_order_is_never_inferred(gold: list[object], predictions: list[object]) -> None:
    with pytest.raises(ValidationError, match="explicit index_map"):
        accuracy(gold, predictions, labels=LABELS)


@pytest.mark.parametrize("gold", [["missing"], [None], [True], [3.0], [99]])
def test_invalid_gold_refused(gold: list[object]) -> None:
    with pytest.raises(ValidationError, match="gold label"):
        accuracy(gold, ["entailment"], labels=LABELS, index_map={0: "entailment"})


@pytest.mark.parametrize("guess", ["missing", None, True, 3.0, 99, {"text": "entailment"}])
def test_invalid_predictions_count_as_wrong(guess: object) -> None:
    result = classification_report(
        ["entailment"], [guess], labels=LABELS, index_map={0: "entailment"}
    )
    assert result.accuracy.value == 0
    assert result.invalid_rate.value == 1
    assert result.confusion.invalid_by_gold == (1, 0, 0)


@pytest.mark.parametrize("labels", [(), ("one",), ("one", "one"), ("one", " ")])
def test_invalid_label_spaces_refused(labels: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="ordered label space"):
        accuracy([], [], labels=labels)


def test_a_single_string_is_not_a_collection_of_class_names() -> None:
    with pytest.raises(ValidationError, match="single string"):
        accuracy([], [], labels="entailment")


def test_multiclass_metrics_have_no_arbitrary_small_label_cap() -> None:
    labels = tuple(f"class_{index}" for index in range(40))
    assert accuracy([labels[0]], [labels[0]], labels=labels).value == 1
    assert macro_f1([labels[0]], [labels[0]], labels=labels).value == pytest.approx(1 / 40)


@pytest.mark.parametrize("index_map", [{True: "entailment"}, {0: "missing"}, {"0": "entailment"}])
def test_invalid_index_maps_refused(index_map: Mapping[object, str]) -> None:
    with pytest.raises(ValidationError, match="index_map"):
        accuracy([], [], labels=LABELS, index_map=cast("Mapping[int, str]", index_map))


def test_population_lengths_must_match() -> None:
    with pytest.raises(ValidationError, match="same length"):
        confusion_matrix(["entailment"], [], labels=LABELS)
    with pytest.raises(ValidationError, match="same length"):
        brier(["entailment"], [], labels=LABELS, probability_kind="class_probabilities")


def test_fixed_probability_reference(reference: Fixture) -> None:
    data = reference["three_way"]
    result = brier(
        data["gold"],
        data["probabilities"],
        labels=data["labels"],
        probability_kind="class_probabilities",
    )
    assert result.value == pytest.approx(data["brier_sum_mean"])
    assert result.n == 4
    loss = nll(
        data["gold"],
        data["probabilities"],
        labels=data["labels"],
        probability_kind="class_probabilities",
    )
    assert loss.value == pytest.approx(-math.log(data["nll_gold_probability_product"]) / 4)
    assert loss.clipped == 0
    two = reference["two_way"]
    assert brier(
        two["gold"],
        two["probabilities"],
        labels=two["labels"],
        probability_kind="class_probabilities",
    ).value == pytest.approx(two["brier_sum_mean"])


def test_probability_provenance_and_missing_coverage_are_explicit() -> None:
    result = brier(
        ["entailment", "unknown"],
        [[1.0, 0.0, 0.0], None],
        labels=LABELS,
        probability_kind="restricted_softmax",
        calibration_ref="calibration@012345",
    )
    assert result.value == 0
    assert result.n == result.excluded == 1
    assert result.coverage == 0.5
    assert result.probability_kind == "restricted_softmax"
    assert result.calibration_ref == "calibration@012345"
    assert MetricValue.model_validate_json(result.model_dump_json()) == result


def test_all_missing_and_empty_probabilities_have_distinct_reasons() -> None:
    missing = nll(["entailment"], [None], labels=LABELS, probability_kind="class_probabilities")
    assert missing.value is None
    assert missing.reason == "no_probabilities"
    assert missing.n == 0
    assert missing.excluded == 1
    assert missing.coverage == 0
    empty = nll([], [], labels=LABELS, probability_kind="class_probabilities")
    assert empty.value is None
    assert empty.reason == "empty"
    assert empty.coverage is None


@pytest.mark.parametrize(
    "row",
    [
        [0.2, 0.2, 0.2],
        [1.0, 0.0],
        [float("nan"), 0.0, 1.0],
        [float("inf"), 0.0, 0.0],
        [-0.1, 0.5, 0.6],
        [1.1, 0.0, 0.0],
        [True, 0.0, 0.0],
        ["1", 0, 0],
        "100",
        1,
    ],
)
def test_malformed_probability_rows_are_not_renormalized(row: object) -> None:
    with pytest.raises(ValidationError):
        brier(["entailment"], [row], labels=LABELS, probability_kind="class_probabilities")


def test_topk_cannot_masquerade_as_exact_probabilities() -> None:
    with pytest.raises(ValidationError, match="provenance"):
        brier(["entailment"], [[1.0, 0.0, 0.0]], labels=LABELS, probability_kind="censored_topk")


def test_nll_clipping_count_and_policy() -> None:
    result = nll(
        ["entailment", "unknown"],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        labels=LABELS,
        probability_kind="class_probabilities",
        clip_epsilon=1e-4,
    )
    assert result.value == pytest.approx(-math.log(1e-4) / 2)
    assert result.clip_epsilon == 1e-4
    assert result.clipped == 1
    assert result.n == 2


@pytest.mark.parametrize("epsilon", [0.0, 1.0, -1.0, float("nan"), float("inf"), True])
def test_nll_invalid_clip_policy(epsilon: float) -> None:
    with pytest.raises(ValidationError, match="clip_epsilon"):
        nll([], [], labels=LABELS, probability_kind="class_probabilities", clip_epsilon=epsilon)


@pytest.mark.parametrize("tolerance", [-1.0, 1.0, float("nan"), True])
def test_invalid_normalization_policy(tolerance: float) -> None:
    with pytest.raises(ValidationError, match="tolerance"):
        brier(
            [],
            [],
            labels=LABELS,
            probability_kind="class_probabilities",
            normalization_tolerance=tolerance,
        )


def test_declared_tolerance_does_not_change_the_probability_vector() -> None:
    result = brier(
        ["entailment"],
        [[0.99, 0.0, 0.0]],
        labels=LABELS,
        probability_kind="class_probabilities",
        normalization_tolerance=0.02,
    )
    assert result.value == pytest.approx(0.0001)
    assert result.normalization_tolerance == 0.02


def test_data_models_refuse_inconsistent_serialized_artifacts() -> None:
    with pytest.raises(ModelValidationError, match="conserve"):
        ConfusionMatrix(labels=LABELS, counts=((0, 0, 0),) * 3, invalid_by_gold=(0, 0, 0), n=1)
    with pytest.raises(ModelValidationError, match="dimensions"):
        ConfusionMatrix(labels=LABELS, counts=((1, 0),), invalid_by_gold=(0,), n=1)
    with pytest.raises(ModelValidationError, match="coverage"):
        MetricValue(value=0.5, n=2, excluded=2, coverage=1.0)
    with pytest.raises(ModelValidationError):
        MetricValue(value=float("nan"), n=1)
    with pytest.raises(ModelValidationError):
        ClassScore(label="a", support=1, predicted=1, precision=2.0, recall=0.0, f1=0.0)
    with pytest.raises(ModelValidationError):
        LabelSpace(labels=("a", "b"), unexpected=True)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "values",
    [
        {"value": None, "n": 1, "reason": "empty"},
        {"value": 0.0, "n": 1, "clipped": 2},
        {"value": 0.0, "n": 1, "clip_epsilon": 0.0},
        {"value": 0.0, "n": 1, "normalization_tolerance": float("nan")},
        {"value": None, "n": 0, "excluded": 1, "reason": "empty"},
        {"value": None, "n": 0, "reason": "no_probabilities"},
        {"value": 0.0, "n": 1, "probability_kind": "censored_topk"},
    ],
)
def test_metric_artifact_accounting_cannot_claim_inconsistent_coverage(
    values: dict[str, object],
) -> None:
    with pytest.raises(ModelValidationError):
        MetricValue.model_validate(values)
