"""Independent integer count reports, including invalid-output and forged-model boundaries."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import cast

import pytest

from strata_forge.core.errors import ValidationError
from strata_forge.metrics import (
    ConfusionMatrix,
    classification_report,
    classification_report_from_confusion,
)


def matrix() -> ConfusionMatrix:
    return ConfusionMatrix(
        labels=("A", "B", "C"),
        counts=((2, 1, 0), (0, 1, 0), (0, 0, 0)),
        invalid_by_gold=(1, 1, 0),
        n=6,
    )


def test_known_count_report_keeps_invalid_false_negatives_and_absent_categories() -> None:
    original = matrix()
    report = classification_report_from_confusion(original)
    assert report.confusion == original
    assert report.confusion is not original
    assert report.accuracy.value == float(Fraction(1, 2))
    assert report.macro_f1.value is not None
    assert math.isclose(report.macro_f1.value, float(Fraction(7, 18)), rel_tol=0, abs_tol=1e-15)
    assert report.invalid_rate.value == float(Fraction(1, 3))
    assert tuple(row.support for row in report.per_class) == (4, 2, 0)
    assert tuple(row.predicted for row in report.per_class) == (2, 2, 0)
    assert tuple(row.f1 for row in report.per_class) == (float(Fraction(2, 3)), 0.5, 0.0)
    assert report.accuracy.n == report.invalid_rate.n == report.macro_f1.n == 6


@pytest.mark.parametrize("factor", [1, 3, 2**80])
def test_integer_multiplicity_preserves_rates_and_records_actual_weighted_count(
    factor: int,
) -> None:
    original = matrix()
    weighted = ConfusionMatrix(
        labels=original.labels,
        counts=tuple(tuple(value * factor for value in row) for row in original.counts),
        invalid_by_gold=tuple(value * factor for value in original.invalid_by_gold),
        n=original.n * factor,
    )
    report = classification_report_from_confusion(weighted)
    assert report.accuracy.value == 0.5
    assert report.invalid_rate.value == float(Fraction(1, 3))
    assert report.macro_f1.value is not None
    assert math.isclose(report.macro_f1.value, float(Fraction(7, 18)), rel_tol=0, abs_tol=1e-15)
    assert report.accuracy.n == 6 * factor
    assert report.per_class[0].support == 4 * factor


def test_count_report_equals_explicit_expanded_rows_under_permuted_label_order() -> None:
    original = matrix()
    gold: list[object] = []
    predicted: list[object] = []
    for index, label in enumerate(original.labels):
        for column, pred in enumerate(original.labels):
            gold.extend([label] * original.counts[index][column])
            predicted.extend([pred] * original.counts[index][column])
        gold.extend([label] * original.invalid_by_gold[index])
        predicted.extend([None] * original.invalid_by_gold[index])
    expected = classification_report(gold, predicted, labels=original.labels)
    assert classification_report_from_confusion(original) == expected
    perm = (2, 0, 1)
    reordered = ConfusionMatrix(
        labels=tuple(original.labels[i] for i in perm),
        counts=tuple(tuple(original.counts[i][j] for j in perm) for i in perm),
        invalid_by_gold=tuple(original.invalid_by_gold[i] for i in perm),
        n=6,
    )
    assert classification_report_from_confusion(reordered) == classification_report(
        gold, predicted, labels=reordered.labels
    )


def test_empty_counts_and_all_invalid_outputs_preserve_existing_undefined_policy() -> None:
    empty = classification_report_from_confusion(
        ConfusionMatrix(labels=("A", "B"), counts=((0, 0), (0, 0)), invalid_by_gold=(0, 0), n=0)
    )
    for metric in (empty.accuracy, empty.macro_f1, empty.invalid_rate):
        assert metric.value is None
        assert metric.reason == "empty"
        assert metric.n == 0
    invalid = classification_report_from_confusion(
        ConfusionMatrix(labels=("A", "B"), counts=((0, 0), (0, 0)), invalid_by_gold=(3, 0), n=3)
    )
    assert invalid.accuracy.value == invalid.macro_f1.value == 0.0
    assert invalid.invalid_rate.value == 1.0
    assert invalid.per_class[0].support == 3


@pytest.mark.parametrize(
    "patch",
    [
        {"labels": ("A",)},
        {"labels": ("A", "A", "C")},
        {"labels": ("A", " ", "C")},
        {"labels": ["A", "B", "C"]},
        {"counts": ((2, 1), (0, 1))},
        {"counts": ((2, 1, 0),)},
        {"counts": ((2, 1, 0), (0, -1, 0), (0, 0, 0))},
        {"counts": ((2.0, 1, 0), (0, 1, 0), (0, 0, 0))},
        {"counts": ((True, 1, 0), (0, 1, 0), (0, 0, 0))},
        {"invalid_by_gold": (1, 1)},
        {"invalid_by_gold": (-1, 3, 0)},
        {"invalid_by_gold": (True, 1, 0)},
        {"n": 5},
        {"n": True},
        {"n": -1},
        {"n": 6.0},
        {"unknown": "fictional"},
    ],
)
def test_manually_changed_or_constructed_models_cannot_bypass_original_validation(
    patch: dict[str, object],
) -> None:
    forged = matrix().model_copy(update=patch)
    with pytest.raises(ValidationError, match=r"^confusion matrix is invalid$") as caught:
        classification_report_from_confusion(forged)
    assert caught.value.__context__ is caught.value.__cause__ is None
    raw = matrix().model_dump(mode="python")
    raw.update(patch)
    constructed = ConfusionMatrix.model_construct(**raw)
    # model_construct ignores forbidden extras, while an actual hidden extra above is refused.
    if "unknown" not in patch:
        with pytest.raises(ValidationError, match=r"^confusion matrix is invalid$"):
            classification_report_from_confusion(constructed)


def test_valid_model_construct_is_rebuilt_instead_of_trusted_or_blanket_rejected() -> None:
    constructed = ConfusionMatrix.model_construct(**matrix().model_dump(mode="python"))
    report = classification_report_from_confusion(constructed)
    assert report == classification_report_from_confusion(matrix())
    assert report.confusion is not constructed
    object.__setattr__(constructed, "n", 999)
    assert report.confusion.n == 6


def test_exact_entry_rejects_subclass_and_nonmodels_before_property_access() -> None:
    calls: list[str] = []

    class Hostile(ConfusionMatrix):
        def __getattribute__(self, name: str) -> object:
            calls.append(name)
            raise AssertionError("must not inspect a subclass")

    subclass = object.__new__(Hostile)
    for value in (None, {}, matrix().model_dump(), subclass):
        with pytest.raises(ValidationError, match=r"^confusion matrix is invalid$"):
            classification_report_from_confusion(cast("ConfusionMatrix", value))
    assert calls == []


def test_hostile_nested_scalars_and_containers_are_not_coerced_or_evaluated() -> None:
    calls: list[str] = []

    class BadInt(int):
        def __lt__(self, other: object) -> bool:
            calls.append("lt")
            raise AssertionError("must not compare subclass")

        def __add__(self, other: object) -> int:
            calls.append("add")
            raise AssertionError("must not add subclass")

    class BadStr(str):
        def strip(self, *args: object, **kwargs: object) -> str:
            calls.append("strip")
            raise AssertionError("must not coerce subclass")

        def __hash__(self) -> int:
            calls.append("hash")
            raise AssertionError("must not hash subclass")

    class BadTuple(tuple[object, ...]):
        def __iter__(self):
            calls.append("iter")
            raise AssertionError("must not walk subclass")

    for patch in (
        {"n": BadInt(6)},
        {"invalid_by_gold": (BadInt(1), 1, 0)},
        {"labels": ("A", BadStr("B"), "C")},
        {"counts": (BadTuple((2, 1, 0)), (0, 1, 0), (0, 0, 0))},
    ):
        with pytest.raises(ValidationError, match=r"^confusion matrix is invalid$"):
            classification_report_from_confusion(matrix().model_copy(update=patch))
    assert calls == []


def test_forged_storage_and_extra_metadata_do_not_execute_callbacks() -> None:
    calls: list[str] = []

    class BadDict(dict[str, object]):
        def copy(self) -> dict[str, object]:
            calls.append("copy")
            raise AssertionError("must not copy a subclass")

    class BadExtra:
        def __bool__(self) -> bool:
            calls.append("bool")
            raise AssertionError("must not test extra truthiness")

    forged = matrix()
    object.__setattr__(forged, "__dict__", BadDict(forged.__dict__))
    with pytest.raises(ValidationError, match=r"^confusion matrix is invalid$"):
        classification_report_from_confusion(forged)
    forged = matrix()
    object.__setattr__(forged, "__pydantic_extra__", BadExtra())
    with pytest.raises(ValidationError, match=r"^confusion matrix is invalid$"):
        classification_report_from_confusion(forged)
    assert calls == []


def test_nonstring_storage_key_never_rehashes_or_compares_untrusted_key() -> None:
    calls: list[str] = []

    class Key(str):
        def __hash__(self) -> int:
            calls.append("hash")
            return super().__hash__()

        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            return super().__eq__(other)

    forged = matrix()
    storage = cast("dict[object, object]", object.__getattribute__(forged, "__dict__"))
    storage[Key("fictional_extra")] = object()
    calls.clear()
    with pytest.raises(ValidationError, match=r"^confusion matrix is invalid$"):
        classification_report_from_confusion(forged)
    assert calls == []


@pytest.mark.parametrize("field", ["labels", "counts", "invalid_by_gold", "n"])
def test_missing_original_field_is_a_fixed_validation_failure(field: str) -> None:
    forged = matrix()
    object.__delattr__(forged, field)
    with pytest.raises(ValidationError, match=r"^confusion matrix is invalid$") as caught:
        classification_report_from_confusion(forged)
    assert caught.value.__context__ is caught.value.__cause__ is None


@pytest.mark.parametrize(
    "patch",
    [
        {"counts": [[2, 1, 0], [0, 1, 0], [0, 0, 0]]},
        {"counts": ([2, 1, 0], (0, 1, 0), (0, 0, 0))},
        {"counts": (2, (0, 1, 0), (0, 0, 0))},
        {"invalid_by_gold": [1, 1, 0]},
    ],
)
def test_mutable_or_nonrow_shapes_are_not_coerced(patch: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match=r"^confusion matrix is invalid$"):
        classification_report_from_confusion(matrix().model_copy(update=patch))
