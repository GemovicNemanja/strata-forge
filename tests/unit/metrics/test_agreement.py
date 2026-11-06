"""Synthetic nominal counts, exact selector boundaries and reference-provenance refusals."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import pytest
from pydantic import ValidationError as ModelValidationError

from strata_forge.core.errors import ValidationError
from strata_forge.metrics.agreement import (
    AGREEMENT_VERSION,
    RESEARCH_ANNOTATION_THRESHOLD,
    AgreementEstimate,
    AgreementGate,
    HumanRatings,
    PairwiseAgreement,
    ReferencePrevalence,
    RegisteredReference,
    pairwise_agreement,
    reference_prevalence,
    research_annotation_gate,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

LABELS = ("E", "C", "U")


class Case(TypedDict):
    name: str
    labels: list[str]
    counts: list[list[int]]
    percent: list[int]
    kappa_chance: list[int]
    ac1_chance: list[int]
    kappa: list[int]
    ac1: list[int]


class Fixture(TypedDict):
    metric_version: str
    cases: list[Case]


def _human(name: str, ratings: Sequence[str], ids: Sequence[str] | None = None) -> HumanRatings:
    return HumanRatings(
        rater_id=name,
        rater_kind="human",
        item_ids=tuple(ids) if ids is not None else tuple(f"item-{i}" for i in range(len(ratings))),
        ratings=tuple(ratings),
    )


def _from_matrix(
    matrix: Sequence[Sequence[int]], labels: Sequence[str] = LABELS
) -> PairwiseAgreement:
    a: list[str] = []
    b: list[str] = []
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            a.extend([labels[i]] * count)
            b.extend([labels[j]] * count)
    return pairwise_agreement(_human("alice", a), _human("bob", b), labels=labels)


def _references(values: Sequence[str], labels: Sequence[str] = LABELS) -> ReferencePrevalence:
    return reference_prevalence(
        tuple(
            RegisteredReference(
                item_id=f"item-{i}",
                solver_gold=label,
                adjudicated=None,
                source_ref="synthetic-solver",
                version_ref="fixture-v1",
            )
            for i, label in enumerate(values)
        ),
        labels=labels,
        item_set_ref="synthetic-items-v1",
        reference_version="registered-fixture-v1",
    )


def test_frozen_independently_counted_conformance_fixture() -> None:
    data = cast(
        "Fixture",
        json.loads((Path(__file__).parent / "fixtures/agreement_point_v1.json").read_text()),
    )
    assert data["metric_version"] == AGREEMENT_VERSION
    for case in data["cases"]:
        report = _from_matrix(case["counts"], case["labels"])
        assert report.counts == tuple(tuple(row) for row in case["counts"])
        for actual, expected in (
            (report.percent_agreement.value, case["percent"]),
            (report.cohen_kappa.chance_agreement, case["kappa_chance"]),
            (report.gwet_ac1.chance_agreement, case["ac1_chance"]),
            (report.cohen_kappa.value, case["kappa"]),
            (report.gwet_ac1.value, case["ac1"]),
        ):
            assert actual == float(Fraction(*expected)), case["name"]
        assert report.n == sum(map(sum, case["counts"]))
        assert report.agreement_count == sum(
            case["counts"][i][i] for i in range(len(case["labels"]))
        )
        assert report.confidence_method == "not_produced"
        assert PairwiseAgreement.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize("majority", LABELS)
@pytest.mark.parametrize(("count", "selected"), [(15, "cohen_kappa"), (16, "gwet_ac1")])
def test_inclusive_four_fifths_reference_switch_for_every_category(
    majority: str, count: int, selected: str
) -> None:
    agreement = _from_matrix([[8, 2, 0], [1, 9, 0], [0, 0, 0]])
    other = next(label for label in LABELS if label != majority)
    prevalence = _references([majority] * count + [other] * (20 - count))
    gate = research_annotation_gate(agreement, prevalence)
    assert gate.selected_coefficient == selected
    assert gate.prevalence.n == 20
    assert gate.prevalence.counts[LABELS.index(majority)] == count
    assert gate.agreement == agreement
    assert gate.threshold == RESEARCH_ANNOTATION_THRESHOLD == 0.7
    assert gate.confidence_method == "not_produced"


def test_reference_switch_never_replaces_rater_chance_terms() -> None:
    agreement = _from_matrix([[8, 2, 0], [1, 9, 0], [0, 0, 0]])
    balanced = research_annotation_gate(agreement, _references(["E"] * 15 + ["C"] * 5))
    prevalent = research_annotation_gate(agreement, _references(["E"] * 16 + ["C"] * 4))
    assert balanced.selected_coefficient != prevalent.selected_coefficient
    assert balanced.agreement == prevalent.agreement == agreement
    assert agreement.cohen_kappa.chance_agreement == 0.5
    assert agreement.gwet_ac1.chance_agreement == float(Fraction(399, 1600))
    different_raters = _from_matrix([[16, 0, 0], [0, 4, 0], [0, 0, 0]])
    other = research_annotation_gate(different_raters, prevalent.prevalence)
    assert other.selected_coefficient == prevalent.selected_coefficient
    assert other.agreement.cohen_kappa.chance_agreement != agreement.cohen_kappa.chance_agreement


def test_inclusive_point_threshold_uses_exact_ratio_not_display_rounding() -> None:
    labels = ("yes", "no")
    agreement = _from_matrix([[8, 2], [1, 9]], labels)
    reference = _references(["yes"] * 15 + ["no"] * 5, labels)
    gate = research_annotation_gate(agreement, reference)
    assert gate.selected_value == 0.7
    assert gate.passed is True
    assert (
        research_annotation_gate(agreement, reference, threshold=math.nextafter(0.7, 1)).passed
        is False
    )
    assert research_annotation_gate(agreement, reference, threshold=0.69).passed is True
    assert AgreementGate.model_validate_json(gate.model_dump_json()) == gate


def test_gold_precedes_different_valid_adjudication_and_null_gold_falls_back() -> None:
    records = (
        RegisteredReference(
            item_id="x", solver_gold="E", adjudicated="C", source_ref="solver", version_ref="v1"
        ),
        RegisteredReference(
            item_id="y",
            solver_gold=None,
            adjudicated="U",
            source_ref="adjudication",
            version_ref="v2",
        ),
    )
    result = reference_prevalence(
        records, labels=LABELS, item_set_ref="registered-items", reference_version="refs-v3"
    )
    assert result.counts == (1, 0, 1)
    assert result.item_ids == ("x", "y")
    assert [
        (item.label, item.basis, item.source_ref, item.version_ref) for item in result.references
    ] == [
        ("E", "solver_gold", "solver", "v1"),
        ("U", "adjudicated", "adjudication", "v2"),
    ]
    assert result.item_set_ref == "registered-items"
    assert result.reference_version == "refs-v3"
    assert ReferencePrevalence.model_validate_json(result.model_dump_json()) == result


def test_missing_reference_or_invalid_gold_never_falls_back_to_human_ratings() -> None:
    for gold, adjudicated in [(None, None), ("unknown-category", "E"), ("E", "unknown-category")]:
        record = RegisteredReference(
            item_id="x",
            solver_gold=gold,
            adjudicated=adjudicated,
            source_ref="fixture",
            version_ref="v1",
        )
        with pytest.raises(ValidationError, match=r"^agreement_reference_invalid$"):
            reference_prevalence(
                (record,), labels=LABELS, item_set_ref="items", reference_version="v1"
            )


def test_identical_one_class_raters_report_undefined_kappa_but_defined_ac1() -> None:
    report = _from_matrix([[20, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert report.percent_agreement.value == report.gwet_ac1.value == 1
    assert report.cohen_kappa.value is None
    assert report.cohen_kappa.reason == "chance_agreement_one"
    assert report.cohen_kappa.chance_agreement == 1
    selected_ac1 = research_annotation_gate(report, _references(["E"] * 16 + ["C"] * 4))
    assert selected_ac1.passed is True
    selected_kappa = research_annotation_gate(report, _references(["E"] * 15 + ["C"] * 5))
    assert selected_kappa.passed is selected_kappa.selected_value is None
    assert selected_kappa.reason == "selected_coefficient_undefined"
    assert selected_kappa.selected_reason == "chance_agreement_one"


def test_empty_populations_are_undefined_without_selecting_a_gate() -> None:
    report = pairwise_agreement(_human("alice", []), _human("bob", []), labels=LABELS)
    for estimate in (report.percent_agreement, report.cohen_kappa, report.gwet_ac1):
        assert estimate.value is estimate.chance_agreement is None
        assert estimate.reason == "empty"
    gate = research_annotation_gate(report, _references([]))
    assert gate.selected_coefficient is gate.selected_value is gate.passed is None
    assert gate.reason == gate.selected_reason == "empty"
    assert gate.prevalence.counts == (0, 0, 0)


def test_complete_categories_change_ac1_even_if_one_was_never_used() -> None:
    a = _from_matrix([[8, 2], [1, 9]], ("E", "C"))
    b = _from_matrix([[8, 2, 0], [1, 9, 0], [0, 0, 0]])
    assert a.cohen_kappa.value == b.cohen_kappa.value == 0.7
    assert b.gwet_ac1.value is not None
    assert a.gwet_ac1.value is not None
    assert b.gwet_ac1.value > a.gwet_ac1.value
    assert b.labels == LABELS


def test_negative_coefficients_single_item_and_label_permutation_are_not_clipped() -> None:
    a, b = _human("alice", ["E"]), _human("bob", ["C"])
    binary = pairwise_agreement(a, b, labels=("E", "C"))
    assert binary.cohen_kappa.value == 0
    assert binary.gwet_ac1.value == -1
    report = pairwise_agreement(a, b, labels=LABELS)
    reverse = pairwise_agreement(a, b, labels=("U", "C", "E"))
    assert report.gwet_ac1.value == reverse.gwet_ac1.value == float(Fraction(-1, 3))


@pytest.mark.parametrize(
    "changes",
    [
        {"rater_id": "alice"},
        {"item_ids": ("other",)},
        {"item_ids": ("x", "x"), "ratings": ("E", "C")},
        {"item_ids": ("x", "y"), "ratings": ("E",)},
        {"ratings": (None,)},
        {"ratings": ("missing",)},
        {"rater_kind": "model"},
        {"rater_id": " "},
    ],
)
def test_invalid_human_pairs_refuse_without_echoing_values(changes: dict[str, object]) -> None:
    raw: dict[str, object] = {
        "rater_id": "bob",
        "rater_kind": "human",
        "item_ids": ("x",),
        "ratings": ("E",),
    }
    raw.update(changes)
    b = HumanRatings.model_construct(
        rater_id=raw["rater_id"],
        rater_kind=raw["rater_kind"],
        item_ids=raw["item_ids"],
        ratings=raw["ratings"],
    )
    with pytest.raises(ValidationError, match=r"^agreement_input_invalid$") as error:
        pairwise_agreement(_human("alice", ["E"], ["x"]), b, labels=LABELS)
    assert error.value.__context__ is None


def test_reordered_or_partial_reference_ids_and_categories_refuse() -> None:
    report = _from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    base = _references(["E", "C"])
    for reference in (
        _references(["E"]),
        base.model_copy(
            update={
                "item_ids": tuple(reversed(base.item_ids)),
                "references": tuple(reversed(base.references)),
            }
        ),
        _references(["E", "C"], labels=("C", "E", "U")),
    ):
        with pytest.raises(ValidationError, match=r"^agreement_population_mismatch$"):
            research_annotation_gate(report, reference)
    with pytest.raises(ValidationError, match=r"^agreement_reference_invalid$"):
        reference_prevalence(
            (cast("RegisteredReference", base.references[0]),),
            labels=LABELS,
            item_set_ref="items",
            reference_version="v1",
        )


def test_duplicate_and_missing_provenance_are_not_silently_dropped() -> None:
    good = RegisteredReference(
        item_id="x", solver_gold="E", adjudicated=None, source_ref="source", version_ref="v1"
    )
    for records in (
        (good, good),
        (good.model_copy(update={"source_ref": ""}),),
        (good.model_copy(update={"version_ref": ""}),),
    ):
        with pytest.raises(ValidationError, match=r"^agreement_reference_invalid$"):
            reference_prevalence(
                records, labels=LABELS, item_set_ref="items", reference_version="v1"
            )


@pytest.mark.parametrize("labels", [None, [], ["E"], ["E", "E"], ["E", " "], "EC", ["E", 1]])
def test_label_space_must_be_explicit_complete_and_exact(labels: object) -> None:
    with pytest.raises(ValidationError, match=r"^agreement_labels_invalid$"):
        pairwise_agreement(
            _human("alice", ["E"]), _human("bob", ["E"]), labels=cast("Sequence[str]", labels)
        )


@pytest.mark.parametrize(
    "threshold",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        True,
        "0.7",
        pytest.param(10**10_000, id="huge-integer"),
    ],
)
def test_invalid_threshold_raises_fixed_error(threshold: object) -> None:
    with pytest.raises(ValidationError, match=r"^agreement_threshold_invalid$"):
        research_annotation_gate(
            _from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
            _references(["E", "C"]),
            threshold=cast("float", threshold),
        )


def test_frozen_reports_reject_inconsistent_serialized_or_constructed_results() -> None:
    report = _from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    with pytest.raises(ModelValidationError):
        PairwiseAgreement.__setattr__(report, "n", 9)
    for changed in (
        {"n": 9},
        {"counts": ((0, 0, 0),) * 3},
        {"agreement_count": 0},
        {"cohen_kappa": {"value": 0.0, "chance_agreement": 0.5, "reason": None}},
    ):
        raw = {**report.model_dump(mode="json"), **changed}
        with pytest.raises(ModelValidationError):
            PairwiseAgreement.model_validate_json(json.dumps(raw))
    forged = report.model_copy(update={"agreement_count": 0})
    with pytest.raises(ValidationError, match=r"^agreement_report_invalid$"):
        research_annotation_gate(forged, _references(["E", "C"]))


@pytest.mark.parametrize(
    "changes",
    [
        {"value": float("nan")},
        {"chance_agreement": float("inf")},
        {"value": None, "reason": None},
        {"value": 1.0, "reason": "empty"},
        {"value": None, "chance_agreement": 0.5, "reason": "chance_agreement_one"},
    ],
)
def test_estimates_refuse_nonfinite_or_inconsistent_undefined_state(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ModelValidationError):
        AgreementEstimate.model_validate(
            {"value": 0.5, "chance_agreement": None, "reason": None, **changes}
        )


def test_reference_and_gate_roundtrips_refuse_count_decision_and_confidence_forgery() -> None:
    report = _from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    reference = _references(["E", "C"])
    with pytest.raises(ValidationError, match=r"^agreement_reference_invalid$"):
        research_annotation_gate(report, reference.model_copy(update={"counts": (2, 0, 0)}))
    gate = research_annotation_gate(report, reference)
    for changes in (
        {"passed": False},
        {"confidence_method": "bootstrap"},
        {"confidence_interval": [0.8, 1.0]},
    ):
        with pytest.raises(ModelValidationError):
            AgreementGate.model_validate_json(
                json.dumps({**gate.model_dump(mode="json"), **changes})
            )


def test_invalid_reference_container_and_missing_set_provenance_have_fixed_errors() -> None:
    with pytest.raises(ValidationError, match=r"^agreement_reference_invalid$"):
        reference_prevalence(
            cast("Sequence[RegisteredReference]", None),
            labels=LABELS,
            item_set_ref="items",
            reference_version="v1",
        )
    for item_set, version in [("", "v1"), ("items", " ")]:
        with pytest.raises(ValidationError, match=r"^agreement_reference_invalid$"):
            reference_prevalence(
                (), labels=LABELS, item_set_ref=item_set, reference_version=version
            )


def test_import_and_actual_operations_require_no_numpy_or_ml_packages() -> None:
    program = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"numpy", "torch", "transformers", "scipy", "pandas", "sklearn"}:
            raise AssertionError("heavy import attempted")
sys.meta_path.insert(0, Block())
from strata_forge.metrics.agreement import HumanRatings, pairwise_agreement
a = HumanRatings(rater_id="a", rater_kind="human", item_ids=("1",), ratings=("E",))
b = HumanRatings(rater_id="b", rater_kind="human", item_ids=("1",), ratings=("E",))
assert pairwise_agreement(a,b,labels=("E","C")).gwet_ac1.value == 1
"""
    result = subprocess.run(  # noqa: S603 — fixed synthetic program in the current interpreter.
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
