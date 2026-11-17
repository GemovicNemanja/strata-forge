"""One human against one model: counted exclusions, fixed selection, gate and interval."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError as ModelValidationError

from strata_forge.core.errors import ValidationError
from strata_forge.metrics.agreement import (
    HUMAN_MODEL_VERSION,
    MAX_HUMAN_SKIP_FRACTION,
    MAX_MODEL_NULL_FRACTION,
    HumanAnnotations,
    HumanModelAgreement,
    HumanRatings,
    ModelRatings,
    RegisteredReference,
    agreement_bootstrap,
    human_model_agreement,
    pairwise_agreement,
    reference_prevalence,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

LABELS = ("entail", "contra", "unknown")
E, C, U = LABELS
SEED = 20261104
SHA = "0123456789abcdef" * 4


def _ids(n: int) -> tuple[str, ...]:
    return tuple(f"slice-item-{i:03d}" for i in range(n))


def _human(ratings: Sequence[str | None]) -> HumanAnnotations:
    return HumanAnnotations(
        rater_id="user-1", rater_kind="human", item_ids=_ids(len(ratings)), ratings=tuple(ratings)
    )


def _model(ratings: Sequence[str | None], run: str | None = "run-7") -> ModelRatings:
    return ModelRatings(
        rater_id="model-1",
        rater_kind="model",
        item_ids=_ids(len(ratings)),
        ratings=tuple(ratings),
        model_prompt_sha256=SHA,
        model_run_ref=run,
    )


def _refs(
    gold: Sequence[str | None], adjudicated: Sequence[str | None] | None = None
) -> tuple[RegisteredReference, ...]:
    fallback = adjudicated if adjudicated is not None else [None] * len(gold)
    return tuple(
        RegisteredReference(
            item_id=item_id,
            solver_gold=g,
            adjudicated=a,
            source_ref="synthetic-source",
            version_ref="synthetic-1",
        )
        for item_id, g, a in zip(_ids(len(gold)), gold, fallback, strict=True)
    )


def _run(
    human: Sequence[str | None],
    model: Sequence[str | None],
    refs: Sequence[RegisteredReference],
    **options: float,
) -> HumanModelAgreement:
    return human_model_agreement(
        _human(human),
        _model(model),
        refs,
        labels=LABELS,
        seed=SEED,
        resamples=int(options.pop("resamples", 200)),
        **options,
    )


def _exact_kappa_7_10() -> tuple[list[str], list[str]]:
    """Forty items, marginals 20/20 for both raters, 34 agreements: kappa is exactly 7/10."""
    human = [E] * 17 + [C] * 17 + [E] * 3 + [C] * 3
    model = [E] * 17 + [C] * 17 + [C] * 3 + [E] * 3
    return human, model


def _pairs(n: int, skips: int = 0, nulls: int = 0) -> tuple[list[str | None], list[str | None]]:
    """A mostly agreeing balanced slice, with skips at the start and nulls at the end."""
    human: list[str | None] = [(E, C, U)[i % 3] for i in range(n)]
    model: list[str | None] = list(human)
    for i in range(0, n, 10):
        model[i] = (E, C, U)[(i + 1) % 3]
    for i in range(skips):
        human[i] = None
    for i in range(nulls):
        model[n - 1 - i] = None
    return human, model


def _balanced_refs(n: int) -> tuple[RegisteredReference, ...]:
    return _refs([(E, C, U)[i % 3] for i in range(n)])


# Rater-type separation


def test_model_raters_are_never_accepted_where_two_humans_are_required() -> None:
    model = _model([E, C])
    human = HumanRatings(rater_id="h", rater_kind="human", item_ids=_ids(2), ratings=(E, C))
    with pytest.raises(ValidationError, match="agreement_input_invalid"):
        pairwise_agreement(human, model, labels=LABELS)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="agreement_input_invalid"):
        pairwise_agreement(model, human, labels=LABELS)  # type: ignore[arg-type]
    prevalence = reference_prevalence(
        _refs([E, C]), labels=LABELS, item_set_ref="set", reference_version="v"
    )
    with pytest.raises(ValidationError, match="agreement_input_invalid"):
        agreement_bootstrap(human, model, prevalence, labels=LABELS, seed=SEED)  # type: ignore[arg-type]
    with pytest.raises(ModelValidationError):
        HumanRatings.model_validate({**model.model_dump(), "rater_kind": "human"})


def test_humans_are_never_accepted_where_the_model_is_required() -> None:
    human = _human([E, C])
    pair_human = HumanRatings(rater_id="h", rater_kind="human", item_ids=_ids(2), ratings=(E, C))
    for left, right in ((human, human), (pair_human, _model([E, C])), (_model([E, C]), human)):
        with pytest.raises(ValidationError, match="agreement_input_invalid"):
            human_model_agreement(
                left,  # type: ignore[arg-type]
                right,  # type: ignore[arg-type]
                _refs([E, C]),
                labels=LABELS,
                seed=SEED,
            )
    with pytest.raises(ModelValidationError):
        ModelRatings.model_validate({**human.model_dump(), "rater_kind": "human"})
    with pytest.raises(ModelValidationError):
        HumanAnnotations.model_validate(
            {**human.model_dump(), "model_prompt_sha256": SHA, "model_run_ref": None}
        )


@pytest.mark.parametrize("sha", ["0" * 63, "0" * 65, "A" * 64, "g" * 64, ""])
def test_model_provenance_requires_a_lowercase_64_hex_prompt_digest(sha: str) -> None:
    with pytest.raises(ModelValidationError):
        ModelRatings(
            rater_id="m",
            rater_kind="model",
            item_ids=("1",),
            ratings=(E,),
            model_prompt_sha256=sha,
            model_run_ref=None,
        )
    with pytest.raises(ModelValidationError):
        _model([E], run=" ")


def test_result_carries_model_provenance_but_no_item_ids_or_labels() -> None:
    human, model = _pairs(30)
    result = human_model_agreement(
        _human(human),
        _model(model, run=None),
        _balanced_refs(30),
        labels=LABELS,
        seed=SEED,
        resamples=50,
    )
    assert result.statistic_version == HUMAN_MODEL_VERSION == "nominal-human-vs-model-v1"
    assert (result.model_prompt_sha256, result.model_run_ref) == (SHA, None)
    text = result.model_dump_json()
    assert "slice-item" not in text
    assert all(label not in text for label in LABELS)
    assert HumanModelAgreement.model_validate_json(text) == result


def test_identical_rater_ids_and_misaligned_items_refuse() -> None:
    same = ModelRatings(
        rater_id="user-1",
        rater_kind="model",
        item_ids=_ids(2),
        ratings=(E, C),
        model_prompt_sha256=SHA,
        model_run_ref=None,
    )
    with pytest.raises(ValidationError, match="agreement_input_invalid"):
        human_model_agreement(_human([E, C]), same, _refs([E, C]), labels=LABELS, seed=SEED)
    with pytest.raises(ValidationError, match="agreement_input_invalid"):
        human_model_agreement(
            _human([E, C]), _model([E, C, C]), _refs([E, C]), labels=LABELS, seed=SEED
        )
    with pytest.raises(ValidationError, match="agreement_input_invalid"):
        _run([E, C], [E, "other"], _refs([E, C]))
    with pytest.raises(ValidationError, match="agreement_labels_invalid"):
        human_model_agreement(_human([E]), _model([E]), _refs([E]), labels=(E,), seed=SEED)


# Exclusions


def test_an_item_excluded_for_either_reason_leaves_both_series() -> None:
    human, model = _pairs(30)
    human[0] = human[1] = None
    model[1] = model[2] = None
    result = _run(human, model, _balanced_refs(30))
    assert (result.items, result.human_skipped, result.model_null, result.used) == (30, 2, 2, 27)
    kept = [i for i in range(30) if human[i] is not None and model[i] is not None]
    reduced = pairwise_agreement(
        HumanRatings(
            rater_id="a",
            rater_kind="human",
            item_ids=tuple(str(i) for i in kept),
            ratings=tuple(str(human[i]) for i in kept),
        ),
        HumanRatings(
            rater_id="b",
            rater_kind="human",
            item_ids=tuple(str(i) for i in kept),
            ratings=tuple(str(model[i]) for i in kept),
        ),
        labels=LABELS,
    )
    assert result.coefficient == "cohen_kappa"
    assert result.value is not None
    assert result.value.value == reduced.cohen_kappa.value


@pytest.mark.parametrize(("skips", "status"), [(12, "pass"), (13, "fail")])
def test_human_skips_fail_strictly_above_ten_percent(skips: int, status: str) -> None:
    human, model = _pairs(120, skips=skips)
    result = _run(human, model, _balanced_refs(120))
    assert result.human_skipped == skips
    assert result.status == status
    assert result.reasons == (() if status == "pass" else ("skip_fraction_exceeded",))
    assert result.value is not None
    assert result.interval is not None
    assert result.value.value > 0.7


@pytest.mark.parametrize(("nulls", "status"), [(12, "pass"), (13, "refused")])
def test_model_nulls_refuse_strictly_above_ten_percent(nulls: int, status: str) -> None:
    human, model = _pairs(120, nulls=nulls)
    result = _run(human, model, _balanced_refs(120))
    assert result.model_null == nulls
    assert result.status == status
    if status == "refused":
        assert result.reasons == ("model_null_fraction_exceeded",)
        assert (result.value, result.interval, result.cohen_kappa) == (None, None, None)
        assert result.coefficient == "cohen_kappa"


def test_both_limits_exceeded_fail_with_both_reasons_and_report_the_value() -> None:
    human, model = _pairs(120, skips=13, nulls=13)
    result = _run(human, model, _balanced_refs(120))
    assert result.status == "fail"
    assert result.reasons == ("skip_fraction_exceeded", "model_null_fraction_exceeded")
    assert result.value is not None
    assert result.interval is not None
    payload = result.model_dump(mode="python")
    for forged in (
        {"status": "refused", "value": None, "interval": None},
        {"reasons": ("skip_fraction_exceeded",)},
        {"reasons": ("model_null_fraction_exceeded",)},
    ):
        with pytest.raises(ModelValidationError):
            HumanModelAgreement.model_validate({**payload, **forged})


def test_a_skip_failure_lists_every_other_applicable_reason() -> None:
    human, model = _exact_kappa_7_10()
    skipped: list[str | None] = [None, None, None, None, None, *human[5:]]
    below = _run(skipped, model, _refs([E] * 20 + [C] * 20))
    assert below.status == "fail"
    assert below.reasons == ("skip_fraction_exceeded", "below_threshold")
    assert below.value is not None
    undefined = _run([None, None, E, E, E, E, E, E, E, E], [E] * 10, _refs([E] * 5 + [C] * 5))
    assert (undefined.status, undefined.reasons) == (
        "fail",
        ("skip_fraction_exceeded", "coefficient_undefined"),
    )
    assert (undefined.value, undefined.interval) == (None, None)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "reference_unavailable",
        "model_null_fraction_exceeded",
        "no_used_items",
        "undefined",
    ],
)
def test_a_refused_slice_never_reports_a_value_kappa_or_interval(case: str) -> None:
    human, model = _pairs(120, nulls=13 if case == "model_null_fraction_exceeded" else 0)
    refs: Sequence[RegisteredReference] = _balanced_refs(120)
    options: dict[str, float] = {}
    if case == "empty":
        human, model, refs = [], [], ()
    elif case == "reference_unavailable":
        refs = _refs([None, *[(E, C, U)[i % 3] for i in range(1, 120)]])
    elif case == "no_used_items":
        human = [None] * 60 + human[60:]
        model = model[:60] + [None] * 60
        options = {"max_human_skip_fraction": 0.6, "max_model_null_fraction": 0.6}
    elif case == "undefined":
        human, model, refs = [E] * 10, [E] * 10, _refs([E] * 5 + [C] * 5)
    result = _run(human, model, refs, **options)
    assert result.status == "refused"
    assert len(result.reasons) == 1
    assert (result.value, result.cohen_kappa, result.cohen_kappa_reason, result.interval) == (
        None,
        None,
        None,
        None,
    )
    payload = result.model_dump(mode="python")
    with pytest.raises(ModelValidationError):
        HumanModelAgreement.model_validate(
            {**payload, "value": {"exact": "1", "value": 1.0}, "interval": None}
        )


def test_exclusion_fractions_are_parameters_with_strict_comparison() -> None:
    assert MAX_HUMAN_SKIP_FRACTION == MAX_MODEL_NULL_FRACTION == 0.10
    human, model = _pairs(120, skips=13, nulls=24)
    loose = _run(
        human, model, _balanced_refs(120), max_human_skip_fraction=0.2, max_model_null_fraction=0.2
    )
    assert (loose.status, loose.max_human_skip_fraction, loose.max_model_null_fraction) == (
        "pass",
        0.2,
        0.2,
    )
    clean, _ = _pairs(20)
    one_null = list(clean)
    one_null[5] = None
    assert _run(clean, clean, _balanced_refs(20), max_model_null_fraction=0).status == "pass"
    strict = _run(clean, one_null, _balanced_refs(20), max_model_null_fraction=0)
    assert strict.reasons == ("model_null_fraction_exceeded",)


@pytest.mark.parametrize("fraction", [-0.1, 1, 1.0, 1.1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_exclusion_fractions_refuse(fraction: object) -> None:
    human, model = _pairs(20)
    for name in ("max_human_skip_fraction", "max_model_null_fraction"):
        with pytest.raises(ValidationError, match="agreement_exclusion_limit_invalid"):
            human_model_agreement(
                _human(human),
                _model(model),
                _balanced_refs(20),
                labels=LABELS,
                seed=SEED,
                **{name: fraction},  # type: ignore[arg-type]
            )


def test_unknown_model_labels_are_refused_not_counted_as_nulls() -> None:
    with pytest.raises(ValidationError, match="agreement_input_invalid"):
        _run([E, C, E], [E, C, "Entail"], _refs([E, C, E]))


# Coefficient selection on the reference label over all slice items


@pytest.mark.parametrize(("entail", "selected"), [(16, "gwet_ac1"), (15, "cohen_kappa")])
def test_inclusive_four_fifths_reference_prevalence_selects_ac1_and_reports_kappa(
    entail: int, selected: str
) -> None:
    refs = _refs([E] * entail + [C] * (20 - entail))
    human = [E] * 14 + [C] * 6
    model = [E] * 13 + [C] * 7
    result = _run(human, model, refs)
    assert result.coefficient == selected
    report = pairwise_agreement(
        HumanRatings(rater_id="a", rater_kind="human", item_ids=_ids(20), ratings=tuple(human)),
        HumanRatings(rater_id="b", rater_kind="human", item_ids=_ids(20), ratings=tuple(model)),
        labels=LABELS,
    )
    assert result.value is not None
    if selected == "gwet_ac1":
        assert result.value.value == report.gwet_ac1.value
        assert result.cohen_kappa is not None
        assert result.cohen_kappa.value == report.cohen_kappa.value
    else:
        assert result.value.value == report.cohen_kappa.value
        assert result.cohen_kappa is None


def test_prevalence_counts_excluded_items_and_selection_stays_fixed() -> None:
    refs = _refs([E] * 16 + [C] * 4)
    human: list[str | None] = [*([E] * 14), *([C] * 6)]
    model: list[str | None] = [*([E] * 13), *([C] * 7)]
    human[0] = None
    model[1] = None
    result = _run(human, model, refs)
    assert result.used == 18
    assert 5 * 14 < 4 * 18, "used-only prevalence would select kappa"
    assert result.coefficient == "gwet_ac1"
    assert result.interval is not None
    assert result.interval.coefficient_selection == "full_sample_reference"


def test_reference_label_prefers_solver_gold_then_adjudication() -> None:
    gold: list[str | None] = [None] * 16 + [C] * 4
    adjudicated: list[str | None] = [E for _ in range(20)]
    result = _run([E] * 14 + [C] * 6, [E] * 13 + [C] * 7, _refs(gold, adjudicated))
    assert result.coefficient == "gwet_ac1"
    gold_wins = _run([E] * 14 + [C] * 6, [E] * 13 + [C] * 7, _refs([C] * 5 + [None] * 15, [E] * 20))
    assert gold_wins.coefficient == "cohen_kappa"


def test_any_item_without_a_reference_label_refuses_the_slice() -> None:
    human, model = _pairs(20)
    gold: list[str | None] = [(E, C, U)[i % 3] for i in range(20)]
    gold[7] = None
    result = _run(human, model, _refs(gold))
    assert (result.status, result.reasons, result.coefficient, result.value) == (
        "refused",
        ("reference_unavailable",),
        None,
        None,
    )
    with pytest.raises(ValidationError, match="agreement_reference_invalid"):
        _run(human, model, _refs([*gold[:7], "bogus", *gold[8:]]))
    with pytest.raises(ValidationError, match="agreement_reference_invalid"):
        _run(human, model, tuple(reversed(_refs(gold))))
    with pytest.raises(ValidationError, match="agreement_reference_invalid"):
        _run(human, model, _refs(gold[:19]))


# Gate


def test_threshold_is_inclusive_at_exactly_seven_tenths() -> None:
    human, model = _exact_kappa_7_10()
    refs = _refs([E] * 20 + [C] * 20)
    at = _run(human, model, refs)
    assert at.coefficient == "cohen_kappa"
    assert at.value is not None
    assert (at.value.exact, at.value.value) == ("7/10", 0.7)
    assert (at.status, at.reasons, at.threshold) == ("pass", (), 0.7)
    above = _run(human, model, refs, threshold=0.7000000001)
    assert (above.status, above.reasons) == ("fail", ("below_threshold",))
    assert above.interval is not None


def test_undefined_selected_coefficient_refuses() -> None:
    result = _run([E] * 10, [E] * 10, _refs([E] * 5 + [C] * 5))
    assert (result.status, result.reasons, result.coefficient) == (
        "refused",
        ("coefficient_undefined",),
        "cohen_kappa",
    )
    assert (result.value, result.interval) == (None, None)


def test_ac1_beside_undefined_kappa_reports_the_kappa_reason() -> None:
    result = _run([E] * 10, [E] * 10, _refs([E] * 10))
    assert result.coefficient == "gwet_ac1"
    assert result.status == "pass"
    assert (result.cohen_kappa, result.cohen_kappa_reason) == (None, "chance_agreement_one")


def test_empty_and_fully_excluded_slices_refuse() -> None:
    empty = _run([], [], ())
    assert (empty.status, empty.reasons, empty.items) == ("refused", ("empty",), 0)
    excluded = _run(
        [None, None, E, C],
        [E, C, None, None],
        _refs([E, C, E, C]),
        max_human_skip_fraction=0.6,
        max_model_null_fraction=0.6,
    )
    assert (excluded.status, excluded.reasons, excluded.used) == ("refused", ("no_used_items",), 0)


# Interval and report-only user-solver kappa


def test_interval_is_the_bootstrap_of_the_pair_after_exclusions() -> None:
    human, model = _pairs(60, skips=3, nulls=2)
    refs = _balanced_refs(60)
    result = _run(human, model, refs, resamples=500)
    kept = [i for i in range(60) if human[i] is not None and model[i] is not None]
    ids = tuple(str(i) for i in kept)
    expected = agreement_bootstrap(
        HumanRatings(
            rater_id="a",
            rater_kind="human",
            item_ids=ids,
            ratings=tuple(str(human[i]) for i in kept),
        ),
        HumanRatings(
            rater_id="b",
            rater_kind="human",
            item_ids=ids,
            ratings=tuple(str(model[i]) for i in kept),
        ),
        reference_prevalence(
            tuple(
                r.model_copy(update={"item_id": str(i)}) for i, r in enumerate(refs) if i in kept
            ),
            labels=LABELS,
            item_set_ref="set",
            reference_version="v",
        ),
        labels=LABELS,
        seed=SEED,
        resamples=500,
    )
    assert expected.coefficient == result.coefficient
    assert result.interval == expected.interval
    assert result.interval is not None
    assert (result.interval.seed, result.interval.resamples) == (SEED, 500)


def test_fixed_seed_is_deterministic() -> None:
    human, model = _pairs(120, skips=4, nulls=3)
    first = _run(human, model, _balanced_refs(120), resamples=10_000)
    second = _run(human, model, _balanced_refs(120), resamples=10_000)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert first.interval is not None
    assert first.interval.undefined_resamples == 0


def test_user_solver_kappa_uses_only_solver_gold_items_and_never_gates() -> None:
    human, model = _exact_kappa_7_10()
    gold: list[str | None] = [*human[:30], *([None] * 10)]
    adjudicated: list[str | None] = [None] * 30 + [E] * 5 + [C] * 5
    skipped: list[str | None] = list(human)
    skipped[0] = None
    result = _run(skipped, model, _refs(gold, adjudicated))
    report = result.user_solver_kappa
    assert report.role == "report_only"
    assert report.n == 29
    assert report.value is not None
    assert Fraction(report.value.exact) == 1
    resolved = [g if g is not None else a for g, a in zip(gold, adjudicated, strict=True)]
    no_solver = _run(skipped, model, _refs([None] * 40, resolved))
    assert (no_solver.user_solver_kappa.n, no_solver.user_solver_kappa.reason) == (0, "empty")
    assert (no_solver.status, no_solver.reasons, no_solver.value, no_solver.interval) == (
        result.status,
        result.reasons,
        result.value,
        result.interval,
    )


def test_frozen_result_refuses_forged_decisions_and_counts() -> None:
    human, model = _exact_kappa_7_10()
    result = _run(human, model, _refs([E] * 20 + [C] * 20), resamples=50)
    payload = result.model_dump(mode="python")
    forgeries: list[dict[str, object]] = [
        {"threshold": 0.71},
        {"status": "fail", "reasons": ("below_threshold",)},
        {"status": "fail", "reasons": ("skip_fraction_exceeded",)},
        {"status": "refused", "reasons": ("model_null_fraction_exceeded",)},
        {"status": "refused", "reasons": ()},
        {"reasons": ("model_null_fraction_exceeded", "skip_fraction_exceeded")},
        {"used": 41},
        {"human_skipped": 1},
        {"value": {"exact": "7/10", "value": 0.71}},
        {"value": {"exact": "14/20", "value": 0.7}},
        {"cohen_kappa": {"exact": "7/10", "value": 0.7}},
        {"interval": None},
        {"coefficient": None},
        {"statistic_version": "nominal-two-human-raters-v1"},
        {"model_prompt_sha256": "x"},
    ]
    for change in forgeries:
        with pytest.raises(ModelValidationError):
            HumanModelAgreement.model_validate({**payload, **change})


# The signed skip failure outranks every refusal


def _skipped(human: Sequence[str | None], count: int = 13) -> list[str | None]:
    return [None] * count + list(human[count:])


@pytest.mark.parametrize(
    ("case", "reasons"),
    [
        ("reference_unavailable", ("skip_fraction_exceeded", "reference_unavailable")),
        ("model_null", ("skip_fraction_exceeded", "model_null_fraction_exceeded")),
        ("no_used_items", ("skip_fraction_exceeded", "no_used_items")),
        ("coefficient_undefined", ("skip_fraction_exceeded", "coefficient_undefined")),
        (
            "all",
            (
                "skip_fraction_exceeded",
                "reference_unavailable",
                "model_null_fraction_exceeded",
            ),
        ),
    ],
)
def test_exceeded_skips_fail_and_list_each_refusal_reason(
    case: str, reasons: tuple[str, ...]
) -> None:
    human, model = _pairs(120)
    gold: list[str | None] = [(E, C, U)[i % 3] for i in range(120)]
    options: dict[str, float] = {}
    human = _skipped(human)
    if case in {"reference_unavailable", "all"}:
        gold[50] = None
    if case in {"model_null", "all"}:
        model = model[:107] + [None] * 13
    if case == "no_used_items":
        human = [None] * 60 + human[60:]
        model = model[:60] + [None] * 60
        options = {"max_model_null_fraction": 0.9}
    if case == "coefficient_undefined":
        human = _skipped([E] * 120)
        model = [E] * 120
    result = _run(human, model, _refs(gold), **options)
    assert result.status == "fail"
    assert result.reasons == reasons
    defined = case == "model_null"
    assert (result.value is not None) == defined
    assert (result.interval is not None) == defined
    assert (result.coefficient is None) == (case in {"reference_unavailable", "all"})
    if not defined:
        assert (result.cohen_kappa, result.cohen_kappa_reason) == (None, None)


def test_empty_slices_are_refused_since_zero_items_cannot_exceed_skips() -> None:
    result = _run([], [], (), max_human_skip_fraction=0)
    assert (result.status, result.reasons) == ("refused", ("empty",))


# Policy bounds, reference provenance and validator gaps


@pytest.mark.parametrize("threshold", [-1.5, 1.0000001, 2])
def test_threshold_outside_the_coefficient_range_refuses(threshold: float) -> None:
    human, model = _pairs(20)
    with pytest.raises(ValidationError, match="agreement_threshold_invalid"):
        _run(human, model, _balanced_refs(20), threshold=threshold)


@pytest.mark.parametrize("threshold", [-1, 1])
def test_threshold_bounds_are_inclusive_and_recorded(threshold: float) -> None:
    human, model = _pairs(20)
    assert _run(human, model, _balanced_refs(20), threshold=threshold).threshold == threshold


@pytest.mark.parametrize("field", ["source_ref", "version_ref"])
def test_blank_reference_provenance_refuses(field: str) -> None:
    human, model = _pairs(4)
    refs = list(_balanced_refs(4))
    refs[2] = refs[2].model_copy(update={field: "  "})
    with pytest.raises(ValidationError, match="agreement_reference_invalid"):
        _run(human, model, refs)


def test_validator_refuses_uncounted_report_only_and_impossible_ac1_intervals() -> None:
    result = _run([E] * 14 + [C] * 6, [E] * 13 + [C] * 7, _refs([E] * 16 + [C] * 4))
    assert result.coefficient == "gwet_ac1"
    assert result.interval is not None
    payload = result.model_dump(mode="python")
    solver = {**payload["user_solver_kappa"], "n": 21}
    interval = {**payload["interval"], "undefined_resamples": 1}
    for forged in ({"user_solver_kappa": solver}, {"interval": interval}):
        with pytest.raises(ModelValidationError):
            HumanModelAgreement.model_validate({**payload, **forged})


def test_report_only_interval_cost_never_blocks_a_large_slice() -> None:
    human = [(E, C, U)[i % 3] for i in range(2001)]
    model = [label if i % 10 else (E, C, U)[(i + 1) % 3] for i, label in enumerate(human)]
    result = human_model_agreement(
        _human(human),
        _model(model),
        _refs(human),
        labels=LABELS,
        seed=SEED,
    )
    assert result.status == "pass"
    assert result.interval is not None
    assert result.interval.resamples == 10_000
