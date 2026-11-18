"""The PRM contract: one declared result per step, held to its score kind, and the opt-in
reductions over a sample's scored steps."""

from __future__ import annotations

import pytest

from strata_forge.plugins.manifest import PluginManifestError
from strata_forge.plugins.step_scoring import (
    ContextSegment,
    Step,
    Unscored,
    aggregate,
    check_scores,
    steps_from_rows,
)

STEPS = [Step(item_id="a", step_index=i, context="c", step="s") for i in range(3)]
REASONS = frozenset({"unparseable", "over_token_cap"})


def test_a_probability_is_held_to_the_closed_unit_interval_and_never_clipped() -> None:
    assert check_scores(STEPS, [0.0, 1, 0.3], "probability", REASONS) == [
        (0.0, None),
        (1.0, None),
        (0.3, None),
    ]
    for bad in (1.0000001, -0.0001, 2):
        with pytest.raises(PluginManifestError, match="probability"):
            check_scores(STEPS, [0.1, bad, 0.2], "probability", REASONS)


@pytest.mark.parametrize("kind", ["logit", "unconstrained"])
def test_logits_and_unconstrained_scores_pass_through_unchanged(kind: str) -> None:
    assert check_scores(STEPS, [-12.5, 0, 7.25], kind, REASONS) == [  # type: ignore[arg-type]
        (-12.5, None),
        (0.0, None),
        (7.25, None),
    ]


@pytest.mark.parametrize("kind", ["probability", "logit", "unconstrained"])
def test_nan_and_infinity_are_refused_whatever_the_kind(kind: str) -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(PluginManifestError, match="non-finite"):
            check_scores(STEPS, [0.1, bad, 0.2], kind, REASONS)  # type: ignore[arg-type]


def test_an_unscored_step_needs_a_declared_reason() -> None:
    results = check_scores(STEPS, [0.5, Unscored("unparseable"), 0.1], "probability", REASONS)
    assert results[1] == (None, "unparseable")
    with pytest.raises(PluginManifestError, match="undeclared reason"):
        check_scores(STEPS, [0.5, Unscored("tired"), 0.1], "probability", REASONS)
    with pytest.raises(PluginManifestError, match="neither a number nor Unscored"):
        check_scores(STEPS, [0.5, None, 0.1], "probability", REASONS)


def test_exactly_one_result_per_step() -> None:
    for bad in ([0.1], [0.1, 0.2, 0.3, 0.4], [0.1, True, 0.2], "abc", None):
        with pytest.raises(PluginManifestError):
            check_scores(STEPS, bad, "unconstrained", REASONS)


def test_aggregation_reads_scored_steps_and_counts_the_rest() -> None:
    rows = [("a", 1, 0.5), ("a", 0, 0.9), ("b", 0, 0.2), ("a", 2, None), ("c", 0, None)]
    assert aggregate(rows, "min") == {"a": (0.5, 2, 1), "b": (0.2, 1, 0), "c": (None, 0, 1)}
    assert aggregate(rows, "last")["a"] == (0.5, 2, 1)
    assert aggregate(rows, "mean")["a"][0] == pytest.approx(0.7)
    assert aggregate(rows, "product")["a"][0] == pytest.approx(0.45)
    with pytest.raises(PluginManifestError, match="twice"):
        aggregate([("a", 0, 0.1), ("a", 0, 0.2)], "min")


def test_rows_are_validated_into_steps() -> None:
    rows = [{"item_id": "a", "step_index": 0, "context": "c", "step": "s", "step_type": "claim"}]
    assert steps_from_rows(rows) == [
        Step(item_id="a", step_index=0, context="c", step="s", step_type="claim")
    ]
    for bad in (
        {"item_id": 1, "step_index": 0, "context": "c", "step": "s"},
        {"item_id": "a", "step_index": -1, "context": "c", "step": "s"},
        {"item_id": "a", "step_index": True, "context": "c", "step": "s"},
        {"item_id": "a", "step_index": 0, "context": None, "step": "s"},
        {"item_id": "a", "step_index": 0, "context": "c", "step": "s", "step_type": 3},
    ):
        with pytest.raises(PluginManifestError):
            steps_from_rows([bad])


def test_a_structured_context_reaches_the_scorer_as_its_segments_never_joined() -> None:
    segments = [{"pid": "p1", "text": "All men are mortal."}, {"pid": "p2", "text": "Socrates."}]
    (step,) = steps_from_rows([{"item_id": "a", "step_index": 0, "context": segments, "step": "s"}])
    assert step.context == (
        ContextSegment(pid="p1", text="All men are mortal."),
        ContextSegment(pid="p2", text="Socrates."),
    )
    for bad in (
        [{"pid": "p1"}],
        [{"pid": "p1", "text": "x", "extra": "y"}],
        [{"pid": 1, "text": "x"}],
        ["p1: x"],
    ):
        with pytest.raises(PluginManifestError):
            steps_from_rows([{"item_id": "a", "step_index": 0, "context": bad, "step": "s"}])
