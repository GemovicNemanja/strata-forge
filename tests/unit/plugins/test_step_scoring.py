"""The PRM contract: one finite score per step, and the common reductions over a sample."""

from __future__ import annotations

import pytest

from strata_forge.plugins.manifest import PluginManifestError
from strata_forge.plugins.step_scoring import Step, aggregate, check_scores, steps_from_rows

STEPS = [Step(item_id="a", step_index=i, context="c", step="s") for i in range(3)]


def test_scores_must_be_one_finite_number_per_step() -> None:
    assert check_scores(STEPS, [0.1, 1, 0.3]) == [0.1, 1.0, 0.3]
    for bad in ([0.1], [0.1, 0.2, float("inf")], [0.1, True, 0.2], "abc", None):
        with pytest.raises(PluginManifestError):
            check_scores(STEPS, bad)


def test_aggregation_orders_steps_and_refuses_repeats() -> None:
    rows = [("a", 1, 0.5), ("a", 0, 0.9), ("b", 0, 0.2), ("a", 2, 0.7)]
    assert aggregate(rows, "min") == {"a": 0.5, "b": 0.2}
    assert aggregate(rows, "last") == {"a": 0.7, "b": 0.2}
    assert aggregate(rows, "mean")["a"] == pytest.approx(0.7)
    assert aggregate(rows, "product")["a"] == pytest.approx(0.9 * 0.5 * 0.7)
    with pytest.raises(PluginManifestError, match="twice"):
        aggregate([("a", 0, 0.1), ("a", 0, 0.2)], "min")


def test_rows_are_validated_into_steps() -> None:
    rows = [{"item_id": "a", "step_index": 0, "context": "c", "step": "s"}]
    assert steps_from_rows(rows) == [Step(item_id="a", step_index=0, context="c", step="s")]
    for bad in (
        {"item_id": 1, "step_index": 0, "context": "c", "step": "s"},
        {"item_id": "a", "step_index": -1, "context": "c", "step": "s"},
        {"item_id": "a", "step_index": True, "context": "c", "step": "s"},
        {"item_id": "a", "step_index": 0, "context": None, "step": "s"},
    ):
        with pytest.raises(PluginManifestError):
            steps_from_rows([bad])
