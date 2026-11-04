"""Analytic softmax cases and strict, portable calibration provenance."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError as ModelValidationError

from strata_forge.core.errors import ValidationError
from strata_forge.metrics import CalibrationScores, TemperatureCalibration, temperature_scale
from strata_forge.metrics import calibration as calibration_module

if TYPE_CHECKING:
    from collections.abc import Iterable


def test_analytic_temperature_values_and_offset_invariance() -> None:
    one = temperature_scale([[0.0, math.log(3)]], temperature=1.0)
    two = temperature_scale([[0.0, math.log(3)]], temperature=2.0)
    assert one[0] == pytest.approx((0.25, 0.75))
    assert two[0] == pytest.approx((1 / (1 + math.sqrt(3)), math.sqrt(3) / (1 + math.sqrt(3))))
    assert temperature_scale([[100.0, 100.0 + math.log(3)]], temperature=1.0)[0] == pytest.approx(
        one[0]
    )


def test_extreme_finite_logits_stay_normalized() -> None:
    result = temperature_scale([[1e308, -1e308], [-1e308, -1e308]], temperature=1e-300)
    assert result == ((1.0, 0.0), (0.5, 0.5))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan"), True])
def test_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValidationError, match="temperature"):
        temperature_scale([], temperature=temperature)


@pytest.mark.parametrize(
    "logits",
    [
        [[0.0]],
        [[0.0, 1.0], [0.0, 1.0, 2.0]],
        [[float("inf"), 1.0]],
        [[True, 1.0]],
        [["0", 1.0]],
        ["01"],
        [None],
    ],
)
def test_malformed_logits(logits: Iterable[object]) -> None:
    with pytest.raises(ValidationError):
        temperature_scale(logits, temperature=1.0)


def test_missing_numpy_has_an_install_hint_and_empty_input_needs_no_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def absent(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(calibration_module, "import_module", absent)
    assert temperature_scale([], temperature=1.0) == ()
    with pytest.raises(ImportError, match=r"strata-forge\[scoring\]"):
        temperature_scale([[0.0, 1.0]], temperature=1.0)


def test_numpy_scalars_and_arrays_are_accepted_without_index_order_guessing() -> None:
    import numpy as np

    from strata_forge.metrics import accuracy, brier

    labels = ("yes", "no")
    assert (
        accuracy(
            np.array([3, 5]), np.array([3, 5]), labels=labels, index_map={3: "yes", 5: "no"}
        ).value
        == 1
    )
    assert (
        brier(
            np.array([3]),
            np.array([[0.75, 0.25]]),
            labels=labels,
            index_map={3: "yes"},
            probability_kind="class_probabilities",
        ).value
        == 0.125
    )
    assert temperature_scale(np.array([[0.0, math.log(3)]]), temperature=1.0)[0] == pytest.approx(
        (0.25, 0.75)
    )


@pytest.fixture
def component() -> TemperatureCalibration:
    return TemperatureCalibration(
        labels=("yes", "no"),
        temperature=2.0,
        fit_ref="owner/validation@" + "a" * 40,
        config="binary",
        split="validation",
        n=40,
        probability_kind="restricted_softmax",
        before=CalibrationScores(brier=0.4, nll=0.9),
        after=CalibrationScores(brier=0.3, nll=0.8),
    )


def test_component_round_trip_and_explicit_label_order(component: TemperatureCalibration) -> None:
    assert TemperatureCalibration.model_validate_json(component.model_dump_json()) == component
    expected = (1 / (1 + math.sqrt(3)), math.sqrt(3) / (1 + math.sqrt(3)))
    assert component.apply([[0.0, math.log(3)]], labels=("yes", "no"))[0] == pytest.approx(expected)
    with pytest.raises(ValidationError, match="label order"):
        component.apply([[0.0, 1.0]], labels=("no", "yes"))
    with pytest.raises(ValidationError, match="label count"):
        component.apply([[0.0, 1.0, 2.0]], labels=("yes", "no"))
    assert component.apply([], labels=("yes", "no")) == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.0),
        ("temperature", float("inf")),
        ("n", 0),
        ("probability_kind", "censored_topk"),
        ("labels", ("yes", "yes")),
        ("nll_clip_epsilon", 1.0),
        ("extra", 1),
    ],
)
def test_component_rejects_invalid_serialized_fields(
    component: TemperatureCalibration, field: str, value: object
) -> None:
    values = component.model_dump()
    values[field] = value
    with pytest.raises(ModelValidationError):
        TemperatureCalibration.model_validate(values)


def test_omitted_calibration_scores_are_null() -> None:
    scores = CalibrationScores()
    assert scores.brier is None
    assert scores.nll is None
    with pytest.raises(ModelValidationError):
        CalibrationScores(brier=float("nan"))


def test_evaluation_reexports_point_metrics_without_forking_definitions() -> None:
    from strata_forge.evals import classification as evaluation
    from strata_forge.metrics import classification as leaf

    assert evaluation.accuracy is leaf.accuracy
    assert evaluation.brier is leaf.brier
    assert evaluation.nll is leaf.nll


def test_metric_leaf_import_boundaries_and_lazy_numpy() -> None:
    root = Path(calibration_module.__file__).parent
    for source in root.glob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("strata_forge.")
            ):
                assert node.module.startswith(("strata_forge.core.", "strata_forge.metrics."))
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(alias.name != "numpy" for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "numpy"
