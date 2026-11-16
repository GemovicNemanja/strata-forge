"""Fabricated supplied-logit objectives and CPU gradients, with no model or data assets."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError as ModelValidationError

from strata_forge.training.verifier_loss import (
    OBJECTIVE_PRESET,
    EntailmentTarget,
    ResearchLossRow,
    ThreeWayTarget,
    VerifierLossError,
    research_verifier_loss,
)

ORDER = ("E", "C", "U")


def stack() -> Any:
    # The installed optional tensor API is the deliberate dynamic test boundary.
    try:
        import torch
    except ImportError:
        with pytest.raises(ImportError, match="finetuning"):
            research_verifier_loss(None, class_order=ORDER, rows=())
        return None
    return torch


def e_row(**kwargs: Any) -> ResearchLossRow:
    return ResearchLossRow(primary=ThreeWayTarget(label_dist=(1.0, 0.0, 0.0)), **kwargs)


def test_fabricated_primary_values_to_one_micro() -> None:
    torch = stack()
    if torch is None:
        return
    logits = torch.tensor([[2.0, 0.5, 0.0]], dtype=torch.float64)
    three = research_verifier_loss(logits, class_order=ORDER, rows=(e_row(),))
    two = research_verifier_loss(
        logits,
        class_order=ORDER,
        rows=(ResearchLossRow(primary=EntailmentTarget(entailment_target=0)),),
    )
    probabilities = logits.softmax(-1)[0]
    ce3 = -logits.log_softmax(-1)[0, 0]
    brier3 = ((probabilities - torch.tensor([1, 0, 0])) ** 2).sum()
    ce2 = -torch.logsumexp(logits.log_softmax(-1)[0, 1:], dim=0)
    brier2 = 2 * probabilities[0] ** 2
    for actual, expected in (
        (ce3, 0.306356),
        (brier3, 0.106534),
        (three.primary, 0.317009),
        (ce2, 1.332279),
        (brier2, 1.083759),
        (two.primary, 1.440655),
    ):
        assert math.isclose(float(actual), expected, abs_tol=1e-6, rel_tol=0)
    mixed = research_verifier_loss(
        logits.repeat(2, 1),
        class_order=ORDER,
        rows=(e_row(), ResearchLossRow(primary=EntailmentTarget(entailment_target=0))),
    )
    assert mixed.preset_id == OBJECTIVE_PRESET == "verifier_grouped_aux_v1"
    assert math.isclose(float(mixed.primary), 0.878832, abs_tol=1e-6, rel_tol=0)
    assert mixed.primary_rows == 2
    assert mixed.flag_rows == mixed.support_rows == 0


def test_fabricated_auxiliary_masks_and_per_row_support_means() -> None:
    torch = stack()
    if torch is None:
        return
    rows = (
        e_row(flag_target=True, premise_ids=("p1", "p2", "p3"), support_set=("p1", "p3")),
        e_row(flag_target=False, premise_ids=("p1", "p2"), support_set=()),
        e_row(premise_ids=("unlabeled",)),
    )
    logits = torch.tensor([[2.0, 0.5, 0.0]] * 3, requires_grad=True)
    flags = torch.tensor([1.0, -0.5, 90.0], requires_grad=True)
    supports = tuple(
        torch.tensor(v, requires_grad=True) for v in ([2.0, -1.0, 0.0], [0.5, -2.0], [90.0])
    )
    terms = research_verifier_loss(
        logits, class_order=ORDER, rows=rows, flag_logits=flags, support_logits=supports
    )
    assert (terms.primary_rows, terms.flag_rows, terms.support_rows) == (3, 2, 2)
    assert math.isclose(float(terms.flag.detach()), 0.393669, abs_tol=1e-6, rel_tol=0)
    a = research_verifier_loss(
        logits[:1],
        class_order=ORDER,
        rows=rows[:1],
        flag_logits=flags[:1],
        support_logits=supports[:1],
    )
    b = research_verifier_loss(
        logits[1:2],
        class_order=ORDER,
        rows=rows[1:2],
        flag_logits=flags[1:2],
        support_logits=supports[1:2],
    )
    assert math.isclose(float(a.support.detach()), 0.377779, abs_tol=1e-6, rel_tol=0)
    assert math.isclose(float(b.support.detach()), 0.550502, abs_tol=1e-6, rel_tol=0)
    assert math.isclose(float(terms.support.detach()), 0.464141, abs_tol=1e-6, rel_tol=0)
    flattened_wrong = (3 * a.support + 2 * b.support) / 5
    assert abs(float((terms.support - flattened_wrong).detach())) > 0.01
    assert torch.equal(terms.total, terms.primary + terms.flag + terms.support)
    terms.total.backward()
    assert float(flags.grad[2]) == 0.0
    assert torch.equal(supports[2].grad, torch.zeros(1))
    assert float(flags.grad[1]) > 0.0  # Explicit false is supervised, not masked.
    assert bool((supports[1].grad > 0).all())  # An empty set is eligible all-negative supervision.


def test_all_null_auxiliary_targets_produce_connected_zero_without_overflow() -> None:
    torch = stack()
    if torch is None:
        return
    logits = torch.tensor([[3e38, 3e38, 3e38]], requires_grad=True)
    flags = torch.tensor([3e38], requires_grad=True)
    support = torch.tensor([3e38, 3e38], requires_grad=True)
    terms = research_verifier_loss(
        logits,
        class_order=ORDER,
        rows=(e_row(premise_ids=("a", "b")),),
        flag_logits=flags,
        support_logits=(support,),
    )
    assert float(terms.flag.detach()) == float(terms.support.detach()) == 0.0
    assert terms.flag_rows == terms.support_rows == 0
    terms.total.backward()
    assert bool(torch.isfinite(logits.grad).all())
    assert torch.equal(flags.grad, torch.zeros(1))
    assert torch.equal(support.grad, torch.zeros(2))


@pytest.mark.parametrize("target", [0, 1])
def test_grouped_loss_does_not_supervise_c_against_u(target: int) -> None:
    torch = stack()
    if torch is None:
        return
    logits = torch.tensor(
        [[math.log(v) for v in p] for p in ((0.25, 0.70, 0.05), (0.25, 0.10, 0.65))],
        dtype=torch.float64,
        requires_grad=True,
    )
    row = ResearchLossRow(primary=EntailmentTarget(entailment_target=target))
    a = research_verifier_loss(logits[:1], class_order=ORDER, rows=(row,))
    b = research_verifier_loss(logits[1:], class_order=ORDER, rows=(row,))
    assert math.isclose(float((a.total - b.total).detach()), 0, abs_tol=1e-12, rel_tol=0)
    (a.total + b.total).backward()
    assert bool(torch.isfinite(logits.grad).all())


def test_explicit_class_order_and_soft_distribution_are_preserved() -> None:
    torch = stack()
    if torch is None:
        return
    rows = (
        ResearchLossRow(primary=ThreeWayTarget(label_dist=(0.2, 0.3, 0.5))),
        ResearchLossRow(primary=EntailmentTarget(entailment_target=0)),
    )
    logits = torch.tensor([[2.0, 0.5, 0.0], [-1.0, 3.0, 2.0]], dtype=torch.float64)
    original = research_verifier_loss(logits, class_order=ORDER, rows=rows)
    permuted = research_verifier_loss(
        logits[:, [2, 0, 1]],
        class_order=("U", "E", "C"),
        rows=(ResearchLossRow(primary=ThreeWayTarget(label_dist=(0.5, 0.2, 0.3))), rows[1]),
    )
    assert math.isclose(float(original.total), float(permuted.total), abs_tol=1e-12, rel_tol=0)


@pytest.mark.parametrize("target", [0, 1])
def test_stable_extreme_logits_and_gradients(target: int) -> None:
    torch = stack()
    if torch is None:
        return
    logits = torch.tensor(
        [[10000.0, -10000.0, -10001.0], [-10000.0, 10000.0, 10001.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    row = ResearchLossRow(primary=EntailmentTarget(entailment_target=target))
    terms = research_verifier_loss(logits, class_order=ORDER, rows=(row, row))
    assert bool(torch.isfinite(terms.total))
    terms.total.backward()
    assert bool(torch.isfinite(logits.grad).all())


def test_complete_mixed_batch_autograd_matches_numeric_gradients() -> None:
    torch = stack()
    if torch is None:
        return
    rows = (
        ResearchLossRow(
            primary=ThreeWayTarget(label_dist=(0.2, 0.3, 0.5)),
            flag_target=True,
            premise_ids=("a", "b"),
            support_set=("b",),
        ),
        ResearchLossRow(
            primary=EntailmentTarget(entailment_target=0),
            premise_ids=("a", "b", "c"),
            support_set=(),
        ),
        ResearchLossRow(primary=EntailmentTarget(entailment_target=1), flag_target=False),
    )
    inputs = (
        torch.tensor(
            [[2.0, 0.5, 0.0], [0.1, -0.5, 0.6], [0.7, 0.0, -0.2]],
            dtype=torch.float64,
            requires_grad=True,
        ),
        torch.tensor([1.0, -0.5, 0.1], dtype=torch.float64, requires_grad=True),
        torch.tensor([2.0, -1.0], dtype=torch.float64, requires_grad=True),
        torch.tensor([0.5, -2.0, 0.1], dtype=torch.float64, requires_grad=True),
        torch.empty(0, dtype=torch.float64, requires_grad=True),
    )

    def objective(primary: Any, flags: Any, a: Any, b: Any, c: Any) -> Any:
        return research_verifier_loss(
            primary, class_order=ORDER, rows=rows, flag_logits=flags, support_logits=(a, b, c)
        ).total

    assert torch.autograd.gradcheck(objective, inputs, eps=1e-6, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_low_precision_inputs_accumulate_in_float32(dtype: str) -> None:
    torch = stack()
    if torch is None:
        return
    logits = torch.tensor([[2.0, 0.5, 0.0]], dtype=getattr(torch, dtype), requires_grad=True)
    terms = research_verifier_loss(logits, class_order=ORDER, rows=(e_row(),))
    assert terms.total.dtype == torch.float32
    assert math.isclose(float(terms.total.detach()), 0.317009, abs_tol=1e-6, rel_tol=0)
    terms.total.backward()
    assert bool(torch.isfinite(logits.grad).all())


@pytest.mark.parametrize(
    "fields",
    [
        {"premise_ids": ("p", "p")},
        {"premise_ids": ("p",), "support_set": ("other",)},
        {"premise_ids": ("p",), "support_set": ("p", "p")},
        {"premise_ids": (), "support_set": ()},
        {"flag_target": 0},
    ],
)
def test_invalid_resolved_masks_and_ids_refuse(fields: dict[str, Any]) -> None:
    with pytest.raises(ModelValidationError):
        e_row(**fields)


@pytest.mark.parametrize(
    "distribution", [(1.1, -0.1, 0.0), (0.0, 0.0, 0.0), (math.nan, 0.0, 1.0), (math.inf, 0.0, 0.0)]
)
def test_invalid_distributions_are_not_repaired(distribution: tuple[float, ...]) -> None:
    with pytest.raises(ModelValidationError):
        ThreeWayTarget.model_validate({"label_dist": distribution})


@pytest.mark.parametrize("value", [True, False, "1", 0.5, -1, 2])
def test_binary_target_is_strict_zero_or_one(value: object) -> None:
    with pytest.raises(ModelValidationError):
        EntailmentTarget.model_validate({"entailment_target": value})


@pytest.mark.parametrize(
    "case",
    [
        "flag_missing",
        "support_missing",
        "primary_shape",
        "flag_shape",
        "support_shape",
        "integer",
        "nonfinite",
        "empty",
        "order",
        "tampered",
    ],
)
def test_invalid_tensor_or_tampered_target_boundary_has_fixed_error(case: str) -> None:
    torch = stack()
    if torch is None:
        return
    kwargs: dict[str, Any] = {"class_order": ORDER, "rows": (e_row(),)}
    logits = torch.zeros((1, 3))
    if case == "flag_missing":
        kwargs["rows"] = (e_row(flag_target=True),)
    if case == "support_missing":
        kwargs["rows"] = (e_row(premise_ids=("p",), support_set=()),)
    if case == "primary_shape":
        logits = torch.zeros((1, 2))
    if case == "flag_shape":
        kwargs["flag_logits"] = torch.zeros(2)
    if case == "support_shape":
        kwargs["support_logits"] = (torch.zeros(2),)
    if case == "integer":
        logits = torch.zeros((1, 3), dtype=torch.int64)
    if case == "nonfinite":
        logits[0, 0] = math.nan
    if case == "empty":
        kwargs["rows"] = ()
    if case == "order":
        kwargs["class_order"] = ("E", "E", "U")
    if case == "tampered":
        kwargs["rows"] = (e_row().model_copy(update={"flag_target": "FICTIONAL_PRIVATE_TARGET"}),)
    with pytest.raises(VerifierLossError) as captured:
        research_verifier_loss(logits, **kwargs)
    assert str(captured.value).startswith("verifier_loss_")
    assert "FICTIONAL_PRIVATE_TARGET" not in str(captured.value)
    assert captured.value.__cause__ is captured.value.__context__ is None


def test_no_weight_smoothing_or_free_coefficient_arguments() -> None:
    for field in (
        "sample_weight",
        "class_weights",
        "label_smoothing",
        "brier_weight",
        "flag_weight",
        "support_weight",
    ):
        kwargs: dict[str, Any] = {field: 0.5}
        with pytest.raises(TypeError, match="unexpected keyword"):
            research_verifier_loss(None, class_order=ORDER, rows=(), **kwargs)


def test_module_import_does_not_import_ml_or_change_generic_defaults() -> None:
    source = Path(__file__).resolve().parents[3] / "src"
    script = f"""
import builtins,sys
sys.path.insert(0,{json.dumps(str(source))})
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {{'torch','transformers','trl','peft','datasets'}}:
        raise AssertionError('unexpected ML import')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from strata_forge.training.verifier_loss import OBJECTIVE_PRESET,ResearchLossRow
from strata_forge.training.classifier import ClassifierConfig
cfg=ClassifierConfig(labels=('E','C','U'))
assert cfg.objective=='cross_entropy' and cfg.brier_weight==0
assert cfg.label_smoothing==0 and cfg.class_weights is None
assert OBJECTIVE_PRESET=='verifier_grouped_aux_v1'
assert not any(name in sys.modules for name in ('torch','transformers','trl','peft','datasets'))
"""
    result = subprocess.run(  # noqa: S603 — fixed interpreter and source-owned import guard.
        [sys.executable, "-I", "-c", script],
        env={"PATH": os.defpath, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
