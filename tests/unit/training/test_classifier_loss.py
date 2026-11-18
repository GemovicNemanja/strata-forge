"""The classifier objective: exact values, the two-way grouping, masks and gradients (CPU, torch)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from strata_forge.training.classifier_loss import LossWeights, classifier_loss

torch: Any = pytest.importorskip("torch")

_PRESET = LossWeights(brier_weight=0.1, flag_weight=1.0, support_weight=1.0)


def _loss(logits: Any, **kwargs: Any) -> Any:
    batch = logits.shape[0]
    defaults: dict[str, Any] = {
        "target": torch.zeros(batch, logits.shape[1]),
        "two_way": torch.zeros(batch, dtype=torch.bool),
        "two_way_target": torch.zeros(batch),
        "positive_index": 0,
        "weights": _PRESET,
    }
    return classifier_loss(logits, **{**defaults, **kwargs})


class TestFabricatedExample:
    """The worked example of the objective's definition, reproduced to 1e-6.

    Logits (2.0, 0.5, 0.0): a three-way row whose label is the first, and a two-way row whose
    target is "not the first". The flag and support terms use their own made-up logits.
    """

    def test_primary_term(self) -> None:
        logits = torch.tensor([[2.0, 0.5, 0.0], [2.0, 0.5, 0.0]], dtype=torch.float64)
        terms = _loss(
            logits,
            target=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64),
            two_way=torch.tensor([False, True]),
            two_way_target=torch.tensor([0.0, 0.0], dtype=torch.float64),
        )
        assert float(terms.primary) == pytest.approx(0.878832, abs=1e-6)

    def test_each_row(self) -> None:
        z = torch.tensor([[2.0, 0.5, 0.0]], dtype=torch.float64)
        full = _loss(z, target=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))
        two = _loss(
            z, two_way=torch.tensor([True]), two_way_target=torch.tensor([0.0], dtype=torch.float64)
        )
        assert float(full.primary) == pytest.approx(0.306356 + 0.1 * 0.106534, abs=1e-6)
        assert float(two.primary) == pytest.approx(1.332279 + 0.1 * 1.083759, abs=1e-6)

    def test_flag_term_masks_the_null_row(self) -> None:
        terms = _loss(
            torch.zeros(3, 3),
            flag_logits=torch.tensor([1.0, -0.5, 9.0]),
            flag_target=torch.tensor([1.0, 0.0, -1.0]),
        )
        assert float(terms.flag) == pytest.approx(0.393669, abs=1e-6)

    def test_support_term_averages_per_row_then_over_eligible_rows(self) -> None:
        terms = _loss(
            torch.zeros(3, 3),
            support_logits=torch.tensor([[2.0, -1.0, 0.0], [0.5, -2.0, 0.0], [5.0, 5.0, 5.0]]),
            support_target=torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            support_valid=torch.tensor([[True, True, True], [True, True, False], [True] * 3]),
            support_row=torch.tensor([True, True, False]),
        )
        assert float(terms.support) == pytest.approx(0.464141, abs=1e-6)

    def test_total_adds_the_weighted_terms(self) -> None:
        weights = LossWeights(brier_weight=0.1, flag_weight=2.0, support_weight=0.5)
        terms = _loss(
            torch.tensor([[2.0, 0.5, 0.0]]),
            target=torch.tensor([[1.0, 0.0, 0.0]]),
            weights=weights,
            flag_logits=torch.tensor([1.0]),
            flag_target=torch.tensor([1.0]),
        )
        expected = float(terms.primary) + 2.0 * float(terms.flag)
        assert float(terms.total) == pytest.approx(expected, abs=1e-6)


class TestClosedForm:
    def test_without_brier_it_is_cross_entropy_with_the_softmax_gradient(self) -> None:
        logits = torch.tensor([[0.3, -1.2, 2.0], [1.0, 1.0, -0.5]], requires_grad=True)
        target = torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.8, 0.0]])
        terms = _loss(logits, target=target, weights=LossWeights())
        expected = torch.nn.functional.cross_entropy(logits.detach(), target)
        assert float(terms.total.detach()) == pytest.approx(float(expected), abs=1e-6)
        terms.total.backward()
        softmax = logits.detach().softmax(-1)
        assert torch.allclose(logits.grad, (softmax - target) / 2, atol=1e-6)

    def test_the_loss_is_float32_under_a_bf16_forward(self) -> None:
        logits = torch.tensor([[0.3, -1.2, 2.0]], dtype=torch.bfloat16)
        terms = _loss(logits, target=torch.tensor([[0.0, 0.0, 1.0]]))
        assert terms.total.dtype == torch.float32


class TestTwoWayRows:
    def test_the_negative_labels_are_never_trained_against_each_other(self) -> None:
        # Moving mass between the two negative labels at a fixed LSE(z_neg) leaves the loss alone.
        def loss_at(c: float, u: float) -> float:
            return float(
                _loss(
                    torch.tensor([[1.0, c, u]], dtype=torch.float64),
                    two_way=torch.tensor([True]),
                    two_way_target=torch.tensor([0.0], dtype=torch.float64),
                ).total
            )

        lse = math.log(math.exp(0.4) + math.exp(-0.7))
        shifted_c = 0.0
        shifted_u = math.log(math.exp(lse) - math.exp(shifted_c))
        assert loss_at(0.4, -0.7) == pytest.approx(loss_at(shifted_c, shifted_u), abs=1e-12)

    def test_the_negative_gradient_follows_the_negative_split(self) -> None:
        logits = torch.tensor([[0.2, 0.9, -0.4]], dtype=torch.float64, requires_grad=True)
        _loss(
            logits,
            two_way=torch.tensor([True]),
            two_way_target=torch.tensor([1.0], dtype=torch.float64),
        ).total.backward()
        p = logits.detach().softmax(-1)[0]
        grad = logits.grad[0]
        # The gradient on each negative label is its share of the negative mass, so the split
        # between them is untouched by the update.
        assert float(grad[1] / grad[2]) == pytest.approx(float(p[1] / p[2]), rel=1e-9)

    def test_the_positive_label_is_configurable(self) -> None:
        z = torch.tensor([[0.0, 3.0, 0.0]], dtype=torch.float64)
        terms = _loss(
            z,
            two_way=torch.tensor([True]),
            two_way_target=torch.tensor([1.0], dtype=torch.float64),
            positive_index=1,
            weights=LossWeights(),
        )
        assert float(terms.total) == pytest.approx(-math.log(float(z.softmax(-1)[0, 1])), abs=1e-9)

    def test_numeric_gradient(self) -> None:
        def fn(z: Any) -> Any:
            return _loss(
                z,
                target=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64),
                two_way=torch.tensor([False, True]),
                two_way_target=torch.tensor([0.0, 1.0], dtype=torch.float64),
                flag_logits=z[:, 0] * 0.5,
                flag_target=torch.tensor([1.0, -1.0], dtype=torch.float64),
            ).total

        z = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(fn, (z,))


class TestMasks:
    def test_no_known_flag_contributes_exactly_zero(self) -> None:
        terms = _loss(
            torch.zeros(2, 3),
            flag_logits=torch.tensor([3.0, -3.0]),
            flag_target=torch.tensor([-1.0, -1.0]),
        )
        assert float(terms.flag) == 0.0

    def test_no_eligible_support_row_contributes_exactly_zero(self) -> None:
        terms = _loss(
            torch.zeros(2, 3),
            support_logits=torch.ones(2, 2),
            support_target=torch.ones(2, 2),
            support_valid=torch.ones(2, 2, dtype=torch.bool),
            support_row=torch.tensor([False, False]),
        )
        assert float(terms.support) == 0.0
        assert torch.isfinite(terms.total)

    def test_unmapped_heads_add_nothing(self) -> None:
        terms = _loss(torch.tensor([[2.0, 0.5, 0.0]]), target=torch.tensor([[1.0, 0.0, 0.0]]))
        assert float(terms.total) == pytest.approx(float(terms.primary))
