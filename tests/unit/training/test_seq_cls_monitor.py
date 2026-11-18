"""In-training eval metrics for seq_cls: monitoring only (no torch)."""

from __future__ import annotations

import pytest

from strata_forge.training.seq_cls_monitor import (
    MONITOR_LABEL,
    monitor_eval_steps,
    monitor_keys,
    monitor_metrics,
)

_LABELS = ["entailment", "contradiction", "neutral"]


def test_four_evals_an_epoch() -> None:
    assert monitor_eval_steps(24, 4) == 2  # 6 steps an epoch
    assert monitor_eval_steps(1000, 32) == 8  # 32 steps an epoch
    assert monitor_eval_steps(3, 8) == 1


def test_three_and_two_way_rows_are_scored_apart() -> None:
    logits = [
        [3.0, 0.0, 0.0],  # entailment, gold entailment
        [0.0, 2.0, 0.0],  # contradiction, gold neutral
        [0.0, 0.0, 1.0],  # neutral, gold neutral
        [1.0, 0.0, 0.0],  # two-way: predicts entailed, gold entailed
        [0.0, 1.0, 0.0],  # two-way: predicts not_entailed, gold entailed
    ]
    target = [[1, 0, 0], [0, 0, 1], [0, 0.2, 0.8], [0, 0, 0], [0, 0, 0]]
    out = monitor_metrics(
        logits, target, [0, 0, 0, 1, 1], [0, 0, 0, 1, 1], labels=_LABELS, positive_index=0
    )
    assert out["monitor/3way/rows"] == 3
    assert out["monitor/3way/accuracy"] == pytest.approx(2 / 3)
    assert out["monitor/3way/f1/entailment"] == 1.0
    assert out["monitor/3way/f1/contradiction"] == 0.0
    assert out["monitor/3way/f1/neutral"] == pytest.approx(2 / 3)
    assert out["monitor/3way/macro_f1"] == pytest.approx((1 + 0 + 2 / 3) / 3)
    assert out["monitor/2way/rows"] == 2
    assert out["monitor/2way/accuracy"] == 0.5
    assert out["monitor/2way/f1_entailed"] == pytest.approx(2 / 3)
    assert set(out) <= set(monitor_keys(_LABELS))


def test_a_group_with_no_rows_is_absent() -> None:
    out = monitor_metrics(
        [[1.0, 0.0, 0.0]], [[1, 0, 0]], [0], [0], labels=_LABELS, positive_index=0
    )
    assert not any(k.startswith("monitor/2way/") for k in out)


def test_the_label_says_what_the_numbers_are_not() -> None:
    assert MONITOR_LABEL == "monitoring (ID dev, final checkpoint is the model)"
