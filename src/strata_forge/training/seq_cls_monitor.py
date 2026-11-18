"""In-training eval metrics for ``seq_cls``: MONITORING ONLY.

The final checkpoint is the model. These numbers watch a run on an in-distribution dev split
(another split of the training dataset, at the same commit); they never choose a checkpoint, stop a
run or stand for the model's score, and every place they appear carries :data:`MONITOR_LABEL`.

Each eval reads the classifier's argmax, uncalibrated, over rows rendered by the same encoder the
trainer uses (:mod:`strata_forge.training.verifier_render`), and scores them with forge's own
metrics (:func:`strata_forge.metrics.classification.classification_report`). Metric keys, stable:

- ``monitor/3way/rows``, ``monitor/3way/accuracy``, ``monitor/3way/macro_f1`` and
  ``monitor/3way/f1/<label>`` per declared label, over the three-way rows (gold is the label, or the
  largest share of a ``label_dist``);
- ``monitor/2way/rows``, ``monitor/2way/accuracy`` and ``monitor/2way/f1_entailed`` over the
  two-way rows (predicted positive when the argmax is the declared positive label);
- ``eval_loss``, the training objective on the split, as the trainer reports it.

A group with no row in the split is absent. Cadence: every ``ceil(steps_per_epoch / 4)`` steps
(:func:`monitor_eval_steps`) and once more on the final checkpoint.

Pure: no torch, so it is tested without the training extra.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from strata_forge.metrics.classification import classification_report

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "MONITOR_LABEL",
    "MONITOR_PREFIX",
    "monitor_eval_steps",
    "monitor_keys",
    "monitor_metrics",
]

#: How every in-training eval number is labelled, wherever it appears.
MONITOR_LABEL = "monitoring (ID dev, final checkpoint is the model)"
#: The prefix of every monitoring metric key.
MONITOR_PREFIX = "monitor/"
_ENTAILED = "entailed"
_NOT_ENTAILED = "not_entailed"


def monitor_eval_steps(train_rows: int, batch_size: int) -> int:
    """Evaluate four times an epoch: every ``ceil(steps_per_epoch / 4)`` optimizer steps."""
    steps_per_epoch = max(1, math.ceil(train_rows / max(1, batch_size)))
    return max(1, math.ceil(steps_per_epoch / 4))


def monitor_keys(labels: Sequence[str]) -> list[str]:
    """Every key :func:`monitor_metrics` can report for ``labels``, in a stable order."""
    return [
        f"{MONITOR_PREFIX}3way/rows",
        f"{MONITOR_PREFIX}3way/accuracy",
        f"{MONITOR_PREFIX}3way/macro_f1",
        *(f"{MONITOR_PREFIX}3way/f1/{label}" for label in labels),
        f"{MONITOR_PREFIX}2way/rows",
        f"{MONITOR_PREFIX}2way/accuracy",
        f"{MONITOR_PREFIX}2way/f1_entailed",
    ]


def _argmax(row: Sequence[float]) -> int:
    # The first maximum on a tie: the lowest class index, deterministically.
    return max(range(len(row)), key=row.__getitem__)


def monitor_metrics(
    logits: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    two_way: Sequence[Any],
    two_way_target: Sequence[float],
    *,
    labels: Sequence[str],
    positive_index: int,
) -> dict[str, float]:
    """The monitoring metrics for one eval pass, keyed as :func:`monitor_keys` lists them."""
    out: dict[str, float] = {}
    three_gold: list[str] = []
    three_pred: list[str] = []
    two_gold: list[str] = []
    two_pred: list[str] = []
    for row, gold, is_two_way, positive in zip(
        logits, target, two_way, two_way_target, strict=True
    ):
        guess = _argmax(row)
        if bool(is_two_way):
            two_gold.append(_ENTAILED if float(positive) >= 0.5 else _NOT_ENTAILED)
            two_pred.append(_ENTAILED if guess == positive_index else _NOT_ENTAILED)
        else:
            three_gold.append(labels[_argmax(gold)])
            three_pred.append(labels[guess])
    if three_gold:
        report = classification_report(three_gold, three_pred, labels=list(labels))
        out[f"{MONITOR_PREFIX}3way/rows"] = float(len(three_gold))
        out[f"{MONITOR_PREFIX}3way/accuracy"] = float(report.accuracy.value or 0.0)
        out[f"{MONITOR_PREFIX}3way/macro_f1"] = float(report.macro_f1.value or 0.0)
        for score in report.per_class:
            out[f"{MONITOR_PREFIX}3way/f1/{score.label}"] = float(score.f1)
    if two_gold:
        report = classification_report(two_gold, two_pred, labels=[_ENTAILED, _NOT_ENTAILED])
        out[f"{MONITOR_PREFIX}2way/rows"] = float(len(two_gold))
        out[f"{MONITOR_PREFIX}2way/accuracy"] = float(report.accuracy.value or 0.0)
        out[f"{MONITOR_PREFIX}2way/f1_entailed"] = float(report.per_class[0].f1)
    return out
