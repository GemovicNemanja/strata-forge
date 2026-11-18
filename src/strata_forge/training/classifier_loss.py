"""The sequence-classification objective: grouped soft cross-entropy, Brier and masked aux terms.

One batch-level function, :func:`classifier_loss`, vectorised over the batch (no per-row Python),
so it costs the same at 600k rows as at 600. It defines the objective exactly; the denominators are
part of the contract, not an implementation detail:

**Primary head.** The model emits ``K`` logits ``z`` and ``p = softmax(z)`` at temperature 1.

- A *full* row has a target distribution ``y`` over the ``K`` labels (one-hot from the label, or a
  declared soft ``label_dist``): ``CE = -sum_k y_k log p_k`` and ``Brier = sum_k (p_k - y_k)^2``
  (the sum form, on ``[0, 2]``).
- A *two-way* row says only whether the positive label holds (``t = 1``) or not (``t = 0``). The
  negative class is the AGGREGATE of every other label, because the source does not say which one
  applies: with ``q = p_pos``, ``log q = z_pos - LSE(z)`` and ``log(1 - q) = LSE(z_neg) - LSE(z)``
  (both computed stably), ``CE = -[t log q + (1 - t) log(1 - q)]`` and ``Brier = 2 (q - t)^2``.
  Nothing in it depends on how the negative labels split their mass, so a two-way row never
  trains one negative label against another.
- ``L_primary = mean over every row of (CE + brier_weight * Brier)``; every row has a primary
  label, so the denominator is the batch's row count.

**Auxiliary heads** (each present only when its role is mapped; a null target is masked, and an
empty mask makes its term exactly zero, never a division by zero):

- *Flag*: one logit per row. ``L_flag`` is the mean BCE-with-logits over the rows whose flag is
  not null.
- *Support*: one logit per premise. Each eligible row (support set not null, at least one
  premise) contributes the mean BCE over its premises; ``L_support`` is the mean of those per-row
  means, so long rows do not dominate.

``L = L_primary + flag_weight * L_flag + support_weight * L_support``. Every mean is over the batch
as given: the trainer refuses gradient accumulation, because accumulating would change these
denominators.

Torch is imported inside the function (the ``[finetuning]`` extra), so importing this module needs
no heavy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["LossTerms", "LossWeights", "classifier_loss"]


@dataclass(frozen=True)
class LossWeights:
    """Coefficients of the objective. ``brier_weight=0`` is plain (grouped) soft cross-entropy."""

    brier_weight: float = 0.0
    flag_weight: float = 1.0
    support_weight: float = 1.0


@dataclass(frozen=True)
class LossTerms:
    """The total loss and its parts, each a 0-dim tensor (the parts detached)."""

    total: Any
    primary: Any
    flag: Any
    support: Any


def _torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:  # pragma: no cover - the [finetuning] extra is installed with it
        msg = "the [finetuning] extra is required: pip install 'strata-forge[finetuning]'"
        raise ImportError(msg) from exc


def classifier_loss(
    logits: Any,
    *,
    target: Any,
    two_way: Any,
    two_way_target: Any,
    positive_index: int,
    weights: LossWeights,
    flag_logits: Any = None,
    flag_target: Any = None,
    support_logits: Any = None,
    support_target: Any = None,
    support_valid: Any = None,
    support_row: Any = None,
) -> LossTerms:
    """The objective for one batch. Shapes, with ``B`` rows, ``K`` labels, ``P`` premise slots:

    - ``logits`` ``[B, K]`` float; ``target`` ``[B, K]`` (a distribution; ignored on two-way rows);
    - ``two_way`` ``[B]`` bool; ``two_way_target`` ``[B]`` float in ``{0, 1}`` (two-way rows only);
    - ``flag_logits`` ``[B]`` and ``flag_target`` ``[B]`` float, ``-1`` where the flag is null;
    - ``support_logits`` / ``support_target`` / ``support_valid`` ``[B, P]`` and ``support_row``
      ``[B]`` bool (the support set is not null).

    The loss is computed in at least float32 (float64 stays float64), so a bf16 forward pass does
    not round the objective.
    """
    torch: Any = _torch()
    functional: Any = torch.nn.functional
    dtype: Any = torch.float64 if logits.dtype == torch.float64 else torch.float32
    logits32: Any = logits.to(dtype)
    lse: Any = torch.logsumexp(logits32, dim=-1)
    log_p: Any = logits32 - lse.unsqueeze(-1)
    p: Any = log_p.exp()
    target32: Any = target.to(dtype)

    ce_full: Any = -(target32 * log_p).sum(-1)
    brier_full: Any = ((p - target32) ** 2).sum(-1)

    num_labels = int(logits32.shape[-1])
    negative = [k for k in range(num_labels) if k != positive_index]
    log_q: Any = log_p[:, positive_index]
    log_not_q: Any = torch.logsumexp(logits32[:, negative], dim=-1) - lse
    t: Any = two_way_target.to(dtype)
    ce_two: Any = -(t * log_q + (1.0 - t) * log_not_q)
    brier_two: Any = 2.0 * (log_q.exp() - t) ** 2

    is_two: Any = two_way.bool()
    per_row: Any = torch.where(
        is_two,
        ce_two + weights.brier_weight * brier_two,
        ce_full + weights.brier_weight * brier_full,
    )
    primary: Any = per_row.mean()
    zero: Any = primary.new_zeros(())

    flag: Any = zero
    if flag_logits is not None and flag_target is not None:
        flag_t: Any = flag_target.to(dtype)
        known: Any = flag_t >= 0
        bce: Any = functional.binary_cross_entropy_with_logits(
            flag_logits.to(dtype), flag_t.clamp(min=0.0), reduction="none"
        )
        flag = (bce * known).sum() / known.sum().clamp(min=1)

    support: Any = zero
    if support_logits is not None:
        valid: Any = support_valid.to(dtype)
        bce_s: Any = functional.binary_cross_entropy_with_logits(
            support_logits.to(dtype), support_target.to(dtype), reduction="none"
        )
        per_premise_count: Any = valid.sum(-1)
        row_mean: Any = (bce_s * valid).sum(-1) / per_premise_count.clamp(min=1)
        eligible: Any = support_row.bool() & (per_premise_count > 0)
        support = (row_mean * eligible).sum() / eligible.sum().clamp(min=1)

    total: Any = primary + weights.flag_weight * flag + weights.support_weight * support
    return LossTerms(
        total=total, primary=primary.detach(), flag=flag.detach(), support=support.detach()
    )
