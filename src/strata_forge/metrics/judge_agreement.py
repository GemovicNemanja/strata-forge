"""How far an LLM judge's scores agree with gold (human) scores of the same outputs.

One criterion at a time, over the cases that have BOTH a gold score and a judge score: the
contingency table (gold rows, judge columns, in scale order), the share of exact agreement,
Cohen's kappa, and for an integer scale the mean absolute error and the quadratically weighted
kappa. Every value is computed exactly (``Fraction``) and reported as a float, or as null with a
reason when it is undefined: no cases, or a chance agreement of one (every case in one cell).

These are point values. An interval over the agreement share or the absolute error is a cluster
bootstrap of a per-case value (:func:`strata_forge.metrics.cluster_bootstrap.cluster_mean_bootstrap`),
which the run computes, since only it knows the clusters. Nothing here decides a gate.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.core.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["JUDGE_AGREEMENT_VERSION", "JudgeAgreement", "judge_agreement"]

JUDGE_AGREEMENT_VERSION = "judge-gold-agreement-v1"
type KappaUndefined = Literal["empty", "chance_agreement_one"]


class JudgeAgreement(BaseModel):
    """Agreement of judge scores with gold scores on one criterion."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)

    metric_version: Literal["judge-gold-agreement-v1"] = JUDGE_AGREEMENT_VERSION
    scale: Literal["pass_fail", "ordinal"]
    values: tuple[int, ...]
    n: int = Field(ge=0)
    #: ``counts[g][j]``: cases gold scored ``values[g]`` and the judge ``values[j]``.
    counts: tuple[tuple[int, ...], ...]
    percent_agreement: float | None
    cohen_kappa: float | None
    cohen_kappa_undefined: KappaUndefined | None
    #: Integer scales only; null for pass/fail.
    mean_absolute_error: float | None
    quadratic_weighted_kappa: float | None
    quadratic_weighted_kappa_undefined: KappaUndefined | None


def _kappa(observed: Fraction, chance: Fraction) -> tuple[float | None, KappaUndefined | None]:
    if chance == 1:
        return None, "chance_agreement_one"
    return float((observed - chance) / (1 - chance)), None


def judge_agreement(
    gold: Sequence[int],
    judged: Sequence[int],
    *,
    values: Sequence[int],
    scale: Literal["pass_fail", "ordinal"],
) -> JudgeAgreement:
    """Agreement over paired ``(gold, judged)`` scores, each one of ``values`` (in scale order)."""
    order = {value: index for index, value in enumerate(values)}
    if len(order) != len(values) or len(order) < 2:
        raise ValidationError("judge_agreement_values_invalid")
    if len(gold) != len(judged):
        raise ValidationError("judge_agreement_population_mismatch")
    if any(type(v) is not int or v not in order for v in (*gold, *judged)):
        raise ValidationError("judge_agreement_score_invalid")
    size = len(values)
    counts = [[0] * size for _ in range(size)]
    for g, j in zip(gold, judged, strict=True):
        counts[order[g]][order[j]] += 1
    n = len(gold)
    ordinal = scale == "ordinal"
    if not n:
        return JudgeAgreement(
            scale=scale,
            values=tuple(values),
            n=0,
            counts=tuple(tuple(row) for row in counts),
            percent_agreement=None,
            cohen_kappa=None,
            cohen_kappa_undefined="empty",
            mean_absolute_error=None,
            quadratic_weighted_kappa=None,
            quadratic_weighted_kappa_undefined="empty" if ordinal else None,
        )
    rows = [sum(row) for row in counts]
    cols = [sum(counts[g][j] for g in range(size)) for j in range(size)]
    observed = Fraction(sum(counts[i][i] for i in range(size)), n)
    chance = Fraction(sum(r * c for r, c in zip(rows, cols, strict=True)), n * n)
    kappa, kappa_undefined = _kappa(observed, chance)
    mae: float | None = None
    qwk: float | None = None
    qwk_undefined: KappaUndefined | None = None
    if ordinal:
        mae = float(Fraction(sum(abs(g - j) for g, j in zip(gold, judged, strict=True)), n))
        weight = [
            [Fraction((g - j) ** 2, (size - 1) ** 2) for j in range(size)] for g in range(size)
        ]
        disagreement = sum(
            weight[g][j] * Fraction(counts[g][j], n) for g in range(size) for j in range(size)
        )
        expected = sum(
            weight[g][j] * Fraction(rows[g] * cols[j], n * n)
            for g in range(size)
            for j in range(size)
        )
        if expected == 0:
            qwk_undefined = "chance_agreement_one"
        else:
            qwk = float(1 - disagreement / expected)
    return JudgeAgreement(
        scale=scale,
        values=tuple(values),
        n=n,
        counts=tuple(tuple(row) for row in counts),
        percent_agreement=float(observed),
        cohen_kappa=kappa,
        cohen_kappa_undefined=kappa_undefined,
        mean_absolute_error=mae,
        quadratic_weighted_kappa=qwk,
        quadratic_weighted_kappa_undefined=qwk_undefined,
    )
