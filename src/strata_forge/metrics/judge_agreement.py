"""How far an LLM judge's scores agree with gold (human) scores of the same outputs.

One criterion at a time, over the cases that have BOTH a gold score and a judge score: the
contingency table (gold rows, judge columns, in scale order), the share of exact agreement,
Cohen's kappa, and for an integer scale the mean absolute error and the linearly and
quadratically weighted kappas. Every value is computed exactly (``Fraction``) and reported as a float, or as null with a
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
#: Why a value is null: no paired cases, a chance agreement of one (every case in one cell, so a
#: kappa divides by zero), or a value that only an integer scale has.
type Undefined = Literal["empty", "chance_agreement_one", "not_an_integer_scale"]


class JudgeAgreement(BaseModel):
    """Agreement of judge scores with gold scores on one criterion.

    Every value is a float, or null with the reason beside it in ``<name>_undefined``.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)

    metric_version: Literal["judge-gold-agreement-v1"] = JUDGE_AGREEMENT_VERSION
    scale: Literal["pass_fail", "ordinal"]
    values: tuple[int, ...]
    n: int = Field(ge=0)
    #: ``counts[g][j]``: cases gold scored ``values[g]`` and the judge ``values[j]``.
    counts: tuple[tuple[int, ...], ...]
    percent_agreement: float | None
    percent_agreement_undefined: Undefined | None
    cohen_kappa: float | None
    cohen_kappa_undefined: Undefined | None
    #: Integer scales only (``not_an_integer_scale`` on pass/fail).
    mean_absolute_error: float | None
    mean_absolute_error_undefined: Undefined | None
    linear_weighted_kappa: float | None
    linear_weighted_kappa_undefined: Undefined | None
    quadratic_weighted_kappa: float | None
    quadratic_weighted_kappa_undefined: Undefined | None


def _weighted_kappa(
    counts: list[list[int]], rows: list[int], cols: list[int], n: int, power: int
) -> tuple[float | None, Undefined | None]:
    """``1 - sum(w * observed) / sum(w * expected)`` with ``w = (|g - j| / (k - 1)) ** power``."""
    size = len(counts)
    weight = [[Fraction(abs(g - j), size - 1) ** power for j in range(size)] for g in range(size)]
    observed = sum(
        weight[g][j] * Fraction(counts[g][j], n) for g in range(size) for j in range(size)
    )
    expected = sum(
        weight[g][j] * Fraction(rows[g] * cols[j], n * n) for g in range(size) for j in range(size)
    )
    if expected == 0:
        return None, "chance_agreement_one"
    return float(1 - observed / expected), None


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
    integer_only: Undefined | None = None if ordinal else "not_an_integer_scale"
    table = tuple(tuple(row) for row in counts)
    if not n:
        empty: Undefined = "empty"
        return JudgeAgreement(
            scale=scale,
            values=tuple(values),
            n=0,
            counts=table,
            percent_agreement=None,
            percent_agreement_undefined=empty,
            cohen_kappa=None,
            cohen_kappa_undefined=empty,
            mean_absolute_error=None,
            mean_absolute_error_undefined=integer_only or empty,
            linear_weighted_kappa=None,
            linear_weighted_kappa_undefined=integer_only or empty,
            quadratic_weighted_kappa=None,
            quadratic_weighted_kappa_undefined=integer_only or empty,
        )
    rows = [sum(row) for row in counts]
    cols = [sum(counts[g][j] for g in range(size)) for j in range(size)]
    observed = Fraction(sum(counts[i][i] for i in range(size)), n)
    chance = Fraction(sum(r * c for r, c in zip(rows, cols, strict=True)), n * n)
    kappa: float | None = None
    kappa_undefined: Undefined | None = "chance_agreement_one"
    if chance != 1:
        kappa, kappa_undefined = float((observed - chance) / (1 - chance)), None
    mae: float | None = None
    lwk = qwk = None
    lwk_undefined = qwk_undefined = integer_only
    if ordinal:
        mae = float(Fraction(sum(abs(g - j) for g, j in zip(gold, judged, strict=True)), n))
        lwk, lwk_undefined = _weighted_kappa(counts, rows, cols, n, 1)
        qwk, qwk_undefined = _weighted_kappa(counts, rows, cols, n, 2)
    return JudgeAgreement(
        scale=scale,
        values=tuple(values),
        n=n,
        counts=table,
        percent_agreement=float(observed),
        percent_agreement_undefined=None,
        cohen_kappa=kappa,
        cohen_kappa_undefined=kappa_undefined,
        mean_absolute_error=mae,
        mean_absolute_error_undefined=integer_only,
        linear_weighted_kappa=lwk,
        linear_weighted_kappa_undefined=lwk_undefined,
        quadratic_weighted_kappa=qwk,
        quadratic_weighted_kappa_undefined=qwk_undefined,
    )
