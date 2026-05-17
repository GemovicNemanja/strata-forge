"""CI eval-regression gate.

Given a tuple of :class:`Outcome` instances, the gate decides whether
the run "passes" against caller-supplied thresholds. Two checks ship:

- **Pass-rate threshold** per grader, optionally with a Wilson
  lower-bound (95 %) to guard against small-sample false positives.
- **Cost cap** on total USD spend across all trials.

The result carries a list of human-readable issues plus the
machine-readable rates and bounds so CI scripts can both ``exit``
with the right code and surface useful diagnostics in the log.

The gate is pure Python — no dependencies beyond :mod:`math`. It
imports cleanly without any extras installed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from forge.evals.metrics import pass_rate_by_grader

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from forge.evals.experiment import Outcome

__all__ = [
    "CIGateResult",
    "CIGateThresholds",
    "evaluate_ci_gate",
    "wilson_lower_bound",
]

_DEFAULT_Z = 1.96  # 95 % CI


def wilson_lower_bound(passed: int, total: int, *, z: float = _DEFAULT_Z) -> float:
    """Return the Wilson score lower bound for a binomial pass rate.

    Returns ``0.0`` for ``total == 0`` (undefined; defensive default).
    The Wilson lower bound is the conservative answer to "given
    ``passed`` of ``total`` passed, what's the lowest plausible
    underlying pass rate at confidence ``z``?". CI gates use this
    instead of the raw pass rate so small samples don't pass simply
    by happening to score high.
    """
    if total <= 0:
        return 0.0
    p_hat = passed / total
    denom = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2.0 * total)) / denom
    margin = (
        z * math.sqrt(p_hat * (1.0 - p_hat) / total + (z * z) / (4.0 * total * total))
    ) / denom
    return max(0.0, center - margin)


class CIGateThresholds(BaseModel):
    """Pass criteria for :func:`evaluate_ci_gate`.

    Attributes:
        min_pass_rate: Default minimum pass rate per grader (any
            grader without an entry in ``per_grader_pass_rate``
            uses this). Default ``0.85``.
        max_cost_usd: Maximum permissible total cost across all
            trials in the run. Default ``1.0``.
        per_grader_pass_rate: Optional grader-name → minimum pass
            rate map. When a grader is listed here, its specific
            threshold overrides ``min_pass_rate``.
        use_wilson_ci: When ``True`` (default), compare the Wilson
            95 % lower bound against the threshold instead of the
            point estimate. Conservative; flips small samples to
            failures unless they're decisively above the threshold.
        wilson_z: ``z`` for the Wilson computation. Default
            ``1.96`` (95 % CI).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_pass_rate: float = Field(default=0.85, ge=0.0, le=1.0)
    max_cost_usd: float = Field(default=1.0, ge=0.0)
    per_grader_pass_rate: dict[str, float] = Field(default={})
    use_wilson_ci: bool = True
    wilson_z: float = Field(default=_DEFAULT_Z, gt=0.0)


class CIGateResult(BaseModel):
    """Verdict from :func:`evaluate_ci_gate`.

    Attributes:
        passed: ``True`` when every check cleared the threshold.
        issues: Human-readable description of the failures (empty
            when ``passed``).
        pass_rates_by_grader: Per-grader point estimates.
        wilson_lower_bounds_by_grader: Per-grader Wilson 95 % lower
            bounds.
        total_cost_usd: Sum of ``trial.cost_usd`` across outcomes.
        total_trials: Number of outcomes considered.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    issues: tuple[str, ...] = ()
    pass_rates_by_grader: dict[str, float] = Field(default={})
    wilson_lower_bounds_by_grader: dict[str, float] = Field(default={})
    total_cost_usd: float
    total_trials: int


def _grader_passed_counts(outcomes: Sequence[Outcome]) -> dict[str, tuple[int, int]]:
    """Return per-grader ``(passed, total)`` counts."""
    tallies: dict[str, tuple[int, int]] = {}
    for outcome in outcomes:
        for result in outcome.grader_results:
            passed, total = tallies.get(result.grader_name, (0, 0))
            tallies[result.grader_name] = (
                passed + (1 if result.passed else 0),
                total + 1,
            )
    return tallies


def evaluate_ci_gate(
    outcomes: Sequence[Outcome],
    *,
    thresholds: CIGateThresholds | None = None,
    per_grader_pass_rate: Mapping[str, float] | None = None,
) -> CIGateResult:
    """Decide whether ``outcomes`` clears the gate's thresholds.

    Args:
        outcomes: The outcome tuple produced by
            :func:`run_experiment`.
        thresholds: A :class:`CIGateThresholds` instance. When
            omitted, the gate uses default thresholds.
        per_grader_pass_rate: Convenience override — merged on top
            of ``thresholds.per_grader_pass_rate`` so callers can
            tighten a single grader without rebuilding the whole
            threshold object.

    Returns:
        A :class:`CIGateResult` whose ``passed`` is ``True`` iff
        every check cleared its threshold.
    """
    effective = thresholds or CIGateThresholds()
    per_grader: dict[str, float] = dict(effective.per_grader_pass_rate)
    if per_grader_pass_rate:
        per_grader.update(per_grader_pass_rate)

    total_trials = len(outcomes)
    total_cost = sum(o.trial.cost_usd for o in outcomes)
    issues: list[str] = []

    tallies = _grader_passed_counts(outcomes)
    point_rates: dict[str, float] = {}
    wilson_rates: dict[str, float] = {}

    if not outcomes:
        issues.append("no outcomes provided to evaluate")

    for grader_name, (passed_count, total) in tallies.items():
        threshold = per_grader.get(grader_name, effective.min_pass_rate)
        point = passed_count / total if total else 0.0
        wilson = wilson_lower_bound(passed_count, total, z=effective.wilson_z)
        point_rates[grader_name] = point
        wilson_rates[grader_name] = wilson
        comparator = wilson if effective.use_wilson_ci else point
        comparator_label = (
            f"wilson95={wilson:.4f}" if effective.use_wilson_ci else f"rate={point:.4f}"
        )
        if comparator < threshold:
            issues.append(
                f"grader {grader_name!r}: {comparator_label} < threshold {threshold:.4f} "
                f"(passed {passed_count}/{total})"
            )

    if total_cost > effective.max_cost_usd:
        issues.append(f"total cost ${total_cost:.6f} > cap ${effective.max_cost_usd:.6f}")

    # Make sure missing graders (referenced in per_grader but absent in outcomes)
    # surface as issues so a typo in the per_grader map doesn't silently pass.
    for missing in sorted(set(per_grader) - set(tallies)):
        issues.append(f"grader {missing!r} listed in thresholds but absent from outcomes")

    # Substitute the metric-side rates for callers who want both views.
    # (pass_rate_by_grader gives the same numbers but is the canonical entry point.)
    point_rates_canonical = pass_rate_by_grader(outcomes)
    return CIGateResult(
        passed=not issues,
        issues=tuple(issues),
        pass_rates_by_grader=point_rates_canonical,
        wilson_lower_bounds_by_grader=wilson_rates,
        total_cost_usd=total_cost,
        total_trials=total_trials,
    )
