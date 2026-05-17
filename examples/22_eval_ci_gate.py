"""Demo the CI eval-regression gate on synthetic outcomes.

No provider keys or extras needed. The example fabricates a small
:class:`Outcome` tuple — the kind :func:`run_experiment` would
produce — and shows three scenarios:

1. A run that passes both pass-rate and cost thresholds.
2. A run that fails the pass-rate threshold (small sample falls
   below the Wilson lower bound).
3. A run that passes the pass-rate threshold but blows the cost cap.

In a real CI pipeline, the workflow is::

    outcomes = await run_experiment(...)
    result = evaluate_ci_gate(outcomes, thresholds=CIGateThresholds(...))
    if not result.passed:
        for issue in result.issues:
            print(f"::error::{issue}")
        sys.exit(1)

Usage::

    uv run python examples/22_eval_ci_gate.py
"""

from __future__ import annotations

import sys

from forge.evals import (
    CIGateThresholds,
    GraderResult,
    Outcome,
    Trial,
    evaluate_ci_gate,
)
from forge.llm.responses import Usage


def _outcome(*, passed: bool, cost_usd: float = 0.001) -> Outcome:
    return Outcome(
        trial=Trial(
            experiment_name="demo",
            model="m",
            provider="p",
            item_id="i",
            response_text="x",
            usage=Usage(input_tokens=1, output_tokens=1),
            cost_usd=cost_usd,
            cache_hit=False,
            latency_ms=10.0,
        ),
        grader_results=(
            GraderResult(grader_name="acc", score=1.0 if passed else 0.0, passed=passed),
        ),
    )


def _print_result(label: str, result: object) -> None:
    print(f"\n--- {label} ---")
    print(f"passed:    {result.passed}")  # type: ignore[attr-defined]
    print(f"point pass-rate: {result.pass_rates_by_grader}")  # type: ignore[attr-defined]
    print(f"wilson 95% LB:   {result.wilson_lower_bounds_by_grader}")  # type: ignore[attr-defined]
    print(f"total cost USD:  {result.total_cost_usd:.4f}")  # type: ignore[attr-defined]
    if not result.passed:  # type: ignore[attr-defined]
        print("issues:")
        for issue in result.issues:  # type: ignore[attr-defined]
            print(f"  - {issue}")


def _main() -> int:
    thresholds = CIGateThresholds(min_pass_rate=0.85, max_cost_usd=1.0)

    # 1) 50 outcomes, 100% passing, all cheap — passes everything.
    passing_run = [_outcome(passed=True) for _ in range(50)]
    _print_result("Healthy run", evaluate_ci_gate(passing_run, thresholds=thresholds))

    # 2) Only 5 outcomes, all passing — Wilson 95% LB ~0.57 < 0.85.
    flaky_small_run = [_outcome(passed=True) for _ in range(5)]
    _print_result(
        "Small sample falls below Wilson LB",
        evaluate_ci_gate(flaky_small_run, thresholds=thresholds),
    )

    # 3) Pass rate OK but the budget exploded.
    expensive_run = [_outcome(passed=True, cost_usd=0.05) for _ in range(50)]
    _print_result(
        "Pass rate fine, cost cap blown",
        evaluate_ci_gate(expensive_run, thresholds=thresholds),
    )

    # Demonstrate the typical CI script exit code path against the first scenario.
    result = evaluate_ci_gate(passing_run, thresholds=thresholds)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(_main())
