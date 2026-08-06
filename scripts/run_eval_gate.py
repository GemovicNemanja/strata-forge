"""Run the canonical eval-regression gate.

This is the script the nightly CI eval-gate job invokes. It:

1. Builds a tiny canonical dataset checked into this script (the
   capital-cities pack used elsewhere — small, deterministic,
   doesn't need an external store).
2. Runs :func:`strata_forge.evals.runner.run_experiment` against one
   configured model.
3. Compares the outcomes against the committed thresholds via
   :func:`strata_forge.evals.ci_gate.evaluate_ci_gate`.
4. Exits non-zero if the gate fails.

Usage::

    uv run python scripts/run_eval_gate.py --model claude-haiku-4-5

The script intentionally avoids the configured ``DatasetStore`` —
the gate needs to be deterministic across hosts, so the dataset
is baked in.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from strata_forge.datasets.schema import Dataset, DatasetItem
from strata_forge.evals.ci_gate import CIGateThresholds, evaluate_ci_gate
from strata_forge.evals.experiment import Experiment, SamplingParams
from strata_forge.evals.graders.exact import ExactMatch
from strata_forge.evals.runner import run_experiment
from strata_forge.llm.client import LLMClient

_CAPITALS: tuple[tuple[str, str, str], ...] = (
    ("fr", "France", "Paris"),
    ("jp", "Japan", "Tokyo"),
    ("de", "Germany", "Berlin"),
    ("au", "Australia", "Canberra"),
    ("ca", "Canada", "Ottawa"),
    ("it", "Italy", "Rome"),
    ("es", "Spain", "Madrid"),
    ("pt", "Portugal", "Lisbon"),
    ("gr", "Greece", "Athens"),
    ("eg", "Egypt", "Cairo"),
    ("ke", "Kenya", "Nairobi"),
    ("ng", "Nigeria", "Abuja"),
    ("za", "South Africa", "Pretoria"),
    ("in", "India", "Delhi"),
    ("th", "Thailand", "Bangkok"),
    ("vn", "Vietnam", "Hanoi"),
    ("kr", "South Korea", "Seoul"),
    ("ru", "Russia", "Moscow"),
    ("ie", "Ireland", "Dublin"),
    ("pl", "Poland", "Warsaw"),
)


def _build_dataset() -> Dataset:
    # Sample size is set so a perfect model clears the 0.80 Wilson lower
    # bound: at n=20, k=20 → wilson95 ≈ 0.84.
    items = tuple(
        DatasetItem(
            id=f"cap-{code}",
            input={"question": f"What is the capital of {country}? Answer with one word."},
            expected_output=capital,
        )
        for code, country, capital in _CAPITALS
    )
    return Dataset(name="eval-gate-capitals", items=items)


# Committed thresholds. Tightening these signals an intentional
# performance commitment; loosening them is a regression worth
# reviewing in code.
THRESHOLDS = CIGateThresholds(
    min_pass_rate=0.80,
    max_cost_usd=0.10,
    use_wilson_ci=True,
)


async def _main(*, model: str) -> int:
    dataset = _build_dataset()
    client = LLMClient(model=model)
    grader = ExactMatch()

    experiment = Experiment(
        name=f"eval-gate-capitals-{model}",
        models=(model,),
        dataset_name=dataset.name,
        grader_names=(grader.name,),
        sampling=SamplingParams(temperature=0.0, max_tokens=20),
    )

    outcomes = await run_experiment(
        experiment,
        dataset=dataset,
        clients={model: client},
        graders=(grader,),
        concurrency=3,
        continue_on_error=True,
    )
    result = evaluate_ci_gate(outcomes, thresholds=THRESHOLDS)

    print(f"=== eval gate report: {experiment.name}")
    print(f"  trials:   {result.total_trials}")
    print(f"  passed:   {result.passed}")
    print(f"  cost:     ${result.total_cost_usd:.5f}")
    for grader_name, rate in result.pass_rates_by_grader.items():
        wilson = result.wilson_lower_bounds_by_grader.get(grader_name, float("nan"))
        print(f"  grader {grader_name!r}: rate={rate:.3f} wilson_lo={wilson:.3f}")
    if result.issues:
        print("  issues:")
        for issue in result.issues:
            print(f"    - {issue}")
    return 0 if result.passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5",
        help="Logical model name from the Forge registry.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(model=args.model)))


if __name__ == "__main__":
    main()
