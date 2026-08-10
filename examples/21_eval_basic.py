"""Run a small experiment end-to-end: dataset, runner, ExactMatch, metrics.

Requires a configured provider (defaults to Anthropic). Skips cleanly
when keys aren't set.

The example:

1. Builds a 4-item trivia :class:`Dataset` with reference answers.
2. Constructs a single-model :class:`Experiment` with one
   :class:`ExactMatch` grader.
3. Runs :func:`run_experiment`, then prints the
   :func:`render_markdown` report and a couple of metric numbers.

Usage::

    uv run python examples/21_eval_basic.py
    uv run python examples/21_eval_basic.py --model gpt-5.5 --provider openai
"""

from __future__ import annotations

import asyncio

from _common import parse_args, require_env  # type: ignore[import-not-found]

from strata_forge.datasets import Dataset, DatasetItem
from strata_forge.evals import (
    ExactMatch,
    Experiment,
    SamplingParams,
    mean_score,
    pass_rate,
    render_markdown,
    run_experiment,
)
from strata_forge.llm.client import LLMClient
from strata_forge.llm.messages import SystemMessage, UserMessage


def _dataset() -> Dataset:
    items = (
        DatasetItem.from_input({"q": "What is the capital of France?"}, expected_output="Paris"),
        DatasetItem.from_input({"q": "What is 2 + 2?"}, expected_output="4"),
        DatasetItem.from_input(
            {"q": "What color is the sky on a clear day?"}, expected_output="blue"
        ),
        DatasetItem.from_input(
            {"q": "Who painted the Mona Lisa?"}, expected_output="Leonardo da Vinci"
        ),
    )
    return Dataset(name="trivia", items=items, description="Tiny trivia eval set.")


def _renderer(item: DatasetItem) -> list:
    return [
        SystemMessage(
            content="Answer with just the answer — no preamble, no punctuation, lowercase if possible."
        ),
        UserMessage(content=str(item.input["q"])),
    ]


async def _main() -> None:
    args = parse_args(description="Run a small evaluation experiment.")
    require_env(args.provider)

    dataset = _dataset()
    client = LLMClient(model=args.model, provider=args.provider)

    experiment = Experiment(
        name="trivia-baseline",
        models=(args.model,),
        prompts=("answer-only",),
        dataset_name=dataset.name,
        grader_names=("exact_match",),
        sampling=SamplingParams(max_tokens=50),
    )

    outcomes = await run_experiment(
        experiment,
        dataset=dataset,
        clients={args.model: client},
        graders=[ExactMatch(case_sensitive=False)],
        prompt_renderers={"answer-only": _renderer},
        concurrency=4,
    )

    print(render_markdown(outcomes, title="Trivia eval (baseline)"))
    print()
    print("--- summary stats ---")
    print(f"pass_rate   = {pass_rate(outcomes, grader_name='exact_match'):.1%}")
    print(f"mean_score  = {mean_score(outcomes, grader_name='exact_match'):.3f}")


if __name__ == "__main__":
    asyncio.run(_main())
