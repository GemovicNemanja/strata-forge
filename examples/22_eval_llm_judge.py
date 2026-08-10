"""Grade open-ended responses with :class:`LLMJudge`.

Requires a configured provider (defaults to Anthropic). Skips cleanly
when keys aren't set.

ExactMatch isn't useful for open-ended answers — "Paris, France" and
"Paris" wouldn't match. :class:`LLMJudge` lets a (typically stronger)
model score the response on a free-form criterion. This example asks
the candidate model to summarize a paragraph, then asks the same
model — acting as judge — whether the summary captures the key idea.

Usage::

    uv run python examples/22_eval_llm_judge.py
    uv run python examples/22_eval_llm_judge.py --model gpt-5.5 --provider openai
"""

from __future__ import annotations

import asyncio

from _common import parse_args, require_env  # type: ignore[import-not-found]

from strata_forge.datasets import Dataset, DatasetItem
from strata_forge.evals import (
    Experiment,
    LLMJudge,
    SamplingParams,
    pass_rate,
    render_markdown,
    run_experiment,
)
from strata_forge.llm.client import LLMClient
from strata_forge.llm.messages import SystemMessage, UserMessage


def _dataset() -> Dataset:
    items = (
        DatasetItem.from_input(
            {
                "paragraph": (
                    "Photosynthesis converts light energy into chemical energy stored in "
                    "glucose. Plants, algae, and some bacteria perform photosynthesis "
                    "using chlorophyll to absorb sunlight."
                )
            },
            expected_output="Photosynthesis converts light into chemical energy.",
        ),
        DatasetItem.from_input(
            {
                "paragraph": (
                    "The Treaty of Versailles, signed in 1919, formally ended World War I. "
                    "It imposed reparations on Germany and reshaped European borders."
                )
            },
            expected_output="The Treaty of Versailles ended WWI in 1919 with reparations on Germany.",
        ),
        DatasetItem.from_input(
            {
                "paragraph": (
                    "TCP guarantees in-order, reliable delivery of bytes between two "
                    "endpoints using sequence numbers, acknowledgments, and retransmission."
                )
            },
            expected_output="TCP provides reliable, in-order byte delivery via sequence numbers and retransmission.",
        ),
    )
    return Dataset(name="summary", items=items)


def _renderer(item: DatasetItem) -> list:
    return [
        SystemMessage(content="Summarize the paragraph in one short sentence."),
        UserMessage(content=str(item.input["paragraph"])),
    ]


async def _main() -> None:
    args = parse_args(description="Grade summaries with an LLM judge.")
    require_env(args.provider)

    dataset = _dataset()
    candidate = LLMClient(model=args.model, provider=args.provider)
    judge = LLMJudge(
        client=candidate,  # using the same model as judge — swap to a stronger one in production
        criteria=(
            "Does the candidate summary capture the key idea of the paragraph? "
            "Length and exact wording don't matter; faithfulness to the core "
            "claim does. Score 0.0 to 1.0."
        ),
        pass_threshold=0.7,
    )

    experiment = Experiment(
        name="summary-eval",
        models=(args.model,),
        prompts=("summarize",),
        dataset_name=dataset.name,
        grader_names=("llm_judge",),
        sampling=SamplingParams(max_tokens=80),
    )

    outcomes = await run_experiment(
        experiment,
        dataset=dataset,
        clients={args.model: candidate},
        graders=[judge],
        prompt_renderers={"summarize": _renderer},
        concurrency=3,
    )

    print(render_markdown(outcomes, title="Summary eval (LLM judge)"))
    print(f"\noverall pass rate = {pass_rate(outcomes, grader_name='llm_judge'):.1%}")


if __name__ == "__main__":
    asyncio.run(_main())
