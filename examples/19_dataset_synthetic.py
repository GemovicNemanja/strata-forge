"""Synthetic data: ``self_instruct`` then ``distill`` against a real LLM.

Requires a configured provider (defaults to Anthropic). Skips cleanly
when keys aren't set.

The example:

1. Defines two seed Q&A items.
2. Asks the LLM to invent more items in the same shape via
   :func:`self_instruct`.
3. Strips the LLM-supplied ``expected_output`` from the new items and
   reruns them through :func:`distill` so a (potentially different)
   teacher model produces the labels.

Usage::

    uv run python examples/19_dataset_synthetic.py
    uv run python examples/19_dataset_synthetic.py --model gpt-5.5 --provider openai
"""

from __future__ import annotations

import asyncio

from _common import parse_args, require_env  # type: ignore[import-not-found]

from forge.datasets import Dataset, DatasetItem, distill, self_instruct
from forge.llm.client import LLMClient


def _seeds() -> tuple[DatasetItem, ...]:
    return (
        DatasetItem.from_input(
            {"question": "What is the capital of France?"},
            expected_output="Paris",
        ),
        DatasetItem.from_input(
            {"question": "Who wrote 'Hamlet'?"},
            expected_output="William Shakespeare",
        ),
    )


async def _main() -> None:
    args = parse_args(
        description="Demo self_instruct + distill against a real LLM.",
    )
    require_env(args.provider)

    client = LLMClient(model=args.model, provider=args.provider)

    print("--- self_instruct: generating 3 new items from 2 seeds ---")
    generated = await self_instruct(
        seeds=_seeds(),
        instructions=(
            "Produce one-fact trivia Q&A items. Each input has a single "
            "'question' field; expected_output is the short answer."
        ),
        n=3,
        client=client,
        name="trivia-synth",
        batch_size=3,
    )
    for item in generated.items:
        print(f"  [{item.id[:8]}...] Q: {item.input['question']}")
        print(f"          A: {item.expected_output}")

    print("\n--- distill: re-label the generated items with the same teacher ---")
    unlabeled = Dataset(
        name="trivia-synth",
        items=tuple(
            DatasetItem(id=item.id, input=dict(item.input), expected_output=None)
            for item in generated.items
        ),
    )
    relabeled = await distill(
        inputs=unlabeled,
        teacher=client,
        name="trivia-distilled",
        concurrency=3,
    )
    for item in relabeled.items:
        print(f"  [{item.id[:8]}...] Q: {item.input['question']}")
        print(f"          A: {item.expected_output}")


if __name__ == "__main__":
    asyncio.run(_main())
