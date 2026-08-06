"""BudgetContext enforces a cost ceiling — the second call would exceed it.

The first call lands inside the budget; the second projected spend
trips :class:`strata_forge.core.errors.BudgetExceededError` *before* the
provider call is made, so spend never overshoots.

Usage::

    uv run python examples/09_budget_ceiling.py
"""

from __future__ import annotations

import asyncio

from _common import parse_args, print_summary, require_env

from strata_forge.core.budget import BudgetContext
from strata_forge.core.errors import BudgetExceededError
from strata_forge.llm import LLMClient, Message


async def _main() -> None:
    args = parse_args(
        description="BudgetContext demo — pre-call spend ceiling.",
    )
    require_env(args.provider)

    client = LLMClient(args.model, provider=args.provider)

    # Deliberately tiny budget — enough for one cheap call.
    async with BudgetContext(max_usd=0.001) as budget:
        print("--- first call (should succeed) ---")
        first = await client.complete(
            [Message.user("Reply with 'hi'.")],
            max_tokens=4,
        )
        print_summary(first, label="first")
        print(f"budget spent: ${budget.spent_usd:.6f} / ${budget.max_usd}")

        print()
        print("--- second call (expected: BudgetExceededError) ---")
        try:
            await client.complete([Message.user("Write a long essay about cats.")])
        except BudgetExceededError as exc:
            print(f"caught: {exc}")
            print(f"budget after: ${budget.spent_usd:.6f} / ${budget.max_usd}")


if __name__ == "__main__":
    asyncio.run(_main())
