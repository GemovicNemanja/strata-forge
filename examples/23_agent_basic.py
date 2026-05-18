"""Build an :class:`Agent` with a built-in tool and run it once.

Requires a configured provider (defaults to Anthropic). Skips
cleanly when keys aren't set.

The example uses the ``calculator`` tool — a safe arithmetic
evaluator — so the LLM can compute exact answers instead of
hallucinating them. The agent's ``.run()`` returns an
:class:`AgentResult` carrying the final text plus the conversation
that was sent.

Usage::

    uv run python examples/23_agent_basic.py
    uv run python examples/23_agent_basic.py --model gpt-5.5 --provider openai
"""

from __future__ import annotations

import asyncio

from _common import parse_args, require_env  # type: ignore[import-not-found]

from forge.agents import Agent, calculator
from forge.llm.client import LLMClient


async def _main() -> None:
    args = parse_args(description="Run an agent with the calculator tool.")
    require_env(args.provider)

    client = LLMClient(model=args.model, provider=args.provider)
    agent = Agent(
        "math-helper",
        client=client,
        system_prompt=(
            "You are a careful math assistant. When a calculation is needed, "
            "use the calculator tool — don't guess. Reply with the final "
            "answer in plain English."
        ),
        tools=[calculator],
    )

    question = "What is (15 * 23) / 4 - 2**3?"
    print(f"--- question ---\n{question}\n")
    result = await agent.run(question)
    print(f"--- answer ---\n{result.text}\n")
    print(
        f"[final-iteration: cost=${result.cost_usd:.6f} "
        f"latency={result.latency_ms:.1f}ms "
        f"route={result.final_response.route.model}@{result.final_response.route.provider}]"
    )


if __name__ == "__main__":
    asyncio.run(_main())
