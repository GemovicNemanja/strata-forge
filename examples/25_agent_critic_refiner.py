"""Critic-refiner multi-agent pattern.

Requires a configured provider. Skips cleanly when keys aren't set.

The drafter produces an initial answer; the critic reviews it and
either approves or returns feedback. The loop continues until the
critic approves or ``max_rounds`` is hit. In production setups, the
critic is often a stronger model than the drafter; here we use the
same model for both to keep the example self-contained.

Usage::

    uv run python examples/25_agent_critic_refiner.py
"""

from __future__ import annotations

import asyncio

from _common import parse_args, require_env  # type: ignore[import-not-found]

from strata_forge.agents import Agent, critic_refiner_run
from strata_forge.llm.client import LLMClient


async def _main() -> None:
    args = parse_args(
        description="Critic-refiner pattern: drafter + critic loop.",
    )
    require_env(args.provider)

    client = LLMClient(model=args.model, provider=args.provider)

    drafter = Agent(
        "drafter",
        client=client,
        system_prompt=(
            "You are a writer producing short product descriptions. "
            "Aim for two crisp sentences, vivid but factual."
        ),
    )
    critic = Agent(
        "critic",
        client=client,
        system_prompt=(
            "You are an editor reviewing product descriptions. Approve when "
            "the draft is two sentences, vivid, factual, and avoids cliches. "
            "Reject with specific actionable feedback otherwise."
        ),
    )

    request = (
        "Write a product description for a noise-cancelling sleep mask with built-in speakers."
    )
    print(f"--- request ---\n{request}\n")
    final = await critic_refiner_run(
        drafter=drafter,
        critic=critic,
        user_input=request,
        max_rounds=3,
    )
    print(f"--- final draft ---\n{final.text}\n")
    print(f"[final cost: ${final.cost_usd:.6f}]")


if __name__ == "__main__":
    asyncio.run(_main())
