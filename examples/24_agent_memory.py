"""Multi-turn agent backed by :class:`ConversationMemory`.

Requires a configured provider. Skips cleanly when keys aren't set.

The agent keeps its own history via :class:`ConversationMemory`.
Between turns the example trims the history to a token budget so it
demonstrates how to keep memory usage bounded over a long
conversation. The system message is preserved across every trim.

Usage::

    uv run python examples/24_agent_memory.py
"""

from __future__ import annotations

import asyncio

from _common import parse_args, require_env  # type: ignore[import-not-found]

from forge.agents import Agent, ConversationMemory
from forge.llm.client import LLMClient


async def _main() -> None:
    args = parse_args(description="Multi-turn agent with conversation memory.")
    require_env(args.provider)

    client = LLMClient(model=args.model, provider=args.provider)
    agent = Agent(
        "remembering-assistant",
        client=client,
        system_prompt=(
            "You are a concise assistant. Remember facts the user mentions "
            "across the conversation and reuse them when relevant."
        ),
    )
    memory = ConversationMemory(system_message=agent.system_prompt)

    turns = [
        "My name is Sam and my favourite colour is teal.",
        "What did I just tell you about myself?",
        "Suggest a logo background colour for my notebook.",
    ]

    for i, user_input in enumerate(turns, start=1):
        # Append the user message before the call so it's in the history
        # the agent sees.
        memory.append_user(user_input)
        result = await agent.run(memory.non_system_messages)
        memory.append_assistant(result.text)

        # Keep history bounded — trim to 1000 tokens between turns.
        memory.trim_to_tokens(1000, model=args.model)

        print(f"--- turn {i} ---")
        print(f"USER: {user_input}")
        print(f"ASSISTANT: {result.text}")
        print(
            f"[history size: {len(memory.non_system_messages)} messages, "
            f"cost=${result.cost_usd:.6f}]\n"
        )


if __name__ == "__main__":
    asyncio.run(_main())
