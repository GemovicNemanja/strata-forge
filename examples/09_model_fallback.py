"""Model-level fallback: drop to a different logical model when the first exhausts.

The two-axis chain tries every provider for Claude first, then advances
to GPT-5.5 if every Claude route has been exhausted. Bare strings in
the chain expand to "use the registry default route".

Usage::

    uv run python examples/09_model_fallback.py
"""

from __future__ import annotations

import asyncio

from _common import print_summary, require_env

from strata_forge.llm import LLMClient, Message


async def _main() -> None:
    for provider in ("anthropic", "openai"):
        require_env(provider)

    client = LLMClient.with_fallbacks(["claude-opus-4-7", "gpt-5.5"])
    response = await client.complete([Message.user("Reply with the single word 'fallback'.")])
    print(f"--- served by {response.route.model}@{response.route.provider} ---")
    print_summary(response)


if __name__ == "__main__":
    asyncio.run(_main())
