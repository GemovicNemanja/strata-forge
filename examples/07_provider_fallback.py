"""Provider-level failover: same logical model, multiple provider routes.

Anthropic's Claude is served by three different providers in this demo.
If Anthropic returns a rate-limit (or any retryable failure), the
fallback runner advances to Bedrock; if Bedrock also fails, Vertex
picks up. The cache key is provider-agnostic, so a hit on one provider
serves identical requests across all three.

Usage::

    uv run python examples/07_provider_fallback.py
"""

from __future__ import annotations

import asyncio

from _common import print_summary, require_env

from forge.llm import LLMClient, Message, ModelFallback


async def _main() -> None:
    # All three Anthropic routes must be configured for the demo to be
    # meaningful end-to-end; we skip if any one is missing.
    for provider in ("anthropic", "bedrock", "vertex"):
        require_env(provider)

    client = LLMClient(
        chain=[
            ModelFallback(
                model="claude-opus-4-7",
                providers=("anthropic", "bedrock", "vertex"),
            ),
        ],
    )
    response = await client.complete(
        [Message.user("Reply with the single word 'fallback'.")]
    )
    print(f"--- served by {response.route.provider} ---")
    print_summary(response)


if __name__ == "__main__":
    asyncio.run(_main())
