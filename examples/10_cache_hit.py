"""Cache hit demo: the same request served from cache on the second call.

The provider-agnostic cache key means the cached entry survives
provider failover — a hit recorded against Anthropic is just as valid
when the next request would have hit Bedrock.

Usage::

    uv run python examples/10_cache_hit.py
"""

from __future__ import annotations

import asyncio

from _common import parse_args, print_summary, require_env

from strata_forge.llm import InMemoryCache, LLMClient, Message


async def _main() -> None:
    args = parse_args(
        description="In-memory cache demo — same call twice; second is a hit.",
    )
    require_env(args.provider)

    cache = InMemoryCache(max_size=32)
    client = LLMClient(args.model, provider=args.provider, cache=cache)
    messages = [Message.user("Reply with the single word 'cached'.")]

    print("--- first call (miss, hits the provider) ---")
    first = await client.complete(messages)
    print_summary(first, label="first")
    print()
    print("--- second call (hit, returns cached response) ---")
    second = await client.complete(messages)
    print_summary(second, label="second")
    print()
    print(f"cache_hit changed: {first.cache_hit} -> {second.cache_hit}")
    print(f"cost_usd changed:  ${first.cost_usd:.6f} -> ${second.cost_usd:.6f}")


if __name__ == "__main__":
    asyncio.run(_main())
