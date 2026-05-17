"""Streaming: print chunks to stdout as they arrive.

Provider-level failover and the cache are intentionally NOT applied to
streams — mid-stream provider switch is impractical.

Usage::

    uv run python examples/04_streaming.py --model claude-opus-4-7 --provider anthropic
"""

from __future__ import annotations

import asyncio
import sys

from _common import parse_args, require_env

from forge.llm import LLMClient, Message


async def _main() -> None:
    args = parse_args(
        description="Streaming completion demo.",
        extra_args=[
            (
                "--prompt",
                {
                    "default": "Write a haiku about distributed systems.",
                    "help": "Prompt to send to the model.",
                },
            ),
        ],
    )
    require_env(args.provider)

    client = LLMClient(args.model, provider=args.provider)
    print(f"--- streaming from {args.model}@{args.provider} ---")
    final_chunk = None
    async for chunk in await client.stream([Message.user(args.prompt)]):
        sys.stdout.write(chunk.delta_text)
        sys.stdout.flush()
        final_chunk = chunk
    print()
    if final_chunk is not None and final_chunk.usage is not None:
        print(f"[tokens={final_chunk.usage.total_tokens}]")


if __name__ == "__main__":
    asyncio.run(_main())
