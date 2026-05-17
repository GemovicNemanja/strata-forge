"""Basic non-streaming completion against any registered (model, provider).

Usage::

    uv run python examples/01_basic_completion.py
    uv run python examples/01_basic_completion.py --model gpt-5.5 --provider openai
    uv run python examples/01_basic_completion.py --model claude-opus-4-7 --provider bedrock
"""

from __future__ import annotations

import asyncio

from _common import parse_args, print_summary, require_env

from forge.llm import LLMClient, Message


async def _main() -> None:
    args = parse_args(
        description="Basic LLM completion demo.",
        extra_args=[
            (
                "--prompt",
                {
                    "default": "Say hello in one short sentence.",
                    "help": "Prompt to send to the model.",
                },
            ),
        ],
    )
    require_env(args.provider)

    client = LLMClient(args.model, provider=args.provider)
    response = await client.complete([Message.user(args.prompt)])
    print_summary(response)


if __name__ == "__main__":
    asyncio.run(_main())
