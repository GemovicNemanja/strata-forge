"""Structured output: ask the model for a Pydantic-validated object.

The dispatch picks the right channel per provider — OpenAI's strict
``response_format``, Gemini's ``response_schema``, or Anthropic's
forced-tool emulation — without the caller knowing the difference.

Usage::

    uv run python examples/02_structured_output.py --model gpt-5.5 --provider openai
"""

from __future__ import annotations

import asyncio

from _common import parse_args, print_summary, require_env
from pydantic import BaseModel, Field

from strata_forge.llm import LLMClient, Message


class Summary(BaseModel):
    """Outline of an arbitrary blob of text."""

    title: str = Field(..., description="Short title for the text.")
    bullets: list[str] = Field(..., description="3-5 key points.")


async def _main() -> None:
    args = parse_args(
        description="Structured-output demo: parse the model's reply into a Pydantic model.",
        extra_args=[
            (
                "--text",
                {
                    "default": (
                        "Forge is a typed, async-first baseline for AI experimentation "
                        "across major LLM providers. It bundles caching, structured output, "
                        "fallbacks, and tool calling out of the box."
                    ),
                    "help": "Text to summarize.",
                },
            ),
        ],
    )
    require_env(args.provider)

    client = LLMClient(args.model, provider=args.provider)
    response = await client.complete_structured(
        [Message.user(f"Summarize this text:\n\n{args.text}")],
        schema=Summary,
    )

    print("--- parsed Summary ---")
    print(f"title:   {response.parsed.title}")
    print("bullets:")
    for bullet in response.parsed.bullets:
        print(f"  - {bullet}")
    print()
    print_summary(response, label="metrics")


if __name__ == "__main__":
    asyncio.run(_main())
