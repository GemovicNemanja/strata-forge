"""Multimodal: send an image and ask the model to describe it.

Each provider gets the image in its native wire format:
OpenAI's ``image_url``, Anthropic's ``image`` content block, or Gemini's
``inline_data``/``file_data``.

Usage::

    uv run python examples/03_image_input.py --model gpt-5.5 --provider openai \\
        --image https://upload.wikimedia.org/wikipedia/commons/8/87/Tabby_cat.jpg
"""

from __future__ import annotations

import asyncio

from _common import parse_args, print_summary, require_env

from forge.llm import ImageContent, LLMClient, TextPart, UserMessage


async def _main() -> None:
    args = parse_args(
        description="Image-input demo: describe an image at a URL.",
        extra_args=[
            (
                "--image",
                {
                    "default": (
                        "https://upload.wikimedia.org/wikipedia/commons/8/87/Tabby_cat.jpg"
                    ),
                    "help": "URL of the image to describe.",
                },
            ),
            (
                "--prompt",
                {
                    "default": "Describe this image in one sentence.",
                    "help": "Prompt to accompany the image.",
                },
            ),
        ],
    )
    require_env(args.provider)

    client = LLMClient(args.model, provider=args.provider)
    response = await client.complete(
        [
            UserMessage(
                content=[
                    TextPart(text=args.prompt),
                    ImageContent.from_url(args.image),
                ],
            )
        ]
    )
    print_summary(response)


if __name__ == "__main__":
    asyncio.run(_main())
