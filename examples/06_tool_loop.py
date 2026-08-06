"""Multi-turn tool loop: ``run_tool_loop`` drives the conversation to a final answer.

The loop runs a completion, invokes any requested tools, appends the
tool results to the history, and repeats until the model exits tool-use
mode or ``max_iterations`` is hit.

Usage::

    uv run python examples/06_tool_loop.py --model gpt-5.5 --provider openai
"""

from __future__ import annotations

import asyncio

from _common import parse_args, print_summary, require_env
from pydantic import BaseModel, Field

from strata_forge.llm import LLMClient, Message, tool


class WeatherArgs(BaseModel):
    location: str = Field(..., description="City, country.")


class ConvertArgs(BaseModel):
    celsius: float = Field(..., description="Temperature in Celsius.")


@tool
async def get_weather(args: WeatherArgs) -> dict[str, float | str]:
    """Get the current weather for a city. Returns degrees Celsius."""
    return {"location": args.location, "temp_c": 18.0, "conditions": "partly cloudy"}


@tool
async def celsius_to_fahrenheit(args: ConvertArgs) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return args.celsius * 9 / 5 + 32


async def _main() -> None:
    args = parse_args(
        description="Multi-turn tool-loop demo (run_tool_loop).",
    )
    require_env(args.provider)

    client = LLMClient(args.model, provider=args.provider)
    response = await client.run_tool_loop(
        [
            Message.user(
                "What's the weather in Tokyo? Convert the temperature to Fahrenheit "
                "and explain it in one sentence."
            )
        ],
        tools=[get_weather, celsius_to_fahrenheit],
        max_iterations=6,
    )
    print_summary(response)


if __name__ == "__main__":
    asyncio.run(_main())
