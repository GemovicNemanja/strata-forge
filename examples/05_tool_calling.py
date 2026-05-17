"""Single-call tool surface: the model emits a tool call; the example invokes it.

``run_tool_loop`` (example 06) is the convenient multi-turn API; this
demo exposes the underlying surface where the caller decides whether to
invoke each tool.

Usage::

    uv run python examples/05_tool_calling.py --model claude-opus-4-7 --provider anthropic
"""

from __future__ import annotations

import asyncio

from _common import parse_args, print_summary, require_env
from pydantic import BaseModel, Field

from forge.llm import LLMClient, Message, tool


class WeatherArgs(BaseModel):
    location: str = Field(..., description="City, country.")
    units: str = Field("celsius", description="Either 'celsius' or 'fahrenheit'.")


@tool
async def get_weather(args: WeatherArgs) -> dict[str, object]:
    """Get the current weather for a location."""
    # Dummy — pretend we hit a weather API.
    return {"temp": 18, "units": args.units, "conditions": "partly cloudy"}


async def _main() -> None:
    args = parse_args(
        description="Single-call tool demo — surface tool calls without auto-invoke.",
    )
    require_env(args.provider)

    client = LLMClient(args.model, provider=args.provider)
    response = await client.complete(
        [Message.user("What's the weather like in Tokyo right now?")],
        tools=[get_weather],
    )

    if response.finish_reason == "tool_use":
        print("--- model requested tool call(s) ---")
        for call in response.tool_calls:
            print(f"  {call.name}({call.arguments})")
            result = await get_weather.invoke(call.arguments)
            print(f"  -> {result}")
    else:
        print("--- model answered without calling a tool ---")
    print()
    print_summary(response, label="metrics")


if __name__ == "__main__":
    asyncio.run(_main())
