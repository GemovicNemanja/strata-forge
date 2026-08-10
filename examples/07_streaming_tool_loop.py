"""Streaming tool loop: ``stream_tool_loop`` yields the tool conversation live.

Where ``06_tool_loop.py`` buffers the whole conversation into one final
response, ``stream_tool_loop`` is an async generator that yields a typed
``LoopEvent`` for each thing that happens — assistant text deltas, each
tool call as it fires, each tool result, and exactly one terminal event
(``Done`` or ``LoopError``). This is the shape a UI consumes to show the
model working in real time.

Usage::

    uv run python examples/07_streaming_tool_loop.py            # claude-haiku-4-5 @ anthropic
    uv run python examples/07_streaming_tool_loop.py --model gpt-5.5 --provider openai
"""

from __future__ import annotations

import asyncio

from _common import parse_args, require_env
from pydantic import BaseModel, Field

from strata_forge.llm import (
    Done,
    IterationStart,
    LLMClient,
    LoopError,
    Message,
    TextDelta,
    ToolCallStarted,
    ToolResult,
    tool,
)


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
    args = parse_args(description="Streaming tool-loop demo (stream_tool_loop).")
    require_env(args.provider)

    client = LLMClient(args.model, provider=args.provider)

    # NOTE: stream_tool_loop is a true async generator — iterate it directly;
    # do NOT `await` the call first (unlike client.stream(...)).
    events = client.stream_tool_loop(
        [
            Message.user(
                "What's the weather in Tokyo? Convert the temperature to Fahrenheit "
                "and explain it in one sentence."
            )
        ],
        tools=[get_weather, celsius_to_fahrenheit],
        max_iterations=6,
    )

    async for event in events:
        match event:
            case IterationStart(index=index):
                print(f"\n--- turn {index} ---")
            case TextDelta(text=text):
                print(text, end="", flush=True)
            case ToolCallStarted(name=name, arguments=arguments):
                print(f"\n[tool ->] {name}({arguments})")
            case ToolResult(name=name, content=content, is_error=is_error):
                tag = "tool !!" if is_error else "tool <-"
                print(f"[{tag}] {name} -> {content}")
            case Done(finish_reason=finish_reason, usage=usage):
                tokens = usage.total_tokens if usage else "n/a"
                print(f"\n\n[done] finish_reason={finish_reason} tokens={tokens}")
            case LoopError(message=message, error_type=error_type, exceeded_max_iterations=capped):
                detail = " (max_iterations hit)" if capped else ""
                print(f"\n\n[error{detail}] {error_type}: {message}")


if __name__ == "__main__":
    asyncio.run(_main())
