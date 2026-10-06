"""Live smoke test of every registered OpenAI route that speaks the Responses API.

For each model, two legs through :meth:`LLMClient.stream_tool_loop`, the path a chat agent
takes:

1. A prompt that must call the declaration-only tool ``echo_value``. The loop has to suspend
   with :class:`PendingToolCalls` carrying the turn's provider items (encrypted reasoning,
   the call reference).
2. The suspended conversation, round-tripped through JSON the way a continuation token
   carries it, plus the tool result. The model has to answer with the echoed value, which
   proves the replayed items were accepted.

The key is read from ``OPENAI_API_KEY`` and never printed (it is also scrubbed from any
error text). The script spends real tokens; ``gpt-5.5-pro`` is skipped unless
``--include-pro`` is given because it is slow and priced at $30 / $180 per million tokens.

Usage::

    OPENAI_API_KEY=... uv run python scripts/smoke_responses.py
    OPENAI_API_KEY=... uv run python scripts/smoke_responses.py --models gpt-6.1-sol gpt-6-luna

Exits non-zero when any model fails either leg.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
from dataclasses import dataclass, field

from pydantic import SecretStr

from strata_forge.llm.client import LLMClient
from strata_forge.llm.loop_events import Done, LoopError, PendingToolCalls, TextDelta
from strata_forge.llm.messages import (
    AssistantMessage,
    Message,
    ToolResultMessage,
)
from strata_forge.llm.providers import OpenAIProvider
from strata_forge.llm.providers.config import OpenAIConfig
from strata_forge.llm.registry import registry
from strata_forge.llm.tools import ToolDeclaration

_ECHO = ToolDeclaration(
    name="echo_value",
    description="Echo a value back to the caller. Always call it when asked to.",
    parameters={
        "type": "object",
        "properties": {"value": {"type": "string", "description": "The value to echo."}},
        "required": ["value"],
        "additionalProperties": False,
    },
)
_SLOW_MODELS = frozenset({"gpt-5.5-pro"})


@dataclass
class _Row:
    model: str
    leg1: str = "-"
    items: str = "-"
    delta_bytes: int = 0
    leg2: str = "-"
    tokens: str = "-"
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.leg1 == "pending" and self.leg2 == "stop"


def _responses_models(include_pro: bool) -> list[str]:
    names: list[str] = []
    for model in registry.list_models(vendor="openai"):
        route = model.route_for("openai")
        if route is None or route.wire_api != "responses":
            continue
        if model.name in _SLOW_MODELS and not include_pro:
            continue
        names.append(model.name)
    return names


async def _smoke(model: str, key: str, max_output_tokens: int) -> _Row:
    row = _Row(model=model)
    provider = OpenAIProvider(OpenAIConfig(api_key=SecretStr(key)))
    client = LLMClient(model, provider="openai", provider_clients={"openai": provider})
    token = f"pong-{secrets.token_hex(4)}"
    prompt = [
        Message.system("You are a test harness. Follow the instructions exactly."),
        Message.user(f"Call echo_value with value {token!r}, then reply with the tool's result."),
    ]

    pending: PendingToolCalls | None = None
    async for event in client.stream_tool_loop(
        prompt, tools=[_ECHO], max_iterations=2, max_tokens=max_output_tokens
    ):
        if isinstance(event, PendingToolCalls):
            pending = event
        elif isinstance(event, LoopError):
            row.errors.append(f"leg 1: {event.error_type}: {event.message}")
        elif isinstance(event, Done):
            row.leg1 = f"done:{event.finish_reason}"
    if pending is None:
        if not row.errors:
            row.errors.append("leg 1 did not call echo_value")
        return row
    row.leg1 = "pending"
    turn = pending.messages[0]
    if isinstance(turn, AssistantMessage) and turn.provider_items is not None:
        row.items = ",".join(item.kind for item in turn.provider_items.items)
    else:
        row.errors.append("leg 1 suspended without provider items")

    # Round-trip the delta through JSON, as a continuation token does between legs.
    blob = json.dumps([m.model_dump(mode="json") for m in pending.messages])
    row.delta_bytes = len(blob.encode("utf-8"))
    restored = [
        AssistantMessage.model_validate(m)
        if m["role"] == "assistant"
        else ToolResultMessage.model_validate(m)
        for m in json.loads(blob)
    ]
    results = [ToolResultMessage(tool_call_id=c.id, content=token) for c in pending.calls]

    answer: list[str] = []
    async for event in client.stream_tool_loop(
        [*prompt, *restored, *results],
        tools=[_ECHO],
        max_iterations=1,
        max_tokens=max_output_tokens,
    ):
        if isinstance(event, TextDelta):
            answer.append(event.text)
        elif isinstance(event, Done):
            row.leg2 = event.finish_reason
            if event.usage is not None:
                row.tokens = f"{event.usage.input_tokens}/{event.usage.output_tokens}"
        elif isinstance(event, LoopError):
            row.errors.append(f"leg 2: {event.error_type}: {event.message}")
        elif isinstance(event, PendingToolCalls):
            row.leg2 = "pending-again"
    if row.leg2 == "stop" and token not in "".join(answer):
        row.errors.append("leg 2 answer does not contain the echoed value")
    return row


def _scrub(text: str, key: str) -> str:
    return text.replace(key, "<OPENAI_API_KEY>") if key else text


async def _main(models: list[str], key: str, max_output_tokens: int) -> int:
    rows: list[_Row] = []
    for model in models:
        try:
            row = await _smoke(model, key, max_output_tokens)
        except Exception as exc:
            row = _Row(model=model, errors=[f"{type(exc).__name__}: {exc}"])
        rows.append(row)

    print("model | leg1 | provider items | delta bytes | leg2 | in/out tokens | result")
    for row in rows:
        result = "ok" if row.ok else "FAIL"
        print(
            f"{row.model} | {row.leg1} | {row.items} | {row.delta_bytes} | {row.leg2} | "
            f"{row.tokens} | {result}"
        )
        for error in row.errors:
            print(f"    {_scrub(error, key)}")
    return 0 if all(row.ok for row in rows) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--models", nargs="*", help="Registry names (default: every route)")
    parser.add_argument("--include-pro", action="store_true", help="Also run gpt-5.5-pro")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=25_000,
        help="max_output_tokens per turn; it bounds reasoning AND the answer",
    )
    args = parser.parse_args()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        sys.exit(2)
    models = args.models or _responses_models(args.include_pro)
    sys.exit(asyncio.run(_main(models, key, args.max_output_tokens)))


if __name__ == "__main__":
    main()
