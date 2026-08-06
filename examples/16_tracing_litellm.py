"""Install the LiteLLM Langfuse callback, then make a `LLMClient` call.

This is the production tracing path: a single
``install_litellm_callback()`` at startup auto-traces every subsequent
LiteLLM call — including ones from libraries that don't know about
Forge — without any other code change.

Requires a provider key (for the LLM call) and Langfuse credentials
(otherwise the callback installer no-ops). When Langfuse is configured,
the resulting trace appears in the configured Langfuse instance under
the LiteLLM-default trace name.

Usage::

    uv run python examples/16_tracing_litellm.py --model claude-opus-4-7 --provider anthropic
"""

from __future__ import annotations

import asyncio

from _common import parse_args, print_summary, require_env

from strata_forge.llm import LLMClient, Message
from strata_forge.tracing import (
    get_client,
    install_litellm_callback,
    is_litellm_callback_installed,
)


async def _main() -> None:
    args = parse_args(
        description="LiteLLM call traced via the Langfuse callback.",
        extra_args=[
            (
                "--prompt",
                {
                    "default": "Reply with the single word 'hello'.",
                    "help": "Prompt to send.",
                },
            ),
        ],
    )
    require_env(args.provider)

    installed = install_litellm_callback()
    print("--- tracing setup ---")
    print(f"  Langfuse client:           {get_client()!r}")
    print(f"  callback install result:   {installed}")
    print(f"  callback registered now:   {is_litellm_callback_installed()}")
    if not installed:
        print("  (Langfuse not configured — the call will succeed but no trace is recorded.)")
    print()

    client = LLMClient(args.model, provider=args.provider)
    response = await client.complete([Message.user(args.prompt)])
    print_summary(response)


if __name__ == "__main__":
    asyncio.run(_main())
