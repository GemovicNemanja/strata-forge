"""Render a `PromptTemplate` and send the messages through `LLMClient`.

This is the typical end-to-end path: declare a versioned prompt with a
stable persona, render it with per-call variables, hand the result to
the LLM client. Provider keys required.

Usage::

    uv run python examples/13_prompt_with_llm.py --model claude-opus-4-7 --provider anthropic
"""

from __future__ import annotations

import asyncio

from _common import parse_args, print_summary, require_env

from forge.llm import LLMClient
from forge.prompts import PromptTemplate, render


async def _main() -> None:
    args = parse_args(
        description="Prompt-driven LLM call: render a template then complete.",
        extra_args=[
            (
                "--persona",
                {
                    "default": "a concise physicist",
                    "help": "Stable variable — role/persona for the model.",
                },
            ),
            (
                "--topic",
                {
                    "default": "entropy",
                    "help": "Dynamic variable — what to explain.",
                },
            ),
        ],
    )
    require_env(args.provider)

    template = PromptTemplate(
        name="explainer",
        stable_section=(
            "You are {{ persona }}. Always think step by step before "
            "answering. Be concise; cite sources when relevant."
        ),
        dynamic_section="In one paragraph, explain: {{ topic }}",
        stable_variables=("persona",),
        dynamic_variables=("topic",),
    )
    rendered = render(
        template,
        {"persona": args.persona, "topic": args.topic},
        model=args.model,
    )

    print(f"--- prompt cache hints (target: {args.model}@{args.provider}) ---")
    print(f"cache_stable_prefix:   {rendered.cache_hints.cache_stable_prefix}")
    print(f"stable_token_estimate: {rendered.cache_hints.stable_token_estimate}")
    print()

    client = LLMClient(args.model, provider=args.provider)
    response = await client.complete(rendered.messages)
    print_summary(response)


if __name__ == "__main__":
    asyncio.run(_main())
