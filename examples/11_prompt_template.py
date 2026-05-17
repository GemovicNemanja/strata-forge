"""Build a `PromptTemplate`, render it, and print the resulting messages.

No provider keys needed — rendering is a pure local transformation. The
example demonstrates the stable/dynamic split that drives provider
prompt caching: the system prompt + persona live in `stable_section`,
the per-call question in `dynamic_section`.

Usage::

    uv run python examples/11_prompt_template.py
    uv run python examples/11_prompt_template.py --topic "the speed of light"
"""

from __future__ import annotations

import argparse

from forge.prompts import PromptTemplate, render


def _main() -> None:
    parser = argparse.ArgumentParser(description="Render a prompt template demo.")
    parser.add_argument(
        "--persona",
        default="a concise physicist",
        help="Stable variable — the role/persona for the model.",
    )
    parser.add_argument(
        "--topic",
        default="entropy",
        help="Dynamic variable — what the user is asking about.",
    )
    args = parser.parse_args()

    template = PromptTemplate(
        name="explainer",
        stable_section=(
            "You are {{ persona }}. Always think step by step before "
            "answering. Be concise; cite sources when relevant."
        ),
        dynamic_section="In one paragraph, explain: {{ topic }}",
        stable_variables=("persona",),
        dynamic_variables=("topic",),
        description="A simple explainer prompt with a stable persona.",
    )
    result = render(template, {"persona": args.persona, "topic": args.topic})

    print("--- messages ---")
    for i, msg in enumerate(result.messages):
        role = type(msg).__name__.removesuffix("Message").lower()
        print(f"[{i}] role={role}")
        print(f"    {msg.content}")
    print()
    print("--- cache hints ---")
    print(f"cache_stable_prefix:   {result.cache_hints.cache_stable_prefix}")
    print(f"stable_token_estimate: {result.cache_hints.stable_token_estimate}")
    print(f"stable_digest:         {result.cache_hints.stable_digest[:16]}...")


if __name__ == "__main__":
    _main()
