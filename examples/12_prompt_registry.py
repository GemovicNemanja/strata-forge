"""Version-track prompts via `PromptRegistry` + `InMemoryPromptStore`.

No provider keys needed. The example puts two versions of the same
template, lists them, retrieves a specific version, and demonstrates
that validation fires before a bad template can reach the store.

Usage::

    uv run python examples/12_prompt_registry.py
"""

from __future__ import annotations

import asyncio

from strata_forge.prompts import (
    InMemoryPromptStore,
    PromptRegistry,
    PromptTemplate,
    PromptValidationError,
)


async def _main() -> None:
    registry = PromptRegistry(InMemoryPromptStore())

    v1 = await registry.put(
        PromptTemplate(
            name="greet",
            stable_section="You are friendly.",
            dynamic_section="Hi {{ name }}",
            dynamic_variables=("name",),
        )
    )
    print(f"stored greet @ version {v1}")

    v2 = await registry.put(
        PromptTemplate(
            name="greet",
            stable_section="You are friendly and concise.",
            dynamic_section="Hi {{ name }}, what's up?",
            dynamic_variables=("name",),
        )
    )
    print(f"stored greet @ version {v2}")

    print(f"versions: {await registry.versions('greet')}")
    print(f"names:    {await registry.list_names()}")

    latest = await registry.get("greet")
    print(f"latest stable_section: {latest.stable_section!r}")

    earliest = await registry.get("greet", v1)
    print(f"v1 stable_section:    {earliest.stable_section!r}")

    print()
    print("--- validation: a bad template never reaches the store ---")
    bad = PromptTemplate(
        name="bad",
        stable_section="",
        dynamic_section="Hi {{ name }}",  # referenced but not declared
    )
    try:
        await registry.put(bad)
    except PromptValidationError as exc:
        print(f"rejected: {exc}")
    print(f"names after rejection: {await registry.list_names()}")


if __name__ == "__main__":
    asyncio.run(_main())
