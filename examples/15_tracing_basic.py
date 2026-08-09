"""Wrap a sync and an async function in `@traced`.

No provider keys needed. When Langfuse isn't configured the wrapper is
a transparent pass-through, so the example runs locally and prints the
return values either way. When `LANGFUSE_PUBLIC_KEY` /
`LANGFUSE_SECRET_KEY` are set, the script also lands a trace for each
call in the configured Langfuse instance.

Usage::

    uv run python examples/15_tracing_basic.py
    LANGFUSE_PUBLIC_KEY=pk... LANGFUSE_SECRET_KEY=sk... \\
        uv run python examples/15_tracing_basic.py
"""

from __future__ import annotations

import asyncio

from strata_forge.core.ids import correlation_id_var
from strata_forge.tracing import get_client, traced


@traced
def sync_demo(x: int, y: int) -> int:
    """Sync function — captured as a top-level Langfuse trace when configured."""
    cid = correlation_id_var.get()
    print(f"  sync_demo running; correlation_id={cid!r}")
    return x + y


@traced(name="async-demo", tags=["example", "v1"])
async def async_demo(message: str) -> str:
    """Async function — same decorator, picks the async wrapper automatically."""
    cid = correlation_id_var.get()
    print(f"  async_demo running; correlation_id={cid!r}")
    return f"echo: {message}"


async def _main() -> None:
    print(f"Langfuse client: {get_client()!r}")
    print()
    print("--- sync_demo(2, 3) ---")
    print(f"  result = {sync_demo(2, 3)}")
    print()
    print("--- await async_demo('hi') ---")
    print(f"  result = {await async_demo('hi')}")


if __name__ == "__main__":
    asyncio.run(_main())
