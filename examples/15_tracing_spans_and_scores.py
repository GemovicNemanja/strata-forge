"""Nest `traced_span` inside `@traced`, then score the trace.

The script demonstrates the full local-side observability story:

- ``@traced`` opens a top-level trace.
- ``traced_span`` blocks inside the function open named child
  observations linked to the trace via ``correlation_id_var``.
- ``record_numeric_metric`` and ``score_trace`` attach measurement
  and grader output to the trace by ID.

When Langfuse isn't configured, every call is a no-op; the script
runs to completion either way. Output prints the trace ID Forge would
publish so the wiring is visible.

Usage::

    uv run python examples/15_tracing_spans_and_scores.py
"""

from __future__ import annotations

import asyncio

from forge.core.ids import correlation_id_var
from forge.tracing import (
    record_numeric_metric,
    score_trace,
    traced,
    traced_span,
)


@traced(name="example-pipeline", tags=["example"])
async def pipeline(query: str) -> str:
    trace_id = correlation_id_var.get()
    print(f"  pipeline: trace_id={trace_id!r}")

    async with traced_span("preprocess", metadata={"step": "lowercase"}):
        normalized = query.lower()
        await asyncio.sleep(0.01)

    async with traced_span("look-up", metadata={"step": "search"}):
        # Pretend we hit an index here.
        await asyncio.sleep(0.02)
        hits = ["doc_1", "doc_2"]

    # Record a per-call numeric metric (latency would normally live on
    # the span; this is for shape).
    if trace_id is not None:
        await record_numeric_metric(
            "hits_returned", len(hits), trace_id=trace_id, comment="top-k=2"
        )

    answer = f"Found {len(hits)} docs for {normalized!r}"

    # The "grader" in a real eval would score after the call; we do it
    # inline here for demonstration.
    if trace_id is not None:
        await score_trace(
            trace_id,
            "helpfulness",
            4.5,
            comment="strong match in both docs",
        )

    return answer


async def _main() -> None:
    print("--- pipeline('Where is Tokyo?') ---")
    answer = await pipeline("Where is Tokyo?")
    print(f"  answer = {answer!r}")


if __name__ == "__main__":
    asyncio.run(_main())
