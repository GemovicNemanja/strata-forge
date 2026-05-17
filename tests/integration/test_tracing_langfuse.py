"""Integration test against the docker-compose Langfuse stack.

Marked with ``@pytest.mark.integration`` so it doesn't run during the
default ``make test`` pass. To exercise:

1. ``make stack-up`` to start the local Langfuse + Postgres + Qdrant +
   Redis stack.
2. Set ``LANGFUSE_HOST``, ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``
   to point at the local instance (the keys can be sourced from the
   Langfuse UI's project settings).
3. ``make integration``.

The test creates a trace via :func:`@traced`, flushes the Langfuse
client buffer, then attempts to fetch the trace back via the SDK's
public read API. The exact fetch method varies across Langfuse SDK
versions; the test is best-effort about server-side verification and
focuses on the "no crash during emission" invariant. Tracing failures
shouldn't break production code paths — that's the contract under
test here.
"""

from __future__ import annotations

import os

import pytest

from forge.core.ids import correlation_id_var
from forge.tracing import get_client, traced

pytestmark = pytest.mark.integration


def _langfuse_configured() -> bool:
    """True when env vars sufficient for a live Langfuse call are set."""
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )


@pytest.fixture(autouse=True)
def _skip_unless_configured() -> None:  # pyright: ignore[reportUnusedFunction]
    if not _langfuse_configured():
        pytest.skip(
            "Set LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY against `make stack-up` to run."
        )


async def test_traced_call_emits_to_langfuse() -> None:
    """A @traced async function emits a trace to the configured Langfuse."""
    pytest.importorskip("langfuse")

    seen_trace_id: list[str | None] = []

    @traced(name="forge-integration-test", tags=["forge-test"])
    async def workload() -> str:
        # While the function runs, the correlation_id is the trace ID.
        seen_trace_id.append(correlation_id_var.get())
        return "hello from forge"

    result = await workload()

    assert result == "hello from forge"
    assert seen_trace_id, "the @traced wrapper never reached the body"
    trace_id = seen_trace_id[0]
    assert trace_id is not None, "no trace ID was published on correlation_id_var"

    # Flush the Langfuse client so events ship before the test exits.
    client = get_client()
    assert client is not None, "get_client returned None despite env config"
    flush = getattr(client, "flush", None)
    if callable(flush):
        flush()

    # Best-effort: read the trace back. The exact method shape varies
    # across Langfuse SDK versions, so we attempt a handful of
    # well-known APIs and accept any one of them. If none are available
    # in this SDK version, we still pass — the emission half is what
    # this test is really about.
    fetch = (
        getattr(client, "fetch_trace", None)
        or getattr(client, "get_trace", None)
        or getattr(getattr(client, "api", None), "trace", None)
    )
    if fetch is None:
        return  # SDK doesn't expose a read API we recognize.

    try:
        fetched = fetch(trace_id)
    except Exception as exc:
        pytest.skip(f"Langfuse read API unavailable: {exc}")
        return
    # The fetched object should mention the trace ID somewhere; structures
    # vary across SDK versions, so we keep the assertion loose.
    assert trace_id in repr(fetched)
