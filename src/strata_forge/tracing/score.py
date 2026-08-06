"""Attach grader feedback to a Langfuse trace or observation.

A "score" in Langfuse is a named, typed value attached to a trace or
to a specific observation (span) inside a trace. Scores are how the
eval module (Phase 2.4) will record grader output —
``score_trace(trace_id, "helpfulness", 4.2)`` writes the value
into Langfuse where it's queryable for analytics and CI eval gates.

Both helpers are async to match the rest of :mod:`strata_forge.tracing`.
When Langfuse isn't configured (no client) or the call fails, the
function is a silent no-op — scoring is best-effort observability, it
never breaks the production code path.

Score values may be numeric (``int`` / ``float``), categorical
(``str``), or boolean. Langfuse infers the wire-format data type from
the Python type unless the caller supplies ``data_type`` explicitly.
"""

from __future__ import annotations

import contextlib
from typing import Any

from strata_forge.tracing.client import get_client

__all__ = [
    "ScoreValue",
    "score_observation",
    "score_trace",
]


type ScoreValue = float | int | str | bool


async def score_trace(
    trace_id: str,
    name: str,
    value: ScoreValue,
    *,
    comment: str | None = None,
    data_type: str | None = None,
) -> None:
    """Attach a score to a Langfuse trace.

    Args:
        trace_id: Langfuse trace ID — typically the value of
            :data:`strata_forge.core.ids.correlation_id_var` while a `@traced`
            function is running.
        name: What's being scored (e.g. ``"helpfulness"``,
            ``"accuracy"``). Reused across runs so Langfuse can chart
            distributions over time.
        value: The score — numeric, categorical, or boolean. Langfuse
            infers the data type from the Python type unless
            ``data_type`` overrides.
        comment: Optional free-form note explaining the score (the
            grader's reasoning, for example).
        data_type: Optional explicit Langfuse data type
            (``"NUMERIC"``, ``"CATEGORICAL"``, ``"BOOLEAN"``). Most
            callers omit; the SDK infers correctly.

    Silently no-ops when Langfuse isn't configured. Failures are
    swallowed via :func:`contextlib.suppress` — scoring never breaks
    the calling code.
    """
    client = get_client()
    if client is None:
        return

    kwargs: dict[str, Any] = {"trace_id": trace_id, "name": name, "value": value}
    if comment is not None:
        kwargs["comment"] = comment
    if data_type is not None:
        kwargs["data_type"] = data_type

    # Langfuse SDK v4 renamed `score` → `create_score`.
    with contextlib.suppress(Exception):
        client.create_score(**kwargs)


async def score_observation(
    observation_id: str,
    name: str,
    value: ScoreValue,
    *,
    trace_id: str | None = None,
    comment: str | None = None,
    data_type: str | None = None,
) -> None:
    """Attach a score to a specific observation (span) within a trace.

    Same shape and semantics as :func:`score_trace` but targets one
    observation rather than the whole trace. Supply ``trace_id`` when
    you know it (Langfuse uses it to expedite scoring); otherwise the
    backend resolves it from ``observation_id``.
    """
    client = get_client()
    if client is None:
        return

    kwargs: dict[str, Any] = {
        "observation_id": observation_id,
        "name": name,
        "value": value,
    }
    if trace_id is not None:
        kwargs["trace_id"] = trace_id
    if comment is not None:
        kwargs["comment"] = comment
    if data_type is not None:
        kwargs["data_type"] = data_type

    # Langfuse SDK v4 renamed `score` → `create_score`.
    with contextlib.suppress(Exception):
        client.create_score(**kwargs)
