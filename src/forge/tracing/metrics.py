"""Record application metrics on a Langfuse trace or observation.

Distinct in intent from :mod:`forge.tracing.score`: metrics measure
*facts* about a call — token count, latency, cost, which model was
selected — rather than grader judgments about quality. The underlying
mechanism is the same Langfuse score API with the wire-format
``data_type`` set explicitly, so consumers (queries, dashboards,
alerts) can filter on data type to separate "what happened" from "how
well it went."

Both helpers are async, silent no-ops when Langfuse isn't configured,
and swallow client errors via :func:`contextlib.suppress` — metric
recording is best-effort observability and never breaks the caller.
Calling either without a ``trace_id`` *or* an ``observation_id`` is a
programmer error and raises :class:`ValueError` even when Langfuse
isn't configured, so the bug surfaces in dev rather than going silent
in production.
"""

from __future__ import annotations

import contextlib
from typing import Any

from forge.tracing.client import get_client

__all__ = [
    "record_categorical_metric",
    "record_numeric_metric",
]


def _target_kwargs(
    *,
    trace_id: str | None,
    observation_id: str | None,
) -> dict[str, Any]:
    """Build the trace/observation target kwargs and validate the pair.

    Raises :class:`ValueError` when neither is supplied — the metric
    has nowhere to attach. Validation runs unconditionally so the bug
    surfaces in dev (no Langfuse) instead of silently dropping data
    only in production (Langfuse configured).
    """
    if trace_id is None and observation_id is None:
        msg = "record_*_metric requires either trace_id or observation_id"
        raise ValueError(msg)
    kwargs: dict[str, Any] = {}
    if trace_id is not None:
        kwargs["trace_id"] = trace_id
    if observation_id is not None:
        kwargs["observation_id"] = observation_id
    return kwargs


async def record_numeric_metric(
    name: str,
    value: float | int,
    *,
    trace_id: str | None = None,
    observation_id: str | None = None,
    comment: str | None = None,
) -> None:
    """Record a numeric application metric (tokens, latency, cost, …).

    Args:
        name: Metric name (e.g. ``"input_tokens"``, ``"latency_ms"``,
            ``"cost_usd"``). Reused across runs so Langfuse can chart
            over time.
        value: Numeric value.
        trace_id: Attach to this trace. Required unless
            ``observation_id`` is set.
        observation_id: Attach to this observation. Required unless
            ``trace_id`` is set.
        comment: Optional free-form note (units, computation
            description, …).

    Raises:
        ValueError: When neither ``trace_id`` nor ``observation_id``
            is supplied.
    """
    target = _target_kwargs(trace_id=trace_id, observation_id=observation_id)
    client = get_client()
    if client is None:
        return

    kwargs: dict[str, Any] = {
        **target,
        "name": name,
        "value": value,
        "data_type": "NUMERIC",
    }
    if comment is not None:
        kwargs["comment"] = comment

    with contextlib.suppress(Exception):
        client.score(**kwargs)


async def record_categorical_metric(
    name: str,
    category: str,
    *,
    trace_id: str | None = None,
    observation_id: str | None = None,
    comment: str | None = None,
) -> None:
    """Record a categorical application metric (model, route, …).

    Args:
        name: Metric name (e.g. ``"model"``, ``"provider"``,
            ``"finish_reason"``).
        category: The discrete category value as a string.
        trace_id / observation_id / comment: Same as
            :func:`record_numeric_metric`.

    Raises:
        ValueError: When neither ``trace_id`` nor ``observation_id``
            is supplied.
    """
    target = _target_kwargs(trace_id=trace_id, observation_id=observation_id)
    client = get_client()
    if client is None:
        return

    kwargs: dict[str, Any] = {
        **target,
        "name": name,
        "value": category,
        "data_type": "CATEGORICAL",
    }
    if comment is not None:
        kwargs["comment"] = comment

    with contextlib.suppress(Exception):
        client.score(**kwargs)
