"""``traced_span`` — async context manager for nested observations.

Use ``async with traced_span(name)`` inside a `@traced` function (or
anywhere a trace is active via :data:`forge.core.ids.correlation_id_var`)
to open a Langfuse span. The span links to the active trace, records
its wall-clock duration, captures success / error outcome, and yields
the span object so callers can attach additional fields via
``span.update(...)`` if they want.

Spans are flat within their trace in this implementation — sequential
spans inside a single ``@traced`` function appear as siblings under
the trace, not nested within each other. Nesting spans within spans
is a refinement for later; for now keeping the model simple matches
Phase 2.2's scope.

Like the rest of :mod:`forge.tracing`, the context manager is a
transparent no-op when Langfuse isn't configured or span creation
fails — the block runs to completion, span errors are swallowed, and
the wrapped logic propagates exceptions unchanged.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

from forge.core.ids import correlation_id_var
from forge.tracing.client import get_client

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = [
    "traced_span",
]


def _create_span(
    client: Any,
    *,
    name: str,
    trace_id: str | None,
    metadata: dict[str, Any] | None,
) -> Any:
    """Create a Langfuse span; return ``None`` on any error.

    Span creation failure must not break the wrapped block. We swallow
    the exception and the caller falls through to the no-trace path.
    """
    kwargs: dict[str, Any] = {"name": name, "metadata": metadata or {}}
    if trace_id is not None:
        kwargs["trace_id"] = trace_id
    try:
        return client.span(**kwargs)
    except Exception:
        return None


@contextlib.asynccontextmanager  # pyright: ignore[reportDeprecated]
# ^ pyright sees a deprecation on the overload that takes a callable returning
# an async generator (its preferred shape is the @asynccontextmanager-typing
# helper from typing_extensions, but only under some configurations). The
# function is fully supported at runtime in Python 3.14; the ignore is a
# pyright/typeshed quirk, not a deprecation we need to act on.
async def traced_span(
    name: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> AsyncIterator[Any]:
    """Open a Langfuse span under the active trace for the duration of a block.

    Yields the Langfuse span object, or ``None`` when Langfuse isn't
    configured or span creation fails. Callers can attach additional
    fields via ``span.update(...)`` inside the block.

    On exit:

    - **Success**: ``span.end`` records ``output={"status": "ok"}`` plus
      the elapsed milliseconds in metadata.
    - **Exception**: ``span.end`` records the error and re-raises the
      original exception. The error path never masks the real failure.

    ``span.end`` failures are swallowed — observability never breaks
    production.
    """
    client = get_client()
    if client is None:
        yield None
        return

    trace_id = correlation_id_var.get()
    span = _create_span(client, name=name, trace_id=trace_id, metadata=metadata)
    if span is None:
        yield None
        return

    start = time.perf_counter()
    try:
        yield span
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        with contextlib.suppress(Exception):
            span.end(
                output={"error": repr(exc)},
                level="ERROR",
                metadata={"duration_ms": elapsed_ms},
            )
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000
        with contextlib.suppress(Exception):
            span.end(
                output={"status": "ok"},
                metadata={"duration_ms": elapsed_ms},
            )
