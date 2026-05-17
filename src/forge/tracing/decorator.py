"""`@traced` decorator — wrap a sync or async function in a Langfuse trace.

Two forms are supported::

    @traced
    async def my_workflow(x: int) -> int:
        ...

    @traced(name="custom-name", tags=["v1", "experiment"])
    def my_pipeline(x: int) -> int:
        ...

While the wrapped function runs, the active Langfuse trace ID is set
on :data:`forge.core.ids.correlation_id_var` so the structlog logger's
``correlation_id`` field carries it automatically. Nested LiteLLM
calls — auto-traced via :func:`forge.tracing.install_litellm_callback`
— appear under the same trace.

When Langfuse isn't configured (or the SDK call to create a trace
fails for any reason), the wrapper is a transparent pass-through: the
wrapped function runs normally with no overhead and no exception
escapes the tracing path. **Tracing failures never break production
code paths.**
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from typing import TYPE_CHECKING, Any, overload

from forge.core.ids import correlation_id_var, set_correlation_id
from forge.tracing.client import get_client

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "traced",
]


def _safe_create_trace(
    client: Any,
    *,
    name: str,
    tags: list[str] | None,
) -> Any:
    """Create a trace, returning ``None`` on any error.

    Langfuse SDK errors (network blip, schema mismatch, etc.) must not
    break the wrapped function. We swallow them here and the wrapper
    falls back to no-trace mode for that call.
    """
    try:
        return client.trace(name=name, tags=tags or [])
    except Exception:
        return None


def _safe_update_trace(trace: Any, **fields: Any) -> None:
    """Apply ``fields`` to ``trace`` via ``trace.update``; swallow errors.

    The defensive ``trace is None`` short-circuit guards against future
    callers; current call sites always pass a non-None trace.
    """
    if trace is None:  # pragma: no cover — defensive guard
        return
    with contextlib.suppress(Exception):
        trace.update(**fields)


@overload
def traced[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def traced[**P, R](
    fn: None = None,
    /,
    *,
    name: str | None = ...,
    tags: list[str] | None = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def traced[**P, R](
    fn: Callable[P, R] | None = None,
    /,
    *,
    name: str | None = None,
    tags: list[str] | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap a function in a Langfuse trace.

    Args:
        fn: When used as the bare ``@traced`` form, the wrapped
            function. Pass ``None`` (or use the parameterized
            ``@traced(...)`` form) to inject options.
        name: Trace name; defaults to ``fn.__name__``.
        tags: Tags attached to the Langfuse trace.

    Returns:
        Either the wrapped callable (bare form) or a decorator that
        applies the options.
    """

    def _wrap(func: Callable[P, R]) -> Callable[P, R]:
        trace_name = name if name is not None else func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                client = get_client()
                if client is None:
                    return await func(*args, **kwargs)

                trace = _safe_create_trace(client, name=trace_name, tags=tags)
                if trace is None:
                    return await func(*args, **kwargs)

                token = set_correlation_id(getattr(trace, "id", None))
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    _safe_update_trace(
                        trace,
                        output={"error": repr(exc)},
                        level="ERROR",
                    )
                    raise
                finally:
                    correlation_id_var.reset(token)
                _safe_update_trace(trace, output={"status": "ok"})
                return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            client = get_client()
            if client is None:
                return func(*args, **kwargs)

            trace = _safe_create_trace(client, name=trace_name, tags=tags)
            if trace is None:
                return func(*args, **kwargs)

            token = set_correlation_id(getattr(trace, "id", None))
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                _safe_update_trace(
                    trace,
                    output={"error": repr(exc)},
                    level="ERROR",
                )
                raise
            finally:
                correlation_id_var.reset(token)
            _safe_update_trace(trace, output={"status": "ok"})
            return result

        return sync_wrapper

    if fn is not None:
        return _wrap(fn)
    return _wrap
