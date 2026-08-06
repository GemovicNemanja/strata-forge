"""structlog setup with contextvar-based correlation-id injection.

`configure_logging` is called once per process (typically by the CLI entry
point or by tests). After it runs, `get_logger` returns a structlog logger that
emits pretty output to a TTY and one JSON object per line to anything else, and
every record carries the currently-bound `correlation_id` if one is set.

The `traced_span` context manager wraps a unit of work and emits paired
`<name>.start` / `<name>.end` records with elapsed time; on exception it emits
`<name>.error` with the exception type and message before re-raising.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import structlog

from strata_forge.core.ids import correlation_id_var

if TYPE_CHECKING:
    from collections.abc import Generator

    from structlog.types import EventDict, FilteringBoundLogger, Processor, WrappedLogger

__all__ = ["add_correlation_id", "configure_logging", "get_logger", "traced_span"]


def add_correlation_id(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """structlog processor: inject the current correlation_id into every record.

    Exposed as a public helper so users who want to compose their own structlog
    configuration (instead of calling ``configure_logging``) can still pick up
    the contextvar-based correlation-id behavior.
    """
    cid = correlation_id_var.get()
    if cid is not None:
        event_dict["correlation_id"] = cid
    return event_dict


def configure_logging(*, level: str = "INFO", json: bool | None = None) -> None:
    """Configure structlog process-wide.

    Pretty-prints to a TTY; emits one JSON object per line to a pipe or file.
    The current `correlation_id` is auto-injected into every record.

    Args:
        level: Log-level name (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
            ``CRITICAL``). Unknown names fall back to ``INFO``.
        json: Force JSON output (``True``) or pretty output (``False``).
            ``None`` (the default) auto-detects from ``sys.stdout.isatty()``.
    """
    if json is None:
        json = not sys.stdout.isatty()

    log_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        add_correlation_id,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Processor
    if json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a structlog logger, optionally named.

    The returned logger respects whatever level / format `configure_logging`
    most recently set up.
    """
    return structlog.get_logger(name)


@contextmanager
def traced_span(name: str, **fields: Any) -> Generator[None]:
    """Emit paired ``<name>.start`` / ``<name>.end`` records with elapsed time.

    On exception, emits ``<name>.error`` with ``error_type`` / ``error`` /
    ``elapsed_s`` and re-raises. Additional keyword arguments are attached to
    every record emitted for the span.
    """
    logger = get_logger("strata_forge.span")
    logger.info(f"{name}.start", **fields)
    start = time.perf_counter()
    try:
        yield
    except BaseException as exc:
        elapsed = time.perf_counter() - start
        logger.error(
            f"{name}.error",
            elapsed_s=round(elapsed, 6),
            error_type=type(exc).__name__,
            error=str(exc),
            **fields,
        )
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info(f"{name}.end", elapsed_s=round(elapsed, 6), **fields)
