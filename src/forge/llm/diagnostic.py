"""NDJSON diagnostic dump — one record per LLM call attempt.

When ``FORGE_DIAGNOSTIC_ENABLED=1`` (via :class:`forge.config.DiagnosticConfig`),
:func:`write_diagnostic_record` appends a JSON line to
``FORGE_DIAGNOSTIC_PATH`` (default ``./forge-diagnostic.ndjson``) for every
completed LLM attempt. The dump is independent of Langfuse — it works
offline and is intended for postmortem debugging, replay, and offline
analysis.

One record per *attempt*, not per "user call": a tool loop with three
iterations emits three records; a fallback chain that traverses two
providers before succeeding emits two records (one error, one success).

Field shape is captured by :class:`DiagnosticRecord`. Callers
(:class:`LLMClient`, mostly) construct records and pass them in;
this module owns serialization and the actual file write.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
from anyio.to_thread import run_sync
from pydantic import BaseModel, ConfigDict, Field

from forge.config import get_settings

__all__ = [
    "DiagnosticRecord",
    "make_error_field",
    "write_diagnostic_record",
]


class DiagnosticRecord(BaseModel):
    """One line in the NDJSON diagnostic dump.

    The record is intentionally provider-agnostic at the schema level:
    every field is plain JSON-serializable, so consumers (replay tools,
    analytics scripts) don't need to import any Forge classes to read
    the dump.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: str = Field(description="ISO-8601 UTC timestamp of when the record was built.")
    correlation_id: str | None = Field(
        default=None,
        description="The active `forge.core.ids.correlation_id`, if set.",
    )
    request_hash: str = Field(
        description=(
            "Provider-agnostic SHA-256 of the request — typically produced "
            "by `forge.llm.cache.cache_key`."
        )
    )
    model: str = Field(description="Logical (registry) model name.")
    provider: str = Field(description="The provider that served (or attempted to serve) the call.")
    provider_model_id: str = Field(
        description="The provider-specific model identifier actually dispatched."
    )
    messages: list[dict[str, Any]] = Field(
        default=[],
        description="Conversation history in wire format at request time.",
    )
    response_text: str = Field(
        default="",
        description="The model's textual output for this attempt (empty on error).",
    )
    tool_calls: list[dict[str, Any]] = Field(
        default=[],
        description="Tool calls emitted in this attempt's response.",
    )
    finish_reason: str | None = Field(
        default=None,
        description="The provider's finish_reason, normalized to Forge's FinishReason set.",
    )
    usage: dict[str, int] | None = Field(
        default=None,
        description="Token usage counters (input, output, cache_read, cache_write).",
    )
    cost_usd: float | None = Field(
        default=None,
        description="Cost of the call in USD, computed against the registry pricing.",
    )
    latency_ms: float | None = Field(
        default=None,
        description="Wall-clock latency of the provider call in milliseconds.",
    )
    cache_hit: bool = Field(
        default=False,
        description="True when the response was served from cache; cost_usd is then 0.",
    )
    error: dict[str, str] | None = Field(
        default=None,
        description=(
            "Populated on failure: {'type': '<ExceptionClassName>', 'message': '...'}. "
            "None on success."
        ),
    )


def make_error_field(exc: BaseException) -> dict[str, str]:
    """Convert an exception into the ``error`` field shape used in records.

    Keeps the record schema simple (string-only) so consumers don't need
    to know about Forge's exception hierarchy to read the dump.
    """
    return {"type": type(exc).__name__, "message": str(exc)}


# Serialize concurrent writes from the same process. anyio.Lock binds to the
# event loop lazily on first acquire, so module-level construction is safe.
_write_lock: anyio.Lock = anyio.Lock()


def _utcnow_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="microseconds")


def _append_sync(out_path: Path, line: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(line)


async def write_diagnostic_record(
    record: DiagnosticRecord,
    *,
    path: Path | str | None = None,
) -> None:
    """Append ``record`` to the NDJSON dump file.

    No-op (returns immediately) when ``FORGE_DIAGNOSTIC_ENABLED`` is
    false. The file is opened in append mode; one record per line; a
    trailing newline is added. The actual write runs on a worker thread
    via :func:`anyio.to_thread.run_sync` so we don't block the event
    loop on filesystem stalls.

    Args:
        record: The :class:`DiagnosticRecord` to serialize.
        path: Override the default destination (mostly for tests).
            ``None`` reads ``FORGE_DIAGNOSTIC_PATH`` from settings.
    """
    if path is None:
        diag = get_settings().diagnostic
        if not diag.enabled:
            return
        out_path = Path(diag.path)
    else:
        # Explicit path bypasses the enabled gate so test fixtures and
        # one-off captures don't need to fiddle with env vars.
        out_path = Path(path)

    line = record.model_dump_json() + "\n"
    async with _write_lock:
        await run_sync(_append_sync, out_path, line)


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Re-exported so callers (e.g. :class:`LLMClient`) building records
    can stamp them without re-importing :mod:`datetime`.
    """
    return _utcnow_iso()
