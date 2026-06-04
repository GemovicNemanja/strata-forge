"""Streaming accumulators — turn an async stream of ``ResponseChunk`` into
either incremental state (during streaming) or a final aggregated value
(after the stream completes).

Three building blocks:

- :func:`accumulate_text` joins every chunk's ``delta_text`` into one string.
- :func:`accumulate_tool_calls` rebuilds complete :class:`ToolCall` instances
  from streamed :class:`ToolCallDelta` updates.
- :class:`JSONAccumulator` collects JSON-fragment strings (typically from
  structured-output streaming) and lets callers check completeness and
  parse the final value.

Higher-level helpers in ``forge.llm.client`` wrap these into the public
streaming API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from forge.core.errors import ValidationError
from forge.llm.messages import ToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from forge.llm.responses import ResponseChunk, ToolCallDelta

__all__ = [
    "JSONAccumulator",
    "StreamingToolCallAccumulator",
    "accumulate_text",
    "accumulate_tool_calls",
]


# ---------------------------------------------------------------------------
# Tool-call accumulation
# ---------------------------------------------------------------------------


@dataclass
class _ToolCallBuffer:
    """Mutable scratch space for one streaming tool call."""

    id: str | None = None
    name: str | None = None
    arguments_text: str = ""


async def accumulate_text(chunks: AsyncIterator[ResponseChunk]) -> str:
    """Concatenate every chunk's ``delta_text`` into one string.

    Chunks without text content are silently skipped. The returned string
    is empty when the stream produces no text deltas (e.g. tool-only
    responses).
    """
    parts: list[str] = []
    async for chunk in chunks:
        if chunk.delta_text:
            parts.append(chunk.delta_text)
    return "".join(parts)


def _finalize_buffers(buffers: dict[int, _ToolCallBuffer]) -> list[ToolCall]:
    """Rebuild complete :class:`ToolCall`s from accumulated buffers.

    Calls are returned one per ``index`` in ascending order.

    Raises:
        ValidationError: When a call's ``id`` or ``name`` never arrived,
            or when its accumulated arguments aren't valid JSON resolving
            to an object.
    """
    result: list[ToolCall] = []
    for index in sorted(buffers):
        buf = buffers[index]
        if buf.id is None:
            msg = f"Streaming tool call at index {index} missing its id"
            raise ValidationError(msg)
        if buf.name is None:
            msg = f"Streaming tool call at index {index} missing its name"
            raise ValidationError(msg)
        arguments: dict[str, Any]
        if buf.arguments_text:
            try:
                parsed: Any = json.loads(buf.arguments_text)
            except json.JSONDecodeError as exc:
                msg = (
                    f"Streaming tool call {buf.id!r} (index {index}) has "
                    f"invalid JSON arguments: {exc.msg}"
                )
                raise ValidationError(msg) from exc
            if not isinstance(parsed, dict):
                msg = (
                    f"Streaming tool call {buf.id!r} (index {index}) arguments "
                    f"resolved to {type(parsed).__name__}, expected an object"
                )
                raise ValidationError(msg)
            arguments = cast("dict[str, Any]", parsed)
        else:
            arguments = {}
        result.append(ToolCall(id=buf.id, name=buf.name, arguments=arguments))
    return result


class StreamingToolCallAccumulator:
    """Incrementally rebuild :class:`ToolCall`s from streamed deltas.

    Unlike :func:`accumulate_tool_calls`, which consumes an entire stream
    in one call, this lets a caller feed each chunk's
    :class:`ToolCallDelta` fragments in via :meth:`add` while doing other
    work between chunks (e.g. yielding text deltas to a UI), then rebuild
    the whole calls at an iteration boundary via :meth:`finalize`. The
    multi-turn streaming tool loop in :mod:`forge.llm.client`
    (``LLMClient.stream_tool_loop``) is the primary consumer.
    """

    def __init__(self) -> None:
        self._buffers: dict[int, _ToolCallBuffer] = {}

    def add(self, deltas: Iterable[ToolCallDelta]) -> None:
        """Fold one chunk's tool-call deltas into the per-index buffers.

        The first delta for an ``index`` carries the call's ``id`` and
        ``name``; later deltas append ``arguments_delta`` fragments.
        """
        for delta in deltas:
            buf = self._buffers.setdefault(delta.index, _ToolCallBuffer())
            if delta.id is not None:
                buf.id = delta.id
            if delta.name is not None:
                buf.name = delta.name
            if delta.arguments_delta:
                buf.arguments_text += delta.arguments_delta

    @property
    def has_calls(self) -> bool:
        """Whether any tool-call deltas have been accumulated so far."""
        return bool(self._buffers)

    def finalize(self) -> list[ToolCall]:
        """Rebuild the complete tool calls, one per index in ascending order.

        Raises:
            ValidationError: When a call's ``id`` or ``name`` never
                arrived, or its accumulated arguments aren't valid JSON
                resolving to an object.
        """
        return _finalize_buffers(self._buffers)


async def accumulate_tool_calls(
    chunks: AsyncIterator[ResponseChunk],
) -> list[ToolCall]:
    """Rebuild complete :class:`ToolCall` instances from streamed deltas.

    Streaming providers send each tool call as a sequence of
    :class:`ToolCallDelta` updates keyed by ``index``. The first delta for
    a given index carries the call's ``id`` and ``name``; later deltas
    typically only carry ``arguments_delta`` fragments that concatenate
    into a JSON-encoded argument object. This helper buffers those
    fragments, parses the final JSON, and returns one :class:`ToolCall`
    per index in ascending order.

    Raises:
        ValidationError: When a call's ``id`` or ``name`` never arrived,
            or when its accumulated arguments aren't valid JSON.
    """
    accumulator = StreamingToolCallAccumulator()
    async for chunk in chunks:
        accumulator.add(chunk.delta_tool_calls)
    return accumulator.finalize()


# ---------------------------------------------------------------------------
# Partial-JSON accumulator for structured-output streaming
# ---------------------------------------------------------------------------


class JSONAccumulator:
    """Buffer JSON-fragment strings; check completeness and parse the final value.

    Designed for structured-output streaming, where the model emits text
    chunks that together form one JSON value. Callers feed each chunk in,
    optionally peek at ``is_complete()`` to know when to stop, and finally
    call ``parse()`` once the stream terminates.

    The accumulator deliberately does NOT attempt to coerce partial JSON
    into best-effort values during streaming. Hand-rolling a partial JSON
    parser is error-prone; if you need to surface intermediate state to
    users, render the raw ``text`` instead and parse only at the end.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> None:
        """Append a chunk of JSON-encoded text to the buffer."""
        if chunk:
            self._buffer += chunk

    def reset(self) -> None:
        """Clear the buffer."""
        self._buffer = ""

    @property
    def text(self) -> str:
        """The accumulated JSON text so far. Read-only."""
        return self._buffer

    def is_complete(self) -> bool:
        """Return ``True`` iff the buffer currently holds a parseable JSON value."""
        if not self._buffer.strip():
            return False
        try:
            json.loads(self._buffer)
        except json.JSONDecodeError:
            return False
        return True

    def parse(self) -> Any:
        """Parse the buffer as JSON and return the value.

        Raises:
            ValidationError: When the buffer is empty or doesn't hold a
                valid JSON value. The error message includes the JSON
                parser's position info to aid debugging.
        """
        if not self._buffer.strip():
            msg = "JSONAccumulator buffer is empty"
            raise ValidationError(msg)
        try:
            return json.loads(self._buffer)
        except json.JSONDecodeError as exc:
            msg = f"JSONAccumulator buffer is not valid JSON: {exc.msg} at pos {exc.pos}"
            raise ValidationError(msg) from exc
