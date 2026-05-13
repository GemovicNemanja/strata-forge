"""Response types for LLM calls — both batch and streaming.

``LLMResponse`` is the value an awaited ``LLMClient.complete`` returns.
``ResponseChunk`` is the value yielded by ``LLMClient.stream`` for each
incremental chunk; the final chunk carries the ``finish_reason`` and the
post-call ``usage``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

# Runtime imports — Pydantic needs the actual classes at model-build time to
# resolve field annotations. Keeping them out of a TYPE_CHECKING block.
from forge.llm.messages import ToolCall  # noqa: TC001
from forge.llm.routing import ModelRoute  # noqa: TC001

__all__ = [
    "FinishReason",
    "LLMResponse",
    "ResponseChunk",
    "ToolCallDelta",
    "Usage",
]


type FinishReason = Literal[
    "stop",
    "tool_use",
    "length",
    "content_filter",
    "error",
]


class Usage(BaseModel):
    """Token accounting for a single completion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        """Input + output. Cache reads/writes are reported separately."""
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """The result of a non-streaming LLM call."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    text: str = ""
    tool_calls: list[ToolCall] = []
    finish_reason: FinishReason
    usage: Usage
    cost_usd: float = Field(ge=0)
    route: ModelRoute
    cache_hit: bool = False
    latency_ms: float = Field(default=0.0, ge=0)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class ToolCallDelta(BaseModel):
    """One incremental update to a tool call during streaming.

    Consumers accumulate ``arguments_delta`` strings into a buffer keyed
    by ``index`` until the stream completes, then JSON-parse the final
    buffer. ``id`` and ``name`` arrive once, in the first delta for that
    index; subsequent deltas leave them ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


class ResponseChunk(BaseModel):
    """One chunk emitted during a streaming completion.

    ``delta_text`` is the new text fragment (may be empty when only tool
    calls are streaming). ``finish_reason`` and ``usage`` are present
    only on the final chunk.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    delta_text: str = ""
    delta_tool_calls: list[ToolCallDelta] = []
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
