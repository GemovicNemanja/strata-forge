"""Event types yielded by :meth:`strata_forge.llm.client.LLMClient.stream_tool_loop`.

The streaming tool loop emits a flat stream of typed events that together
describe a multi-iteration tool-use conversation as it happens: assistant
text deltas, the tool calls the model requested, the results fed back, the
boundaries between iterations, and exactly one terminal event. Consumers
(e.g. an SSE bridge) switch on the event type and forward each as a named
event; they never re-run the loop.

These mirror the :mod:`strata_forge.llm.responses` conventions: frozen Pydantic
models with ``extra="forbid"``. :data:`LoopEvent` is the discriminated
union; every concrete event carries a ``type`` literal that doubles as the
discriminator (and as the wire event name for SSE consumers).

Terminal-event contract: a run ends with **exactly one** of :class:`Done`
(the model exited tool-use mode and produced an answer),
:class:`PendingToolCalls` (the model called declaration-only tools the
caller must execute out-of-band before resuming), or :class:`LoopError`
(a provider/transport error, a malformed streamed tool call, or the
iteration cap being hit). See ``stream_tool_loop`` for the raise-vs-emit
policy that decides which failures raise synchronously versus surface as
a terminal :class:`LoopError`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

# Runtime imports — Pydantic needs the actual classes at model-build time to
# resolve field annotations. Keeping them out of a TYPE_CHECKING block.
from strata_forge.llm.messages import AssistantMessage, ToolCall, ToolResultMessage  # noqa: TC001
from strata_forge.llm.responses import FinishReason, Usage  # noqa: TC001

__all__ = [
    "Done",
    "IterationStart",
    "LoopError",
    "LoopEvent",
    "PendingToolCalls",
    "TextDelta",
    "ToolCallStarted",
    "ToolResult",
]


class _BaseEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class IterationStart(_BaseEvent):
    """Marks the boundary at the top of each loop iteration.

    ``index`` is 0-based: the first model turn is iteration 0. Consumers
    use this to group the text/tool events that follow into one "step".
    """

    type: Literal["iteration_start"] = "iteration_start"
    index: int


class TextDelta(_BaseEvent):
    """An incremental fragment of assistant text from the current iteration."""

    type: Literal["text_delta"] = "text_delta"
    text: str
    iteration: int


class ToolCallStarted(_BaseEvent):
    """A fully-assembled tool call the model requested this iteration.

    Emitted once per call AFTER the iteration's stream completes and the
    streamed ``ToolCallDelta`` fragments have been accumulated into a whole
    call (``id`` + ``name`` + parsed ``arguments``) — never with partial
    arguments. Exactly one :class:`ToolResult` with the matching ``id``
    follows — unless the call targets a
    :class:`~strata_forge.llm.tools.ToolDeclaration`, in which case the run
    suspends with a terminal :class:`PendingToolCalls` carrying it.
    """

    type: Literal["tool_call_started"] = "tool_call_started"
    id: str
    name: str
    arguments: dict[str, object]
    iteration: int


class ToolResult(_BaseEvent):
    """The result of executing one tool call, fed back to the model.

    ``is_error=True`` covers an unregistered tool, an arguments-validation
    failure, or the tool function raising — all of which are stringified
    into ``content`` and fed back to the model rather than raised (mirrors
    ``run_tool_loop``).
    """

    type: Literal["tool_result"] = "tool_result"
    id: str
    name: str
    content: str
    is_error: bool
    iteration: int


class PendingToolCalls(_BaseEvent):
    """Terminal event: the model called tools the caller must execute.

    Emitted when the current turn requested at least one
    :class:`~strata_forge.llm.tools.ToolDeclaration`-targeted call. Any
    executable :class:`~strata_forge.llm.tools.Tool` calls in the same turn were
    already invoked (in model call order) and their :class:`ToolResult`
    events emitted; ``calls`` carries only the unexecuted
    declaration-targeted calls, in model order.

    ``messages`` is the conversation delta this run appended — the
    assistant turns plus the tool results already executed, everything
    after the caller's input. To resume, execute the pending calls
    out-of-band and re-invoke ``stream_tool_loop`` with
    ``input + messages + one ToolResultMessage per pending call``.

    ``iterations_used`` (== ``iteration + 1``: the suspended turn consumed
    a model call) supports cross-run budgeting — pass the remaining budget
    as ``max_iterations`` on resume. ``usage`` is the suspended turn's
    streamed usage when the provider reported it (per-turn, not a sum —
    same accounting as :class:`Done`).
    """

    type: Literal["pending_tool_calls"] = "pending_tool_calls"
    calls: list[ToolCall]
    messages: list[AssistantMessage | ToolResultMessage]
    iteration: int
    iterations_used: int
    usage: Usage | None = None


class Done(_BaseEvent):
    """Terminal event: the model exited tool-use mode and produced an answer.

    ``finish_reason`` is the final iteration's reason (``"stop"`` /
    ``"length"`` / ``"content_filter"``). ``usage`` is the final
    iteration's streamed usage when the provider reported it (``None``
    otherwise) — NOT a sum across iterations; cumulative accounting is the
    caller's job via tracing, exactly as ``run_tool_loop`` / ``AgentResult``
    document.
    """

    type: Literal["done"] = "done"
    finish_reason: FinishReason
    usage: Usage | None = None


class LoopError(_BaseEvent):
    """Terminal event: the loop could not complete.

    Carries the exception class name + message for a provider/transport
    error or a malformed streamed tool call. ``exceeded_max_iterations``
    flags the iteration-cap case specifically so a UI can offer a "retry
    with a higher limit" affordance. Pre-flight failures (a bad
    ``max_iterations`` argument, a capability-gate violation) raise
    synchronously instead of producing this event; only failures that
    occur once events have already begun flowing surface here.
    """

    type: Literal["error"] = "error"
    message: str
    error_type: str
    exceeded_max_iterations: bool = False


type LoopEvent = (
    IterationStart | TextDelta | ToolCallStarted | ToolResult | PendingToolCalls | Done | LoopError
)
