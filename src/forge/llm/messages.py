"""Message types for chat-style LLM calls.

The four concrete message types — ``SystemMessage``, ``UserMessage``,
``AssistantMessage``, ``ToolResultMessage`` — each carry a ``role`` literal
that doubles as a discriminator. The ``Message`` namespace class exposes
ergonomic factory methods (``Message.user(...)``) used throughout the API.

Tool calls are represented in this module (``ToolCall``) because they are
fundamentally part of the message shape: an ``AssistantMessage`` carries
the model's tool-call requests, and a ``ToolResultMessage`` carries the
matching result. ``validate_conversation`` enforces the invariant that
every tool-result ID has a prior matching call.
"""

from __future__ import annotations

from typing import Any, Literal, NoReturn, final

from pydantic import BaseModel, ConfigDict, Field

from forge.core.errors import ValidationError

__all__ = [
    "AnyMessage",
    "AssistantMessage",
    "ContentPart",
    "Message",
    "Role",
    "SystemMessage",
    "TextPart",
    "ToolCall",
    "ToolResultMessage",
    "UserMessage",
    "validate_conversation",
]


type Role = Literal["system", "user", "assistant", "tool"]


# ---------------------------------------------------------------------------
# Content parts (multimodal building blocks)
# ---------------------------------------------------------------------------


class ContentPart(BaseModel):
    """Base for typed pieces of a multipart message body.

    Concrete subclasses declare their own ``type: Literal[...]`` field
    plus their payload. ``TextPart`` is the only concrete subclass at this
    layer; ``ImagePart`` is added by ``forge.llm.multimodal``. Callers
    narrow via ``isinstance(part, TextPart)`` rather than reading ``type``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class TextPart(ContentPart):
    """A text segment within a multipart message body."""

    type: Literal["text"] = "text"
    text: str


# ---------------------------------------------------------------------------
# Tool call (one request from the model to invoke a tool)
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A single tool-invocation request emitted by the assistant.

    ``arguments`` is a JSON object whose schema is determined by the
    invoked tool. Validation against the tool's argument model happens at
    the seam where the tool is actually invoked, not here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Concrete message types
# ---------------------------------------------------------------------------


class _BaseMessage(BaseModel):
    """Internal base — every concrete message type subclasses it.

    Each concrete subclass declares its own ``role: Literal[...]``; the
    base intentionally does not declare ``role`` so that the subclass
    fields are first declarations rather than narrowing overrides.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class SystemMessage(_BaseMessage):
    """A system prompt — sets context, persona, or constraints for the model."""

    role: Literal["system"] = "system"
    content: str


class UserMessage(_BaseMessage):
    """A user turn. Content may be plain text or a list of typed parts."""

    role: Literal["user"] = "user"
    content: str | list[ContentPart]


class AssistantMessage(_BaseMessage):
    """A model turn. May include text, tool calls, or both."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] = []


class ToolResultMessage(_BaseMessage):
    """The result of executing a tool call.

    ``tool_call_id`` MUST reference an earlier ``ToolCall.id`` in the
    same conversation; ``validate_conversation`` enforces this.
    """

    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str
    is_error: bool = False


type AnyMessage = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage


# ---------------------------------------------------------------------------
# Factory namespace — `Message.user(...)`, `Message.assistant(...)`, ...
# ---------------------------------------------------------------------------


@final
class Message:
    """Static-method namespace for ergonomic message construction.

    Use ``Message.system(...)`` / ``Message.user(...)`` etc. instead of
    instantiating the concrete classes directly. The namespace exists so
    callers don't have to import four classes for the common case.
    """

    def __new__(cls) -> NoReturn:
        msg = "Message is a namespace; call its static methods directly"
        raise TypeError(msg)

    @staticmethod
    def system(content: str) -> SystemMessage:
        return SystemMessage(content=content)

    @staticmethod
    def user(content: str | list[ContentPart]) -> UserMessage:
        return UserMessage(content=content)

    @staticmethod
    def assistant(
        content: str | None = None,
        *,
        tool_calls: list[ToolCall] | None = None,
    ) -> AssistantMessage:
        return AssistantMessage(content=content, tool_calls=tool_calls or [])

    @staticmethod
    def tool_result(
        tool_call_id: str,
        content: str,
        *,
        is_error: bool = False,
    ) -> ToolResultMessage:
        return ToolResultMessage(
            tool_call_id=tool_call_id,
            content=content,
            is_error=is_error,
        )


# ---------------------------------------------------------------------------
# Conversation-level validation
# ---------------------------------------------------------------------------


def validate_conversation(messages: list[AnyMessage]) -> None:
    """Verify every ``ToolResultMessage`` references a prior ``ToolCall.id``.

    Raises ``forge.core.errors.ValidationError`` on the first violation,
    naming the offending message index and the unmatched tool_call_id. A
    duplicate ``tool_call_id`` across multiple ``ToolCall`` instances is
    also rejected — every call must have a unique id.
    """
    seen_call_ids: set[str] = set()
    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                if call.id in seen_call_ids:
                    raise ValidationError(
                        f"Duplicate ToolCall id {call.id!r} at message index {index}",
                    )
                seen_call_ids.add(call.id)
        elif isinstance(message, ToolResultMessage) and message.tool_call_id not in seen_call_ids:
            raise ValidationError(
                f"ToolResultMessage at index {index} references unknown "
                f"tool_call_id {message.tool_call_id!r}",
            )
