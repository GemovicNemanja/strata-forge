"""In-process conversation memory — append-only with optional trimming.

A :class:`ConversationMemory` holds the rolling history of an agent's
interactions: a system message (immutable for the memory's lifetime)
plus an append-only list of subsequent messages.

Two trim modes are supported:

- :meth:`trim_to_messages` — keep the last ``N`` non-system messages.
- :meth:`trim_to_tokens` — drop oldest non-system messages until the
  total token count fits a budget (via :func:`forge.llm.count_tokens`).

Both trim modes preserve the system message — the agent's
instructions don't get evicted under any memory pressure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from forge.llm.messages import AssistantMessage, SystemMessage, UserMessage
from forge.llm.tokens import count_tokens

if TYPE_CHECKING:
    from collections.abc import Iterable

    from forge.llm.messages import AnyMessage

__all__ = [
    "ConversationMemory",
]


def _message_text(message: AnyMessage) -> str:
    """Extract the text content of a message for token counting.

    Handles list-of-content-parts (:class:`UserMessage` multimodal)
    and tool-result / assistant variants. Returns an empty string for
    messages whose content isn't textual (e.g. image-only).
    """
    content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # UserMessage.content can be a list of parts (text + image).
    parts: list[str] = []
    if isinstance(content, list):
        items = cast("list[Any]", content)
        for part in items:
            if isinstance(part, str):
                parts.append(part)
            else:
                text: Any = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
    return " ".join(parts)


class ConversationMemory:
    """A rolling conversation history with a fixed system message.

    Args:
        system_message: Optional system message that opens every
            rendered conversation. When provided, this message is
            never evicted by :meth:`trim_to_messages` or
            :meth:`trim_to_tokens`. Can be a :class:`SystemMessage`
            instance or a plain string (which is wrapped).
    """

    def __init__(self, *, system_message: SystemMessage | str | None = None) -> None:
        if system_message is None:
            self._system: SystemMessage | None = None
        elif isinstance(system_message, SystemMessage):
            self._system = system_message
        else:
            self._system = SystemMessage(content=system_message)
        self._history: list[AnyMessage] = []

    @property
    def system_message(self) -> SystemMessage | None:
        return self._system

    @property
    def messages(self) -> tuple[AnyMessage, ...]:
        """All messages in chronological order — system first when set."""
        if self._system is None:
            return tuple(self._history)
        return (self._system, *self._history)

    @property
    def non_system_messages(self) -> tuple[AnyMessage, ...]:
        """The history excluding the system message."""
        return tuple(self._history)

    def append(self, message: AnyMessage) -> None:
        """Append one message to the history.

        Raises :class:`ValueError` when a :class:`SystemMessage` is
        appended — system messages live in the dedicated slot and
        can't be inserted mid-conversation.
        """
        if isinstance(message, SystemMessage):
            err = (
                "ConversationMemory holds at most one system message — pass it to "
                "the constructor or use replace_system_message()."
            )
            raise ValueError(err)
        self._history.append(message)

    def extend(self, messages: Iterable[AnyMessage]) -> None:
        """Append several messages."""
        for message in messages:
            self.append(message)

    def replace_system_message(self, system_message: SystemMessage | str | None) -> None:
        """Replace the system message (or clear it with ``None``)."""
        if system_message is None:
            self._system = None
        elif isinstance(system_message, SystemMessage):
            self._system = system_message
        else:
            self._system = SystemMessage(content=system_message)

    def clear(self) -> None:
        """Drop every non-system message. The system message survives."""
        self._history.clear()

    def __len__(self) -> int:
        return len(self.messages)

    def trim_to_messages(self, max_messages: int) -> None:
        """Keep the last ``max_messages`` non-system messages.

        The system message (when present) is preserved on top of the
        budget — a memory with a system message and ``max_messages=2``
        renders 3 messages total when read.

        Raises :class:`ValueError` when ``max_messages`` is negative.
        """
        if max_messages < 0:
            err = f"max_messages must be >= 0; got {max_messages}"
            raise ValueError(err)
        if max_messages == 0:
            self._history.clear()
        elif len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def trim_to_tokens(self, max_tokens: int, *, model: str | None = None) -> None:
        """Drop oldest non-system messages until the total fits ``max_tokens``.

        Token counts use :func:`forge.llm.count_tokens` against the
        named ``model`` (or the default encoding when ``model`` is
        ``None``). System message tokens count toward the budget but
        the system message itself is never evicted.

        When ``max_tokens`` is small enough that even the system
        message exceeds the budget, no trimming yields a fit — the
        memory keeps the system message plus an empty history.

        Raises :class:`ValueError` when ``max_tokens`` is negative.
        """
        if max_tokens < 0:
            err = f"max_tokens must be >= 0; got {max_tokens}"
            raise ValueError(err)

        def _tokens_for(message: AnyMessage) -> int:
            return count_tokens(_message_text(message), model=model)

        system_tokens = _tokens_for(self._system) if self._system else 0
        history_tokens = [_tokens_for(m) for m in self._history]
        total = system_tokens + sum(history_tokens)

        if total <= max_tokens:
            return

        # Drop from the oldest end of the non-system history until we fit.
        drop = 0
        while drop < len(self._history) and total > max_tokens:
            total -= history_tokens[drop]
            drop += 1
        self._history = self._history[drop:]

    def append_user(self, content: str) -> None:
        """Convenience: append a :class:`UserMessage` with text ``content``."""
        self._history.append(UserMessage(content=content))

    def append_assistant(self, content: str) -> None:
        """Convenience: append an :class:`AssistantMessage` with text ``content``."""
        self._history.append(AssistantMessage(content=content))
