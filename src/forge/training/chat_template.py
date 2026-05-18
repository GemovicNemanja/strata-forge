"""Chat-template formatting for fine-tuning.

Converts Forge :class:`AnyMessage` conversations into the
single-string formats that HuggingFace tokenizers and TRL trainers
consume. The two supported targets are:

- ``apply_chat_template`` — use the tokenizer's own Jinja chat
  template (the canonical HF approach). Works with any model whose
  tokenizer has a ``chat_template``.
- ``conversation_to_text`` — fall back to ``role: content`` line
  formatting when no chat template is available.

This module is dependency-light: ``transformers`` is imported
lazily inside :func:`apply_chat_template` so importing
:mod:`forge.training.chat_template` works without the
``[finetuning]`` extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from forge.llm.messages import (
    AssistantMessage,
    ContentPart,
    SystemMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from forge.llm.messages import AnyMessage

__all__ = [
    "apply_chat_template",
    "conversation_to_dicts",
    "conversation_to_text",
]


def _content_to_text(content: str | list[ContentPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        parts.append(f"[non-text: {part.__class__.__name__}]")
    return "".join(parts)


def _message_role(message: AnyMessage) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, UserMessage):
        return "user"
    if isinstance(message, AssistantMessage):
        return "assistant"
    return "tool"


def conversation_to_dicts(
    messages: Sequence[AnyMessage],
) -> list[dict[str, str]]:
    """Convert a conversation to the standard ``[{role, content}]`` shape.

    The shape is the one HuggingFace tokenizers expect for
    ``apply_chat_template``. Tool calls and multimodal content
    parts collapse to text — the trainer side is text-only.
    """
    out: list[dict[str, str]] = []
    for message in messages:
        role = _message_role(message)
        text = _content_to_text(message.content)
        out.append({"role": role, "content": text})
    return out


def conversation_to_text(
    messages: Sequence[AnyMessage],
    *,
    role_prefix: str = "",
    role_suffix: str = ": ",
    message_separator: str = "\n\n",
) -> str:
    """Render a conversation as plain text without a tokenizer.

    Falls back to ``role: content`` formatting. Useful when no
    chat template is available or for ad-hoc inspection.
    """
    dicts = conversation_to_dicts(messages)
    lines = [f"{role_prefix}{d['role']}{role_suffix}{d['content']}" for d in dicts]
    return message_separator.join(lines)


def apply_chat_template(
    messages: Sequence[AnyMessage],
    *,
    tokenizer: Any,
    add_generation_prompt: bool = False,
    tokenize: bool = False,
) -> str | list[int]:
    """Apply a HF tokenizer's chat template to a Forge conversation.

    Args:
        messages: The Forge conversation.
        tokenizer: A HF tokenizer with an ``apply_chat_template``
            method. Forge does not import ``transformers`` itself;
            callers pass the tokenizer in.
        add_generation_prompt: Forwarded to
            ``apply_chat_template``. Set to ``True`` when preparing
            a prompt for generation (training-style usage leaves it
            ``False``).
        tokenize: When ``True``, returns the token IDs list; when
            ``False`` (default), returns the formatted string.

    Returns:
        Either the formatted prompt string (``tokenize=False``) or
        the list of token IDs (``tokenize=True``).

    Raises:
        ValueError: If the tokenizer has no ``apply_chat_template``
            method or no ``chat_template`` set.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        err = (
            "tokenizer has no apply_chat_template method — pass a HF "
            "tokenizer or use conversation_to_text() instead"
        )
        raise ValueError(err)
    dicts = conversation_to_dicts(messages)
    result: Any = tokenizer.apply_chat_template(
        dicts,
        add_generation_prompt=add_generation_prompt,
        tokenize=tokenize,
    )
    if tokenize:
        return [int(x) for x in result]
    return str(result)
