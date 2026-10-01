"""OpenAI Responses API wire format: request building and response parsing.

A route whose registry entry sets ``wire_api: responses`` speaks OpenAI's Responses API
(``POST /v1/responses``) instead of Chat Completions. This module turns Forge's typed
conversation into a Responses request body, and turns the provider's typed output (the
streamed events, or the final response object) back into Forge's ``ResponseChunk`` and
``LLMResponse`` shapes.

Requests are stateless: ``store`` is always ``false``, so the provider keeps nothing between
calls. Reasoning comes back as encrypted content instead, and Forge carries each turn's output
items on ``AssistantMessage.provider_items`` and replays them, in order, on the next request to
the same provider. A turn from any other provider, or one without items, is sent as plain
assistant text plus one ``function_call`` item per tool call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from strata_forge.llm.errors import map_responses_error
from strata_forge.llm.messages import (
    AssistantMessage,
    CallRef,
    ProviderItems,
    ReasoningItem,
    SystemMessage,
    TextItem,
    ToolCall,
    UserMessage,
)
from strata_forge.llm.multimodal import ImageContent
from strata_forge.llm.responses import ResponseChunk, ToolCallDelta, Usage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from strata_forge.llm.messages import AnyMessage, ContentPart, ProviderItem, ResponsesProvider
    from strata_forge.llm.responses import FinishReason
    from strata_forge.llm.tools import AnyTool

__all__ = [
    "INCLUDE_ENCRYPTED_REASONING",
    "ParsedResponse",
    "ResponsesStreamParser",
    "build_input",
    "build_request",
    "parse_response",
    "parse_usage",
    "redact_encrypted_content",
    "requested_effort",
]


INCLUDE_ENCRYPTED_REASONING = "reasoning.encrypted_content"
"""The ``include`` value that returns reasoning as replayable encrypted content.

OpenAI returns it by default when ``store`` is ``false``; Azure requires it explicitly.
"""

_PHASES = frozenset({"commentary", "final_answer"})


# ---------------------------------------------------------------------------
# Duck-typed access (LiteLLM yields Pydantic models, tests pass plain dicts)
# ---------------------------------------------------------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return cast("dict[str, Any]", obj).get(key, default)
    return getattr(obj, key, default)


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return dict(cast("dict[str, Any]", obj))
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return {}


def _as_list(obj: Any) -> list[Any]:
    if isinstance(obj, (list, tuple)):
        return list(cast("Sequence[Any]", obj))
    return []


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _event_type(event: Any) -> str:
    raw = _get(event, "type")
    value = getattr(raw, "value", raw)
    return value if isinstance(value, str) else ""


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def _user_content(content: str | list[ContentPart]) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, ImageContent):
            parts.append(part.to_openai_responses_format())
        else:
            parts.append({"type": "input_text", "text": cast("Any", part).text})
    return parts


def _function_call(call: ToolCall, *, item_id: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "function_call",
        "call_id": call.id,
        "name": call.name,
        "arguments": json.dumps(call.arguments),
    }
    if item_id is not None:
        item["id"] = item_id
    return item


def _plain_assistant_text(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": text}


def _replayed_item(item: ProviderItem, calls: Mapping[str, ToolCall]) -> dict[str, Any]:
    if isinstance(item, ReasoningItem):
        return {
            "type": "reasoning",
            "id": item.id,
            "summary": [{"type": "summary_text", "text": text} for text in item.summary],
            "encrypted_content": item.encrypted_content,
        }
    if isinstance(item, TextItem):
        message: dict[str, Any] = {
            "type": "message",
            "id": item.id,
            "role": "assistant",
            "content": [{"type": "output_text", "text": item.text, "annotations": []}],
        }
        if item.phase is not None:
            message["phase"] = item.phase
        return message
    return _function_call(calls[item.call_id], item_id=item.id)


def _assistant_items(message: AssistantMessage, provider: str) -> list[dict[str, Any]]:
    """One assistant turn as Responses input items.

    The turn's own ``provider_items`` are replayed verbatim (ids, ``phase``, encrypted
    reasoning) when they came from ``provider``; ``status`` is never sent. Otherwise the turn
    is synthesized from its plain ``content`` and ``tool_calls``, which is valid input for any
    model and only loses the reasoning context.
    """
    items = message.provider_items
    if items is None or items.provider != provider:
        out: list[dict[str, Any]] = []
        if message.content:
            out.append(_plain_assistant_text(message.content))
        out.extend(_function_call(call) for call in message.tool_calls)
        return out

    calls = {call.id: call for call in message.tool_calls}
    has_text = any(isinstance(item, TextItem) for item in items.items)
    pending_text = message.content if (message.content and not has_text) else None
    replayed_calls: set[str] = set()
    out = []
    for item in items.items:
        if isinstance(item, CallRef):
            if pending_text is not None:
                out.append(_plain_assistant_text(pending_text))
                pending_text = None
            replayed_calls.add(item.call_id)
        out.append(_replayed_item(item, calls))
    if pending_text is not None:
        out.append(_plain_assistant_text(pending_text))
    out.extend(_function_call(call) for call in message.tool_calls if call.id not in replayed_calls)
    return out


def build_input(
    messages: Sequence[AnyMessage],
    provider: str,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Split a conversation into the Responses ``instructions`` and ``input`` items.

    The leading system messages become ``instructions``. A later system message (the
    structured-output reprompt, for one) becomes a ``developer`` message item in place, so it
    keeps its position in the conversation.
    """
    leading: list[str] = []
    index = 0
    while index < len(messages):
        head = messages[index]
        if not isinstance(head, SystemMessage):
            break
        leading.append(head.content)
        index += 1

    items: list[dict[str, Any]] = []
    for message in messages[index:]:
        if isinstance(message, SystemMessage):
            items.append({"type": "message", "role": "developer", "content": message.content})
        elif isinstance(message, UserMessage):
            items.append(
                {"type": "message", "role": "user", "content": _user_content(message.content)}
            )
        elif isinstance(message, AssistantMessage):
            items.extend(_assistant_items(message, provider))
        else:
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
    instructions = "\n\n".join(leading) if leading else None
    return instructions, items


def _text_format(response_format: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a Chat Completions ``response_format`` into a Responses ``text.format``."""
    if response_format.get("type") == "json_schema":
        spec = cast("Mapping[str, Any]", response_format.get("json_schema") or {})
        return {"type": "json_schema", **dict(spec)}
    return dict(response_format)


def requested_effort(extras: Mapping[str, Any] | None) -> str | None:
    """The reasoning effort a caller's ``provider_extras`` request, in either spelling."""
    if not extras:
        return None
    reasoning = extras.get("reasoning")
    if isinstance(reasoning, dict):
        effort = cast("dict[str, Any]", reasoning).get("effort")
        if isinstance(effort, str):
            return effort
    effort = extras.get("reasoning_effort")
    return effort if isinstance(effort, str) else None


def build_request(
    *,
    messages: Sequence[AnyMessage],
    provider: str,
    tools: Sequence[AnyTool] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    response_format: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    """Build the Responses request body (everything but ``model`` and auth).

    ``max_tokens`` becomes ``max_output_tokens``, which bounds reasoning AND visible output.
    ``response_format`` becomes ``text.format``. In ``extras``, a Chat Completions style
    ``reasoning_effort`` becomes ``reasoning.effort`` and a ``response_format`` becomes
    ``text.format``; every other key is forwarded verbatim and wins over the computed body.
    """
    instructions, input_items = build_input(messages, provider)
    body: dict[str, Any] = {
        "input": input_items,
        "store": False,
        "include": [INCLUDE_ENCRYPTED_REASONING],
    }
    if instructions is not None:
        body["instructions"] = instructions
    if tools:
        body["tools"] = [tool.to_openai_responses_schema() for tool in tools]
    if max_tokens is not None:
        body["max_output_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if response_format is not None:
        body["text"] = {"format": _text_format(response_format)}

    if extras:
        remaining = dict(extras)
        effort = remaining.pop("reasoning_effort", None)
        if effort is not None:
            reasoning = remaining.get("reasoning")
            merged = dict(cast("dict[str, Any]", reasoning)) if isinstance(reasoning, dict) else {}
            merged.setdefault("effort", effort)
            remaining["reasoning"] = merged
        extra_format = remaining.pop("response_format", None)
        if isinstance(extra_format, dict):
            body["text"] = {"format": _text_format(cast("dict[str, Any]", extra_format))}
        body.update(remaining)

    if stream:
        body["stream"] = True
    return body


def redact_encrypted_content(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy request input items with every ``encrypted_content`` replaced by its size.

    The diagnostic dump records the request; the encrypted reasoning is opaque and large.
    """
    redacted: list[dict[str, Any]] = []
    for item in items:
        copy = dict(item)
        content = copy.get("encrypted_content")
        if isinstance(content, str):
            copy["encrypted_content"] = f"<{len(content)} bytes>"
        redacted.append(copy)
    return redacted


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_usage(raw: Any) -> Usage:
    """Map a Responses ``usage`` object onto Forge's :class:`Usage`.

    OpenAI's ``input_tokens`` INCLUDES cached reads and cache writes, while
    :func:`~strata_forge.llm.cost.compute_cost` prices each bucket separately, so the
    input count is reported net of both.
    """
    if raw is None:
        return Usage(input_tokens=0, output_tokens=0)
    input_total = int(_get(raw, "input_tokens", 0) or 0)
    output_tokens = int(_get(raw, "output_tokens", 0) or 0)
    details = _get(raw, "input_tokens_details")
    cache_read = int(_get(details, "cached_tokens", 0) or 0)
    cache_write = int(_get(details, "cache_write_tokens", 0) or 0)
    return Usage(
        input_tokens=max(0, input_total - cache_read - cache_write),
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def _message_text(item: Mapping[str, Any]) -> tuple[str, bool]:
    """A message item's visible text, and whether any of it was a refusal."""
    texts: list[str] = []
    refused = False
    for part in _as_list(item.get("content")):
        part_type = _get(part, "type")
        if part_type == "output_text":
            text = _get(part, "text")
            if isinstance(text, str):
                texts.append(text)
        elif part_type == "refusal":
            refusal = _get(part, "refusal")
            if isinstance(refusal, str):
                texts.append(refusal)
                refused = True
    return "".join(texts), refused


def _provider_items(
    provider: ResponsesProvider,
    items: Sequence[Mapping[str, Any]],
) -> ProviderItems | None:
    """The replayable subset of a turn's output items, in output order.

    A reasoning item without ``encrypted_content`` cannot be replayed statelessly and is
    dropped; so is any item without an ``id``. Hosted-tool items never occur (Forge sends
    function tools only).
    """
    out: list[ProviderItem] = []
    for item in items:
        item_type = item.get("type")
        item_id = _str_or_none(item.get("id"))
        if item_id is None:
            continue
        if item_type == "reasoning":
            encrypted = _str_or_none(item.get("encrypted_content"))
            if encrypted is None:
                continue
            summary = tuple(
                text
                for part in _as_list(item.get("summary"))
                if isinstance(text := _get(part, "text"), str)
            )
            out.append(ReasoningItem(id=item_id, encrypted_content=encrypted, summary=summary))
        elif item_type == "message":
            text, _ = _message_text(item)
            phase = item.get("phase")
            out.append(TextItem(id=item_id, phase=phase if phase in _PHASES else None, text=text))
        elif item_type == "function_call":
            call_id = _str_or_none(item.get("call_id"))
            if call_id is not None:
                out.append(CallRef(id=item_id, call_id=call_id))
    if not out:
        return None
    return ProviderItems(provider=provider, items=tuple(out))


def _incomplete_reason(response: Any) -> FinishReason:
    reason = _get(_get(response, "incomplete_details"), "reason")
    return "content_filter" if reason == "content_filter" else "length"


def _failure(response: Any, *, model: str | None, provider: str | None) -> Exception:
    error = _get(response, "error")
    return map_responses_error(
        _str_or_none(_get(error, "code")),
        _str_or_none(_get(error, "message")) or "The response failed",
        model=model,
        provider=provider,
    )


def _lenient_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(cast("dict[str, Any]", raw))
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(cast("dict[str, Any]", parsed)) if isinstance(parsed, dict) else {}


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    """A non-streamed Responses result, in Forge's terms."""

    text: str
    tool_calls: list[ToolCall]
    finish_reason: FinishReason
    usage: Usage
    provider_items: ProviderItems | None


def parse_response(
    response: Any,
    *,
    provider: ResponsesProvider,
    model: str | None = None,
) -> ParsedResponse:
    """Parse a complete (non-streamed) Responses API response object.

    Raises:
        ProviderError: The response's ``status`` is ``failed``.
    """
    if _get(response, "status") == "failed":
        raise _failure(response, model=model, provider=provider)
    items = [_as_dict(raw) for raw in _as_list(_get(response, "output"))]
    texts: list[str] = []
    refused = False
    calls: list[ToolCall] = []
    for item in items:
        if item.get("type") == "message":
            text, item_refused = _message_text(item)
            texts.append(text)
            refused = refused or item_refused
        elif item.get("type") == "function_call":
            calls.append(
                ToolCall(
                    id=_str_or_none(item.get("call_id")) or "",
                    name=_str_or_none(item.get("name")) or "",
                    arguments=_lenient_arguments(item.get("arguments")),
                )
            )
    finish: FinishReason
    if _get(response, "status") == "incomplete":
        finish = _incomplete_reason(response)
    elif calls:
        finish = "tool_use"
    else:
        finish = "content_filter" if refused else "stop"
    return ParsedResponse(
        text="".join(texts),
        tool_calls=calls,
        finish_reason=finish,
        usage=parse_usage(_get(response, "usage")),
        provider_items=_provider_items(provider, items),
    )


class ResponsesStreamParser:
    """Turn streamed Responses API events into :class:`ResponseChunk`\\ s.

    Tool calls are keyed by the event's ``output_index`` (argument deltas carry no call id or
    name) and renumbered into dense tool slots 0, 1, ... in the order they open. The final
    ``response.completed`` / ``response.incomplete`` event produces the last chunk, with the
    usage, the finish reason and the turn's :class:`~strata_forge.llm.messages.ProviderItems`.

    The terminal event's ``response.output`` is authoritative: anything the stream did not
    deliver as deltas (a stream LiteLLM fakes from a non-streamed call carries only text deltas)
    is emitted on that final chunk. ``response.failed`` and ``error`` raise a mapped
    ``ProviderError``. A stream that ends without a terminal event leaves :attr:`finished`
    ``False``; the caller must treat that as a failure, never as a clean stop.
    """

    def __init__(
        self,
        provider: ResponsesProvider,
        *,
        model: str | None = None,
    ) -> None:
        self._provider: ResponsesProvider = provider
        self._model = model
        self._slots: dict[int, int] = {}
        self._args_seen: set[int] = set()
        self._items: dict[int, dict[str, Any]] = {}
        self._text_indexes: set[int] = set()
        self._text_streamed = False
        self._refused = False
        self.finished = False

    def _slot(self, output_index: int) -> tuple[int, bool]:
        """The tool slot for ``output_index``, and whether it was just opened."""
        slot = self._slots.get(output_index)
        if slot is not None:
            return slot, False
        slot = len(self._slots)
        self._slots[output_index] = slot
        return slot, True

    def _call_delta(self, output_index: int, item: Mapping[str, Any]) -> ToolCallDelta:
        """Everything about a finished call the stream has not delivered yet."""
        slot, _ = self._slot(output_index)
        arguments = item.get("arguments")
        delta = ToolCallDelta(
            index=slot,
            id=_str_or_none(item.get("call_id")),
            name=_str_or_none(item.get("name")),
            arguments_delta=(
                arguments if isinstance(arguments, str) and slot not in self._args_seen else ""
            ),
        )
        self._args_seen.add(slot)
        return delta

    def feed(self, event: Any) -> ResponseChunk | None:
        """Fold one streamed event; return the chunk it produces, if any.

        Raises:
            ProviderError: On ``response.failed`` or an ``error`` event.
        """
        event_type = _event_type(event)
        if event_type in ("response.output_text.delta", "response.refusal.delta"):
            delta = _str_or_none(_get(event, "delta")) or ""
            if not delta:
                return None
            self._text_streamed = True
            self._text_indexes.add(int(_get(event, "output_index", 0) or 0))
            if event_type == "response.refusal.delta":
                self._refused = True
            return ResponseChunk(delta_text=delta)
        if event_type == "response.output_item.added":
            item = _as_dict(_get(event, "item"))
            if item.get("type") != "function_call":
                return None
            output_index = int(_get(event, "output_index", 0) or 0)
            slot, _ = self._slot(output_index)
            arguments = _str_or_none(item.get("arguments")) or ""
            if arguments:
                self._args_seen.add(slot)
            return ResponseChunk(
                delta_tool_calls=[
                    ToolCallDelta(
                        index=slot,
                        id=_str_or_none(item.get("call_id")),
                        name=_str_or_none(item.get("name")),
                        arguments_delta=arguments,
                    )
                ]
            )
        if event_type == "response.function_call_arguments.delta":
            output_index = int(_get(event, "output_index", 0) or 0)
            slot, _ = self._slot(output_index)
            delta = _str_or_none(_get(event, "delta")) or ""
            if not delta:
                return None
            self._args_seen.add(slot)
            return ResponseChunk(
                delta_tool_calls=[ToolCallDelta(index=slot, arguments_delta=delta)]
            )
        if event_type == "response.output_item.done":
            output_index = int(_get(event, "output_index", 0) or 0)
            item = _as_dict(_get(event, "item"))
            self._items[output_index] = item
            if item.get("type") == "message" and output_index not in self._text_indexes:
                # A message whose text never arrived as deltas: deliver it whole, once.
                text, refused = _message_text(item)
                self._text_indexes.add(output_index)
                self._refused = self._refused or refused
                return ResponseChunk(delta_text=text) if text else None
            if item.get("type") != "function_call":
                return None
            return ResponseChunk(delta_tool_calls=[self._call_delta(output_index, item)])
        if event_type in ("response.completed", "response.incomplete"):
            return self._finish(
                _get(event, "response"), incomplete=event_type.endswith("incomplete")
            )
        if event_type == "response.failed":
            raise _failure(_get(event, "response"), model=self._model, provider=self._provider)
        if event_type == "error":
            raise map_responses_error(
                _str_or_none(_get(event, "code")),
                _str_or_none(_get(event, "message")) or "The response stream reported an error",
                model=self._model,
                provider=self._provider,
            )
        return None

    def _finish(self, response: Any, *, incomplete: bool) -> ResponseChunk:
        self.finished = True
        status = _get(response, "status")
        if status == "failed":
            raise _failure(response, model=self._model, provider=self._provider)

        late_calls: list[ToolCallDelta] = []
        late_text: list[str] = []
        for output_index, raw in enumerate(_as_list(_get(response, "output"))):
            if output_index in self._items:
                continue
            item = _as_dict(raw)
            self._items[output_index] = item
            if item.get("type") == "function_call":
                late_calls.append(self._call_delta(output_index, item))
            elif item.get("type") == "message" and not self._text_streamed:
                text, refused = _message_text(item)
                late_text.append(text)
                self._refused = self._refused or refused

        ordered = [self._items[index] for index in sorted(self._items)]
        finish: FinishReason
        if incomplete or status == "incomplete":
            finish = _incomplete_reason(response)
        elif any(item.get("type") == "function_call" for item in ordered):
            finish = "tool_use"
        else:
            finish = "content_filter" if self._refused else "stop"

        return ResponseChunk(
            delta_text="".join(late_text),
            delta_tool_calls=late_calls,
            finish_reason=finish,
            usage=parse_usage(_get(response, "usage")),
            provider_items=_provider_items(self._provider, ordered),
        )
