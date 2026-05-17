"""``LLMClient`` — the main async LLM surface.

Ties every LLM building block together:

- Route resolution via :mod:`forge.llm.routing` + :mod:`forge.llm.registry`.
- Provider-agnostic response cache (:mod:`forge.llm.cache`).
- Two-axis fallback + per-call retry (:mod:`forge.llm.fallback` +
  :mod:`forge.core.retry`).
- Structured output (:mod:`forge.llm.schemas`).
- Tool calling and the multi-turn loop (:mod:`forge.llm.tools`).
- Multimodal content (:mod:`forge.llm.multimodal`).
- Streaming accumulators (:mod:`forge.llm.streaming`).
- Cost / token accounting via the registry + :mod:`forge.llm.cost`.
- ``BudgetContext`` enforcement (:mod:`forge.core.budget`).
- NDJSON diagnostic dump (:mod:`forge.llm.diagnostic`).

Two construction modes:

- :class:`LLMClient` ``(model="...", provider="...")`` — single logical
  model on a fixed route (or the registry default if ``provider`` is
  omitted).
- :meth:`LLMClient.with_fallbacks` — a fallback chain. Each entry can
  pin a list of providers for that model; bare strings expand to
  "default route only" entries.

Streaming bypasses the cache (no useful "final shape" mid-stream) and
runs on the head entry of the chain only — provider-level failover
mid-stream is impractical and explicitly out of scope.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel

from forge.core.budget import current_budget
from forge.core.errors import RegistryError
from forge.core.ids import get_correlation_id
from forge.llm.cache import cache_key as compute_cache_key
from forge.llm.cost import compute_cost
from forge.llm.diagnostic import (
    DiagnosticRecord,
    make_error_field,
    utcnow_iso,
    write_diagnostic_record,
)
from forge.llm.errors import map_litellm_exception
from forge.llm.fallback import (
    ModelFallback,
    normalize_fallback_chain,
    run_with_fallback,
)
from forge.llm.messages import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
    validate_conversation,
)
from forge.llm.multimodal import ImageContent
from forge.llm.providers import (
    AnthropicProvider,
    AzureProvider,
    BedrockProvider,
    OpenAICompatProvider,
    OpenAIProvider,
    ProviderClient,
    VertexProvider,
)
from forge.llm.registry import ProviderName, registry
from forge.llm.responses import LLMResponse, ResponseChunk, ToolCall, ToolCallDelta, Usage
from forge.llm.routing import ModelRoute, resolve
from forge.llm.schemas import (
    StructuredOutputError,
    make_reprompt_instruction,
    parse_json_response,
    to_anthropic_forced_tool_schema,
    to_gemini_response_schema,
    to_openai_response_format,
)
from forge.llm.tools import Tool, ToolLoopExceededError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from forge.llm.cache import CacheBackend
    from forge.llm.fallback import FallbackEntry
    from forge.llm.messages import AnyMessage, ContentPart, TextPart
    from forge.llm.responses import FinishReason

__all__ = [
    "LLMClient",
    "StructuredResponse",
]


# ---------------------------------------------------------------------------
# StructuredResponse — LLMResponse + parsed Pydantic instance
# ---------------------------------------------------------------------------


class StructuredResponse[M: BaseModel](LLMResponse):
    """:class:`LLMResponse` plus a validated Pydantic instance of the schema.

    Returned by :meth:`LLMClient.complete_structured`. Generic in the
    schema type so ``resp.parsed`` has the precise model type at the
    call site.
    """

    parsed: M


# ---------------------------------------------------------------------------
# Provider client registry
# ---------------------------------------------------------------------------


def _default_provider_clients() -> dict[ProviderName, ProviderClient]:
    """Construct one provider client of each kind, env-configured."""
    return {
        "openai": OpenAIProvider(),
        "anthropic": AnthropicProvider(),
        "vertex": VertexProvider(),
        "bedrock": BedrockProvider(),
        "azure": AzureProvider(),
        "openai_compat": OpenAICompatProvider(),
    }


# ---------------------------------------------------------------------------
# Wire-format helpers
# ---------------------------------------------------------------------------


class _ImagePart(Protocol):
    """Structural type for the multimodal image content part."""

    def to_openai_format(self) -> dict[str, Any]: ...
    def to_anthropic_format(self) -> dict[str, Any]: ...
    def to_gemini_format(self) -> dict[str, Any]: ...


def _image_part_for_provider(part: _ImagePart, provider: ProviderName) -> dict[str, Any]:
    if provider == "anthropic" or provider == "bedrock":
        return part.to_anthropic_format()
    if provider == "vertex":
        return part.to_gemini_format()
    # OpenAI / Azure / openai_compat all accept the OpenAI shape.
    return part.to_openai_format()


def _text_part(part: TextPart) -> dict[str, Any]:
    return {"type": "text", "text": part.text}


def _content_to_wire(
    content: str | list[ContentPart],
    provider: ProviderName,
) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    out: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, ImageContent):
            out.append(_image_part_for_provider(part, provider))
        else:
            # Any other ContentPart subclass — assume it has a `type` and
            # serializes via `model_dump`. The only built-in concrete subclass
            # at this layer is TextPart.
            out.append(_text_part(cast("TextPart", part)))
    return out


def _tool_call_to_wire(call: ToolCall) -> dict[str, Any]:
    """OpenAI-shape tool call (the canonical LiteLLM input format)."""
    import json

    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.arguments),
        },
    }


def _message_to_wire(msg: AnyMessage, provider: ProviderName) -> dict[str, Any]:
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    if isinstance(msg, UserMessage):
        return {"role": "user", "content": _content_to_wire(msg.content, provider)}
    if isinstance(msg, AssistantMessage):
        result: dict[str, Any] = {"role": "assistant"}
        if msg.content is not None:
            result["content"] = msg.content
        if msg.tool_calls:
            result["tool_calls"] = [_tool_call_to_wire(tc) for tc in msg.tool_calls]
        return result
    # Must be ToolResultMessage by elimination from AnyMessage.
    return {
        "role": "tool",
        "tool_call_id": msg.tool_call_id,
        "content": msg.content,
    }


# ---------------------------------------------------------------------------
# Tool schema dispatch per provider
# ---------------------------------------------------------------------------


def _tools_for_provider(tools: Sequence[Tool], provider: ProviderName) -> list[dict[str, Any]]:
    if provider == "anthropic" or provider == "bedrock":
        return [t.to_anthropic_schema() for t in tools]
    if provider == "vertex":
        return [t.to_gemini_schema() for t in tools]
    return [t.to_openai_schema() for t in tools]


# ---------------------------------------------------------------------------
# Response normalization (OpenAI-shaped — LiteLLM normalizes for us)
# ---------------------------------------------------------------------------


def _normalize_response(
    raw: Any,
    *,
    route: ModelRoute,
    latency_ms: float,
) -> LLMResponse:
    """Convert a LiteLLM-shaped response into :class:`LLMResponse`.

    LiteLLM normalizes provider responses into an OpenAI-compatible
    shape (``choices[0].message`` + ``usage``). We duck-type that shape
    so tests can pass simple namespaces; real LiteLLM calls also work.
    """
    raw_any: Any = raw
    choice = raw_any.choices[0]
    message = choice.message
    finish_reason = _normalize_finish_reason(_duck_get(choice, "finish_reason"))

    text = _duck_get(message, "content") or ""
    tool_calls = _parse_tool_calls(_duck_get(message, "tool_calls"))
    # If the model emitted tool calls but the provider's finish_reason didn't
    # signal them, normalize to "tool_use" so callers can rely on it.
    if tool_calls and finish_reason != "tool_use":
        finish_reason = "tool_use"

    usage = _parse_usage(_duck_get(raw_any, "usage"))
    cost = compute_cost(usage, route.model)

    return LLMResponse(
        text=cast("str", text),
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        cost_usd=cost,
        route=route,
        cache_hit=False,
        latency_ms=latency_ms,
    )


def _normalize_finish_reason(raw: Any) -> FinishReason:
    # LiteLLM uses OpenAI's vocabulary; map to ours.
    mapping = {
        "stop": "stop",
        "tool_calls": "tool_use",
        "tool_use": "tool_use",
        "function_call": "tool_use",
        "length": "length",
        "content_filter": "content_filter",
        "max_tokens": "length",
    }
    if isinstance(raw, str) and raw in mapping:
        return cast("FinishReason", mapping[raw])
    return "stop"


def _duck_get(obj: Any, key: str, default: Any = None) -> Any:
    """Duck-typed accessor: tries attribute, then dict-key, else default.

    LiteLLM returns rich pydantic-y objects in production; tests pass
    plain dicts or :class:`types.SimpleNamespace` namespaces. This helper
    smooths over the difference so the normalizer works on either.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return cast("dict[str, Any]", obj).get(key, default)
    return getattr(obj, key, default)


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    import json

    if not raw:
        return []
    out: list[ToolCall] = []
    for entry in raw:
        fn = _duck_get(entry, "function")
        if fn is None:
            continue
        call_id_raw = _duck_get(entry, "id", "")
        name_raw = _duck_get(fn, "name", "")
        args_raw = _duck_get(fn, "arguments", "{}")
        call_id = call_id_raw if isinstance(call_id_raw, str) else ""
        name = name_raw if isinstance(name_raw, str) else ""
        arguments: dict[str, Any] = {}
        if isinstance(args_raw, str):
            try:
                parsed_args: Any = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                parsed_args = {}
            if isinstance(parsed_args, dict):
                arguments = dict(cast("dict[str, Any]", parsed_args))
        elif isinstance(args_raw, dict):
            arguments = dict(cast("dict[str, Any]", args_raw))
        out.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return out


def _parse_usage(raw: Any) -> Usage:
    if raw is None:
        return Usage(input_tokens=0, output_tokens=0)
    input_tokens = _duck_get(raw, "prompt_tokens", 0) or 0
    output_tokens = _duck_get(raw, "completion_tokens", 0) or 0
    details = _duck_get(raw, "prompt_tokens_details")
    cache_read = 0
    if details is not None:
        cache_read = _duck_get(details, "cached_tokens", 0) or 0
    return Usage(
        input_tokens=int(cast("int", input_tokens)),
        output_tokens=int(cast("int", output_tokens)),
        cache_read_tokens=int(cast("int", cache_read)),
    )


def _normalize_stream_chunk(raw: Any) -> ResponseChunk:
    raw_any: Any = raw
    choice = raw_any.choices[0]
    delta = _duck_get(choice, "delta")
    delta_text = ""
    delta_tool_calls: list[ToolCallDelta] = []
    if delta is not None:
        delta_text = _duck_get(delta, "content", "") or ""
        raw_tc = _duck_get(delta, "tool_calls")
        if raw_tc:
            for entry in raw_tc:
                idx_raw = _duck_get(entry, "index", 0)
                call_id = _duck_get(entry, "id")
                fn = _duck_get(entry, "function")
                name: str | None = None
                args_delta = ""
                if fn is not None:
                    name = _duck_get(fn, "name")
                    args_delta = _duck_get(fn, "arguments", "") or ""
                delta_tool_calls.append(
                    ToolCallDelta(
                        index=int(cast("int", idx_raw)),
                        id=call_id if isinstance(call_id, str) else None,
                        name=name if isinstance(name, str) else None,
                        arguments_delta=args_delta if isinstance(args_delta, str) else "",
                    )
                )
    finish_reason_raw = _duck_get(choice, "finish_reason")
    normalized_reason: FinishReason | None = (
        _normalize_finish_reason(finish_reason_raw) if finish_reason_raw is not None else None
    )
    usage_raw = _duck_get(raw_any, "usage")
    usage = _parse_usage(usage_raw) if usage_raw is not None else None
    return ResponseChunk(
        delta_text=cast("str", delta_text),
        delta_tool_calls=delta_tool_calls,
        finish_reason=normalized_reason,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Async LLM client with cache, retry, fallback, budget, and diagnostic.

    Args:
        model: Logical model name. Required unless ``chain`` is given.
        provider: Pin to a specific provider for ``model``. If ``None``,
            the registry's default route is used.
        chain: A fallback chain (mutually exclusive with ``model``). Use
            :meth:`with_fallbacks` for ergonomic construction.
        cache: Optional cache backend. ``None`` disables caching.
        provider_clients: Override the auto-instantiated provider client
            map. Tests use this to inject mocked clients.
        retry_max_attempts: Max retries per provider attempt before the
            attempt is counted as exhausted.
        retry_initial_wait: Base wait between retries (seconds).
        retry_max_wait: Cap on wait between retries (seconds).
        strict_bad_request: When ``True``, a
            :exc:`ProviderBadRequestError` aborts the fallback chain
            immediately. Default ``False``: advance to the next provider.
    """

    def __init__(
        self,
        model: str | None = None,
        provider: ProviderName | None = None,
        *,
        chain: Sequence[FallbackEntry] | None = None,
        cache: CacheBackend | None = None,
        provider_clients: Mapping[ProviderName, ProviderClient] | None = None,
        retry_max_attempts: int = 5,
        retry_initial_wait: float = 1.0,
        retry_max_wait: float = 30.0,
        strict_bad_request: bool = False,
    ) -> None:
        if (model is None) == (chain is None):
            err = "LLMClient: pass exactly one of `model` or `chain`"
            raise ValueError(err)
        if chain is not None:
            self._chain: list[ModelFallback] = normalize_fallback_chain(chain)
        else:
            # `model` is not None here — the XOR check above guarantees it,
            # but we cast for the type checker rather than asserting (S101).
            self._chain = [
                ModelFallback(
                    model=cast("str", model),
                    providers=(provider,) if provider is not None else None,
                )
            ]

        self._cache = cache
        self._provider_clients: dict[ProviderName, ProviderClient] = (
            dict(provider_clients) if provider_clients is not None else _default_provider_clients()
        )
        self._retry_max_attempts = retry_max_attempts
        self._retry_initial_wait = retry_initial_wait
        self._retry_max_wait = retry_max_wait
        self._strict_bad_request = strict_bad_request

    # --- Constructors -----------------------------------------------------

    @classmethod
    def with_fallbacks(
        cls,
        chain: Sequence[FallbackEntry],
        **kwargs: Any,
    ) -> LLMClient:
        """Build a client whose calls run through a two-axis fallback chain."""
        return cls(chain=chain, **kwargs)

    # --- Public API -------------------------------------------------------

    async def complete(
        self,
        messages: Sequence[AnyMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: Sequence[Tool] | None = None,
        response_format: dict[str, Any] | None = None,
        provider_extras: Mapping[ProviderName, Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        """Run a non-streaming completion through the configured chain.

        Returns the first successful :class:`LLMResponse`. On cache hit
        the response has ``cache_hit=True`` and ``cost_usd=0.0``; the
        original (cached) ``usage`` and ``route`` are preserved.
        """
        validate_conversation(list(messages))
        self._check_tool_capability(tools)

        # Cache key uses the chain's HEAD model name. The cache is
        # provider-agnostic by construction (see `cache_key`), so a hit
        # is valid regardless of which provider would have served the call.
        head_model = self._chain[0].model
        wire_messages_canonical = [_message_to_wire(m, "openai") for m in messages]
        tools_canonical = _tools_for_provider(tools, "openai") if tools else None
        cache_k = compute_cache_key(
            model=head_model,
            messages=wire_messages_canonical,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format=response_format,
            tools=tools_canonical,
            provider_extras=_flatten_extras(provider_extras),
        )

        if self._cache is not None:
            cached = await self._cache.get(cache_k)
            if cached is not None:
                return cached.model_copy(update={"cache_hit": True, "cost_usd": 0.0})

        async def _attempt(route: ModelRoute) -> LLMResponse:
            response = await self._invoke(
                route=route,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                tools=tools,
                response_format=response_format,
                provider_extras=provider_extras,
            )
            await self._consume_budget(response)
            if self._cache is not None:
                await self._cache.set(cache_k, response)
            return response

        return await run_with_fallback(
            self._chain,
            _attempt,
            strict_bad_request=self._strict_bad_request,
            retry_max_attempts=self._retry_max_attempts,
            retry_initial_wait=self._retry_initial_wait,
            retry_max_wait=self._retry_max_wait,
        )

    async def complete_structured[M: BaseModel](
        self,
        messages: Sequence[AnyMessage],
        *,
        schema: type[M],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        max_reprompt_attempts: int = 3,
        provider_extras: Mapping[ProviderName, Mapping[str, Any]] | None = None,
    ) -> StructuredResponse[M]:
        """Completion that returns a validated Pydantic instance.

        Dispatch by route:

        - OpenAI / Azure / openai_compat: native ``response_format``.
        - Anthropic / Bedrock: forced-tool emulation.
        - Vertex: ``response_schema`` on Gemini, forced-tool for Claude.

        When the model emits text that fails Pydantic validation, the
        client reprompts with the parse error included, up to
        ``max_reprompt_attempts`` times before raising
        :exc:`StructuredOutputError`.
        """
        validate_conversation(list(messages))
        head_entry = self._chain[0]
        head_provider = head_entry.providers[0] if head_entry.providers else None
        head_route = resolve(head_entry.model, head_provider)
        text_messages: list[AnyMessage] = list(messages)
        last_error: Exception | None = None

        for _ in range(max_reprompt_attempts + 1):
            response_format, extra_tools, extra_extras = _structured_payload_for_route(
                head_route, schema
            )
            extras_combined = _merge_extras(provider_extras, extra_extras)

            response = await self.complete(
                messages=text_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                tools=extra_tools,
                response_format=response_format,
                provider_extras=extras_combined,
            )

            try:
                parsed_text = _extract_structured_text(response, schema_name=schema.__name__)
                parsed = parse_json_response(parsed_text, schema)
            except Exception as exc:
                last_error = exc
                # Append a reprompt instruction and retry against the same chain.
                text_messages = list(messages)
                text_messages.append(
                    SystemMessage(
                        content=make_reprompt_instruction(schema, parse_error=str(exc)),
                    )
                )
                continue

            return StructuredResponse[M](
                text=response.text,
                tool_calls=response.tool_calls,
                finish_reason=response.finish_reason,
                usage=response.usage,
                cost_usd=response.cost_usd,
                route=response.route,
                cache_hit=response.cache_hit,
                latency_ms=response.latency_ms,
                parsed=parsed,
            )

        msg = (
            f"Structured-output reprompt loop exhausted after {max_reprompt_attempts + 1} "
            f"attempt(s) without producing a valid {schema.__name__}"
        )
        raise StructuredOutputError(
            msg,
            attempts=max_reprompt_attempts + 1,
            last_error=last_error,
        )

    async def stream(
        self,
        messages: Sequence[AnyMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: Sequence[Tool] | None = None,
        provider_extras: Mapping[ProviderName, Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[ResponseChunk]:
        """Stream completion chunks from the head entry of the chain.

        Streaming bypasses the cache and the fallback loop (a stream
        midway through can't be cleanly transferred to a new provider).
        """
        validate_conversation(list(messages))
        self._check_tool_capability(tools)

        entry = self._chain[0]
        first_provider = entry.providers[0] if entry.providers else None
        route = resolve(entry.model, first_provider)
        provider_client = self._provider_clients[route.provider]
        wire = [_message_to_wire(m, route.provider) for m in messages]
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            kwargs["top_p"] = top_p
        if tools is not None:
            kwargs["tools"] = _tools_for_provider(tools, route.provider)
        if provider_extras is not None:
            extras = provider_extras.get(route.provider)
            if extras is not None:
                kwargs.update(extras)

        return self._stream_chunks(provider_client, route, wire, kwargs)

    async def _stream_chunks(
        self,
        provider_client: ProviderClient,
        route: ModelRoute,
        wire: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> AsyncIterator[ResponseChunk]:
        try:
            async for raw in provider_client.astream(
                provider_model_id=route.provider_model_id,
                messages=wire,
                **kwargs,
            ):
                yield _normalize_stream_chunk(raw)
        except Exception as exc:
            mapped = map_litellm_exception(exc, model=route.model, provider=route.provider)
            raise mapped from exc

    async def run_tool_loop(
        self,
        messages: Sequence[AnyMessage],
        *,
        tools: Sequence[Tool],
        max_iterations: int = 8,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        provider_extras: Mapping[ProviderName, Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        """Run a multi-turn tool-use loop until the model stops calling tools.

        Each iteration:
          1. Run a completion.
          2. If ``finish_reason != "tool_use"``, return the response.
          3. Otherwise, invoke every tool call (validated against the
             tool's Pydantic model), append the assistant + tool-result
             messages to the conversation, and loop.

        Raises:
            ToolLoopExceededError: When ``max_iterations`` is hit without
                the model exiting tool-use mode.
        """
        if max_iterations < 1:
            err = f"max_iterations must be >= 1, got {max_iterations}"
            raise ValueError(err)
        self._check_tool_capability(tools)

        tools_by_name: dict[str, Tool] = {t.name: t for t in tools}
        history: list[AnyMessage] = list(messages)

        for _ in range(max_iterations):
            response = await self.complete(
                messages=history,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                tools=tools,
                provider_extras=provider_extras,
            )
            if response.finish_reason != "tool_use" or not response.tool_calls:
                return response

            history.append(
                AssistantMessage(content=response.text or None, tool_calls=response.tool_calls)
            )
            for call in response.tool_calls:
                tool_obj = tools_by_name.get(call.name)
                if tool_obj is None:
                    history.append(
                        ToolResultMessage(
                            tool_call_id=call.id,
                            content=f"Tool {call.name!r} is not registered",
                            is_error=True,
                        )
                    )
                    continue
                try:
                    result = await tool_obj.invoke(call.arguments)
                except Exception as exc:
                    history.append(
                        ToolResultMessage(
                            tool_call_id=call.id,
                            content=f"{type(exc).__name__}: {exc}",
                            is_error=True,
                        )
                    )
                    continue
                history.append(
                    ToolResultMessage(
                        tool_call_id=call.id,
                        content=_stringify_tool_result(result),
                    )
                )

        # `last_response` is unreachable as None since max_iterations >= 1 was
        # enforced above; raise unconditionally if we exit the loop.
        raise ToolLoopExceededError(
            f"Tool loop did not terminate within {max_iterations} iterations",
            max_iterations=max_iterations,
            iterations=max_iterations,
        )

    # --- Internals --------------------------------------------------------

    def _check_tool_capability(self, tools: Sequence[Tool] | None) -> None:
        if not tools:
            return
        for entry in self._chain:
            try:
                model_entry = registry.get(entry.model)
            except RegistryError:
                # Unknown models surface during fallback; skip here.
                continue
            if not model_entry.capabilities.tool_calling:
                err = f"Model {entry.model!r} does not support tool calling per the registry"
                raise RegistryError(err, model=entry.model, reason="capability_missing")

    async def _invoke(
        self,
        *,
        route: ModelRoute,
        messages: Sequence[AnyMessage],
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        tools: Sequence[Tool] | None,
        response_format: dict[str, Any] | None,
        provider_extras: Mapping[ProviderName, Mapping[str, Any]] | None,
    ) -> LLMResponse:
        """One provider attempt — actual HTTP via the provider client."""
        provider_client = self._provider_clients[route.provider]
        wire = [_message_to_wire(m, route.provider) for m in messages]
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            kwargs["top_p"] = top_p
        if tools is not None:
            kwargs["tools"] = _tools_for_provider(tools, route.provider)
        if response_format is not None:
            kwargs["response_format"] = response_format
        if provider_extras is not None:
            extras = provider_extras.get(route.provider)
            if extras is not None:
                kwargs.update(extras)

        start = time.perf_counter()
        try:
            raw = await provider_client.acompletion(
                provider_model_id=route.provider_model_id,
                messages=wire,
                **kwargs,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            mapped = map_litellm_exception(exc, model=route.model, provider=route.provider)
            await self._record_failure(
                route=route,
                wire=wire,
                latency_ms=latency_ms,
                error=mapped,
            )
            raise mapped from exc

        latency_ms = (time.perf_counter() - start) * 1000
        response = _normalize_response(raw, route=route, latency_ms=latency_ms)
        await self._record_success(route=route, wire=wire, response=response)
        return response

    async def _consume_budget(self, response: LLMResponse) -> None:
        budget = current_budget()
        if budget is None:
            return
        await budget.consume(
            usd=response.cost_usd,
            tokens=response.usage.total_tokens,
        )

    async def _record_success(
        self,
        *,
        route: ModelRoute,
        wire: list[dict[str, Any]],
        response: LLMResponse,
    ) -> None:
        record = DiagnosticRecord(
            timestamp=utcnow_iso(),
            correlation_id=get_correlation_id(),
            request_hash="",  # filled by callers that compute one upstream
            model=route.model,
            provider=route.provider,
            provider_model_id=route.provider_model_id,
            messages=wire,
            response_text=response.text,
            tool_calls=[_tool_call_to_wire(tc) for tc in response.tool_calls],
            finish_reason=response.finish_reason,
            usage=response.usage.model_dump(),
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            cache_hit=response.cache_hit,
            error=None,
        )
        await write_diagnostic_record(record)

    async def _record_failure(
        self,
        *,
        route: ModelRoute,
        wire: list[dict[str, Any]],
        latency_ms: float,
        error: BaseException,
    ) -> None:
        record = DiagnosticRecord(
            timestamp=utcnow_iso(),
            correlation_id=get_correlation_id(),
            request_hash="",
            model=route.model,
            provider=route.provider,
            provider_model_id=route.provider_model_id,
            messages=wire,
            response_text="",
            tool_calls=[],
            finish_reason=None,
            usage=None,
            cost_usd=None,
            latency_ms=latency_ms,
            cache_hit=False,
            error=make_error_field(error),
        )
        await write_diagnostic_record(record)


# ---------------------------------------------------------------------------
# Structured-output dispatch helpers (per-provider)
# ---------------------------------------------------------------------------


def _structured_payload_for_route(
    route: ModelRoute,
    schema: type[BaseModel],
) -> tuple[
    dict[str, Any] | None,
    Sequence[Tool] | None,
    Mapping[ProviderName, Mapping[str, Any]] | None,
]:
    """Build the provider-specific payload to coax structured output.

    Returns ``(response_format, tools, provider_extras)`` — any of which
    may be ``None`` for that provider.
    """
    if route.provider == "openai" or route.provider == "azure" or route.provider == "openai_compat":
        return to_openai_response_format(schema), None, None

    if route.provider == "vertex" and not route.provider_model_id.startswith("claude"):
        gemini_schema = to_gemini_response_schema(schema)
        # Gemini uses generation_config.response_schema and response_mime_type.
        extras: dict[ProviderName, dict[str, Any]] = {
            "vertex": {
                "response_format": {"type": "json_object"},
                "response_mime_type": "application/json",
                "response_schema": gemini_schema,
            }
        }
        return None, None, extras

    # Anthropic, Bedrock, or Claude-on-Vertex: forced-tool emulation.
    _tool_schema, _tool_choice = to_anthropic_forced_tool_schema(schema)
    # We don't have a public "force this tool" knob on `complete()` — the
    # caller passes the tool via tools= and we route via provider_extras.
    # For Phase 1, surface the forced tool through provider_extras so it
    # lands directly in LiteLLM's payload.
    extras = {
        route.provider: {
            "tools": [_tool_schema],
            "tool_choice": _tool_choice,
        }
    }
    return None, None, extras


def _extract_structured_text(response: LLMResponse, *, schema_name: str) -> str:
    """Pull out the JSON payload from a structured-output response.

    For OpenAI/Vertex(Gemini), the text is in ``response.text``. For the
    forced-tool emulation path, it's the tool call's arguments.
    """
    import json

    if response.text:
        return response.text
    # Forced-tool emulation: look for a tool call matching the schema name.
    for call in response.tool_calls:
        if call.name == schema_name:
            return json.dumps(call.arguments)
    # Fall back: any tool call works (Anthropic may name it differently).
    if response.tool_calls:
        return json.dumps(response.tool_calls[0].arguments)
    err = "Structured response contained neither text nor a tool call"
    raise StructuredOutputError(err, attempts=1)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _flatten_extras(
    provider_extras: Mapping[ProviderName, Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    """Flatten the per-provider extras into a single canonical dict.

    Used only for the cache key — sorting by provider name keeps the
    digest stable across logically equivalent inputs.
    """
    if provider_extras is None:
        return None
    return {provider: dict(kwargs) for provider, kwargs in sorted(provider_extras.items())}


def _merge_extras(
    a: Mapping[ProviderName, Mapping[str, Any]] | None,
    b: Mapping[ProviderName, Mapping[str, Any]] | None,
) -> dict[ProviderName, dict[str, Any]] | None:
    if a is None and b is None:
        return None
    out: dict[ProviderName, dict[str, Any]] = {}
    for src in (a, b):
        if src is None:
            continue
        for provider, kwargs in src.items():
            out.setdefault(provider, {}).update(kwargs)
    return out


def _stringify_tool_result(result: Any) -> str:
    import json

    if isinstance(result, str):
        return result
    try:
        return json.dumps(result)
    except TypeError, ValueError:
        return str(result)
