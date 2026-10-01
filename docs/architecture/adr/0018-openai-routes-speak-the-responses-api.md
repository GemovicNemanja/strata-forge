# 0018 — OpenAI routes speak the Responses API through `litellm.aresponses`

**Status:** Accepted
**Extends:** [ADR 0001](0001-litellm-as-transport.md) (LiteLLM stays the transport)

## Context

Every provider call went through `litellm.acompletion`, so the `openai` and `azure` routes spoke
OpenAI's Chat Completions API. OpenAI's current models no longer take tools there:

- "GPT-6 Astra and GPT-6.1 Sol support Chat Completions, but tool calling requires Responses."
  (developers.openai.com/api/docs/guides/latest-model)
- "Starting with GPT-5.4, Chat Completions does not support tool calling with `reasoning_effort`
  values other than `none`" (developers.openai.com/api/docs/guides/migrate-to-responses), and
  GPT-6 Luna and GPT-5.5 default to `medium`.
- gpt-5.5-pro is not served on Chat Completions at all.

A registry flag (`tool_call_reasoning_effort`) that forced effort `none` on tool calls covered one
model, at the cost of disabling its reasoning whenever it used tools. Chat Completions remains
supported, but OpenAI recommends the Responses API for reasoning models, and an agent loop on
these models needs tools.

The Responses API also changes how reasoning survives between turns. With `store: false` (no
server-side retention) the provider keeps nothing, and a reasoning model's context for the next
turn is the `encrypted_content` of its reasoning items, which the caller must replay in order,
together with each assistant message's `phase`.

LiteLLM 1.83 offers two ways in:

1. A Chat Completions bridge (`openai/responses/<id>`, or automatic for some `gpt-5.N` names).
   It drops an assistant turn's text when the turn also has tool calls, never carries `phase`,
   and turns `response.incomplete` / `response.failed` / `error` events into an empty chunk, so a
   truncated or failed turn reads as a clean stop.
2. `litellm.aresponses`, a pass-through: the OpenAI config applies no mapping, exceptions still go
   through `litellm.exception_type` (so `map_litellm_exception` keeps working), and the streaming
   iterator fires LiteLLM's success and failure callbacks, so the Langfuse tracing in
   `strata_forge.tracing` keeps working.

## Decision

- A registry route carries `wire_api: chat_completions | responses`. `responses` is valid only on
  the `openai` and `azure` providers. Every `openai` route of an OpenAI model uses it; an `azure`
  route uses it where Azure documents the model for its Responses API. `openai_compat`
  (OpenRouter, a self-hosted vLLM, Ollama), Anthropic, Vertex and Bedrock stay on Chat
  Completions.
- `ProviderClient` gains `aresponses` / `aresponses_stream` over `litellm.aresponses`. Forge builds
  the request body and parses the typed event stream itself (`strata_forge.llm.responses_wire`);
  the bridge in (1) is not used. The official `openai` SDK is not added as a dependency.
- Requests are stateless: `store: false`, `include: ["reasoning.encrypted_content"]`, never
  `previous_response_id`. Each turn's output items (encrypted reasoning, text with its `phase`,
  function-call references) ride on `AssistantMessage.provider_items`, a closed, typed union
  tagged with the provider that produced it. They are replayed verbatim to that provider and
  replaced by plain text plus `function_call` items for any other, so a provider switch or a
  caller that drops them degrades the reasoning context, never the request.
- A failure inside the stream (`response.failed`, `error`) raises a `ProviderError` subclass, a
  `response.incomplete` turn reports `length` or `content_filter`, and a stream that ends before
  its terminal event raises. Truncation is never reported as success.
- LiteLLM fakes a stream (one blocking call, replayed as deltas) for a model its model map lacks,
  and its bundled map trails OpenAI's lineup, so streaming would depend on its import-time fetch
  of the remote map. Before a Responses call Forge registers a model LiteLLM cannot look up from
  its own registry entry (streaming flag, limits, prices); a model LiteLLM maps is left alone.
- No tool-capable OpenAI model has a Chat Completions route: an `azure` route exists only where
  Azure documents the model for its Responses API.
- `capabilities.sampling_params` records whether a model takes `temperature` / `top_p` at its
  default reasoning effort; the client refuses them pre-flight where the provider would reject
  them.
- The LiteLLM floor rises to the version the Responses path is tested against (`>=1.83,<2`);
  releases before 1.66 have no `aresponses`.

## Consequences

**Positive**

- Every registered OpenAI model calls tools at its default reasoning effort, and the
  effort-pinning flag is gone.
- Reasoning context survives a tool turn without OpenAI retaining any conversation state.
- Truncated and failed turns are reported truthfully.

**Negative**

- Forge owns a second wire format: the request builder and the event parser must track the
  Responses API. The parser is written against the documented event types and also accepts a
  stream LiteLLM fakes from a non-streamed call (a model registered with `streaming: false`),
  where the terminal event carries everything.
- Forge writes into LiteLLM's process-wide model map. The entries are confined to models LiteLLM
  does not map and carry no `mode`, so they change only streaming and LiteLLM's own cost figure
  for those models.
- `provider_items` are opaque and large (the encrypted reasoning). A caller that persists a
  conversation between calls (a continuation token, a database row) carries them or loses the
  reasoning context; it must never edit them, because a modified blob fails the next request.
- `max_output_tokens` bounds reasoning as well as output, so a caller's token budget for a
  reasoning model must be much larger than its expected answer.

## Alternatives considered

1. **The LiteLLM Chat Completions bridge.** Smallest change, but it loses preamble text and
   `phase` and hides incomplete and failed responses. Rejected.
2. **The official `openai` SDK inside `providers/`.** A new pinned dependency, its own exception
   mapping and its own tracing hook, with nothing `litellm.aresponses` lacks. Rejected.
3. **`previous_response_id` with stored responses.** Smaller requests, but OpenAI then retains
   every conversation for at least 30 days, and a resumed conversation depends on a server-side
   object the caller cannot inspect or move between keys. Rejected.
