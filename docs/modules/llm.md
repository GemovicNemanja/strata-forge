# `forge.llm` — async LLM client

The `forge.llm` package is the typed, async LLM surface used by every other
module that needs to talk to a model. It wraps LiteLLM ([ADR 0001]) and adds
the layer that Forge actually wants to depend on: Pydantic messages and
responses, structured output, tool calling, multimodal input, streaming,
two-axis fallback, a provider-agnostic cache, a model registry, cost
accounting, an NDJSON diagnostic dump, and budget integration.

This document is the canonical API reference for the module. The
architectural rationale for individual decisions lives in the ADRs linked
inline. The implementation lives under `src/forge/llm/`; module-specific
agent rules live in [`src/forge/llm/CLAUDE.md`](../../src/forge/llm/CLAUDE.md).

> **TL;DR.** Build an `LLMClient`, call `complete` / `stream` /
> `complete_structured` / `run_tool_loop` / `stream_tool_loop`. Configure provider-level and
> model-level fallback with `with_fallbacks`. Wrap call sites in a
> `BudgetContext` for cost ceilings. Use the `[redis]` extra if you want
> the cache to outlive a single process.

[ADR 0001]: ../architecture/adr/0001-litellm-as-transport.md

---

## Contents

- [Quickstart](#quickstart)
- [Model registry](#model-registry)
- [Public API](#public-api)
  - [`LLMClient`](#llmclient)
  - [Messages and content parts](#messages-and-content-parts)
  - [Responses](#responses)
- [Routing](#routing)
- [Two-axis fallback](#two-axis-fallback)
- [Cache](#cache)
- [Structured output](#structured-output)
- [Tool calling](#tool-calling)
- [Multimodal](#multimodal)
- [Streaming](#streaming)
- [Cost, budgets, and the diagnostic dump](#cost-budgets-and-the-diagnostic-dump)
- [Errors](#errors)
- [Sync wrappers](#sync-wrappers)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```python
from forge.llm import LLMClient, Message

client = LLMClient("claude-opus-4-7")
resp = await client.complete([Message.user("Say hello.")])
print(resp.text, resp.usage.total_tokens, resp.cost_usd)
```

The first positional argument is a logical model name registered in
`registry_data.yaml`. Without an explicit `provider=`, Forge takes the
registry's default route for that model. Pin a specific provider when you
want the call to go through it:

```python
client = LLMClient("claude-opus-4-7", provider="bedrock")
```

Every method on `LLMClient` is `async def`. Sync wrappers for CLI and
notebook ergonomics live in [`forge.sync`](#sync-wrappers).

---

## Model registry

The registry is curated by [ADR 0004]: latest foundation models from
Anthropic, OpenAI, and Google. Adding new vendors requires a superseding
ADR. Pricing is in USD per million tokens.

| Logical name | Vendor | Tier | Context | Routes | Input / Output |
|---|---|---|---|---|---|
| `claude-opus-4-7` | Anthropic | flagship | 1 M | anthropic (default), bedrock, vertex | $5.00 / $25.00 |
| `claude-sonnet-4-6` | Anthropic | balanced | 1 M | anthropic (default), bedrock, vertex | $3.00 / $15.00 |
| `claude-haiku-4-5` | Anthropic | fast | 200 K | anthropic (default), bedrock, vertex | $1.00 / $5.00 |
| `gpt-5.5` | OpenAI | flagship | 400 K | openai (default), azure | $5.00 / $30.00 |
| `gpt-5.5-pro` | OpenAI | reasoning | 400 K | openai (default), azure | $30.00 / $180.00 |
| `gpt-5.5-thinking` | OpenAI | reasoning | 400 K | openai (default), azure | $5.00 / $30.00 |
| `gpt-5.5-instant` | OpenAI | fast | 128 K | openai (default), azure | $1.25 / $10.00 |
| `gemini-3.1-pro` | Google | flagship | 2 M | vertex (default) | $2.00 / $12.00 |
| `gemini-3.1-flash-lite` | Google | fast | 1 M | vertex (default) | $0.10 / $1.00 |

Aliases (`opus`, `sonnet`, `haiku`, `gpt55`, `gemini-pro`, …) resolve to
their canonical name before lookup. The source of truth is
[`src/forge/llm/registry_data.yaml`](../../src/forge/llm/registry_data.yaml);
edits to it must keep YAML and this table in sync.

Programmatic access:

```python
from forge.llm import registry

model = registry.get("opus")        # alias -> Model("claude-opus-4-7")
all_models = registry.list_models() # 9 entries
default_route = model.default_route()
```

[ADR 0004]: ../architecture/adr/0004-model-registry-scope.md

---

## Public API

### `LLMClient`

```python
client = LLMClient(
    model,                       # str — logical model name (or use chain=)
    provider=None,               # ProviderName — pin route (default route otherwise)
    *,
    chain=None,                  # Sequence[FallbackEntry] — mutually exclusive with model
    cache=None,                  # CacheBackend | None
    provider_clients=None,       # Mapping[ProviderName, ProviderClient] | None
    retry_max_attempts=5,
    retry_initial_wait=1.0,
    retry_max_wait=30.0,
    strict_bad_request=False,
    require_tool_support=False,  # raise pre-flight for unconfirmable tool models
)
```

Pass **exactly one** of `model=` or `chain=` — they're mutually exclusive.
Use [`LLMClient.with_fallbacks`](#two-axis-fallback) as a friendlier
constructor for chains.

Methods:

| Method | Returns | Bypasses cache? | Bypasses fallback? |
|---|---|---|---|
| `complete(messages, *, ...)` | `LLMResponse` | no | no |
| `complete_structured(messages, *, schema, ...)` | `StructuredResponse[M]` | no | no |
| `stream(messages, *, ...)` | `AsyncIterator[ResponseChunk]` | yes | yes |
| `run_tool_loop(messages, *, tools, max_iterations=8, ...)` | `LLMResponse` | no (per attempt) | no |
| `stream_tool_loop(messages, *, tools, max_iterations=8, ...)` | `AsyncIterator[LoopEvent]` | yes | yes |

All methods validate the conversation (every `ToolResultMessage`
references a prior `ToolCall.id`) before dispatching, and surface
provider errors as `ProviderError` subclasses — never raw LiteLLM
exceptions. The one exception to the *raise* convention is
`stream_tool_loop`: an in-stream provider error becomes a terminal
`LoopError` event rather than a raised exception (see
[Streaming tool loop](#streaming-tool-loop)).

### Messages and content parts

Every conversation is a sequence of `AnyMessage` — one of
`SystemMessage`, `UserMessage`, `AssistantMessage`, `ToolResultMessage`.
The ergonomic way to build them is via the `Message` namespace:

```python
from forge.llm import Message

conversation = [
    Message.system("You are a concise assistant."),
    Message.user("Two facts about kangaroos."),
]
```

User content can be either a plain `str` or a list of `ContentPart`
instances — `TextPart` and `ImageContent` are the built-in concrete
parts. See [Multimodal](#multimodal) for the image case.

`AssistantMessage` carries both `content: str | None` and
`tool_calls: list[ToolCall]`. `ToolResultMessage` carries
`tool_call_id`, `content`, and `is_error: bool`. The full hierarchy
lives in [`messages.py`](../../src/forge/llm/messages.py).

### Responses

`LLMResponse` (frozen Pydantic model):

| Field | Type | Notes |
|---|---|---|
| `text` | `str` | The assistant's text reply. Empty when only tool calls were emitted. |
| `tool_calls` | `list[ToolCall]` | One per tool call requested. |
| `finish_reason` | `FinishReason` | `"stop"` / `"tool_use"` / `"length"` / `"content_filter"` / `"error"`. |
| `usage` | `Usage` | `input_tokens`, `output_tokens`, plus cache read/write counters. |
| `cost_usd` | `float` | Computed against the registry's per-million rates. `0.0` on a cache hit. |
| `route` | `ModelRoute` | The `(model, provider, provider_model_id)` actually used. |
| `cache_hit` | `bool` | `True` when served from cache. |
| `latency_ms` | `float` | Wall-clock latency of the provider call. |

`StructuredResponse[M]` is `LLMResponse` plus `parsed: M`, where `M` is
the Pydantic model passed to `complete_structured`.

---

## Routing

`forge.llm.routing.resolve(model, provider=None)` translates a logical
`(model, provider)` request into a concrete `ModelRoute`:

```python
from forge.llm import resolve

route = resolve("opus", provider="bedrock")
# ModelRoute(model="claude-opus-4-7",
#            provider="bedrock",
#            provider_model_id="anthropic.claude-opus-4-7")
```

Aliases are resolved before lookup. Unknown models raise
`RegistryError(reason="unknown_model")`; an unsupported `(model,
provider)` combo raises `RegistryError(reason="unsupported_route")`.

---

## Two-axis fallback

[ADR 0005] defines two orthogonal failover axes:

- **Provider-level (inner loop):** within one `ModelFallback`, try the
  same logical model across `providers=(...)` in order.
- **Model-level (outer loop):** when every provider for an entry has
  been exhausted, advance to the next entry — which may be a different
  logical model.

```python
from forge.llm import LLMClient, ModelFallback

# Same Claude model, three provider routes; then drop to GPT.
client = LLMClient.with_fallbacks([
    ModelFallback(
        model="claude-opus-4-7",
        providers=("anthropic", "bedrock", "vertex"),
    ),
    ModelFallback(model="gpt-5.5", providers=("openai", "azure")),
])
```

Bare strings expand to "default route only, no provider-level failover":

```python
client = LLMClient.with_fallbacks(["claude-opus-4-7", "gpt-5.5"])
```

Per-error rules (see also the docstring on `run_with_fallback`):

| Error | Default | Strict (`strict_bad_request=True`) |
|---|---|---|
| `ProviderContentFilterError` | re-raise (terminal for the whole chain) | re-raise |
| `ProviderBadRequestError` | advance | re-raise, wrapped in `FallbackExhaustedError` |
| `ProviderAuthError` | advance | advance |
| `ProviderRateLimitError` | retry + advance | retry + advance |
| `ProviderTimeoutError` | retry + advance | retry + advance |
| `ProviderServerError` | retry + advance | retry + advance |
| `RegistryError` (from `resolve()`) | advance | advance |
| Anything else | propagate unchanged | propagate unchanged |

Each provider attempt is wrapped in `forge.core.retry.retry` (exponential
backoff + jitter), so a transient rate-limit retries before the provider
is counted as exhausted. The retry policy is configured by the
`retry_max_attempts` / `retry_initial_wait` / `retry_max_wait` kwargs on
`LLMClient`.

When the entire chain fails, `FallbackExhaustedError.causes` carries
every `(model, provider, error)` triple in attempt order for
postmortem inspection.

[ADR 0005]: ../architecture/adr/0005-two-axis-fallback.md

---

## Cache

The cache key is computed by `cache_key(...)` over the **logical** model
name plus the canonical request — sampling params, response_format, tool
schemas, and any `provider_extras`. The provider serving the call is
**not** in the key. That's intentional: a hit cached when Anthropic
served the call is just as valid when the next call would have gone to
Bedrock — including the provider would defeat the cache during
provider-level failover.

```python
from forge.llm import InMemoryCache, LLMClient, Message

cache = InMemoryCache(max_size=1024)
client = LLMClient("claude-opus-4-7", cache=cache)

first  = await client.complete([Message.user("hi")])      # MISS
second = await client.complete([Message.user("hi")])      # HIT (same logical key)
# second.cache_hit == True; second.cost_usd == 0.0
```

Backends:

- `InMemoryCache(max_size=1024)` — bounded LRU, in-process.
- `RedisCache(url=..., prefix="forge:llm:", ttl_seconds=None)` —
  requires the `[redis]` extra. Lazy-imports `redis.asyncio`; raises a
  helpful `ImportError` if the extra is missing. Values are pickled
  (treat the Redis instance as trusted).

Streaming **never** hits the cache: chunks have provider-specific
shapes and no useful "final form" mid-stream.

---

## Structured output

`complete_structured` returns a Pydantic-validated instance plus the
usual `LLMResponse` fields:

```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    bullets: list[str]

resp = await client.complete_structured(
    [Message.user("Summarize: ...")],
    schema=Summary,
)
print(resp.parsed.title, resp.parsed.bullets)
```

Dispatch by route:

- **OpenAI / Azure / openai_compat** — sets `response_format` to OpenAI's
  strict `json_schema` payload. The schema is tightened to the strict
  mode subset (`additionalProperties: false`, every property required).
- **Vertex (Gemini)** — uses `response_mime_type="application/json"` and
  `response_schema` with the OpenAPI subset of JSON Schema Gemini accepts.
- **Anthropic / Bedrock / Claude-on-Vertex** — no native channel; emits
  a forced tool call whose `input_schema` mirrors the desired shape, then
  parses the tool-call arguments back through the Pydantic model.

If the model emits text that fails Pydantic validation,
`complete_structured` reprompts with the parse error included up to
`max_reprompt_attempts` (default 3) times. If every attempt fails, the
call raises `StructuredOutputError` with `attempts` and `last_error`.

For a hand-rolled reprompt loop, the building blocks are exported:
`pydantic_to_json_schema`, `to_openai_response_format`,
`to_anthropic_forced_tool_schema`, `to_gemini_response_schema`,
`make_reprompt_instruction`, `parse_json_response`.

---

## Tool calling

[ADR 0006] makes tool calling first-class in `forge.llm`. Tools are
async functions with a single Pydantic-typed argument; the `@tool`
decorator wraps one into a `Tool` instance.

```python
from pydantic import BaseModel, Field
from forge.llm import tool

class WeatherArgs(BaseModel):
    location: str = Field(..., description="City, country")

@tool
async def get_weather(args: WeatherArgs) -> dict:
    """Get the current weather for a location."""
    return {"temp_c": 18, "city": args.location}
```

The decorator extracts the Pydantic args model from the function's
annotation (works under `from __future__ import annotations` via
`typing.get_type_hints`); name defaults to `func.__name__`; description
defaults to the docstring.

### Single call (manual invoke)

```python
resp = await client.complete(
    [Message.user("Weather in Tokyo?")],
    tools=[get_weather],
)
if resp.finish_reason == "tool_use":
    for call in resp.tool_calls:
        result = await get_weather.invoke(call.arguments)
        # `invoke` validates args via the Pydantic model before calling
        # the wrapped function; bad args raise `ValidationError`.
```

### Multi-turn loop

`run_tool_loop` does the whole conversation for you:

```python
final = await client.run_tool_loop(
    [Message.user("Weather in Tokyo, then convert to Fahrenheit.")],
    tools=[get_weather, celsius_to_fahrenheit],
    max_iterations=8,
)
print(final.text)
```

Per iteration: run `complete()`, and if `finish_reason == "tool_use"`,
validate + invoke each tool, append the assistant message and the
tool-result messages to the history, loop. Tool exceptions and unknown
tool names are surfaced to the model as `is_error=True` results rather
than terminating the loop. If the model never exits tool-use mode,
`ToolLoopExceededError` carries the iteration cap.

### Capability gate

If you pass `tools=` to a model whose registry entry has
`capabilities.tool_calling = False`, the call raises
`RegistryError(reason="capability_missing")` **before** any HTTP
happens. No silent fallback to "the model will probably ignore it."

An `openai_compat` / OpenRouter model is **not** in the curated registry
(its ids are operator-specific — [ADR 0004]), so its tool-calling
capability can't be confirmed. By default such a model is let through and
the provider decides at call time. Pass `require_tool_support=True` to the
`LLMClient` constructor to instead raise
`RegistryError(reason="capability_unknown")` pre-flight for any
unconfirmable model — letting a caller (e.g. a server driving an agent
loop) cleanly degrade to a tool-less path rather than hit an opaque
provider rejection mid-stream.

### Provider serialization

`Tool.to_openai_schema()`, `to_anthropic_schema()`, `to_gemini_schema()`
return the three native wire shapes. `LLMClient` picks the right one for
the resolved route automatically.

[ADR 0006]: ../architecture/adr/0006-tool-calling-as-llm-primitive.md

---

## Multimodal

`ImageContent` is the multimodal building block:

```python
from forge.llm import ImageContent, Message, TextPart, UserMessage

await client.complete([
    UserMessage(content=[
        TextPart(text="What's in this image?"),
        ImageContent.from_url("https://example.com/cat.jpg"),
    ]),
])
```

Construction options: `from_url(url)`, `from_path(path)` (auto-detects
MIME), `from_bytes(data, mime_type)`. Each provider gets the image in
its native wire format — OpenAI's `image_url`, Anthropic's `image`
content block with `source.type`, Gemini's `inline_data` /
`file_data`. The model registry's `capabilities.vision` flag indicates
which models accept images.

A `downscale_image(data, *, max_dimension=2048, quality=85)` helper is
available behind the `[multimodal]` extra (Pillow); the example scripts
demonstrate basic usage in
[`examples/03_image_input.py`](../../examples/03_image_input.py).

---

## Streaming

```python
async for chunk in await client.stream([Message.user("Write a haiku.")]):
    print(chunk.delta_text, end="", flush=True)
```

Each `ResponseChunk` carries `delta_text`, `delta_tool_calls`, and
optional `finish_reason` + `usage` (present on the final chunk when the
provider reports them). The streaming utilities in
[`streaming.py`](../../src/forge/llm/streaming.py) provide accumulators
for the common postprocessing patterns:

- `accumulate_text(chunks)` → the concatenated text.
- `accumulate_tool_calls(chunks)` → the assembled `list[ToolCall]`
  (validates that every tool call has an id, a name, and JSON-object
  arguments).
- `StreamingToolCallAccumulator` — the incremental form: feed each
  chunk's `delta_tool_calls` via `add(...)` while doing other work, then
  `finalize()` to rebuild the `list[ToolCall]` at an iteration boundary.
  Backs `stream_tool_loop`.
- `JSONAccumulator` — feed `delta_text` fragments, then `parse()` when
  the buffer is a complete JSON value.

Streaming **bypasses cache and fallback** — a stream midway through
can't be cleanly transferred to a new provider. Use the async API
directly (sync wrappers buffer the whole stream before yielding, by
necessity).

### Streaming tool loop

`stream_tool_loop` is the streaming counterpart of `run_tool_loop`: it
drives the same multi-turn tool-use conversation, but instead of
buffering it into one final `LLMResponse`, it yields a flat stream of
typed `LoopEvent`s as the conversation unfolds — so a UI can show the
model's tool use live.

```python
from forge.llm import (
    LLMClient, Message, IterationStart, TextDelta,
    ToolCallStarted, ToolResult, Done, LoopError,
)

client = LLMClient("claude-opus-4-7")
async for event in client.stream_tool_loop(
    [Message.user("Weather in Tokyo, then convert to Fahrenheit.")],
    tools=[get_weather, celsius_to_fahrenheit],
):
    match event:
        case TextDelta(text=t):              print(t, end="")
        case ToolCallStarted(name=n):        ...   # tool about to run
        case ToolResult(name=n, is_error=e): ...   # tool returned
        case Done(finish_reason=r):          ...   # success, terminal
        case LoopError(message=m):           ...   # failure, terminal
```

The event union lives in `forge.llm.loop_events`; every event carries a
`type` literal (the discriminator):

| Event | Emitted | Fields |
|---|---|---|
| `IterationStart` | at the top of each iteration | `index` (0-based) |
| `TextDelta` | per assistant text fragment | `text`, `iteration` |
| `ToolCallStarted` | once per call, after its args are fully assembled | `id`, `name`, `arguments`, `iteration` |
| `ToolResult` | once per result, after the tool returns | `id`, `name`, `content`, `is_error`, `iteration` |
| `Done` | terminal — model exited tool-use mode | `finish_reason`, `usage` |
| `LoopError` | terminal — the loop could not complete | `message`, `error_type`, `exceeded_max_iterations` |

Every run ends with **exactly one** terminal event. The tool-result
feedback semantics mirror `run_tool_loop`: an unregistered tool, or a
tool that raises, becomes an `is_error` result fed back to the model,
never an exception out of the loop. An empty `tools` sequence degenerates
to a single streamed turn (one `IterationStart`, its `TextDelta`s, a
`Done`).

Two things distinguish it from `run_tool_loop`:

- **It is a true async generator** — iterate it directly
  (`async for event in client.stream_tool_loop(...)`); do **not** `await`
  the call first, unlike `stream`.
- **Raise vs emit.** Pre-flight failures — a bad `max_iterations`, or a
  capability-gate violation — *raise* synchronously, matching
  `run_tool_loop`. Failures that occur once events are already flowing —
  a provider/transport error, a malformed streamed tool call, or the
  iteration cap — surface as a terminal `LoopError` (with
  `exceeded_max_iterations=True` for the cap), so an in-flight stream
  always ends cleanly rather than raising mid-iteration.

`usage` on `Done` is the final iteration's usage only — not a sum across
iterations; cumulative accounting is the caller's job via tracing. See
[ADR 0014](../architecture/adr/0014-streaming-tool-loop-event-protocol.md)
for the event-protocol rationale.

---

## Cost, budgets, and the diagnostic dump

### Cost

Every `LLMResponse` carries `cost_usd` computed against the registry's
per-million-token rates, taking into account cache-read / cache-write
prices when usage reports them. `cache_hit=True` responses report
`cost_usd=0.0`.

### Budgets

`BudgetContext` from `forge.core.budget` enforces a USD or token
ceiling around a block of code:

```python
from forge.core.budget import BudgetContext

async with BudgetContext(max_usd=1.00) as budget:
    await client.complete([...])
    await client.complete([...])
    print(budget.spent_usd, "spent so far")
```

`LLMClient` consumes against the active budget after every successful
call. If the next call would exceed the ceiling, `BudgetExceededError`
fires *before* the spend happens. Nested budgets share the parent
ceiling unless explicitly `isolated=True`.

### Diagnostic NDJSON dump

Set `FORGE_DIAGNOSTIC_ENABLED=true` (and optionally
`FORGE_DIAGNOSTIC_PATH=./forge-diagnostic.ndjson`) to append one
JSON line per LLM attempt — including each iteration of a tool loop.
Fields: timestamp, correlation_id, request hash, model, provider
route, messages, response text, tool calls, finish_reason, usage,
cost, latency, cache_hit, error (on failure).

The record schema (`DiagnosticRecord` in
[`diagnostic.py`](../../src/forge/llm/diagnostic.py)) is plain
JSON-serializable so replay/analytics scripts don't need to import
Forge.

---

## Errors

Every exception raised from `forge.llm` is a `ForgeError` subclass — no
raw LiteLLM exceptions ever bubble out. The seam is
[`map_litellm_exception`](../../src/forge/llm/errors.py).

| Exception | When it fires |
|---|---|
| `ProviderAuthError` | 401 / 403 from the provider. |
| `ProviderRateLimitError` | 429. Retried before counting as exhausted. |
| `ProviderTimeoutError` | Request deadline elapsed. Retried. |
| `ProviderBadRequestError` | 400 — params the provider rejects. Default: advance; strict: abort. |
| `ProviderServerError` | 5xx. Retried. |
| `ProviderContentFilterError` | Content policy hit. Terminal — short-circuits the whole chain. |
| `BudgetExceededError` | Active `BudgetContext` would be exceeded. |
| `ValidationError` | Pydantic validation failed (tool args, structured-output parse). |
| `CacheError` | Backend operation failed (Redis unreachable, etc.). |
| `RegistryError` | Unknown model, unsupported route, missing capability, ... |
| `FallbackExhaustedError` | Every entry in a chain failed. `.causes` carries the route history. |
| `ToolLoopExceededError` | `run_tool_loop` hit `max_iterations`. |
| `StructuredOutputError` | Reprompt loop exhausted without producing valid JSON. |

---

## Sync wrappers

Every async surface has a sync facade in `forge.sync` for CLI and
notebook use:

```python
from forge import sync
from forge.llm import Message

resp = sync.complete([Message.user("hi")], model="claude-opus-4-7")
parsed = sync.complete_structured([...], schema=Summary, model="gpt-5.5")
for chunk in sync.stream([...], model="claude-opus-4-7"):
    print(chunk.delta_text, end="")
final = sync.run_tool_loop([...], tools=[get_weather], model="gpt-5.5")
```

Each wraps a single `asyncio.run`. Construction style: pass `client=` to
reuse a pre-built `LLMClient`, or pass `model=` / `provider=` / `chain=`
for a one-shot. Sync `stream()` drains every chunk into a list before
returning the iterator — `asyncio.run` is single-shot, so true
chunk-by-chunk sync streaming is structurally impossible. Use the async
API when you need incremental output.

---

## Troubleshooting

**`RegistryError: Unknown model: 'gpt-5'`.**
The registry only knows the names listed above. Use the closest
canonical name or one of its registered aliases (`gpt55`, `opus`, …).

**`RegistryError: ... has no route for provider 'X'`.**
The `(model, provider)` combo isn't a registered route. Check the
[registry table](#model-registry) — e.g. Gemini doesn't have an
Anthropic or Bedrock route.

**`RegistryError: ... does not support tool calling`.**
You passed `tools=` to a model whose `capabilities.tool_calling = False`
in the registry. No model in the current registry actually trips this
gate, but tightening capabilities later will surface it here.

**`RegistryError: Tool support for model '...' cannot be confirmed`.**
You constructed the client with `require_tool_support=True` and passed
`tools=` to an `openai_compat` / OpenRouter model the registry doesn't
track (`reason="capability_unknown"`). Either drop `require_tool_support`
(let the provider decide), or route to a model whose tool-calling
capability is known.

**`FallbackExhaustedError` after only one attempt.**
Either you're passing a single-entry chain whose only provider raised a
non-retryable error, or `strict_bad_request=True` is set and a 400
short-circuited the chain. Inspect `exc.causes` for the `(model,
provider, error)` triple.

**`ProviderContentFilterError` skipping the rest of the chain.**
This is intentional — the same content fails on every other provider, so
retrying is wasted work. Adjust the prompt or accept the refusal.

**Cache hits don't seem to work across providers.**
They should — `cache_key` excludes the provider on purpose. Verify the
sampling params, `provider_extras`, and tool list are identical between
the two calls; any divergence changes the canonical hash.

**`StructuredOutputError: ... reprompt loop exhausted`.**
The model couldn't produce a schema-conformant response within
`max_reprompt_attempts` (default 3). Inspect `exc.last_error`. Common
causes: schema too restrictive, prompt unclear, or the model literally
can't output JSON reliably — try a stronger model or simplify the schema.

**`ImportError: RedisCache requires the [redis] extra`.**
Install with `pip install ai-forge[redis]` (or `uv sync --extra redis`)
before instantiating `RedisCache`.

**No diagnostic file written.**
The dump is gated on `FORGE_DIAGNOSTIC_ENABLED=true`. Set the env var
and rerun; the file appears at `FORGE_DIAGNOSTIC_PATH` (default
`./forge-diagnostic.ndjson`).
