# Run a tool-calling agent

Tool calling is a primitive of `strata_forge.llm`, not a feature of an agent framework layered on
top. A `Tool` is a name, a description, a Pydantic argument model, and an async function;
`LLMClient` knows how to render it for every provider and how to drive a multi-turn loop over it.
`strata_forge.agents` adds packaging — a system prompt, built-in tools, memory, and multi-agent
composition — over exactly those primitives. Understanding the bottom layer first makes the top
layer obvious.

**Prerequisites.** The base install and one configured provider. Nothing on this page needs an
extra.

**Running the snippets.** A block that ends in `asyncio.run(main())` is a complete program. The
rest are fragments: put them inside an `async def main()` and invoke it the same way, or paste them
into a notebook cell, where top-level `await` is legal.

---

## 1. Declare a tool

The `@tool` decorator turns an async function taking a single Pydantic model into a `Tool`. The
function's docstring becomes the description the model reads, and the model's field descriptions
become the parameter documentation — so write both for the model, not for yourself.

```python
from pydantic import BaseModel, Field

from strata_forge.llm import tool


class LookupArgs(BaseModel):
    sku: str = Field(description="Stock keeping unit, e.g. 'AB-1042'.")


@tool
async def inventory_lookup(args: LookupArgs) -> dict[str, object]:
    """Return the current stock level and warehouse for a SKU."""
    return {"sku": args.sku, "in_stock": 42, "warehouse": "rotterdam"}
```

The decorator raises `TypeError` at import time if the signature is wrong — one parameter,
annotated with a `BaseModel` subclass. `inventory_lookup.to_openai_schema()`,
`.to_anthropic_schema()`, and `.to_gemini_schema()` render the provider-specific forms;
`await inventory_lookup.invoke({"sku": "AB-1"})` validates the arguments and calls the function.

## 2. One call, where the caller decides

Passing `tools=` to `complete` surfaces the model's requests without acting on them. Use this
when a tool needs confirmation, rate limiting, or a human in the loop.

```python
import asyncio

from strata_forge.llm import LLMClient, Message

client = LLMClient("claude-opus-4-8", provider="anthropic")

async def main() -> None:
    response = await client.complete(
        [Message.user("How many AB-1042 do we have?")],
        tools=[inventory_lookup],
    )
    if response.finish_reason == "tool_use":
        for call in response.tool_calls:
            print(call.name, call.arguments)
            print(await inventory_lookup.invoke(call.arguments))

asyncio.run(main())
```

See
[`examples/05_tool_calling.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/05_tool_calling.py).

## 3. Let the loop drive

`run_tool_loop` does the obvious thing repeatedly: complete, invoke every requested tool, append
the assistant and tool-result messages, repeat until the model stops asking. It returns the final
`LLMResponse`.

```python
response = await client.run_tool_loop(
    [Message.user("How many AB-1042 do we have, and where?")],
    tools=[inventory_lookup],
    max_iterations=6,
)
print(response.text)
```

Two behaviours that matter in production:

- **A tool that raises does not break the loop.** The exception is turned into a
  `ToolResultMessage` with `is_error=True` and fed back to the model, which usually corrects
  itself or explains the failure. An unregistered tool name gets the same treatment. This is
  deliberate: a model that mistypes an argument should get a chance to fix it.
- **Hitting the cap raises.** `ToolLoopExceededError` fires when `max_iterations` turns pass
  without the model exiting tool-use mode, carrying `max_iterations` and `iterations` so you can
  log the runaway rather than guess at it.

See
[`examples/06_tool_loop.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/06_tool_loop.py).

## 4. Stream the loop for a UI

`stream_tool_loop` is the same loop as a flat stream of typed events, which is the shape a user
interface consumes: an `IterationStart` per turn, `TextDelta` for assistant text, a
`ToolCallStarted` and `ToolResult` per tool round trip, and exactly one terminal event.

```python
from strata_forge.llm import (
    Done,
    IterationStart,
    LoopError,
    TextDelta,
    ToolCallStarted,
    ToolResult,
)

events = client.stream_tool_loop(
    [Message.user("How many AB-1042 do we have, and where?")],
    tools=[inventory_lookup],
    max_iterations=6,
)

async for event in events:
    match event:
        case IterationStart(index=index):
            print(f"--- turn {index}")
        case TextDelta(text=text):
            print(text, end="", flush=True)
        case ToolCallStarted(name=name, arguments=arguments):
            print(f"\n[call] {name}({arguments})")
        case ToolResult(name=name, content=content, is_error=is_error):
            print(f"[{'error' if is_error else 'ok'}] {name} -> {content}")
        case Done(finish_reason=reason, usage=usage):
            print(f"\n[done] {reason}")
        case LoopError(message=message, exceeded_max_iterations=capped):
            print(f"\n[failed] {message}{' (cap hit)' if capped else ''}")
```

`stream_tool_loop` is a true async generator — iterate it directly. Do **not** `await` the call
first, unlike `client.stream(...)`. See
[`examples/07_streaming_tool_loop.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/07_streaming_tool_loop.py)
and [ADR 0014](../architecture/adr/0014-streaming-tool-loop-event-protocol.md).

## 5. Tools your process cannot execute

Sometimes the tool runs somewhere else — in a browser, on the user's machine, behind a
confirmation dialog. A `ToolDeclaration` is a tool with a schema and no function. Mix them into
`stream_tool_loop`'s `tools` and the loop suspends when the model calls one:

```python
from strata_forge.llm import PendingToolCalls, ToolDeclaration

open_file = ToolDeclaration(
    name="open_file",
    description="Open a file in the user's editor.",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
)

async for event in client.stream_tool_loop(messages, tools=[inventory_lookup, open_file]):
    if isinstance(event, PendingToolCalls):
        # Execute event.calls out of band, then resume with:
        #   messages + event.messages + one ToolResultMessage per pending call,
        #   and max_iterations - event.iterations_used
        break
```

Executable tools requested in the same turn run first, with their `ToolResult` events, before the
run suspends. `PendingToolCalls` carries the conversation delta so far, so resuming is
concatenation rather than replay.
[ADR 0015](../architecture/adr/0015-client-executed-tools-suspend-the-streaming-loop.md) explains
why suspension beats a callback.

## 6. Package it as an Agent

`Agent` binds a name, a client, a system prompt, and a tool set. `run` dispatches to
`run_tool_loop` when there are tools and to a single `complete` when there are not.

```python
from strata_forge.agents import Agent, calculator
from strata_forge.llm import LLMClient

agent = Agent(
    "inventory-desk",
    client=LLMClient("claude-opus-4-8", provider="anthropic"),
    system_prompt="Answer stock questions. Use the tools; never guess a number.",
    tools=[inventory_lookup, calculator],
    max_iterations=8,
)

result = await agent.run("How many AB-1042 do we have, and what is that times 3?")
print(result.text, result.final_response.route.provider)
```

`run_structured(..., output_schema=MyModel)` returns an `AgentResult` whose `parsed` field holds
a validated instance; `run_streaming(...)` yields the same `LoopEvent` stream as
`stream_tool_loop`.

`AgentResult.cost_usd` and `.latency_ms` are the **final** completion's numbers, not the sum
across loop iterations, and `AgentResult.messages` is the conversation that was sent in, without
the intermediate tool-call and tool-result messages. For cumulative accounting across a whole
loop, trace it — see [`docs/modules/tracing.md`](../modules/tracing.md).

## 7. Built-in tools

Four tools ship ready to use. Two are `Tool` instances; two are factories, because they need
configuration you must supply.

| Symbol | Kind | Notes |
|---|---|---|
| `calculator` | `Tool` | Safe arithmetic evaluator over a restricted AST. No `eval`. |
| `fetch_url` | `Tool` | HTTP GET returning truncated body text. Follows redirects. |
| `fs_read_tool(allowed_dirs=[...])` | factory | Reads files under the allowed roots only; resolves symlinks before the check, so escapes fail. |
| `web_search_tool(backend=...)` | factory | You supply an async `(query, max_results) -> Sequence[SearchResult]`; the tool renders results as JSON. There is no bundled search provider. |

```python
from pathlib import Path

from strata_forge.agents import fetch_url, fs_read_tool

reader = fs_read_tool(allowed_dirs=[Path("./docs").resolve()])
agent = Agent("doc-helper", client=client, tools=[reader, fetch_url])
```

`fs_read_tool` raises `ValueError` when `allowed_dirs` is empty — there is no "read anything"
mode, on purpose.

## 8. Memory is the caller's

`Agent` is stateless across runs and takes no memory argument. `ConversationMemory` holds the
history and you pass it in; that keeps the agent re-entrant and makes the truncation policy an
explicit decision rather than a hidden one.

```python
from strata_forge.agents import ConversationMemory

memory = ConversationMemory(system_message=agent.system_prompt)

for turn in ("My name is Sam.", "What did I just tell you?"):
    memory.append_user(turn)
    result = await agent.run(memory.non_system_messages)
    memory.append_assistant(result.text)
    memory.trim_to_tokens(4000, model="claude-opus-4-8")
```

The system message lives in a dedicated slot and survives every trim; `trim_to_messages` and
`trim_to_tokens` only evict from the rolling history. Appending a `SystemMessage` raises.

`EpisodicMemory` is the long-term counterpart: text keyed by embedding, over any `VectorStore`.
The Protocol is the same one `strata_forge.rag` implements, so the Qdrant store from
[Build a RAG pipeline over Qdrant](build-a-rag-pipeline-over-qdrant.md) drops in unchanged.

```python
from strata_forge.agents import EpisodicMemory
from strata_forge.rag import LiteLLMEmbedder, QdrantVectorStore

embedder = LiteLLMEmbedder("text-embedding-3-small")
memory = EpisodicMemory(
    store=QdrantVectorStore(collection_name="agent-memory", embedding_dimensions=1536),
    embed=embedder.embed,
)
await memory.add("Sam prefers teal.", metadata={"turn": 1})
for hit in await memory.search("what colour does Sam like?", top_k=3):
    print(hit.score, hit.item.text)
```

One instance must use one embedding function for both writes and queries — the store holds raw
vectors and cannot detect a mismatch.

See
[`examples/25_agent_memory.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/25_agent_memory.py).

## 9. Compose several agents

Two patterns ship, both plain async functions over `Agent`s rather than a framework.

```python
from strata_forge.agents import critic_refiner_run, handoff

# Route to a specialist. The router picks a key from `specialists` via structured output.
answer = await handoff(
    router=Agent("router", client=fast_client, system_prompt="Pick the best specialist."),
    specialists={"billing": billing_agent, "shipping": shipping_agent},
    user_input="Where is my parcel?",
)

# Draft, critique, refine until the critic approves or max_rounds is hit.
final = await critic_refiner_run(
    drafter=drafter,
    critic=critic,
    user_input="Write a two-sentence product description.",
    max_rounds=3,
)
```

`handoff` returns the chosen specialist's `AgentResult`; the router's intermediate decision is
not exposed in the return value, so trace the run if you need to audit routing. It raises
`ValueError` when `specialists` is empty or the router names something that is not in the map.

See
[`examples/24_agent_basic.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/24_agent_basic.py)
and
[`examples/26_agent_critic_refiner.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/26_agent_critic_refiner.py).

---

## Where to go next

- [`docs/modules/agents.md`](../modules/agents.md) — the full agent surface.
- [`docs/modules/llm.md`](../modules/llm.md) — tool schemas, streaming, structured output.
- [ADR 0006](../architecture/adr/0006-tool-calling-as-llm-primitive.md) — why tools live in
  `strata_forge.llm` and the agents module reuses them rather than re-implementing them.
- [ADR 0011](../architecture/adr/0011-agents-thin-wrapper-over-forge-llm.md) — why `Agent` is a
  thin composition and not a framework.
- [Control cost and reliability](control-cost-and-reliability.md) — an agent loop is the fastest
  way to spend money by accident; put a ceiling on it.
