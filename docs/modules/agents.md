# `forge.agents` — agent runtime, built-in tools, memory, multi-agent patterns

`forge.agents` is a thin composition layer on top of `forge.llm`.
The agent runtime never re-implements tool calling, structured
output, the multi-turn tool loop, or message handling — every one
of those primitives comes from `forge.llm` directly. See
[ADR 0011](../architecture/adr/0011-agents-thin-wrapper-over-forge-llm.md)
for the design rationale.

Integration points:

- **Agent runtime:** `Agent` + `AgentResult`. `Agent.run()`
  dispatches through `LLMClient.run_tool_loop` when tools are
  present and `LLMClient.complete` when not.
  `Agent.run_structured(output_schema=)` uses
  `LLMClient.complete_structured`. `Agent.run_streaming()` yields
  the tool loop as a live stream of typed `LoopEvent`s.
- **Built-in tools:** `calculator`, `fetch_url` (ready-to-use
  instances); `fs_read_tool(allowed_dirs=)`,
  `web_search_tool(backend=)` (factories needing caller config).
- **Memory:** `ConversationMemory` for in-process turn history with
  token-budget trimming; `EpisodicMemory` against a pluggable
  `VectorStore` Protocol (concrete `InMemoryVectorStore` ships
  here; Qdrant backend lands with `forge.rag`).
- **Multi-agent patterns:** `handoff(router=, specialists=, ...)`
  and `critic_refiner_run(drafter=, critic=, ...)`. Both are pure
  compositions over `Agent`.

Module rules: [`src/forge/agents/CLAUDE.md`](../../src/forge/agents/CLAUDE.md).
Source: [`src/forge/agents/`](../../src/forge/agents/).

---

## Contents

- [Quickstart](#quickstart)
- [Agent and AgentResult](#agent-and-agentresult)
- [Built-in tools](#built-in-tools)
- [Memory](#memory)
- [Multi-agent patterns](#multi-agent-patterns)
- [Lazy-import contract](#lazy-import-contract)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```python
import asyncio
from forge.agents import Agent, calculator
from forge.llm.client import LLMClient

async def main() -> None:
    client = LLMClient(model="claude-opus-4-7", provider="anthropic")
    agent = Agent(
        "math-helper",
        client=client,
        system_prompt="Use the calculator tool for any arithmetic.",
        tools=[calculator],
    )
    result = await agent.run("What is (15 * 23) / 4 - 2**3?")
    print(result.text)

asyncio.run(main())
```

End-to-end demos:

- [`examples/23_agent_basic.py`](../../examples/23_agent_basic.py)
  — agent with the calculator tool.
- [`examples/24_agent_memory.py`](../../examples/24_agent_memory.py)
  — multi-turn agent backed by `ConversationMemory`.
- [`examples/25_agent_critic_refiner.py`](../../examples/25_agent_critic_refiner.py)
  — critic-refiner pattern with drafter + critic agents.

---

## Agent and AgentResult

```python
class Agent:
    def __init__(
        self,
        name: str,
        *,
        client: LLMClient,
        system_prompt: str | None = None,
        tools: Sequence[Tool] = (),
        max_iterations: int = 8,
    ) -> None: ...

    async def run(self, user_input: str | Sequence[AnyMessage], **kw) -> AgentResult: ...
    async def run_streaming(
        self, user_input: str | Sequence[AnyMessage], **kw,
    ) -> AsyncIterator[LoopEvent]: ...
    async def run_structured(
        self, user_input, *, output_schema: type[M], **kw,
    ) -> AgentResult: ...
```

`Agent.run()` builds the conversation as
`[SystemMessage(system_prompt), <user_input>]`, then dispatches
through `LLMClient.run_tool_loop` (when `tools` are non-empty) or
`LLMClient.complete` (otherwise). `Agent.run_structured()` invokes
`LLMClient.complete_structured` against the same composed messages
and returns a result whose `.parsed` is the validated Pydantic
instance.

```python
class AgentResult(BaseModel):
    text: str
    messages: tuple[AnyMessage, ...]   # input conversation
    final_response: LLMResponse         # raw final-iteration response
    parsed: Any = None                  # set by run_structured
```

`AgentResult.messages` carries the conversation that was *sent* to
the LLM (system + user). Intermediate tool-call / tool-result
messages from inside `run_tool_loop` are not included here —
`LLMClient.run_tool_loop` doesn't expose them. For full iteration
visibility, install the LiteLLM Langfuse callback via
`forge.tracing.install_litellm_callback()` or set
`FORGE_DIAGNOSTIC=1` for the NDJSON dump.

For *live* visibility, use `Agent.run_streaming()` — the streaming
counterpart that yields typed `LoopEvent`s (`IterationStart` /
`TextDelta` / `ToolCallStarted` / `ToolResult` / `Done` / `LoopError`)
as the conversation unfolds, surfacing the intermediate tool calls and
results that `run()` collapses into a final answer. It delegates to
[`LLMClient.stream_tool_loop`](llm.md#streaming-tool-loop) with the
agent's tools and `max_iterations`; a tool-less agent degenerates to a
single streamed turn. There is no streaming variant of
`run_structured`.

```python
async for event in agent.run_streaming("Weather in Tokyo?"):
    ...  # match on the event type — see the llm.md streaming-tool-loop table
```

---

## Built-in tools

### `calculator`

A safe arithmetic evaluator. Parses expressions into an AST, walks
them against a whitelist of numeric nodes (BinOp, UnaryOp, Constant,
the arithmetic operators), and only then compiles+evals. Rejects
function calls, attribute access, names, strings, lists, and
comparisons.

```python
from forge.agents import Agent, calculator
agent = Agent("math", client=client, tools=[calculator])
```

### `fetch_url`

HTTP GET via `httpx.AsyncClient` (already in core deps via
`litellm`). Pydantic-validated URL; capped body length; per-request
timeout; follows redirects.

```python
from forge.agents import Agent, fetch_url
agent = Agent("reader", client=client, tools=[fetch_url])
```

### `fs_read_tool`

**Factory.** Returns a sandboxed file-reader Tool. The target path
and the allow list are both resolved (symlinks followed) before
membership check, so parent-traversal and symlink-escape attempts
fail safely.

```python
from pathlib import Path
from forge.agents import Agent, fs_read_tool

reader = fs_read_tool(allowed_dirs=[Path("docs/").resolve()])
agent = Agent("doc-bot", client=client, tools=[reader])
```

### `web_search_tool`

**Factory.** Forge stays vendor-agnostic about search APIs — pass
in a backend (async `(query, max_results) -> Sequence[SearchResult]`)
that wraps Tavily / SerpAPI / DuckDuckGo / your internal index.

```python
from forge.agents import Agent, SearchResult, web_search_tool

async def my_backend(query: str, n: int) -> list[SearchResult]:
    # ... call your search provider ...
    return [SearchResult(title=..., url=..., snippet=...)]

search = web_search_tool(backend=my_backend)
agent = Agent("researcher", client=client, tools=[search])
```

`SearchResult` allows extra fields (Pydantic `extra="allow"`) so
provider-specific metadata (score, published_at, …) passes through.

---

## Memory

### `ConversationMemory`

In-process append-only history with a dedicated system message slot
that's never evicted under memory pressure.

```python
from forge.agents import ConversationMemory

memory = ConversationMemory(system_message="be helpful")
memory.append_user("hi")
memory.append_assistant("hello")
memory.trim_to_messages(20)            # keep last 20 non-system messages
memory.trim_to_tokens(2_000, model="claude-opus-4-7")  # token-budget trim
```

`memory.messages` returns the full conversation (system + history)
as a tuple; `memory.non_system_messages` returns just the history.
Both trim modes drop oldest non-system messages first; the system
message survives every trim.

### `EpisodicMemory`

Vector-backed long-term memory against a pluggable `VectorStore`
Protocol. The in-process `InMemoryVectorStore` works for tests and
prototyping; `forge.rag`'s Qdrant backend (Phase 4) will satisfy
the same Protocol.

```python
from forge.agents import EpisodicMemory, InMemoryVectorStore

async def my_embed(text: str) -> list[float]:
    # ... call your embedding provider ...
    return [...]

memory = EpisodicMemory(store=InMemoryVectorStore(), embed=my_embed)
await memory.add("user prefers concise answers")
await memory.add("user is allergic to peanuts")

results = await memory.search("what should I avoid putting in the meal?")
for r in results:
    print(f"{r.score:.3f} — {r.item.text}")
```

The `VectorStore` Protocol exposes `add` / `search` / `delete` /
`clear`. Anything satisfying that shape — including the future
`QdrantVectorStore` — drops into `EpisodicMemory` unchanged.

---

## Multi-agent patterns

Both functions are pure compositions — no new runtime.

### `handoff`

A router agent picks one specialist from a catalog; the chosen
specialist responds to the original user input.

```python
from forge.agents import Agent, handoff

router = Agent("router", client=client, system_prompt="Pick the right specialist.")
math_agent = Agent("math", client=client, system_prompt="You solve math problems.")
text_agent = Agent("text", client=client, system_prompt="You write prose.")

result = await handoff(
    router=router,
    specialists={"math": math_agent, "text": text_agent},
    user_input="rewrite this paragraph in formal English",
)
print(result.text)   # text_agent's answer
```

The router is invoked via `run_structured` with `RouterChoice` as
the schema. Unknown specialists raise `ValueError`; empty catalogs
raise `ValueError`.

### `critic_refiner_run`

A drafter produces an answer; a critic reviews it and either
approves or returns feedback. The loop continues until approval or
`max_rounds` is hit. On exhaustion, the most recent draft is
returned (partial progress beats raising).

```python
from forge.agents import Agent, critic_refiner_run

drafter = Agent("drafter", client=client, system_prompt="Write product copy.")
critic = Agent("critic", client=stronger_client, system_prompt="Approve only when concise + vivid.")

final = await critic_refiner_run(
    drafter=drafter,
    critic=critic,
    user_input="Write a product description for a sleep mask.",
    max_rounds=3,
)
```

The critic is invoked via `run_structured` with `CritiqueVerdict`
as the schema. Feedback is rolled into the drafter's next prompt as
a system + user message bundle.

---

## Lazy-import contract

`import forge.agents` works without any optional extras. The
built-in tools' SDK dependencies are minimal:

- `calculator` — stdlib only.
- `fetch_url` — `httpx` (core via `litellm`).
- `fs_read` — stdlib `pathlib` + `asyncio.to_thread` for the read.
- `web_search` — no SDK; the caller supplies the backend.

Memory primitives are likewise stdlib-only. `EpisodicMemory`
accepts any `VectorStore` Protocol implementation — the concrete
adapter for Qdrant lands in `forge.rag` (Phase 4).

---

## Troubleshooting

- **"`Agent name must be non-empty`"** / **"`max_iterations must be >= 1`"**: invalid construction args; pass a name and a positive iteration cap.
- **`RegistryError: model X does not support tool calling`**: the model in the agent's `LLMClient` lacks `tool_calling=True` in the registry. Either pick a tool-capable model or remove the tools from the agent.
- **`PermissionError: path X is outside the allowed directories`**: the `fs_read` sandbox rejected the path. Add the directory to `allowed_dirs` at tool construction, or use an explicit path inside an existing allowed root.
- **`ValueError: handoff: router picked X, which isn't in the specialists catalog`**: the router LLM emitted an unknown specialist name. Make the router's system prompt enumerate the available specialists more explicitly, or run a sanity-check verification of the catalog the router sees.
- **Critic loops forever / `max_rounds` exhausted**: the critic is too strict, or the drafter can't act on the feedback. Inspect via Langfuse traces (every iteration is captured) and either soften the critic's criteria or improve the drafter's instructions.
- **Multi-modal `UserMessage` content lost from `ConversationMemory` token counts**: the token extractor counts text parts only; images aren't counted toward `trim_to_tokens`. Image tokens are computed by `forge.llm.tokens` only when paired with a real model.
