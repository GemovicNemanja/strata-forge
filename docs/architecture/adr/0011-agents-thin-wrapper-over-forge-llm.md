# ADR 0011 — `strata_forge.agents` is a thin wrapper over `strata_forge.llm`, not a parallel runtime

**Status:** Accepted
**Date:** Initial scaffolding for `strata_forge.agents`
**Supersedes:** —
**Superseded by:** —

## Context

`strata_forge.llm` deliberately owns tool calling, structured output,
and the multi-turn tool loop (ADR 0006). The agent layer sits above
it: a higher-level builder that wires a system prompt, a set of
tools, and an `LLMClient` into a single object whose
``.run(user_input)`` method produces a final answer.

The natural temptation when building agents is to ship a *parallel*
runtime — a new tool abstraction, a new message type, a new loop
implementation, a new structured-output dispatcher. Many agent
libraries do this. We're explicitly rejecting that shape.

## Decision

`strata_forge.agents` is a **thin, composition-only layer over
`strata_forge.llm`**. Concretely:

1. **No new tool abstraction.** Agents accept `strata_forge.llm.Tool`
   instances built with the existing `strata_forge.llm.tool`
   decorator. The agent's tools field is typed as
   ``tuple[Tool, ...]``; nothing else satisfies it.
2. **No new message types.** Agent input is either a string
   (wrapped into a `UserMessage`) or a sequence of the
   existing `strata_forge.llm.AnyMessage` union members. The agent
   prepends a `SystemMessage` from its system prompt and
   passes the result through verbatim.
3. **No new tool loop.** Agents delegate to
   `LLMClient.run_tool_loop` when tools are present and to
   `LLMClient.complete` otherwise. The agent never iterates
   over tool calls itself.
4. **No new structured-output path.** When an agent needs typed
   output, it calls `LLMClient.complete_structured` — same
   schema reprompt logic, same per-provider dispatch, same
   `StructuredResponse`.
5. **`AgentResult` is the only new shape.** It bundles the final
   text, the conversation that was sent to the LLM, and the raw
   `LLMResponse`. The raw response is exposed so callers
   that already understand `strata_forge.llm` semantics never need to
   learn an agent-specific accounting layer.

```
                                     strata_forge.agents.Agent
                                            │
                                            │  .run() / .run_structured()
                                            ▼
                              strata_forge.llm.LLMClient
                                            │
                          ┌─────────────────┼──────────────────┐
                          ▼                 ▼                  ▼
                 .run_tool_loop()    .complete()    .complete_structured()
                          │                 │                  │
                          ▼                 ▼                  ▼
                     strata_forge.llm.Tool ◄── @tool decorator ──► Pydantic schema
```

Built-in tools are concrete `Tool` instances constructed via `tool`;
memory is layered on top of the message-history shape `LLMClient`
already understands; multi-agent patterns compose multiple `Agent`
instances rather than introducing a new orchestration primitive.

## Consequences

**Positive**

- **One source of truth for tool calling.** Bugs in the tool loop
  are fixed in one place. The agent module can't drift from the
  LLM client's semantics because it doesn't have its own.
- **Token, cost, and tracing accounting works automatically.** The
  Langfuse callback registered by `strata_forge.tracing` traces every
  LLM call regardless of whether it originated from a bare
  `LLMClient.complete` or from `Agent.run`. The diagnostic NDJSON
  dump catches every iteration of the agent's tool loop.
- **Migration ergonomics.** Users who already use
  `LLMClient.run_tool_loop` directly can wrap their existing tools
  in an `Agent` without rewriting them. There's nothing to port.
- **Type checking sees through the layer.** Since `Agent` doesn't
  introduce new generics or wrap the response type, pyright's
  inference flows from `LLMClient` straight through to the caller.

**Negative**

- **The agent's view of intermediate tool calls is limited.**
  `LLMClient.run_tool_loop` returns only the final
  `LLMResponse`; intermediate tool-call / tool-result
  messages live inside the method. `AgentResult.messages`
  therefore carries the *input* conversation plus the final
  assistant message, not every iteration. Callers that need the
  full iteration trace rely on `strata_forge.tracing` (Langfuse traces
  capture every call) or the `FORGE_DIAGNOSTIC` NDJSON dump.
- **Some agent-library features don't have an obvious home.**
  Things like "rate-limit my agent's tool calls" or "give my
  agent a persistent budget across runs" don't fit a thin wrapper.
  Those land in adjacent modules (`strata_forge.core.budget` for the
  budget case) rather than reshaping the agent itself.

**Mitigations**

- The "full iteration trace" gap is real but Langfuse-traced
  setups already get the data they need. We document the
  tracing-dependency in the module reference doc so users who
  care know where to look.
- Power features that don't fit the thin-wrapper shape are
  evaluated case-by-case; the bar for adding them is "does this
  belong somewhere else in `strata_forge.*` instead?" before we extend
  the agent.

## Alternatives considered

1. **Re-implement tool calling inside `strata_forge.agents`.** Maximizes
   flexibility but doubles the surface area of "how does Forge do
   tool calling?", with the inevitable drift between the two
   implementations. Rejected on consistency grounds.

2. **`strata_forge.agents` exposes `Tool`, `@tool`, messages as
   re-exports without any new shape.** Smaller still, but loses
   the convenience of "I want one object that knows about my
   tools and my system prompt." The `Agent` shape pays for itself
   in ergonomics. Kept the re-exports anyway — `from strata_forge.agents
   import tool, UserMessage` works for users who don't want to
   know that the primitives live in `strata_forge.llm`.

3. **Build the agent on PydanticAI directly.** Loses the
   provider-agnostic Forge surface (PydanticAI has its own
   provider seam), and ships a transitive dep on a project we
   don't otherwise depend on. Rejected — Forge already has every
   primitive PydanticAI exposes, plus the registry, fallback
   chain, and caching layer in `strata_forge.llm`.
