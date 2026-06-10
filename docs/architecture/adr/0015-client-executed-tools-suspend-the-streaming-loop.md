# ADR 0015 — Client-executed tools suspend the streaming loop

**Status:** Accepted
**Date:** Addition of `ToolDeclaration` + `PendingToolCalls`
**Supersedes:** —
**Superseded by:** —
**Amends:** [ADR 0014] (the terminal-event contract gains a third terminal event)

## Context

[ADR 0006] made tools executable Python functions, and [ADR 0014]'s
`stream_tool_loop` executes every requested call inline. That assumes the
loop's host can execute every tool. It can't always: a server streaming a
tool loop to a browser may need the *browser* to perform an action (mutate
UI state, drive an app) and feed the result back. The model must be able to
call such a tool — the provider needs its schema — but forge has no function
to invoke for it, and an `AsyncIterator` is one-directional: the consumer
cannot inject a result mid-stream.

The design question: how does `stream_tool_loop` offer tools it cannot
execute, and what happens when the model calls one?

## Decision

Two additions, no new loop:

1. **`ToolDeclaration`** (`tools.py`) — a frozen dataclass carrying
   `name`, `description`, and a raw JSON-Schema `parameters` dict, with the
   same `to_{openai,anthropic,gemini}_schema()` surface as `Tool` but no
   `fn`. It is passed in the **same `tools=` sequence** as executable tools
   (`type AnyTool = Tool | ToolDeclaration`); `complete`/`stream` accept it
   for serialization, and only the streaming loop distinguishes it.

2. **`PendingToolCalls`** (`loop_events.py`) — a third terminal event.
   When a turn requests at least one declaration-targeted call, the loop
   executes the turn's executable calls first (in model call order, with
   their `ToolResult` events), then yields
   `PendingToolCalls(calls, messages, iteration, iterations_used, usage)`
   and returns. `calls` are the unexecuted declaration-targeted calls;
   `messages` is the **conversation delta** appended this run (assistant
   turns + executed tool results — everything after the caller's input).

**Resumption is plain input, not a new API.** The caller executes the
pending calls out-of-band and re-invokes
`stream_tool_loop(input + event.messages + one ToolResultMessage per
pending call)`. This works because `validate_conversation` validates by
construction-order invariants (every result id references a prior call id),
not by shape templates — a mid-loop conversation is a valid conversation.
`iterations_used` lets the caller pass the remaining budget as
`max_iterations` on resume so the cap is meaningful across runs.

Supporting decisions:

- **Declarations mix into `tools=`, not a separate parameter.** Provider
  serialization (`_tools_for_provider`), the capability gate, and the
  per-name lookup all iterate one list; a second parameter would force
  every one of them to merge two. A declaration is simply "a tool forge
  cannot execute" — nothing in forge is "a browser".
- **Duplicate tool names are a pre-flight `ValidationError`.** With
  declarations in the mix, a shared name makes execute-vs-suspend
  ambiguous. (The previous behavior — silent last-wins shadowing among
  executable tools — was never meaningful.) Matches ADR 0014's
  raise-vs-emit policy: nothing has streamed yet, so it raises.
- **The event exposes forge message types, not provider wire dicts.**
  `AssistantMessage`/`ToolResultMessage`/`ToolCall` are already public API;
  re-deriving the delta from `TextDelta`/`ToolCallStarted`/`ToolResult`
  events would re-implement the loop's bookkeeping — exactly what ADR 0014
  decision 1 rejects. The private `_message_to_wire` format stays private.
- **A suspension on the last budgeted iteration is a suspension**, not an
  `exceeded_max_iterations` `LoopError` — the turn completed; the model is
  waiting on the caller, not looping.
- **`run_tool_loop` is untouched.** It keeps `Sequence[Tool]`: a buffered
  loop has no way to suspend, and pyright now enforces that declarations
  only flow where suspension exists.

## Consequences

**Positive**

- A host can offer caller-executed tools with zero loop forking: one loop
  implementation, one feedback semantics, one accumulator.
- Stateless resumption: everything needed to continue lives in
  `input + event.messages + results`, so a server can suspend across
  HTTP requests without holding state.
- Mixed turns work: server-executable calls in a suspending turn still
  execute, in model order, before suspension.

**Negative**

- **Three terminal events.** Consumers switching on the union must handle
  `PendingToolCalls` or fail on suspension; the type system surfaces this
  (the union changed).
- **Resumption correctness is the caller's job.** Forge validates the
  conversation but cannot know whether the caller's results are faithful;
  a caller that drops a pending call's result gets a pre-flight
  `ValidationError` only if ids mismatch, not if content lies.
- **Declarations skip argument validation.** A `Tool` validates arguments
  through its Pydantic model on invoke; a declaration has no model, so the
  caller receives the model's arguments as parsed JSON, verbatim, and must
  validate them itself.

## Alternatives considered

1. **A callback (`on_client_tool=`) the loop awaits.** Keeps the stream
   alive but forces the host to block the HTTP response while the browser
   round-trips — unusable over SSE, which is one-directional. Rejected.
2. **A separate `client_tools=` parameter.** Same semantics, but every
   internal consumer of `tools` (serialization, gate, name map) must merge
   two sequences, and the "one list, one namespace" collision check
   becomes two-list bookkeeping. Rejected.
3. **Expose provider wire dicts in the event** (let the caller replay them
   verbatim). Leaks the private wire format and couples consumers to
   LiteLLM's shape. The typed messages are the public, stable currency.
   Rejected.

[ADR 0006]: 0006-tool-calling-as-llm-primitive.md
[ADR 0014]: 0014-streaming-tool-loop-event-protocol.md
