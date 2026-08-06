# ADR 0014 — The streaming tool loop is a typed event protocol, not a streamed `LLMResponse`

**Status:** Accepted
**Date:** Addition of `LLMClient.stream_tool_loop`
**Supersedes:** —
**Superseded by:** —
**Amended by:** [ADR 0015](0015-client-executed-tools-suspend-the-streaming-loop.md) — the terminal-event contract gains a third terminal event, `PendingToolCalls` (suspension on caller-executed `ToolDeclaration`s)

## Context

[ADR 0006] made tool calling a first-class `strata_forge.llm` primitive, and
shipped two complementary surfaces: `run_tool_loop` (the buffered
multi-turn loop, returning one final `LLMResponse`) and `stream` (a
single streamed turn, yielding `ResponseChunk`s). They don't compose:
`stream` runs no tool loop, and `run_tool_loop` buffers — so there was
no way to *stream* a multi-turn tool-use conversation.

[ADR 0011] called this gap out explicitly as a consequence of the thin
agent wrapper: `run_tool_loop` returns only the final response, so the
intermediate tool calls and results are invisible to the caller, leaving
Langfuse traces and the `FORGE_DIAGNOSTIC` NDJSON dump as the only ways
to see inside the loop.

A consumer that wants to *show the model working* — text as it streams,
each tool call as it fires, each tool result as it returns — needs a
third surface. The design question is its **shape**: what does a
streaming, multi-iteration tool loop yield?

## Decision

`LLMClient.stream_tool_loop` yields a flat stream of **typed events** —
a discriminated union (`LoopEvent`) of frozen Pydantic models in
[`loop_events.py`](../../src/strata_forge/llm/loop_events.py) — not a stream of
`ResponseChunk`s and not a buffered `LLMResponse`.

```
IterationStart  — top of each iteration (index)
TextDelta       — one assistant text fragment (text, iteration)
ToolCallStarted — one fully-assembled tool call (id, name, arguments, iteration)
ToolResult      — one tool result fed back (id, name, content, is_error, iteration)
Done            — terminal: model exited tool-use mode (finish_reason, usage)
LoopError       — terminal: the loop could not complete (message, error_type, exceeded_max_iterations)
```

Five decisions pin the shape:

1. **A semantic event union, not raw chunks.** `ResponseChunk` describes
   one provider turn; it has no vocabulary for "a tool is about to run"
   or "iteration 2 began". A consumer building tool-step UI would have to
   re-derive those boundaries by re-implementing the loop's
   bookkeeping — exactly the duplication [ADR 0011] rejects. The events
   *are* the loop's semantics, surfaced once.

2. **Tool calls are emitted whole, never partial.** Streamed tool-call
   arguments arrive as `ToolCallDelta` fragments; `ToolCallStarted` is
   emitted only after the iteration's stream drains and the fragments are
   reassembled and JSON-parsed (via the shared
   `StreamingToolCallAccumulator`). A consumer never sees a half-built
   argument object.

3. **A new method, not a flag on `run_tool_loop`.** Streaming changes the
   return type from `LLMResponse` to `AsyncIterator[LoopEvent]`; bolting
   it onto `run_tool_loop` (e.g. a `stream=True` flag) would make the
   return type conditional and break every existing caller and its
   tests. `stream_tool_loop` sits beside `run_tool_loop`, which is left
   untouched. The two share all feedback semantics (assistant + tool-
   result message append, unknown-tool and tool-exception handling as
   `is_error` results, the capability gate).

4. **Raise vs emit — pre-flight raises, in-stream emits.** Failures the
   caller can fix *before* any event flows — a bad `max_iterations`
   (`ValueError`) or a capability-gate violation (`RegistryError`) —
   **raise synchronously**, matching `run_tool_loop` so callers handle
   them the same way. Failures *after* events have begun — a
   provider/transport error, a malformed streamed tool call, or the
   iteration cap — surface as a **terminal `LoopError` event**. Once an
   async generator has yielded, raising out of it would force every
   consumer to wrap iteration in `try/except` *and* handle a partial
   stream; a terminal event means the contract is uniform: **every run
   ends with exactly one terminal event** (`Done` or `LoopError`). This
   is the one deliberate divergence from `run_tool_loop`, which raises
   `ToolLoopExceededError` on cap exhaustion — the streaming loop sets
   `LoopError(exceeded_max_iterations=True)` instead.

5. **Events live in their own module.** `loop_events.py` keeps the union
   out of `responses.py` (whose `LLMResponse`/`ResponseChunk` are the
   non-loop surface) and cohesive with the streaming-loop method that
   produces them. They are re-exported from both `strata_forge.llm` and
   `strata_forge.agents`, since `Agent.run_streaming` yields them.

`usage` on `Done` is the final iteration's usage only — not a sum across
iterations — consistent with `run_tool_loop` / `AgentResult` (see
[ADR 0011]); cumulative accounting stays a tracing concern.

## Consequences

**Positive**

- **The loop's intermediate state is finally visible** without
  re-implementing it or reading traces out of band — closing the gap
  [ADR 0011] documented. UIs (e.g. a chat that renders tool-step cards)
  consume the events directly.
- **One tool-loop implementation, still.** `stream_tool_loop` reuses
  `stream`, the shared accumulator, the same feedback rules, and the same
  capability gate. It does not fork the loop semantics; a fix to the
  feedback rules lands in both.
- **A uniform terminal contract.** Exactly one `Done` or `LoopError`
  ends every run, so consumers don't need to special-case raised
  exceptions mid-stream — important for an SSE bridge that must always
  close its stream cleanly.

**Negative**

- **Two ways to fail.** Pre-flight raises but in-stream emits, so a
  caller must both wrap the *call set-up* and inspect for a terminal
  `LoopError`. This is the cost of not raising out of a generator
  mid-stream; it's documented on the method and here.
- **A second tool-loop surface to keep in step.** `run_tool_loop` and
  `stream_tool_loop` must stay behaviorally aligned on feedback
  semantics. They share helpers to minimize drift, but a change to one's
  semantics requires a matching change (and test) in the other.
- **No streamed structured output.** `stream_tool_loop` streams text and
  tool steps, not a typed schema; `run_structured` has no streaming
  counterpart. Structured streaming is out of scope.

## Alternatives considered

1. **Stream `ResponseChunk`s across iterations.** Reuses the existing
   streaming type, but a `ResponseChunk` can't express iteration
   boundaries or "tool is about to run / has returned", so consumers
   re-derive the loop's structure — the duplication this codebase avoids.
   Rejected.

2. **A `stream=True` flag on `run_tool_loop`.** Smaller surface, but
   makes the return type conditional (`LLMResponse` vs
   `AsyncIterator[...]`), a breaking change for every caller and a typing
   hazard. Rejected in favor of a sibling method.

3. **Raise `ToolLoopExceededError` / provider errors out of the
   generator** (full parity with `run_tool_loop`). Cleanest *symbolically*
   but forces every consumer to handle both a raised exception and a
   truncated event stream. The terminal-`LoopError` contract is simpler
   to consume and is what an SSE/event bridge actually wants. Pre-flight
   errors still raise, because no events have flowed yet.

[ADR 0006]: 0006-tool-calling-as-llm-primitive.md
[ADR 0011]: 0011-agents-thin-wrapper-over-forge-llm.md
