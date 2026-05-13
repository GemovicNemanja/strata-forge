# ADR 0002 — Async-only public API, sync wrappers in a single module

**Status:** Accepted
**Date:** Initial scaffolding
**Supersedes:** —
**Superseded by:** —

## Context

Forge's primary workloads are I/O-bound: HTTP calls to LLM providers, vector store lookups, dataset uploads, remote-compute submissions. They benefit from `asyncio` concurrency (batch inference, agent loops, parallel grader execution). At the same time, two important usage modes are inherently synchronous: command-line invocations and notebook cells where users don't want to think about event loops.

A common anti-pattern in async Python libraries is to maintain two parallel APIs — one async, one sync — drifting over time and doubling the test matrix. We don't want that.

## Decision

- Every public function on every Forge module is `async`.
- Sync wrappers live exclusively in `forge.sync`. Each wrapper is a thin shim over the async equivalent (`return asyncio.run(forge.llm.complete(...))`).
- Internal calls between Forge modules are async end-to-end. We never call `httpx.Client` (sync) or `requests` from production code paths.
- The CLI (`forge.cli`) calls into `forge.sync` for ergonomics; commands that need async features (streaming) handle their own event loops via `anyio` / `asyncio`.

## Consequences

**Positive**

- One implementation per function — no parallel sync code path to maintain.
- Async fits naturally with streaming, batch inference, agent multi-turn loops, parallel evaluation, and parallel retrieval.
- LiteLLM, Pydantic, HTTPX, anyio — our entire stack is async-friendly already.
- Sync wrappers add roughly one line per wrapped function; the cost is trivial.

**Negative**

- Notebook users in a pure-sync mental model must remember to use `forge.sync` for one-shot calls. Mitigation: `from forge.sync import complete, stream, run_tool_loop, …` is one import.
- New contributors need familiarity with `async def` / `await`. We accept this as table stakes for modern Python work.
- Pure-sync libraries we depend on (e.g. some training utilities) must be wrapped via `anyio.to_thread.run_sync` when called from async paths. We do this inline where needed.

**Mitigations**

- `forge.sync` is generated mechanically wherever feasible — when adding a new async public function, add the sync wrapper in the same PR; CI lint can check for the pair.
- Marimo notebooks (top-level `await` supported) sidestep most of the awkwardness.

## Alternatives considered

1. **Sync-only public API, async only internally.** Forces async-savvy users to fight the abstraction, blocks event-loop integration with agent frameworks (PydanticAI is async-native).
2. **Dual sync + async APIs at parity.** Doubles maintenance and test surface; the two implementations drift.
3. **Async-only, no sync wrappers.** Forces CLI and notebook authors to write `asyncio.run(...)` everywhere; small but constant friction. The `forge.sync` namespace removes it cheaply.
