# `strata_forge.sync` — synchronous facades for scripts and notebooks

Every public method of `strata_forge.llm.LLMClient` is `async def`. That is a deliberate choice —
the workloads are I/O-bound, and maintaining two parallel implementations is how async libraries
rot ([ADR 0002](../architecture/adr/0002-async-only-public-api.md)). `strata_forge.sync` is the one
place where that rule is relaxed: a small set of blocking facades that exist so a shell script or
a notebook cell does not have to write `asyncio.run(...)` at every call site.

It is a single module, `src/strata_forge/sync.py`, and it wraps exactly four `LLMClient` methods:

| Function | Wraps | Required arguments |
|---|---|---|
| `complete` | `LLMClient.complete` | `messages` |
| `complete_structured` | `LLMClient.complete_structured` | `messages`, `schema=` |
| `stream` | `LLMClient.stream` | `messages` |
| `run_tool_loop` | `LLMClient.run_tool_loop` | `messages`, `tools=` |

Nothing else has a facade. `LLMClient.stream_tool_loop` is async-only, and no module outside
`strata_forge.llm` — not `evals`, not `agents`, not `rag`, not `storage`, not `compute` — is
wrapped here. For those, call the async API from an async context.

Source:
[`src/strata_forge/sync.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/src/strata_forge/sync.py).
The async surface these wrap is documented in [`strata_forge.llm`](llm.md).

---

## Contents

- [Quickstart](#quickstart)
- [Choosing the client](#choosing-the-client)
- [The event-loop rule](#the-event-loop-rule)
- [Notebooks](#notebooks)
- [Streaming is drained, not streamed](#streaming-is-drained-not-streamed)
- [Cost of a one-shot loop](#cost-of-a-one-shot-loop)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```python
from pydantic import BaseModel

from strata_forge import sync
from strata_forge.llm import Message


class Summary(BaseModel):
    title: str
    bullets: list[str]


response = sync.complete([Message.user("Say hi in three words.")], model="claude-opus-4-7")
print(response.text, response.route.provider, response.usage.total_tokens, response.cost_usd)

structured = sync.complete_structured(
    [Message.user("Summarize the Apollo program.")],
    schema=Summary,
    model="claude-opus-4-7",
)
print(structured.parsed.title, structured.parsed.bullets)

for chunk in sync.stream([Message.user("Count to five.")], model="claude-opus-4-7"):
    print(chunk.delta_text, end="")
```

The return types are the async ones, unchanged: `complete` gives an `LLMResponse`,
`complete_structured` gives a `StructuredResponse[M]` (an `LLMResponse` subclass, so `.text`,
`.usage`, and `.cost_usd` are all still there alongside `.parsed`), `stream` gives an
`Iterator[ResponseChunk]`, and `run_tool_loop` gives the final `LLMResponse`. Everything else
about the call — routing, fallback, caching, budgets, tracing, the diagnostic dump — behaves
exactly as it does through the async client, because it *is* the async client underneath.

Every keyword the async method accepts passes straight through: `temperature`, `max_tokens`,
`top_p`, `tools`, `response_format`, `provider_extras`, and so on.

---

## Choosing the client

Each function takes either a pre-built client or the arguments to construct one:

```python
from strata_forge import sync
from strata_forge.llm import LLMClient, Message

# One-shot: build a client per call.
sync.complete([Message.user("hi")], model="claude-opus-4-7", provider="anthropic")

# Reused: build once, pass it in.
client = LLMClient("claude-opus-4-7")
sync.complete([Message.user("hi")], client=client)
sync.complete([Message.user("and again")], client=client)
```

Pass `client=` **or** some combination of `model=` / `provider=` / `chain=`, never both — doing so
raises a `ValueError`:

```
Pass either `client` or (`model`/`provider`/`chain`), not both
```

The `chain=` form takes the same `FallbackEntry` sequence as the async client, so two-axis
fallback works from a sync call site too.

Build the client once whenever it carries state you want to survive between calls: a configured
cache, injected `provider_clients`, non-default retry or fallback policy. For a throwaway script,
constructing one per call is fine.

---

## The event-loop rule

Each wrapper is one `asyncio.run(...)`. That call creates a fresh event loop, runs the coroutine
to completion, and closes the loop. `asyncio.run` refuses to nest, so **calling any
`strata_forge.sync` function from inside a running event loop raises**:

```
RuntimeError: asyncio.run() cannot be called from a running event loop
```

This is not a bug to work around; it is the boundary of what a sync facade can do. If you are
already inside an event loop, you are in async code, and the async API is right there:

```python
from strata_forge.llm import LLMClient, Message


async def handler() -> str:
    client = LLMClient("claude-opus-4-7")
    response = await client.complete([Message.user("hi")])  # not sync.complete
    return response.text
```

The same applies to any framework that runs your code inside a loop — an async web handler, a
`TaskGroup`, an async test (`pytest.mark.anyio` / `pytest.mark.asyncio`), or an async CLI command.
In all of those, `await client.complete(...)` is both correct and simpler.

Note that the failure mode is loud but late: the `RuntimeError` surfaces at the call, and Python
additionally warns that the underlying coroutine was never awaited. If you see that warning in
otherwise-working code, a sync wrapper has been called from an async context somewhere.

---

## Notebooks

Notebook kernels are the common case where a loop is already running and it is not obvious.

- **Jupyter / IPython** run cells inside an event loop and support top-level `await`. Write
  `await client.complete(...)` directly in the cell; the sync wrappers will raise there.
- **marimo** also supports top-level `await` in cells, with the same guidance.
- **A plain `.py` script** has no loop until you start one, so `sync.complete(...)` works, and so
  does wrapping your own coroutine in `asyncio.run(main())`.

If you are unsure which situation you are in, ask the runtime:

```python
import asyncio

try:
    asyncio.get_running_loop()
    print("a loop is running -- use `await`")
except RuntimeError:
    print("no loop -- strata_forge.sync is safe here")
```

The rule of thumb: if `await` is legal where you are standing, use it and skip this module
entirely. `strata_forge.sync` earns its keep only where `await` is not legal.

---

## Streaming is drained, not streamed

`sync.stream` returns an `Iterator[ResponseChunk]`, but it collects every chunk into a list before
that iterator is handed back. Nothing is yielded incrementally: the call blocks until the model
has finished, then you iterate over the recorded chunks.

That is structural, not an oversight. `asyncio.run` runs a coroutine to completion and closes its
loop; there is no way to suspend inside it, yield a value to sync code, and resume. So a sync
streaming API can offer chunk *objects*, but not chunk *timing*.

Use it when you want the per-chunk structure — deltas, tool-call fragments, finish reasons — from
a script, and reach for `LLMClient.stream` from async code when you want tokens to appear on
screen as they arrive.

---

## Cost of a one-shot loop

Each wrapper call spins up an event loop and tears it down. For interactive and script use this is
irrelevant next to the network round trip. It becomes relevant in two situations:

- **Many calls in a tight loop.** Sequential `sync.complete` calls cannot overlap, so N calls take
  the sum of N latencies. The async client with `asyncio.gather` (or
  `strata_forge.compute.BatchInferenceRunner` for a bounded fan-out) does them concurrently.
- **Long-lived connections.** A fresh loop per call means HTTP connection pools set up under one
  loop cannot be reused by the next. Passing `client=` preserves the client's own state — the
  response cache in particular — but the loop, and anything bound to it, is new each time.

If either applies, write an `async def main()` and call `asyncio.run(main())` once around the
whole batch instead of once per call.

---

## Troubleshooting

**`RuntimeError: asyncio.run() cannot be called from a running event loop`.**
A sync wrapper was called from inside an event loop — a notebook cell, an async handler, an async
test. Use `await client.complete(...)` (or the relevant `LLMClient` method) instead. See
[The event-loop rule](#the-event-loop-rule).

**`RuntimeWarning: coroutine 'LLMClient.complete' was never awaited`.**
The same problem seen from the other side: `asyncio.run` rejected the coroutine, so nothing
consumed it. Fix the call site rather than silencing the warning.

**`ValueError` complaining that you passed both a client and construction arguments.**
Exactly what it says — drop either the `client=` argument or the `model=` / `provider=` /
`chain=` arguments.

**`sync.stream` prints nothing until the very end.**
Expected. Chunks are collected before the iterator is returned. Use `LLMClient.stream` from async
code for incremental output.

**I want a sync version of `stream_tool_loop` / the agent runner / the eval runner.**
There isn't one, and adding one would not help: those surfaces are multi-turn and event-driven, so
running them through a single `asyncio.run` would collapse exactly the interleaving that makes
them useful. Write an `async def main()` and call `asyncio.run(main())` once at the top.

**My sync calls are slow in aggregate.**
They run strictly one after another. Move the whole batch into one `async def` and use
`asyncio.gather` so the requests overlap.
