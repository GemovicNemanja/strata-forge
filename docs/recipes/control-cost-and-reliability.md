# Control cost and reliability in production

Four mechanisms decide what a call to `LLMClient.complete` actually costs and how hard it tries:
a **budget ceiling**, a **two-axis fallback chain**, a **provider-agnostic cache**, and a **retry
policy**. They are independent, they compose, and the order in which they compose is what makes
them predictable. This recipe explains that order and then configures each layer.

**Prerequisites.** The base install. `RedisCache` needs the `[redis]` extra; nothing else here
does.

**Running the snippets.** A block that ends in `asyncio.run(main())` is a complete program. The
rest are fragments: put them inside an `async def main()` and invoke it the same way, or paste them
into a notebook cell, where top-level `await` is legal.

---

## What one `complete()` call actually does

```
complete(messages, ...)
  ├─ validate conversation, check tool capability          (raises before any network I/O)
  ├─ compute cache key  ──▶ HIT ──▶ return copy: cache_hit=True, cost_usd=0.0   [stops here]
  └─ run_with_fallback(chain)
       for each model entry in the chain:            ← outer axis: models
         for each provider route on that entry:      ← inner axis: providers
           retry(attempt)                            ← tenacity, exponential backoff + jitter
             └─ attempt: call provider → consume budget → write cache → return
```

Read that sequence carefully, because three of its consequences are not obvious:

1. **A cache hit costs nothing and consumes no budget.** It short-circuits before the fallback
   loop, so it is never retried, never routed, and never charged.
2. **Retries happen inside one provider attempt.** The chain only advances to the next provider
   after the retry budget for the current one is exhausted. A five-attempt retry policy across a
   three-provider chain is up to fifteen provider calls.
3. **The budget is consumed after the provider returns.** `BudgetContext` raises before it
   records spend, but the client only records spend once it has a response — so the ceiling stops
   the *next* call, not the one that crossed the line. Treat it as a circuit breaker, not a
   pre-authorization.

`stream()` sits outside all of this: it bypasses the cache, the fallback loop, the retry wrapper,
and budget accounting, because a stream that is already emitting tokens cannot be cleanly moved
to another provider. `complete_structured()` and `run_tool_loop()` both go through `complete()`
and therefore inherit everything above — a tool loop with eight iterations is eight budget
consumptions and eight cache lookups.

---

## 1. Put a ceiling on it

`BudgetContext` is an async context manager backed by a contextvar, so it covers everything
inside the block — including tasks you spawn concurrently — without being threaded through any
signature.

```python
import asyncio

from strata_forge.core import BudgetContext, BudgetExceededError
from strata_forge.llm import LLMClient, Message

client = LLMClient("claude-opus-4-8", provider="anthropic")

async def main() -> None:
    async with BudgetContext(max_usd=2.00, max_tokens=1_000_000) as budget:
        try:
            await client.complete([Message.user("Summarise this quarter.")])
        except BudgetExceededError as exc:
            print(f"stopped at ${exc.spent_usd} against a ${exc.limit_usd} ceiling")
        print(budget.spent_usd, budget.available_usd())

asyncio.run(main())
```

Either ceiling may be `None`; the other is still enforced and both are still tracked.
`available_usd()` and `available_tokens()` return the remaining headroom, or `None` when that
axis is unlimited.

### Nested budgets

Nested contexts are linked. A child starts at zero spend, and `consume` charges the child **and
every linked ancestor atomically** — if any budget in the chain would be exceeded, nothing is
recorded anywhere. That gives you a per-step ceiling inside a per-run ceiling without any
bookkeeping:

```python
async with BudgetContext(max_usd=10.00) as run_budget:      # whole job
    for item in items:
        async with BudgetContext(max_usd=0.05):             # one item
            await handle(item)
    print(run_budget.spent_usd)                             # sum of every step
```

Pass `isolated=True` to break the link — useful for a sub-task that must not consume the parent's
headroom, for example a cheap classifier running alongside an expensive main path.

### What the budget does not see

- **A single call that overshoots.** Spend is recorded post-hoc; cap the damage with
  `max_tokens=` on the call itself.
- **Embeddings.** `strata_forge.rag`'s embedder calls LiteLLM directly and never touches
  `LLMClient`. Meter ingestion separately.
- **Streaming.** `stream()` does no budget accounting at all.
- **Cache hits.** By design — nothing was spent.

If you want a pre-flight estimate, `strata_forge.llm.count_tokens` gives you the input side
before you call. The client does not estimate on your behalf.

See
[`examples/10_budget_ceiling.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/10_budget_ceiling.py).

## 2. Build the fallback chain

A chain has two axes. The **inner** axis is provider routes for one logical model — the same
weights served by Anthropic, Bedrock, and Vertex. The **outer** axis is different models
entirely. Failover walks providers first, because a different route to the same model preserves
your evaluation results in a way that a different model does not.

```python
from strata_forge.llm import LLMClient, ModelFallback

client = LLMClient.with_fallbacks(
    [
        ModelFallback(model="claude-opus-4-8", providers=("anthropic", "bedrock", "vertex")),
        ModelFallback(model="gpt-5.5", providers=("openai", "azure")),
        "claude-haiku-4-5",   # bare string: registry default route, no provider failover
    ],
)
response = await client.complete([Message.user("Hello.")])
print(response.route.model, response.route.provider)
```

A bare string expands to `ModelFallback(model=..., providers=None)`, meaning "the registry's
default route, no provider-level failover". Always read `response.route` rather than assuming
you got the head of the chain.

### The error routing table

Each error class has one behaviour, and it is not configurable per call:

| Error | Default | With `strict_bad_request=True` |
|---|---|---|
| `ProviderContentFilterError` | re-raise immediately | re-raise immediately |
| `ProviderBadRequestError` | advance to next route | abort the chain |
| `ProviderAuthError` | advance to next route | advance to next route |
| `ProviderRateLimitError` | retry, then advance | retry, then advance |
| `ProviderTimeoutError` | retry, then advance | retry, then advance |
| `ProviderServerError` | retry, then advance | retry, then advance |
| `RegistryError` | advance to next route | advance to next route |

Content filtering short-circuits the whole chain because the same content will be refused
everywhere; retrying it burns money and latency for a guaranteed refusal. Everything else
accumulates into `FallbackExhaustedError.causes` as `(model, provider, error)` triples:

```python
from strata_forge.core import FallbackExhaustedError

try:
    response = await client.complete(messages)
except FallbackExhaustedError as exc:
    for model, provider, error in exc.causes:
        print(f"{model}@{provider or 'default'}: {type(error).__name__}: {error}")
```

Two operational notes. **Auth errors advance silently** — a chain will happily route around a
provider whose key you fat-fingered and serve you a different model, so alert on
`response.route.provider` drift rather than on exceptions. And **bad requests advance by
default**, on the theory that one provider rejecting a parameter is not a reason to fail the
request; set `strict_bad_request=True` when you would rather hear about the malformed call.

See
[`examples/08_provider_fallback.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/08_provider_fallback.py)
and
[`examples/09_model_fallback.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/09_model_fallback.py),
plus [ADR 0005](../architecture/adr/0005-two-axis-fallback.md) for why the two axes are separate.

## 3. Cache

The cache key is a SHA-256 over the canonicalised request: the chain's **head model name**,
messages, sampling parameters, response format, tool schemas, and provider extras. It contains no
provider identity, which is the property that makes it useful — an entry written when Anthropic
served the call is a valid hit when the chain would next have routed to Bedrock.

```python
from strata_forge.llm import InMemoryCache, LLMClient

client = LLMClient("claude-opus-4-8", provider="anthropic", cache=InMemoryCache(max_size=2048))

first = await client.complete(messages)    # miss: hits the provider
second = await client.complete(messages)   # hit: cache_hit=True, cost_usd=0.0
```

`InMemoryCache` is a bounded LRU, right for a single process or as a hot tier. `RedisCache`
shares entries across processes:

```python
from strata_forge.llm import RedisCache

cache = RedisCache(url="redis://localhost:6379/0", prefix="forge:llm:", ttl_seconds=3600)
```

It needs the `[redis]` extra and serialises with `pickle`, so the Redis instance must be
trusted — never point it at a store other processes can write to. `make stack-up` brings up a
local Redis alongside Qdrant and Postgres.

Two things to watch. Changing the head model of a chain changes every key, so a chain reorder
invalidates the cache wholesale. And a cached response replays the original `usage` and `route`
with `cost_usd` zeroed — if you are summing `cost_usd` for billing, hits correctly contribute
nothing, but if you are summing `usage.total_tokens` for capacity planning, they still count.

See
[`examples/11_cache_hit.py`](https://github.com/GemovicNemanja/strata-forge/blob/main/examples/11_cache_hit.py).

## 4. Retry policy

Retry is per provider attempt, using tenacity with exponential backoff plus jitter. The knobs
live on the client constructor and apply to every attempt in the chain:

```python
client = LLMClient.with_fallbacks(
    ["claude-opus-4-8", "gpt-5.5"],
    retry_max_attempts=4,      # total attempts per provider, including the first
    retry_initial_wait=0.5,    # seconds
    retry_max_wait=15.0,       # cap on the backoff
    strict_bad_request=True,
)
```

Only transient failures are retried — `ProviderRateLimitError`, `ProviderTimeoutError`,
`ProviderServerError`, exported as `DEFAULT_RETRY_ON`. Auth, bad-request, and content-filter
errors are excluded on purpose: the request itself is the problem, so retrying it is pure waste.

Budget the worst case deliberately. `retry_max_attempts=5` over a three-entry chain with three
provider routes on the first entry is up to 25 provider calls before `FallbackExhaustedError`,
and every rate-limit backoff is real wall-clock latency your caller is waiting through. A
user-facing path usually wants fewer attempts and a shorter cap than a batch job does.

The same decorator is available for your own code:

```python
from strata_forge.core import DEFAULT_RETRY_ON, retry

@retry(max_attempts=3, initial_wait=0.5, retry_on=DEFAULT_RETRY_ON)
async def fetch_reference_data() -> dict[str, str]:
    ...
```

It works on sync and async callables and re-raises the final exception rather than wrapping it.

---

## Putting it together

```python
from strata_forge.core import BudgetContext
from strata_forge.llm import InMemoryCache, LLMClient, Message, ModelFallback

client = LLMClient.with_fallbacks(
    [
        ModelFallback(model="claude-opus-4-8", providers=("anthropic", "bedrock")),
        ModelFallback(model="gpt-5.5", providers=("openai", "azure")),
    ],
    cache=InMemoryCache(max_size=4096),
    retry_max_attempts=3,
    retry_initial_wait=0.5,
    retry_max_wait=10.0,
)

async def score_batch(prompts: list[str]) -> list[str]:
    async with BudgetContext(max_usd=5.00) as budget:
        results = []
        for prompt in prompts:
            response = await client.complete([Message.user(prompt)], max_tokens=500)
            results.append(response.text)
        print(f"spent ${budget.spent_usd:.4f} over {len(results)} calls")
        return results
```

For genuinely large batches, `strata_forge.compute.BatchInferenceRunner` adds bounded concurrency
over a single client and an `on_error="collect"` mode that returns per-prompt failures instead of
aborting — see
[Fine-tune on remote compute and serve the result](fine-tune-on-remote-compute-and-serve.md).

## Observability

None of this is worth much if you cannot see it after the fact. `strata_forge.tracing` sends
traces to Langfuse when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set and does
nothing at all when they are not. Independently, setting `FORGE_DIAGNOSTIC_ENABLED=1` appends one
NDJSON record per completed call to `FORGE_DIAGNOSTIC_PATH` (default
`./forge-diagnostic.ndjson`) — route, usage, cost, latency, correlation id, and the full request
and response text. That last part makes it the fastest way to answer "which provider actually
served that request, and what did it say", and also the reason it is off by default and belongs
on a machine you control. See [`docs/modules/tracing.md`](../modules/tracing.md).

## Where to go next

- [`docs/modules/llm.md`](../modules/llm.md) — the full client surface, registry, and error map.
- [`docs/modules/core.md`](../modules/core.md) — `BudgetContext`, `retry`, correlation ids.
- [ADR 0003](../architecture/adr/0003-exception-hierarchy.md) — why every provider failure is
  normalised into a `ForgeError` subclass at the transport seam.
- [ADR 0005](../architecture/adr/0005-two-axis-fallback.md) — the two-axis design.
