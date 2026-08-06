# ADR 0008 — `strata_forge.tracing` is layered above every other module, not threaded through them

**Status:** Accepted
**Date:** Initial scaffolding for `strata_forge.tracing`
**Supersedes:** —
**Superseded by:** —

## Context

Phase 1 deliberately left observability as a separate concern. The LLM
module produces correlation IDs (`strata_forge.core.ids.correlation_id_var`)
and emits NDJSON diagnostic records when `FORGE_DIAGNOSTIC_ENABLED=1`,
but it never imports anything from a tracing layer — by design.

Phase 2.2 introduces that tracing layer. The obvious shape would have
been to add `from strata_forge.tracing import traced` calls inside `strata_forge.llm`
so every completion is traced automatically. We're explicitly rejecting
that shape, for four reasons:

1. **Dependency direction.** `strata_forge.tracing` already needs to import
   message and response types from `strata_forge.llm` to render trace inputs
   and outputs sensibly. If `strata_forge.llm` also imports from
   `strata_forge.tracing`, we get a circular dependency that has to be broken
   by lazy imports or partial-module hacks — both of which are
   load-order hazards.

2. **Optional-extras hygiene.** `strata_forge.tracing` is gated by the
   `[langfuse]` extra. A Forge install without the extra still needs
   `strata_forge.llm` to work end-to-end. Threading tracing through the LLM
   module either makes the extra effectively required (every call
   tries to invoke a tracing function) or sprinkles `try/except
   ImportError` everywhere — neither is clean.

3. **Testing surface.** Unit tests for the LLM client shouldn't have
   to mock a tracing layer they don't care about. The current 922
   unit tests pass without any tracing setup; that's a feature.

4. **Composition with non-Forge code.** The Langfuse Python SDK
   integrates cleanly into LiteLLM via a callback registered at
   process startup. Wiring it once in app initialization gives every
   downstream LiteLLM call automatic tracing — including calls from
   third-party libraries that go through LiteLLM but don't know about
   Forge. Putting the integration inside `strata_forge.llm` would miss those
   call sites.

## Decision

`strata_forge.tracing` is a **cross-cutting layer that wraps every other
module from above**. The dependency arrow points one way:

```
                 strata_forge.tracing
                    │  (imports from)
                    ▼
strata_forge.llm    strata_forge.prompts    (strata_forge.evals, strata_forge.agents, …)
                    │
                    ▼
                strata_forge.core
```

`strata_forge.tracing` may import:

- `strata_forge.core` — for the structlog logger and the
  `correlation_id_var` ContextVar (so trace IDs propagate through log
  records).
- `strata_forge.config` — for the `LangfuseConfig` sub-model.
- `strata_forge.llm` — for the message/response types, only for serializing
  trace inputs/outputs into Langfuse's payload shape.
- `strata_forge.prompts` — for the same reason (when later sub-phases need
  to attach prompt metadata to a trace).
- `langfuse` — lazy-imported behind the `[langfuse]` extra.

`strata_forge.llm`, `strata_forge.prompts`, and every other downstream module MUST
NOT import from `strata_forge.tracing`. The integration points are:

- **For LLM calls.** `strata_forge.tracing.litellm_callback.install_litellm_callback()`
  registers LiteLLM's existing Langfuse callback. The callback runs
  inside LiteLLM, observes every completion, and reports to Langfuse —
  with zero `strata_forge.llm` code change. Applications call this once at
  startup; CI / scripts that don't want tracing don't call it.

- **For non-LLM code.** `@traced` and `traced_span()` are applied at
  the call site by whoever wants the trace — eval runners, agent
  loops, RAG pipelines. They are *additive*; un-decorated functions
  work exactly as before.

- **For scoring / metrics.** `score_trace`, `score_observation`,
  `record_numeric_metric`, `record_categorical_metric` are called
  after the fact by whoever has the trace ID — graders, post-hoc
  pipelines.

When Langfuse isn't configured (no public/secret key in
`LangfuseConfig`), every public function in `strata_forge.tracing` is a
silent no-op. There's never a code path where missing Langfuse
configuration crashes a production call.

## Consequences

**Positive**

- The Phase 1 LLM module stays untouched. Its 564 lines and 922 unit
  tests don't need updates to gain Langfuse tracing — `install_litellm_callback()`
  flips it on at process startup.
- Test isolation: unit tests for any downstream module remain free of
  tracing concerns unless that module explicitly opted in via
  `@traced`.
- Optional-extras hygiene: a Forge install without `[langfuse]`
  imports `strata_forge.tracing` cleanly; only `get_client()` and
  `install_litellm_callback()` lazy-import `langfuse`, and both
  no-op gracefully without credentials.
- Composition: third-party code that uses LiteLLM directly (not via
  `strata_forge.llm`) gets the same trace records.

**Negative**

- The integration is less discoverable. A reader of `strata_forge.llm.client`
  won't see *any* mention of tracing and won't know it happens. The
  module reference doc compensates with an explicit cross-link.
- Trace coverage depends on `install_litellm_callback()` being called
  before any LLM activity. Forgetting the call means no traces; the
  failure mode is silent ("why aren't my traces showing up?").

**Mitigations**

- The `strata-forge` CLI's entry-point installs the callback automatically
  when Langfuse is configured. Scripts using the library directly
  must install it themselves; the docs spell this out and the
  example scripts demonstrate it.
- `strata-forge doctor` reports whether the callback has been installed and
  whether Langfuse is reachable, so the silent-failure case surfaces
  on diagnostic runs.

## Alternatives considered

1. **`strata_forge.llm` imports `strata_forge.tracing` and wraps every call.**
   Rejected for the four reasons in Context — circular dep,
   extras hygiene, test surface, and missing non-Forge LiteLLM calls.

2. **Inversion of control: `strata_forge.llm` exposes hooks, `strata_forge.tracing`
   subscribes.** Cleaner in theory, but requires building a generic
   hook bus when we already have LiteLLM's callback mechanism for
   free. Rejected on "use the upstream extension point" grounds.

3. **No `strata_forge.tracing` module at all; users wire LiteLLM's callback
   themselves.** Loses the Forge-flavored conveniences (`@traced` on
   non-LLM code, score helpers, `correlation_id` integration).
   Rejected: we ship enough syntactic sugar that it's worth the
   module.
