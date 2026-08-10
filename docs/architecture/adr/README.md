# Architecture decision records

An ADR captures one architectural decision and the reasoning that produced it, so that a reader six
months later can tell the difference between a considered trade-off and an accident. Every record
here is `Accepted`, and none has been withdrawn — one clause of ADR 0006 was later overtaken by
ADR 0011, which its header records.

Start with [`../overview.md`](../overview.md) for the narrative and
[`../module-boundaries.md`](../module-boundaries.md) for the import rules those decisions imply.

## Index

| # | Title | Decision |
|---|---|---|
| [0001](0001-litellm-as-transport.md) | LiteLLM as the provider transport | Let LiteLLM own HTTP, auth, streaming, and retry plumbing across all six provider surfaces; strata-forge owns a typed Pydantic layer above it, and provider SDKs are imported only inside `llm/providers/`. |
| [0002](0002-async-only-public-api.md) | Async-only public API, sync wrappers in a single module | Public I/O is `async` end to end; synchronous convenience lives in exactly one module (`strata_forge.sync`) as `asyncio.run` shims, so there is never a second code path to keep in step. |
| [0003](0003-exception-hierarchy.md) | Exception-based error model with a unified `ForgeError` hierarchy | Failures raise rather than return. Everything roots at `ForgeError`, LiteLLM exceptions normalize into the `ProviderError` subtree at the seam, and `Result`-style returns were explicitly rejected. |
| [0004](0004-model-registry-scope.md) | Model registry scoped to latest OpenAI, Anthropic, Google foundation models | Ship registry entries only for current frontier models from three vendors; self-hosted OpenAI-compatible endpoints go through the `openai_compat` provider instead of registry rows, and a new vendor needs a new ADR. |
| [0005](0005-two-axis-fallback.md) | Two-axis fallback (model-level + provider-level) | `ModelFallback(model, providers=[...])` fails over between routes for one logical model; the outer list fails over between models. Per-error rules decide which axis advances, and a content-filter refusal short-circuits both. |
| [0006](0006-tool-calling-as-llm-primitive.md) | Tool calling, structured output, and multimodal as `strata_forge.llm` primitives | All three ship inside the LLM module rather than as add-ons, so higher layers compose `Tool`, `@tool`, `run_tool_loop`, and the message types instead of reimplementing them. Its sketch of the agent module as a PydanticAI builder was later overtaken by [ADR 0011](0011-agents-thin-wrapper-over-forge-llm.md). |
| [0007](0007-stable-prefix-dynamic-suffix-prompts.md) | Prompts are structurally split into stable prefix and dynamic suffix | Every `PromptTemplate` must declare both sections and tag its variables to one of them, making prompt-cache friendliness a property the type system checks rather than a convention authors remember. |
| [0008](0008-tracing-as-cross-cutting.md) | `strata_forge.tracing` is layered above every other module, not threaded through them | Tracing imports downward and nothing imports it. Model calls are captured by registering LiteLLM's Langfuse callback; everything else is opt-in at the call site via `@traced` and `traced_span`. |
| [0009](0009-datasets-langfuse-canonical-hf-exchange.md) | Langfuse is canonical for Forge datasets; HF Datasets is the exchange format | Datasets persist in Langfuse so traces, annotations, and eval data stay in one system; Hugging Face Datasets is the import/export bridge; frozen Pydantic models with content-hash versions are the canonical in-memory shape. |
| [0010](0010-evals-experiment-as-data-pluggable-graders.md) | Evals: experiments as data, graders as pluggable async protocols, four-axis matrix | An `Experiment` is frozen declarative data with content-hash identity, graders are caller-supplied async `Protocol` implementations, and the runner walks the `models × prompts × items × graders` matrix and returns outcomes for downstream layers to aggregate. |
| [0011](0011-agents-thin-wrapper-over-forge-llm.md) | `strata_forge.agents` is a thin wrapper over `strata_forge.llm`, not a parallel runtime | The agent layer adds `Agent` and `AgentResult` and nothing else — no tool abstraction, no message types, no loop, no structured-output path. Building on PydanticAI was considered and rejected. |
| [0012](0012-rag-protocols-and-vector-store-relocation.md) | RAG module owns the vector-store / embedder / chunker / retriever Protocols | Vector-store primitives move out of `agents.memory` into `strata_forge.rag`, fixing the dependency arrow to `agents → rag`, and RAG anchors on runtime-checkable Protocols so implementations need no inheritance. The embedder calls LiteLLM directly rather than through `LLMClient`. |
| [0013](0013-compute-task-and-backend-shapes.md) | `strata_forge.compute` ships task-as-data + a Backend Protocol | A `Task` is frozen, YAML-round-trippable data, and every execution target implements one async `Backend` Protocol covering the whole job lifecycle, so local subprocess, SSH, and SkyPilot are interchangeable at the call site. |
| [0014](0014-streaming-tool-loop-event-protocol.md) | The streaming tool loop is a typed event protocol, not a streamed `LLMResponse` | `stream_tool_loop` yields a flat discriminated union of loop events with tool calls emitted whole, never partial, and sits beside `run_tool_loop` as a separate method rather than behind a flag that would make the return type conditional. |
| [0015](0015-client-executed-tools-suspend-the-streaming-loop.md) | Client-executed tools suspend the streaming loop | `ToolDeclaration` describes a tool the library cannot run; when the model calls one, the loop finishes its executable calls, emits a terminal `PendingToolCalls` event, and returns. Resumption is ordinary message input, not a second API. |
| [0016](0016-backend-read-file.md) | `Backend.read_file` for workdir-confined side-channel reads | Add one `Backend` method for reading a file a job writes alongside its output, confined to the job's working directory by a shared path guard and returning `""` when the file does not exist yet. The SkyPilot implementation is deferred. |

## When a change needs an ADR

Write one when a change would alter something a future contributor could not re-derive from the
code alone:

- It contradicts an architecture principle in
  [`CLAUDE.md`](https://github.com/GemovicNemanja/strata-forge/blob/main/CLAUDE.md) §3 — a
  synchronous public I/O surface, a provider SDK reached from outside `llm/providers/`, a module
  importing across a boundary the table in
  [`../module-boundaries.md`](../module-boundaries.md) forbids.
- It changes a cross-module contract: an event protocol, a Protocol's method set, the shape of a
  persisted or serialized type, the error taxonomy.
- It picks one dependency or vendor over a live alternative, or reverses a choice an existing ADR
  made.
- It adds a top-level module, or a new provider vendor to the model registry.

Routine work does not: adding a grader, a chunker, a CLI flag, a registry entry for a model from a
vendor already covered, or a test. If you find yourself writing a long PR description arguing for a
design, that argument belongs in an ADR.

## Writing one

Create `docs/architecture/adr/NNNN-short-slug.md`, taking the next free number in sequence. Numbers
are never reused, and a rejected proposal keeps its number with `Status: Rejected` so the sequence
stays a complete record of what was considered.

```markdown
# ADR NNNN — Short imperative title

**Status:** Proposed | Accepted | Rejected | Superseded
**Date:** YYYY-MM-DD
**Supersedes:** —
**Superseded by:** —

## Context

The forces in play: the requirement, the constraints, what the alternatives were, and what makes
the choice non-obvious. Written so a reader who was not there can reconstruct the pressure.

## Decision

What was decided, in the present tense and in enough concrete detail — signatures, type shapes,
module paths — that a reader can check the code against it.

## Consequences

What this buys, what it costs, and what it forecloses. Name the downsides explicitly; an ADR with
no costs section is a decision that was not examined.
```

Keep the whole record under roughly 200 lines. An ADR is a decision, not a design document — API
detail belongs in `docs/modules/<name>.md`, and usage belongs in `docs/recipes/`. Cross-link both
ways: the ADR names the modules it governs, and those modules' reference pages link back to it.

## Immutability

**A merged ADR is never edited.** Its Context describes the situation at the time it was written and
its Decision records what was chosen then; rewriting either destroys the only account of why the
code looks the way it does. When a decision no longer holds, write a new ADR that supersedes it:

1. The new ADR fills in `**Supersedes:** ADR NNNN` and restates the old context before explaining
   what changed.
2. The old ADR's `**Status:**` becomes `Superseded` and its `**Superseded by:**` names the new
   number. Those two header fields are the only lines an existing ADR may ever change.

Everything else about the old record stays exactly as merged, including any statement that has since
become inaccurate. If you want a description of how the code behaves today, that is what
[`../overview.md`](../overview.md), [`../module-boundaries.md`](../module-boundaries.md), and
[`docs/modules/`](../../modules/README.md) are for.
