# strata_forge.prompts

Typed prompt authoring backed by a sandboxed Jinja2 environment and a
store-agnostic registry. Templates are **structurally split** into a
stable prefix and a dynamic suffix so provider prompt caching works by
default — Anthropic markers, OpenAI auto-caching, and Gemini
`CachedContent` hints all derive from the same template definition. See
[ADR 0007](../../../docs/architecture/adr/0007-stable-prefix-dynamic-suffix-prompts.md)
for the rationale.

The module compiles `(PromptTemplate, variables)` to
`list[strata_forge.llm.AnyMessage]` plus per-message cache hints; the LLM
client uses those hints to populate provider-specific cache mechanics.
Storage is pluggable: an in-memory store for tests and quick scripts, a
Langfuse-backed store (lazy import behind the `[langfuse]` extra) for
versioned production prompts.

> **Status.** Implementation in progress as part of Phase 2. See
> [`docs/roadmap.md`](../../../docs/roadmap.md) for the per-sub-phase
> deliverable list.

Module rules: [`CLAUDE.md`](CLAUDE.md). Full reference once shipped:
`docs/modules/prompts.md`.
