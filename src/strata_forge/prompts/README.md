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

The independent modular API composes literal text, declared scalar variables and exact pinned
modules into immutable chat or textual-artifact builds. It freezes transitive dependencies,
declared constraints, portable content identities and UTF-8 source maps. Rendering never
reparses inserted values. It performs no I/O, model calls, publication or permission checks.

`CompositionRegistry`, `compile_surface` and `render_build` are async; pure byte/schema helpers
and snapshot inspection are synchronous under ADR 0023. Python is authoritative. The Node
reference consumer passes the same byte/render fixtures and cannot publish an alternative
compiler identity. Existing Jinja2 templates and stores keep their API.

`consume_build` accepts local caller-owned canonical bytes with mandatory expected SHA-256,
byte size and an explicit `LocalBuildSource`; `render_consumed` rechecks that snapshot and
delegates to the same modular renderer. Descriptive partitions/native locators confer no
remote access or provenance attestation. The private Node 24 package in
[`packages/prompt-sdk/`](../../../packages/prompt-sdk/README.md) consumes the same fixtures and
contains the extracted reference kernel, with no separate compiler or hidden distribution build.
Neither consumer fetches content, stores a cache, calls models or captures attempts.

Module rules: [`CLAUDE.md`](CLAUDE.md). References:
[`Jinja2/cache-aware API`](../../../docs/modules/prompts.md),
[`modular composition API`](../../../docs/modules/prompt-composition.md).
