# strata_forge.prompts

Typed prompt authoring backed by a sandboxed Jinja2 environment and a store-agnostic registry.
Templates are **structurally split** into a stable prefix and a dynamic suffix, so the part of a
prompt that provider caches can reuse is identified in the template definition rather than guessed
at call time. See
[ADR 0007](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/adr/0007-stable-prefix-dynamic-suffix-prompts.md)
for the rationale.

`render(template, variables)` compiles a `PromptTemplate` into a `RenderedPrompt`: the
`strata_forge.llm` messages to send, the stable/dynamic split with its digests, and a `CacheHints`
record describing the cacheable prefix for the caller to act on or record. `PromptRegistry` stores
versioned templates through a pluggable backend — `InMemoryPromptStore` for tests and scripts,
`LangfusePromptStore` (needs the `[langfuse]` extra) for versioned production prompts.

Reference:
[docs/modules/prompts.md](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/prompts.md).
