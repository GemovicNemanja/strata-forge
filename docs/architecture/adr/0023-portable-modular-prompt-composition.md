# ADR 0023 — portable modular prompt composition

Status: accepted for implementation by the platform coordinator; runtime implementation and cross-language acceptance are separate evidence.

## Context

Reusable instructions need immutable composition, exact output attribution and preserved message/tool/skill boundaries. The existing Jinja2 template API remains supported. The earlier TypeScript experiment does not define production canonical bytes or portable source offsets. A browser preview and a Python worker must not assign different identities to the same build.

## Decision

Forge's pure Python compiler is authoritative. Persisted resources, permissions, publication and moving channels belong to the server; the browser consumes builds. A TypeScript consumer or renderer must pass the same fixtures. It cannot publish an alternative compiler identity.

Use an explicit ordered AST of literal text, declared variable references and pinned module versions. Literal text never becomes a template implicitly: braces, Jinja, f-string fragments and Markdown remain literal. Existing template dialects require separately named import adapters with before/after equality checks. No arbitrary expressions, filesystem includes, remote imports or callable resolution are admitted.

The [portable contract](../../contracts/modular-prompts-v1.md) defines canonical JSON bytes, immutable dependency locks, target boundaries, UTF-8 source offsets and rendering identity. Builds and renders are separate immutable values. Fetch, local render, caller-reported dispatch, instrumented request and provider response remain distinct evidence. Composition alone makes no claim about what reached a model.

Compiler errors cover structural impossibility. Duplicate passages and possible natural-language conflicts are advisory; instructions are never silently rewritten or reordered. Dependency impact is an explicit candidate operation, never mutation of a published build.

## Consequences

Python and TypeScript must share byte/hash/source-map fixtures, including Unicode and newline edge cases. The portable profile deliberately limits numeric values to safe integers; generation settings with floats belong to a separate execution-capture contract. The existing Jinja2 renderer retains its own API and semantics. This decision does not install persistence, an editor, MCP, a model-backed analyzer or a deployment mechanism.
