# ADR 0006 — Tool calling, structured output, and multimodal as `strata_forge.llm` primitives

**Status:** Accepted
**Date:** Initial scaffolding
**Supersedes:** —
**Superseded by:** partially — [ADR 0011](0011-agents-thin-wrapper-over-forge-llm.md), which
rejects PydanticAI for the agent runtime

## Context

Tool calling is a feature of every modern LLM provider. It's needed by agents (multi-turn loops with autonomous tool selection), but it's also needed by:

- Eval runners (a grader might call a tool to fetch reference data).
- One-off scripts (research workflows where the LLM picks among helper functions).
- Structured-output flows (some providers expose structured output *as* a forced tool call, so the implementation seam is shared).
- RAG pipelines (a retrieval step exposed as a tool the LLM can choose to invoke).

If tool plumbing lived in the agent module, each of these consumers would need to either (a) pull in the agent runtime as a transitive dependency for one capability they actually want, or (b) reimplement provider-specific tool serialization. Both options are bad.

The same logic applies to **structured output** (Pydantic-typed responses) and **multimodal image input**: they are LLM features, not agent features.

## Decision

Tool calling is a first-class feature of `strata_forge.llm`, shipped in `src/strata_forge/llm/tools.py`. The agent module (`strata_forge.agents`) reuses these primitives rather than reimplementing them. The same module also ships `strata_forge.llm.schemas` (structured output) and `strata_forge.llm.multimodal` (image input).

Concretely, `strata_forge.llm` exposes:

- `Tool` protocol: `.name`, `.description`, `.parameters_schema` (Pydantic model class), `.invoke(args)`.
- `@tool` decorator: wraps an `async def fn(args: ArgsModel) -> ...` into a `Tool`. Schema is generated from the Pydantic args model.
- Per-provider schema serializers (`to_openai_tool_schema`, `to_anthropic_tool_schema`, `to_gemini_tool_schema`).
- Message types: `AssistantMessage(content=..., tool_calls=[ToolCall(id, name, arguments)])`, `ToolResultMessage(tool_call_id, content, is_error=False)`.
- `LLMClient.run_tool_loop(messages, tools, max_iterations)` — multi-turn loop: completion → if `tool_use`, invoke each tool, append result messages, completion again, until `finish_reason != "tool_use"` or `max_iterations` hit. Cost/token/budget accrue across iterations.
- Capability gate: requesting tools against a model whose registry entry doesn't support `tool_calling` raises `RegistryError` before any provider call.

The agent module's contribution is: PydanticAI agent builder, built-in tools (`web_search`, `fs_read`, `fetch_url`, `calculator`), `ConversationMemory`, `EpisodicMemory`, multi-agent patterns. The agent module does NOT reimplement `Tool`, `@tool`, `run_tool_loop`, or message types — those are imported from `strata_forge.llm`.

## Consequences

**Positive**

- Eval runners, ad-hoc scripts, RAG pipelines, and agents share the same tool implementation and message types. Bug fixes and provider-format updates land in one place.
- Provider-specific quirks (OpenAI tool schema vs Anthropic tool-use format vs Gemini function declarations) live in one place — the `to_*_tool_schema` functions.
- The agent module stays small and focused: it composes primitives, it doesn't define them.
- Structured output can be expressed as a forced single-tool call internally when that gives better fidelity than provider JSON modes — sharing infrastructure with tool calling is what enables this.

**Negative**

- `strata_forge.llm` is larger and more featured than a strict "just send completions" abstraction. The module is harder to read end-to-end.

**Mitigations**

- Tool-related code is in `tools.py` (separate file from `client.py`'s completion core). `client.py` imports from `tools.py`; the dependency direction is clean.
- Structured output is in `schemas.py`; multimodal in `multimodal.py`. The module is internally well-separated.
- VCR cassettes cover the full tool-loop flow per provider — regressions are caught.

## Alternatives considered

1. **Tools live in `strata_forge.agents`; other consumers import from there.** Forces every tool-using consumer to take the agent module as a transitive dep. Couples agents to evals/RAG/scripts in a direction that doesn't match the dependency graph. Rejected.
2. **A separate `strata_forge.tools` module.** A reasonable design, but tool serialization is tightly coupled to provider message construction — splitting them creates a circular-feeling dependency between `strata_forge.llm` and `strata_forge.tools`. Rejected on cohesion grounds.
3. **No first-class tool support in `strata_forge.llm`; everyone writes their own.** Loses provider-format normalization, multi-turn loop semantics, capability gates. Defeats the "batteries included" pillar. Rejected.
