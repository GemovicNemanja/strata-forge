# forge.agents

A thin agent layer on top of `forge.llm`. An `Agent` bundles an
`LLMClient`, a system prompt, and a set of tools; its `.run()`
method dispatches through `LLMClient.run_tool_loop`
(when tools are present) or `LLMClient.complete` (when not), and
returns an `AgentResult` carrying the final text plus the input
conversation.

By design, the agent layer does not re-implement tool calling,
structured output, or the multi-turn tool loop — every primitive
comes straight from `forge.llm`. See
[ADR 0011](../../../docs/architecture/adr/0011-agents-thin-wrapper-over-forge-llm.md)
for the rationale.

The 3.1 foundation ships the `Agent` runtime and the `AgentResult`
shape. Built-in tools (Phase 3.2), conversation and episodic memory
(Phase 3.3), and multi-agent patterns — hand-off and critic-refiner
(Phase 3.4) — land in subsequent sub-phases. See
`docs/roadmap.md` for status.
