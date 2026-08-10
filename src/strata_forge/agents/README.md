# strata_forge.agents

A thin agent layer on top of `strata_forge.llm`. An `Agent` bundles an `LLMClient`, a system
prompt, and a set of tools; its `.run()` method dispatches through `LLMClient.run_tool_loop` (when
tools are present) or `LLMClient.complete` (when not), and returns an `AgentResult` carrying the
final text plus the conversation that produced it.

By design, the agent layer does not re-implement tool calling, structured output, or the multi-turn
tool loop — every primitive comes straight from `strata_forge.llm`. See
[ADR 0011](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/adr/0011-agents-thin-wrapper-over-forge-llm.md)
for the rationale.

Alongside `Agent` the module ships built-in tools (`calculator`, `fetch_url`, `fs_read_tool`,
`web_search_tool`), `ConversationMemory` and vector-backed `EpisodicMemory` that the caller threads
into `.run()`, and the multi-agent helpers `handoff` and `critic_refiner_run`. No optional extra is
required.

Reference:
[docs/modules/agents.md](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/agents.md).
