# strata_forge.llm

Provider-abstracted async LLM client built on LiteLLM. Owns the typed layer: Pydantic-validated messages and responses, structured output (provider-native channel with reprompt fallback), tool calling with a multi-turn `run_tool_loop`, multimodal image input, streaming with partial-JSON and tool-call accumulators, two-axis fallback (different model OR same model on a different provider route), provider-agnostic response caching, and a model registry scoped to the latest OpenAI / Anthropic / Google foundation models.

> Implementation pending. See `docs/roadmap.md` for current status.

The separate `generation` data module defines closed local-generation controls and preserved
channel/result shapes, with a finite inert JSON-schema dialect. These types grant no
native capability. The additive generate_one operation refuses without fixed native endpoint
observation; it leaves existing completion/stream APIs unchanged. See
[ADR0029](../../../docs/architecture/adr/0029-closed-local-generation-protocol.md).
