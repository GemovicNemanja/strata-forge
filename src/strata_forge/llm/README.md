# strata_forge.llm

Provider-abstracted async LLM client built on LiteLLM. `LLMClient` is the entry point;
`Message` and `LLMResponse` are the typed shapes every call goes through. The module owns the whole
typed layer: Pydantic-validated messages and responses, structured output (provider-native channel
with a reprompt fallback), tool calling via `Tool` / `@tool` and a multi-turn `run_tool_loop`,
multimodal image input, streaming with partial-JSON and tool-call accumulators, two-axis fallback
(different model, or the same model on a different provider route), a provider-agnostic response
cache, and a curated model registry with pricing and capability metadata.

The client itself needs no optional extra and picks provider credentials up from the usual
environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …); `RedisCache` needs `[redis]` and
image downscaling needs `[multimodal]`.

Reference:
[docs/modules/llm.md](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/llm.md).
