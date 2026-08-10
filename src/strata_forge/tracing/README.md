# strata_forge.tracing

Langfuse-based observability layered above every other module. `install_litellm_callback()`
registers LiteLLM's Langfuse callback once, so every LLM call is traced without `strata_forge.llm`
importing anything from here. A `@traced` decorator and the `traced_span()` async context manager
cover non-LLM code (eval graders, agent loops, RAG pipelines); `score_trace` / `score_observation`
and the `record_*_metric` helpers attach feedback and application metrics to existing traces.

Everything degrades to a silent no-op when Langfuse is unconfigured, so instrumented code runs
unchanged without credentials. Needs the `[langfuse]` extra plus `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY`.

Cross-cutting by design: no other `strata_forge.*` module imports `strata_forge.tracing`. See
[ADR 0008](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/adr/0008-tracing-as-cross-cutting.md)
for the rationale.

Reference:
[docs/modules/tracing.md](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/tracing.md).
