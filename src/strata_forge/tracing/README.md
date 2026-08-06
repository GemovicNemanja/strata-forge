# strata_forge.tracing

Langfuse-based observability layered above every other Forge module.
The LiteLLM Langfuse callback (registered by
:func:`install_litellm_callback`) auto-traces every LLM call through
LiteLLM without `strata_forge.llm` importing anything from this module. A
`@traced` decorator and `traced_span()` async context manager cover
non-LLM code (eval graders, agent loops, RAG pipelines); score and
metric helpers attach feedback and application metrics to traces.

Cross-cutting by design: no other `strata_forge.*` module imports
`strata_forge.tracing`. See
[ADR 0008](../../../docs/architecture/adr/0008-tracing-as-cross-cutting.md)
for the rationale.

> **Status.** Implementation in progress as part of Phase 2.2. See
> [`docs/roadmap.md`](../../../docs/roadmap.md) for the per-sub-phase
> deliverable list.

Module rules: [`CLAUDE.md`](CLAUDE.md). Full reference once shipped:
`docs/modules/tracing.md`.
