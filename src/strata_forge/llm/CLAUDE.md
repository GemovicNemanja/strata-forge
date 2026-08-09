# Agent rules — strata_forge.llm

`strata_forge.llm` is the central LLM abstraction. Every higher-level module that talks to a model goes through here.

## Purpose

Provider-abstracted async LLM client built on LiteLLM ([ADR 0001](../../../docs/architecture/adr/0001-litellm-as-transport.md)). The module owns the typed layer: Pydantic-validated messages and responses, structured output, tool calling, multimodal image input, streaming, two-axis fallback ([ADR 0005](../../../docs/architecture/adr/0005-two-axis-fallback.md)), provider-agnostic caching, model registry ([ADR 0004](../../../docs/architecture/adr/0004-model-registry-scope.md)), cost/token accounting, and the diagnostic NDJSON dump. Tool calling, structured output, and multimodal are first-class here ([ADR 0006](../../../docs/architecture/adr/0006-tool-calling-as-llm-primitive.md)) — they are NOT in the agent module.

## Boundaries

- **Owns:** `client.py`, `messages.py`, `responses.py`, `loop_events.py`, `schemas.py`, `tools.py`, `multimodal.py`, `streaming.py`, `tokens.py`, `cost.py`, `cache.py`, `fallback.py`, `routing.py`, `errors.py`, `registry.py`, `registry_data.yaml`, `diagnostic.py`, `providers/`.
- **Imports from inside `forge`:** `strata_forge.core`, `strata_forge.config`. Nothing higher.
- **Imports of provider SDKs** (`anthropic`, `openai`, `google-cloud-aiplatform`, `boto3`): ONLY inside `src/strata_forge/llm/providers/`. Everywhere else uses `LLMClient` or the `provider_extras` passthrough.
- **Does NOT:** manage prompts, store datasets, run evaluations, build agents, perform retrieval, orchestrate remote compute.

## Public API

The module's `__init__.py` re-exports a curated surface:

- `LLMClient` — the main async client.
- Message types: `Message`, `SystemMessage`, `UserMessage`, `AssistantMessage`, `ToolResultMessage`.
- Response types: `LLMResponse`, `ResponseChunk`, `ToolCall`, `FinishReason`.
- `ModelFallback` — explicit two-axis fallback entry.
- `Tool`, `@tool` — tool calling primitives. `ToolDeclaration` (+ the `AnyTool` union) — a declaration-only tool the caller executes out-of-band ([ADR 0015](../../../docs/architecture/adr/0015-client-executed-tools-suspend-the-streaming-loop.md)).
- Streaming tool loop: `LLMClient.stream_tool_loop` + the `LoopEvent` union (`IterationStart`, `TextDelta`, `ToolCallStarted`, `ToolResult`, `PendingToolCalls`, `Done`, `LoopError`) from `loop_events.py` ([ADR 0014](../../../docs/architecture/adr/0014-streaming-tool-loop-event-protocol.md)). A turn that calls a `ToolDeclaration` suspends with terminal `PendingToolCalls`; resume by re-invoking with `input + event.messages + a ToolResultMessage per pending call` ([ADR 0015](../../../docs/architecture/adr/0015-client-executed-tools-suspend-the-streaming-loop.md)).
- `ImageContent` — multimodal image input.
- Errors: anything raised from this module is a `ProviderError` subclass or another `ForgeError` subclass.

The `strata_forge.sync` namespace re-exports `complete`, `stream`, `complete_structured`, `run_tool_loop` as sync wrappers.

## Internal patterns

- Every provider call routes through LiteLLM via the `ProviderClient` base. LiteLLM exceptions are normalized at the seam in `errors.py` (`map_litellm_exception`) — raw LiteLLM exceptions never bubble out.
- The model registry (`registry.py` + `registry_data.yaml`) is the source of truth for pricing, capability flags, and provider routes. Pricing is NEVER hard-coded in Python.
- `routing.py` resolves `(logical_model, optional_provider)` → concrete `ModelRoute(provider, provider_model_id)`. Aliases are resolved before lookup.
- Cache keys are **provider-agnostic**: a hit on `claude-opus-4-7` is valid regardless of which provider served the original call. Cache key = SHA256(canonical(logical model, messages, sampling params, response_format, tool schemas hash, provider_extras hash)). Streaming responses are NOT cached.
- Fallback: provider-level (inner) within a `ModelFallback`; model-level (outer) across entries. See ADR 0005 for the advance/abort rules per error class.
- Tool capability gate: requesting tools against a model whose registry entry has `capabilities.tool_calling = false` raises `RegistryError` **before** any provider call. An `openai_compat`/OpenRouter model isn't in the registry, so it's let through by default; construct the client with `require_tool_support=True` to instead raise `RegistryError(reason="capability_unknown")` pre-flight for such unconfirmable models.
- Diagnostic dump: when `FORGE_DIAGNOSTIC_ENABLED=1`, every completed call (including each iteration of a tool loop) appends a JSON record to `${FORGE_DIAGNOSTIC_PATH:-./forge-diagnostic.ndjson}`. Independent of Langfuse.

## Test expectations

- Unit tests under `tests/unit/llm/`, one file per source module.
- Coverage: the enforced gate is the repo-wide 85 % line floor (`fail_under` in `pyproject.toml`). Per root CLAUDE.md §5, `llm/` is held to a higher bar by convention — treat a drop below 90 % here as a regression even though no per-module gate enforces it.
- VCR cassettes under `tests/vcr/cassettes/<provider>/` cover the matrix of (provider × scenario): basic completion, streaming, structured output, multimodal, single tool call, multi-turn tool loop, rate-limit retry, content filter, provider-level fallthrough, model-level fallthrough.
- Snapshot tests (`syrupy`) for per-provider tool schema serialization and structured-output schema serialization.
- Hypothesis property tests for cache key stability and token counter monotonicity.
- Live integration tests (`@pytest.mark.live`) are env-gated; nightly only.

## Gotchas

- Provider SDKs imported outside `providers/` is a hard violation. The escape hatch is `provider_extras` — never bypass that.
- The cache key is intentionally provider-agnostic; do NOT include `route.provider` in it. Including the provider would mean a cache miss on every provider failover, defeating the cache's purpose.
- When LiteLLM updates its exception classes, `map_litellm_exception` must be updated — the snapshot tests in `tests/unit/llm/test_errors.py` catch the drift.
- Tool argument validation happens on the way IN (`@tool` validates against the Pydantic args model) AND the way OUT (provider tool-call args are parsed back through the Pydantic model). Both checks must remain.
- `run_tool_loop` accrues cost/token across iterations; it must consult any active `BudgetContext` between iterations and raise `BudgetExceededError` cleanly.
- Never log full prompts at `INFO` level — `DEBUG` only. The diagnostic NDJSON dump is the appropriate channel for full request/response payloads.
- Pricing edits go in `registry_data.yaml` with a source-link comment. CI ought to fail if a registry entry's `pricing_per_million_tokens` has changed without a coincident YAML comment update — that's a future linter task.

## When to update this file

- Adding a new provider module (e.g. `strata_forge.llm.providers.mistral`).
- Adding a new public capability to `LLMClient` (e.g. embeddings, batch API).
- Changing the cache key formula.
- Changing the fallback advance/abort policy.
- Adding new model entries to the registry (no ADR needed for new entries within the OpenAI/Anthropic/Google scope; ADR required for new vendors).
- Changing the LiteLLM transport seam.
