# Agent rules — strata_forge.agents

`strata_forge.agents` is a thin composition layer over :mod:`strata_forge.llm`.
The agent runtime never re-implements tool calling, structured
output, the tool loop, or message handling — every one of those
primitives comes from :mod:`strata_forge.llm` directly. See
[ADR 0011](../../../docs/architecture/adr/0011-agents-thin-wrapper-over-forge-llm.md)
for the rationale.

## Purpose

- :class:`Agent` — a single-agent runtime that bundles an
  :class:`LLMClient`, a system prompt, and a set of tools.
  ``.run(user_input)`` dispatches through
  :meth:`LLMClient.run_tool_loop` (when tools are present) or
  :meth:`LLMClient.complete` (when not).
  ``.run_streaming(user_input)`` yields the tool loop as live
  :data:`LoopEvent`s via :meth:`LLMClient.stream_tool_loop`.
- :class:`AgentResult` — the output shape: text + input messages
  + raw :class:`LLMResponse` (so callers see Forge-canonical
  accounting fields, not an agent-specific wrapper).
- Built-in tools: ``web_search_tool``, ``fetch_url``,
  ``fs_read_tool``, ``calculator``. Each is a concrete :class:`Tool`
  built via :func:`strata_forge.llm.tool`, with a Pydantic args model
  (:class:`WebSearchArgs`, :class:`FetchURLArgs`,
  :class:`FSReadArgs`, :class:`CalculatorArgs`).
- Memory: :class:`ConversationMemory` for in-process message-history
  trimming; :class:`EpisodicMemory` against the pluggable
  :class:`VectorStore` Protocol. The Protocol,
  :class:`InMemoryVectorStore`, :class:`VectorItem`, and
  :class:`VectorSearchResult` live in :mod:`strata_forge.rag` per
  ADR 0012 — ``memory/vector_store.py`` is a re-export shim, and
  :class:`strata_forge.rag.QdrantVectorStore` is the persistent
  implementation of the same Protocol.
- Multi-agent patterns: :func:`handoff` (router-driven delegation,
  returning a :class:`RouterChoice`) and :func:`critic_refiner_run`
  (critique/refine loop over a :class:`CritiqueVerdict`).

## Boundaries

- **Owns:** `agent.py`, `multi_agent.py`, `tools/`
  (`web_search.py`, `fetch_url.py`, `fs_read.py`, `calculator.py`),
  `memory/` (`conversation.py`, `episodic.py`, `vector_store.py`).
- **Imports from inside `forge`:** :mod:`strata_forge.core` (errors),
  :mod:`strata_forge.config` (settings if built-in tools need them),
  :mod:`strata_forge.llm` (everything tool/message/response-related),
  optionally :mod:`strata_forge.prompts` (when rendering structured
  prompts as system messages), and :mod:`strata_forge.rag` for the
  vector-store Protocol.
- **Does NOT import** :mod:`strata_forge.tracing` (that module wraps
  every Forge module from above; the dependency arrow points
  one way — see ADR 0008).
- **Does NOT re-implement** any primitive that lives in
  :mod:`strata_forge.llm`. The agent's tools field is
  ``tuple[strata_forge.llm.Tool, ...]`` — no new tool abstraction. Tool
  invocation goes through :meth:`Tool.invoke`. The multi-turn
  loop goes through :meth:`LLMClient.run_tool_loop`. Structured
  output goes through :meth:`LLMClient.complete_structured`.
- **External deps:** Pydantic and ``httpx`` (both in the core
  install) — ``fetch_url`` uses ``httpx.AsyncClient``. No optional
  extras: ``web_search_tool`` is a factory that takes a
  caller-supplied :data:`SearchBackend` callable rather than
  depending on a search SDK.

## Public API

The module's ``__init__.py`` re-exports exactly these symbols
(mirror any change here into ``__all__``):

- Agent runtime: :class:`Agent`, :class:`AgentResult`.
- Built-in tools: :func:`web_search_tool`, :func:`fs_read_tool`,
  :data:`fetch_url`, :data:`calculator`, plus their args models
  :class:`WebSearchArgs`, :class:`FSReadArgs`,
  :class:`FetchURLArgs`, :class:`CalculatorArgs` and the
  web-search shapes :data:`SearchBackend`, :class:`SearchResult`.
- Memory: :class:`ConversationMemory`, :class:`EpisodicMemory`,
  :data:`EmbedFn`, and the vector-store names re-exported from
  :mod:`strata_forge.rag`: :class:`VectorStore`,
  :class:`InMemoryVectorStore`, :class:`VectorItem`,
  :class:`VectorSearchResult`.
- Multi-agent: :func:`handoff`, :func:`critic_refiner_run`,
  :class:`RouterChoice`, :class:`CritiqueVerdict`.
- Message types and tool primitives forwarded from
  :mod:`strata_forge.llm` for convenience:
  :data:`AnyMessage`, :class:`SystemMessage`,
  :class:`UserMessage`, :class:`AssistantMessage`,
  :class:`ToolResultMessage`, :class:`Message`, :class:`Tool`,
  :class:`ToolLoopExceededError`, :func:`tool`.
- Streaming-loop events forwarded from :mod:`strata_forge.llm`:
  :data:`LoopEvent` and its members (:class:`IterationStart`,
  :class:`TextDelta`, :class:`ToolCallStarted`,
  :class:`ToolResult`, :class:`Done`, :class:`LoopError`) —
  yielded by :meth:`Agent.run_streaming`.

Errors raised from this module are :class:`ForgeError` subclasses.
The agent itself raises:

- :class:`ValueError` for invalid construction args (empty name,
  ``max_iterations < 1``).
- Anything :class:`LLMClient` raises propagates verbatim —
  :class:`RegistryError`, :class:`BudgetExceededError`,
  :class:`ToolLoopExceededError`, the various
  :class:`ProviderError` subclasses.

## Internal patterns

- **Composition, not inheritance.** :class:`Agent` does not
  subclass :class:`LLMClient`; it holds one. Adapting an
  existing :class:`LLMClient` workflow into an Agent is one
  constructor call.
- **Tuple-typed tools field.** Same rationale as in
  :mod:`strata_forge.datasets` and :mod:`strata_forge.evals`: structural
  immutability so callers can rely on the agent's tool list
  being stable across runs.
- **Run vs. run_structured.** ``run()`` covers the
  tool-loop / complete dispatch; ``run_structured()`` covers
  the typed-output path. They're separate methods rather than
  one with a polymorphic return type so pyright can infer the
  caller's :attr:`AgentResult.parsed` type via the generic
  ``output_schema``.
- **Intermediate tool calls are not captured in
  AgentResult.messages.** :meth:`LLMClient.run_tool_loop`
  returns only the final response; the iteration trace stays
  inside the method. Callers that need it post-hoc consult
  Langfuse (when :mod:`strata_forge.tracing` is wired in via
  ``install_litellm_callback``) or the
  ``FORGE_DIAGNOSTIC_ENABLED`` NDJSON dump; callers that need it *live*
  use :meth:`Agent.run_streaming`, which surfaces every
  iteration's calls and results as :data:`LoopEvent`s without a
  second loop implementation.

## Test expectations

- Unit tests under ``tests/unit/agents/``, one file per source
  module.
- Coverage: the enforced gate is the repo-wide 85 % line floor
  (``fail_under`` in ``pyproject.toml``); treat a drop in this module
  as a regression.
- Tests use ``AsyncMock`` against the :class:`LLMClient` surface
  — no live network.
- Each built-in tool gets its own test file with the network /
  filesystem seam mocked.
- The agent's behavior with real LiteLLM-backed clients is
  covered by VCR cassettes in :mod:`strata_forge.llm`'s test suite;
  there's nothing agent-specific to record at the seam.

## Gotchas

- **Don't add a new tool abstraction.** If the agent needs a
  feature that :class:`strata_forge.llm.Tool` doesn't have, extend
  :class:`Tool` in :mod:`strata_forge.llm` — don't fork it.
- **Don't capture intermediate tool calls by re-implementing
  the loop.** If full-trace capture is needed, extend
  :meth:`LLMClient.run_tool_loop` (e.g. with an optional
  output-history parameter) and surface that through Agent —
  but keep one loop implementation across the codebase.
- **Never log raw user input at INFO.** Same rule as
  :mod:`strata_forge.llm`: agents see arbitrary user content, much
  of which may be PII. Use DEBUG, or rely on Langfuse / the
  diagnostic dump for full payloads.
- **Memory state lives outside Agent.** A memory instance
  (:class:`ConversationMemory`, :class:`EpisodicMemory`) is passed
  in by the caller and owned by the caller; :class:`Agent` itself
  holds no conversation state across runs.

## When to update this file

- Adding a new public class/function to ``__init__.py``.
- Adding a new built-in tool.
- Changing the agent's dispatch logic (e.g. supporting
  combined tools + structured output).
- Wiring in a new optional extra.
- Changing the message-flow contract with :mod:`strata_forge.llm`.
