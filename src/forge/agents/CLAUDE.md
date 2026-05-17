# Agent rules — forge.agents

`forge.agents` is a thin composition layer over :mod:`forge.llm`.
The agent runtime never re-implements tool calling, structured
output, the tool loop, or message handling — every one of those
primitives comes from :mod:`forge.llm` directly. See
[ADR 0011](../../../docs/architecture/adr/0011-agents-thin-wrapper-over-forge-llm.md)
for the rationale.

## Purpose

- :class:`Agent` — a single-agent runtime that bundles an
  :class:`LLMClient`, a system prompt, and a set of tools.
  ``.run(user_input)`` dispatches through
  :meth:`LLMClient.run_tool_loop` (when tools are present) or
  :meth:`LLMClient.complete` (when not).
- :class:`AgentResult` — the output shape: text + input messages
  + raw :class:`LLMResponse` (so callers see Forge-canonical
  accounting fields, not an agent-specific wrapper).
- Built-in tools (Phase 3.2): ``web_search``, ``fetch_url``,
  ``fs_read``, ``calculator``. Each is a concrete :class:`Tool`
  built via :func:`forge.llm.tool`.
- Memory (Phase 3.3): :class:`ConversationMemory` for in-process
  message-history trimming; :class:`EpisodicMemory` against a
  pluggable vector store (concrete adapter lands when
  :mod:`forge.rag` ships in Phase 4).
- Multi-agent patterns (Phase 3.4): hand-off, critic-refiner.

## Boundaries

- **Owns:** `agent.py`, `tools/` (Phase 3.2), `memory/`
  (Phase 3.3), multi-agent helpers (Phase 3.4).
- **Imports from inside `forge`:** :mod:`forge.core` (errors),
  :mod:`forge.config` (settings if built-in tools need them),
  :mod:`forge.llm` (everything tool/message/response-related),
  optionally :mod:`forge.prompts` (when rendering structured
  prompts as system messages), and :mod:`forge.rag` when its
  vector-store Protocol lands.
- **Does NOT import** :mod:`forge.tracing` (that module wraps
  every Forge module from above; the dependency arrow points
  one way — see ADR 0008).
- **Does NOT re-implement** any primitive that lives in
  :mod:`forge.llm`. The agent's tools field is
  ``tuple[forge.llm.Tool, ...]`` — no new tool abstraction. Tool
  invocation goes through :meth:`Tool.invoke`. The multi-turn
  loop goes through :meth:`LLMClient.run_tool_loop`. Structured
  output goes through :meth:`LLMClient.complete_structured`.
- **External deps:** Pydantic. No optional extras at the
  foundation sub-phase; built-in tools (Phase 3.2) may add
  lazy imports.

## Public API

The module's ``__init__.py`` re-exports:

- Agent runtime: :class:`Agent`, :class:`AgentResult`.
- Message types and tool primitives forwarded from
  :mod:`forge.llm` for convenience:
  :data:`AnyMessage`, :class:`SystemMessage`,
  :class:`UserMessage`, :class:`AssistantMessage`,
  :class:`ToolResultMessage`, :class:`Message`, :class:`Tool`,
  :class:`ToolLoopExceededError`, :func:`tool`.

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
  :mod:`forge.datasets` and :mod:`forge.evals`: structural
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
  inside the method. Callers that need it consult Langfuse
  (when :mod:`forge.tracing` is wired in via
  ``install_litellm_callback``) or the
  ``FORGE_DIAGNOSTIC`` NDJSON dump.

## Test expectations

- Unit tests under ``tests/unit/agents/``, one file per source
  module.
- Coverage target: ≥ 90 % line.
- Tests use ``AsyncMock`` against the :class:`LLMClient` surface
  — no live network.
- Built-in tools (Phase 3.2) get one mocked-SDK test each.
- The agent's behavior with real LiteLLM-backed clients is
  covered by VCR cassettes in :mod:`forge.llm`'s test suite;
  there's nothing agent-specific to record at the seam.

## Gotchas

- **Don't add a new tool abstraction.** If the agent needs a
  feature that :class:`forge.llm.Tool` doesn't have, extend
  :class:`Tool` in :mod:`forge.llm` — don't fork it.
- **Don't capture intermediate tool calls by re-implementing
  the loop.** If full-trace capture is needed, extend
  :meth:`LLMClient.run_tool_loop` (e.g. with an optional
  output-history parameter) and surface that through Agent —
  but keep one loop implementation across the codebase.
- **Never log raw user input at INFO.** Same rule as
  :mod:`forge.llm`: agents see arbitrary user content, much
  of which may be PII. Use DEBUG, or rely on Langfuse / the
  diagnostic dump for full payloads.
- **Memory state lives outside Agent.** When Phase 3.3 ships
  :class:`ConversationMemory`, agents take a memory instance as
  a constructor argument; the agent itself stays stateless
  across runs.

## When to update this file

- Adding a new public class/function to ``__init__.py``.
- Adding a new built-in tool.
- Changing the agent's dispatch logic (e.g. supporting
  combined tools + structured output).
- Wiring in a new optional extra.
- Changing the message-flow contract with :mod:`forge.llm`.
