"""Single-agent runtime — a thin wrapper over :class:`strata_forge.llm.LLMClient`.

An :class:`Agent` holds a system prompt, a set of tools (instances of
:class:`strata_forge.llm.Tool`), and a reference to an :class:`LLMClient`.
Calling :meth:`Agent.run` builds the full message list (system +
user) and dispatches it through :meth:`LLMClient.run_tool_loop` when
tools are present, or :meth:`LLMClient.complete` otherwise. The
returned :class:`AgentResult` bundles the final text, the conversation
that was sent, and the raw :class:`LLMResponse`.

See [ADR 0011](../../../docs/architecture/adr/0011-agents-thin-wrapper-over-forge-llm.md)
for the no-reimplementation design principle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ConfigDict

from strata_forge.llm.messages import (
    AnyMessage,
    SystemMessage,
    UserMessage,
)
from strata_forge.llm.responses import (
    LLMResponse,  # noqa: TC001 — Pydantic needs runtime resolution
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from strata_forge.llm.client import LLMClient
    from strata_forge.llm.loop_events import LoopEvent
    from strata_forge.llm.registry import ProviderName
    from strata_forge.llm.tools import Tool

__all__ = [
    "Agent",
    "AgentResult",
]


M = TypeVar("M", bound=BaseModel)


class AgentResult(BaseModel):
    """The outcome of an :meth:`Agent.run` invocation.

    Attributes:
        text: The final assistant text — empty when the model exited
            with no textual content (rare, but possible after a tool
            loop that terminated immediately).
        messages: The conversation sent to the LLM (system prompt +
            user input). Intermediate tool-call / tool-result
            messages from inside the tool loop are NOT included
            here — :class:`LLMClient.run_tool_loop` doesn't expose
            them, and capturing them would require re-implementing
            the loop. Callers that need the full trace consult
            ``strata_forge.tracing`` (Langfuse) or the
            ``FORGE_DIAGNOSTIC`` NDJSON dump.
        final_response: The raw final :class:`LLMResponse` from the
            underlying client. Carries usage, cost, latency, the
            resolved provider route, and ``cache_hit`` exactly as
            ``strata_forge.llm`` produced them.
        parsed: When :meth:`Agent.run_structured` produced a typed
            output, the validated Pydantic instance lives here.
            ``None`` for plain :meth:`Agent.run` calls.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    text: str
    messages: tuple[AnyMessage, ...]
    final_response: LLMResponse
    parsed: Any = None

    @property
    def cost_usd(self) -> float:
        """Convenience accessor: final-response cost.

        Note this is the **final completion's** cost only; cost
        across all tool-loop iterations is not aggregated. Use
        Langfuse tracing for cumulative accounting.
        """
        return self.final_response.cost_usd

    @property
    def latency_ms(self) -> float:
        """Convenience accessor: final-response latency."""
        return self.final_response.latency_ms


class Agent:
    """A single-agent runtime built on :class:`LLMClient`.

    Args:
        name: Human-readable identifier. Surfaces in multi-agent
            handoff messages (Phase 3.4) and report output.
        client: The :class:`LLMClient` the agent uses. Models with
            ``tool_calling=False`` in the registry can still be
            wrapped in an :class:`Agent`, but calling :meth:`run`
            with non-empty ``tools`` raises
            :class:`strata_forge.core.errors.RegistryError`.
        system_prompt: Optional system message prepended to every
            run. When ``None``, the agent sends only the user input.
        tools: Tool instances built via :func:`strata_forge.llm.tool`. May
            be empty.
        max_iterations: Tool-loop iteration cap forwarded to
            :meth:`LLMClient.run_tool_loop`. Default 8 — same as
            :meth:`LLMClient.run_tool_loop`.
    """

    def __init__(
        self,
        name: str,
        *,
        client: LLMClient,
        system_prompt: str | None = None,
        tools: Sequence[Tool] = (),
        max_iterations: int = 8,
    ) -> None:
        if not name:
            err = "Agent name must be non-empty"
            raise ValueError(err)
        if max_iterations < 1:
            err = f"max_iterations must be >= 1; got {max_iterations}"
            raise ValueError(err)
        self._name = name
        self._client = client
        self._system_prompt = system_prompt
        self._tools: tuple[Tool, ...] = tuple(tools)
        self._max_iterations = max_iterations

    @property
    def name(self) -> str:
        return self._name

    @property
    def client(self) -> LLMClient:
        return self._client

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    @property
    def tools(self) -> tuple[Tool, ...]:
        return self._tools

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def _build_messages(self, user_input: str | Sequence[AnyMessage]) -> list[AnyMessage]:
        messages: list[AnyMessage] = []
        if self._system_prompt is not None:
            messages.append(SystemMessage(content=self._system_prompt))
        if isinstance(user_input, str):
            messages.append(UserMessage(content=user_input))
        else:
            messages.extend(user_input)
        return messages

    async def run(
        self,
        user_input: str | Sequence[AnyMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        provider_extras: Mapping[ProviderName, Mapping[str, Any]] | None = None,
    ) -> AgentResult:
        """Run the agent over ``user_input`` and return the result.

        When the agent has tools, dispatches through
        :meth:`LLMClient.run_tool_loop`; otherwise calls
        :meth:`LLMClient.complete` once.

        Args:
            user_input: A user-message string (wrapped into
                :class:`UserMessage`) or a sequence of
                :class:`AnyMessage` to pass through verbatim. The
                agent's system prompt is prepended either way.
            temperature, max_tokens, top_p: Forwarded to the
                underlying call.
            provider_extras: Forwarded to the underlying call.

        Returns:
            An :class:`AgentResult` with the final text, the input
            conversation, and the raw :class:`LLMResponse`.
        """
        messages = self._build_messages(user_input)
        if self._tools:
            response = await self._client.run_tool_loop(
                messages,
                tools=self._tools,
                max_iterations=self._max_iterations,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                provider_extras=provider_extras,
            )
        else:
            response = await self._client.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                provider_extras=provider_extras,
            )
        return AgentResult(
            text=response.text,
            messages=tuple(messages),
            final_response=response,
        )

    async def run_streaming(
        self,
        user_input: str | Sequence[AnyMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        provider_extras: Mapping[ProviderName, Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[LoopEvent]:
        """Stream the agent's tool-use loop as typed events.

        Builds the message list (system prompt + ``user_input``) like
        :meth:`run`, then delegates to
        :meth:`LLMClient.stream_tool_loop` with the agent's tools and
        ``max_iterations``, yielding each
        :data:`~strata_forge.llm.loop_events.LoopEvent` as it arrives. When the
        agent has NO tools, the loop degenerates to a single streamed turn
        (one ``IterationStart``, its ``TextDelta``s, and a terminal
        ``Done``).

        This is a true async generator — iterate it directly
        (``async for event in agent.run_streaming(...)``). There is no
        streaming variant of :meth:`run_structured`.
        """
        messages = self._build_messages(user_input)
        async for event in self._client.stream_tool_loop(
            messages,
            tools=self._tools,
            max_iterations=self._max_iterations,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            provider_extras=provider_extras,
        ):
            yield event

    async def run_structured(
        self,
        user_input: str | Sequence[AnyMessage],
        *,
        output_schema: type[M],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        max_reprompt_attempts: int = 3,
        provider_extras: Mapping[ProviderName, Mapping[str, Any]] | None = None,
    ) -> AgentResult:
        """Run the agent and require a typed output.

        Calls :meth:`LLMClient.complete_structured` against the
        agent's system prompt + user input. Tools are NOT applied
        in this path — the structured-output dispatch in
        :mod:`strata_forge.llm` already uses forced-tool mode internally
        for providers that need it, and combining application tools
        with that machinery has too many cross-product edge cases
        for the foundation sub-phase. Run :meth:`Agent.run` first
        when the agent needs to call tools before producing a
        structured answer.

        Args:
            user_input: Same shape as :meth:`run`.
            output_schema: The Pydantic schema the response must
                validate against. The validated instance lives at
                :attr:`AgentResult.parsed`.
            temperature, max_tokens, top_p: Forwarded.
            max_reprompt_attempts: Forwarded to
                :meth:`LLMClient.complete_structured`.
            provider_extras: Forwarded.

        Returns:
            An :class:`AgentResult` whose ``parsed`` carries the
            validated Pydantic instance.
        """
        messages = self._build_messages(user_input)
        response = await self._client.complete_structured(
            messages=messages,
            schema=output_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            max_reprompt_attempts=max_reprompt_attempts,
            provider_extras=provider_extras,
        )
        return AgentResult(
            text=response.text,
            messages=tuple(messages),
            final_response=response,
            parsed=response.parsed,
        )
