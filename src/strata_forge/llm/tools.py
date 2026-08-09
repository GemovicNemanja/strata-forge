"""Tool calling primitives.

A :class:`Tool` wraps an async Python function with a Pydantic args model
and the human-readable name + description the LLM needs in order to call
it. The :func:`tool` decorator is the usual way to construct one.

Schema serialization for each provider lives in
``strata_forge.llm.providers.{openai,anthropic,vertex}``; :class:`Tool` delegates
to them so callers don't need to import the right module for each
provider. The schema converters are also re-exported here for callers who
already have a Pydantic model and just want the wire format.

:exc:`ToolLoopExceededError` lives here too; it's raised by the
``LLMClient.run_tool_loop`` helper when the maximum number of tool-use
iterations is exceeded.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_type_hints, overload

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from strata_forge.core.errors import ForgeError, ValidationError
from strata_forge.llm.providers.anthropic import to_anthropic_tool_schema
from strata_forge.llm.providers.openai import to_openai_tool_schema
from strata_forge.llm.providers.vertex import to_gemini_tool_schema

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = [
    "AnyTool",
    "Tool",
    "ToolDeclaration",
    "ToolFunc",
    "ToolLoopExceededError",
    "to_anthropic_tool_schema",
    "to_gemini_tool_schema",
    "to_openai_tool_schema",
    "tool",
]


# Note: `Any` here is intentional — the actual Pydantic args model is per-tool
# and tracked at runtime via :attr:`Tool.parameters_model`. Static type safety
# is preserved for tool authors via the generic typevar in the :func:`tool`
# decorator overloads below.
type ToolFunc = Callable[[Any], Awaitable[Any]]


class ToolLoopExceededError(ForgeError):
    """Raised when a tool-use loop exceeds its maximum iteration count.

    The error carries ``max_iterations`` and ``iterations`` (the actual
    count reached) so callers can decide whether to retry with a higher
    limit or surface a "model went into a tool loop" error to the user.
    """

    def __init__(
        self,
        message: str,
        *,
        max_iterations: int,
        iterations: int,
    ) -> None:
        super().__init__(message)
        self.max_iterations = max_iterations
        self.iterations = iterations


@dataclass(frozen=True, slots=True)
class Tool:
    """An async tool that the LLM can invoke.

    Construct via :func:`tool` rather than instantiating directly; the
    decorator inspects the wrapped function's signature to pull the
    Pydantic arguments model out automatically.

    Attributes:
        name: The tool's identifier, used in the LLM's tool-call schema.
        description: Human-readable description shown to the model.
        parameters_model: Pydantic ``BaseModel`` subclass describing the
            shape of the tool's arguments. ``invoke()`` validates the raw
            dict against this model before calling the wrapped function.
        fn: The async function to invoke. Receives a validated instance
            of ``parameters_model``; can return anything.
    """

    name: str
    description: str
    parameters_model: type[BaseModel]
    fn: ToolFunc

    def parameters_schema(self) -> dict[str, Any]:
        """Return the Pydantic-derived JSON Schema for this tool's args."""
        return self.parameters_model.model_json_schema()

    def to_openai_schema(self) -> dict[str, Any]:
        """OpenAI-format tool descriptor."""
        return to_openai_tool_schema(self.name, self.description, self.parameters_schema())

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Anthropic-format tool descriptor."""
        return to_anthropic_tool_schema(self.name, self.description, self.parameters_schema())

    def to_gemini_schema(self) -> dict[str, Any]:
        """Gemini-format tool (function-declaration) descriptor."""
        return to_gemini_tool_schema(self.name, self.description, self.parameters_schema())

    async def invoke(self, args: dict[str, Any]) -> Any:
        """Validate ``args`` against the parameter model and call the function.

        Raises:
            ValidationError: When ``args`` doesn't satisfy the Pydantic
                model's constraints. The Pydantic error chain is preserved
                via ``__cause__``.
        """
        try:
            validated = self.parameters_model.model_validate(args)
        except PydanticValidationError as exc:
            msg = f"Tool {self.name!r} received invalid arguments: {exc}"
            raise ValidationError(msg) from exc
        return await self.fn(validated)


@dataclass(frozen=True, slots=True)
class ToolDeclaration:
    """A tool the model may call but forge cannot execute.

    A declaration carries everything the provider needs to offer the tool
    to the model — name, description, and a raw JSON-Schema ``parameters``
    object — but no function. It exists for callers whose tools execute
    out-of-band (e.g. a server streaming to a browser that performs the
    action): in ``LLMClient.stream_tool_loop``, a call targeting a
    declaration suspends the run with a terminal
    :class:`~strata_forge.llm.loop_events.PendingToolCalls` event instead of
    being invoked; the caller executes it elsewhere and resumes the loop
    with the grown conversation.

    ``parameters`` is passed through to the provider verbatim — forge does
    not validate JSON-Schema semantics (a malformed schema surfaces as a
    provider error at call time).
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def parameters_schema(self) -> dict[str, Any]:
        """Return the raw JSON Schema for this tool's args (verbatim)."""
        return self.parameters

    def to_openai_schema(self) -> dict[str, Any]:
        """OpenAI-format tool descriptor."""
        return to_openai_tool_schema(self.name, self.description, self.parameters_schema())

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Anthropic-format tool descriptor."""
        return to_anthropic_tool_schema(self.name, self.description, self.parameters_schema())

    def to_gemini_schema(self) -> dict[str, Any]:
        """Gemini-format tool (function-declaration) descriptor."""
        return to_gemini_tool_schema(self.name, self.description, self.parameters_schema())


# Anything acceptable in a `tools=` sequence: executable tools and
# declaration-only tools share the schema-serialization surface, so provider
# dispatch treats them uniformly; only the tool loop distinguishes them.
type AnyTool = Tool | ToolDeclaration


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


@overload
def tool[M: BaseModel](fn: Callable[[M], Awaitable[Any]], /) -> Tool: ...


@overload
def tool(
    fn: None = None,
    /,
    *,
    name: str | None = ...,
    description: str | None = ...,
) -> Callable[[ToolFunc], Tool]: ...


def tool(
    fn: ToolFunc | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[ToolFunc], Tool]:
    """Wrap an async function into a :class:`Tool`.

    The wrapped function must take **exactly one** argument typed as a
    Pydantic ``BaseModel`` subclass. The model becomes the tool's
    parameter schema; the function's docstring (or ``description``
    override) becomes the tool description; the function's name (or
    ``name`` override) becomes the tool name.

    Usage::

        class WeatherArgs(BaseModel):
            location: str
            units: str = "celsius"

        @tool
        async def get_weather(args: WeatherArgs) -> dict[str, Any]:
            \"\"\"Get the current weather for a location.\"\"\"
            ...

        @tool(name="lookup_time", description="Get the current time in a tz.")
        async def time_in(args: TimezoneArgs) -> str:
            ...
    """

    def _wrap(func: ToolFunc) -> Tool:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if len(params) != 1:
            msg = (
                f"@tool function {func.__name__!r} must take exactly one argument; "
                f"got {len(params)}"
            )
            raise TypeError(msg)
        param = params[0]
        # Resolve string annotations (PEP 563 / `from __future__ import annotations`).
        # `get_type_hints` evaluates them in the function's own globals so we get
        # real classes back rather than strings.
        try:
            hints = get_type_hints(func, include_extras=True)
        except (NameError, TypeError) as exc:  # forward ref to a name not in scope
            msg = (
                f"@tool function {func.__name__!r} parameter {param.name!r} "
                f"has an unresolvable annotation: {exc}"
            )
            raise TypeError(msg) from exc
        annotation = hints.get(param.name, inspect.Parameter.empty)
        if annotation is inspect.Parameter.empty:
            msg = (
                f"@tool function {func.__name__!r} parameter {param.name!r} "
                "must have a Pydantic BaseModel annotation"
            )
            raise TypeError(msg)
        if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
            msg = (
                f"@tool function {func.__name__!r} parameter {param.name!r} "
                f"must be a Pydantic BaseModel subclass, got {annotation!r}"
            )
            raise TypeError(msg)

        tool_name = name if name is not None else func.__name__
        if description is not None:
            tool_description = description
        else:
            doc = func.__doc__
            tool_description = doc.strip() if doc else ""

        return Tool(
            name=tool_name,
            description=tool_description,
            parameters_model=annotation,
            fn=func,
        )

    if fn is not None:
        return _wrap(fn)
    return _wrap
