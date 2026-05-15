"""Unit tests for `forge.llm.tools`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from forge.core.errors import ForgeError, ValidationError
from forge.llm.tools import Tool, ToolLoopExceededError, tool


class _WeatherArgs(BaseModel):
    location: str = Field(..., description="City, country")
    units: str = "celsius"


class _EmptyArgs(BaseModel):
    pass


class TestToolLoopExceededError:
    def test_carries_counters(self) -> None:
        err = ToolLoopExceededError(
            "loop went too long",
            max_iterations=8,
            iterations=8,
        )
        assert err.max_iterations == 8
        assert err.iterations == 8
        assert str(err) == "loop went too long"

    def test_is_forge_error(self) -> None:
        err = ToolLoopExceededError("x", max_iterations=1, iterations=1)
        assert isinstance(err, ForgeError)


class TestToolConstruction:
    def test_construct_directly(self) -> None:
        async def _fn(args: _WeatherArgs) -> str:
            return f"{args.location}: sunny"

        t = Tool(
            name="get_weather",
            description="Get current weather.",
            parameters_model=_WeatherArgs,
            fn=_fn,
        )
        assert t.name == "get_weather"
        assert t.description == "Get current weather."
        assert t.parameters_model is _WeatherArgs

    def test_is_frozen(self) -> None:
        async def _fn(args: _EmptyArgs) -> None:
            return None

        t = Tool(name="x", description="d", parameters_model=_EmptyArgs, fn=_fn)
        with pytest.raises((AttributeError, TypeError)):
            t.name = "y"  # type: ignore[misc]

    def test_parameters_schema_matches_pydantic(self) -> None:
        async def _fn(args: _WeatherArgs) -> None:
            return None

        t = Tool(name="x", description="d", parameters_model=_WeatherArgs, fn=_fn)
        assert t.parameters_schema() == _WeatherArgs.model_json_schema()


class TestToolSchemaDelegation:
    """`Tool.to_*_schema` must produce the same output as calling the
    underlying provider serializer with `name`, `description`, and the
    Pydantic-derived JSON schema."""

    def _make(self) -> Tool:
        async def _fn(args: _WeatherArgs) -> None:
            return None

        return Tool(
            name="get_weather",
            description="Get current weather for a city.",
            parameters_model=_WeatherArgs,
            fn=_fn,
        )

    def test_openai_schema(self) -> None:
        t = self._make()
        schema = t.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert schema["function"]["description"] == "Get current weather for a city."
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "location" in params["properties"]

    def test_anthropic_schema(self) -> None:
        t = self._make()
        schema = t.to_anthropic_schema()
        # Anthropic has a flatter shape — no outer "type": "function" wrapper.
        assert "function" not in schema
        assert schema["name"] == "get_weather"
        assert schema["description"] == "Get current weather for a city."
        assert schema["input_schema"]["type"] == "object"

    def test_gemini_schema(self) -> None:
        t = self._make()
        schema = t.to_gemini_schema()
        # Gemini uses `parameters` like OpenAI but no outer wrapper.
        assert "function" not in schema
        assert "input_schema" not in schema
        assert schema["name"] == "get_weather"
        assert schema["description"] == "Get current weather for a city."
        assert schema["parameters"]["type"] == "object"


class TestToolInvoke:
    async def test_invokes_with_validated_args(self) -> None:
        captured: list[_WeatherArgs] = []

        async def _fn(args: _WeatherArgs) -> str:
            captured.append(args)
            return f"weather in {args.location}: 18°{args.units}"

        t = Tool(name="get_weather", description="d", parameters_model=_WeatherArgs, fn=_fn)
        result = await t.invoke({"location": "Tokyo", "units": "celsius"})
        assert result == "weather in Tokyo: 18°celsius"
        assert len(captured) == 1
        assert isinstance(captured[0], _WeatherArgs)
        assert captured[0].location == "Tokyo"

    async def test_defaults_filled_in(self) -> None:
        async def _fn(args: _WeatherArgs) -> str:
            return args.units

        t = Tool(name="x", description="d", parameters_model=_WeatherArgs, fn=_fn)
        result = await t.invoke({"location": "Paris"})
        assert result == "celsius"

    async def test_invalid_args_raise_validation_error(self) -> None:
        async def _fn(args: _WeatherArgs) -> None:
            return None

        t = Tool(name="get_weather", description="d", parameters_model=_WeatherArgs, fn=_fn)
        with pytest.raises(ValidationError, match="get_weather"):
            await t.invoke({"units": "celsius"})  # missing required "location"

    async def test_validation_error_preserves_pydantic_cause(self) -> None:
        async def _fn(args: _WeatherArgs) -> None:
            return None

        t = Tool(name="x", description="d", parameters_model=_WeatherArgs, fn=_fn)
        with pytest.raises(ValidationError) as exc_info:
            await t.invoke({})
        assert exc_info.value.__cause__ is not None

    async def test_extra_args_with_default_model_are_ignored(self) -> None:
        # Pydantic by default ignores extras; explicit forbid would change this.
        async def _fn(args: _WeatherArgs) -> str:
            return args.location

        t = Tool(name="x", description="d", parameters_model=_WeatherArgs, fn=_fn)
        result = await t.invoke({"location": "Berlin", "extra": "ignored"})
        assert result == "Berlin"

    async def test_function_returning_dict(self) -> None:
        async def _fn(args: _WeatherArgs) -> dict[str, Any]:
            return {"temp_c": 18, "city": args.location}

        t = Tool(name="x", description="d", parameters_model=_WeatherArgs, fn=_fn)
        result = await t.invoke({"location": "Berlin"})
        assert result == {"temp_c": 18, "city": "Berlin"}


class TestToolDecoratorBareForm:
    """`@tool` applied directly to a function."""

    def test_returns_tool_instance(self) -> None:
        @tool
        async def get_weather(args: _WeatherArgs) -> str:
            """Get current weather for a city."""
            return args.location

        assert isinstance(get_weather, Tool)

    def test_name_from_function(self) -> None:
        @tool
        async def get_weather(args: _WeatherArgs) -> str:
            """Get current weather."""
            return args.location

        assert get_weather.name == "get_weather"

    def test_description_from_docstring(self) -> None:
        @tool
        async def get_weather(args: _WeatherArgs) -> str:
            """Get current weather for a city."""
            return args.location

        assert get_weather.description == "Get current weather for a city."

    def test_description_strips_whitespace(self) -> None:
        @tool
        async def get_weather(args: _WeatherArgs) -> str:
            """
            Get current weather.
            """
            return args.location

        # Leading/trailing whitespace gone, but inner formatting kept.
        assert get_weather.description.startswith("Get current weather.")
        assert not get_weather.description.startswith(" ")
        assert not get_weather.description.endswith(" ")

    def test_description_empty_when_no_docstring(self) -> None:
        @tool
        async def get_weather(args: _WeatherArgs) -> str:
            return args.location

        assert get_weather.description == ""

    def test_parameters_model_extracted_from_annotation(self) -> None:
        @tool
        async def get_weather(args: _WeatherArgs) -> str:
            """d"""
            return args.location

        assert get_weather.parameters_model is _WeatherArgs

    async def test_invoke_runs_underlying_function(self) -> None:
        @tool
        async def get_weather(args: _WeatherArgs) -> str:
            """d"""
            return args.location.upper()

        result = await get_weather.invoke({"location": "tokyo"})
        assert result == "TOKYO"


class TestToolDecoratorFactoryForm:
    """`@tool(name=..., description=...)` for explicit overrides."""

    def test_name_override(self) -> None:
        @tool(name="weather")
        async def get_weather(args: _WeatherArgs) -> str:
            """Default desc."""
            return args.location

        assert get_weather.name == "weather"
        # Description still comes from docstring when not overridden.
        assert get_weather.description == "Default desc."

    def test_description_override(self) -> None:
        @tool(description="Custom description.")
        async def get_weather(args: _WeatherArgs) -> str:
            """Ignored docstring."""
            return args.location

        assert get_weather.name == "get_weather"
        assert get_weather.description == "Custom description."

    def test_both_overrides(self) -> None:
        @tool(name="w", description="d")
        async def get_weather(args: _WeatherArgs) -> str:
            return args.location

        assert get_weather.name == "w"
        assert get_weather.description == "d"

    def test_no_kwargs_works_like_bare(self) -> None:
        # @tool() with no kwargs is a degenerate factory form.
        @tool()
        async def get_weather(args: _WeatherArgs) -> str:
            """Doc."""
            return args.location

        assert isinstance(get_weather, Tool)
        assert get_weather.name == "get_weather"
        assert get_weather.description == "Doc."


class TestToolDecoratorValidation:
    """Each test below intentionally passes a malformed function to
    :func:`tool`. We invoke ``tool`` directly (rather than as a decorator)
    so pyright sees a normal call site we can ignore narrowly, and we
    cast the function through ``Any`` since the whole point is that its
    signature does not match :data:`ToolFunc`.
    """

    def test_function_with_zero_args_rejected(self) -> None:
        async def no_args() -> None:
            return None

        with pytest.raises(TypeError, match="exactly one argument"):
            tool(no_args)  # pyright: ignore[reportCallIssue, reportArgumentType]

    def test_function_with_two_args_rejected(self) -> None:
        async def too_many(a: _WeatherArgs, b: _WeatherArgs) -> None:
            return None

        with pytest.raises(TypeError, match="exactly one argument"):
            tool(too_many)  # pyright: ignore[reportCallIssue, reportArgumentType]

    def test_function_without_annotation_rejected(self) -> None:
        async def no_annotation(args) -> None:  # type: ignore[no-untyped-def]
            return None

        with pytest.raises(TypeError, match="must have a Pydantic BaseModel annotation"):
            tool(no_annotation)  # pyright: ignore[reportUnknownArgumentType]

    def test_function_with_non_basemodel_annotation_rejected(self) -> None:
        async def wrong_type(args: dict[str, Any]) -> None:
            return None

        with pytest.raises(TypeError, match="must be a Pydantic BaseModel subclass"):
            tool(wrong_type)  # pyright: ignore[reportCallIssue, reportArgumentType]

    def test_function_with_primitive_annotation_rejected(self) -> None:
        async def wrong_type(args: str) -> None:
            return None

        with pytest.raises(TypeError, match="must be a Pydantic BaseModel subclass"):
            tool(wrong_type)  # pyright: ignore[reportCallIssue, reportArgumentType]
