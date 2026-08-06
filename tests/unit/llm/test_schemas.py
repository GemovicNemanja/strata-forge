"""Unit tests for `strata_forge.llm.schemas`."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from pydantic import BaseModel, Field

from strata_forge.core.errors import ForgeError, ValidationError
from strata_forge.llm.schemas import (
    StructuredOutputError,
    make_reprompt_instruction,
    parse_json_response,
    pydantic_to_json_schema,
    to_anthropic_forced_tool_schema,
    to_gemini_response_schema,
    to_openai_response_format,
)

# ---------------------------------------------------------------------------
# Sample models
# ---------------------------------------------------------------------------


class _Address(BaseModel):
    city: str
    country: str


class _Person(BaseModel):
    """A person with a nested address."""

    name: str = Field(..., description="Full name")
    age: int = Field(..., ge=0, description="Age in years")
    address: _Address


class _Summary(BaseModel):
    title: str
    bullets: list[str]


class _Empty(BaseModel):
    pass


# ---------------------------------------------------------------------------
# StructuredOutputError
# ---------------------------------------------------------------------------


class TestStructuredOutputError:
    def test_is_forge_error(self) -> None:
        err = StructuredOutputError("failed", attempts=3)
        assert isinstance(err, ForgeError)

    def test_carries_attempts(self) -> None:
        err = StructuredOutputError("x", attempts=3)
        assert err.attempts == 3
        assert err.last_error is None

    def test_carries_last_error(self) -> None:
        cause = ValueError("bad json")
        err = StructuredOutputError("x", attempts=3, last_error=cause)
        assert err.last_error is cause


# ---------------------------------------------------------------------------
# pydantic_to_json_schema
# ---------------------------------------------------------------------------


class TestPydanticToJsonSchema:
    def test_simple_model(self) -> None:
        schema = pydantic_to_json_schema(_Summary)
        assert schema["type"] == "object"
        assert "title" in schema["properties"]
        assert "bullets" in schema["properties"]
        assert schema["properties"]["bullets"]["type"] == "array"

    def test_inlines_refs_by_default(self) -> None:
        schema = pydantic_to_json_schema(_Person)
        # `$defs` should be stripped after inlining.
        assert "$defs" not in schema
        address_schema = schema["properties"]["address"]
        # The reference should have been replaced with the resolved object.
        assert "$ref" not in address_schema
        assert address_schema["type"] == "object"
        assert "city" in address_schema["properties"]

    def test_inline_refs_off_keeps_defs(self) -> None:
        schema = pydantic_to_json_schema(_Person, inline_refs=False)
        assert "$defs" in schema
        # Reference preserved.
        addr_node = schema["properties"]["address"]
        assert "$ref" in addr_node

    def test_returns_a_new_dict(self) -> None:
        s1 = pydantic_to_json_schema(_Summary)
        s2 = pydantic_to_json_schema(_Summary)
        s1["mutated"] = True
        assert "mutated" not in s2

    def test_empty_model(self) -> None:
        schema = pydantic_to_json_schema(_Empty)
        assert schema["type"] == "object"
        # No properties — but the schema is still well-formed.
        assert schema.get("properties", {}) == {}


# ---------------------------------------------------------------------------
# OpenAI response_format
# ---------------------------------------------------------------------------


class TestOpenAIResponseFormat:
    def test_basic_shape(self) -> None:
        fmt = to_openai_response_format(_Summary)
        assert fmt["type"] == "json_schema"
        js = fmt["json_schema"]
        assert js["name"] == "_Summary"
        assert js["strict"] is True
        assert js["schema"]["type"] == "object"

    def test_name_override(self) -> None:
        fmt = to_openai_response_format(_Summary, name="custom")
        assert fmt["json_schema"]["name"] == "custom"

    def test_strict_mode_sets_additional_properties_false(self) -> None:
        fmt = to_openai_response_format(_Summary)
        schema = fmt["json_schema"]["schema"]
        assert schema["additionalProperties"] is False

    def test_strict_mode_makes_all_properties_required(self) -> None:
        fmt = to_openai_response_format(_Summary)
        schema = fmt["json_schema"]["schema"]
        assert set(schema["required"]) == {"title", "bullets"}

    def test_strict_mode_recurses_into_nested_objects(self) -> None:
        fmt = to_openai_response_format(_Person)
        schema = fmt["json_schema"]["schema"]
        # Top-level
        assert schema["additionalProperties"] is False
        # Nested address — after inlining, should also be locked down.
        addr = schema["properties"]["address"]
        assert addr["type"] == "object"
        assert addr["additionalProperties"] is False
        assert set(addr["required"]) == {"city", "country"}

    def test_non_strict_mode_no_additional_properties_added(self) -> None:
        fmt = to_openai_response_format(_Summary, strict=False)
        schema = fmt["json_schema"]["schema"]
        assert "additionalProperties" not in schema
        assert fmt["json_schema"]["strict"] is False


# ---------------------------------------------------------------------------
# Anthropic forced tool
# ---------------------------------------------------------------------------


class TestAnthropicForcedToolSchema:
    def test_returns_tool_and_choice(self) -> None:
        tool, choice = to_anthropic_forced_tool_schema(_Summary)
        assert tool["name"] == "_Summary"
        assert tool["input_schema"]["type"] == "object"
        # OpenAI-spec tool_choice; LiteLLM translates to Anthropic on dispatch.
        assert choice == {"type": "function", "function": {"name": "_Summary"}}

    def test_default_description_present(self) -> None:
        tool, _ = to_anthropic_forced_tool_schema(_Summary)
        assert "JSON" in tool["description"] or "json" in tool["description"]

    def test_custom_name(self) -> None:
        tool, choice = to_anthropic_forced_tool_schema(_Summary, name="emit_summary")
        assert tool["name"] == "emit_summary"
        assert choice["function"]["name"] == "emit_summary"

    def test_custom_description(self) -> None:
        tool, _ = to_anthropic_forced_tool_schema(_Summary, description="my desc")
        assert tool["description"] == "my desc"

    def test_input_schema_inlined(self) -> None:
        tool, _ = to_anthropic_forced_tool_schema(_Person)
        # $defs should be inlined by default.
        assert "$defs" not in tool["input_schema"]
        addr = tool["input_schema"]["properties"]["address"]
        assert addr["type"] == "object"


# ---------------------------------------------------------------------------
# Gemini response_schema
# ---------------------------------------------------------------------------


class TestGeminiResponseSchema:
    def test_basic_shape(self) -> None:
        schema = to_gemini_response_schema(_Summary)
        assert schema["type"] == "object"

    def test_strips_additional_properties(self) -> None:
        # Gemini doesn't recognize additionalProperties; we strip it everywhere.
        schema = to_gemini_response_schema(_Person)

        def _has_key(obj: Any, key: str) -> bool:
            if isinstance(obj, dict):
                d = cast("dict[str, Any]", obj)
                if key in d:
                    return True
                return any(_has_key(v, key) for v in d.values())
            if isinstance(obj, list):
                items = cast("list[Any]", obj)
                return any(_has_key(x, key) for x in items)
            return False

        assert not _has_key(schema, "additionalProperties")

    def test_strips_defs_and_meta_schema(self) -> None:
        schema = to_gemini_response_schema(_Person)
        assert "$defs" not in schema
        assert "$schema" not in schema

    def test_nested_models_inlined(self) -> None:
        schema = to_gemini_response_schema(_Person)
        addr = schema["properties"]["address"]
        assert addr["type"] == "object"
        assert "city" in addr["properties"]


# ---------------------------------------------------------------------------
# Reprompt instruction
# ---------------------------------------------------------------------------


class TestMakeRepromptInstruction:
    def test_includes_schema(self) -> None:
        text = make_reprompt_instruction(_Summary)
        # The schema should appear as readable JSON in the instruction.
        assert "title" in text
        assert "bullets" in text

    def test_instructs_for_pure_json(self) -> None:
        text = make_reprompt_instruction(_Summary)
        assert "ONLY" in text
        assert "JSON" in text

    def test_includes_error_when_provided(self) -> None:
        text = make_reprompt_instruction(_Summary, parse_error="missing 'title' field")
        assert "missing 'title' field" in text
        # Should indicate this is a correction round.
        assert "previous" in text.lower() or "invalid" in text.lower()

    def test_omits_error_section_when_first_attempt(self) -> None:
        text = make_reprompt_instruction(_Summary)
        assert "previous" not in text.lower()


# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_parses_bare_object(self) -> None:
        text = '{"title": "Hi", "bullets": ["a", "b"]}'
        result = parse_json_response(text, _Summary)
        assert isinstance(result, _Summary)
        assert result.title == "Hi"
        assert result.bullets == ["a", "b"]

    def test_strips_whitespace(self) -> None:
        text = '   \n  {"title": "Hi", "bullets": []}  \n  '
        result = parse_json_response(text, _Summary)
        assert result.title == "Hi"

    def test_strips_markdown_fence_with_lang(self) -> None:
        text = '```json\n{"title": "Hi", "bullets": []}\n```'
        result = parse_json_response(text, _Summary)
        assert result.title == "Hi"

    def test_strips_markdown_fence_without_lang(self) -> None:
        text = '```\n{"title": "Hi", "bullets": []}\n```'
        result = parse_json_response(text, _Summary)
        assert result.title == "Hi"

    def test_parses_nested_model(self) -> None:
        data = {
            "name": "Alice",
            "age": 30,
            "address": {"city": "Berlin", "country": "DE"},
        }
        result = parse_json_response(json.dumps(data), _Person)
        assert result.name == "Alice"
        assert result.address.city == "Berlin"

    def test_invalid_json_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="not valid JSON"):
            parse_json_response("{not json", _Summary)

    def test_invalid_json_chains_decode_error(self) -> None:
        with pytest.raises(ValidationError) as info:
            parse_json_response("{not json", _Summary)
        assert isinstance(info.value.__cause__, json.JSONDecodeError)

    def test_schema_mismatch_raises_validation_error(self) -> None:
        # Missing required field "bullets".
        text = '{"title": "Hi"}'
        with pytest.raises(ValidationError, match="does not match"):
            parse_json_response(text, _Summary)

    def test_schema_mismatch_mentions_model_name(self) -> None:
        text = '{"title": "Hi"}'
        with pytest.raises(ValidationError, match="_Summary"):
            parse_json_response(text, _Summary)

    def test_schema_mismatch_chains_pydantic_error(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        text = '{"title": "Hi"}'
        with pytest.raises(ValidationError) as info:
            parse_json_response(text, _Summary)
        assert isinstance(info.value.__cause__, PydanticValidationError)

    def test_negative_age_rejected(self) -> None:
        # Pydantic field constraint (ge=0) should be enforced.
        data = {"name": "Alice", "age": -1, "address": {"city": "x", "country": "y"}}
        with pytest.raises(ValidationError):
            parse_json_response(json.dumps(data), _Person)

    def test_unfenced_backticks_inside_value_preserved(self) -> None:
        # A markdown fence is only stripped when the entire text starts with it.
        text = '{"title": "use ```code``` here", "bullets": []}'
        result = parse_json_response(text, _Summary)
        assert "```code```" in result.title

    def test_single_line_fence_left_alone(self) -> None:
        # A single line starting with ``` and no newline isn't a real fence —
        # we don't attempt to strip it (would corrupt the body).
        text = "```not a real fence"
        with pytest.raises(ValidationError):
            parse_json_response(text, _Summary)
