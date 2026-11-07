"""Bounded schema/output checks preserve supplied semantics instead of repairing generations."""

from __future__ import annotations

import copy
import json
from typing import Any, cast

import pytest

from strata_forge.llm._generation_schema import (
    GenerationError,
    canonical_json,
    conforms,
    parse_json,
    validate_schema,
)

# Deliberately heterogeneous JSON shapes exercise the strict portable boundary.
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": ["string", "null"], "minLength": 1, "maxLength": 4},
        "count": {"type": "integer", "minimum": 0, "maximum": 3},
        "ratio": {"type": "number", "minimum": 0, "maximum": 1},
        "flags": {"type": "array", "items": {"type": "boolean"}, "maxItems": 2},
        "label": {"type": "string", "enum": ["a", "b"], "maxLength": 1},
    },
    "required": ["text", "count", "ratio", "flags", "label"],
    "additionalProperties": False,
}
BASE: dict[str, Any] = {"text": None, "count": 2, "ratio": 0.5, "flags": [True], "label": "a"}


@pytest.mark.parametrize(
    ("field", "value", "valid"),
    [
        ("text", None, True),
        ("text", "a\n\u03b1", True),
        ("text", "", False),
        ("text", "xxxxx", False),
        ("count", 0, True),
        ("count", 3.0, True),
        ("count", True, False),
        ("count", 1.5, False),
        ("count", 4, False),
        ("ratio", 1, True),
        ("ratio", 1.1, False),
        ("flags", [True, False], True),
        ("flags", [0], False),
        ("flags", [True] * 3, False),
        ("label", "b", True),
        ("label", "c", False),
    ],
)
def test_matches_actual_json_schema_semantics_without_coercion(
    field: str, value: object, valid: bool
) -> None:
    from jsonschema import Draft202012Validator

    raw = copy.deepcopy(BASE)
    raw[field] = value
    schema = validate_schema(SCHEMA)
    assert conforms(schema, parse_json(json.dumps(raw), max_bytes=4096)) is valid
    assert cast("Any", Draft202012Validator)(SCHEMA).is_valid(raw) is valid


def test_closed_objects_missing_or_extra_keys_fail() -> None:
    schema = validate_schema(SCHEMA)
    for value in ({**BASE, "extra": 1}, {k: v for k, v in BASE.items() if k != "text"}):
        assert conforms(schema, parse_json(json.dumps(value), max_bytes=4096)) is False


@pytest.mark.parametrize(
    "keyword",
    [
        "$ref",
        "$defs",
        "$schema",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "pattern",
        "format",
        "patternProperties",
        "unevaluatedProperties",
        "dependentSchemas",
        "default",
        "title",
        "description",
    ],
)
def test_unsupported_keywords_refuse_instead_of_disappearing(keyword: str) -> None:
    raw = copy.deepcopy(SCHEMA)
    raw[keyword] = "https://fixture.invalid/schema"
    with pytest.raises(GenerationError):
        validate_schema(raw)
    assert raw[keyword] == "https://fixture.invalid/schema"


@pytest.mark.parametrize(
    "change",
    [
        "additional_true",
        "missing_required",
        "duplicate_required",
        "required_subset",
        "nullable_order",
        "float_max_items",
        "bool_max_length",
        "missing_string_bound",
        "bad_bounds",
        "enum_wrong_type",
        "enum_duplicate",
        "enum_outside_bound",
        "missing_array_items",
        "properties_over_cap",
        "deep",
    ],
)
def test_invalid_shapes_and_resource_bounds(change: str) -> None:
    raw = copy.deepcopy(SCHEMA)
    if change == "additional_true":
        raw["additionalProperties"] = True
    elif change == "missing_required":
        del raw["required"]
    elif change == "duplicate_required":
        raw["required"].append("text")
    elif change == "required_subset":
        raw["required"].remove("text")
    elif change == "nullable_order":
        raw["properties"]["text"]["type"] = ["null", "string"]
    elif change == "float_max_items":
        raw["properties"]["flags"]["maxItems"] = 2.0
    elif change == "bool_max_length":
        raw["properties"]["label"]["maxLength"] = True
    elif change == "missing_string_bound":
        del raw["properties"]["label"]["maxLength"]
    elif change == "bad_bounds":
        raw["properties"]["ratio"].update(minimum=2, maximum=1)
    elif change == "enum_wrong_type":
        raw["properties"]["label"]["enum"] = [1]
    elif change == "enum_duplicate":
        raw["properties"]["label"]["enum"] = ["a", "a"]
    elif change == "enum_outside_bound":
        raw["properties"]["label"]["enum"] = ["long"]
    elif change == "missing_array_items":
        del raw["properties"]["flags"]["items"]
    elif change == "properties_over_cap":
        raw = {
            "type": "object",
            "properties": {str(i): {"type": "null"} for i in range(65)},
            "required": [str(i) for i in range(65)],
            "additionalProperties": False,
        }
    else:
        child = {"type": "null"}
        for _ in range(10):
            child = {"type": "array", "items": child, "maxItems": 1}
        raw["properties"]["deep"] = child
        raw["required"].append("deep")
    with pytest.raises(GenerationError):
        validate_schema(raw)


@pytest.mark.parametrize(
    "raw",
    [
        b"\xff",
        b'{"a":1,"a":2}',
        b'{"kind":"x","\\u006bind":"y"}',
        b"{} {}",
        b"```json\n{}\n```",
        b"NaN",
        b"Infinity",
        b"1e9999",
        b"9007199254740992",
        b'"\\ud800"',
        '"\ud800\\ude00"',
        '"\\ud83d\ude00"',
    ],
)
def test_raw_codec_refuses_without_body_context(raw: str | bytes) -> None:
    with pytest.raises(GenerationError) as caught:
        parse_json(raw, max_bytes=4096)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_schema_is_copied_without_inserting_defaults_or_normalizing_text() -> None:
    source: dict[str, Any] = {
        "type": "object",
        "properties": {"\u03b1": {"type": "string", "maxLength": 8}},
        "required": ["\u03b1"],
        "additionalProperties": False,
    }
    before = canonical_json(source)
    actual = validate_schema(source)
    assert canonical_json(actual) == before
    source["properties"]["\u03b1"]["maxLength"] = 0
    assert actual["properties"] != {"\u03b1": {"type": "string", "maxLength": 0}}
    value = {"\u03b1": "a\r\ne\u0301\u2028"}
    assert conforms(actual, parse_json(json.dumps(value), max_bytes=4096))


def test_byte_node_and_cycle_bounds_refuse_without_expansion() -> None:
    with pytest.raises(GenerationError):
        parse_json(b'"' + b"a" * 4096 + b'"', max_bytes=4096)
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(GenerationError):
        canonical_json(cycle)
    with pytest.raises(GenerationError):
        canonical_json(["x"] * 1000, max_bytes=64)
    with pytest.raises(GenerationError):
        validate_schema({"type": "object", "unknown": "x" * 16384})


@pytest.mark.parametrize(("minimum", "maximum"), [(1.2, 1.3), (1e20, 1e21), (-1e21, -1e20)])
def test_integer_interval_with_no_supported_value_refuses(minimum: float, maximum: float) -> None:
    raw = copy.deepcopy(SCHEMA)
    raw["properties"]["count"].update(minimum=minimum, maximum=maximum)
    with pytest.raises(GenerationError):
        validate_schema(raw)
