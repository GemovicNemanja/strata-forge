"""Finite JSON data and schema checks for explicitly constrained local generation."""

from __future__ import annotations

import json
import math
from typing import cast

from strata_forge.core.errors import ForgeError

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]

MAX_SCHEMA_BYTES = 16 * 1024
MAX_SCHEMA_DEPTH = 8
MAX_SCHEMA_NODES = 256
MAX_SCHEMA_PROPERTIES = 64
MAX_SAFE_INTEGER = 2**53 - 1


class GenerationError(ForgeError):
    """A fixed generation boundary refusal without raw request/provider material."""


def scalar_text(value: object) -> bool:
    return type(value) is str and not any(0xD800 <= ord(char) <= 0xDFFF for char in value)


def _pairs(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise GenerationError("generation_json_invalid")
        result[key] = value
    return result


def _integer(value: str) -> int:
    if len(value) > 17:
        raise GenerationError("generation_json_invalid")
    parsed = int(value)
    if abs(parsed) > MAX_SAFE_INTEGER:
        raise GenerationError("generation_json_invalid")
    return parsed


def _float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise GenerationError("generation_json_invalid")
    return parsed


def _constant(value: str) -> JsonValue:
    raise GenerationError("generation_json_invalid")


def _data(value: object, depth: int, budget: list[int]) -> bool:
    budget[1] -= 1
    if depth > 32 or budget[1] < 0:
        return False
    if value is None or type(value) is bool:
        budget[0] -= 4 if value is None or value is True else 5
        return budget[0] >= 0
    if type(value) is int:
        if abs(value) > MAX_SAFE_INTEGER:
            return False
        budget[0] -= len(str(value))
        return budget[0] >= 0
    if type(value) is float:
        budget[0] -= len(repr(value))
        return budget[0] >= 0 and math.isfinite(value)
    if type(value) is str:
        if len(value) > budget[0] or not scalar_text(value):
            return False
        budget[0] -= len(value.encode("utf-8"))
        return budget[0] >= 0
    if type(value) is list:
        return all(_data(child, depth + 1, budget) for child in cast("list[object]", value))
    if type(value) is dict:
        return all(
            type(key) is str and _data(key, depth + 1, budget) and _data(child, depth + 1, budget)
            for key, child in cast("dict[object, object]", value).items()
        )
    return False


def canonical_json(value: object, *, max_bytes: int = 1024 * 1024) -> bytes:
    """Encode finite scalar JSON without altering string or property ordering semantics."""
    if not _data(value, 0, [max_bytes, max_bytes]):
        raise GenerationError("generation_json_invalid")
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        raise GenerationError("generation_json_invalid")
    return encoded


def parse_json(raw: str | bytes, *, max_bytes: int) -> JsonValue:
    """Reject invalid UTF-8, duplicate keys and nonfinite values before typed conversion."""
    failed = False
    parsed: object = None
    try:
        encoded = raw.encode("utf-8") if type(raw) is str else raw
        if type(encoded) is not bytes or len(encoded) > max_bytes:
            failed = True
        else:
            parsed = json.loads(
                encoded.decode("utf-8"),
                object_pairs_hook=_pairs,
                parse_int=_integer,
                parse_float=_float,
                parse_constant=_constant,
            )
    except UnicodeError, ValueError, RecursionError, GenerationError:
        failed = True
    # Raising outside the handler prevents a JSONDecodeError from retaining the source body.
    if failed or not _data(parsed, 0, [max_bytes, max_bytes]):
        raise GenerationError("generation_json_invalid")
    return cast("JsonValue", parsed)


def _bound(value: object, *, maximum: int) -> bool:
    return type(value) is int and 0 <= value <= maximum


def _number(value: object) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _matches_type(kind: str, value: JsonValue) -> bool:
    if kind == "null":
        return value is None
    if kind == "boolean":
        return type(value) is bool
    if kind == "string":
        return type(value) is str
    if kind == "integer":
        return (
            _number(value)
            and abs(cast("int | float", value)) <= MAX_SAFE_INTEGER
            and (type(value) is int or cast("float", value).is_integer())
        )
    if kind == "number":
        return _number(value)
    if kind == "array":
        return type(value) is list
    return kind == "object" and type(value) is dict


def validate_schema(value: object) -> dict[str, JsonValue]:
    """Copy one bounded, closed schema dialect without keyword removal or coercion."""
    encoded = canonical_json(value, max_bytes=MAX_SCHEMA_BYTES)
    if len(encoded) > MAX_SCHEMA_BYTES:
        raise GenerationError("generation_schema_unsupported")
    copied = parse_json(encoded, max_bytes=MAX_SCHEMA_BYTES)
    nodes = properties = 0

    def walk(raw: JsonValue, depth: int) -> None:
        nonlocal nodes, properties
        nodes += 1
        if type(raw) is not dict or depth > MAX_SCHEMA_DEPTH or nodes > MAX_SCHEMA_NODES:
            raise GenerationError("generation_schema_unsupported")
        schema = cast("dict[str, JsonValue]", raw)
        declared = schema.get("type")
        kinds = [declared] if type(declared) is str else declared
        if (
            type(kinds) is not list
            or not kinds
            or any(type(kind) is not str for kind in kinds)
            or any(
                kind not in ("object", "array", "string", "integer", "number", "boolean", "null")
                for kind in kinds
            )
            or (len(kinds) != 1 and (len(kinds) != 2 or kinds[1] != "null" or kinds[0] == "null"))
        ):
            raise GenerationError("generation_schema_unsupported")
        kind = cast("str", kinds[0])
        allowed = {"type"}
        if kind == "object":
            allowed |= {"properties", "required", "additionalProperties"}
            props, required = schema.get("properties"), schema.get("required")
            if (
                type(props) is not dict
                or type(required) is not list
                or schema.get("additionalProperties") is not False
            ):
                raise GenerationError("generation_schema_unsupported")
            props = cast("dict[str, JsonValue]", props)
            properties += len(props)
            if properties > MAX_SCHEMA_PROPERTIES or any(
                not key or len(key.encode("utf-8")) > 128 for key in props
            ):
                raise GenerationError("generation_schema_unsupported")
            if (
                any(type(key) is not str for key in required)
                or len(required) != len(props)
                or set(cast("list[str]", required)) != set(props)
            ):
                raise GenerationError("generation_schema_unsupported")
            for child in props.values():
                walk(child, depth + 1)
        elif kind == "array":
            allowed |= {"items", "minItems", "maxItems"}
            minimum, maximum = schema.get("minItems", 0), schema.get("maxItems")
            if not _bound(maximum, maximum=128) or not _bound(
                minimum, maximum=cast("int", maximum)
            ):
                raise GenerationError("generation_schema_unsupported")
            walk(schema.get("items"), depth + 1)
        elif kind == "string":
            allowed |= {"minLength", "maxLength", "enum"}
            minimum, maximum = schema.get("minLength", 0), schema.get("maxLength")
            if not _bound(maximum, maximum=4096) or not _bound(
                minimum, maximum=cast("int", maximum)
            ):
                raise GenerationError("generation_schema_unsupported")
        elif kind in ("number", "integer"):
            allowed |= {"minimum", "maximum", "enum"}
            for field in ("minimum", "maximum"):
                if field in schema and not _number(schema[field]):
                    raise GenerationError("generation_schema_unsupported")
            if (
                "minimum" in schema
                and "maximum" in schema
                and cast("float", schema["minimum"]) > cast("float", schema["maximum"])
            ):
                raise GenerationError("generation_schema_unsupported")
            if kind == "integer":
                lower = max(
                    -MAX_SAFE_INTEGER, cast("float", schema.get("minimum", -MAX_SAFE_INTEGER))
                )
                upper = min(
                    MAX_SAFE_INTEGER, cast("float", schema.get("maximum", MAX_SAFE_INTEGER))
                )
                if math.ceil(lower) > math.floor(upper):
                    raise GenerationError("generation_schema_unsupported")
        else:
            allowed.add("enum")
        if set(schema) - allowed:
            raise GenerationError("generation_schema_unsupported")
        if "enum" in schema:
            values = schema["enum"]
            if type(values) is not list or not 1 <= len(values) <= 64:
                raise GenerationError("generation_schema_unsupported")
            seen: list[JsonValue] = []
            for item in values:
                if (
                    not any(_matches_type(cast("str", selected), item) for selected in kinds)
                    or item in seen
                ):
                    raise GenerationError("generation_schema_unsupported")
                if not conforms(schema, item):
                    raise GenerationError("generation_schema_unsupported")
                seen.append(item)

    walk(copied, 0)
    if type(copied) is not dict or copied.get("type") != "object":
        raise GenerationError("generation_schema_unsupported")
    return copied


def conforms(schema: dict[str, JsonValue], value: JsonValue) -> bool:
    """Check generated JSON against an already validated schema without executing user code."""
    declared = schema["type"]
    kinds = [declared] if type(declared) is str else cast("list[JsonValue]", declared)
    if not any(_matches_type(cast("str", kind), value) for kind in kinds):
        return False
    if value is None:
        return "enum" not in schema or value in cast("list[JsonValue]", schema["enum"])
    if "enum" in schema and value not in cast("list[JsonValue]", schema["enum"]):
        return False
    if type(value) is dict:
        props = cast("dict[str, dict[str, JsonValue]]", schema["properties"])
        return set(value) == set(props) and all(
            conforms(props[key], child) for key, child in value.items()
        )
    if type(value) is list:
        return cast("int", schema.get("minItems", 0)) <= len(value) <= cast(
            "int", schema["maxItems"]
        ) and all(conforms(cast("dict[str, JsonValue]", schema["items"]), child) for child in value)
    if type(value) is str:
        return (
            cast("int", schema.get("minLength", 0))
            <= len(value)
            <= cast("int", schema["maxLength"])
        )
    if _number(value):
        numeric = cast("float", value)
        return ("minimum" not in schema or numeric >= cast("float", schema["minimum"])) and (
            "maximum" not in schema or numeric <= cast("float", schema["maximum"])
        )
    return True
