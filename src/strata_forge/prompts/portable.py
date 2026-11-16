"""Exact bounded JSON bytes for the strata-prompt-json-v1 portable profile."""

from __future__ import annotations

import hashlib
import json
import re
from typing import cast

from strata_forge.prompts.composition_errors import PromptCompositionError

type PortableValue = None | bool | str | int | list[PortableValue] | dict[str, PortableValue]
MAX_SAFE_INTEGER = 9_007_199_254_740_991
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_JSON_DEPTH = 64
_INTEGER = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")


def _consume(count: int, budget: list[int]) -> None:
    budget[0] += count
    if budget[0] > MAX_JSON_BYTES:
        raise PromptCompositionError("portable_bytes_limit")


def _validate(value: object, depth: int, ancestors: set[int], budget: list[int]) -> None:
    if depth > MAX_JSON_DEPTH:
        raise PromptCompositionError("portable_depth_limit")
    if value is None or type(value) is bool:
        _consume(4 if value is None or value else 5, budget)
        return
    if type(value) is int:
        if abs(value) > MAX_SAFE_INTEGER:
            raise PromptCompositionError("portable_integer_range")
        _consume(len(str(value)), budget)
        return
    if type(value) is str:
        if len(value) > MAX_JSON_BYTES:
            raise PromptCompositionError("portable_bytes_limit")
        if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            raise PromptCompositionError("portable_unicode_invalid")
        extra = sum(1 if c in '"\\\b\t\n\f\r' else 5 if ord(c) < 32 else 0 for c in value)
        _consume(len(value.encode("utf-8")) + extra + 2, budget)
        return
    if type(value) not in (list, dict):
        raise PromptCompositionError("portable_type_invalid")
    identity = id(value)
    if identity in ancestors:
        raise PromptCompositionError("portable_cycle")
    ancestors.add(identity)
    if type(value) is dict:
        fields = cast("dict[object, object]", value)
        _consume(2 + max(0, len(fields) - 1) + len(fields), budget)
        for key, child in fields.items():
            if type(key) is not str or any(not 0x20 <= ord(c) <= 0x7E for c in key):
                raise PromptCompositionError("portable_key_invalid")
            _validate(key, depth + 1, ancestors, budget)
            _validate(child, depth + 1, ancestors, budget)
    else:
        children = cast("list[object]", value)
        _consume(2 + max(0, len(children) - 1), budget)
        for child in children:
            _validate(child, depth + 1, ancestors, budget)
    ancestors.remove(identity)


def canonical_bytes(value: object) -> bytes:
    """Encode the bounded portable profile without whitespace or Unicode normalization."""
    _validate(value, 0, set(), [0])
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_JSON_BYTES:
        raise PromptCompositionError("portable_bytes_limit")
    return encoded


def _integer(text: str) -> int:
    if len(text) > 17 or not _INTEGER.fullmatch(text):
        raise PromptCompositionError("portable_number_invalid")
    value = int(text)
    if abs(value) > MAX_SAFE_INTEGER:
        raise PromptCompositionError("portable_integer_range")
    return value


def _no_number(_text: str) -> PortableValue:
    raise PromptCompositionError("portable_number_invalid")


def _object(pairs: list[tuple[str, PortableValue]]) -> dict[str, PortableValue]:
    result: dict[str, PortableValue] = {}
    for key, value in pairs:
        if key in result:
            raise PromptCompositionError("portable_duplicate_key")
        result[key] = value
    return result


def parse_portable_json(raw: bytes | str) -> PortableValue:
    """Reject duplicate keys and invalid numeric spellings before host conversion."""
    parsed: PortableValue | None = None
    failure = "portable_json_invalid"
    valid = False
    try:
        if type(raw) not in (bytes, str):
            raise PromptCompositionError("portable_type_invalid")
        encoded = raw if isinstance(raw, bytes) else raw.encode("utf-8")
        if len(encoded) > MAX_JSON_BYTES:
            raise PromptCompositionError("portable_bytes_limit")
        if encoded.startswith(b"\xef\xbb\xbf"):
            raise PromptCompositionError("portable_json_invalid")
        parsed = cast(
            "PortableValue",
            json.loads(
                encoded.decode("utf-8"),
                parse_int=_integer,
                parse_float=_no_number,
                parse_constant=_no_number,
                object_pairs_hook=_object,
            ),
        )
        _validate(parsed, 0, set(), [0])
        valid = True
    except PromptCompositionError as error:
        failure = error.diagnostic.code
    except ValueError, UnicodeError, RecursionError:
        valid = False
    if not valid:
        raise PromptCompositionError(failure)
    return parsed


def portable_sha256(value: object) -> str:
    """Fingerprint a domain-bearing portable object using its exact canonical bytes."""
    if type(value) is not dict:
        raise PromptCompositionError("portable_domain_required")
    fields = cast("dict[str, object]", value)
    if type(fields.get("domain")) is not str or not fields["domain"]:
        raise PromptCompositionError("portable_domain_required")
    return hashlib.sha256(canonical_bytes(fields)).hexdigest()
