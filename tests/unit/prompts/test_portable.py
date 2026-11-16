"""Independent byte/hash fixtures and refusal cases for the portable prompt JSON profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from strata_forge.prompts import portable
from strata_forge.prompts.composition_errors import PromptCompositionError
from strata_forge.prompts.portable import canonical_bytes, parse_portable_json, portable_sha256

FIXTURES = Path(__file__).parents[2] / "fixtures/prompts/portable-bytes-v1.json"


def test_coordinator_byte_and_hash_fixtures() -> None:
    fixture = json.loads(FIXTURES.read_text(encoding="utf-8"))
    for case in fixture["cases"]:
        encoded = canonical_bytes(case["value"])
        assert encoded.hex() == case["canonical_utf8_hex"]
        assert hashlib.sha256(encoded).hexdigest() == case["sha256"]
        assert portable_sha256(case["value"]) == case["sha256"]
        assert parse_portable_json(encoded) == case["value"]


@pytest.mark.parametrize(
    "raw",
    [
        "-0",
        "-00",
        "+1",
        "01",
        "1.0",
        "1e0",
        "1E2",
        "NaN",
        "Infinity",
        "-Infinity",
        "9007199254740992",
        "-9007199254740992",
    ],
)
def test_numeric_spelling_refuses_before_conversion(raw: str) -> None:
    with pytest.raises(PromptCompositionError) as caught:
        parse_portable_json(raw)
    assert caught.value.__context__ is None
    assert raw not in str(caught.value)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":{"b":1,"b":1}}', '[{"a":null,"a":false}]'])
def test_duplicate_keys_are_never_overwritten(raw: str) -> None:
    with pytest.raises(PromptCompositionError, match="portable_duplicate_key") as caught:
        parse_portable_json(raw)
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        float("nan"),
        float("inf"),
        b"text",
        (1, 2),
        {1, 2},
        object(),
        {1: "x"},
        {"é": "x"},
        {"\n": "x"},
        chr(0xD800),
        chr(0xDFFF),
    ],
)
def test_nonportable_host_values_refuse(value: object) -> None:
    with pytest.raises(PromptCompositionError):
        canonical_bytes(value)


@pytest.mark.parametrize(
    "raw", [b"\xef\xbb\xbf{}", b"\xff", b'{"secret":"\xff"}', '"\\ud800"', '"\\udfff"']
)
def test_invalid_utf8_and_surrogates_refuse_without_raw_context(raw: bytes | str) -> None:
    with pytest.raises(PromptCompositionError) as caught:
        parse_portable_json(raw)
    assert caught.value.__context__ is None


def test_scalar_and_whitespace_bytes_are_exact() -> None:
    assert canonical_bytes([True, False, 0, -1]) == b"[true,false,0,-1]"
    assert canonical_bytes("e\u0301") != canonical_bytes("é")
    assert canonical_bytes("\r\n") != canonical_bytes("\n")
    assert parse_portable_json('"\\ud83d\\ude00"') == "😀"
    assert canonical_bytes("\ufeff").hex() == "22efbbbf22"
    assert parse_portable_json('"\\ufeff"') == "\ufeff"
    assert (
        canonical_bytes({"": "", "__proto__": "literal", "a\\b": "/"})
        == b'{"":"","__proto__":"literal","a\\\\b":"/"}'
    )


def test_cycles_and_depth_are_bounded() -> None:
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(PromptCompositionError, match="portable_cycle"):
        canonical_bytes(cycle)
    value: object = None
    for _ in range(65):
        value = [value]
    with pytest.raises(PromptCompositionError, match="portable_depth_limit"):
        canonical_bytes(value)


def test_encoded_size_is_accounted_before_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(portable, "MAX_JSON_BYTES", 12)
    assert canonical_bytes("\x00") == b'"\\u0000"'
    for value in ("a" * 13, "\x00\x00", ["a", "bbbbbb"], {"a": "bbbbbb"}):
        with pytest.raises(PromptCompositionError, match="portable_bytes_limit"):
            canonical_bytes(value)
    with pytest.raises(PromptCompositionError, match="portable_bytes_limit"):
        parse_portable_json('"' + "a" * 12 + '"')


@pytest.mark.parametrize("value", [{}, {"domain": ""}, {"domain": 1}, []])
def test_hash_requires_a_named_domain(value: object) -> None:
    with pytest.raises(PromptCompositionError, match="portable_domain_required"):
        portable_sha256(value)
