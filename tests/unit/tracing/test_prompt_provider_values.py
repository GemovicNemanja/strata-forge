"""Synthetic typed values and exact projections, with no provider/credential work."""

from __future__ import annotations

import hashlib
import math
import struct
from decimal import Decimal
from fractions import Fraction

import pytest

from strata_forge.prompts import portable
from strata_forge.prompts.portable import canonical_bytes
from strata_forge.tracing import prompt_provider_values as values
from strata_forge.tracing.prompt_provider_values import (
    ProviderValueError,
    encode_provider_value,
    project_chat,
    project_response,
    project_responses,
)


@pytest.mark.parametrize(
    ("native", "tag"),
    [
        (None, ["null"]),
        (False, ["bool", False]),
        (True, ["bool", True]),
        (0, ["integer", "0"]),
        (1, ["integer", "1"]),
        (-1, ["integer", "-1"]),
        (2**80, ["integer", str(2**80)]),
        (1.0, ["binary64", "3ff0000000000000"]),
        (0.0, ["binary64", "0000000000000000"]),
        (-0.0, ["binary64", "8000000000000000"]),
        (5e-324, ["binary64", "0000000000000001"]),
        (float.fromhex("0x1.fffffffffffffp1023"), ["binary64", "7fefffffffffffff"]),
        ("😀e\u0301\r\n{{literal}}\x00", ["string", "😀e\u0301\r\n{{literal}}\x00"]),
        ([1, None], ["array", [["integer", "1"], ["null"]]]),
        ({"注記": 0.7}, ["object", [["注記", ["binary64", struct.pack(">d", 0.7).hex()]]]]),
    ],
)
def test_exact_scalar_type_bits_order_and_unicode(native: object, tag: object) -> None:
    assert encode_provider_value(native) == canonical_bytes(tag)


def test_native_object_order_is_preserved_and_type_presence_hashes_are_distinct() -> None:
    assert encode_provider_value({"a": 1, "b": 2}) != encode_provider_value({"b": 2, "a": 1})
    distinct = [
        project_chat("fixture", [], item).canonical_tagged_sha256
        for item in [
            {},
            {"temperature": None},
            {"temperature": False},
            {"temperature": 0},
            {"temperature": 1},
            {"temperature": 1.0},
            {"temperature": -0.0},
            {"temperature": 0.0},
        ]
    ]
    assert len(set(distinct)) == len(distinct)
    assert project_chat("fixture", [], {"temperature": 0.7, "top_p": 1.0}) == project_chat(
        "fixture", [], {"top_p": 1.0, "temperature": 0.7}
    )


def test_unallowlisted_values_are_not_read_or_hashed() -> None:
    class Secret:
        def __getattribute__(self, _name: str) -> object:
            raise AssertionError("PRIVATE CONFIG CANARY")

        def __repr__(self) -> str:
            raise AssertionError("PRIVATE CONFIG CANARY")

    first = project_chat(
        "fixture",
        [{"role": "user", "content": "synthetic"}],
        {"temperature": 0.7, "api_key": Secret(), "headers": Secret()},
    )
    second = project_chat(
        "fixture",
        [{"role": "user", "content": "synthetic"}],
        {"temperature": 0.7, "api_key": object(), "headers": object()},
    )
    assert first == second
    assert first.status == "observed"
    assert first.omitted_fields_count == 2
    request: dict[str, object] = {
        "input": [],
        "store": False,
        "include": [],
        "extra_headers": Secret(),
        "base_url": Secret(),
    }
    response = project_responses("fixture", request)
    assert response.status == "observed"
    assert response.omitted_fields_count == 2
    assert response.selected_present_fields_count == 3


def test_custom_values_keys_cycles_and_nonfinite_numbers_are_never_coerced() -> None:
    accesses: list[str] = []

    class Custom:
        def __iter__(self) -> object:
            accesses.append("iter")
            raise AssertionError("CANARY")

        def __str__(self) -> str:
            accesses.append("str")
            raise AssertionError("CANARY")

        def __float__(self) -> float:
            accesses.append("float")
            raise AssertionError("CANARY")

        def model_dump(self) -> object:
            accesses.append("dump")
            raise AssertionError("CANARY")

    class Key(str):
        def __eq__(self, _other: object) -> bool:
            accesses.append("keyeq")
            raise AssertionError("CANARY")

        __hash__ = str.__hash__

    cycle: list[object] = []
    cycle.append(cycle)
    for unsupported in [
        Custom(),
        Decimal("0.7"),
        Fraction(1, 2),
        (1, 2),
        {Key("temperature"): 1},
        cycle,
        math.inf,
        -math.inf,
        math.nan,
        "\ud800",
    ]:
        with pytest.raises(ProviderValueError, match="projection_unsupported") as caught:
            encode_provider_value(unsupported)
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None
    assert project_chat("fixture", [], {Key("temperature"): 1}).status == "unsupported"
    assert accesses == []


def test_opaque_response_does_not_access_custom_serialization_and_has_unknown_counts() -> None:
    class Opaque:
        def __getattribute__(self, _name: str) -> object:
            raise AssertionError("RESPONSE CANARY")

    result = project_response("acompletion", Opaque())
    assert result.status == "opaque"
    assert result.canonical_tagged_sha256 is None
    assert result.selected_present_fields_count is result.omitted_fields_count is None
    responses: list[dict[str, object]] = [
        {"choices": []},
        {"output": [], "usage": {"total": 0}, "secret": Opaque()},
    ]
    for item in responses:
        observed = project_response("aresponses", item)
        assert observed.status == "observed"


def nested(kind: str, depth: int) -> object:
    value: object = None
    for _ in range(depth):
        value = [value] if kind == "array" else {"x": value}
    return value


def test_native_and_weighted_frame_depth_boundaries_use_unchanged_portable_limits() -> None:
    assert values.MAX_PROJECTION_DEPTH == 32
    assert (
        values.MAX_TAGGED_DEPTH + values.PROJECTION_FRAME_DEPTH_RESERVATION
        == portable.MAX_JSON_DEPTH
        == 64
    )
    for kind, supported in [("array", 28), ("object", 18)]:
        assert project_chat("fixture", nested(kind, supported), {}).status == "observed"
        assert project_responses("fixture", {"input": nested(kind, supported)}).status == "observed"
        assert (
            project_response("acompletion", {"choices": nested(kind, supported)}).status
            == "observed"
        )
        assert project_chat("fixture", nested(kind, supported + 1), {}).status == "limit"
        with pytest.raises(ProviderValueError, match="projection_limit"):
            encode_provider_value(nested(kind, supported + 1))


def test_nodes_map_integer_and_byte_caps_never_hash_a_prefix() -> None:
    assert encode_provider_value([None] * 65_535)
    for value in [[None] * 65_536, {str(index): None for index in range(4097)}, 10**1024]:
        with pytest.raises(ProviderValueError, match="projection_limit"):
            encode_provider_value(value)
    assert encode_provider_value({str(index): None for index in range(4096)})
    assert encode_provider_value(10**1023)
    overhead = len(b'["string",""]')
    payload = "x" * (16 * 1024 * 1024 - overhead)
    encoded = encode_provider_value(payload)
    assert len(encoded) == 16 * 1024 * 1024
    assert project_chat("fixture", payload, {}).status == "limit"  # fixed framing also counts
    with pytest.raises(ProviderValueError, match="projection_limit"):
        encode_provider_value(payload + "x")
    limited = project_responses("fixture", {"input": [None] * 65_536})
    assert limited.status == "limit"
    assert limited.canonical_tagged_sha256 is limited.encoded_size_bytes is None


def test_underlying_portable_limit_remains_limit_with_fixed_error_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(portable, "MAX_JSON_BYTES", 32)
    with pytest.raises(ProviderValueError, match="projection_limit") as caught:
        encode_provider_value("synthetic" * 20)
    assert caught.value.__context__ is None
    assert project_chat("fixture", [], {}).status == "limit"


def test_projection_is_explicit_domain_method_presence_and_never_raw_json_wire() -> None:
    first = project_chat("fixture", [], {"temperature": 0.7})
    expected: dict[str, object] = {
        "domain": "strata-provider-input-projection-v1",
        "profile": "forge-provider-client-input/v1",
        "method": "acompletion",
        "provider_model_id": ["string", "fixture"],
        "fields": [
            [name, ["present", ["array", []]]]
            if name == "messages"
            else [name, ["present", ["binary64", struct.pack(">d", 0.7).hex()]]]
            if name == "temperature"
            else [name, ["absent"]]
            for name in values.CHAT_FIELDS
        ],
    }
    assert first.canonical_tagged_sha256 == hashlib.sha256(canonical_bytes(expected)).hexdigest()
    assert first.encoded_size_bytes == len(canonical_bytes(expected))
    assert (
        first.canonical_tagged_sha256
        != hashlib.sha256(b'{"temperature":0.7,"messages":[]}').hexdigest()
    )


def test_omitted_top_level_names_are_not_scanned_or_validated_as_projected_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = values._quoted_size  # pyright: ignore[reportPrivateUsage] - prove no traversal of the exact omitted name
    unknown = "omitted_" + "x" * 100_000 + "\ud800"

    def checked(text: str) -> int:
        assert text is not unknown
        return original(text)

    monkeypatch.setattr(values, "_quoted_size", checked)
    first = project_chat("fixture", [], {unknown: object()})
    second = project_chat("fixture", [], {"other_omitted": object()})
    assert first.status == "observed"
    assert first.omitted_fields_count == 1
    assert first == second


def test_actual_portable_safe_integer_capacity_refusal_has_limit_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(portable, "MAX_SAFE_INTEGER", 0)
    with pytest.raises(ProviderValueError, match="projection_limit") as caught:
        values._encode({"domain": "fixture", "fixed": 1})  # pyright: ignore[reportPrivateUsage] - actual fixed-frame canon capacity boundary
    assert caught.value.__context__ is None
    assert encode_provider_value(
        2**80
    )  # Native integers remain tagged strings, never a widened canon integer.


def test_shared_explicit_native_recipes_and_actual_projection_vectors() -> None:
    import json
    from pathlib import Path
    from typing import cast

    fixture = cast(
        "dict[str, object]",
        json.loads(
            (
                Path(__file__).parents[3] / "tests/fixtures/prompts/provider-input-v1.json"
            ).read_text()
        ),
    )

    def decode(item: dict[str, object]) -> object:
        kind = item["kind"]
        if kind == "null":
            return None
        if kind in {"string", "boolean"}:
            return item["value"]
        if kind == "integer":
            return int(cast("str", item["decimal"]))
        if kind == "binary64":
            return struct.unpack(">d", bytes.fromhex(cast("str", item["bits"])))[0]
        if kind == "array":
            return [decode(child) for child in cast("list[dict[str, object]]", item["items"])]
        if kind == "object":
            return {
                key: decode(child)
                for key, child in cast("list[tuple[str, dict[str, object]]]", item["entries"])
            }
        raise AssertionError("fixture_recipe_invalid")

    for vector in cast("list[dict[str, object]]", fixture["vectors"]):
        raw = encode_provider_value(decode(cast("dict[str, object]", vector["native_recipe"])))
        assert raw.hex() == vector["tagged_utf8_hex"]
        assert hashlib.sha256(raw).hexdigest() == vector["tagged_sha256"]
    for vector in cast("list[dict[str, object]]", fixture["projections"]):
        model = decode(cast("dict[str, object]", vector["model_recipe"]))
        arguments = cast(
            "dict[str, object]", decode(cast("dict[str, object]", vector["arguments_recipe"]))
        )
        if vector["method"] == "acompletion":
            messages = arguments.pop("messages")
            projection = project_chat(model, messages, arguments)
        else:
            projection = project_responses(model, arguments)
        assert projection.model_dump(mode="json") == vector["projection"]
        raw = bytes.fromhex(cast("str", vector["framed_utf8_hex"]))
        assert projection.canonical_tagged_sha256 == hashlib.sha256(raw).hexdigest()
        assert projection.encoded_size_bytes == len(raw)
