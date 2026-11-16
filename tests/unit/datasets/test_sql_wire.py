"""Known lossless tagged values and malformed stdlib protocol controls; no native SQL."""

from __future__ import annotations

import struct
from datetime import UTC, date, datetime, time
from decimal import Decimal

import pytest

from strata_forge.datasets import _sql_wire as wire


@pytest.mark.parametrize(
    ("dtype", "native", "expected"),
    [
        (
            {"kind": "integer", "bits": 64, "signed": True},
            9007199254740993,
            {"kind": "integer", "value": "9007199254740993"},
        ),
        (
            {"kind": "integer", "bits": 128, "signed": True},
            -(2**127),
            {"kind": "integer", "value": "-170141183460469231731687303715884105728"},
        ),
        (
            {"kind": "decimal", "precision": 12, "scale": 2},
            Decimal("1.20"),
            {"kind": "decimal", "value": "1.20"},
        ),
        ({"kind": "float", "bits": 64}, -0.0, {"kind": "float", "value": "-0x0.0p+0"}),
        ({"kind": "float", "bits": 64}, float("nan"), {"kind": "float", "value": "nan"}),
        (
            {"kind": "float", "bits": 64},
            float("inf"),
            {"kind": "float", "value": "positive_infinity"},
        ),
        (
            {"kind": "float", "bits": 64},
            -float("inf"),
            {"kind": "float", "value": "negative_infinity"},
        ),
        ({"kind": "boolean"}, False, {"kind": "boolean", "value": False}),
        ({"kind": "text"}, "雪😀", {"kind": "text", "value": "雪😀"}),
        ({"kind": "bytes"}, b"\x00\xff", {"kind": "bytes", "base64": "AP8="}),
        ({"kind": "date"}, date(2040, 1, 2), {"kind": "date", "value": "2040-01-02"}),
        (
            {"kind": "time", "unit": "us"},
            time(3, 4, 5, 6),
            {"kind": "time", "value": "03:04:05.000006"},
        ),
        (
            {"kind": "timestamp", "unit": "us", "timezone": "utc"},
            datetime(2040, 1, 2, 3, 4, 5, 6, tzinfo=UTC),
            {"kind": "timestamp", "value": "2040-01-02T03:04:05.000006+00:00"},
        ),
        ({"kind": "text"}, None, {"kind": "null"}),
    ],
)
def test_exact_known_native_values(dtype: object, native: object, expected: object) -> None:
    assert wire.native_cell(dtype, native) == expected
    assert wire.checked_cell(dtype, expected) == expected


def test_nested_struct_uses_ordered_values_and_preserves_prototype_names() -> None:
    dtype = {
        "kind": "struct",
        "fields": [
            {"name": "__proto__", "type": {"kind": "text"}},
            {"name": "constructor", "type": {"kind": "list", "item": {"kind": "boolean"}}},
        ],
    }
    expected = {
        "kind": "struct",
        "values": [
            {"kind": "text", "value": "literal"},
            {"kind": "list", "values": [{"kind": "boolean", "value": True}, {"kind": "null"}]},
        ],
    }
    assert (
        wire.native_cell(dtype, {"constructor": [True, None], "__proto__": "literal"}) == expected
    )
    assert wire.checked_cell(dtype, expected) == expected


@pytest.mark.parametrize(
    ("dtype", "native"),
    [
        ({"kind": "integer", "bits": 64, "signed": True}, True),
        ({"kind": "integer", "bits": 8, "signed": False}, 256),
        ({"kind": "integer", "bits": 128, "signed": True}, 2**127),
        ({"kind": "decimal", "precision": 2, "scale": 2}, Decimal("1.20")),
        ({"kind": "decimal", "precision": 12, "scale": 2}, Decimal("1.2")),
        ({"kind": "float", "bits": 32}, 1.0000000000000002),
        ({"kind": "float", "bits": 32}, 1e300),
        ({"kind": "date"}, date.max),
        ({"kind": "date"}, date.min),
        (
            {"kind": "timestamp", "unit": "ms", "timezone": "none"},
            datetime(2040, 1, 2, microsecond=1),
        ),
        ({"kind": "timestamp", "unit": "us", "timezone": "utc"}, datetime(2040, 1, 2)),
        ({"kind": "text"}, "\ud800"),
        ({"kind": "text"}, object()),
        ({"kind": "array", "length": 2, "item": {"kind": "boolean"}}, [True]),
    ],
)
def test_loss_or_wrong_native_shape_refuses(dtype: object, native: object) -> None:
    with pytest.raises(wire.WireError):
        wire.native_cell(dtype, native)


@pytest.mark.parametrize(
    "kind", ["timestamp_ns", "interval", "map", "union", "uuid", "enum", "unknown"]
)
def test_unsupported_types_refuse_before_value_conversion(kind: str) -> None:
    with pytest.raises(wire.WireError) as caught:
        wire.native_cell({"kind": kind}, object())
    assert caught.value.code == "result_type_unsupported"


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":-0}',
        b'{"a":1.0}',
        b'{"a":NaN}',
        b'{"a":9007199254740992}',
        b'{"a":"\\ud800"}',
        b"[] trailing",
    ],
)
def test_nonportable_json_refuses_without_raw_context(raw: bytes) -> None:
    with pytest.raises(wire.WireError) as caught:
        wire.parse(raw, limit=1024)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_canonical_unicode_escaping_and_special_keys() -> None:
    expected = b'{"__proto__":"data","a":"\\n\\u0000\\"","z":"\xe9\x9b\xaa"}'
    assert wire.canonical({"z": "雪", "a": '\n\x00"', "__proto__": "data"}, limit=1024) == expected
    assert wire.parse(expected, limit=1024) == {"z": "雪", "a": '\n\x00"', "__proto__": "data"}


def test_framing_fragmentations_trailing_and_eof_are_explicit() -> None:
    expected = struct.pack("!I", 7) + b'{"a":1}'
    assert wire.frame({"a": 1}, limit=128) == expected
    for split in range(len(expected) + 1):
        reader = wire.FrameDecoder(limit=128)
        reader.feed(expected[:split])
        reader.feed(expected[split:])
        assert reader.finish() == {"a": 1}
        with pytest.raises(wire.WireError):
            reader.feed(b"")
    reader = wire.FrameDecoder(limit=128)
    with pytest.raises(wire.WireError):
        reader.feed(expected + b"x")
    short = wire.FrameDecoder(limit=128)
    short.feed(expected[:-1])
    with pytest.raises(wire.WireError):
        short.finish()
    huge = wire.FrameDecoder(limit=128)
    with pytest.raises(wire.WireError):
        huge.feed(struct.pack("!I", 129))


def test_cell_byte_limit_includes_json_escaping_and_nested_counts() -> None:
    with pytest.raises(wire.WireError):
        wire.native_cell({"kind": "text"}, "\x00" * 11000)
    with pytest.raises(wire.WireError):
        wire.native_cell({"kind": "list", "item": {"kind": "boolean"}}, [True] * 1025)


@pytest.mark.parametrize(
    "cell",
    [
        {"kind": "integer", "value": "01"},
        {"kind": "integer", "value": "1", "extra": True},
        {"kind": "integer", "value": "+1"},
    ],
)
def test_encoded_values_are_canonical_and_closed(cell: object) -> None:
    with pytest.raises(wire.WireError):
        wire.checked_cell({"kind": "integer", "bits": 64, "signed": True}, cell)


def test_rejected_output_chunk_cannot_be_discarded_then_report_success() -> None:
    decoder = wire.FrameDecoder(limit=32)
    decoder.feed(wire.frame({"v": 1}, limit=32))
    with pytest.raises(wire.WireError):
        decoder.feed(b"X" * 33)
    with pytest.raises(wire.WireError):
        decoder.finish()


def test_unsupported_decimal_never_reaches_unbounded_formatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original = builtins.format

    def guard(value: object, format_spec: str = "") -> str:
        if type(value) is Decimal:
            raise AssertionError("unsupported decimal reached formatting")
        return original(value, format_spec)

    value = Decimal("1E+1000000")
    monkeypatch.setattr(builtins, "format", guard)
    with pytest.raises(wire.WireError):
        wire.native_cell({"kind": "decimal", "precision": 38, "scale": 0}, value)


def test_impossible_object_cardinality_refuses_before_key_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(_values: object) -> bool:
        raise AssertionError("oversized object reached all-key traversal")

    monkeypatch.setattr(wire, "all", forbidden, raising=False)
    with pytest.raises(wire.WireError):
        wire.query({str(index): None for index in range(100)})
