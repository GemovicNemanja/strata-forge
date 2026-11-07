"""Bounded stdlib-only values shared by the local SQL parent and fixed worker image."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import struct
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import cast
from zoneinfo import ZoneInfo

type JSON = None | bool | int | str | list[JSON] | dict[str, JSON]
SAFE_INTEGER = 9_007_199_254_740_991
MAX_MANIFEST = 131072
MAX_REQUEST = 196608
MAX_RESULT = 2097152
MAX_CELL = 65536
PROFILE = "sql_local_arm64_v1"
CODES = frozenset(
    {
        "invalid_spec",
        "source_mismatch",
        "source_limit",
        "query_refused",
        "query_failed",
        "result_type_unsupported",
        "output_limit",
        "deadline",
        "resource_limit",
        "busy",
        "unavailable",
        "profile_refused",
        "worker_failed",
        "cleanup_unresolved",
    }
)
_HEX = re.compile(r"[a-f0-9]+\Z")
_INT = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")


class WireError(Exception):
    """Internal worker protocol refusal, never an exposed native diagnostic."""

    def __init__(self, code: object = "unavailable") -> None:
        self.code = code if type(code) is str and code in CODES else "unavailable"
        super().__init__("sql_unavailable")


def _need(test: bool, code: str = "invalid_spec") -> None:
    if not test:
        raise WireError(code)


def _text(value: object, limit: int, *, empty: bool = False) -> str:
    _need(type(value) is str)
    value = cast("str", value)
    _need(len(value) <= limit and (empty or bool(value)))
    _need(not any(0xD800 <= ord(c) <= 0xDFFF for c in value))
    _need(len(value.encode("utf-8")) <= limit)
    return value


def _count(value: object, low: int, high: int) -> int:
    _need(type(value) is int)
    value = cast("int", value)
    _need(low <= value <= high)
    return value


def _sha(value: object, length: int = 64) -> str:
    text = _text(value, length)
    _need(len(text) == length and _HEX.fullmatch(text) is not None)
    return text


def _obj(value: object, keys: set[str], optional: set[str] | None = None) -> dict[str, JSON]:
    _need(type(value) is dict)
    source = cast("dict[object, object]", value)
    _need(len(keys) <= len(source) <= len(keys | (optional or set())))
    _need(all(type(k) is str for k in source))
    _need(keys <= source.keys() <= keys | (optional or set()))
    return cast("dict[str, JSON]", source)


def _items(value: object, low: int, high: int) -> list[JSON]:
    _need(type(value) is list)
    result = cast("list[JSON]", value)
    _need(low <= len(result) <= high)
    return result


def _portable(
    value: object, depth: int, ancestors: set[int], budget: list[int], limit: int
) -> None:
    _need(depth <= 24)
    if value is None or type(value) is bool:
        budget[0] += 5 if value is False else 4
    elif type(value) is int:
        _need(abs(value) <= SAFE_INTEGER)
        budget[0] += len(str(value))
    elif type(value) is str:
        text = _text(value, limit, empty=True)
        budget[0] += (
            len(text.encode("utf-8"))
            + 2
            + sum(1 if c in '\\"\b\t\n\f\r' else 5 if ord(c) < 32 else 0 for c in text)
        )
    elif type(value) in (list, dict):
        identity = id(value)
        _need(identity not in ancestors)
        ancestors.add(identity)
        if type(value) is list:
            children = cast("list[object]", value)
            _need(len(children) <= limit)
            budget[0] += 2 + max(0, len(children) - 1)
            for child in children:
                _portable(child, depth + 1, ancestors, budget, limit)
        else:
            fields = cast("dict[object, object]", value)
            _need(len(fields) <= limit)
            budget[0] += 2 + max(0, len(fields) - 1) + len(fields)
            for key, child in fields.items():
                _need(type(key) is str)
                _need(all(32 <= ord(c) <= 126 for c in cast("str", key)))
                _portable(key, depth + 1, ancestors, budget, limit)
                _portable(child, depth + 1, ancestors, budget, limit)
        ancestors.remove(identity)
    else:
        raise WireError()
    _need(budget[0] <= limit, "output_limit")


def canonical(value: object, *, limit: int) -> bytes:
    """Encode closed portable values without implicit string conversion or normalization."""
    _portable(value, 0, set(), [0], limit)
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    _need(len(encoded) <= limit, "output_limit")
    return encoded


def _integer(text: str) -> int:
    _need(len(text) <= 17 and _INT.fullmatch(text) is not None)
    result = int(text)
    _need(abs(result) <= SAFE_INTEGER)
    return result


def _no_float(_text_value: str) -> JSON:
    raise WireError("invalid_spec")


def _object(pairs: list[tuple[str, JSON]]) -> dict[str, JSON]:
    result: dict[str, JSON] = {}
    for key, value in pairs:
        _need(key not in result)
        result[key] = value
    return result


def parse(raw: object, *, limit: int) -> JSON:
    """Reject duplicate fields, unsafe numbers and nonportable values before projection."""
    _need(type(raw) is bytes and len(raw) <= limit)
    result: JSON | None = None
    failed = False
    try:
        result = cast(
            "JSON",
            json.loads(
                cast("bytes", raw).decode("utf-8"),
                object_pairs_hook=_object,
                parse_int=_integer,
                parse_float=_no_float,
                parse_constant=_no_float,
            ),
        )
        canonical(result, limit=limit)
    except ValueError, UnicodeError, RecursionError, WireError:
        failed = True
    if failed:
        raise WireError("invalid_spec")
    return result


def digest(value: object, *, limit: int = MAX_MANIFEST) -> str:
    return hashlib.sha256(canonical(value, limit=limit)).hexdigest()


def _version(value: object) -> None:
    _need(type(value) is int and value == 1)


def query(value: object) -> dict[str, JSON]:
    """Validate one query value without parsing or executing SQL in the parent."""
    data = _obj(
        value, {"version", "sql", "input_manifest_sha256"}, {"max_result_rows", "timeout_ms"}
    )
    _version(data["version"])
    result: dict[str, JSON] = {
        "version": 1,
        "sql": _text(data["sql"], 32768),
        "input_manifest_sha256": _sha(data["input_manifest_sha256"]),
        "max_result_rows": _count(data.get("max_result_rows", 1000), 1, 10000),
        "timeout_ms": _count(data.get("timeout_ms", 5000), 100, 10000),
    }
    canonical(result, limit=MAX_REQUEST)
    return result


def logical_type(
    value: object, *, depth: int = 0, nodes: list[int] | None = None
) -> dict[str, JSON]:
    """Close the lossless first result-type profile, including bounded nested shapes."""
    nodes = [0] if nodes is None else nodes
    nodes[0] += 1
    _need(depth <= 8 and nodes[0] <= 1024, "source_limit")
    initial = _obj(
        value,
        {"kind"},
        {"bits", "signed", "precision", "scale", "unit", "timezone", "item", "length", "fields"},
    )
    kind = _text(initial["kind"], 16)
    result: dict[str, JSON]
    if kind in ("null", "boolean", "text", "bytes", "date"):
        _obj(value, {"kind"})
        result = {"kind": kind}
    elif kind == "integer":
        data = _obj(value, {"kind", "bits", "signed"})
        bits = _count(data["bits"], 8, 128)
        _need(bits in (8, 16, 32, 64, 128) and type(data["signed"]) is bool)
        result = {"kind": kind, "bits": bits, "signed": data["signed"]}
    elif kind == "float":
        data = _obj(value, {"kind", "bits"})
        bits = _count(data["bits"], 32, 64)
        _need(bits in (32, 64))
        result = {"kind": kind, "bits": bits}
    elif kind == "decimal":
        data = _obj(value, {"kind", "precision", "scale"})
        precision = _count(data["precision"], 1, 38)
        result = {
            "kind": kind,
            "precision": precision,
            "scale": _count(data["scale"], 0, precision),
        }
    elif kind == "time":
        data = _obj(value, {"kind", "unit"})
        _need(type(data["unit"]) is str and data["unit"] == "us")
        result = {"kind": kind, "unit": "us"}
    elif kind == "timestamp":
        data = _obj(value, {"kind", "unit", "timezone"})
        _need(type(data["unit"]) is str and data["unit"] in ("s", "ms", "us"))
        _need(type(data["timezone"]) is str and data["timezone"] in ("none", "utc"))
        result = {"kind": kind, "unit": data["unit"], "timezone": data["timezone"]}
    elif kind in ("list", "array"):
        data = _obj(value, {"kind", "item"} | ({"length"} if kind == "array" else set()))
        result = {"kind": kind, "item": logical_type(data["item"], depth=depth + 1, nodes=nodes)}
        if kind == "array":
            result["length"] = _count(data["length"], 1, 1024)
    elif kind == "struct":
        data = _obj(value, {"kind", "fields"})
        fields: list[JSON] = []
        names: set[str] = set()
        for item in _items(data["fields"], 1, 1024):
            entry = _obj(item, {"name", "type"})
            name = _text(entry["name"], 256)
            _need(name.casefold() not in names)
            names.add(name.casefold())
            fields.append(
                {"name": name, "type": logical_type(entry["type"], depth=depth + 1, nodes=nodes)}
            )
        result = {"kind": kind, "fields": fields}
    else:
        raise WireError("result_type_unsupported")
    return result


def columns(value: object, *, source: bool) -> list[JSON]:
    result: list[JSON] = []
    names: set[str] = set()
    nodes = [0]
    for item in _items(value, 1, 128):
        entry = _obj(item, {"name", "type"})
        name = _text(entry["name"], 256)
        _need(not source or name.casefold() not in names)
        names.add(name.casefold())
        result.append({"name": name, "type": logical_type(entry["type"], nodes=nodes)})
    return result


def manifest(value: object) -> dict[str, JSON]:
    """Bind ordered complete local bytes/schema and descriptive source assertions."""
    data = _obj(
        value,
        {
            "version",
            "source_assertion",
            "parts",
            "schema",
            "schema_sha256",
            "declared_population",
            "expected_rows",
            "part_count",
        },
    )
    _version(data["version"])
    origin = _obj(data["source_assertion"], {"system", "name", "config", "split", "revision"})
    revision = _obj(origin["revision"], {"algorithm", "value"})
    algorithm = _text(revision["algorithm"], 16)
    _need(algorithm in ("git_sha1", "sha256"))
    declared: dict[str, JSON] = {
        key: _text(origin[key], 256) for key in ("system", "name", "config", "split")
    }
    declared["revision"] = {
        "algorithm": algorithm,
        "value": _sha(revision["value"], 40 if algorithm == "git_sha1" else 64),
    }
    parts: list[JSON] = []
    size, rows = 0, 0
    for ordinal, item in enumerate(_items(data["parts"], 1, 32)):
        part = _obj(item, {"ordinal", "sha256", "size_bytes", "expected_rows"})
        _need(_count(part["ordinal"], 0, 31) == ordinal)
        part_size = _count(part["size_bytes"], 1, 67108864)
        part_rows = _count(part["expected_rows"], 0, 1000000)
        size += part_size
        rows += part_rows
        _need(size <= 67108864 and rows <= 1000000, "source_limit")
        parts.append(
            {
                "ordinal": ordinal,
                "sha256": _sha(part["sha256"]),
                "size_bytes": part_size,
                "expected_rows": part_rows,
            }
        )
    schema = columns(data["schema"], source=True)
    schema_sha = _sha(data["schema_sha256"])
    _need(digest(schema) == schema_sha, "source_mismatch")
    _need(
        type(data["declared_population"]) is str
        and data["declared_population"] == "complete_named_split"
    )
    _need(_count(data["expected_rows"], 0, 1000000) == rows)
    _need(_count(data["part_count"], 1, 32) == len(parts))
    result: dict[str, JSON] = {
        "version": 1,
        "source_assertion": declared,
        "parts": parts,
        "schema": schema,
        "schema_sha256": schema_sha,
        "declared_population": "complete_named_split",
        "expected_rows": rows,
        "part_count": len(parts),
    }
    canonical(result, limit=MAX_MANIFEST)
    return result


def installation(value: object) -> dict[str, JSON]:
    data = _obj(
        value,
        {
            "version",
            "profile",
            "docker_executable",
            "engine_socket",
            "image_id",
            "worker_bundle_sha256",
        },
    )
    _version(data["version"])
    _need(type(data["profile"]) is str and data["profile"] == PROFILE)
    for key in ("docker_executable", "engine_socket"):
        path = _text(data[key], 4096)
        _need(path.startswith("/") and all(ord(c) >= 32 and ord(c) != 127 for c in path))
        _need(all(part not in ("", ".", "..") for part in path[1:].split("/")))
    image = _text(data["image_id"], 71)
    _need(image.startswith("sha256:") and _sha(image[7:]) == image[7:])
    _sha(data["worker_bundle_sha256"])
    canonical(data, limit=16384)
    return data.copy()


def _native_cell(dtype: dict[str, JSON], value: object, depth: int) -> dict[str, JSON]:
    _need(depth <= 8, "output_limit")
    if value is None:
        return {"kind": "null"}
    kind = dtype["kind"]
    if kind == "boolean":
        _need(type(value) is bool, "result_type_unsupported")
        result: dict[str, JSON] = {"kind": kind, "value": cast("bool", value)}
    elif kind == "integer":
        _need(type(value) is int, "result_type_unsupported")
        number = cast("int", value)
        bits = cast("int", dtype["bits"])
        signed = dtype["signed"] is True
        _need(
            -(1 << (bits - 1)) <= number < 1 << (bits - 1) if signed else 0 <= number < 1 << bits,
            "result_type_unsupported",
        )
        result = {"kind": kind, "value": str(number)}
    elif kind == "decimal":
        _need(type(value) is Decimal, "result_type_unsupported")
        decimal = cast("Decimal", value)
        _need(decimal.is_finite(), "result_type_unsupported")
        scale = cast("int", dtype["scale"])
        precision = cast("int", dtype["precision"])
        _need(
            decimal.as_tuple().exponent == -scale and len(decimal.as_tuple().digits) <= precision,
            "result_type_unsupported",
        )
        text = format(decimal, "f")
        result = {"kind": kind, "value": text}
    elif kind == "float":
        _need(type(value) is float, "result_type_unsupported")
        number = cast("float", value)
        if dtype["bits"] == 32 and math.isfinite(number):
            _need(abs(number) <= 3.4028234663852886e38, "result_type_unsupported")
            _need(
                struct.unpack("!f", struct.pack("!f", number))[0].hex() == number.hex(),
                "result_type_unsupported",
            )
        text = (
            "nan"
            if math.isnan(number)
            else "positive_infinity"
            if number == math.inf
            else "negative_infinity"
            if number == -math.inf
            else number.hex()
        )
        result = {"kind": kind, "value": text}
    elif kind == "text":
        result = {"kind": kind, "value": _text(value, MAX_CELL, empty=True)}
    elif kind == "bytes":
        _need(type(value) is bytes and len(value) <= MAX_CELL, "output_limit")
        result = {"kind": kind, "base64": base64.b64encode(cast("bytes", value)).decode("ascii")}
    elif kind == "date":
        _need(type(value) is date, "result_type_unsupported")
        day = cast("date", value)
        _need(day not in (date.min, date.max), "result_type_unsupported")
        result = {"kind": kind, "value": day.isoformat()}
    elif kind == "time":
        _need(type(value) is time, "result_type_unsupported")
        clock = cast("time", value)
        _need(clock.tzinfo is None, "result_type_unsupported")
        result = {"kind": kind, "value": clock.isoformat(timespec="microseconds")}
    elif kind == "timestamp":
        _need(type(value) is datetime, "result_type_unsupported")
        moment = cast("datetime", value)
        _need(
            moment.replace(tzinfo=None) not in (datetime.min, datetime.max),
            "result_type_unsupported",
        )
        if dtype["timezone"] == "utc":
            _need(type(moment.tzinfo) in (timezone, ZoneInfo), "result_type_unsupported")
            _need(moment.utcoffset() == timedelta(0), "result_type_unsupported")
        else:
            _need(moment.tzinfo is None, "result_type_unsupported")
        unit = dtype["unit"]
        _need(
            unit == "us" or moment.microsecond % (1000 if unit == "ms" else 1000000) == 0,
            "result_type_unsupported",
        )
        result = {"kind": kind, "value": moment.isoformat(timespec="microseconds")}
    elif kind in ("list", "array"):
        _need(type(value) in (list, tuple), "result_type_unsupported")
        members = cast("list[object] | tuple[object, ...]", value)
        _need(len(members) <= 1024, "output_limit")
        _need(kind != "array" or len(members) == dtype["length"], "result_type_unsupported")
        child = cast("dict[str, JSON]", dtype["item"])
        result = {
            "kind": kind,
            "values": [_native_cell(child, item, depth + 1) for item in members],
        }
    elif kind == "struct":
        _need(type(value) is dict, "result_type_unsupported")
        fields = cast("dict[object, object]", value)
        _need(all(type(key) is str for key in fields), "result_type_unsupported")
        definitions = cast("list[dict[str, JSON]]", dtype["fields"])
        _need(len(fields) == len(definitions), "result_type_unsupported")
        values: list[JSON] = []
        for definition in definitions:
            name = cast("str", definition["name"])
            _need(name in fields, "result_type_unsupported")
            values.append(
                _native_cell(cast("dict[str, JSON]", definition["type"]), fields[name], depth + 1)
            )
        result = {"kind": kind, "values": values}
    else:
        raise WireError("result_type_unsupported")
    canonical(result, limit=MAX_CELL)
    return result


def native_cell(dtype: object, value: object) -> dict[str, JSON]:
    """Encode only the admitted exact builtin/native values, without generic conversion."""
    return _native_cell(logical_type(dtype), value, 0)


def validate_cell(dtype: object, value: object) -> dict[str, JSON]:
    """Revalidate encoded cells by lossless roundtrip through their closed native profile."""
    shape = logical_type(dtype)
    cell = _obj(value, {"kind"}, {"value", "base64", "values"})
    _text(cell["kind"], 16)
    if cell["kind"] == "null":
        _obj(cell, {"kind"})
        return {"kind": "null"}
    _need(cell["kind"] == shape["kind"], "result_type_unsupported")
    kind = shape["kind"]
    native: object
    if kind == "boolean":
        _obj(cell, {"kind", "value"})
        native = cell["value"]
    elif kind in ("integer", "decimal", "float", "date", "time", "timestamp", "text"):
        _obj(cell, {"kind", "value"})
        text = _text(cell["value"], MAX_CELL, empty=kind == "text")
        if kind == "integer":
            _need(len(text) <= 40 and _INT.fullmatch(text) is not None)
            native = int(text)
        elif kind == "decimal":
            _need(len(text) <= 42 and re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", text) is not None)
            native = Decimal(text)
        elif kind == "float":
            native = {
                "nan": math.nan,
                "positive_infinity": math.inf,
                "negative_infinity": -math.inf,
            }.get(text)
            if native is None:
                native = float.fromhex(text)
        elif kind == "date":
            native = date.fromisoformat(text)
        elif kind == "time":
            native = time.fromisoformat(text)
        elif kind == "timestamp":
            native = datetime.fromisoformat(text)
        else:
            native = text
    elif kind == "bytes":
        _obj(cell, {"kind", "base64"})
        text = _text(cell["base64"], MAX_CELL, empty=True)
        native = base64.b64decode(text, validate=True)
    elif kind in ("list", "array", "struct"):
        _obj(cell, {"kind", "values"})
        values = _items(cell["values"], 0, 1024)
        if kind == "struct":
            definitions = cast("list[dict[str, JSON]]", shape["fields"])
            _need(len(values) == len(definitions))
            expected: dict[str, JSON] = {
                "kind": kind,
                "values": [
                    validate_cell(d["type"], v) for d, v in zip(definitions, values, strict=True)
                ],
            }
        else:
            _need(kind != "array" or len(values) == shape["length"])
            expected = {
                "kind": kind,
                "values": [validate_cell(shape["item"], item) for item in values],
            }
        _need(canonical(expected, limit=MAX_CELL) == canonical(cell, limit=MAX_CELL))
        return expected
    else:
        raise WireError("result_type_unsupported")
    expected = native_cell(shape, native)
    _need(canonical(expected, limit=MAX_CELL) == canonical(cell, limit=MAX_CELL))
    return expected


def checked_cell(dtype: object, value: object) -> dict[str, JSON]:
    result: dict[str, JSON] | None = None
    try:
        result = validate_cell(dtype, value)
    except ValueError, OverflowError, binascii.Error, WireError:
        result = None
    if result is None:
        raise WireError("result_type_unsupported")
    return result


def frame(value: object, *, limit: int) -> bytes:
    """Encode one bounded length-prefixed JSON frame, never a general RPC envelope."""
    body = canonical(value, limit=limit)
    return struct.pack("!I", len(body)) + body


class FrameDecoder:
    """One bounded frame with explicit EOF and no silent trailing-data acceptance."""

    def __init__(self, *, limit: int) -> None:
        _count(limit, 1, MAX_RESULT)
        self.limit = limit
        self.buffer = bytearray()
        self.expected: int | None = None
        self.ended = False

    def feed(self, chunk: object) -> None:
        code: str | None = None
        try:
            self._feed(chunk)
        except WireError as refusal:
            code = refusal.code
        if code is not None:
            self.ended = True
            self.buffer.clear()
            self.expected = None
            raise WireError(code)

    def _feed(self, chunk: object) -> None:
        _need(not self.ended and type(chunk) is bytes)
        data = cast("bytes", chunk)
        _need(len(data) <= 65536 and len(self.buffer) + len(data) <= self.limit + 4, "output_limit")
        self.buffer.extend(data)
        if self.expected is None and len(self.buffer) >= 4:
            expected = cast("int", struct.unpack("!I", self.buffer[:4])[0])
            _need(0 < expected <= self.limit, "output_limit")
            self.expected = expected
        if self.expected is not None:
            _need(len(self.buffer) <= self.expected + 4)

    def finish(self) -> JSON:
        _need(not self.ended)
        self.ended = True
        _need(self.expected is not None and len(self.buffer) == self.expected + 4)
        return parse(bytes(self.buffer[4:]), limit=self.limit)


def result_table(value: object) -> dict[str, JSON]:
    """Validate a complete bounded typed table, never an execution or access attestation."""
    data = _obj(
        value,
        {
            "version",
            "columns",
            "rows",
            "returned_rows",
            "result_total_rows",
            "result_truncated",
            "coverage",
        },
    )
    _version(data["version"])
    schema = columns(data["columns"], source=False)
    values = _items(data["rows"], 0, 10000)
    rows: list[JSON] = []
    for item in values:
        row = _items(item, len(schema), len(schema))
        rows.append(
            [
                checked_cell(cast("dict[str, JSON]", column)["type"], cell)
                for column, cell in zip(schema, row, strict=True)
            ]
        )
    _need(_count(data["returned_rows"], 0, 10000) == len(rows))
    _need(type(data["result_truncated"]) is bool)
    truncated = data["result_truncated"] is True
    if truncated:
        _need(bool(rows) and data["result_total_rows"] is None)
    else:
        _need(_count(data["result_total_rows"], 0, 10000) == len(rows))
    coverage = _obj(
        data["coverage"],
        {
            "input_manifest_materialized_complete",
            "upstream_inventory_verified",
            "query_scan_coverage",
            "result_state",
        },
    )
    _need(coverage["input_manifest_materialized_complete"] is True)
    _need(coverage["upstream_inventory_verified"] is False)
    _need(
        type(coverage["query_scan_coverage"]) is str
        and coverage["query_scan_coverage"] == "not_measured"
    )
    _need(
        type(coverage["result_state"]) is str
        and coverage["result_state"] == ("row_prefix" if truncated else "exhausted")
    )
    result: dict[str, JSON] = {
        "version": 1,
        "columns": schema,
        "rows": rows,
        "returned_rows": len(rows),
        "result_total_rows": None if truncated else len(rows),
        "result_truncated": truncated,
        "coverage": coverage.copy(),
    }
    canonical(result, limit=MAX_RESULT)
    return result


def source_bundle(files: object) -> str:
    """Hash exactly the two fixed image sources; descriptive consistency, not authority."""
    _need(type(files) is dict, "profile_refused")
    sources = cast("dict[object, object]", files)
    _need(len(sources) == 2 and all(type(key) is str for key in sources), "profile_refused")
    _need(set(sources) == {"_sql_worker.py", "_sql_wire.py"}, "profile_refused")
    hashed = hashlib.sha256()
    for name in ("_sql_worker.py", "_sql_wire.py"):
        raw = sources[name]
        _need(type(raw) is bytes and len(raw) <= 131072, "profile_refused")
        hashed.update(name.encode("ascii") + b"\0" + hashlib.sha256(cast("bytes", raw)).digest())
    return hashed.hexdigest()
