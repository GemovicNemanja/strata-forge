"""Closed stdlib wire for the fixed native dataset transformation profile."""

from __future__ import annotations

import hashlib
import re
from typing import cast

if __package__:
    from strata_forge.datasets import _sql_wire as sql
else:
    import _sql_wire as sql

PROFILE = "dataset_transform_arm64_v1"
MAX_BYTES = 67108864
MAX_TEXT_BYTES = 16777216
MAX_ROWS = 1000000
MAX_COLUMNS = 128
SOURCES = ("_sql_wire.py", "_sql_worker.py", "_dataset_wire.py", "_dataset_worker.py")
type JSON = sql.JSON


def need(value: bool, code: str = "invalid_spec") -> None:
    if not value:
        raise sql.WireError(code)


def obj(value: object, keys: set[str]) -> dict[str, JSON]:
    need(type(value) is dict)
    data = cast("dict[str, JSON]", value)
    need(len(data) == len(keys) and all(type(k) is str for k in data) and data.keys() == keys)
    return data


def count(value: object, low: int, high: int) -> int:
    need(type(value) is int and low <= value <= high)
    return cast("int", value)


def text(value: object, maximum: int) -> str:
    need(type(value) is str)
    value = cast("str", value)
    need(len(value) <= maximum and all(not 0xD800 <= ord(c) <= 0xDFFF for c in value))
    need(len(value.encode("utf-8")) <= maximum)
    return value


def sha(value: object) -> str:
    value = text(value, 64)
    need(re.fullmatch(r"[0-9a-f]{64}", value) is not None)
    return value


def alias(value: object) -> str:
    value = text(value, 32)
    need(
        re.fullmatch(r"[a-z][a-z0-9_]{0,31}", value) is not None and not value.startswith("strata_")
    )
    return value


def columns(value: object) -> list[JSON]:
    need(type(value) is list and 1 <= len(cast("list[object]", value)) <= MAX_COLUMNS)
    result: list[JSON] = []
    names: set[str] = set()
    for item in cast("list[object]", value):
        col = obj(item, {"name", "type", "nullable"})
        name = text(col["name"], 256)
        need(bool(name) and name.casefold() not in names and type(col["nullable"]) is bool)
        names.add(name.casefold())
        dtype = sql.logical_type(col["type"])
        need(
            dtype["kind"] in {"text", "boolean", "integer", "float", "decimal", "date", "timestamp"}
        )
        if dtype["kind"] == "integer":
            need(dtype == {"kind": "integer", "bits": 64, "signed": True})
        if dtype["kind"] == "float":
            need(dtype == {"kind": "float", "bits": 64})
        if dtype["kind"] == "timestamp":
            need(dtype["unit"] == "us")
        result.append({"name": name, "type": dtype, "nullable": col["nullable"]})
    sql.canonical(result, limit=sql.MAX_MANIFEST)
    return result


def input_manifest(value: object) -> dict[str, JSON]:
    data = obj(value, {"version", "format", "parts"})
    need(type(data["version"]) is int and data["version"] == 1)
    fmt = text(data["format"], 8)
    need(fmt in {"parquet", "csv", "jsonl"})
    need(type(data["parts"]) is list)
    raw = cast("list[object]", data["parts"])
    need(1 <= len(raw) <= (32 if fmt == "parquet" else 1))
    parts: list[JSON] = []
    total = 0
    for index, item in enumerate(raw):
        part = obj(item, {"ordinal", "sha256", "size_bytes"})
        need(count(part["ordinal"], 0, 31) == index)
        size = count(part["size_bytes"], 1, MAX_BYTES if fmt == "parquet" else MAX_TEXT_BYTES)
        total += size
        parts.append({"ordinal": index, "sha256": sha(part["sha256"]), "size_bytes": size})
    need(total <= (MAX_BYTES if fmt == "parquet" else MAX_TEXT_BYTES), "source_limit")
    result: dict[str, JSON] = {"version": 1, "format": fmt, "parts": parts}
    sql.canonical(result, limit=sql.MAX_MANIFEST)
    return result


def convert_spec(value: object) -> dict[str, JSON]:
    data = obj(value, {"version", "format", "schema", "input_manifest_sha256", "timeout_ms"})
    need(type(data["version"]) is int and data["version"] == 1)
    fmt = text(data["format"], 8)
    need(fmt in {"parquet", "csv", "jsonl"})
    schema = None if fmt == "parquet" else columns(data["schema"])
    need(fmt != "parquet" or data["schema"] is None)
    return {
        "version": 1,
        "format": fmt,
        "schema": schema,
        "input_manifest_sha256": sha(data["input_manifest_sha256"]),
        "timeout_ms": count(data["timeout_ms"], 100, 10000),
    }


def query_spec(value: object) -> dict[str, JSON]:
    data = obj(value, {"version", "sql", "inputs_sha256", "timeout_ms", "max_result_rows"})
    need(type(data["version"]) is int and data["version"] == 1)
    query = text(data["sql"], 32768)
    need(bool(query))
    return {
        "version": 1,
        "sql": query,
        "inputs_sha256": sha(data["inputs_sha256"]),
        "timeout_ms": count(data["timeout_ms"], 100, 10000),
        "max_result_rows": count(data["max_result_rows"], 1, 1000),
    }


def named_inputs(value: object) -> list[JSON]:
    need(type(value) is list and 1 <= len(cast("list[object]", value)) <= 4)
    result: list[JSON] = []
    names: set[str] = set()
    sizes = rows = parts = 0
    for item in cast("list[object]", value):
        data = obj(item, {"alias", "manifest"})
        name = alias(data["alias"])
        need(name not in names)
        names.add(name)
        manifest = sql.manifest(data["manifest"])
        parts += cast("int", manifest["part_count"])
        rows += cast("int", manifest["expected_rows"])
        sizes += sum(
            cast("int", cast("dict[str, JSON]", p)["size_bytes"])
            for p in cast("list[JSON]", manifest["parts"])
        )
        need(sizes <= MAX_BYTES and rows <= MAX_ROWS and parts <= 32, "source_limit")
        result.append({"alias": name, "manifest": manifest})
    sql.canonical(result, limit=sql.MAX_MANIFEST)
    return result


def installation(value: object) -> dict[str, JSON]:
    data = obj(
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
    need(type(data["profile"]) is str and data["profile"] == PROFILE)
    checked = sql.installation({**data, "profile": sql.PROFILE})
    checked["profile"] = PROFILE
    return checked


def source_bundle(files: object) -> str:
    need(type(files) is dict)
    files = cast("dict[str, bytes]", files)
    need(
        len(files) == len(SOURCES)
        and all(type(k) is str for k in files)
        and set(files) == set(SOURCES)
    )
    values: dict[str, JSON] = {}
    for name in SOURCES:
        raw = files[name]
        need(type(raw) is bytes and 0 < len(raw) <= 131072, "profile_refused")
        values[name] = hashlib.sha256(raw).hexdigest()
    return sql.digest({"profile": PROFILE, "sources": values})
