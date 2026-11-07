"""Fixed image-only local SQL worker; importing this file executes no native SQL."""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import resource
import stat
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from strata_forge.datasets import _sql_wire as wire
else:
    if __package__:
        wire = importlib.import_module("strata_forge.datasets._sql_wire")
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        wire = importlib.import_module("_sql_wire")


class _NativeType(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def children(self) -> list[tuple[str, object]]: ...


class _StatementKind(Protocol):
    @property
    def name(self) -> str: ...


class _Statement(Protocol):
    @property
    def type(self) -> _StatementKind: ...


class _Connection(Protocol):
    @property
    def description(self) -> list[tuple[object, ...]]: ...
    def execute(self, query: str, parameters: object = None) -> _Connection: ...
    def fetchone(self) -> tuple[object, ...] | None: ...
    def fetchmany(self, size: int = 1) -> list[tuple[object, ...]]: ...
    def extract_statements(self, query: str) -> list[_Statement]: ...
    def close(self) -> None: ...


class _Engine(Protocol):
    @property
    def __version__(self) -> str: ...
    def connect(self, database: str, *, config: dict[str, object]) -> _Connection: ...


def _need(ok: bool, code: str = "profile_refused") -> None:
    if not ok:
        raise wire.WireError(code)


def bundle_sha256() -> str:
    """Hash only the fixed worker/image codec files, never caller-selected paths."""
    sources: dict[str, bytes] = {}
    for name in ("_sql_worker.py", "_sql_wire.py"):
        descriptor = os.open(Path(__file__).parent / name, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(descriptor)
            _need(stat.S_ISREG(info.st_mode) and info.st_size <= 131072)
            with os.fdopen(os.dup(descriptor), "rb") as source:
                raw = source.read(131073)
            _need(len(raw) <= 131072)
            sources[name] = raw
        finally:
            os.close(descriptor)
    return wire.source_bundle(sources)


def expected_ready(operation_id: str, bundle: str) -> dict[str, wire.JSON]:
    return {
        "version": 1,
        "kind": "ready",
        "operation_id": operation_id,
        "profile": wire.PROFILE,
        "bundle_sha256": bundle,
        "python": "3.14.5",
        "duckdb": "2.0.0",
        "pytz": "2026.6",
        "uid": 65532,
        "gid": 65532,
        "memory_bytes": 268435456,
        "memory_swap_bytes": 0,
        "pids": 32,
        "cpu_quota": 100000,
        "cpu_period": 100000,
        "nofile": 64,
        "file_bytes": 33554432,
        "core_bytes": 0,
        "seccomp": 2,
        "no_new_privs": 1,
        "cap_eff": "0000000000000000",
        "python_enable_replacements": False,
        "python_scan_all_frames": False,
    }


def _read_kernel(name: str) -> str:
    with (Path("/sys/fs/cgroup") / name).open("rb") as source:
        raw = source.read(128)
    _need(len(raw) < 128)
    return raw.decode("ascii").strip()


def _profile() -> None:
    _need(sys.version_info[:3] == (3, 14, 5) and sys.platform == "linux")
    _need(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1)
    _need(os.getuid() == 65532 and os.getgid() == 65532 and os.getpid() == 1)
    for key, pair in (
        (resource.RLIMIT_NOFILE, (64, 64)),
        (resource.RLIMIT_FSIZE, (33554432, 33554432)),
        (resource.RLIMIT_CORE, (0, 0)),
    ):
        _need(resource.getrlimit(key) == pair)
    _need(_read_kernel("memory.max") == "268435456")
    _need(_read_kernel("memory.swap.max") == "0")
    _need(_read_kernel("pids.max") == "32")
    _need(_read_kernel("cpu.max") == "100000 100000")
    with Path("/proc/self/status").open("rb") as source:
        status = source.read(16384)
    _need(len(status) < 16384)
    fields = dict(line.split(":", 1) for line in status.decode("ascii").splitlines() if ":" in line)
    _need(fields.get("Seccomp", "").strip() == "2")
    _need(fields.get("NoNewPrivs", "").strip() == "1")
    _need(fields.get("CapEff", "").strip() == "0000000000000000")


def _configured(engine: _Engine) -> _Connection:
    conn = engine.connect(
        ":memory:",
        config={
            "threads": 1,
            "async_threads": 1,
            "external_threads": 1,
            "memory_limit": "96MB",
            "max_temp_directory_size": "32MB",
            "temp_directory": tempfile.mkdtemp(prefix="duckdb-"),
            "allow_unsigned_extensions": False,
            "allow_community_extensions": False,
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
        },
    )
    for sql in (
        "SET enable_logging=false",
        "SET enable_progress_bar=false",
        "SET python_enable_replacements=false",
        "SET python_scan_all_frames=false",
        "SET allow_persistent_secrets=false",
        "SET allow_extension_repositories='forbidden'",
        "SET TimeZone='UTC'",
    ):
        conn.execute(sql)
    _settings(conn, locked=False)
    return conn


def _settings(conn: _Connection, *, locked: bool) -> None:
    names = [
        "python_enable_replacements",
        "python_scan_all_frames",
        "allow_persistent_secrets",
        "allow_extension_repositories",
        "TimeZone",
        "enable_logging",
        "enable_progress_bar",
        "allow_unsigned_extensions",
        "allow_community_extensions",
        "autoinstall_known_extensions",
        "autoload_known_extensions",
    ]
    expected = dict.fromkeys(names, "false")
    expected.update(
        {"allow_extension_repositories": "forbidden", "TimeZone": "UTC", "enable_logging": "0"}
    )
    if locked:
        names += ["enable_external_access", "lock_configuration"]
        expected.update({"enable_external_access": "false", "lock_configuration": "true"})
    found = conn.execute(
        "SELECT name,value FROM duckdb_settings() WHERE name IN (SELECT unnest(?))", [names]
    ).fetchmany(32)
    actual: dict[str, str] = {}
    for row in found:
        _need(len(row) == 2 and type(row[0]) is str and type(row[1]) is str)
        actual[cast("str", row[0])] = cast("str", row[1])
    _need(actual == expected)


def _native_type(value: object, depth: int = 0) -> dict[str, wire.JSON]:
    _need(depth <= 8, "result_type_unsupported")
    native = cast("_NativeType", value)
    name = native.id
    primitive: dict[str, dict[str, wire.JSON]] = {
        "boolean": {"kind": "boolean"},
        "varchar": {"kind": "text"},
        "blob": {"kind": "bytes"},
        "date": {"kind": "date"},
        "time": {"kind": "time", "unit": "us"},
        "timestamp": {"kind": "timestamp", "unit": "us", "timezone": "none"},
        "timestamp_s": {"kind": "timestamp", "unit": "s", "timezone": "none"},
        "timestamp_ms": {"kind": "timestamp", "unit": "ms", "timezone": "none"},
        "timestamp with time zone": {"kind": "timestamp", "unit": "us", "timezone": "utc"},
        "float": {"kind": "float", "bits": 32},
        "double": {"kind": "float", "bits": 64},
        "null": {"kind": "null"},
    }
    if name in primitive:
        return primitive[name]
    integers = {
        "tinyint": 8,
        "smallint": 16,
        "integer": 32,
        "bigint": 64,
        "hugeint": 128,
        "utinyint": 8,
        "usmallint": 16,
        "uinteger": 32,
        "ubigint": 64,
        "uhugeint": 128,
    }
    if name in integers:
        return {"kind": "integer", "bits": integers[name], "signed": not name.startswith("u")}
    children = native.children if name in ("decimal", "list", "array", "struct") else []
    if name == "decimal":
        fields = dict(children)
        _need(set(fields) == {"precision", "scale"}, "result_type_unsupported")
        return wire.logical_type(
            {"kind": "decimal", "precision": fields["precision"], "scale": fields["scale"]}
        )
    if name == "list":
        _need(len(children) == 1 and children[0][0] == "child", "result_type_unsupported")
        return {"kind": "list", "item": _native_type(children[0][1], depth + 1)}
    if name == "array":
        fields = dict(children)
        _need(set(fields) == {"child", "size"}, "result_type_unsupported")
        return wire.logical_type(
            {
                "kind": "array",
                "length": fields["size"],
                "item": _native_type(fields["child"], depth + 1),
            }
        )
    if name == "struct":
        _need(0 < len(children) <= 1024, "result_type_unsupported")
        return {
            "kind": "struct",
            "fields": [
                {"name": key, "type": _native_type(child, depth + 1)} for key, child in children
            ],
        }
    raise wire.WireError("result_type_unsupported")


def _columns(conn: _Connection, *, source: bool) -> list[wire.JSON]:
    description = conn.description
    _need(0 < len(description) <= 128, "output_limit")
    return wire.columns(
        [{"name": row[0], "type": _native_type(row[1])} for row in description], source=source
    )


def _physical_schema(conn: _Connection, path: str) -> None:
    conn.execute("SELECT name,num_children FROM parquet_schema(?)", [path])
    pending = [1]
    names: list[set[str]] = [set()]
    seen = 0
    while batch := conn.fetchmany(64):
        for name, children in batch:
            _need(
                type(name) is str and (children is None or type(children) is int), "source_mismatch"
            )
            while pending and pending[-1] == 0:
                pending.pop()
                names.pop()
            _need(bool(pending), "source_mismatch")
            pending[-1] -= 1
            text = cast("str", name)
            _need(len(text) <= 256 and len(text.encode("utf-8")) <= 256, "source_limit")
            folded = text.casefold()
            _need(folded not in names[-1], "source_mismatch")
            names[-1].add(folded)
            seen += 1
            _need(seen <= 4096 and len(pending) <= 24, "source_limit")
            count = 0 if children is None else cast("int", children)
            _need(0 <= count <= 1024, "source_limit")
            if count:
                pending.append(count)
                names.append(set())
    _need(all(left == 0 for left in pending), "source_mismatch")


def _scalar_count(conn: _Connection) -> int:
    row = conn.execute("SELECT count(*) FROM dataset").fetchone()
    _need(row is not None and len(row) == 1 and type(row[0]) is int, "source_mismatch")
    return cast("int", cast("tuple[object, ...]", row)[0])


def _hash_part(path: str, part: dict[str, wire.JSON], deadline: float) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        _need(stat.S_ISREG(info.st_mode) and info.st_size == part["size_bytes"], "source_mismatch")
        hashed = hashlib.sha256()
        seen = 0
        while chunk := os.read(descriptor, 65536):
            seen += len(chunk)
            _need(seen <= cast("int", part["size_bytes"]), "source_mismatch")
            _need(time.monotonic() < deadline, "deadline")
            hashed.update(chunk)
        _need(
            seen == part["size_bytes"] and hashed.hexdigest() == part["sha256"], "source_mismatch"
        )
    finally:
        os.close(descriptor)


def _preload(conn: _Connection, manifest: dict[str, wire.JSON], deadline: float) -> None:
    # The relative worker guard supplements the parent's original absolute timer;
    # host and container monotonic clocks need not share an epoch.
    parts = cast("list[dict[str, wire.JSON]]", manifest["parts"])
    expected_schema = manifest["schema"]
    total = 0
    for ordinal, part in enumerate(parts):
        _need(time.monotonic() < deadline, "deadline")
        path = f"/input/part-{ordinal:06d}.parquet"
        _hash_part(path, part, deadline)
        _physical_schema(conn, path)
        conn.execute(
            "SELECT * FROM read_parquet(?,hive_partitioning=false,union_by_name=false) LIMIT 0",
            [path],
        )
        _need(_columns(conn, source=True) == expected_schema, "source_mismatch")
        if ordinal == 0:
            conn.execute(
                "CREATE TABLE dataset AS SELECT * FROM read_parquet(?,hive_partitioning=false,union_by_name=false) LIMIT 0",
                [path],
            )
        conn.execute(
            "INSERT INTO dataset SELECT * FROM read_parquet(?,hive_partitioning=false,union_by_name=false)",
            [path],
        )
        observed = _scalar_count(conn)
        _need(observed - total == part["expected_rows"] and observed <= 1000000, "source_mismatch")
        total = observed
    _need(total == manifest["expected_rows"], "source_mismatch")
    expected_names = {f"part-{i:06d}.parquet" for i in range(len(parts))}
    found: set[str] = set()
    for item in Path("/input").iterdir():
        _need(item.name in expected_names and item.name not in found, "source_mismatch")
        found.add(item.name)
    _need(found == expected_names, "source_mismatch")
    for ordinal, part in enumerate(parts):
        _hash_part(f"/input/part-{ordinal:06d}.parquet", part, deadline)
    conn.execute("SET enable_external_access=false")
    conn.execute("SET lock_configuration=true")
    _settings(conn, locked=True)


def _adapt(dtype: dict[str, wire.JSON], value: object, utc: object) -> object:
    if value is None:
        return None
    kind = dtype["kind"]
    if kind == "timestamp" and dtype["timezone"] == "utc" and type(value) is datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is utc else value
    if kind in ("list", "array"):
        _need(
            type(value) in (list, tuple) and len(cast("list[object]", value)) <= 1024,
            "output_limit",
        )
        return [
            _adapt(cast("dict[str, wire.JSON]", dtype["item"]), item, utc)
            for item in cast("list[object]", value)
        ]
    if kind == "struct":
        _need(type(value) is dict, "result_type_unsupported")
        fields = cast("dict[str, object]", value)
        definitions = cast("list[dict[str, wire.JSON]]", dtype["fields"])
        _need(
            len(fields) == len(definitions) and all(type(key) is str for key in fields),
            "result_type_unsupported",
        )
        _need(all(field["name"] in fields for field in definitions), "result_type_unsupported")
        return {
            cast("str", field["name"]): _adapt(
                cast("dict[str, wire.JSON]", field["type"]), fields[cast("str", field["name"])], utc
            )
            for field in definitions
        }
    return value


def _read_request() -> dict[str, wire.JSON]:
    prefix = sys.stdin.buffer.read(4)
    _need(len(prefix) == 4, "invalid_spec")
    import struct

    size = cast("int", struct.unpack("!I", prefix)[0])
    _need(0 < size <= wire.MAX_REQUEST, "invalid_spec")
    raw = sys.stdin.buffer.read(size)
    _need(len(raw) == size and sys.stdin.buffer.read(1) == b"", "invalid_spec")
    decoded = wire.parse(raw, limit=wire.MAX_REQUEST)
    _need(type(decoded) is dict, "invalid_spec")
    data = cast("dict[str, wire.JSON]", decoded)
    _need(
        set(data) == {"version", "operation_id", "query", "manifest", "request_sha256"}
        and type(data["version"]) is int
        and data["version"] == 1,
        "invalid_spec",
    )
    return data


def _table(
    conn: _Connection, spec: dict[str, wire.JSON], utc: object, deadline: float
) -> dict[str, wire.JSON]:
    sql = cast("str", spec["sql"])
    statements = conn.extract_statements(sql)
    _need(len(statements) == 1 and statements[0].type.name == "SELECT", "query_refused")
    conn.execute(sql)
    columns = _columns(conn, source=False)
    types = [cast("dict[str, wire.JSON]", column)["type"] for column in columns]
    rows: list[wire.JSON] = []
    limit = cast("int", spec["max_result_rows"])
    truncated = False
    encoded_size = len(wire.canonical(columns, limit=wire.MAX_MANIFEST))
    while len(rows) <= limit:
        _need(time.monotonic() < deadline, "deadline")
        batch = conn.fetchmany(min(64, limit + 1 - len(rows)))
        if not batch:
            break
        for row in batch:
            if len(rows) == limit:
                truncated = True
                break
            _need(len(row) == len(types), "worker_failed")
            encoded: list[wire.JSON] = [
                wire.native_cell(dtype, _adapt(cast("dict[str, wire.JSON]", dtype), cell, utc))
                for dtype, cell in zip(types, row, strict=True)
            ]
            encoded_size += len(wire.canonical(encoded, limit=wire.MAX_RESULT)) + 1
            _need(encoded_size <= wire.MAX_RESULT - 65536, "output_limit")
            rows.append(encoded)
        if truncated:
            break
    _need(time.monotonic() < deadline, "deadline")
    return {
        "version": 1,
        "columns": columns,
        "rows": rows,
        "returned_rows": len(rows),
        "result_total_rows": None if truncated else len(rows),
        "result_truncated": truncated,
        "coverage": {
            "input_manifest_materialized_complete": True,
            "upstream_inventory_verified": False,
            "query_scan_coverage": "not_measured",
            "result_state": "row_prefix" if truncated else "exhausted",
        },
    }


def main() -> int:
    operation = ""
    request_sha: str | None = None
    conn: _Connection | None = None
    output: dict[str, wire.JSON] | None = None
    code = "profile_refused"
    try:
        _need(len(sys.argv) == 3 and sys.argv[1] == "--operation")
        operation = sys.argv[2]
        _need(len(operation) == 32 and all(c in "0123456789abcdef" for c in operation))
        _profile()
        logging.disable(sys.maxsize)
        engine = cast("_Engine", importlib.import_module("duckdb"))
        pytz = importlib.import_module("pytz")
        _need(engine.__version__ == "2.0.0" and pytz.__version__ == "2026.6")
        conn = _configured(engine)
        bundle = bundle_sha256()
        sys.stdout.buffer.write(wire.frame(expected_ready(operation, bundle), limit=8192))
        sys.stdout.buffer.flush()
        request = _read_request()
        _need(request["operation_id"] == operation, "invalid_spec")
        spec = wire.query(request["query"])
        manifest = wire.manifest(request["manifest"])
        _need(
            wire.canonical(request["query"], limit=wire.MAX_REQUEST)
            == wire.canonical(spec, limit=wire.MAX_REQUEST),
            "invalid_spec",
        )
        _need(
            wire.canonical(request["manifest"], limit=wire.MAX_MANIFEST)
            == wire.canonical(manifest, limit=wire.MAX_MANIFEST),
            "invalid_spec",
        )
        unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
        request_sha = wire.digest(unsigned, limit=wire.MAX_REQUEST)
        _need(
            request["request_sha256"] == request_sha
            and spec["input_manifest_sha256"] == wire.digest(manifest),
            "source_mismatch",
        )
        deadline = time.monotonic() + cast("int", spec["timeout_ms"]) / 1000
        code = "source_mismatch"
        _preload(conn, manifest, deadline)
        _need(time.monotonic() < deadline, "deadline")
        code = "query_failed"
        table = _table(conn, spec, pytz.UTC, deadline)
        conn.close()
        conn = None
        output = {
            "version": 1,
            "kind": "result",
            "operation_id": operation,
            "request_sha256": request_sha,
            "manifest_sha256": wire.digest(manifest),
            "table": table,
            "bundle_sha256": bundle,
            "source_observed": {
                "parts": manifest["part_count"],
                "rows": manifest["expected_rows"],
                "bytes": sum(
                    cast("int", cast("dict[str, wire.JSON]", part)["size_bytes"])
                    for part in cast("list[wire.JSON]", manifest["parts"])
                ),
                "schema_sha256": manifest["schema_sha256"],
            },
        }
        wire.canonical(output, limit=wire.MAX_RESULT)
    except wire.WireError as refusal:
        output = None
        code = refusal.code
    except Exception:
        output = None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                output = None
    if output is None:
        output = {
            "version": 1,
            "kind": "error",
            "operation_id": operation,
            "request_sha256": request_sha,
            "code": code,
        }
    sys.stdout.buffer.write(wire.frame(output, limit=wire.MAX_RESULT))
    sys.stdout.buffer.flush()
    return 0 if output["kind"] == "result" else 1


if __name__ == "__main__":
    try:
        status = main()
    except BaseException:
        status = 1
    raise SystemExit(status)
