"""Fixed image-only conversion, named-table query and Parquet materialization."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# This fixed sibling composes only the unchanged image worker helpers, never its profile.
import csv
import hashlib
import importlib
import json
import logging
import math
import os
import re
import resource
import stat
import sys
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import TextIO

if __package__:
    from strata_forge.datasets import _dataset_wire as wire
    from strata_forge.datasets import _sql_wire as sql
    from strata_forge.datasets import _sql_worker as native
else:
    sys.path.insert(0, str(Path(__file__).parent))
    import _dataset_wire as wire
    import _sql_wire as sql
    import _sql_worker as native


# These fixed native helpers are shared with the unchanged read-only worker;
# its profile, preload and entrypoint are never invoked by the writer.


def bundle_sha256() -> str:
    sources: dict[str, bytes] = {}
    for name in wire.SOURCES:
        fd = os.open(Path(__file__).parent / name, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            wire.need(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= 131072, "profile_refused")
            sources[name] = os.read(fd, 131073)
        finally:
            os.close(fd)
    return wire.source_bundle(sources)


def expected_ready(operation_id: str, bundle: str) -> dict[str, sql.JSON]:
    result = native.expected_ready(operation_id, bundle)
    result.update({"profile": wire.PROFILE, "memory_bytes": 536870912, "file_bytes": 67108864})
    return result


def _profile() -> None:
    wire.need(sys.version_info[:3] == (3, 14, 5) and sys.platform == "linux", "profile_refused")
    wire.need(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1, "profile_refused")
    wire.need(os.getuid() == 65532 and os.getgid() == 65532 and os.getpid() == 1, "profile_refused")
    for key, pair in (
        (resource.RLIMIT_NOFILE, (64, 64)),
        (resource.RLIMIT_FSIZE, (67108864, 67108864)),
        (resource.RLIMIT_CORE, (0, 0)),
    ):
        wire.need(resource.getrlimit(key) == pair, "profile_refused")
    for name, expected in (
        ("memory.max", "536870912"),
        ("memory.swap.max", "0"),
        ("pids.max", "32"),
        ("cpu.max", "100000 100000"),
    ):
        wire.need(native._read_kernel(name) == expected, "profile_refused")
    status = Path("/proc/self/status").read_text()
    for expected in ("NoNewPrivs:\t1", "Seccomp:\t2", "CapEff:\t0000000000000000"):
        wire.need(expected in status, "profile_refused")
    wire.need(not list(Path("/output").iterdir()), "profile_refused")


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _type(dtype: dict[str, sql.JSON]) -> str:
    kind = dtype["kind"]
    if kind == "decimal":
        return f"DECIMAL({dtype['precision']},{dtype['scale']})"
    if kind == "timestamp":
        return "TIMESTAMPTZ" if dtype["timezone"] == "utc" else "TIMESTAMP"
    return {
        "text": "VARCHAR",
        "boolean": "BOOLEAN",
        "integer": "BIGINT",
        "float": "DOUBLE",
        "date": "DATE",
    }[cast("str", kind)]


def _scalar(value: object, dtype: dict[str, sql.JSON], *, is_csv: bool, nullable: bool) -> object:
    if value is None or (is_csv and value == r"\N"):
        wire.need(nullable, "source_mismatch")
        return None
    kind = dtype["kind"]
    if kind == "text":
        result: object = wire.text(value, sql.MAX_CELL)
    elif kind == "boolean":
        wire.need(value in ("true", "false") if is_csv else type(value) is bool, "source_mismatch")
        result = value == "true" if is_csv else value
    elif kind == "integer":
        if is_csv:
            wire.need(
                type(value) is str and re.fullmatch(r"-?(0|[1-9][0-9]*)", value) is not None,
                "source_mismatch",
            )
            value = int(cast("str", value))
        wire.need(type(value) is int and -(2**63) <= value < 2**63, "source_mismatch")
        result = value
    elif kind == "float":
        if is_csv:
            wire.need(
                type(value) is str
                and re.fullmatch(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?", value)
                is not None,
                "source_mismatch",
            )
            value = float(cast("str", value))
        wire.need(type(value) in (int, float), "source_mismatch")
        result = float(cast("int | float", value))
        wire.need(math.isfinite(result), "source_mismatch")
    elif kind == "decimal":
        value = wire.text(value, 128)
        wire.need(
            re.fullmatch(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?", value) is not None, "source_mismatch"
        )
        number = Decimal(value)
        wire.need(number.as_tuple().exponent == -cast("int", dtype["scale"]), "source_mismatch")
        wire.need(not (number.is_zero() and number.is_signed()), "source_mismatch")
        result = number
    elif kind == "date":
        value = wire.text(value, 10)
        wire.need(re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is not None, "source_mismatch")
        result = date.fromisoformat(value)
    else:
        value = wire.text(value, 32)
        utc = dtype["timezone"] == "utc"
        wire.need(
            re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,6})?"
                + ("Z" if utc else ""),
                value,
            )
            is not None,
            "source_mismatch",
        )
        result = datetime.fromisoformat(value)
    sql.native_cell(dtype, result)
    return result


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    wire.need(
        len(pairs) <= wire.MAX_COLUMNS and len({k for k, _ in pairs}) == len(pairs),
        "source_mismatch",
    )
    return dict(pairs)


def _csv_grammar(source: TextIO, deadline: float) -> None:
    state = "start"
    pending_cr = False
    count = 0
    while chunk := source.read(65536):
        count += len(chunk)
        wire.need(count <= wire.MAX_TEXT_BYTES, "source_limit")
        wire.need(time.monotonic() < deadline, "deadline")
        for char in chunk:
            if pending_cr:
                wire.need(char == "\n", "source_mismatch")
                pending_cr = False
                continue
            if char == "\r":
                pending_cr = True
                if state != "quoted":
                    state = "start"
            elif char == "\n":
                if state != "quoted":
                    state = "start"
            elif state == "quoted":
                if char == '"':
                    state = "after_quote"
            elif state == "after_quote":
                if char == '"':
                    state = "quoted"
                else:
                    wire.need(char == ",", "source_mismatch")
                    state = "start"
            elif char == ",":
                state = "start"
            elif char == '"':
                wire.need(state == "start", "source_mismatch")
                state = "quoted"
            else:
                state = "unquoted"
    wire.need(not pending_cr and state != "quoted", "source_mismatch")


def _memory_settings(conn: native._Connection) -> None:
    row = conn.execute(
        "SELECT current_setting('memory_limit'), current_setting('max_temp_directory_size')"
    ).fetchone()
    wire.need(
        type(row) is tuple and len(row) == 2 and all(type(v) is str for v in row), "profile_refused"
    )
    wire.need(row == ("183.1 MiB", "61.0 MiB"), "profile_refused")


def _text_input(conn: native._Connection, spec: dict[str, sql.JSON], deadline: float) -> None:
    schema = cast("list[dict[str, sql.JSON]]", spec["schema"])
    names = [cast("str", col["name"]) for col in schema]
    definitions = ",".join(
        _identifier(cast("str", c["name"])) + " " + _type(cast("dict[str, sql.JSON]", c["type"]))
        for c in schema
    )
    conn.execute("CREATE TABLE dataset (" + definitions + ")")
    insertion = "INSERT INTO dataset VALUES (" + ",".join("?" for _ in schema) + ")"  # noqa: S608 — fixed placeholders, values remain bound.
    count = 0
    csv.field_size_limit(sql.MAX_CELL)
    with Path("/input/part-000000.bin").open(encoding="utf-8", newline="") as source:
        wire.need(source.read(1) != "\ufeff", "source_mismatch")
        source.seek(0)
        if spec["format"] == "csv":
            _csv_grammar(source, deadline)
            source.seek(0)
            reader = csv.reader(source, strict=True)
            wire.need(next(reader, None) == names, "source_mismatch")
            for row in reader:
                count += 1
                wire.need(count <= wire.MAX_ROWS and time.monotonic() < deadline, "source_limit")
                wire.need(len(row) == len(schema), "source_mismatch")
                conn.execute(
                    insertion,
                    [
                        _scalar(
                            v,
                            cast("dict[str, sql.JSON]", c["type"]),
                            is_csv=True,
                            nullable=c["nullable"] is True,
                        )
                        for v, c in zip(row, schema, strict=True)
                    ],
                )
        else:
            for line in source:
                if line in ("\n", "\r\n"):
                    continue
                count += 1
                wire.need(count <= wire.MAX_ROWS and time.monotonic() < deadline, "source_limit")
                raw: object = json.loads(line, object_pairs_hook=_unique_pairs)
                wire.need(
                    type(raw) is dict and set(cast("dict[str, object]", raw)) == set(names),
                    "source_mismatch",
                )
                values = cast("dict[str, object]", raw)
                conn.execute(
                    insertion,
                    [
                        _scalar(
                            values[name],
                            cast("dict[str, sql.JSON]", c["type"]),
                            is_csv=False,
                            nullable=c["nullable"] is True,
                        )
                        for name, c in zip(names, schema, strict=True)
                    ],
                )
    conn.execute("SELECT * FROM dataset LIMIT 0")
    wire.need(
        native._columns(conn, source=True)
        == [{"name": c["name"], "type": c["type"]} for c in schema],
        "source_mismatch",
    )


def _parquet(
    conn: native._Connection,
    *,
    alias: str,
    parts: list[dict[str, sql.JSON]],
    expected: dict[str, sql.JSON] | None,
    start: int,
    deadline: float,
) -> int:
    schema: list[sql.JSON] | None = None
    total = 0
    target = _identifier(alias)
    for index, part in enumerate(parts):
        path = f"/input/part-{start + index:06d}.bin"
        native._hash_part(path, part, deadline)
        native._physical_schema(conn, path)
        conn.execute(
            "SELECT * FROM read_parquet(?,hive_partitioning=false,union_by_name=false) LIMIT 0",
            [path],
        )
        found = native._columns(conn, source=True)
        if schema is None:
            schema = found
            conn.execute(
                "CREATE TABLE "  # noqa: S608 — escaped identifier, bound path.
                + target
                + " AS SELECT * FROM read_parquet(?,hive_partitioning=false,union_by_name=false) LIMIT 0",
                [path],
            )
        wire.need(
            schema == found and (expected is None or found == expected["schema"]), "source_mismatch"
        )
        row = conn.execute("SELECT count(*) FROM read_parquet(?)", [path]).fetchone()
        wire.need(row is not None and type(row[0]) is int, "source_mismatch")
        observed = cast("int", cast("tuple[object, ...]", row)[0])
        total += observed
        wire.need(total <= wire.MAX_ROWS, "source_limit")
        wire.need(expected is None or part["expected_rows"] == observed, "source_mismatch")
        conn.execute(
            "INSERT INTO "  # noqa: S608 — escaped identifier, bound path.
            + target
            + " SELECT * FROM read_parquet(?,hive_partitioning=false,union_by_name=false)",
            [path],
        )
    wire.need(expected is None or total == expected["expected_rows"], "source_mismatch")
    return total


def _validated_output(
    conn: native._Connection, utc: object, deadline: float
) -> tuple[list[sql.JSON], int]:
    conn.execute("SELECT * FROM strata_result")
    columns = native._columns(conn, source=True)
    total = 0
    while batch := conn.fetchmany(64):
        wire.need(time.monotonic() < deadline, "deadline")
        total += len(batch)
        wire.need(total <= wire.MAX_ROWS, "output_limit")
        for row in batch:
            for column, value in zip(columns, row, strict=True):
                dtype = cast("dict[str, sql.JSON]", cast("dict[str, sql.JSON]", column)["type"])
                sql.native_cell(dtype, native._adapt(dtype, value, utc))
    return columns, total


def _read_request() -> dict[str, sql.JSON]:
    import struct

    prefix = sys.stdin.buffer.read(4)
    wire.need(len(prefix) == 4)
    size = cast("int", struct.unpack("!I", prefix)[0])
    wire.need(0 < size <= sql.MAX_REQUEST)
    raw = sys.stdin.buffer.read(size)
    wire.need(len(raw) == size and sys.stdin.buffer.read(1) == b"")
    return wire.obj(
        sql.parse(raw, limit=sql.MAX_REQUEST),
        {"version", "kind", "operation_id", "spec", "inputs", "request_sha256"},
    )


def main() -> int:
    operation = ""
    request_sha: str | None = None
    conn: native._Connection | None = None
    output: dict[str, sql.JSON] | None = None
    code = "profile_refused"
    try:
        wire.need(len(sys.argv) == 3 and sys.argv[1] == "--operation")
        operation = sys.argv[2]
        wire.need(re.fullmatch(r"[0-9a-f]{32}", operation) is not None)
        _profile()
        logging.disable(sys.maxsize)
        engine = cast("native._Engine", importlib.import_module("duckdb"))
        pytz = importlib.import_module("pytz")
        wire.need(engine.__version__ == "2.0.0" and pytz.__version__ == "2026.6", "profile_refused")
        conn = native._configured(engine)
        conn.execute("SET memory_limit='192MB'")
        conn.execute("SET max_temp_directory_size='64MB'")
        _memory_settings(conn)
        bundle = bundle_sha256()
        sys.stdout.buffer.write(sql.frame(expected_ready(operation, bundle), limit=8192))
        sys.stdout.buffer.flush()
        request = _read_request()
        wire.need(
            type(request["version"]) is int
            and request["version"] == 1
            and request["operation_id"] == operation
        )
        unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
        request_sha = sql.digest(unsigned, limit=sql.MAX_REQUEST)
        wire.need(request["request_sha256"] == request_sha, "source_mismatch")
        kind = request["kind"]
        wire.need(type(kind) is str and kind in {"convert", "query", "materialize"})
        spec = (
            wire.convert_spec(request["spec"])
            if kind == "convert"
            else wire.query_spec(request["spec"])
        )
        wire.need(spec == request["spec"])
        deadline = time.monotonic() + cast("int", spec["timeout_ms"]) / 1000
        count = 0
        if kind == "convert":
            manifest = wire.input_manifest(request["inputs"])
            wire.need(
                spec["format"] == manifest["format"]
                and spec["input_manifest_sha256"] == sql.digest(manifest),
                "source_mismatch",
            )
            parts = cast("list[dict[str, sql.JSON]]", manifest["parts"])
            count = len(parts)
            if spec["format"] == "parquet":
                _parquet(
                    conn, alias="dataset", parts=parts, expected=None, start=0, deadline=deadline
                )
            else:
                native._hash_part("/input/part-000000.bin", parts[0], deadline)
                _text_input(conn, spec, deadline)
        else:
            inputs = wire.named_inputs(request["inputs"])
            wire.need(spec["inputs_sha256"] == sql.digest(inputs), "source_mismatch")
            for raw in inputs:
                source = cast("dict[str, sql.JSON]", raw)
                manifest = cast("dict[str, sql.JSON]", source["manifest"])
                parts = cast("list[dict[str, sql.JSON]]", manifest["parts"])
                _parquet(
                    conn,
                    alias=cast("str", source["alias"]),
                    parts=parts,
                    expected=manifest,
                    start=count,
                    deadline=deadline,
                )
                count += len(parts)
        wire.need(
            {p.name for p in Path("/input").iterdir()}
            == {f"part-{i:06d}.bin" for i in range(count)},
            "source_mismatch",
        )
        conn.execute("SET allowed_paths=['/output/data.parquet']")
        conn.execute("SET enable_external_access=false")
        conn.execute("SET lock_configuration=true")
        native._settings(conn, locked=True)
        _memory_settings(conn)
        row = conn.execute("SELECT current_setting('allowed_paths')").fetchone()
        wire.need(row is not None and row[0] == ["/output/data.parquet"], "profile_refused")
        query = "SELECT * FROM dataset" if kind == "convert" else cast("str", spec["sql"])
        statements = conn.extract_statements(query)
        wire.need(len(statements) == 1 and statements[0].type.name == "SELECT", "query_refused")
        conn.execute("CREATE TABLE strata_result AS " + query)
        if kind == "query":
            table = native._table(
                conn,
                {"sql": "SELECT * FROM strata_result", "max_result_rows": spec["max_result_rows"]},
                pytz.UTC,
                deadline,
            )
            result: dict[str, sql.JSON] = {"table": table, "output": None}
        else:
            schema, rows = _validated_output(conn, pytz.UTC, deadline)
            conn.execute(
                "COPY strata_result TO '/output/data.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            path = Path("/output/data.parquet")
            size = path.stat().st_size
            wire.need(0 < size <= wire.MAX_BYTES, "output_limit")
            part: dict[str, sql.JSON] = {"sha256": "", "size_bytes": size}
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(65536):
                    wire.need(time.monotonic() < deadline, "deadline")
                    digest.update(chunk)
            part["sha256"] = digest.hexdigest()
            result = {
                "table": None,
                "output": {
                    **part,
                    "schema": schema,
                    "schema_sha256": sql.digest(schema),
                    "rows": rows,
                },
            }
        conn.close()
        conn = None
        wire.need(time.monotonic() < deadline, "deadline")
        output = {
            "version": 1,
            "kind": "result",
            "operation_id": operation,
            "request_sha256": request_sha,
            "bundle_sha256": bundle,
            **result,
        }
        sql.canonical(output, limit=sql.MAX_RESULT)
    except sql.WireError as refusal:
        output = None
        code = refusal.code
    except Exception:
        output = None
        code = "worker_failed"
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
    sys.stdout.buffer.write(sql.frame(output, limit=sql.MAX_RESULT))
    sys.stdout.buffer.flush()
    return 0 if output["kind"] == "result" else 1


if __name__ == "__main__":
    try:
        status = main()
    except BaseException:
        status = 1
    raise SystemExit(status)
