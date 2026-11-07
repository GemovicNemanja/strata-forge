"""Explicit owned Docker opt-in. Missing installation is a prerequisite failure, never a pass."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from strata_forge.datasets._sql_container import SQLContainerOwner
from strata_forge.datasets.sql import LocalSQLExecutor
from strata_forge.datasets.sql_types import (
    LocalParquetFiles,
    LocalSQLInstallation,
    SQLExecutionError,
    SQLInputManifest,
    SQLQuerySpec,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from strata_forge.datasets import _sql_wire as wire

pytestmark = pytest.mark.integration
_ROOT = Path(__file__).parents[2] / "fixtures/headless-sql"


def installed() -> LocalSQLInstallation:
    path = os.environ.get("STRATA_TEST_SQL_INSTALLATION")
    if not path:
        pytest.fail(
            "Select this actual Docker suite only with an explicitly prepared STRATA_TEST_SQL_INSTALLATION; no setup or pull occurs in query."
        )
    return LocalSQLInstallation.from_wire(json.loads(Path(path).read_bytes()))


def manifest() -> SQLInputManifest:
    return SQLInputManifest.from_wire(json.loads((_ROOT / "typed-manifest.json").read_bytes()))


def spec(query: str, limit: int = 1000, timeout: int = 10000) -> SQLQuerySpec:
    return SQLQuerySpec.from_wire(
        {
            "version": 1,
            "sql": query,
            "input_manifest_sha256": manifest().digest(),
            "max_result_rows": limit,
            "timeout_ms": timeout,
        }
    )


async def execute(query: str, *, limit: int = 1000) -> dict[str, wire.JSON]:
    with (_ROOT / "typed.parquet").open("rb") as source:
        async with LocalSQLExecutor(installation=installed()) as executor:
            result = await executor.query(
                spec=spec(query, limit),
                source=LocalParquetFiles.from_open_fds(manifest=manifest(), fds=(source.fileno(),)),
            )
    return result.wire_payload()


@pytest.mark.parametrize(
    "query",
    [
        "SELECT id FROM dataset ORDER BY id",
        "WITH x AS (SELECT * FROM dataset) SELECT id FROM x ORDER BY id",
        "SELECT id FROM query_table('dataset') ORDER BY id",
    ],
)
async def test_actual_complete_split_one_relation(query: str) -> None:
    result = await execute(query)
    table = cast("dict[str,wire.JSON]", result["table"])
    assert table["rows"] == [
        [{"kind": "integer", "value": "1"}],
        [{"kind": "integer", "value": "3"}],
        [{"kind": "integer", "value": "9007199254740993"}],
    ]
    assert table["result_truncated"] is False
    receipt = cast("dict[str,wire.JSON]", result["receipt"])
    assert receipt["source_observed"] == {
        "parts": 1,
        "rows": 3,
        "bytes": 1461,
        "schema_sha256": manifest().wire_payload()["schema_sha256"],
    }
    assert cast("dict[str,wire.JSON]", receipt["cleanup"])["container_removed"] is True


async def test_actual_decimal_utc_nested_golden_and_prefix_distinct() -> None:
    result = await execute(
        "SELECT id,amount,zoned,nested,details FROM dataset ORDER BY id", limit=2
    )
    table = cast("dict[str,wire.JSON]", result["table"])
    rows = cast("list[list[dict[str,wire.JSON]]]", table["rows"])
    assert rows[0][1] == {"kind": "decimal", "value": "1.25"}
    assert rows[0][2] == {"kind": "timestamp", "value": "2040-01-02T03:04:05.000006+00:00"}
    assert rows[0][3] == {
        "kind": "list",
        "values": [{"kind": "integer", "value": "1"}, {"kind": "null"}],
    }
    assert table["result_truncated"] is True
    assert table["result_total_rows"] is None
    assert cast("dict[str,wire.JSON]", table["coverage"])["result_state"] == "row_prefix"


@pytest.mark.parametrize(
    "query", ["DESCRIBE dataset", "SUMMARIZE dataset", "SELECT * FROM pragma_table_info('dataset')"]
)
async def test_actual_native_select_kind_metadata_queries(query: str) -> None:
    table = cast("dict[str,wire.JSON]", (await execute(query))["table"])
    assert table["returned_rows"]


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM dataset",
        "SELECT 1; SELECT 2",
        "SELECT * FROM query('DELETE FROM dataset RETURNING *')",
        "SELECT * FROM query('UPDATE dataset SET id=2 RETURNING *')",
        "SELECT * FROM query('CREATE TABLE changed AS SELECT 1')",
        "SELECT * FROM query('PRAGMA enable_profiling')",
        "SELECT * FROM query('SET TimeZone=''Europe/Belgrade''')",
        "SELECT * FROM query_table('dataset; DELETE FROM dataset')",
        "SELECT * FROM read_text('/etc/passwd')",
        "SELECT repeat('x',30000) FROM range(1000)",
        "SELECT 1::TIMESTAMP_NS",
    ],
)
async def test_actual_dynamic_mutation_external_and_codec_paths_refuse_whole(query: str) -> None:
    with pytest.raises(SQLExecutionError) as caught:
        await execute(query)
    assert caught.value.code in {
        "query_refused",
        "query_failed",
        "output_limit",
        "result_type_unsupported",
    }
    assert str(caught.value) == "sql_execution_refused"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


async def test_actual_durable_no_source_mutation_across_queries() -> None:
    table = cast("dict[str,wire.JSON]", (await execute("SELECT count(*) FROM dataset"))["table"])
    assert table["rows"] == [[{"kind": "integer", "value": "3"}]]


async def test_actual_oversized_cell_has_fixed_refusal_no_result() -> None:
    # The accepted shared scalar validator uses invalid_spec for a text cap refusal.
    with pytest.raises(SQLExecutionError) as caught:
        await execute("SELECT repeat('x',65537)")
    assert caught.value.code == "invalid_spec"
    assert caught.value.__context__ is None


async def test_actual_repeated_cancel_holds_admission_through_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reading, waiting, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_frame = cast(
        "Callable[[SQLContainerOwner, asyncio.StreamReader, int], Awaitable[bytes]]",
        vars(SQLContainerOwner)["_frame"],
    )
    original_command = cast("Callable[..., Awaitable[bytes]]", vars(SQLContainerOwner)["_command"])

    async def observed_frame(
        self: SQLContainerOwner, reader: asyncio.StreamReader, limit: int
    ) -> bytes:
        if limit == 2097152:
            reading.set()
        return await original_frame(self, reader, limit)

    async def observed_command(self: SQLContainerOwner, *args: str, deadline: float) -> bytes:
        if args[0] == "wait":
            waiting.set()
            await release.wait()
        return await original_command(self, *args, deadline=deadline)

    monkeypatch.setattr(SQLContainerOwner, "_frame", observed_frame)
    monkeypatch.setattr(SQLContainerOwner, "_command", observed_command)
    with (_ROOT / "typed.parquet").open("rb") as opened:
        source = LocalParquetFiles.from_open_fds(manifest=manifest(), fds=(opened.fileno(),))
        async with LocalSQLExecutor(installation=installed()) as executor:
            running = asyncio.create_task(
                executor.query(
                    spec=spec(
                        "WITH RECURSIVE x(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM x) SELECT sum(i) FROM x"
                    ),
                    source=source,
                )
            )
            try:
                await asyncio.wait_for(reading.wait(), 5)
                await asyncio.sleep(0.03)
                running.cancel()
                await asyncio.wait_for(waiting.wait(), 5)
                running.cancel()
                await asyncio.sleep(0.01)
                assert not running.done()
                with pytest.raises(SQLExecutionError) as caught:
                    await executor.query(spec=spec("SELECT 1"), source=source)
                assert caught.value.code == "busy"
            finally:
                release.set()
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
            # Confirmed cleanup permits an explicit new query; cancellation never replays.
            result = await executor.query(spec=spec("SELECT count(*) FROM dataset"), source=source)
            assert cast("dict[str,wire.JSON]", result.wire_payload()["table"])["rows"] == [
                [{"kind": "integer", "value": "3"}]
            ]


async def test_actual_absolute_budget_retires_nonreturning_native_query() -> None:
    with (_ROOT / "typed.parquet").open("rb") as opened:
        async with LocalSQLExecutor(installation=installed()) as executor:
            with pytest.raises(SQLExecutionError) as caught:
                await executor.query(
                    spec=spec(
                        "WITH RECURSIVE x(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM x) SELECT sum(i) FROM x",
                        timeout=2500,
                    ),
                    source=LocalParquetFiles.from_open_fds(
                        manifest=manifest(), fds=(opened.fileno(),)
                    ),
                )
            assert caught.value.code == "deadline"
            result = await executor.query(
                spec=spec("SELECT 1"),
                source=LocalParquetFiles.from_open_fds(manifest=manifest(), fds=(opened.fileno(),)),
            )
            assert cast("dict[str,wire.JSON]", result.wire_payload()["table"])["rows"] == [
                [{"kind": "integer", "value": "1"}]
            ]


async def test_actual_missing_immutable_image_refuses_without_pull() -> None:
    profile = installed().wire_payload()
    profile["image_id"] = "sha256:" + "0" * 64
    with (_ROOT / "typed.parquet").open("rb") as opened:
        async with LocalSQLExecutor(
            installation=LocalSQLInstallation.from_wire(profile)
        ) as executor:
            with pytest.raises(SQLExecutionError) as caught:
                await executor.query(
                    spec=spec("SELECT 1"),
                    source=LocalParquetFiles.from_open_fds(
                        manifest=manifest(), fds=(opened.fileno(),)
                    ),
                )
            assert caught.value.code == "unavailable"


async def test_actual_case_colliding_physical_parquet_is_refused(tmp_path: Path) -> None:
    data = (_ROOT / "typed.parquet").read_bytes().replace(b"zoned", b"Stamp")
    assert data != (_ROOT / "typed.parquet").read_bytes()
    target = tmp_path / "collision.parquet"
    target.write_bytes(data)
    import hashlib

    payload = manifest().wire_payload()
    part = cast("dict[str,wire.JSON]", cast("list[wire.JSON]", payload["parts"])[0])
    part["sha256"] = hashlib.sha256(data).hexdigest()
    changed = SQLInputManifest.from_wire(payload)
    request = SQLQuerySpec.from_wire(
        {
            "version": 1,
            "sql": "SELECT 1",
            "input_manifest_sha256": changed.digest(),
            "timeout_ms": 10000,
        }
    )
    with target.open("rb") as opened:
        async with LocalSQLExecutor(installation=installed()) as executor:
            with pytest.raises(SQLExecutionError) as caught:
                await executor.query(
                    spec=request,
                    source=LocalParquetFiles.from_open_fds(
                        manifest=changed, fds=(opened.fileno(),)
                    ),
                )
    assert caught.value.code == "source_mismatch"
