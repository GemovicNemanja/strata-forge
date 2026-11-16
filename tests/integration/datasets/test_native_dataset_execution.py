"""Explicit actual fixed-image dataset tests using generated fictional bytes only."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# Fault controls inspect fixed private cleanup/worker boundaries without runtime bypasses.
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import cast

import pytest

from strata_forge.datasets import _sql_wire as sql
from strata_forge.datasets.dataset_execution import DatasetResult, LocalDatasetExecutor
from strata_forge.datasets.dataset_execution_types import (
    DatasetConvertSpec,
    DatasetExecutionError,
    DatasetInputManifest,
    DatasetInstallation,
    DatasetQuerySpec,
    NamedParquetFiles,
    ProvidedDatasetFiles,
)
from strata_forge.datasets.sql_types import LocalParquetFiles, SQLInputManifest

pytestmark = pytest.mark.integration


@pytest.fixture
def installation() -> DatasetInstallation:
    selected = os.environ.get("STRATA_TEST_DATASET_INSTALLATION")
    if selected is None:
        pytest.fail("explicit native dataset installation required; setup is never automatic")
    return DatasetInstallation.from_wire(json.loads(Path(selected).read_bytes()))


def columns() -> list[sql.JSON]:
    return [
        {"name": "id", "type": {"kind": "integer", "bits": 64, "signed": True}, "nullable": False},
        {"name": "text", "type": {"kind": "text"}, "nullable": True},
    ]


async def convert(
    executor: LocalDatasetExecutor,
    root: Path,
    data: bytes,
    *,
    fmt: str = "jsonl",
    schema: list[sql.JSON] | None = None,
) -> DatasetResult:
    path = root / ("input-" + hashlib.sha256(data).hexdigest())
    path.write_bytes(data)
    manifest = DatasetInputManifest.from_wire(
        {
            "version": 1,
            "format": fmt,
            "parts": [
                {"ordinal": 0, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
            ],
        }
    )
    spec = DatasetConvertSpec.from_wire(
        {
            "version": 1,
            "format": fmt,
            "schema": None if fmt == "parquet" else (columns() if schema is None else schema),
            "input_manifest_sha256": manifest.digest(),
            "timeout_ms": 10000,
        }
    )
    fd = os.open(path, os.O_RDONLY)
    os.lseek(fd, 1, os.SEEK_SET)
    try:
        result = await executor.convert(
            spec=spec, source=ProvidedDatasetFiles(manifest=manifest, fds=(fd,))
        )
        assert os.lseek(fd, 0, os.SEEK_CUR) == 1
        assert os.fstat(fd).st_size == len(data)
        assert executor._staging is None
        return result
    finally:
        os.close(fd)


def named(result: DatasetResult, root: Path, alias: str) -> tuple[NamedParquetFiles, int]:
    payload = result.wire_payload()
    descriptor = cast("dict[str, sql.JSON]", payload["output"])
    data = result.output_bytes()
    path = root / (alias + ".parquet")
    path.write_bytes(data)
    manifest = SQLInputManifest.from_wire(
        {
            "version": 1,
            "source_assertion": {
                "system": "local",
                "name": "fictional",
                "config": "default",
                "split": "provided",
                "revision": {"algorithm": "sha256", "value": descriptor["sha256"]},
            },
            "parts": [
                {
                    "ordinal": 0,
                    "sha256": descriptor["sha256"],
                    "size_bytes": len(data),
                    "expected_rows": descriptor["rows"],
                }
            ],
            "schema": descriptor["schema"],
            "schema_sha256": descriptor["schema_sha256"],
            "declared_population": "complete_named_split",
            "expected_rows": descriptor["rows"],
            "part_count": 1,
        }
    )
    fd = os.open(path, os.O_RDONLY)
    return NamedParquetFiles(
        alias=alias, source=LocalParquetFiles.from_open_fds(manifest=manifest, fds=(fd,))
    ), fd


def query_spec(
    sources: tuple[NamedParquetFiles, ...], statement: str, limit: int = 1000
) -> DatasetQuerySpec:
    return DatasetQuerySpec.from_wire(
        {
            "version": 1,
            "sql": statement,
            "inputs_sha256": sql.digest([source.checked()[0] for source in sources]),
            "timeout_ms": 10000,
            "max_result_rows": limit,
        }
    )


def table(result: DatasetResult) -> dict[str, sql.JSON]:
    return cast("dict[str, sql.JSON]", result.wire_payload()["table"])


async def test_native_jsonl_join_materialize_and_exact_readback(
    installation: DatasetInstallation, tmp_path: Path
) -> None:
    async with LocalDatasetExecutor(installation=installation) as executor:
        left = await convert(
            executor, tmp_path, '{"id":1,"text":"雪"}\n{"id":2,"text":"{{literal}}"}\n'.encode()
        )
        right = await convert(executor, tmp_path, b'{"id":1,"text":"A"}\n{"id":2,"text":null}\n')
        first, fd1 = named(left, tmp_path, "items")
        second, fd2 = named(right, tmp_path, "labels")
        third, fd3 = named(left, tmp_path, "other_items")
        fourth, fd4 = named(right, tmp_path, "other_labels")
        try:
            sources = (first, second, third, fourth)
            statement = "SELECT items.id,items.text,labels.text AS label FROM items JOIN labels USING(id) JOIN other_items USING(id) JOIN other_labels USING(id) ORDER BY items.id"
            spec = query_spec(sources, statement, limit=1)
            observed = table(await executor.query(spec=spec, sources=sources))
            assert observed["returned_rows"] == 1
            assert observed["result_truncated"] is True
            materialized = await executor.materialize(spec=spec, sources=sources)
            descriptor = cast("dict[str, sql.JSON]", materialized.wire_payload()["output"])
            assert descriptor["rows"] == 2
            final, final_fd = named(materialized, tmp_path, "result")
            try:
                restored = table(
                    await executor.query(
                        spec=query_spec((final,), "SELECT * FROM result ORDER BY id"),
                        sources=(final,),
                    )
                )
                assert restored["returned_rows"] == 2
                assert restored["rows"] == [
                    [
                        {"kind": "integer", "value": "1"},
                        {"kind": "text", "value": "雪"},
                        {"kind": "text", "value": "A"},
                    ],
                    [
                        {"kind": "integer", "value": "2"},
                        {"kind": "text", "value": "{{literal}}"},
                        {"kind": "null"},
                    ],
                ]
                assert json.dumps(materialized.wire_payload())[0] == "{"
                assert materialized.model_dump() == {}
            finally:
                os.close(final_fd)
        finally:
            for fd in (fd1, fd2, fd3, fd4):
                os.close(fd)


async def test_native_csv_quotes_crlf_empty_and_null(
    installation: DatasetInstallation, tmp_path: Path
) -> None:
    async with LocalDatasetExecutor(installation=installation) as executor:
        result = await convert(
            executor, tmp_path, b'id,text\r\n1,"a""b\r\nc"\r\n2,\r\n3,\\N\r\n', fmt="csv"
        )
        source, fd = named(result, tmp_path, "items")
        try:
            rows = table(
                await executor.query(
                    spec=query_spec((source,), "SELECT * FROM items ORDER BY id"), sources=(source,)
                )
            )["rows"]
            assert rows == [
                [{"kind": "integer", "value": "1"}, {"kind": "text", "value": 'a"b\r\nc'}],
                [{"kind": "integer", "value": "2"}, {"kind": "text", "value": ""}],
                [{"kind": "integer", "value": "3"}, {"kind": "null"}],
            ]
        finally:
            os.close(fd)


@pytest.mark.parametrize("raw", [b'id,text\n1,a"b\n', b"id,text\r1,hello\r", b'id,text\n1,"a"x\n'])
async def test_native_csv_grammar_refusal_then_healthy_reuse(
    installation: DatasetInstallation, tmp_path: Path, raw: bytes
) -> None:
    async with LocalDatasetExecutor(installation=installation) as executor:
        with pytest.raises(DatasetExecutionError) as caught:
            await convert(executor, tmp_path, raw, fmt="csv")
        assert caught.value.code == "source_mismatch"
        assert caught.value.__context__ is None
        assert executor._staging is None
        observation = executor.failure_observation(caught.value)
        assert observation is not None
        failure = observation.wire_payload()
        assert failure is not None
        assert failure["code"] == "source_mismatch"
        assert failure["kind"] == "convert"
        assert failure["reusable_at_settlement"] is True
        assert failure["cleanup"] == {
            "container_waited": True,
            "stopped_pid_zero": True,
            "container_removed": True,
            "attach_reaped": True,
            "staging_removed": True,
            "exit_code": 1,
            "oom_killed": False,
        }
        original_owner = executor._owner
        good = await convert(executor, tmp_path, b"id,text\n1,healthy\n", fmt="csv")
        assert cast("dict[str, sql.JSON]", good.wire_payload()["output"])["rows"] == 1
        assert executor._owner is original_owner
        assert observation.wire_payload() is None


@pytest.mark.parametrize(
    ("statement", "code"),
    [("DROP TABLE items", "query_refused"), ("SELECT missing_column FROM items", "worker_failed")],
)
async def test_native_failed_query_observation_then_same_executor_health(
    installation: DatasetInstallation, tmp_path: Path, statement: str, code: str
) -> None:
    async with LocalDatasetExecutor(installation=installation) as executor:
        converted = await convert(executor, tmp_path, b'{"id":1,"text":"fictional"}\n')
        source, fd = named(converted, tmp_path, "items")
        original_owner = executor._owner
        try:
            with pytest.raises(DatasetExecutionError) as caught:
                await executor.query(spec=query_spec((source,), statement), sources=(source,))
            assert caught.value.code == code
            observation = executor.failure_observation(caught.value)
            assert observation is not None
            failure = observation.wire_payload()
            assert failure is not None
            assert failure["code"] == code
            assert failure["kind"] == "query"
            assert failure["reusable_at_settlement"] is True
            assert len(cast("str", failure["operation_id"])) == 32
            assert len(cast("str", failure["request_sha256"])) == 64
            healthy = await executor.query(
                spec=query_spec((source,), "SELECT id FROM items"), sources=(source,)
            )
            assert table(healthy)["returned_rows"] == 1
            assert executor._owner is original_owner
            assert observation.wire_payload() is None
        finally:
            os.close(fd)


async def test_native_empty_and_full_parquet_type_roundtrip(
    installation: DatasetInstallation, tmp_path: Path
) -> None:
    async with LocalDatasetExecutor(installation=installation) as executor:
        empty = await convert(executor, tmp_path, b"id,text\n", fmt="csv")
        assert cast("dict[str, sql.JSON]", empty.wire_payload()["output"])["rows"] == 0
        fixture = Path(__file__).parents[2] / "fixtures/headless-sql/typed.parquet"
        original_manifest = json.loads(fixture.with_name("typed-manifest.json").read_bytes())
        converted = await convert(executor, tmp_path, fixture.read_bytes(), fmt="parquet")
        descriptor = cast("dict[str, sql.JSON]", converted.wire_payload()["output"])
        assert descriptor["schema"] == original_manifest["schema"]
        assert descriptor["rows"] == 3
        source, fd = named(converted, tmp_path, "items")
        try:
            observed = table(
                await executor.query(
                    spec=query_spec((source,), "SELECT * FROM items ORDER BY id"), sources=(source,)
                )
            )
            assert observed["returned_rows"] == 3
            assert len(cast("list[sql.JSON]", observed["columns"])) == 8
            assert "9007199254740993" in json.dumps(observed)
        finally:
            os.close(fd)


@pytest.mark.parametrize(
    "statement",
    [
        "DROP TABLE items",
        "SELECT 1; SELECT 2",
        "SELECT * FROM read_text('/etc/passwd')",
        "SELECT * FROM read_parquet('/output/other.parquet')",
        "SELECT uuid()",
    ],
)
async def test_native_refuses_mutation_external_path_and_unsupported_type(
    installation: DatasetInstallation, tmp_path: Path, statement: str
) -> None:
    async with LocalDatasetExecutor(installation=installation) as executor:
        converted = await convert(executor, tmp_path, b'{"id":1,"text":"fictional"}\n')
        source, fd = named(converted, tmp_path, "items")
        try:
            with pytest.raises(DatasetExecutionError) as caught:
                await executor.query(spec=query_spec((source,), statement), sources=(source,))
            assert caught.value.__context__ is None
            assert statement not in str(caught.value)
            assert executor._staging is None
        finally:
            os.close(fd)


async def test_native_output_row_cap_precedes_cell_scan(
    installation: DatasetInstallation, tmp_path: Path
) -> None:
    async with LocalDatasetExecutor(installation=installation) as executor:
        converted = await convert(executor, tmp_path, b'{"id":1,"text":"fictional"}\n')
        source, fd = named(converted, tmp_path, "items")
        try:
            spec = query_spec((source,), "SELECT i FROM range(1000001) AS t(i)", limit=1)
            prefix = table(await executor.query(spec=spec, sources=(source,)))
            assert prefix["returned_rows"] == 1
            assert prefix["result_truncated"] is True
            with pytest.raises(DatasetExecutionError) as caught:
                await executor.materialize(spec=spec, sources=(source,))
            assert caught.value.code == "output_limit"
        finally:
            os.close(fd)


async def test_native_output_larger_than_default_store_chunk(
    installation: DatasetInstallation, tmp_path: Path
) -> None:
    raw = b"".join(
        json.dumps(
            {"id": i, "text": hashlib.shake_256(str(i).encode()).hexdigest(16384)},
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for i in range(96)
    )
    async with LocalDatasetExecutor(installation=installation) as executor:
        result = await convert(executor, tmp_path, raw)
        assert len(result.output_bytes()) > 1024 * 1024
        assert len(result.output_bytes()) <= 64 * 1024 * 1024
        assert (
            hashlib.sha256(result.output_bytes()).hexdigest()
            == cast("dict[str, sql.JSON]", result.wire_payload()["output"])["sha256"]
        )


async def test_native_repeated_cancel_after_request_handoff_and_reuse(
    installation: DatasetInstallation, tmp_path: Path
) -> None:
    async with LocalDatasetExecutor(installation=installation) as executor:
        converted = await convert(executor, tmp_path, b'{"id":1,"text":"fictional"}\n')
        source, fd = named(converted, tmp_path, "items")
        try:
            spec = query_spec(
                (source,),
                "WITH RECURSIVE t(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM t WHERE i<1000000000) SELECT sum(i) FROM t",
            )
            running = asyncio.create_task(executor.query(spec=spec, sources=(source,)))
            async with asyncio.timeout(5):
                while True:
                    owner = executor._owner
                    if (
                        owner is not None
                        and owner._attach is not None
                        and owner._attach.stdin is not None
                        and owner._attach.stdin.is_closing()
                    ):
                        break
                    await asyncio.sleep(0.005)
            running.cancel()
            await asyncio.sleep(0.005)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            assert executor._staging is None
            assert executor._owner is not None
            assert executor._owner.available
            healthy = await executor.query(
                spec=query_spec((source,), "SELECT id FROM items"), sources=(source,)
            )
            assert table(healthy)["returned_rows"] == 1
        finally:
            os.close(fd)
