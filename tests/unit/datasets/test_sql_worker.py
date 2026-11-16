"""Fixed worker orchestration over fake native rows; actual engine tests remain separate."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING, cast

import pytest

from strata_forge.datasets import _sql_wire as wire
from strata_forge.datasets import _sql_worker as worker

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass
class NativeType:
    id: str
    children: list[tuple[str, object]]


@dataclass
class Kind:
    name: str = "SELECT"


@dataclass
class Statement:
    type: Kind


class Rows:
    def __init__(
        self,
        values: Sequence[tuple[object, ...]],
        dtype: NativeType | None = None,
        statement: str = "SELECT",
    ) -> None:
        self.rows = list(values)
        self.description: list[tuple[object, ...]] = [("value", dtype or NativeType("bigint", []))]
        self.calls: list[str] = []
        self.batches: list[int] = []
        self.kind = statement

    def execute(self, query: str, parameters: object = None) -> Rows:
        assert parameters is None
        self.calls.append(query)
        return self

    def extract_statements(self, query: str) -> list[Statement]:
        return (
            [Statement(Kind(self.kind))]
            if query != "multi"
            else [Statement(Kind()), Statement(Kind())]
        )

    def fetchmany(self, size: int = 1) -> list[tuple[object, ...]]:
        self.batches.append(size)
        result = self.rows[:size]
        del self.rows[:size]
        return result

    def fetchone(self) -> tuple[object, ...] | None:
        result = self.fetchmany(1)
        return result[0] if result else None

    def close(self) -> None:
        return


def table(
    native: Rows, *, limit: int, sql: str = "SELECT value FROM dataset"
) -> dict[str, wire.JSON]:
    fn = cast(
        "Callable[[object, dict[str, wire.JSON], object, float], dict[str, wire.JSON]]",
        vars(worker)["_table"],
    )
    return fn(
        native,
        {"sql": sql, "max_result_rows": limit},
        object(),
        time.monotonic() + 5,
    )


def test_fixed_small_fetch_batches_execute_once_with_one_lookahead() -> None:
    native = Rows([(index,) for index in range(10001)])
    result = table(native, limit=1000)
    assert native.calls == ["SELECT value FROM dataset"]
    assert native.batches == [64] * 15 + [41]
    rows = cast("list[wire.JSON]", result["rows"])
    assert len(rows) == 1000
    assert result["returned_rows"] == 1000
    assert result["result_total_rows"] is None
    assert result["result_truncated"] is True
    assert rows[0] == [{"kind": "integer", "value": "0"}]
    assert rows[-1] == [{"kind": "integer", "value": "999"}]
    assert wire.result_table(result) == result


def test_exhausted_empty_and_exact_limit_are_distinct_from_source_coverage() -> None:
    for rows in ([], [(1,)], [(1,), (2,)]):
        result = table(Rows(rows), limit=2)
        assert result["returned_rows"] == len(rows)
        assert result["result_total_rows"] == len(rows)
        assert result["result_truncated"] is False
        assert result["coverage"] == {
            "input_manifest_materialized_complete": True,
            "upstream_inventory_verified": False,
            "query_scan_coverage": "not_measured",
            "result_state": "exhausted",
        }
        assert wire.result_table(result) == result


@pytest.mark.parametrize("statement", ["INSERT", "COPY", "SET", "ATTACH", "CREATE", "PRAGMA"])
def test_native_statement_refusal_precedes_execution(statement: str) -> None:
    native = Rows([], statement=statement)
    with pytest.raises(wire.WireError) as caught:
        table(native, limit=10)
    assert caught.value.code == "query_refused"
    assert not native.calls


def test_native_multiple_statement_refusal_precedes_execution() -> None:
    native = Rows([])
    with pytest.raises(wire.WireError):
        table(native, limit=10, sql="multi")
    assert not native.calls


def test_native_cell_overflow_is_whole_refusal_not_a_success_prefix() -> None:
    native = Rows([("small",), ("\x00" * 11000,)], NativeType("varchar", []))
    with pytest.raises(wire.WireError) as caught:
        table(native, limit=10)
    assert caught.value.code == "output_limit"
    assert native.calls == ["SELECT value FROM dataset"]


def test_unknown_native_type_refuses_before_any_fetch() -> None:
    native = Rows([(object(),)], NativeType("timestamp_ns", []))
    with pytest.raises(wire.WireError) as caught:
        table(native, limit=10)
    assert caught.value.code == "result_type_unsupported"
    assert not native.batches


def test_exact_native_utc_identity_only_no_unknown_tzinfo_callback() -> None:
    from datetime import datetime, timedelta, tzinfo

    class Hostile(tzinfo):
        def utcoffset(self, _dt: datetime | None) -> timedelta:
            raise AssertionError("arbitrary timezone callback executed")

    dtype: dict[str, wire.JSON] = {"kind": "timestamp", "unit": "us", "timezone": "utc"}
    adapt = cast("Callable[[dict[str, wire.JSON], object, object], object]", vars(worker)["_adapt"])
    native = datetime(2040, 1, 2, tzinfo=Hostile())
    value = adapt(dtype, native, object())
    assert value is native
    with pytest.raises(wire.WireError):
        wire.native_cell(dtype, value)
    std = datetime(2040, 1, 2, tzinfo=UTC)
    assert adapt(dtype, std, object()) is std


def test_profile_values_are_not_configuration_callbacks_or_an_issuer() -> None:
    ready = worker.expected_ready("a" * 32, "b" * 64)
    assert ready["python_enable_replacements"] is False
    assert ready["python_scan_all_frames"] is False
    assert ready["memory_bytes"] == 268435456
    assert "authorized" not in ready
    assert "network" not in ready
    assert "sql" not in ready
