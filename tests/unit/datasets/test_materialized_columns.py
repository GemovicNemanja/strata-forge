"""Pre-CTAS schema controls; actual image execution has separate conformance."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# Fixed worker fault controls never substitute for actual image evidence.
import time
from typing import ClassVar, cast

import pytest

from strata_forge.datasets import _dataset_worker as worker
from strata_forge.datasets import _sql_wire as sql


class NativeInteger:
    id = "bigint"


class Projection:
    def __init__(self, names: list[str]) -> None:
        self.description: list[tuple[object, ...]] = [(name, NativeInteger()) for name in names]


class Connection:
    description: ClassVar[list[tuple[object, ...]]] = []

    def __init__(self, names: list[str]) -> None:
        self.projection = Projection(names)
        self.bound: list[str] = []
        self.executed: list[str] = []

    def sql(self, query: str) -> Projection:
        self.bound.append(query)
        return self.projection

    def execute(self, query: str) -> Connection:
        self.executed.append(query)
        return self


@pytest.mark.parametrize("names", [["same", "same"], ["Case", "case"], ["Straße", "STRASSE"]])
def test_ambiguous_names_refuse_before_ctas(
    monkeypatch: pytest.MonkeyPatch, names: list[str]
) -> None:
    conn = Connection(names)

    def no_output(*_: object) -> tuple[list[sql.JSON], int]:
        pytest.fail("output validation must not run after rejected projection")

    monkeypatch.setattr(worker, "_validated_output", no_output)
    with pytest.raises(sql.WireError) as error:
        worker._materialized_output(
            cast("worker.native._Connection", conn),
            "SELECT exact_sql",
            object(),
            time.monotonic() + 1,
        )
    assert error.value.code == "query_refused"
    assert conn.bound == ["SELECT exact_sql"]
    assert conn.executed == []


def test_unique_projection_is_bound_once_and_ctas_uses_exact_original_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = Connection(["left_id", "right_id"])
    expected: list[sql.JSON] = [
        {"name": name, "type": {"kind": "integer", "bits": 64, "signed": True}}
        for name in ["left_id", "right_id"]
    ]
    utc = object()

    def output(actual: object, zone: object, deadline: float) -> tuple[list[sql.JSON], int]:
        assert actual is conn
        assert zone is utc
        assert time.monotonic() < deadline
        return expected, 2

    monkeypatch.setattr(worker, "_validated_output", output)
    query = ' SELECT id AS "left_id",id+1 AS "right_id"\r\nFROM items '
    assert worker._materialized_output(
        cast("worker.native._Connection", conn), query, utc, time.monotonic() + 1
    ) == (expected, 2)
    assert conn.bound == [query]
    assert conn.executed == ["CREATE TABLE strata_result AS " + query]


@pytest.mark.parametrize("change", ["name", "type"])
def test_ctas_schema_change_refuses_instead_of_publishing_a_coercion(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    conn = Connection(["exact"])
    actual: list[sql.JSON] = [
        {
            "name": "renamed" if change == "name" else "exact",
            "type": {"kind": "text"}
            if change == "type"
            else {"kind": "integer", "bits": 64, "signed": True},
        }
    ]
    monkeypatch.setattr(worker, "_validated_output", lambda *_: (actual, 2))
    with pytest.raises(sql.WireError) as error:
        worker._materialized_output(
            cast("worker.native._Connection", conn), "SELECT exact", object(), time.monotonic() + 1
        )
    assert error.value.code == "result_type_unsupported"


def test_expired_original_budget_cannot_bind_or_start_ctas() -> None:
    conn = Connection(["exact"])
    with pytest.raises(sql.WireError) as error:
        worker._materialized_output(
            cast("worker.native._Connection", conn), "SELECT exact", object(), time.monotonic() - 1
        )
    assert error.value.code == "deadline"
    assert conn.bound == conn.executed == []
