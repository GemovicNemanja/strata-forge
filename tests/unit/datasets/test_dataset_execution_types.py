"""Closed native dataset values; these controls execute no native engine or Docker."""

from __future__ import annotations

import copy
from typing import cast

import pytest

from strata_forge.datasets import _dataset_wire as wire
from strata_forge.datasets import _sql_wire as sql
from strata_forge.datasets.dataset_execution_types import (
    DatasetConvertSpec,
    DatasetExecutionError,
    DatasetInputManifest,
    DatasetInstallation,
    DatasetQuerySpec,
)


def manifest(fmt: str = "csv") -> dict[str, sql.JSON]:
    return {
        "version": 1,
        "format": fmt,
        "parts": [{"ordinal": 0, "sha256": "1" * 64, "size_bytes": 20}],
    }


def schema() -> list[sql.JSON]:
    return [
        {"name": "id", "type": {"kind": "integer", "bits": 64, "signed": True}, "nullable": False}
    ]


def convert() -> dict[str, sql.JSON]:
    return {
        "version": 1,
        "format": "csv",
        "schema": schema(),
        "input_manifest_sha256": sql.digest(manifest()),
        "timeout_ms": 5000,
    }


def test_independent_canonical_values_and_fixed_error() -> None:
    original = convert()
    value = DatasetConvertSpec.from_wire(original)
    original["schema"] = []
    assert value.wire_payload() == convert()
    returned = value.wire_payload()
    returned["schema"] = []
    assert value.wire_payload() == convert()
    with pytest.raises(DatasetExecutionError) as caught:
        DatasetConvertSpec.from_wire({**convert(), "fictional_secret": "not-logged"})
    assert caught.value.__context__ is None
    assert "not-logged" not in repr(caught.value)
    assert "id" not in repr(value)


@pytest.mark.parametrize(
    "change",
    [
        {"version": True},
        {"timeout_ms": True},
        {"timeout_ms": 10001},
        {"format": "xml"},
        {"schema": []},
        {"input_manifest_sha256": "a" * 63},
    ],
)
def test_convert_refuses_closed_invalid_fields(change: dict[str, sql.JSON]) -> None:
    with pytest.raises(DatasetExecutionError):
        DatasetConvertSpec.from_wire({**convert(), **change})


def test_parquet_derives_schema_and_no_defaults() -> None:
    assert DatasetConvertSpec.from_wire({**convert(), "format": "parquet", "schema": None})
    with pytest.raises(DatasetExecutionError):
        DatasetConvertSpec.from_wire({**convert(), "format": "parquet"})
    missing = convert()
    del missing["timeout_ms"]
    with pytest.raises(DatasetExecutionError):
        DatasetConvertSpec.from_wire(missing)


@pytest.mark.parametrize("bad", ["input", "schema", "columns"])
def test_manifest_and_schema_limits(bad: str) -> None:
    if bad == "input":
        data = manifest()
        cast("list[dict[str, sql.JSON]]", data["parts"])[0]["size_bytes"] = wire.MAX_TEXT_BYTES + 1
        with pytest.raises(DatasetExecutionError):
            DatasetInputManifest.from_wire(data)
    else:
        value = convert()
        if bad == "schema":
            value["schema"] = [*schema(), *schema()]
        else:
            value["schema"] = [
                {"name": f"c{i}", "type": {"kind": "text"}, "nullable": True} for i in range(129)
            ]
        with pytest.raises(DatasetExecutionError):
            DatasetConvertSpec.from_wire(value)


def test_query_and_installation_use_distinct_profile() -> None:
    spec = {
        "version": 1,
        "sql": "select * from items",
        "inputs_sha256": "1" * 64,
        "timeout_ms": 5000,
        "max_result_rows": 1000,
    }
    assert DatasetQuerySpec.from_wire(spec).wire_payload() == spec
    with pytest.raises(DatasetExecutionError):
        DatasetQuerySpec.from_wire({**spec, "max_result_rows": 1001})
    data = {
        "version": 1,
        "profile": wire.PROFILE,
        "docker_executable": "/fictional/docker",
        "engine_socket": "/fictional/socket",
        "image_id": "sha256:" + "2" * 64,
        "worker_bundle_sha256": "3" * 64,
    }
    assert DatasetInstallation.from_wire(data).wire_payload() == data
    with pytest.raises(DatasetExecutionError):
        DatasetInstallation.from_wire({**data, "profile": sql.PROFILE})


def test_subclass_and_constructed_canonical_refuse() -> None:
    class Other(DatasetInputManifest):
        pass

    with pytest.raises(DatasetExecutionError):
        Other.from_wire(manifest())
    value = DatasetInputManifest.model_construct(canonical=b'{"fictional":"canary"}')
    with pytest.raises(DatasetExecutionError):
        value.wire_payload()


def test_new_bundle_binds_all_four_fixed_sources() -> None:
    values = dict.fromkeys(wire.SOURCES, b"fictional")
    digest = wire.source_bundle(values)
    values = copy.copy(values)
    values["_dataset_worker.py"] = b"changed"
    assert wire.source_bundle(values) != digest
    with pytest.raises(sql.WireError):
        wire.source_bundle({**values, "unselected.py": b"ignored"})
