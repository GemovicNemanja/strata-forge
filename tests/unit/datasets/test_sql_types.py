"""Closed local SQL values bind content, never authority or an executable engine."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from strata_forge.datasets.sql_types import (
    LocalParquetFiles,
    LocalSQLInstallation,
    SQLExecutionError,
    SQLInputManifest,
    SQLQuerySpec,
)

_FIXTURE = Path(__file__).parents[2] / "fixtures/headless-sql/contract-v1.json"


def fixture() -> dict[str, object]:
    return json.loads(_FIXTURE.read_bytes())


def query() -> dict[str, object]:
    return {"version": 1, "sql": "SELECT 'fictional雪'", "input_manifest_sha256": "a" * 64}


def test_metadata_contract_manifest_hash_is_independent_and_exact() -> None:
    data = fixture()
    manifest = SQLInputManifest.from_wire(data["manifest"])
    assert manifest.digest() == data["manifest_sha256"]
    assert manifest.wire_payload() == data["manifest"]
    assert data["actual_parquet_bytes_provided"] is False
    assert data["authority"] == "none"


def test_query_exact_bytes_defaults_and_privacy() -> None:
    value = SQLQuerySpec.from_wire(query())
    assert value.sql == "SELECT 'fictional雪'"
    assert value.timeout_ms == 5000
    assert value.max_result_rows == 1000
    assert value.wire_payload() == query() | {"timeout_ms": 5000, "max_result_rows": 1000}
    assert "fictional" not in repr(value)
    assert "fictional" not in str(value)
    expected = json.dumps(
        value.wire_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    assert value.canonical == expected
    assert value.digest() == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("version", True),
        ("version", 2),
        ("timeout_ms", False),
        ("timeout_ms", 99),
        ("timeout_ms", 10001),
        ("max_result_rows", 0),
        ("max_result_rows", 10001),
        ("sql", ""),
        ("sql", "x" * 32769),
        ("sql", "\ud800"),
        ("input_manifest_sha256", "a" * 64 + "\n"),
        ("input_manifest_sha256", "A" * 64),
        ("engine", "fictional"),
        ("image", "fictional"),
        ("path", "/not-opened"),
        ("authorized", True),
        ("callback", "not-imported"),
    ],
)
def test_query_forbids_unsupported_or_malformed_fields(key: str, bad: object) -> None:
    request = query() | {key: bad}
    with pytest.raises(SQLExecutionError) as caught:
        SQLQuerySpec.from_wire(request)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "fictional" not in str(caught.value)


def test_returned_payload_and_original_nested_input_cannot_mutate_manifest() -> None:
    source = fixture()["manifest"]
    value = SQLInputManifest.from_wire(cast("dict[str, object]", source))
    original = value.canonical
    # Independent JSON copies ensure callers do not receive a prepared dictionary alias.
    output = value.wire_payload()
    output["parts"] = []
    assert value.canonical == original
    assert value.wire_payload() == source


@pytest.mark.parametrize(
    "change", ["order", "bytes", "rows", "sha", "schema", "collision", "extra", "empty"]
)
def test_manifest_cross_fields_fail_closed(change: str) -> None:
    source = copy.deepcopy(fixture()["manifest"])
    assert isinstance(source, dict)
    if change == "order":
        source["parts"][0]["ordinal"] = 1
    elif change == "bytes":
        source["parts"][0]["size_bytes"] = 67108865
    elif change == "rows":
        source["expected_rows"] += 1
    elif change == "sha":
        source["schema_sha256"] = "0" * 64
    elif change == "schema":
        source["schema"][0]["type"]["bits"] = 32
    elif change == "collision":
        source["schema"][1]["name"] = "ID"
    elif change == "empty":
        source["parts"] = []
    else:
        source["authorized"] = True
    with pytest.raises(SQLExecutionError):
        SQLInputManifest.from_wire(cast("dict[str, object]", source))


def test_forged_construct_copy_subclass_and_instance_metadata_do_not_bypass() -> None:
    bad = SQLQuerySpec.model_construct(canonical=b'{"version":true}')
    with pytest.raises(SQLExecutionError):
        bad.wire_payload()
    good = SQLQuerySpec.from_wire(query())
    copied = good.model_copy(update={"canonical": b"fictional"})
    with pytest.raises(SQLExecutionError):
        copied.digest()
    object.__setattr__(good, "KIND", "installation")
    object.__setattr__(good, "LIMIT", 0)
    assert good.sql == query()["sql"]

    class Other(SQLQuerySpec):
        pass

    with pytest.raises(SQLExecutionError):
        Other.from_wire(query())


def test_pydantic_direct_validation_has_a_fixed_content_free_error() -> None:
    with pytest.raises(SQLExecutionError) as caught:
        SQLQuerySpec(canonical=b"fictional-private-query")
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "private" not in str(caught.value)


def test_installation_is_only_an_operator_description() -> None:
    data = {
        "version": 1,
        "profile": "sql_local_arm64_v1",
        "docker_executable": "/not-opened/docker",
        "engine_socket": "/not-opened/socket",
        "image_id": "sha256:" + "b" * 64,
        "worker_bundle_sha256": "c" * 64,
    }
    value = LocalSQLInstallation.from_wire(data)
    assert value.wire_payload() == data
    assert "socket" not in repr(value)
    for key, bad in [
        ("image_id", "moving:tag"),
        ("engine_socket", "tcp://outside"),
        ("docker_executable", "/bad/../docker"),
    ]:
        with pytest.raises(SQLExecutionError):
            LocalSQLInstallation.from_wire(data | {key: bad})


def test_file_descriptions_do_not_open_or_authorize_anything() -> None:
    manifest = SQLInputManifest.from_wire(fixture()["manifest"])
    source = LocalParquetFiles.from_open_fds(manifest=manifest, fds=(999999,))
    assert source.checked() == (manifest, (999999,))
    assert "999999" not in repr(source)
    with pytest.raises(SQLExecutionError):
        LocalParquetFiles.from_open_fds(manifest=manifest, fds=(True,))
    forged = source.model_copy(update={"fds": (-1,)})
    with pytest.raises(SQLExecutionError):
        forged.checked()


class Hostile:
    def __str__(self) -> str:
        raise AssertionError("str executed")

    def __repr__(self) -> str:
        raise AssertionError("repr executed")

    def __hash__(self) -> int:
        raise AssertionError("hash executed")

    def __eq__(self, _other: object) -> bool:
        raise AssertionError("eq executed")


@pytest.mark.parametrize("code", [None, {}, [], Hostile(), True])
def test_error_constructor_does_not_execute_rejected_values(code: object) -> None:
    error = SQLExecutionError(code)
    assert error.code == "unavailable"
    assert str(error) == "sql_execution_refused"
