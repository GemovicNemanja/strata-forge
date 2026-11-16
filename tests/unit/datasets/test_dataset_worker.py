"""Scalar conversion controls, not native worker/profile execution evidence."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# Fault controls inspect fixed private cleanup/worker boundaries without runtime bypasses.
import io
import time
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from strata_forge.datasets import _dataset_worker as worker
from strata_forge.datasets import _sql_wire as sql
from strata_forge.datasets._dataset_worker import _scalar


@pytest.mark.parametrize(
    ("dtype", "value", "expected"),
    [
        ({"kind": "text"}, "雪 {{literal}}", "雪 {{literal}}"),
        ({"kind": "text"}, "", ""),
        ({"kind": "boolean"}, "false", False),
        ({"kind": "boolean"}, "true", True),
        ({"kind": "integer", "bits": 64, "signed": True}, "9007199254740993", 9007199254740993),
        ({"kind": "float", "bits": 64}, "1.25", 1.25),
        ({"kind": "decimal", "precision": 4, "scale": 2}, "1.25", Decimal("1.25")),
        ({"kind": "date"}, "2040-01-02", date(2040, 1, 2)),
        (
            {"kind": "timestamp", "unit": "us", "timezone": "none"},
            "2040-01-02T03:04:05.000006",
            datetime(2040, 1, 2, 3, 4, 5, 6),
        ),
    ],
)
def test_explicit_csv_scalar(dtype: dict[str, sql.JSON], value: str, expected: object) -> None:
    assert _scalar(value, dtype, is_csv=True, nullable=False) == expected


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        ({"kind": "boolean"}, "True"),
        ({"kind": "integer", "bits": 64, "signed": True}, "01"),
        ({"kind": "integer", "bits": 64, "signed": True}, "+1"),
        ({"kind": "integer", "bits": 64, "signed": True}, str(2**63)),
        ({"kind": "float", "bits": 64}, "NaN"),
        ({"kind": "float", "bits": 64}, "1e999"),
        ({"kind": "decimal", "precision": 4, "scale": 2}, "1.2"),
        ({"kind": "decimal", "precision": 4, "scale": 2}, "-0.00"),
        ({"kind": "date"}, "20400102"),
        ({"kind": "timestamp", "unit": "us", "timezone": "utc"}, "2040-01-02T03:04:05"),
    ],
)
def test_scalar_refuses_silent_coercion(dtype: dict[str, sql.JSON], value: str) -> None:
    with pytest.raises(sql.WireError):
        _scalar(value, dtype, is_csv=True, nullable=False)


def test_null_marker_and_json_native_types() -> None:
    assert _scalar(r"\N", {"kind": "text"}, is_csv=True, nullable=True) is None
    assert _scalar(None, {"kind": "text"}, is_csv=False, nullable=True) is None
    with pytest.raises(sql.WireError):
        _scalar(r"\N", {"kind": "text"}, is_csv=True, nullable=False)
    with pytest.raises(sql.WireError):
        _scalar(True, {"kind": "integer", "bits": 64, "signed": True}, is_csv=False, nullable=False)


@pytest.mark.parametrize("content", ['x\na"b\n', "x\rhello\r"])
def test_csv_lexical_newline_and_quote_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    path = tmp_path / "input.csv"
    path.write_bytes(content.encode())

    def selected_path(_: object) -> Path:
        return path

    monkeypatch.setattr(worker, "Path", selected_path)

    class NativeText:
        id = "varchar"

    class Connection:
        description: ClassVar[list[tuple[str, NativeText]]] = [("x", NativeText())]

        def execute(self, statement: str, parameters: object = None) -> Connection:
            return self

    spec: dict[str, sql.JSON] = {
        "format": "csv",
        "schema": [{"name": "x", "type": {"kind": "text"}, "nullable": False}],
    }
    with pytest.raises(sql.WireError):
        worker._text_input(
            cast("worker.native._Connection", Connection()), spec, time.monotonic() + 1
        )


@pytest.mark.parametrize(
    "content", ['x\r\n"a""b"\r\n', 'x\n"a\nb"\n', 'x\r\n"a\r\nb"\r\n', 'x\n""\n', "x\nhello"]
)
def test_csv_grammar_keeps_escaped_and_embedded_text(content: str) -> None:
    worker._csv_grammar(io.StringIO(content), time.monotonic() + 1)


@pytest.mark.parametrize(
    "content", ['x\na"b\n', "x\rhello\r", 'x\n"a"x\n', 'x\n"unclosed', 'x\n"a\rb"\n']
)
def test_csv_grammar_refuses_malformed_records(content: str) -> None:
    with pytest.raises(sql.WireError):
        worker._csv_grammar(io.StringIO(content), time.monotonic() + 1)


def test_csv_grammar_handles_chunk_boundary() -> None:
    valid = 'x\n"' + ("a" * 65531) + '\r\nb"\n'
    worker._csv_grammar(io.StringIO(valid), time.monotonic() + 1)
    with pytest.raises(sql.WireError):
        worker._csv_grammar(io.StringIO(valid.replace("\r\n", "\rx", 1)), time.monotonic() + 1)
