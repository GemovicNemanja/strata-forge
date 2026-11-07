"""Scalar conversion controls, not native worker/profile execution evidence."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

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
