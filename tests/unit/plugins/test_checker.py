"""The scores check that runs in the sandbox, and the runner's reading of what it returns."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from strata_forge.pipelines import plugin_runner
from strata_forge.pipelines._common import RunError
from strata_forge.plugins.checker import (
    DIAGNOSTICS_FILENAME,
    RECORD_FILENAME,
    CheckJob,
    main,
)
from strata_forge.plugins.manifest import EntryPointDecl
from strata_forge.plugins.step_scoring import REASON_COLUMN, SCORE_COLUMN

from .conftest import MANIFEST

if TYPE_CHECKING:
    from pathlib import Path

ENTRY = EntryPointDecl.model_validate(
    next(e for e in MANIFEST["entry_points"] if e["name"] == "prm")
).model_copy(update={"unscored_reasons": ("unparseable",), "passthrough_columns": ()})
KEEP = ["item_id", "step_index"]


def _files(tmp_path: Path, scores: list[float | None]) -> tuple[Path, Path, Path]:
    steps = tmp_path / "steps.parquet"
    n = len(scores)
    pq.write_table(
        pa.table(
            {
                "item_id": [f"i{k // 2}" for k in range(n)],
                "step_index": list(range(n)),
                "text": ["t"] * n,
            }
        ),
        steps,
    )
    written = tmp_path / "scores.parquet"
    pq.write_table(
        pa.table(
            {
                "item_id": [f"i{k // 2}" for k in range(n)],
                "step_index": list(range(n)),
                SCORE_COLUMN: pa.array(scores, type=pa.float64()),
                REASON_COLUMN: pa.array(
                    [None if s is not None else "unparseable" for s in scores], type=pa.string()
                ),
            }
        ),
        written,
    )
    out = tmp_path / "out"
    out.mkdir()
    return steps, written, out


def _check(tmp_path: Path, scores: list[float | None], **job: Any) -> dict[str, Any]:
    steps, written, out = _files(tmp_path, scores)
    path = tmp_path / "job.json"
    path.write_text(
        CheckJob(
            entry=ENTRY, scores=written, steps=steps, keep=KEEP, out=out, **job
        ).model_dump_json()
    )
    assert main([str(path)]) == 0
    return json.loads((out / RECORD_FILENAME).read_text())


def test_the_check_counts_and_writes_the_diagnostics_beside_the_scores(tmp_path: Path) -> None:
    record = _check(tmp_path, [0.5, None, 0.9, 0.1], aggregations=("min",))
    assert record["ok"]
    assert record["scoring"] == {
        "score_kind": "probability",
        "rows": 4,
        "scored": 3,
        "unscored": {"unparseable": 1},
    }
    assert record["diagnostics"]["path"] == DIAGNOSTICS_FILENAME
    assert (tmp_path / "out" / DIAGNOSTICS_FILENAME).is_file()
    # What the runner believes of it.
    checked = plugin_runner._check_record(json.dumps(record).encode(), ENTRY, 4)  # pyright: ignore[reportPrivateUsage]
    assert checked["scoring"]["scored"] == 3


def test_scores_out_of_range_are_refused_with_a_reason(tmp_path: Path) -> None:
    record = _check(tmp_path, [0.5, 7.0])
    assert record["ok"] is False
    assert "row 1" in record["error"]
    with pytest.raises(RunError, match="row 1"):
        plugin_runner._check_record(json.dumps(record).encode(), ENTRY, 2)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "scoring",
    [
        {"score_kind": "logit", "rows": 2, "scored": 2, "unscored": {}},
        {"score_kind": "probability", "rows": 3, "scored": 3, "unscored": {}},
        {"score_kind": "probability", "rows": 2, "scored": 1, "unscored": {"made_up": 1}},
        {"score_kind": "probability", "rows": 2, "scored": 5, "unscored": {}},
        {"score_kind": "probability", "rows": 2, "scored": 2, "unscored": {}, "extra": 1},
    ],
)
def test_a_record_that_does_not_add_up_is_refused(scoring: dict[str, Any]) -> None:
    raw = json.dumps({"ok": True, "scoring": scoring}).encode()
    with pytest.raises(RunError, match="infrastructure fault"):
        plugin_runner._check_record(raw, ENTRY, 2)  # pyright: ignore[reportPrivateUsage]


def test_an_oversized_or_malformed_record_is_refused() -> None:
    with pytest.raises(RunError, match="oversized"):
        plugin_runner._check_record(b"x" * (plugin_runner.MAX_CHECK_RECORD_BYTES + 1), ENTRY, 2)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RunError, match="malformed"):
        plugin_runner._check_record(b"[]", ENTRY, 2)  # pyright: ignore[reportPrivateUsage]
