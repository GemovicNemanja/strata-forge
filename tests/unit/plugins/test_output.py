"""The plugin's output is copied without following links before anything reads or pushes it,
and a step scorer's scores are checked by the runner against the input."""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from strata_forge.plugins.manifest import EntryPointDecl
from strata_forge.plugins.output import OutputError, check_scores_file, stage_output

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture
def secret(tmp_path: Path) -> Path:
    folder = tmp_path / "secret_dir"
    folder.mkdir()
    (folder / "id_ed25519").write_text("PRIVATE")
    return folder


def test_regular_files_are_copied_and_hashed(tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / "sub").mkdir(parents=True)
    (out / "a.txt").write_text("alpha")
    (out / "sub" / "b.bin").write_bytes(b"\x00\x01")
    staged = tmp_path / "staged"
    hashes = stage_output(out, staged)
    assert hashes == {
        "a.txt": hashlib.sha256(b"alpha").hexdigest(),
        "sub/b.bin": hashlib.sha256(b"\x00\x01").hexdigest(),
    }
    assert (staged / "sub" / "b.bin").read_bytes() == b"\x00\x01"
    assert not (staged / "a.txt").is_symlink()


def test_an_output_directory_that_is_a_link_is_refused(tmp_path: Path, secret: Path) -> None:
    out = tmp_path / "out"
    out.symlink_to(secret)
    with pytest.raises(OutputError, match="is a link"):
        stage_output(out, tmp_path / "staged")
    assert not (tmp_path / "staged").exists()


def test_a_hard_link_to_a_readable_file_is_refused(tmp_path: Path, secret: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    os.link(secret / "id_ed25519", out / "copy")
    with pytest.raises(OutputError, match="hard link"):
        stage_output(out, tmp_path / "staged")


@pytest.mark.parametrize("target", ["file", "dir"])
def test_a_link_anywhere_inside_is_refused(tmp_path: Path, secret: Path, target: str) -> None:
    out = tmp_path / "out"
    (out / "deep").mkdir(parents=True)
    (out / "deep" / "leak").symlink_to(secret / "id_ed25519" if target == "file" else secret)
    with pytest.raises(OutputError, match="link"):
        stage_output(out, tmp_path / "staged")


def test_a_special_file_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    os.mkfifo(out / "pipe")
    with pytest.raises(OutputError, match="special file"):
        stage_output(out, tmp_path / "staged")


def test_a_missing_output_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(OutputError, match="no output directory"):
        stage_output(tmp_path / "absent", tmp_path / "staged")


ENTRY = EntryPointDecl.model_validate(
    {
        "name": "prm",
        "type": "step_scorer",
        "inputs": [
            {"role": "steps", "required_columns": ["item_id", "step_index", "context", "step"]}
        ],
        "outputs": [{"role": "scores", "kind": "dataset"}],
        "score_kind": "probability",
        "unscored_reasons": ["unparseable"],
    }
)
KEEP = ["item_id", "step_index"]


def _files(
    tmp_path: Path, change: Callable[[dict[str, Any]], None] | None = None
) -> tuple[Path, Path]:
    steps = tmp_path / "steps.parquet"
    pq.write_table(
        pa.table(
            {
                "item_id": ["a", "a", "b"],
                "step_index": [0, 1, 0],
                "context": ["c"] * 3,
                "step": ["x", "y", "z"],
            }
        ),
        str(steps),
    )
    columns: dict[str, Any] = {
        "item_id": pa.array(["a", "a", "b"]),
        "step_index": pa.array([0, 1, 0], type=pa.int64()),
        "score": pa.array([0.1, None, 1.0], type=pa.float64()),
        "reason": pa.array([None, "unparseable", None], type=pa.string()),
    }
    if change is not None:
        change(columns)
    scores = tmp_path / "scores.parquet"
    pq.write_table(pa.table(columns), str(scores))
    return steps, scores


def test_scores_that_match_the_input_are_counted(tmp_path: Path) -> None:
    steps, scores = _files(tmp_path)
    record = check_scores_file(ENTRY, scores, steps, KEEP)
    assert record["rows"] == 3
    assert record["scored"] == 2
    assert record["unscored"] == {"unparseable": 1}


TRUNCATED = "truncated"


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"item_id": pa.array(["a", "a", "c"])}, "'item_id' column does not match"),
        ({"step_index": pa.array([0, 2, 0], type=pa.int64())}, "'step_index'"),
        ({"step_index": pa.array([0, 1, 0], type=pa.int32())}, "'step_index'"),
        ({"score": pa.array([0.1, None, 1.5])}, "probability"),
        ({"score": pa.array([0.1, None, float("nan")])}, "non-finite"),
        ({"score": pa.array([0.1, 0.2, 1.0])}, "exactly one"),
        ({"reason": pa.array([None, "tired", None])}, "undeclared reason"),
        ({"score": pa.array(["0.1", None, "1"])}, "float64"),
        ({"extra": pa.array([1, 2, 3])}, "columns"),
        ({TRUNCATED: True}, "infrastructure fault: 2 score rows for 3"),
    ],
)
def test_scores_outside_the_contract_are_refused(
    tmp_path: Path, updates: dict[str, Any], match: str
) -> None:
    def change(columns: dict[str, Any]) -> None:
        if updates.get(TRUNCATED):
            for name, array in list(columns.items()):
                columns[name] = array.slice(0, 2)
            return
        columns.update(updates)

    steps, scores = _files(tmp_path, change)
    with pytest.raises(OutputError, match=match):
        check_scores_file(ENTRY, scores, steps, KEEP)


def test_an_unreadable_scores_file_is_an_infrastructure_fault(tmp_path: Path) -> None:
    steps, scores = _files(tmp_path)
    scores.write_bytes(b"not parquet")
    with pytest.raises(OutputError, match="cannot be read"):
        check_scores_file(ENTRY, scores, steps, KEEP)
