"""A step scorer's scores, checked in a process of their own, inside the sandbox.

``scores.parquet`` is a file the plugin wrote, and parsing it means running a parquet reader over
bytes a plugin chose. The runner holds the run's credential in memory, so it never parses that
file itself: :mod:`strata_forge.pipelines.plugin_runner` starts
``<engine python> -I -m strata_forge.plugins.checker <job.json>`` in the same sandbox the plugin
ran in (no network, no credential, the engine read-only), with read access to the staged scores
and the input steps and write access to one directory. This process checks the scores against the
input (:func:`~strata_forge.plugins.output.check_scores_file`), writes the opt-in diagnostic
aggregations beside them, and writes :data:`RECORD_FILENAME`: the counts, or why the scores were
refused. The runner reads that small JSON record (bounded, validated field by field) and copies
the diagnostics without following links; nothing else crosses back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.plugins.manifest import (
    EntryPointDecl,  # noqa: TC001 — Pydantic needs runtime resolution
)
from strata_forge.plugins.output import OutputError, check_scores_file
from strata_forge.plugins.step_scoring import Aggregation, aggregate

__all__ = [
    "DIAGNOSTICS_FILENAME",
    "RECORD_FILENAME",
    "CheckJob",
    "main",
    "run",
    "write_diagnostics",
]

#: Where a step-scoring run's opt-in per-item aggregations land: beside the scores, never in them.
DIAGNOSTICS_FILENAME = "aggregates.diagnostic.parquet"
#: The record this process leaves for the runner.
RECORD_FILENAME = "record.json"


class CheckJob(BaseModel):
    """What the runner asks this process to check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry: EntryPointDecl
    scores: Path
    steps: Path
    keep: list[str] = Field(max_length=4096)
    aggregations: tuple[Aggregation, ...] = Field(default=(), max_length=4)
    out: Path


def write_diagnostics(
    outdir: Path, rows: list[tuple[str, int, float | None]], how: tuple[Aggregation, ...]
) -> dict[str, Any] | None:
    """Write the opt-in per-item aggregations, labelled diagnostic, beside the scores."""
    if not how:
        return None
    pa: Any = __import__("pyarrow")
    pq: Any = __import__("pyarrow.parquet", fromlist=["parquet"])
    columns: dict[str, list[Any]] = {
        "item_id": [],
        "aggregation": [],
        "value": [],
        "scored_steps": [],
        "unscored_steps": [],
    }
    for name in how:
        for item_id, (value, n_scored, n_unscored) in aggregate(rows, name).items():
            columns["item_id"].append(item_id)
            columns["aggregation"].append(name)
            columns["value"].append(value)
            columns["scored_steps"].append(n_scored)
            columns["unscored_steps"].append(n_unscored)
    table = pa.table(
        {
            "item_id": pa.array(columns["item_id"], type=pa.string()),
            "aggregation": pa.array(columns["aggregation"], type=pa.string()),
            "value": pa.array(columns["value"], type=pa.float64()),
            "scored_steps": pa.array(columns["scored_steps"], type=pa.int64()),
            "unscored_steps": pa.array(columns["unscored_steps"], type=pa.int64()),
        }
    ).replace_schema_metadata({"strata.diagnostic": "true"})
    pq.write_table(table, str(outdir / DIAGNOSTICS_FILENAME))
    return {"path": DIAGNOSTICS_FILENAME, "diagnostic": True, "aggregations": list(how)}


def run(job: CheckJob) -> dict[str, Any]:
    """Check the scores and write the diagnostics; the record the runner reads back."""
    try:
        record = check_scores_file(job.entry, job.scores, job.steps, job.keep)
    except OutputError as exc:
        return {"ok": False, "error": str(exc)[:2000]}
    rows = record.pop("_rows")
    diagnostics = write_diagnostics(job.out, rows, job.aggregations)
    return {"ok": True, "scoring": record, "diagnostics": diagnostics}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m strata_forge.plugins.checker <job.json>", file=sys.stderr)
        return 2
    job = CheckJob.model_validate_json(Path(args[0]).read_text(encoding="utf-8"))
    record = run(job)
    (job.out / RECORD_FILENAME).write_text(json.dumps(record), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
