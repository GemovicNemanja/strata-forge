"""The child process a plugin's code runs in, on the compute target.

``python -m strata_forge.plugins.host <job.json>`` is started by
:mod:`strata_forge.pipelines.plugin_runner` once it has verified the pinned build and downloaded
the declared inputs. The job file names the distribution, the entry point, the validated
configuration, the local input paths and the output directory; nothing else crosses over:

- the parent holds the run's Hub credential, and this process never receives it (the secrets file
  is deleted before the parent starts work, the environment is an allow-list with every
  credential-shaped name removed, and the Hub libraries are put offline on an empty cache);
- the entry point is looked up by name in the pinned distribution's own metadata, never imported
  from a path;
- a ``step_scorer`` never touches files: this process reads the steps, calls the scorer in
  batches and writes the scores; a ``runner`` gets read-only input paths and one output
  directory, and reports metrics through :class:`PluginContext`.

This is credential separation, not isolation from hostile code: the plugin is code its owner chose
to install and the operator allow-listed, running on the owner's own machine.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.plugins.manifest import (
    STEP_INPUT_COLUMNS,
    STEP_INPUT_ROLE,
    PluginManifestError,
    ScalarValue,
)
from strata_forge.plugins.pins import PluginPin, registered_entry_point
from strata_forge.plugins.step_scoring import (
    SCORE_COLUMNS,
    StepScorer,
    StepScorerSetup,
    check_scores,
    steps_from_rows,
)
from strata_forge.training.progress import JsonlProgressWriter, ProgressEvent

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["SCORES_FILENAME", "HostJob", "PluginContext", "main"]

#: Where a step-scoring run's scores land in the output directory.
SCORES_FILENAME = "scores.parquet"
#: Bounds on what a plugin may report per event, matching the progress channel's own limits.
MAX_METRIC_KEYS = 64
MAX_METRIC_KEY_CHARS = 64
MAX_PHASE_CHARS = 200


class HostJob(BaseModel):
    """The job file the parent writes for this process."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pin: PluginPin
    entry_point: str
    type: Literal["runner", "step_scorer"]
    config: dict[str, ScalarValue]
    datasets: dict[str, Path]
    models: dict[str, Path]
    output_dir: Path
    progress_path: str | None = None
    batch_size: int = Field(default=32, ge=1, le=4096)


@dataclass
class PluginContext:
    """What a ``runner`` entry point is called with."""

    config: dict[str, ScalarValue]
    #: Each declared dataset input, as a local parquet file.
    datasets: dict[str, Path]
    #: Each declared model input, as a local snapshot directory.
    models: dict[str, Path]
    #: The one directory the plugin writes; forge pushes it to the run's private output repo.
    output_dir: Path
    _writer: JsonlProgressWriter | None = field(default=None, repr=False)

    def report(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
        epoch: float | None = None,
    ) -> None:
        """Report numbers the plugin computed. Shown as emitted by the plugin, never inferred."""
        clean: dict[str, float] = {}
        for key, value in list(metrics.items())[:MAX_METRIC_KEYS]:
            number = float(value)
            if math.isfinite(number):
                clean[str(key)[:MAX_METRIC_KEY_CHARS]] = number
        self._emit(ProgressEvent(kind="step", step=step, epoch=epoch, metrics=clean))

    def phase(self, message: str) -> None:
        """Say what the plugin is doing now, in a short phrase."""
        self._emit(ProgressEvent(kind="phase", message=message[:MAX_PHASE_CHARS]))

    def _emit(self, event: ProgressEvent) -> None:
        if self._writer is not None:
            self._writer.emit(event)


def _load(job: HostJob) -> Any:
    ep = registered_entry_point(job.pin, job.entry_point)
    if ep is None:
        msg = f"{job.pin.distribution} registers no plugin entry point {job.entry_point!r}"
        raise PluginManifestError(msg)
    return ep.load()


def _pyarrow() -> tuple[Any, Any]:
    try:
        pa: Any = __import__("pyarrow")
        pq: Any = __import__("pyarrow.parquet", fromlist=["parquet"])
    except ImportError as exc:  # pragma: no cover -- the [hf] extra ships pyarrow
        msg = "the [hf] extra is required to run a step scorer: pip install 'strata-forge[hf]'"
        raise PluginManifestError(msg) from exc
    return pa, pq


def run_step_scorer(factory: Any, job: HostJob, writer: JsonlProgressWriter | None) -> int:
    """Score every step of the ``steps`` input and write :data:`SCORES_FILENAME`. Returns rows."""
    pa, pq = _pyarrow()
    scorer = factory(StepScorerSetup(config=dict(job.config), models=dict(job.models)))
    if not isinstance(scorer, StepScorer):
        msg = f"{job.entry_point!r} did not return a step scorer (an object with score())"
        raise PluginManifestError(msg)
    source = pq.ParquetFile(str(job.datasets[STEP_INPUT_ROLE]))
    total = int(source.metadata.num_rows)
    item_ids: list[str] = []
    indices: list[int] = []
    scores: list[float] = []
    for batch in source.iter_batches(batch_size=job.batch_size, columns=list(STEP_INPUT_COLUMNS)):
        steps = steps_from_rows(batch.to_pylist())
        values = check_scores(steps, scorer.score(steps))
        item_ids.extend(s.item_id for s in steps)
        indices.extend(s.step_index for s in steps)
        scores.extend(values)
        if writer is not None:
            writer.emit(ProgressEvent(kind="step", step=len(scores), total_steps=total))
    table = pa.table(
        {
            SCORE_COLUMNS[0]: pa.array(item_ids, type=pa.string()),
            SCORE_COLUMNS[1]: pa.array(indices, type=pa.int64()),
            SCORE_COLUMNS[2]: pa.array(scores, type=pa.float64()),
        }
    )
    job.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(job.output_dir / SCORES_FILENAME))
    return len(scores)


def run(job: HostJob) -> None:
    writer = JsonlProgressWriter(job.progress_path) if job.progress_path else None
    try:
        target = _load(job)
        if job.type == "step_scorer":
            run_step_scorer(target, job, writer)
            return
        job.output_dir.mkdir(parents=True, exist_ok=True)
        target(
            PluginContext(
                config=dict(job.config),
                datasets=dict(job.datasets),
                models=dict(job.models),
                output_dir=job.output_dir,
                _writer=writer,
            )
        )
    finally:
        if writer is not None:
            writer.close()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m strata_forge.plugins.host <job.json>", file=sys.stderr)
        return 2
    job = HostJob.model_validate_json(Path(args[0]).read_text(encoding="utf-8"))
    run(job)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
