"""The child process a plugin's code runs in, on the compute target.

``<plugin env>/bin/python -I -m strata_forge.plugins.host <job.json>`` is started by
:mod:`strata_forge.pipelines.plugin_runner` inside the sandbox
(:mod:`strata_forge.plugins.sandbox`), from the plugin's own environment
(:mod:`strata_forge.plugins.environment`), never the engine's. ``-I`` keeps the user site
directory, ``PYTHON*`` variables and the working directory off ``sys.path``. The job file names
the pin, the entry point's declared interface, the validated configuration, the local input
paths, the output directory and the operator's limits; nothing else crosses over:

- the runner holds the run's Hub credential in memory only, and this process never receives it:
  not in its environment (an allow-list with every credential-shaped name removed, the Hub
  offline on an empty cache), not in a file, and not through ``/proc`` (the sandbox gives it a
  PID namespace of its own, or another uid);
- before any plugin module is imported, this process applies the operator's limits to itself and
  checks again that its environment holds exactly the pinned build (PEP 610 and every RECORD
  hash); the entry point is looked up by name in that distribution's own metadata, and each
  module on its import path must come from that distribution's own files before it is imported;
- a ``step_scorer`` never touches files: this process reads the steps, calls the scorer in
  batches and writes the scores; a ``runner`` gets read-only input paths and one output
  directory;
- progress goes to the runner as JSON lines on this process's original standard output, which is
  a pipe the runner reads and validates; standard output is pointed at standard error for the
  plugin's own code, so a ``print`` cannot interleave with it. Every metric the plugin reports is
  keyed under :data:`PLUGIN_METRIC_PREFIX`, so it reads as plugin-reported beside the GPU
  telemetry the runner samples outside the sandbox.

The runner re-checks everything this process wrote before anything is pushed: the plugin's code
shares this process, so nothing here is trusted by the runner either.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.plugins.manifest import (
    STEP_INPUT_COLUMNS,
    STEP_INPUT_ROLE,
    STEP_TYPE_COLUMN,
    EntryPointDecl,
    PluginManifestError,
    ScalarValue,
)
from strata_forge.plugins.pins import (
    PluginLimits,
    PluginPin,
    distribution_files,
    installed_mismatch,
    record_mismatch,
    registered_entry_point,
)
from strata_forge.plugins.step_scoring import (
    REASON_COLUMN,
    SCORE_COLUMN,
    StepScorer,
    StepScorerSetup,
    check_scores,
    steps_from_rows,
)
from strata_forge.training.progress import ProgressEvent

if TYPE_CHECKING:
    from collections.abc import Mapping
    from importlib.metadata import EntryPoint

__all__ = [
    "PLUGIN_METRIC_PREFIX",
    "SCORES_FILENAME",
    "HostJob",
    "PluginContext",
    "apply_limits",
    "checked_load",
    "main",
    "score_columns",
    "verify_build",
]

#: Where a step-scoring run's scores land in the output directory.
SCORES_FILENAME = "scores.parquet"
#: Bounds on what a plugin may report per event, matching the progress channel's own limits.
MAX_METRIC_KEYS = 64
MAX_METRIC_KEY_CHARS = 64
MAX_PHASE_CHARS = 200
#: The key prefix of every metric a plugin reports: plugin-reported, never forge-measured.
PLUGIN_METRIC_PREFIX = "plugin."


class HostJob(BaseModel):
    """The job file the parent writes for this process."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pin: PluginPin
    #: The entry point as its manifest declares it (its name, type and scoring contract).
    entry: EntryPointDecl
    config: dict[str, ScalarValue]
    datasets: dict[str, Path]
    models: dict[str, Path]
    output_dir: Path
    batch_size: int = Field(default=32, ge=1, le=4096)
    limits: PluginLimits = Field(default_factory=PluginLimits)


class Channel:
    """The progress pipe to the runner: one JSON line per event."""

    def __init__(self, stream: IO[str] | None) -> None:
        self._stream = stream

    def emit(self, event: ProgressEvent) -> None:
        if self._stream is None:
            return
        try:
            self._stream.write(event.model_dump_json() + "\n")
            self._stream.flush()
        except OSError:
            self._stream = None  # the runner stopped reading; the run's outcome is its call


@dataclass
class PluginContext:
    """What a ``runner`` entry point is called with."""

    config: dict[str, ScalarValue]
    #: Each declared dataset input, as a local parquet file.
    datasets: dict[str, Path]
    #: Each declared model input, as a local snapshot directory.
    models: dict[str, Path]
    #: The one directory the plugin writes; forge pushes a checked copy of it to the run's
    #: private output repo.
    output_dir: Path
    _channel: Channel | None = field(default=None, repr=False)

    def report(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
        epoch: float | None = None,
    ) -> None:
        """Report numbers the plugin computed. Shown as plugin-reported, never inferred: each key
        is recorded under :data:`PLUGIN_METRIC_PREFIX`."""
        clean: dict[str, float] = {}
        for key, value in list(metrics.items())[:MAX_METRIC_KEYS]:
            number = float(value)
            if math.isfinite(number):
                clean[(PLUGIN_METRIC_PREFIX + str(key))[:MAX_METRIC_KEY_CHARS]] = number
        self._emit(ProgressEvent(kind="step", step=step, epoch=epoch, metrics=clean))

    def phase(self, message: str) -> None:
        """Say what the plugin is doing now, in a short phrase."""
        self._emit(ProgressEvent(kind="phase", message=message[:MAX_PHASE_CHARS]))

    def _emit(self, event: ProgressEvent) -> None:
        if self._channel is not None:
            self._channel.emit(event)


def apply_limits(limits: PluginLimits) -> None:
    """Lower this process's resource limits to the operator's (soft and hard, so the plugin
    cannot raise them again). No core files: a dump would write the process's memory to disk."""
    import resource

    def lower(which: int, value: int) -> None:
        _, hard = resource.getrlimit(which)
        cap = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(which, (cap, cap))

    lower(resource.RLIMIT_CPU, limits.max_runtime_seconds)
    lower(resource.RLIMIT_FSIZE, limits.max_file_mb * 1024 * 1024)
    lower(resource.RLIMIT_NOFILE, limits.max_open_files)
    lower(resource.RLIMIT_CORE, 0)
    if limits.max_memory_mb is not None:
        lower(resource.RLIMIT_AS, limits.max_memory_mb * 1024 * 1024)


def verify_build(pin: PluginPin) -> None:
    """Refuse to go on unless this interpreter's environment holds exactly the pinned build."""
    problem = installed_mismatch(pin) or record_mismatch(pin)
    if problem is not None:
        raise PluginManifestError(problem)


def checked_load(pin: PluginPin, ep: EntryPoint) -> Any:
    """Import ``ep``'s module, one package at a time, each only after its file is found to be
    one of the pinned distribution's own files; then resolve its attribute."""
    files = distribution_files(pin)
    if files is None:
        msg = f"{pin.distribution} lists no installed files"
        raise PluginManifestError(msg)
    parts = ep.module.split(".")
    module: Any = None
    for depth in range(1, len(parts) + 1):
        name = ".".join(parts[:depth])
        spec = importlib.util.find_spec(name)
        origin = spec.origin if spec is not None else None
        if origin is None or Path(origin).resolve() not in files:
            msg = f"the entry point's module {name!r} is not one of {pin.distribution}'s files"
            raise PluginManifestError(msg)
        module = importlib.import_module(name)
    target: Any = module
    for attr in (ep.attr or "").split("."):
        if attr:
            target = getattr(target, attr)
    return target


def _load(job: HostJob) -> Any:
    ep = registered_entry_point(job.pin, job.entry.name)
    if ep is None:
        msg = f"{job.pin.distribution} registers no plugin entry point {job.entry.name!r}"
        raise PluginManifestError(msg)
    return checked_load(job.pin, ep)


def _pyarrow() -> tuple[Any, Any]:
    try:
        pa: Any = __import__("pyarrow")
        pq: Any = __import__("pyarrow.parquet", fromlist=["parquet"])
    except ImportError as exc:  # pragma: no cover -- the [hf] extra ships pyarrow
        msg = "the [hf] extra is required to run a step scorer: pip install 'strata-forge[hf]'"
        raise PluginManifestError(msg) from exc
    return pa, pq


def score_columns(entry: EntryPointDecl, input_columns: list[str]) -> list[str]:
    """The input columns a step-scoring run copies into its scores, in order."""
    keep = ["item_id", "step_index"]
    if STEP_TYPE_COLUMN in input_columns:
        keep.append(STEP_TYPE_COLUMN)
    keep.extend(entry.passthrough_columns)
    return keep


def run_step_scorer(factory: Any, job: HostJob, channel: Channel) -> int:
    """Score every step of the ``steps`` input and write :data:`SCORES_FILENAME`. Returns rows.

    The kept input columns are copied from the input file as they are (type and value), and
    ``score`` / ``reason`` are added beside them: exactly one row per input step, in input order.
    """
    pa, pq = _pyarrow()
    entry = job.entry
    if entry.score_kind is None:  # the manifest model refuses this; a typed guard
        msg = f"{entry.name!r} declares no score_kind"
        raise PluginManifestError(msg)
    scorer = factory(StepScorerSetup(config=dict(job.config), models=dict(job.models)))
    if not isinstance(scorer, StepScorer):
        msg = f"{entry.name!r} did not return a step scorer (an object with score())"
        raise PluginManifestError(msg)
    path = str(job.datasets[STEP_INPUT_ROLE])
    source = pq.ParquetFile(path)
    names = list(source.schema_arrow.names)
    read = [*STEP_INPUT_COLUMNS, *([STEP_TYPE_COLUMN] if STEP_TYPE_COLUMN in names else [])]
    total = int(source.metadata.num_rows)
    scores: list[float | None] = []
    reasons: list[str | None] = []
    for batch in source.iter_batches(batch_size=job.batch_size, columns=read):
        steps = steps_from_rows(batch.to_pylist())
        results = check_scores(
            steps, scorer.score(steps), entry.score_kind, frozenset(entry.unscored_reasons)
        )
        scores.extend(score for score, _ in results)
        reasons.extend(reason for _, reason in results)
        channel.emit(ProgressEvent(kind="step", step=len(scores), total_steps=total))
    table = pq.read_table(path, columns=score_columns(entry, names))
    table = table.append_column(SCORE_COLUMN, pa.array(scores, type=pa.float64()))
    table = table.append_column(REASON_COLUMN, pa.array(reasons, type=pa.string()))
    job.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(job.output_dir / SCORES_FILENAME))
    return len(scores)


def run(job: HostJob, channel: Channel | None = None) -> None:
    pipe = channel if channel is not None else Channel(None)
    verify_build(job.pin)
    target = _load(job)
    if job.entry.type == "step_scorer":
        run_step_scorer(target, job, pipe)
        return
    job.output_dir.mkdir(parents=True, exist_ok=True)
    target(
        PluginContext(
            config=dict(job.config),
            datasets=dict(job.datasets),
            models=dict(job.models),
            output_dir=job.output_dir,
            _channel=pipe,
        )
    )


def _progress_pipe() -> IO[str]:
    """Keep the original standard output as the progress pipe, and point file descriptor 1 at
    standard error for everything else this process prints."""
    fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(os.dup(2), "w", buffering=1)
    return os.fdopen(fd, "w", buffering=1)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m strata_forge.plugins.host <job.json>", file=sys.stderr)
        return 2
    job = HostJob.model_validate_json(Path(args[0]).read_text(encoding="utf-8"))
    apply_limits(job.limits)
    channel = Channel(_progress_pipe())
    run(job, channel)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
