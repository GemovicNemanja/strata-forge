"""Structured, append-only training-progress events.

``strata_forge.training`` runners are synchronous and only return once training
finishes; :class:`strata_forge.compute.Backend` exposes ``logs`` (a *pull*) but no push
channel. To let an external orchestrator surface **live** metrics, a runner can
write one :class:`ProgressEvent` per training event to a JSONL file
(:class:`JsonlProgressWriter`) that the orchestrator tails (e.g. ``tail -F`` over
SSH).

Work that has no step to count (downloading a split, launching a model server, uploading
results) reports itself with ``kind="phase"`` events, so the orchestrator can say what is
happening instead of showing an indeterminate wait between two countable milestones.

The file is plain JSONL — one ``ProgressEvent`` per line — so it needs no extra
runtime dependency and degrades gracefully even if a line lands in stdout.
:func:`trainer_callback` bridges TRL / ``transformers`` ``Trainer`` events onto a
writer; ``transformers`` is imported lazily inside the factory, so importing this
module never requires the ``[finetuning]`` extra. :func:`attach` is the one-liner
runners use to wire a writer onto a trainer when a path (or the
``FORGE_PROGRESS_PATH`` env var) is set.
"""

from __future__ import annotations

import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.training.hardware import GpuSampler

__all__ = [
    "MAX_METRIC_KEYS",
    "MAX_METRIC_KEY_CHARS",
    "RUN_STAGES",
    "JsonlProgressWriter",
    "ProgressEvent",
    "ProgressKind",
    "RunStage",
    "attach",
    "coerce_float",
    "coerce_int",
    "numeric_metrics",
    "trainer_callback",
]


# ``phase`` is listed first because it is the only kind that can precede ``start``: it reports
# what a long provisioning step is doing (downloading data, launching a model server, uploading
# results) so the stretches between countable milestones are not a silent gap to whoever tails
# the JSONL. A ``phase`` event carries no step count — but it may carry ``stage`` and ``metrics``,
# because neither is a count: one says WHICH milestone the prose belongs to, the other reports the
# machine underneath it, and both stay meaningful during the long uncountable stretches.
type ProgressKind = Literal["phase", "start", "step", "eval", "checkpoint", "end", "error"]

# The coarse milestone a run is in, as an ID rather than prose. ``message`` says what is happening
# in words that change constantly ("Loading the dataset", "Starting the model server"); ``stage``
# says which of a fixed, ordered set of milestones that sentence belongs to, so an orchestrator can
# render an ordered stepper without pattern-matching English.
#
# A runner only ever reports the last three. ``provision`` and ``install_engine`` describe the VM
# before the runner's own process exists — the orchestrator that submitted the job owns those, and
# infers them from the job's own lifecycle. Dataset loading is reported as ``load_model``: it is
# not the model, but it is the same "getting ready to run" milestone from the watcher's side, and a
# stage the user never sees a separate label for is a stage that should not exist.
type RunStage = Literal["provision", "install_engine", "load_model", "run", "push"]

RUN_STAGES: tuple[RunStage, ...] = (
    "provision",
    "install_engine",
    "load_model",
    "run",
    "push",
)
"""Every stage in order. Consumers rendering a stepper should read this rather than hardcode it."""

# A metric key is a string on a channel that is tailed, relayed and rendered. `message` has had
# a cap since the sink was written; these give `metrics` the same treatment on both axes.
MAX_METRIC_KEYS = 64
MAX_METRIC_KEY_CHARS = 64

# Keys promoted to dedicated :class:`ProgressEvent` fields, so they aren't
# duplicated inside ``metrics``.
_PROMOTED = frozenset({"loss", "eval_loss", "learning_rate", "epoch"})


class ProgressEvent(BaseModel):
    """One structured training-progress record.

    A ``phase`` event is the exception to the shape below: it carries no step, epoch or
    loss, because it marks work that has no step to count. It may still carry ``stage``
    and ``metrics`` — neither is a count, and both stay true while nothing is countable.

    Attributes:
        kind: The lifecycle milestone this event marks, or ``phase`` for a free-form
            report of what a long uncountable step is currently doing.
        stage: Which coarse, ordered milestone this event belongs to, when the runner knows.
            Independent of ``kind``: ``kind`` says what sort of record this is, ``stage`` says
            where in the run it sits. ``None`` on events emitted before a stage is established.
        step: Global optimizer step, when known.
        total_steps: Total planned optimizer steps, when known.
        epoch: Fractional epoch, when known.
        loss: Training (or eval) loss at this point, when reported.
        learning_rate: LR at this step, when reported.
        metrics: Any additional numeric values (eval metrics, grad norm, …),
            excluding the values already promoted to their own fields.
        message: Free-form human-readable detail (the payload of ``phase``, and used by
            ``error``).
        ts: Event timestamp (UTC).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ProgressKind
    stage: RunStage | None = None
    step: int | None = None
    total_steps: int | None = None
    epoch: float | None = None
    loss: float | None = None
    learning_rate: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    message: str = ""
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


def coerce_float(value: Any) -> float | None:
    """Best-effort cast to ``float``; ``None`` for bools, non-numerics and non-finite values.

    Shared with the runners, which report the same trainer numbers from ``train()``'s return
    value rather than from a callback — the same values reached two different ways.

    NaN and infinity are dropped rather than passed through, and that matters more than it looks:
    ``loss=nan`` and ``grad_norm=inf`` are routine in fp16 training, not adversarial. Pydantic
    serializes them to JSON ``null``, which ``metrics: dict[str, float]`` then REFUSES to parse
    back — so an ordinary gradient explosion would emit a progress line the orchestrator cannot
    read. A dropped key degrades to "not reported"; an unparseable line loses the whole event.
    """
    if value is None or isinstance(value, bool):  # bool is an int subclass — reject it
        return None
    try:
        out = float(value)
    except TypeError, ValueError:
        return None
    return out if math.isfinite(out) else None


def coerce_int(value: Any) -> int | None:
    f = coerce_float(value)
    return int(f) if f is not None else None


def _numeric_extras(
    values: dict[str, Any], *, exclude: frozenset[str] = _PROMOTED
) -> dict[str, float]:
    """The numeric entries of ``values`` (as floats), minus ``exclude``.

    Per-event callbacks exclude the keys promoted to their own :class:`ProgressEvent` fields;
    a whole-run summary excludes nothing, because there is nowhere else for those numbers to go.

    Keys are bounded in both length and count. ``values`` is the trainer's log dict, and
    :func:`trainer_callback` is public API: a consumer whose ``compute_metrics`` keys a score on a
    dataset-derived label (per-class accuracy, a common pattern) puts that text straight into the
    file the orchestrator tails and relays onward. Forge itself sets no ``compute_metrics``, so
    this is a bound on a surface rather than a fix for a live leak — but the bound belongs at the
    choke point, not in each caller.
    """
    out: dict[str, float] = {}
    for key, value in values.items():
        if key in exclude:
            continue
        if len(out) >= MAX_METRIC_KEYS:
            break
        coerced = coerce_float(value)
        if coerced is not None:
            out[str(key)[:MAX_METRIC_KEY_CHARS]] = coerced
    return out


def numeric_metrics(train_output: Any) -> dict[str, float]:
    """The numeric entries of a TRL ``TrainOutput``'s metrics, as floats.

    The ``train()``-return counterpart of :func:`trainer_callback`'s per-event extraction: the
    same trainer numbers, reached once at the end instead of as they happen.
    """
    raw: Any = getattr(train_output, "metrics", None)
    if not isinstance(raw, dict):
        return {}
    return _numeric_extras(cast("dict[str, Any]", raw), exclude=frozenset())


class _Pace:
    """Derives training throughput from the wall clock between logged steps.

    ``transformers`` reports ``train_samples_per_second`` once, in the ``TrainOutput`` returned
    after training ends — useful for a report, useless for watching a run. This measures the same
    thing continuously from what every log already carries: the step number, the time it arrived,
    and (when the trainer was asked to count them) the tokens seen so far.

    Cumulative from the first log rather than per-interval, for the same reason the inference
    runner is: a single slow step between two logs would otherwise make the gauge lurch. The first
    log establishes the baseline and reports nothing — one timestamp is not a rate.
    """

    def __init__(self) -> None:
        self._t0: float | None = None
        self._step0: int | None = None
        self._tokens0: float | None = None

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._step0 = None
        self._tokens0 = None

    def tick(self, step: int | None, data: dict[str, Any]) -> dict[str, float]:
        """Rates for this log line, or ``{}`` until there is enough history to divide by."""
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        tokens = coerce_float(data.get("num_tokens"))

        if self._step0 is None:
            self._step0, self._tokens0, self._t0 = step, tokens, now
            return {}

        elapsed = now - self._t0
        if elapsed <= 0:
            return {}

        out: dict[str, float] = {}
        # `self._step0` is an int by here: the baseline branch above returns whenever it is None.
        if step is not None:
            out["steps_per_s"] = (step - self._step0) / elapsed
        # Only when the trainer was configured to count tokens
        # (`TrainingArguments.include_num_input_tokens_seen`); absent, tokens/s is not knowable
        # here and reporting a guess would be worse than reporting nothing.
        if (
            tokens is not None
            and self._tokens0 is not None
            and "train_tokens_per_second" not in data
        ):
            out["tokens_per_s"] = (tokens - self._tokens0) / elapsed
        return out


class JsonlProgressWriter:
    """Append :class:`ProgressEvent`s to a JSONL file, one per line.

    The file is opened in append mode (parent directories are created) and each
    :meth:`emit` writes a single line and flushes, so a tailing reader sees
    events as they happen. Usable as a context manager.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event: ProgressEvent) -> None:
        """Write ``event`` as one JSON line and flush."""
        self._fh.write(event.model_dump_json() + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlProgressWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def trainer_callback(writer: JsonlProgressWriter, *, gpu: GpuSampler | None = None) -> Any:
    """Build a ``transformers.TrainerCallback`` that forwards events to ``writer``.

    ``transformers`` is imported lazily here, so importing
    :mod:`strata_forge.training.progress` never pulls the ``[finetuning]`` extra. The
    returned object is ready to pass to ``trainer.add_callback(...)``.

    ``gpu`` is injectable so a caller can share one sampler with its phase sink — and so a test
    can supply a stub instead of probing whatever hardware the test machine happens to have.
    """
    try:
        transformers_mod: Any = __import__("transformers")
    except ImportError as exc:  # pragma: no cover - exercised via a fake module in tests
        msg = (
            "trainer_callback requires transformers (the [finetuning] extra). "
            "Install it with: pip install 'strata-forge[finetuning]'."
        )
        raise ImportError(msg) from exc

    base: Any = transformers_mod.TrainerCallback
    sampler = gpu if gpu is not None else GpuSampler()
    # Throughput is derived here rather than read off the trainer: `transformers` reports
    # `train_samples_per_second` only in the final TrainOutput, which is exactly too late to
    # watch. Tracking the wall clock between logs gives the same number while it still matters.
    pace = _Pace()

    class _JsonlCallback(base):  # pyright: ignore[reportUntypedBaseClass]
        """Maps a subset of ``TrainerCallback`` hooks onto JSONL events."""

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            pace.start()
            writer.emit(
                ProgressEvent(
                    kind="start",
                    stage="run",
                    step=coerce_int(getattr(state, "global_step", None)),
                    total_steps=coerce_int(getattr(state, "max_steps", None)),
                    epoch=coerce_float(getattr(state, "epoch", None)),
                    metrics=sampler.sample(),
                )
            )

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            del args, control, kwargs
            data = dict(logs or {})
            is_eval = any(key.startswith("eval_") for key in data)
            step = coerce_int(getattr(state, "global_step", None))
            writer.emit(
                ProgressEvent(
                    kind="eval" if is_eval else "step",
                    stage="run",
                    step=step,
                    total_steps=coerce_int(getattr(state, "max_steps", None)),
                    epoch=coerce_float(data.get("epoch", getattr(state, "epoch", None))),
                    loss=coerce_float(data.get("eval_loss") if is_eval else data.get("loss")),
                    learning_rate=coerce_float(data.get("learning_rate")),
                    # Trainer numbers first, then derived pace, then hardware. Later keys win, but
                    # `pace` yields nothing for a key the trainer already reported, so a real
                    # `train_tokens_per_second` from the trainer is never overwritten by an estimate.
                    metrics={**_numeric_extras(data), **pace.tick(step, data), **sampler.sample()},
                )
            )

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            writer.emit(
                ProgressEvent(
                    kind="checkpoint",
                    stage="run",
                    step=coerce_int(getattr(state, "global_step", None)),
                    epoch=coerce_float(getattr(state, "epoch", None)),
                )
            )

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            writer.emit(
                ProgressEvent(
                    kind="end",
                    # Still `run`: this marks the END of training, not the start of pushing.
                    # The runner owns the `push` stage, because it is the one that uploads.
                    stage="run",
                    step=coerce_int(getattr(state, "global_step", None)),
                    total_steps=coerce_int(getattr(state, "max_steps", None)),
                    epoch=coerce_float(getattr(state, "epoch", None)),
                    metrics=sampler.sample(),
                )
            )

    return _JsonlCallback()


def attach(trainer: Any, path: str | Path | None) -> JsonlProgressWriter | None:
    """Attach a JSONL progress callback to ``trainer`` when a path is configured.

    ``path`` wins; otherwise the ``FORGE_PROGRESS_PATH`` env var is used. Returns
    the :class:`JsonlProgressWriter` (so the caller may close it) or ``None`` when
    no path is set and progress streaming is therefore disabled.
    """
    resolved = path or os.environ.get("FORGE_PROGRESS_PATH")
    if not resolved:
        return None
    writer = JsonlProgressWriter(resolved)
    trainer.add_callback(trainer_callback(writer))
    return writer
