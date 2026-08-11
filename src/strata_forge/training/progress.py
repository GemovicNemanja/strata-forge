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

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "JsonlProgressWriter",
    "ProgressEvent",
    "ProgressKind",
    "attach",
    "trainer_callback",
]


# ``phase`` is listed first because it is the only kind that can precede ``start``: it reports
# what a long provisioning step is doing (downloading data, launching a model server, uploading
# results) so the stretches between countable milestones are not a silent gap to whoever tails
# the JSONL. A ``phase`` event carries ``message`` only — never a step count.
type ProgressKind = Literal["phase", "start", "step", "eval", "checkpoint", "end", "error"]

# Keys promoted to dedicated :class:`ProgressEvent` fields, so they aren't
# duplicated inside ``metrics``.
_PROMOTED = frozenset({"loss", "eval_loss", "learning_rate", "epoch"})


class ProgressEvent(BaseModel):
    """One structured training-progress record.

    A ``phase`` event is the exception to the shape below: it carries only ``kind``,
    ``message`` and ``ts``, because it marks work that has no step to count.

    Attributes:
        kind: The lifecycle milestone this event marks, or ``phase`` for a free-form
            report of what a long uncountable step is currently doing.
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
    step: int | None = None
    total_steps: int | None = None
    epoch: float | None = None
    loss: float | None = None
    learning_rate: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    message: str = ""
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _coerce_float(value: Any) -> float | None:
    """Best-effort cast to ``float``; ``None`` for bools / non-numerics."""
    if value is None or isinstance(value, bool):  # bool is an int subclass — reject it
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _coerce_int(value: Any) -> int | None:
    f = _coerce_float(value)
    return int(f) if f is not None else None


def _numeric_extras(values: dict[str, Any]) -> dict[str, float]:
    """The numeric entries of ``values`` (as floats), minus the promoted keys."""
    out: dict[str, float] = {}
    for key, value in values.items():
        if key in _PROMOTED:
            continue
        coerced = _coerce_float(value)
        if coerced is not None:
            out[str(key)] = coerced
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


def trainer_callback(writer: JsonlProgressWriter) -> Any:
    """Build a ``transformers.TrainerCallback`` that forwards events to ``writer``.

    ``transformers`` is imported lazily here, so importing
    :mod:`strata_forge.training.progress` never pulls the ``[finetuning]`` extra. The
    returned object is ready to pass to ``trainer.add_callback(...)``.
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

    class _JsonlCallback(base):  # pyright: ignore[reportUntypedBaseClass]
        """Maps a subset of ``TrainerCallback`` hooks onto JSONL events."""

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            writer.emit(
                ProgressEvent(
                    kind="start",
                    step=_coerce_int(getattr(state, "global_step", None)),
                    total_steps=_coerce_int(getattr(state, "max_steps", None)),
                    epoch=_coerce_float(getattr(state, "epoch", None)),
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
            writer.emit(
                ProgressEvent(
                    kind="eval" if is_eval else "step",
                    step=_coerce_int(getattr(state, "global_step", None)),
                    total_steps=_coerce_int(getattr(state, "max_steps", None)),
                    epoch=_coerce_float(data.get("epoch", getattr(state, "epoch", None))),
                    loss=_coerce_float(data.get("eval_loss") if is_eval else data.get("loss")),
                    learning_rate=_coerce_float(data.get("learning_rate")),
                    metrics=_numeric_extras(data),
                )
            )

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            writer.emit(
                ProgressEvent(
                    kind="checkpoint",
                    step=_coerce_int(getattr(state, "global_step", None)),
                    epoch=_coerce_float(getattr(state, "epoch", None)),
                )
            )

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            writer.emit(
                ProgressEvent(
                    kind="end",
                    step=_coerce_int(getattr(state, "global_step", None)),
                    total_steps=_coerce_int(getattr(state, "max_steps", None)),
                    epoch=_coerce_float(getattr(state, "epoch", None)),
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
