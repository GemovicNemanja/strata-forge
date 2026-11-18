"""The step-scoring contract: how a process reward model (PRM) plugin is fed and what it returns.

A ``step_scorer`` entry point scores reasoning STEPS, one result per step. The platform scores
and records; what the scores are then used for stays in the caller's own code. Forge owns
everything around the scoring so every PRM plugin is driven the same way and cannot reach the
run's credentials or files:

- Forge reads the ``steps`` input (a dataset split pinned at a commit) and checks it has the
  :data:`~strata_forge.plugins.manifest.STEP_INPUT_COLUMNS`: ``item_id`` (the sample a step
  belongs to, a string), ``step_index`` (its position in that sample, from 0), ``context`` (what
  the step is judged against) and ``step`` (the step's text), plus ``step_type`` when the input
  has it. Steps are segmented UPSTREAM and forge never re-splits or re-joins them: ``context`` is
  either a string or a list of ``{pid, text}`` segments, and a list reaches the scorer as that
  list, in order, never joined into one string.
- The plugin's entry point is a factory: called once with the validated configuration and the
  local paths of its declared model inputs (:class:`StepScorerSetup`), it returns a
  :class:`StepScorer`. Forge then calls :meth:`StepScorer.score` with batches of :class:`Step`
  and expects exactly one result per step, in order: a finite number, or :class:`Unscored` with
  one of the reasons the manifest declares.
- The manifest's ``score_kind`` says what the number is. A ``probability`` outside [0, 1] fails
  the run (both ends are allowed, and nothing is clipped); a ``logit`` or ``unconstrained`` score
  is recorded exactly as returned. NaN and infinity fail the run whatever the kind.
- Forge writes one row per input step (``item_id``, ``step_index``, ``step_type`` when present,
  the declared pass-through columns copied unchanged, ``score`` and ``reason``) and checks it
  against the input before anything is pushed: a missing or extra row is an infrastructure fault
  that fails the run.

Per-sample aggregation is not part of the result. :func:`aggregate` offers ``min``, ``mean``,
``last`` and ``product`` over the scored steps, and a run writes them only when asked, to a
separate file labelled diagnostic, never into the scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 -- a runtime type of a pydantic field
from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict

from strata_forge.plugins.manifest import (
    STEP_TYPE_COLUMN,
    PluginManifestError,
    ScalarValue,
    ScoreKind,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

__all__ = [
    "AGGREGATIONS",
    "REASON_COLUMN",
    "SCORE_COLUMN",
    "Aggregation",
    "ContextSegment",
    "Step",
    "StepResult",
    "StepScorer",
    "StepScorerSetup",
    "Unscored",
    "aggregate",
    "check_result",
    "check_scores",
    "steps_from_rows",
]

#: The columns forge adds to every scored step.
SCORE_COLUMN = "score"
REASON_COLUMN = "reason"

#: How step scores may be combined into one number per sample, for a diagnostic read only.
type Aggregation = Literal["min", "mean", "last", "product"]
AGGREGATIONS: tuple[Aggregation, ...] = ("min", "mean", "last", "product")


class ContextSegment(BaseModel):
    """One segment of a structured context, as segmented upstream: an id and its text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pid: str
    text: str


class Step(BaseModel):
    """One step to score, exactly as the ``steps`` input holds it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    step_index: int
    #: A string, or the ``{pid, text}`` segments in their input order, never joined.
    context: str | tuple[ContextSegment, ...]
    step: str
    step_type: str | None = None


@dataclass(frozen=True, slots=True)
class Unscored:
    """A step the scorer did not score, and why: one of its manifest's ``unscored_reasons``."""

    reason: str


#: What a scorer returns for one step: a number, or a declared reason for none.
type StepResult = float | Unscored


class StepScorerSetup(BaseModel):
    """What a step scorer's factory is given: its configuration and its model inputs' paths."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    config: dict[str, ScalarValue]
    models: dict[str, Path]


@runtime_checkable
class StepScorer(Protocol):
    """A loaded process reward model."""

    def score(self, steps: Sequence[Step]) -> Sequence[StepResult]:
        """One result per step, in the order given."""
        ...


def check_result(
    value: object, score_kind: ScoreKind, reasons: Collection[str]
) -> tuple[float | None, str | None]:
    """``(score, reason)`` for one scorer result, refusing anything outside the contract.

    A number must be finite, and a ``probability`` must lie in [0, 1]; it is refused, never
    clipped. An :class:`Unscored` must carry a declared reason.
    """
    if isinstance(value, Unscored):
        if value.reason not in reasons:
            declared = sorted(reasons) or "none"
            msg = (
                f"a step scorer left a step unscored for an undeclared reason {value.reason!r} "
                f"(declared: {declared})"
            )
            raise PluginManifestError(msg)
        return None, value.reason
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = "a step scorer returned something that is neither a number nor Unscored"
        raise PluginManifestError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = "a step scorer returned a non-finite score"
        raise PluginManifestError(msg)
    if score_kind == "probability" and not 0.0 <= number <= 1.0:
        msg = f"a step scorer declared probability scores and returned {number!r}"
        raise PluginManifestError(msg)
    return number, None


def check_scores(
    steps: Sequence[Step], results: object, score_kind: ScoreKind, reasons: Collection[str]
) -> list[tuple[float | None, str | None]]:
    """Refuse a scorer answer that is not one acceptable result per step."""
    if not isinstance(results, list | tuple):
        msg = "a step scorer must return a list with one result per step"
        raise PluginManifestError(msg)
    items = cast("Sequence[object]", results)
    if len(items) != len(steps):
        msg = f"a step scorer returned {len(items)} results for {len(steps)} steps"
        raise PluginManifestError(msg)
    return [check_result(value, score_kind, reasons) for value in items]


def aggregate(
    rows: Sequence[tuple[str, int, float | None]], how: Aggregation
) -> dict[str, tuple[float | None, int, int]]:
    """Per ``item_id``: ``(value, scored_steps, unscored_steps)`` over its scored steps.

    ``min`` (the weakest step), ``mean``, ``last`` (the scored step with the highest
    ``step_index``) or ``product``. Unscored steps (``None``) are counted, never imputed; an item
    with none scored has value ``None``. A repeated ``(item_id, step_index)`` is refused, because
    which of two scores for one step counts is not something to guess.
    A diagnostic read: what a sample's score should be is the caller's decision, not forge's.
    """
    per_item: dict[str, dict[int, float | None]] = {}
    for item_id, index, value in rows:
        steps = per_item.setdefault(item_id, {})
        if index in steps:
            msg = f"item {item_id!r} has step {index} twice"
            raise PluginManifestError(msg)
        steps[index] = value
    out: dict[str, tuple[float | None, int, int]] = {}
    for item_id, steps in per_item.items():
        scored = [v for _, v in sorted(steps.items()) if v is not None]
        unscored = len(steps) - len(scored)
        value: float | None
        if not scored:
            value = None
        else:
            match how:
                case "min":
                    value = min(scored)
                case "mean":
                    value = math.fsum(scored) / len(scored)
                case "last":
                    value = scored[-1]
                case "product":
                    value = math.prod(scored)
        out[item_id] = (value, len(scored), unscored)
    return out


def _context(value: object) -> str | tuple[ContextSegment, ...] | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, list | tuple):
        return None
    segments: list[ContextSegment] = []
    for part in cast("Sequence[object]", value):
        if not isinstance(part, dict):
            return None
        fields = cast("dict[str, object]", part)
        pid, text = fields.get("pid"), fields.get("text")
        if set(fields) != {"pid", "text"} or not isinstance(pid, str) or not isinstance(text, str):
            return None
        segments.append(ContextSegment(pid=pid, text=text))
    return tuple(segments)


def steps_from_rows(rows: Sequence[Mapping[str, object]]) -> list[Step]:
    """Validate raw input rows into :class:`Step` s, naming the first bad row."""
    steps: list[Step] = []
    for n, row in enumerate(rows):
        item_id, index = row.get("item_id"), row.get("step_index")
        context, text = _context(row.get("context")), row.get("step")
        step_type = row.get(STEP_TYPE_COLUMN)
        if (
            not isinstance(item_id, str)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or context is None
            or not isinstance(text, str)
            or (step_type is not None and not isinstance(step_type, str))
        ):
            msg = (
                f"steps row {n}: item_id and step must be strings, step_index a non-negative "
                "integer, context a string or a list of {pid, text} strings, and step_type a "
                "string when present"
            )
            raise PluginManifestError(msg)
        steps.append(
            Step(item_id=item_id, step_index=index, context=context, step=text, step_type=step_type)
        )
    return steps
