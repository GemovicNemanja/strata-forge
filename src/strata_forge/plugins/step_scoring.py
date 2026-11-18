"""The step-scoring contract: how a process reward model (PRM) plugin is fed and what it returns.

A ``step_scorer`` entry point scores reasoning STEPS, one number per step. Forge owns everything
around the scoring so every PRM plugin is driven the same way and cannot reach the run's
credentials or files:

- Forge reads the ``steps`` input (a dataset split pinned at a commit) and checks it has the
  :data:`~strata_forge.plugins.manifest.STEP_INPUT_COLUMNS`: ``item_id`` (the sample a step
  belongs to, a string), ``step_index`` (its position in that sample, from 0), ``context`` (what
  the step is judged against: the problem and the steps accepted before it, already assembled)
  and ``step`` (the step's text). Segmenting a sample into steps and assembling contexts are
  upstream of the scorer, by design: the scorer judges a given step in a given context.
- The plugin's entry point is a factory: called once with the validated configuration and the
  local paths of its declared model inputs (:class:`StepScorerSetup`), it returns a
  :class:`StepScorer`. Forge then calls :meth:`StepScorer.score` with batches of
  :class:`Step` and expects exactly one finite float back per step, in order.
- Forge writes the scores (``item_id``, ``step_index``, ``score``) with the run's evidence and
  pushes them. Combining step scores into a per-sample score is the caller's choice
  (:func:`aggregate` provides the common reductions); forge does not pick one for the scorer.

What a score MEANS (a probability the step is correct, a logit, a reward) is the plugin's to
declare in its manifest description; forge only guarantees one finite number per step.
"""

from __future__ import annotations

import math
from pathlib import Path  # noqa: TC003 -- a runtime type of a pydantic field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from strata_forge.plugins.manifest import PluginManifestError, ScalarValue

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "SCORE_COLUMNS",
    "Aggregation",
    "Step",
    "StepScorer",
    "StepScorerSetup",
    "aggregate",
    "check_scores",
    "steps_from_rows",
]

#: The columns forge writes for a step-scoring run, in order.
SCORE_COLUMNS: tuple[str, ...] = ("item_id", "step_index", "score")

#: How step scores combine into one score per sample.
type Aggregation = Literal["min", "mean", "last", "product"]


class Step(BaseModel):
    """One step to score, exactly as the ``steps`` input holds it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    step_index: int
    context: str
    step: str


class StepScorerSetup(BaseModel):
    """What a step scorer's factory is given: its configuration and its model inputs' paths."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    config: dict[str, ScalarValue]
    models: dict[str, Path]


@runtime_checkable
class StepScorer(Protocol):
    """A loaded process reward model."""

    def score(self, steps: Sequence[Step]) -> Sequence[float]:
        """One finite score per step, in the order given."""
        ...


def check_scores(steps: Sequence[Step], scores: object) -> list[float]:
    """Refuse a scorer answer that is not one finite number per step."""
    if not isinstance(scores, list | tuple):
        msg = "a step scorer must return a list of numbers"
        raise PluginManifestError(msg)
    values: list[float] = []
    for value in scores:  # pyright: ignore[reportUnknownVariableType] -- checked per element
        if isinstance(value, bool) or not isinstance(value, int | float):
            msg = "a step scorer returned a non-numeric score"
            raise PluginManifestError(msg)
        number = float(value)
        if not math.isfinite(number):
            msg = "a step scorer returned a non-finite score"
            raise PluginManifestError(msg)
        values.append(number)
    if len(values) != len(steps):
        msg = f"a step scorer returned {len(values)} scores for {len(steps)} steps"
        raise PluginManifestError(msg)
    return values


def aggregate(rows: Sequence[tuple[str, int, float]], how: Aggregation) -> dict[str, float]:
    """One score per ``item_id`` from ``(item_id, step_index, score)`` rows.

    ``min`` (the weakest step decides), ``mean``, ``last`` (the highest ``step_index``), or
    ``product``. Steps are ordered by ``step_index`` within an item; a repeated index is refused,
    because which of two scores for one step counts is not something to guess.
    """
    per_item: dict[str, dict[int, float]] = {}
    for item_id, index, value in rows:
        steps = per_item.setdefault(item_id, {})
        if index in steps:
            msg = f"item {item_id!r} has step {index} twice"
            raise PluginManifestError(msg)
        steps[index] = value
    out: dict[str, float] = {}
    for item_id, steps in per_item.items():
        ordered = [steps[i] for i in sorted(steps)]
        match how:
            case "min":
                out[item_id] = min(ordered)
            case "mean":
                out[item_id] = math.fsum(ordered) / len(ordered)
            case "last":
                out[item_id] = ordered[-1]
            case "product":
                out[item_id] = math.prod(ordered)
    return out


def steps_from_rows(rows: Sequence[Mapping[str, object]]) -> list[Step]:
    """Validate raw input rows into :class:`Step` s, naming the first bad row."""
    steps: list[Step] = []
    for n, row in enumerate(rows):
        item_id, index = row.get("item_id"), row.get("step_index")
        context, text = row.get("context"), row.get("step")
        if (
            not isinstance(item_id, str)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(context, str)
            or not isinstance(text, str)
        ):
            msg = (
                f"steps row {n}: item_id and context and step must be strings and step_index a "
                "non-negative integer"
            )
            raise PluginManifestError(msg)
        steps.append(Step(item_id=item_id, step_index=index, context=context, step=text))
    return steps
