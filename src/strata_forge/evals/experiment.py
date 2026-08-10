"""Typed shapes for evaluation experiments and their results.

An :class:`Experiment` is the declarative description of an eval
run — pure data, frozen, with content-hash identity.
:func:`strata_forge.evals.run_experiment` resolves the named references
against live registries at execution time.

A :class:`Trial` is one row in the experiment matrix —
``(model, prompt, item)`` plus the LLM response that came back. A
:class:`GraderResult` is one grader's verdict on a Trial. An
:class:`Outcome` binds the two together.

See [ADR 0010](https://github.com/GemovicNemanja/strata-forge/blob/main/docs/architecture/adr/0010-evals-experiment-as-data-pluggable-graders.md).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.llm.responses import Usage  # noqa: TC001 — Pydantic needs runtime resolution

__all__ = [
    "Experiment",
    "GraderResult",
    "Outcome",
    "SamplingParams",
    "Trial",
]


class SamplingParams(BaseModel):
    """LLM sampling parameters held on an :class:`Experiment`.

    Frozen so two callers describing "the same experiment" produce
    identical content hashes. Defaults match :mod:`strata_forge.llm`'s own
    defaults (None = let the model use its server-side default).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0, le=1)


class Experiment(BaseModel):
    """Declarative description of an evaluation run.

    Carries names rather than concrete clients/datasets/graders so
    the same :class:`Experiment` is portable across processes and
    serializable to disk. The runner resolves names against
    :mod:`strata_forge.llm`, :mod:`strata_forge.prompts`, and :mod:`strata_forge.datasets`.

    Attributes:
        name: Human-readable identifier for this experiment.
        models: Logical model names (e.g. ``"claude-opus-4-7"``).
        prompts: Prompt-registry names. Empty tuple means the runner
            will use the dataset items' ``input`` dicts as raw
            messages without rendering through a template.
        dataset_name: Name of the dataset the experiment runs over.
        dataset_version: Specific version, or ``None`` for latest.
        grader_names: Names of graders to apply (resolved by the
            runner via the grader registry, or supplied as
            instances).
        sampling: Sampling parameters applied to every LLM call.
        description: Free-form prose.
        metadata: Arbitrary JSON-serializable annotations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    models: tuple[str, ...] = Field(min_length=1)
    prompts: tuple[str, ...] = ()
    dataset_name: str = Field(min_length=1)
    dataset_version: str | None = None
    grader_names: tuple[str, ...] = Field(min_length=1)
    sampling: SamplingParams = Field(default_factory=SamplingParams)
    description: str = ""
    metadata: dict[str, Any] = Field(default={})


class Trial(BaseModel):
    """One row of the experiment matrix plus the LLM response.

    Attributes:
        experiment_name: Name of the parent :class:`Experiment`.
        model: Logical model name that produced this trial's response.
        provider: Concrete provider route used (e.g. ``"anthropic"``).
        prompt_name: Prompt template used; ``None`` when no template
            was applied (raw item.input was sent verbatim).
        item_id: The :class:`DatasetItem.id` this trial graded.
        response_text: The text the model returned.
        usage: Token accounting for the call.
        cost_usd: Computed cost.
        cache_hit: ``True`` when the response was served from cache.
        latency_ms: Wall-clock latency of the call.
        metadata: Arbitrary JSON-serializable annotations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    prompt_name: str | None = None
    item_id: str = Field(min_length=1)
    response_text: str
    usage: Usage
    cost_usd: float = Field(ge=0)
    cache_hit: bool = False
    latency_ms: float = Field(ge=0)
    metadata: dict[str, Any] = Field(default={})


class GraderResult(BaseModel):
    """One grader's verdict on one Trial.

    ``score`` is a float; graders that emit binary pass/fail use
    0.0 / 1.0 by convention. ``passed`` is the boolean version of
    the verdict — graders are free to derive it from the score with
    their own threshold (e.g. an LLM judge with ``score >= 0.7``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    grader_name: str = Field(min_length=1)
    score: float
    passed: bool
    explanation: str = ""
    metadata: dict[str, Any] = Field(default={})


class Outcome(BaseModel):
    """A :class:`Trial` with every grader result that applied to it.

    Reports iterate over ``Outcome`` instances. Aggregation (accuracy,
    F1, …) computes over the flat sequence of ``grader_results``
    pulled from every outcome.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial: Trial
    grader_results: tuple[GraderResult, ...] = ()

    def by_grader(self, grader_name: str) -> GraderResult | None:
        """Return the result from ``grader_name`` if present, else ``None``."""
        for result in self.grader_results:
            if result.grader_name == grader_name:
                return result
        return None
