"""Async experiment runner — orchestrates the model x prompt x item x grader matrix.

The runner consumes an :class:`Experiment` plus a resolved dataset and
the live objects the experiment names refer to (LLM clients keyed by
model, prompt renderers keyed by prompt name, grader instances).
It generates the cross-product across the first three axes, runs the
LLM calls bounded by ``concurrency`` via an ``asyncio.Semaphore``,
grades each trial against every grader, and returns the collected
:class:`Outcome` tuple.

The runner does not own dataset resolution, prompt rendering, or
metric aggregation. Each of those lives in a separate module so the
runner stays single-purpose.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from strata_forge.evals.experiment import GraderResult, Outcome, Trial
from strata_forge.llm.messages import UserMessage

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from strata_forge.datasets.schema import Dataset, DatasetItem
    from strata_forge.evals.experiment import Experiment
    from strata_forge.evals.graders.base import Grader
    from strata_forge.llm.client import LLMClient
    from strata_forge.llm.messages import AnyMessage

__all__ = [
    "PromptRenderer",
    "run_experiment",
]


type PromptRenderer = Callable[[DatasetItem], list[AnyMessage]]
"""Function that turns one dataset item into a message list for the LLM."""


def _default_renderer(item: DatasetItem) -> list[AnyMessage]:
    """Render an item as a single UserMessage whose content is JSON-encoded item.input."""
    return [UserMessage(content=json.dumps(item.input, sort_keys=True))]


def _validate_runner_inputs(
    experiment: Experiment,
    *,
    clients: Mapping[str, LLMClient],
    prompt_renderers: Mapping[str, PromptRenderer] | None,
    graders: Sequence[Grader],
) -> None:
    missing_clients = [m for m in experiment.models if m not in clients]
    if missing_clients:
        msg = f"Experiment models without clients: {missing_clients!r}"
        raise ValueError(msg)
    if experiment.prompts:
        renderers = prompt_renderers or {}
        missing_prompts = [p for p in experiment.prompts if p not in renderers]
        if missing_prompts:
            msg = f"Experiment prompts without renderers: {missing_prompts!r}"
            raise ValueError(msg)
    grader_names = {g.name for g in graders}
    missing_graders = [g for g in experiment.grader_names if g not in grader_names]
    if missing_graders:
        msg = f"Experiment grader_names not in supplied graders: {missing_graders!r}"
        raise ValueError(msg)
    if len(grader_names) != len(graders):
        msg = "Grader names must be unique across the graders supplied"
        raise ValueError(msg)


async def run_experiment(
    experiment: Experiment,
    *,
    dataset: Dataset,
    clients: Mapping[str, LLMClient],
    graders: Sequence[Grader],
    prompt_renderers: Mapping[str, PromptRenderer] | None = None,
    concurrency: int = 10,
    continue_on_error: bool = False,
) -> tuple[Outcome, ...]:
    """Run an experiment over a dataset; return all outcomes.

    Args:
        experiment: The declarative experiment definition.
        dataset: The :class:`Dataset` whose items the runner iterates
            over. The caller resolves ``experiment.dataset_name`` /
            ``experiment.dataset_version`` against a ``DatasetStore``
            before calling this function.
        clients: Maps logical model name → :class:`LLMClient`. Every
            ``experiment.models`` entry must have a client.
        graders: Concrete grader instances. Their ``.name`` properties
            must cover ``experiment.grader_names`` exactly.
        prompt_renderers: Maps prompt name → ``PromptRenderer``.
            Required when ``experiment.prompts`` is non-empty. When
            ``experiment.prompts`` is empty, ``item.input`` is passed
            verbatim as a JSON-encoded user message.
        concurrency: Maximum in-flight LLM calls.
        continue_on_error: When ``True``, a failing LLM call yields a
            Trial with ``response_text=""`` plus failed grader results
            instead of propagating the exception. Default ``False``:
            propagate the first failure.

    Returns:
        A tuple of :class:`Outcome` instances, one per
        (model, prompt, item) combination, in deterministic order
        (model-major, prompt-major, item-major).

    Raises:
        ValueError: When any name in the experiment isn't covered by
            the supplied clients, prompt renderers, or graders.
    """
    _validate_runner_inputs(
        experiment,
        clients=clients,
        prompt_renderers=prompt_renderers,
        graders=graders,
    )
    grader_by_name = {g.name: g for g in graders}
    selected_graders = tuple(grader_by_name[name] for name in experiment.grader_names)

    prompt_axis: tuple[str | None, ...] = experiment.prompts if experiment.prompts else (None,)
    renderers = prompt_renderers or {}

    semaphore = asyncio.Semaphore(concurrency)

    async def _one_trial(
        model: str,
        prompt_name: str | None,
        item: DatasetItem,
    ) -> Outcome:
        client = clients[model]
        renderer = renderers[prompt_name] if prompt_name is not None else _default_renderer
        messages = renderer(item)
        async with semaphore:
            start = time.monotonic()
            try:
                response = await client.complete(
                    messages=messages,
                    temperature=experiment.sampling.temperature,
                    max_tokens=experiment.sampling.max_tokens,
                    top_p=experiment.sampling.top_p,
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                elapsed_ms = (time.monotonic() - start) * 1000.0
                return _failed_outcome(
                    experiment=experiment,
                    model=model,
                    prompt_name=prompt_name,
                    item=item,
                    error=exc,
                    grader_names=experiment.grader_names,
                    latency_ms=elapsed_ms,
                )

        trial = Trial(
            experiment_name=experiment.name,
            model=response.route.model,
            provider=response.route.provider,
            prompt_name=prompt_name,
            item_id=item.id,
            response_text=response.text,
            usage=response.usage,
            cost_usd=response.cost_usd,
            cache_hit=response.cache_hit,
            latency_ms=response.latency_ms,
        )
        grader_results: list[GraderResult] = []
        for grader in selected_graders:
            try:
                result = await grader.grade(item=item, response=response)
            except Exception as exc:
                if not continue_on_error:
                    raise
                result = GraderResult(
                    grader_name=grader.name,
                    score=0.0,
                    passed=False,
                    explanation=f"grader raised: {exc!r}",
                )
            grader_results.append(result)
        return Outcome(trial=trial, grader_results=tuple(grader_results))

    tasks: list[asyncio.Task[Outcome]] = []
    for model in experiment.models:
        for prompt_name in prompt_axis:
            for item in dataset.items:
                tasks.append(asyncio.create_task(_one_trial(model, prompt_name, item)))
    outcomes = await asyncio.gather(*tasks)
    return tuple(outcomes)


def _failed_outcome(
    *,
    experiment: Experiment,
    model: str,
    prompt_name: str | None,
    item: DatasetItem,
    error: Exception,
    grader_names: tuple[str, ...],
    latency_ms: float,
) -> Outcome:
    """Synthesize an Outcome representing a failed LLM call.

    The trial carries empty text and zero usage; every grader is
    recorded as failed with the exception in the explanation.
    """
    from strata_forge.llm.responses import Usage

    trial = Trial(
        experiment_name=experiment.name,
        model=model,
        provider="<failed>",
        prompt_name=prompt_name,
        item_id=item.id,
        response_text="",
        usage=Usage(input_tokens=0, output_tokens=0),
        cost_usd=0.0,
        cache_hit=False,
        latency_ms=latency_ms,
        metadata={"error": repr(error)},
    )
    failed: list[GraderResult] = [
        GraderResult(
            grader_name=name,
            score=0.0,
            passed=False,
            explanation=f"llm call failed: {error!r}",
        )
        for name in grader_names
    ]
    return Outcome(trial=trial, grader_results=tuple(failed))


# Suppress the unused-Any-import warning when this module is read with the
# typing-only imports stripped (pyright can't tell `Any` is used by typing).
_: Any = None
