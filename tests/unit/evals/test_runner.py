"""Unit tests for `forge.evals.runner`."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from forge.datasets.schema import Dataset, DatasetItem
from forge.evals.experiment import Experiment, GraderResult, SamplingParams
from forge.evals.runner import run_experiment
from forge.llm.messages import UserMessage
from forge.llm.responses import LLMResponse, Usage
from forge.llm.routing import ModelRoute

if TYPE_CHECKING:
    from forge.llm.messages import AnyMessage
    from forge.llm.registry import ProviderName


def _response(
    text: str,
    *,
    model: str = "claude-opus-4-7",
    provider: ProviderName = "anthropic",
) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(input_tokens=10, output_tokens=5),
        cost_usd=0.001,
        route=ModelRoute(model=model, provider=provider, provider_model_id=model),
        latency_ms=42.0,
    )


def _client(
    *texts: str,
    model: str = "claude-opus-4-7",
    provider: ProviderName = "anthropic",
) -> AsyncMock:
    client = AsyncMock()
    responses = [_response(t, model=model, provider=provider) for t in texts]
    client.complete = AsyncMock(side_effect=responses)
    return client


def _item(query: str, *, expected: Any = None) -> DatasetItem:
    return DatasetItem.from_input({"q": query}, expected_output=expected)


class _PassAllGrader:
    @property
    def name(self) -> str:
        return "pass_all"

    async def grade(self, *, item: DatasetItem, response: LLMResponse) -> GraderResult:
        del item, response
        return GraderResult(grader_name=self.name, score=1.0, passed=True)


class _FailAllGrader:
    @property
    def name(self) -> str:
        return "fail_all"

    async def grade(self, *, item: DatasetItem, response: LLMResponse) -> GraderResult:
        del item, response
        return GraderResult(grader_name=self.name, score=0.0, passed=False)


def _experiment(
    *,
    models: tuple[str, ...] = ("claude-opus-4-7",),
    prompts: tuple[str, ...] = (),
    grader_names: tuple[str, ...] = ("pass_all",),
) -> Experiment:
    return Experiment(
        name="exp",
        models=models,
        prompts=prompts,
        dataset_name="ds",
        grader_names=grader_names,
        sampling=SamplingParams(temperature=0.0),
    )


def _dataset(*queries: str) -> Dataset:
    return Dataset(name="ds", items=tuple(_item(q) for q in queries))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    async def test_one_model_one_item_one_grader(self) -> None:
        outcomes = await run_experiment(
            _experiment(),
            dataset=_dataset("a"),
            clients={"claude-opus-4-7": _client("answer-a")},
            graders=[_PassAllGrader()],
        )
        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.trial.model == "claude-opus-4-7"
        assert outcome.trial.provider == "anthropic"
        assert outcome.trial.response_text == "answer-a"
        assert len(outcome.grader_results) == 1
        assert outcome.grader_results[0].passed is True

    async def test_matrix_size_is_models_times_items(self) -> None:
        experiment = _experiment(models=("m1", "m2"))
        outcomes = await run_experiment(
            experiment,
            dataset=_dataset("a", "b"),
            clients={
                "m1": _client("ans-1a", "ans-1b", model="m1"),
                "m2": _client("ans-2a", "ans-2b", model="m2"),
            },
            graders=[_PassAllGrader()],
        )
        # 2 models x 2 items = 4 trials
        assert len(outcomes) == 4

    async def test_every_grader_runs_on_every_trial(self) -> None:
        outcomes = await run_experiment(
            _experiment(grader_names=("pass_all", "fail_all")),
            dataset=_dataset("a"),
            clients={"claude-opus-4-7": _client("ans")},
            graders=[_PassAllGrader(), _FailAllGrader()],
        )
        outcome = outcomes[0]
        names = {r.grader_name for r in outcome.grader_results}
        assert names == {"pass_all", "fail_all"}

    async def test_grader_order_matches_experiment_order(self) -> None:
        outcomes = await run_experiment(
            _experiment(grader_names=("fail_all", "pass_all")),
            dataset=_dataset("a"),
            clients={"claude-opus-4-7": _client("ans")},
            graders=[_PassAllGrader(), _FailAllGrader()],
        )
        names = [r.grader_name for r in outcomes[0].grader_results]
        assert names == ["fail_all", "pass_all"]


# ---------------------------------------------------------------------------
# Prompt renderers
# ---------------------------------------------------------------------------


class TestPromptRenderers:
    async def test_default_renderer_used_when_prompts_empty(self) -> None:
        client = _client("ans")
        await run_experiment(
            _experiment(prompts=()),
            dataset=_dataset("the-question"),
            clients={"claude-opus-4-7": client},
            graders=[_PassAllGrader()],
        )
        call_args = client.complete.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)
        assert "the-question" in messages[0].content

    async def test_custom_renderer_called_per_item(self) -> None:
        renders: list[DatasetItem] = []

        def _my_renderer(item: DatasetItem) -> list[AnyMessage]:
            renders.append(item)
            return [UserMessage(content=f"custom: {item.input['q']}")]

        client = _client("ans-1", "ans-2")
        await run_experiment(
            _experiment(prompts=("my_prompt",)),
            dataset=_dataset("a", "b"),
            clients={"claude-opus-4-7": client},
            graders=[_PassAllGrader()],
            prompt_renderers={"my_prompt": _my_renderer},
        )
        assert len(renders) == 2
        messages = client.complete.call_args_list[0].kwargs["messages"]
        assert "custom:" in messages[0].content

    async def test_prompt_name_in_trial_when_set(self) -> None:
        def _renderer(item: DatasetItem) -> list[AnyMessage]:
            return [UserMessage(content=item.input["q"])]

        outcomes = await run_experiment(
            _experiment(prompts=("p1",)),
            dataset=_dataset("a"),
            clients={"claude-opus-4-7": _client("ans")},
            graders=[_PassAllGrader()],
            prompt_renderers={"p1": _renderer},
        )
        assert outcomes[0].trial.prompt_name == "p1"

    async def test_prompt_name_none_when_no_prompts(self) -> None:
        outcomes = await run_experiment(
            _experiment(prompts=()),
            dataset=_dataset("a"),
            clients={"claude-opus-4-7": _client("ans")},
            graders=[_PassAllGrader()],
        )
        assert outcomes[0].trial.prompt_name is None


# ---------------------------------------------------------------------------
# Sampling parameters passed through
# ---------------------------------------------------------------------------


class TestSamplingPassthrough:
    async def test_sampling_params_forwarded_to_client(self) -> None:
        experiment = Experiment(
            name="x",
            models=("m",),
            dataset_name="d",
            grader_names=("pass_all",),
            sampling=SamplingParams(temperature=0.7, max_tokens=128, top_p=0.95),
        )
        client = _client("ans", model="m")
        await run_experiment(
            experiment,
            dataset=_dataset("a"),
            clients={"m": client},
            graders=[_PassAllGrader()],
        )
        kwargs = client.complete.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 128
        assert kwargs["top_p"] == 0.95


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    async def test_missing_model_client_rejected(self) -> None:
        with pytest.raises(ValueError, match="without clients"):
            await run_experiment(
                _experiment(models=("missing-model",)),
                dataset=_dataset("a"),
                clients={"different-model": _client("ans")},
                graders=[_PassAllGrader()],
            )

    async def test_missing_prompt_renderer_rejected(self) -> None:
        with pytest.raises(ValueError, match="without renderers"):
            await run_experiment(
                _experiment(prompts=("missing-prompt",)),
                dataset=_dataset("a"),
                clients={"claude-opus-4-7": _client("ans")},
                graders=[_PassAllGrader()],
            )

    async def test_missing_grader_rejected(self) -> None:
        with pytest.raises(ValueError, match="grader_names"):
            await run_experiment(
                _experiment(grader_names=("not-supplied",)),
                dataset=_dataset("a"),
                clients={"claude-opus-4-7": _client("ans")},
                graders=[_PassAllGrader()],
            )

    async def test_duplicate_grader_names_rejected(self) -> None:
        # Two grader instances with the same .name violate uniqueness.
        dup1 = _PassAllGrader()
        dup2 = _PassAllGrader()
        with pytest.raises(ValueError, match="unique"):
            await run_experiment(
                _experiment(),
                dataset=_dataset("a"),
                clients={"claude-opus-4-7": _client("ans")},
                graders=[dup1, dup2],
            )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_llm_error_propagates_by_default(self) -> None:
        client = AsyncMock()
        client.complete = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            await run_experiment(
                _experiment(),
                dataset=_dataset("a"),
                clients={"claude-opus-4-7": client},
                graders=[_PassAllGrader()],
            )

    async def test_continue_on_error_yields_failed_outcome(self) -> None:
        client = AsyncMock()
        client.complete = AsyncMock(side_effect=RuntimeError("boom"))
        outcomes = await run_experiment(
            _experiment(),
            dataset=_dataset("a"),
            clients={"claude-opus-4-7": client},
            graders=[_PassAllGrader()],
            continue_on_error=True,
        )
        assert len(outcomes) == 1
        trial = outcomes[0].trial
        assert trial.response_text == ""
        assert trial.provider == "<failed>"
        assert "boom" in trial.metadata["error"]
        result = outcomes[0].grader_results[0]
        assert result.passed is False
        assert "llm call failed" in result.explanation

    async def test_grader_error_propagates_by_default(self) -> None:
        class _BoomGrader:
            @property
            def name(self) -> str:
                return "boom"

            async def grade(self, *, item: DatasetItem, response: LLMResponse) -> GraderResult:
                del item, response
                raise RuntimeError("grader-boom")

        with pytest.raises(RuntimeError, match="grader-boom"):
            await run_experiment(
                _experiment(grader_names=("boom",)),
                dataset=_dataset("a"),
                clients={"claude-opus-4-7": _client("ans")},
                graders=[_BoomGrader()],
            )

    async def test_continue_on_error_with_failing_grader(self) -> None:
        class _BoomGrader:
            @property
            def name(self) -> str:
                return "boom"

            async def grade(self, *, item: DatasetItem, response: LLMResponse) -> GraderResult:
                del item, response
                raise RuntimeError("grader-boom")

        outcomes = await run_experiment(
            _experiment(grader_names=("boom",)),
            dataset=_dataset("a"),
            clients={"claude-opus-4-7": _client("ans")},
            graders=[_BoomGrader()],
            continue_on_error=True,
        )
        result = outcomes[0].grader_results[0]
        assert result.passed is False
        assert "grader raised" in result.explanation


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    async def test_respects_concurrency_limit(self) -> None:
        in_flight = 0
        peak = 0

        async def _slow(**_kwargs: Any) -> LLMResponse:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.005)
            in_flight -= 1
            return _response("ans")

        client = AsyncMock()
        client.complete = AsyncMock(side_effect=_slow)
        # 10 items, concurrency=3 → peak should be ≤ 3.
        await run_experiment(
            _experiment(),
            dataset=Dataset(name="d", items=tuple(_item(str(i)) for i in range(10))),
            clients={"claude-opus-4-7": client},
            graders=[_PassAllGrader()],
            concurrency=3,
        )
        assert peak <= 3
