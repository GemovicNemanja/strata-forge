"""Unit tests for `strata_forge.evals.experiment`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from strata_forge.evals.experiment import (
    Experiment,
    GraderResult,
    Outcome,
    SamplingParams,
    Trial,
)
from strata_forge.llm.responses import Usage


def _usage(input_tokens: int = 10, output_tokens: int = 5) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)


def _trial(**overrides: Any) -> Trial:
    defaults: dict[str, Any] = {
        "experiment_name": "exp",
        "model": "claude-opus-4-7",
        "provider": "anthropic",
        "prompt_name": None,
        "item_id": "item-a",
        "response_text": "Paris",
        "usage": _usage(),
        "cost_usd": 0.001,
        "cache_hit": False,
        "latency_ms": 150.0,
    }
    defaults.update(overrides)
    return Trial(**defaults)


# ---------------------------------------------------------------------------
# SamplingParams
# ---------------------------------------------------------------------------


class TestSamplingParams:
    def test_defaults_to_all_none(self) -> None:
        sp = SamplingParams()
        assert sp.temperature is None
        assert sp.max_tokens is None
        assert sp.top_p is None

    def test_is_frozen(self) -> None:
        sp = SamplingParams(temperature=0.5)
        with pytest.raises(PydanticValidationError, match="frozen"):
            sp.temperature = 0.9  # type: ignore[misc]

    def test_validates_ranges(self) -> None:
        with pytest.raises(PydanticValidationError):
            SamplingParams(temperature=-0.1)
        with pytest.raises(PydanticValidationError):
            SamplingParams(max_tokens=0)
        with pytest.raises(PydanticValidationError):
            SamplingParams(top_p=1.5)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            SamplingParams(unknown=1.0)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


class TestExperiment:
    def test_basic_construction(self) -> None:
        exp = Experiment(
            name="trivia",
            models=("claude-opus-4-7", "gpt-5.5"),
            prompts=("trivia-prompt",),
            dataset_name="trivia",
            grader_names=("exact_match",),
        )
        assert exp.name == "trivia"
        assert exp.models == ("claude-opus-4-7", "gpt-5.5")
        assert exp.dataset_version is None
        assert exp.sampling == SamplingParams()

    def test_models_must_be_non_empty(self) -> None:
        with pytest.raises(PydanticValidationError):
            Experiment(
                name="x",
                models=(),
                dataset_name="d",
                grader_names=("g",),
            )

    def test_grader_names_must_be_non_empty(self) -> None:
        with pytest.raises(PydanticValidationError):
            Experiment(
                name="x",
                models=("m",),
                dataset_name="d",
                grader_names=(),
            )

    def test_prompts_can_be_empty(self) -> None:
        # Empty prompts means "use raw item.input as messages".
        exp = Experiment(
            name="x",
            models=("m",),
            dataset_name="d",
            grader_names=("g",),
        )
        assert exp.prompts == ()

    def test_is_frozen(self) -> None:
        exp = Experiment(name="x", models=("m",), dataset_name="d", grader_names=("g",))
        with pytest.raises(PydanticValidationError, match="frozen"):
            exp.name = "other"  # type: ignore[misc]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            Experiment(
                name="x",
                models=("m",),
                dataset_name="d",
                grader_names=("g",),
                unknown="oops",  # type: ignore[call-arg]
            )

    def test_name_must_be_non_empty(self) -> None:
        with pytest.raises(PydanticValidationError):
            Experiment(name="", models=("m",), dataset_name="d", grader_names=("g",))

    def test_dataset_name_must_be_non_empty(self) -> None:
        with pytest.raises(PydanticValidationError):
            Experiment(name="x", models=("m",), dataset_name="", grader_names=("g",))

    def test_serialize_round_trip(self) -> None:
        # Experiments are declarative — they must round-trip through JSON
        # so they're portable across processes.
        original = Experiment(
            name="trivia",
            models=("claude-opus-4-7",),
            prompts=("p1", "p2"),
            dataset_name="trivia",
            dataset_version="abc123",
            grader_names=("exact_match", "regex_check"),
            sampling=SamplingParams(temperature=0.7, max_tokens=512),
            metadata={"owner": "qa-team"},
        )
        payload = original.model_dump_json()
        rebuilt = Experiment.model_validate_json(payload)
        assert rebuilt == original


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------


class TestTrial:
    def test_basic_construction(self) -> None:
        trial = _trial()
        assert trial.experiment_name == "exp"
        assert trial.model == "claude-opus-4-7"
        assert trial.provider == "anthropic"
        assert trial.item_id == "item-a"
        assert trial.response_text == "Paris"

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            _trial(cost_usd=-0.01)

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            _trial(latency_ms=-1.0)

    def test_is_frozen(self) -> None:
        trial = _trial()
        with pytest.raises(PydanticValidationError, match="frozen"):
            trial.cost_usd = 999.0  # type: ignore[misc]

    def test_prompt_name_optional(self) -> None:
        trial = _trial(prompt_name=None)
        assert trial.prompt_name is None
        trial2 = _trial(prompt_name="my-prompt")
        assert trial2.prompt_name == "my-prompt"

    def test_required_string_fields_non_empty(self) -> None:
        with pytest.raises(PydanticValidationError):
            _trial(model="")
        with pytest.raises(PydanticValidationError):
            _trial(provider="")
        with pytest.raises(PydanticValidationError):
            _trial(item_id="")
        with pytest.raises(PydanticValidationError):
            _trial(experiment_name="")


# ---------------------------------------------------------------------------
# GraderResult
# ---------------------------------------------------------------------------


class TestGraderResult:
    def test_basic_construction(self) -> None:
        r = GraderResult(grader_name="exact", score=1.0, passed=True)
        assert r.grader_name == "exact"
        assert r.score == 1.0
        assert r.passed is True

    def test_score_can_be_any_float(self) -> None:
        # We don't enforce [0, 1] — LLM judges might use other scales.
        r = GraderResult(grader_name="x", score=7.5, passed=True)
        assert r.score == 7.5
        r2 = GraderResult(grader_name="x", score=-1.0, passed=False)
        assert r2.score == -1.0

    def test_grader_name_must_be_non_empty(self) -> None:
        with pytest.raises(PydanticValidationError):
            GraderResult(grader_name="", score=1.0, passed=True)

    def test_is_frozen(self) -> None:
        r = GraderResult(grader_name="x", score=1.0, passed=True)
        with pytest.raises(PydanticValidationError, match="frozen"):
            r.score = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------


class TestOutcome:
    def test_basic_construction(self) -> None:
        trial = _trial()
        results = (
            GraderResult(grader_name="exact", score=1.0, passed=True),
            GraderResult(grader_name="regex", score=0.0, passed=False),
        )
        outcome = Outcome(trial=trial, grader_results=results)
        assert outcome.trial is trial
        assert len(outcome.grader_results) == 2

    def test_by_grader_finds_match(self) -> None:
        trial = _trial()
        results = (
            GraderResult(grader_name="exact", score=1.0, passed=True),
            GraderResult(grader_name="regex", score=0.5, passed=False),
        )
        outcome = Outcome(trial=trial, grader_results=results)
        recovered = outcome.by_grader("regex")
        assert recovered is not None
        assert recovered.grader_name == "regex"
        assert recovered.score == 0.5

    def test_by_grader_returns_none_for_missing(self) -> None:
        outcome = Outcome(
            trial=_trial(),
            grader_results=(GraderResult(grader_name="exact", score=1.0, passed=True),),
        )
        assert outcome.by_grader("nonexistent") is None

    def test_grader_results_are_tuple(self) -> None:
        outcome = Outcome(trial=_trial(), grader_results=())
        assert isinstance(outcome.grader_results, tuple)

    def test_is_frozen(self) -> None:
        outcome = Outcome(trial=_trial(), grader_results=())
        with pytest.raises(PydanticValidationError, match="frozen"):
            outcome.trial = _trial(model="other")  # type: ignore[misc]
