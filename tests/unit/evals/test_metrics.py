"""Unit tests for `forge.evals.metrics`."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from forge.evals.experiment import GraderResult, Outcome, Trial
from forge.evals.metrics import (
    bleu,
    mean_score,
    pass_rate,
    pass_rate_by_grader,
    pass_rate_by_model,
    rouge,
)
from forge.llm.responses import Usage


def _trial(
    *, model: str = "m", expected_text: str | None = None, response_text: str = "x"
) -> Trial:
    metadata: dict[str, Any] = {}
    if expected_text is not None:
        metadata["expected_text"] = expected_text
    return Trial(
        experiment_name="e",
        model=model,
        provider="p",
        item_id="i",
        response_text=response_text,
        usage=Usage(input_tokens=1, output_tokens=1),
        cost_usd=0.0,
        cache_hit=False,
        latency_ms=1.0,
        metadata=metadata,
    )


def _outcome(*results: tuple[str, float, bool], **trial_kwargs: Any) -> Outcome:
    return Outcome(
        trial=_trial(**trial_kwargs),
        grader_results=tuple(
            GraderResult(grader_name=name, score=score, passed=passed)
            for name, score, passed in results
        ),
    )


# ---------------------------------------------------------------------------
# pass_rate
# ---------------------------------------------------------------------------


class TestPassRate:
    def test_all_passing(self) -> None:
        outcomes = [
            _outcome(("g", 1.0, True)),
            _outcome(("g", 1.0, True)),
        ]
        assert pass_rate(outcomes, grader_name="g") == 1.0

    def test_all_failing(self) -> None:
        outcomes = [
            _outcome(("g", 0.0, False)),
            _outcome(("g", 0.0, False)),
        ]
        assert pass_rate(outcomes, grader_name="g") == 0.0

    def test_half_passing(self) -> None:
        outcomes = [
            _outcome(("g", 1.0, True)),
            _outcome(("g", 0.0, False)),
        ]
        assert pass_rate(outcomes, grader_name="g") == 0.5

    def test_no_matching_grader_returns_zero(self) -> None:
        outcomes = [_outcome(("other", 1.0, True))]
        assert pass_rate(outcomes, grader_name="missing") == 0.0

    def test_filters_to_named_grader(self) -> None:
        outcomes = [
            _outcome(("g", 0.0, False), ("other", 1.0, True)),
        ]
        assert pass_rate(outcomes, grader_name="g") == 0.0
        assert pass_rate(outcomes, grader_name="other") == 1.0


# ---------------------------------------------------------------------------
# mean_score
# ---------------------------------------------------------------------------


class TestMeanScore:
    def test_average_of_scores(self) -> None:
        outcomes = [
            _outcome(("g", 0.5, False)),
            _outcome(("g", 0.8, True)),
            _outcome(("g", 0.2, False)),
        ]
        result = mean_score(outcomes, grader_name="g")
        assert abs(result - 0.5) < 1e-9

    def test_empty_returns_zero(self) -> None:
        assert mean_score([], grader_name="g") == 0.0

    def test_no_matching_grader_returns_zero(self) -> None:
        outcomes = [_outcome(("other", 1.0, True))]
        assert mean_score(outcomes, grader_name="missing") == 0.0


# ---------------------------------------------------------------------------
# pass_rate_by_model
# ---------------------------------------------------------------------------


class TestPassRateByModel:
    def test_groups_by_model(self) -> None:
        outcomes = [
            _outcome(("g", 1.0, True), model="m1"),
            _outcome(("g", 1.0, True), model="m1"),
            _outcome(("g", 0.0, False), model="m2"),
            _outcome(("g", 1.0, True), model="m2"),
        ]
        result = pass_rate_by_model(outcomes, grader_name="g")
        assert result == {"m1": 1.0, "m2": 0.5}

    def test_empty_returns_empty_dict(self) -> None:
        assert pass_rate_by_model([], grader_name="g") == {}


# ---------------------------------------------------------------------------
# pass_rate_by_grader
# ---------------------------------------------------------------------------


class TestPassRateByGrader:
    def test_groups_by_grader(self) -> None:
        outcomes = [
            _outcome(("g1", 1.0, True), ("g2", 0.0, False)),
            _outcome(("g1", 0.0, False), ("g2", 0.0, False)),
        ]
        result = pass_rate_by_grader(outcomes)
        assert result == {"g1": 0.5, "g2": 0.0}

    def test_empty_returns_empty_dict(self) -> None:
        assert pass_rate_by_grader([]) == {}


# ---------------------------------------------------------------------------
# BLEU + ROUGE (lazy extra)
# ---------------------------------------------------------------------------


class TestBleuExtraRequired:
    def test_bleu_raises_without_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "nltk.translate.bleu_score", None)
        with pytest.raises(ImportError, match=r"\[evals\] extra"):
            bleu([_outcome(("g", 1.0, True), expected_text="hi", response_text="hi")])

    def test_rouge_raises_without_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "rouge_score", None)
        with pytest.raises(ImportError, match=r"\[evals\] extra"):
            rouge([_outcome(("g", 1.0, True), expected_text="hi", response_text="hi")])


class TestBleuWithFake:
    @pytest.fixture(autouse=True)
    def _install_fake_nltk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bleu_mod = types.ModuleType("nltk.translate.bleu_score")

        class _FakeSmoother:
            method1 = "smoother1"

        def _sentence_bleu(refs: Any, hyp: Any, smoothing_function: Any = None) -> float:
            del smoothing_function
            # Toy implementation: 1.0 when first reference exactly equals hyp,
            # 0.0 otherwise. Sufficient for testing the mean aggregation.
            return 1.0 if refs[0] == hyp else 0.0

        bleu_mod.SmoothingFunction = _FakeSmoother  # type: ignore[attr-defined]
        bleu_mod.sentence_bleu = _sentence_bleu  # type: ignore[attr-defined]
        # The import in metrics.py reads `import nltk.translate.bleu_score`.
        # Provide both the leaf and the parent so the chained import resolves.
        nltk_mod = types.ModuleType("nltk")
        translate_mod = types.ModuleType("nltk.translate")
        translate_mod.bleu_score = bleu_mod  # type: ignore[attr-defined]
        nltk_mod.translate = translate_mod  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "nltk", nltk_mod)
        monkeypatch.setitem(sys.modules, "nltk.translate", translate_mod)
        monkeypatch.setitem(sys.modules, "nltk.translate.bleu_score", bleu_mod)

    def test_bleu_returns_mean_across_outcomes(self) -> None:
        outcomes = [
            _outcome(("g", 1.0, True), expected_text="hello world", response_text="hello world"),
            _outcome(("g", 0.0, False), expected_text="goodbye", response_text="something else"),
        ]
        result = bleu(outcomes)
        # One perfect match (1.0), one no-match (0.0) → mean 0.5.
        assert result == 0.5

    def test_bleu_skips_outcomes_without_expected_text(self) -> None:
        outcomes = [
            _outcome(("g", 1.0, True), response_text="x"),  # no expected_text
            _outcome(("g", 1.0, True), expected_text="hi", response_text="hi"),
        ]
        assert bleu(outcomes) == 1.0

    def test_bleu_returns_zero_when_no_comparable_pairs(self) -> None:
        outcomes = [_outcome(("g", 1.0, True), response_text="x")]
        assert bleu(outcomes) == 0.0


class TestRougeWithFake:
    @pytest.fixture(autouse=True)
    def _install_fake_rouge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rouge_mod = types.ModuleType("rouge_score")
        scorer_mod = types.ModuleType("rouge_score.rouge_scorer")

        class _FakeScore:
            def __init__(self, fmeasure: float) -> None:
                self.fmeasure = fmeasure

        class _FakeScorer:
            def __init__(self, variants: list[str], use_stemmer: bool = True) -> None:
                self.variants = variants

            def score(self, reference: str, hypothesis: str) -> dict[str, _FakeScore]:
                f = 1.0 if reference == hypothesis else 0.0
                return {v: _FakeScore(f) for v in self.variants}

        scorer_mod.RougeScorer = _FakeScorer  # type: ignore[attr-defined]
        rouge_mod.rouge_scorer = scorer_mod  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "rouge_score", rouge_mod)
        monkeypatch.setitem(sys.modules, "rouge_score.rouge_scorer", scorer_mod)

    def test_rouge_returns_mean_fmeasure(self) -> None:
        outcomes = [
            _outcome(("g", 1.0, True), expected_text="hi", response_text="hi"),
            _outcome(("g", 1.0, True), expected_text="hi", response_text="bye"),
        ]
        # 1.0 + 0.0 mean = 0.5
        assert rouge(outcomes) == 0.5

    def test_rouge_variant_passed_through(self) -> None:
        outcomes = [
            _outcome(("g", 1.0, True), expected_text="hi", response_text="hi"),
        ]
        # Variant string is forwarded; fake scorer treats them uniformly.
        assert rouge(outcomes, variant="rouge1") == 1.0
        assert rouge(outcomes, variant="rougeL") == 1.0

    def test_rouge_returns_zero_when_no_comparable_pairs(self) -> None:
        # No outcome has an expected_text, so nothing is scored.
        outcomes = [_outcome(("g", 1.0, True), response_text="x")]
        assert rouge(outcomes) == 0.0
