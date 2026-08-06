"""Unit tests for `strata_forge.evals.graders.semantic`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from strata_forge.datasets.schema import DatasetItem
from strata_forge.evals.graders.semantic import SemanticSimilarity, cosine_similarity
from strata_forge.llm.responses import LLMResponse, Usage
from strata_forge.llm.routing import ModelRoute

if TYPE_CHECKING:
    from collections.abc import Sequence


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1),
        cost_usd=0.0,
        route=ModelRoute(
            model="claude-opus-4-7",
            provider="anthropic",
            provider_model_id="claude-opus-4-7",
        ),
    )


def _item(*, expected: Any = None) -> DatasetItem:
    return DatasetItem(id="a", input={"q": "x"}, expected_output=expected)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors_return_zero(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == 0.0

    def test_antiparallel_vectors_return_minus_one(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - -1.0) < 1e-9

    def test_zero_vector_returns_zero(self) -> None:
        # Undefined direction; defensive 0.0 fallback.
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_known_value(self) -> None:
        # cos angle between (3,4) and (4,3) = (12+12)/(5*5) = 24/25 = 0.96
        result = cosine_similarity([3.0, 4.0], [4.0, 3.0])
        assert abs(result - 24 / 25) < 1e-9


# ---------------------------------------------------------------------------
# SemanticSimilarity grader
# ---------------------------------------------------------------------------


def _embed_factory(
    mapping: dict[str, Sequence[float]],
) -> Any:
    """Build an async embed function that returns the mapping for known texts."""

    async def _embed(text: str) -> Sequence[float]:
        if text not in mapping:
            msg = f"unexpected embed input: {text!r}"
            raise AssertionError(msg)
        return mapping[text]

    return _embed


class TestSemanticSimilarity:
    async def test_identical_text_passes(self) -> None:
        embed = _embed_factory({"hello": [1.0, 0.0], "hello world": [0.5, 0.5]})
        grader = SemanticSimilarity(embed, pass_threshold=0.8)
        result = await grader.grade(item=_item(expected="hello"), response=_response("hello"))
        # Same text → embed gets called with the same string → identical vector.
        assert result.passed is True
        assert result.score == 1.0
        assert result.metadata["cosine_similarity"] == 1.0

    async def test_dissimilar_text_fails(self) -> None:
        embed = _embed_factory(
            {"yes": [1.0, 0.0], "no": [0.0, 1.0]}  # orthogonal
        )
        grader = SemanticSimilarity(embed, pass_threshold=0.5)
        result = await grader.grade(item=_item(expected="yes"), response=_response("no"))
        assert result.passed is False
        assert result.score == 0.0
        assert "0.0000" in result.explanation

    async def test_partial_similarity_passes_above_threshold(self) -> None:
        # cosine([3,4], [4,3]) = 0.96 → above default 0.8 → pass
        embed = _embed_factory({"reference": [3.0, 4.0], "candidate": [4.0, 3.0]})
        grader = SemanticSimilarity(embed)  # default threshold 0.8
        result = await grader.grade(
            item=_item(expected="reference"), response=_response("candidate")
        )
        assert result.passed is True
        assert abs(result.score - 0.96) < 1e-9

    async def test_missing_expected_fails(self) -> None:
        # When item has no expected_output, the grader can't compare.
        async def _embed(text: str) -> Sequence[float]:  # pragma: no cover
            del text
            return [1.0]

        grader = SemanticSimilarity(_embed)
        result = await grader.grade(item=_item(expected=None), response=_response("y"))
        assert result.passed is False
        assert "no expected_output" in result.explanation

    async def test_anti_aligned_clamps_to_zero(self) -> None:
        # cosine [-1] but score floors at 0 to keep metrics non-negative.
        embed = _embed_factory({"a": [1.0, 0.0], "b": [-1.0, 0.0]})
        # Allow score floor to clip below the threshold to verify behaviour.
        grader = SemanticSimilarity(embed, pass_threshold=-0.5)
        result = await grader.grade(item=_item(expected="a"), response=_response("b"))
        # Raw cosine is -1.0; passed compares against threshold using raw,
        # but score is clamped to 0 in the output.
        assert result.score == 0.0
        # -1.0 is NOT >= -0.5, so it should fail
        assert result.passed is False


# ---------------------------------------------------------------------------
# Threshold validation
# ---------------------------------------------------------------------------


class TestThresholdValidation:
    def test_below_minus_one_rejected(self) -> None:
        async def _embed(text: str) -> Sequence[float]:  # pragma: no cover
            del text
            return [0.0]

        with pytest.raises(ValueError, match="pass_threshold"):
            SemanticSimilarity(_embed, pass_threshold=-1.5)

    def test_above_one_rejected(self) -> None:
        async def _embed(text: str) -> Sequence[float]:  # pragma: no cover
            del text
            return [0.0]

        with pytest.raises(ValueError, match="pass_threshold"):
            SemanticSimilarity(_embed, pass_threshold=1.5)

    def test_default_name(self) -> None:
        async def _embed(text: str) -> Sequence[float]:  # pragma: no cover
            del text
            return [0.0]

        grader = SemanticSimilarity(_embed)
        assert grader.name == "semantic_similarity"

    def test_custom_name(self) -> None:
        async def _embed(text: str) -> Sequence[float]:  # pragma: no cover
            del text
            return [0.0]

        grader = SemanticSimilarity(_embed, name="my-sem")
        assert grader.name == "my-sem"

    def test_pass_threshold_property_exposed(self) -> None:
        async def _embed(text: str) -> Sequence[float]:  # pragma: no cover
            del text
            return [0.0]

        grader = SemanticSimilarity(_embed, pass_threshold=0.5)
        assert grader.pass_threshold == 0.5
