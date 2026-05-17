"""Unit tests for `forge.evals.graders.json_grader`."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from forge.datasets.schema import DatasetItem
from forge.evals.graders.json_grader import JSONField, JSONStructure
from forge.llm.responses import LLMResponse, Usage
from forge.llm.routing import ModelRoute


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1),
        cost_usd=0.0,
        route=ModelRoute(
            model="claude-opus-4-7", provider="anthropic", provider_model_id="claude-opus-4-7"
        ),
    )


def _item(*, expected: Any = None) -> DatasetItem:
    return DatasetItem(id="a", input={"q": "x"}, expected_output=expected)


class _Answer(BaseModel):
    answer: str
    confidence: float


# ---------------------------------------------------------------------------
# JSONStructure
# ---------------------------------------------------------------------------


class TestJSONStructure:
    async def test_valid_json_matching_schema_passes(self) -> None:
        grader = JSONStructure(schema=_Answer)
        response = _response(json.dumps({"answer": "Paris", "confidence": 0.95}))
        result = await grader.grade(item=_item(), response=response)
        assert result.passed is True
        assert result.score == 1.0

    async def test_invalid_json_fails(self) -> None:
        grader = JSONStructure(schema=_Answer)
        result = await grader.grade(item=_item(), response=_response("not json at all"))
        assert result.passed is False
        # Pydantic v2 also surfaces JSON parse errors as ValidationError.
        assert "validation failed" in result.explanation or "not JSON" in result.explanation

    async def test_valid_json_wrong_schema_fails(self) -> None:
        grader = JSONStructure(schema=_Answer)
        response = _response(json.dumps({"wrong_field": "Paris"}))
        result = await grader.grade(item=_item(), response=response)
        assert result.passed is False
        assert "validation failed" in result.explanation

    async def test_default_name(self) -> None:
        grader = JSONStructure(schema=_Answer)
        assert grader.name == f"json_structure({_Answer.__name__})"

    async def test_custom_name(self) -> None:
        grader = JSONStructure(schema=_Answer, name="my-shape")
        assert grader.name == "my-shape"

    async def test_extra_fields_handled_by_schema(self) -> None:
        # The grader's pass/fail depends on the schema's own config —
        # we don't override here. _Answer accepts extras by default in
        # Pydantic v2.
        grader = JSONStructure(schema=_Answer)
        response = _response(json.dumps({"answer": "Paris", "confidence": 0.95, "extra": "ok"}))
        result = await grader.grade(item=_item(), response=response)
        assert result.passed is True


# ---------------------------------------------------------------------------
# JSONField
# ---------------------------------------------------------------------------


class TestJSONField:
    async def test_path_equals_item_expected(self) -> None:
        grader = JSONField(path="answer")
        response = _response(json.dumps({"answer": "Paris"}))
        result = await grader.grade(item=_item(expected="Paris"), response=response)
        assert result.passed is True

    async def test_path_not_equal_to_item_expected(self) -> None:
        grader = JSONField(path="answer")
        response = _response(json.dumps({"answer": "London"}))
        result = await grader.grade(item=_item(expected="Paris"), response=response)
        assert result.passed is False
        assert "actual='London'" in result.explanation
        assert "expected='Paris'" in result.explanation

    async def test_nested_path(self) -> None:
        grader = JSONField(path="result.answer")
        response = _response(json.dumps({"result": {"answer": "Paris", "confidence": 0.9}}))
        result = await grader.grade(item=_item(expected="Paris"), response=response)
        assert result.passed is True

    async def test_missing_path_resolves_to_none(self) -> None:
        grader = JSONField(path="result.answer")
        response = _response(json.dumps({"result": {"different": "field"}}))
        result = await grader.grade(item=_item(expected="Paris"), response=response)
        assert result.passed is False

    async def test_explicit_expected_overrides_item(self) -> None:
        grader = JSONField(path="answer", expected="Paris")
        # The item says "London" but the grader compares against "Paris".
        response = _response(json.dumps({"answer": "Paris"}))
        result = await grader.grade(item=_item(expected="London"), response=response)
        assert result.passed is True

    async def test_explicit_expected_none_is_literal(self) -> None:
        # Passing expected=None explicitly asserts the value at path
        # is literally None — distinguishing from "no expected supplied".
        grader = JSONField(path="answer", expected=None)
        response = _response(json.dumps({"answer": None}))
        result = await grader.grade(
            item=_item(expected="Paris"),  # would override if MISSING
            response=response,
        )
        assert result.passed is True

    async def test_explicit_expected_none_fails_when_field_present(self) -> None:
        grader = JSONField(path="answer", expected=None)
        response = _response(json.dumps({"answer": "Paris"}))
        result = await grader.grade(item=_item(), response=response)
        assert result.passed is False

    async def test_invalid_json_fails(self) -> None:
        grader = JSONField(path="answer")
        result = await grader.grade(item=_item(expected="Paris"), response=_response("not json"))
        assert result.passed is False
        assert "not JSON" in result.explanation

    async def test_traversing_non_dict_returns_none(self) -> None:
        grader = JSONField(path="answer.deeper")
        response = _response(json.dumps({"answer": "Paris"}))  # string, not dict
        result = await grader.grade(item=_item(expected="anything"), response=response)
        assert result.passed is False

    async def test_default_name(self) -> None:
        grader = JSONField(path="answer.text")
        assert grader.name == "json_field(answer.text)"

    async def test_custom_name(self) -> None:
        grader = JSONField(path="answer", name="answer-check")
        assert grader.name == "answer-check"

    def test_empty_path_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            JSONField(path="")

    async def test_integer_value_comparison(self) -> None:
        grader = JSONField(path="count", expected=42)
        response = _response(json.dumps({"count": 42}))
        result = await grader.grade(item=_item(), response=response)
        assert result.passed is True
