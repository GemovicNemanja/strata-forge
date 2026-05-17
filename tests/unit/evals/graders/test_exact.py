"""Unit tests for `forge.evals.graders.exact`."""

from __future__ import annotations

import re
from typing import Any

from forge.datasets.schema import DatasetItem
from forge.evals.graders.exact import ExactMatch, Regex
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


# ---------------------------------------------------------------------------
# ExactMatch
# ---------------------------------------------------------------------------


class TestExactMatch:
    async def test_matching_text_passes(self) -> None:
        grader = ExactMatch()
        result = await grader.grade(item=_item(expected="Paris"), response=_response("Paris"))
        assert result.passed is True
        assert result.score == 1.0
        assert result.grader_name == "exact_match"

    async def test_nonmatching_text_fails(self) -> None:
        grader = ExactMatch()
        result = await grader.grade(item=_item(expected="Paris"), response=_response("London"))
        assert result.passed is False
        assert result.score == 0.0
        assert "Paris" in result.explanation
        assert "London" in result.explanation

    async def test_strip_trims_whitespace(self) -> None:
        grader = ExactMatch()  # strip=True by default
        result = await grader.grade(item=_item(expected="Paris"), response=_response("  Paris\n"))
        assert result.passed is True

    async def test_strip_false_keeps_whitespace_significant(self) -> None:
        grader = ExactMatch(strip=False)
        result = await grader.grade(item=_item(expected="Paris"), response=_response("  Paris\n"))
        assert result.passed is False

    async def test_case_insensitive(self) -> None:
        grader = ExactMatch(case_sensitive=False)
        result = await grader.grade(item=_item(expected="paris"), response=_response("PARIS"))
        assert result.passed is True

    async def test_case_sensitive_default(self) -> None:
        grader = ExactMatch()  # case_sensitive=True
        result = await grader.grade(item=_item(expected="paris"), response=_response("PARIS"))
        assert result.passed is False

    async def test_missing_expected_output_fails(self) -> None:
        grader = ExactMatch()
        result = await grader.grade(item=_item(expected=None), response=_response("anything"))
        assert result.passed is False
        assert "no expected_output" in result.explanation

    async def test_non_string_expected_stringified(self) -> None:
        grader = ExactMatch()
        result = await grader.grade(item=_item(expected=42), response=_response("42"))
        assert result.passed is True

    async def test_custom_name(self) -> None:
        grader = ExactMatch(name="my-grader")
        assert grader.name == "my-grader"
        result = await grader.grade(item=_item(expected="x"), response=_response("x"))
        assert result.grader_name == "my-grader"


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------


class TestRegex:
    async def test_matching_pattern_passes(self) -> None:
        grader = Regex(r"\bParis\b")
        result = await grader.grade(item=_item(), response=_response("The capital is Paris."))
        assert result.passed is True
        assert result.score == 1.0

    async def test_nonmatching_pattern_fails(self) -> None:
        grader = Regex(r"\bLondon\b")
        result = await grader.grade(item=_item(), response=_response("The capital is Paris."))
        assert result.passed is False
        assert result.score == 0.0

    async def test_expected_match_false_inverts(self) -> None:
        # "Must NOT contain X"
        grader = Regex(r"forbidden", expected_match=False)
        result = await grader.grade(item=_item(), response=_response("clean response"))
        assert result.passed is True

    async def test_expected_match_false_fails_on_match(self) -> None:
        grader = Regex(r"forbidden", expected_match=False)
        result = await grader.grade(item=_item(), response=_response("contains forbidden text"))
        assert result.passed is False

    async def test_compiled_pattern_accepted(self) -> None:
        pattern = re.compile(r"^[Ee]rror:", re.MULTILINE)
        grader = Regex(pattern)
        result = await grader.grade(
            item=_item(),
            response=_response("preamble\nError: something\nmore"),
        )
        assert result.passed is True

    async def test_default_name_includes_pattern(self) -> None:
        grader = Regex(r"foo")
        assert grader.name == "regex(foo)"

    async def test_custom_name(self) -> None:
        grader = Regex(r"foo", name="contains-foo")
        assert grader.name == "contains-foo"

    async def test_item_expected_output_ignored(self) -> None:
        # Regex doesn't consult item.expected_output — the pattern is
        # the test, not the item's label.
        grader = Regex(r"Paris")
        result_with_label = await grader.grade(
            item=_item(expected="London"), response=_response("Paris is great")
        )
        result_without_label = await grader.grade(
            item=_item(expected=None), response=_response("Paris is great")
        )
        assert result_with_label.passed is result_without_label.passed is True
