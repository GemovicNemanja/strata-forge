"""Exact-match and regex graders.

Both work on ``response.text``. ``ExactMatch`` compares against the
dataset item's ``expected_output`` (stringified). ``Regex`` checks a
caller-supplied pattern — the dataset item's expected output isn't
involved, so this grader works for items that don't carry one.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from strata_forge.evals.experiment import GraderResult

if TYPE_CHECKING:
    from strata_forge.datasets.schema import DatasetItem
    from strata_forge.llm.responses import LLMResponse

__all__ = [
    "ExactMatch",
    "Regex",
]


class ExactMatch:
    """Pass when ``response.text`` equals ``item.expected_output``.

    ``case_sensitive=False`` lower-cases both sides before comparing;
    ``strip=True`` (default) trims leading/trailing whitespace so a
    model's trailing newline doesn't fail the match.

    When ``item.expected_output`` is ``None``, the grader returns a
    failed result with an explanation — exact match against an absent
    label is undefined.
    """

    def __init__(
        self,
        *,
        case_sensitive: bool = True,
        strip: bool = True,
        name: str | None = None,
    ) -> None:
        self.case_sensitive = case_sensitive
        self.strip = strip
        self._name = name or "exact_match"

    @property
    def name(self) -> str:
        return self._name

    async def grade(self, *, item: DatasetItem, response: LLMResponse) -> GraderResult:
        if item.expected_output is None:
            return GraderResult(
                grader_name=self.name,
                score=0.0,
                passed=False,
                explanation="item has no expected_output",
            )
        text = response.text
        expected = str(item.expected_output)
        if self.strip:
            text = text.strip()
            expected = expected.strip()
        if not self.case_sensitive:
            text = text.lower()
            expected = expected.lower()
        matched = text == expected
        return GraderResult(
            grader_name=self.name,
            score=1.0 if matched else 0.0,
            passed=matched,
            explanation=("" if matched else f"expected {expected!r}, got {text!r}"),
        )


class Regex:
    """Pass when ``response.text`` matches a regex.

    ``expected_match=False`` inverts the check — useful for
    "must NOT contain X" assertions.

    The grader's name defaults to ``regex(<pattern>)``; override with
    ``name=`` for cleaner report output.
    """

    def __init__(
        self,
        pattern: str | re.Pattern[str],
        *,
        expected_match: bool = True,
        name: str | None = None,
    ) -> None:
        self.pattern: re.Pattern[str] = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.expected_match = expected_match
        self._name = name or f"regex({self.pattern.pattern})"

    @property
    def name(self) -> str:
        return self._name

    async def grade(self, *, item: DatasetItem, response: LLMResponse) -> GraderResult:
        del item  # not used by this grader
        matched = bool(self.pattern.search(response.text))
        passed = matched == self.expected_match
        return GraderResult(
            grader_name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            explanation=(f"matched={matched}, expected_match={self.expected_match}"),
        )
