"""Unit tests for `strata_forge.evals.graders.base.Grader`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from strata_forge.evals.experiment import GraderResult
from strata_forge.evals.graders import Grader

if TYPE_CHECKING:
    from strata_forge.datasets.schema import DatasetItem
    from strata_forge.llm.responses import LLMResponse


class _DuckTypedGrader:
    """A grader-shaped object that doesn't inherit from anything."""

    @property
    def name(self) -> str:
        return "duck"

    async def grade(self, *, item: DatasetItem, response: LLMResponse) -> GraderResult:
        del item, response
        return GraderResult(grader_name=self.name, score=1.0, passed=True)


class TestRuntimeCheckable:
    def test_duck_typed_object_satisfies_protocol(self) -> None:
        grader = _DuckTypedGrader()
        assert isinstance(grader, Grader)

    def test_object_without_grade_method_does_not_satisfy(self) -> None:
        class _Half:
            @property
            def name(self) -> str:
                return "half"

        assert not isinstance(_Half(), Grader)

    def test_object_without_name_does_not_satisfy(self) -> None:
        class _Half:
            async def grade(
                self, *, item: DatasetItem, response: LLMResponse
            ) -> GraderResult:  # pragma: no cover
                del item, response
                return GraderResult(grader_name="x", score=1.0, passed=True)

        assert not isinstance(_Half(), Grader)
