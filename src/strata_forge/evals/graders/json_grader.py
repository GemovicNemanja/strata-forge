"""JSON-shape and JSON-field graders.

``JSONStructure`` passes when ``response.text`` parses as a given
Pydantic schema. ``JSONField`` passes when a dotted path inside the
parsed JSON equals an expected value.

Both graders are deterministic — no LLM calls. The LLM-driven graders
live in :mod:`strata_forge.evals.graders.llm_judge`,
:mod:`strata_forge.evals.graders.pairwise`, and
:mod:`strata_forge.evals.graders.semantic`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ValidationError

from strata_forge.evals.experiment import GraderResult

if TYPE_CHECKING:
    from strata_forge.datasets.schema import DatasetItem
    from strata_forge.llm.responses import LLMResponse

__all__ = [
    "JSONField",
    "JSONStructure",
]

# Sentinel for "no explicit expected value supplied"; the public
# `expected` argument is compared against this identity to decide
# whether to fall back to ``item.expected_output``.
_MISSING: Any = object()


class JSONStructure:
    """Pass when ``response.text`` parses as ``schema``.

    The grader runs the Pydantic schema's ``model_validate_json`` over
    the response text. Passes when validation succeeds, fails with the
    ``ValidationError`` rendered into ``explanation`` when it doesn't.

    Typical use: confirm a structured-output call produced something
    parseable, independent of whether the values are correct (that's
    what :class:`JSONField` is for).
    """

    def __init__(self, *, schema: type[BaseModel], name: str | None = None) -> None:
        self.schema = schema
        self._name = name or f"json_structure({schema.__name__})"

    @property
    def name(self) -> str:
        return self._name

    async def grade(self, *, item: DatasetItem, response: LLMResponse) -> GraderResult:
        del item  # not used
        try:
            self.schema.model_validate_json(response.text)
        except ValidationError as exc:
            # ValidationError covers both malformed-JSON and schema-violation
            # cases; in Pydantic v2 it subclasses ValueError, so a separate
            # JSON-decode catch isn't needed.
            return GraderResult(
                grader_name=self.name,
                score=0.0,
                passed=False,
                explanation=f"validation failed: {exc!s}",
            )
        return GraderResult(grader_name=self.name, score=1.0, passed=True)


class JSONField:
    """Pass when a JSON path inside ``response.text`` equals a target value.

    ``path`` is a dotted string (e.g. ``"answer.text"``). The grader
    parses the response as JSON, walks the path, and compares the
    value at that path with ``expected`` (when provided) or with
    ``item.expected_output`` (otherwise).

    Missing keys mid-path resolve to ``None`` rather than raising,
    so the grader behaves the same whether the schema is malformed
    or the field is simply absent.
    """

    def __init__(
        self,
        *,
        path: str,
        expected: Any = _MISSING,
        name: str | None = None,
    ) -> None:
        if not path:
            msg = "path must be non-empty"
            raise ValueError(msg)
        self.path = path
        self._expected = expected
        self._has_explicit_expected = expected is not _MISSING
        self._name = name or f"json_field({path})"

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _resolve(data: Any, path: str) -> Any:
        cursor: Any = data
        for part in path.split("."):
            if not isinstance(cursor, dict):
                return None
            cursor = cast("dict[str, Any]", cursor).get(part)
        return cursor

    async def grade(self, *, item: DatasetItem, response: LLMResponse) -> GraderResult:
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as exc:
            return GraderResult(
                grader_name=self.name,
                score=0.0,
                passed=False,
                explanation=f"not JSON: {exc!s}",
            )
        actual = self._resolve(data, self.path)
        expected = self._expected if self._has_explicit_expected else item.expected_output
        passed = actual == expected
        return GraderResult(
            grader_name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            explanation=("" if passed else f"actual={actual!r}, expected={expected!r}"),
        )
