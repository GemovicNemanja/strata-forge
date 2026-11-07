"""Malformed diagnostic selectors cannot execute hashing, equality or coercion hooks."""

from __future__ import annotations

from typing import cast

import pytest

from strata_forge.pipelines.evaluation_analysis_spec import EvaluationAnalysisError


@pytest.mark.parametrize(
    "value",
    [{}, [], None, True, 7, "unknown", "x" * 4096],
    ids=["dict", "list", "null", "bool", "integer", "unknown", "long"],
)
def test_error_selector_has_fixed_fallback(value: object) -> None:
    error = EvaluationAnalysisError(cast("str", value))
    assert error.args == ("analysis_invalid",)
    assert error.__context__ is error.__cause__ is None


def test_error_selector_refuses_hostile_object_and_string_subclass_without_callbacks() -> None:
    calls: list[str] = []

    class Hostile:
        def __hash__(self) -> int:
            calls.append("hash")
            raise AssertionError("must not hash caller data")

        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            raise AssertionError("must not compare caller data")

        def __str__(self) -> str:
            calls.append("str")
            raise AssertionError("must not coerce caller data")

    class HostileString(str):
        def __hash__(self) -> int:
            calls.append("hash")
            raise AssertionError("must not hash a subclass")

        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            raise AssertionError("must not compare a subclass")

        def __str__(self) -> str:
            calls.append("str")
            raise AssertionError("must not coerce a subclass")

        def __len__(self) -> int:
            calls.append("len")
            raise AssertionError("must not inspect a subclass")

    for value in (Hostile(), HostileString("analysis_denied")):
        assert EvaluationAnalysisError(cast("str", value)).args == ("analysis_invalid",)
    assert calls == []


@pytest.mark.parametrize(
    "code",
    [
        "analysis_invalid",
        "analysis_unavailable",
        "analysis_denied",
        "analysis_limit",
        "analysis_input_invalid",
        "analysis_population_mismatch",
        "analysis_source_changed",
        "analysis_deadline",
        "analysis_failed",
    ],
)
def test_only_builtin_fixed_codes_are_preserved(code: str) -> None:
    assert EvaluationAnalysisError(code).args == (code,)
