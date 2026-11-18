from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from strata_forge.pipelines._grading import (
    Rubric,
    parse_grade,
    render_criteria,
    rubric_digest,
    rubric_schema,
)
from strata_forge.pipelines._guided import OutputValidator, schema_problem

_TEMPLATE = "Question: {question}\nAnswer: {output}\nScore it on:\n{criteria}"


def _rubric(**overrides: Any) -> Rubric:
    base: dict[str, Any] = {
        "name": "support quality",
        "template": _TEMPLATE,
        "criteria": [
            {"id": "policy", "description": "Follows the refund policy.", "scale": "pass_fail"},
            {
                "id": "helpfulness",
                "description": "How helpful the answer is.",
                "scale": "ordinal",
                "min": 1,
                "max": 5,
                "pass_at": 4,
            },
        ],
    }
    return Rubric.model_validate(base | overrides)


def test_the_template_must_show_the_output_and_the_criteria() -> None:
    with pytest.raises(ValidationError, match=r"\{output\}"):
        _rubric(template="Score {criteria}")
    with pytest.raises(ValidationError, match=r"\{criteria\}"):
        _rubric(template="Score {output}")
    assert _rubric().column_placeholders() == {"question"}


@pytest.mark.parametrize(
    ("criterion", "message"),
    [
        ({"id": "Bad", "description": "x", "scale": "pass_fail"}, "lowercase"),
        ({"id": "a", "description": "x", "scale": "pass_fail", "min": 0}, "takes no min"),
        ({"id": "a", "description": "x", "scale": "ordinal", "min": 1}, "needs min and max"),
        ({"id": "a", "description": "x", "scale": "ordinal", "min": 0, "max": 100}, "<= 10"),
        (
            {"id": "a", "description": "x", "scale": "ordinal", "min": 1, "max": 5, "pass_at": 1},
            "pass_at",
        ),
    ],
)
def test_criterion_refusals(criterion: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _rubric(criteria=[criterion])


def test_criterion_ids_are_unique() -> None:
    same = {"id": "a", "description": "x", "scale": "pass_fail"}
    with pytest.raises(ValidationError, match="unique"):
        _rubric(criteria=[same, same])


def test_the_schema_is_a_guided_json_schema_with_the_rationale_first() -> None:
    schema = rubric_schema(_rubric())
    assert schema_problem(schema) is None
    assert list(schema["properties"]) == ["rationale", "scores"]
    validator = OutputValidator(schema)
    good = {"rationale": "cites the policy", "scores": {"policy": True, "helpfulness": 4}}
    assert validator.error(json.dumps(good)) is None
    for bad in (
        {**good, "scores": {"policy": True, "helpfulness": 6}},
        {**good, "scores": {"policy": 1, "helpfulness": 4}},
        {**good, "extra": 1},
        {"scores": good["scores"]},
    ):
        assert validator.error(json.dumps(bad)) is not None, bad


def test_parse_grade_reads_a_complete_answer_only() -> None:
    rubric = _rubric()
    grade = parse_grade(
        json.dumps({"rationale": "ok", "scores": {"policy": False, "helpfulness": 5}}), rubric
    )
    assert grade is not None
    assert grade.scores == {"policy": 0, "helpfulness": 5}
    for bad in (
        None,
        "",
        "not json",
        "[]",
        json.dumps({"rationale": "", "scores": {"policy": True, "helpfulness": 3}}),
        json.dumps({"rationale": "x", "scores": {"policy": True}}),
        json.dumps({"rationale": "x", "scores": {"policy": 1, "helpfulness": 3}}),
        json.dumps({"rationale": "x", "scores": {"policy": True, "helpfulness": True}}),
        json.dumps({"rationale": "x", "scores": {"policy": True, "helpfulness": 0}}),
        json.dumps({"rationale": "x", "scores": {"policy": True, "helpfulness": 3, "x": 1}}),
    ):
        assert parse_grade(bad, rubric) is None, bad


def test_scales_passes_and_the_criteria_text() -> None:
    rubric = _rubric()
    policy, helpful = rubric.criteria
    assert policy.values() == [0, 1]
    assert helpful.values() == [1, 2, 3, 4, 5]
    assert (policy.passes(1), policy.passes(0)) == (True, False)
    assert (helpful.passes(4), helpful.passes(3)) == (True, False)
    assert render_criteria(rubric) == (
        "1. policy (true if it passes, false if it fails): Follows the refund policy.\n"
        "2. helpfulness (an integer from 1 to 5): How helpful the answer is."
    )


def test_the_digest_pins_the_rubric() -> None:
    assert rubric_digest(_rubric()) == rubric_digest(_rubric())
    assert rubric_digest(_rubric()) != rubric_digest(_rubric(name="other"))
