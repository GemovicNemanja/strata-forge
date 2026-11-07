"""Completion, channel presence and JSON validity remain independent explicit facts."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from strata_forge.llm.generation import GenerationResponse


def response() -> dict[str, Any]:
    # Nullable channel fields are present in real protocol objects without implying content.
    return {
        "status": "completed",
        "finish_reason": "stop",
        "final_present": True,
        "final_text": "<think>literal</think>",
        "reasoning_present": True,
        "reasoning_text": None,
        "refusal_present": True,
        "refusal_text": None,
        "json_status": "not_requested",
        "schema_status": "not_requested",
        "parsed_json": None,
        "usage": {
            "input_tokens": None,
            "generated_tokens": None,
            "reasoning_tokens": None,
            "final_tokens": None,
        },
    }


def test_present_null_reasoning_and_refusal_do_not_change_literal_final() -> None:
    raw = response()
    parsed = GenerationResponse.model_validate(raw)
    assert parsed.final_text == raw["final_text"]
    assert parsed.reasoning_present
    assert parsed.reasoning_text is None
    assert parsed.refusal_present
    assert parsed.refusal_text is None
    assert parsed.usage.generated_tokens is None


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("final_text", None),
        ("reasoning_present", False),
        ("finish_reason", "length"),
        ("refusal_text", "refused"),
        ("refusal_present", False),
        ("json_status", "invalid"),
        ("schema_status", "valid"),
    ],
)
def test_inconsistent_completed_receipt_refuses(change: str, value: object) -> None:
    raw = response()
    raw[change] = value
    if change == "reasoning_present":
        raw["reasoning_text"] = "separate text"
    if change == "refusal_present":
        raw["refusal_text"] = "separate refusal"
    with pytest.raises(ValidationError):
        GenerationResponse.model_validate(raw)


def test_length_ended_valid_json_retains_validity_without_success() -> None:
    raw = response()
    raw.update(
        status="incomplete_length",
        finish_reason="length",
        final_text='{"a":1}',
        json_status="valid",
        schema_status="valid",
        parsed_json={"a": 1},
    )
    parsed = GenerationResponse.model_validate(raw)
    assert parsed.status == "incomplete_length"
    assert parsed.schema_status == "valid"


@pytest.mark.parametrize(
    ("status", "finish", "refusal"),
    [
        ("incomplete_length", "stop", None),
        ("provider_refusal", "stop", None),
        ("invalid_schema", "stop", None),
        ("invalid_json", "stop", None),
    ],
)
def test_false_outcome_or_validity_cannot_be_constructed(
    status: str, finish: str, refusal: object
) -> None:
    raw = response()
    raw.update(status=status, finish_reason=finish, refusal_text=refusal)
    with pytest.raises(ValidationError):
        GenerationResponse.model_validate(raw)


def test_usage_never_double_counts_reasoning_or_invents_missing_counts() -> None:
    raw = response()
    raw["usage"].update(generated_tokens=7, reasoning_tokens=3, final_tokens=4)
    assert GenerationResponse.model_validate(raw).usage.generated_tokens == 7
    raw["usage"]["final_tokens"] = 7
    with pytest.raises(ValidationError):
        GenerationResponse.model_validate(raw)
