"""Exact inert rendering, safe refusals and complete effective-config digests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from strata_forge.training.verifier_config import VerifierConfig, render_verifier_input


def config(**overrides: object) -> VerifierConfig:
    return VerifierConfig.model_validate(
        {
            "labels": ["E", "C", "U"],
            "template": "Premises:\n{premises}",
            "pair_template": "Hypothesis: {hypothesis}",
            "premise_format": "numbered_lines",
            "max_length": 128,
            "max_premises": 20,
            **overrides,
        }
    )


def test_numbered_rendering_has_exact_boundaries_and_does_not_reinterpret_row_braces() -> None:
    rendered = render_verifier_input(config(), ["A", "B {hypothesis}"], "target {premises}")
    assert rendered.text == "Premises:\n1. A\n2. B {hypothesis}"
    assert rendered.text_pair == "Hypothesis: target {premises}"
    assert rendered.error_code is None


def test_json_is_compact_unicode_and_never_python_repr() -> None:
    rendered = render_verifier_input(config(premise_format="json"), ['é "quoted"', "B\nC"], "H")
    assert rendered.text == 'Premises:\n["é \\"quoted\\"","B\\nC"]'.replace('\\\\"', '\\"')


@pytest.mark.parametrize(
    "template",
    ["{premises.__class__}", "{premises[0]}", "{x}", "{", "}", "{{premises}}", "no placeholders"],
)
def test_unknown_executable_or_missing_placeholders_refuse(template: str) -> None:
    with pytest.raises(ValidationError):
        config(template=template)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_length", 8193),
        ("max_length", True),
        ("max_premises", 21),
        ("max_premises", 0),
        ("labels", ["E", "E"]),
        ("label_tokens", {"E": "a"}),
    ],
)
def test_invalid_declared_renderer_shape(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        config(**{field: value})


@pytest.mark.parametrize(
    ("premises", "hypothesis", "reason"),
    [
        (["a"] * 21, "h", "premise_limit_exceeded"),
        ("a", "h", "input_shape_invalid"),
        ([1], "h", "input_shape_invalid"),
        (["a"], None, "input_shape_invalid"),
        (["a"], "h" * 200_001, "rendered_input_too_long"),
    ],
)
def test_row_refusals_never_slice_or_echo_input(
    premises: object, hypothesis: object, reason: str
) -> None:
    rendered = render_verifier_input(config(), premises, hypothesis)
    assert rendered.error_code == reason
    assert rendered.text is None
    assert rendered.text_pair is None


def test_lower_limits_are_declared_scope_and_config_digest_tracks_effective_changes() -> None:
    value = config(max_premises=1, max_length=32)
    assert render_verifier_input(value, ["a", "b"], "h").error_code == "premise_limit_exceeded"
    assert value.digest() == VerifierConfig.model_validate_json(value.model_dump_json()).digest()
    assert value.digest() != config(max_premises=1, max_length=64).digest()


def test_repeated_placeholders_are_bounded_before_expanding_large_row_values() -> None:
    value = config(template="{premises}" * 10_000)
    rendered = render_verifier_input(value, ["x" * 100_000], "h")
    assert rendered.error_code == "rendered_input_too_long"
    assert rendered.text is None
