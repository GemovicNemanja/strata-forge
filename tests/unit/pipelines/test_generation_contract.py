"""Finite generation wires and stable sample vectors without an engine, network or model."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from strata_forge.llm._generation_schema import GenerationError, canonical_json
from strata_forge.llm.generation import GenerationRequest, revalidate_request
from strata_forge.pipelines._generation_samples import plan_samples, sample_request
from strata_forge.pipelines.generation_spec import (
    GenerationRunSpecV2,
    parse_generation_spec,
    revalidate_generation_spec,
)

# Heterogeneous serialized fixture data intentionally crosses strict Pydantic boundaries.
FIXTURE: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures/generation_v2.json").read_text()
)


def raw_spec() -> dict[str, Any]:
    return copy.deepcopy(FIXTURE["generation_spec"])


def test_golden_wire_digest_and_model_dict_json_roundtrips() -> None:
    raw = raw_spec()
    spec = GenerationRunSpecV2.model_validate(raw)
    assert spec.wire_payload() == raw
    assert (
        canonical_json(spec.wire_payload()).hex() == FIXTURE["generation_spec_canonical_utf8_hex"]
    )
    assert spec.digest() == FIXTURE["generation_spec_sha256"]
    for again in (
        GenerationRunSpecV2.model_validate(spec),
        GenerationRunSpecV2.model_validate(spec.model_dump()),
        GenerationRunSpecV2.model_validate_json(spec.model_dump_json()),
        parse_generation_spec(spec.model_dump_json()),
        revalidate_generation_spec(spec),
    ):
        assert again.wire_payload() == raw
        assert again.digest() == spec.digest()
    assert spec.input_reads() == ((spec.model_ref, "model"), (spec.dataset_ref, "dataset"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("samples", 0),
        ("samples", 17),
        ("samples", True),
        ("samples", "2"),
        ("seed", None),
        ("temperature", True),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("top_p", 0),
        ("top_p", 1.1),
        ("top_k", 0),
        ("top_k", 1025),
        ("min_p", -0.1),
        ("max_tokens", 0),
        ("max_tokens", "12"),
        ("stop", [""]),
        ("stop", ["a", "a"]),
        ("stop", ["x" * 257]),
        ("stop", ["x"] * 9),
        ("stop", ["\ud800"]),
        ("unknown_knob", None),
        ("seed", {"mode": "per_item_sample_v1", "base_seed": True}),
        ("seed", {"mode": "per_item_sample_v1", "base_seed": 2**53}),
    ],
)
def test_closed_generation_controls_refuse(field: str, value: object) -> None:
    raw = raw_spec()
    raw["generation"][field] = value
    with pytest.raises((ValidationError, GenerationError)):
        GenerationRunSpecV2.model_validate(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", "2"),
        ("schema_version", 1),
        ("profile", "unknown"),
        ("model_ref", "fixture/model@main"),
        ("dataset_ref", "fixture/source"),
        ("endpoint", "http://127.0.0.1:1"),
        ("overflow", "drop"),
        ("template", "{uncovered}"),
        ("id_column", None),
    ],
)
def test_top_level_unknown_or_unsupported_is_not_silently_translated(
    field: str, value: object
) -> None:
    raw = raw_spec()
    raw[field] = value
    with pytest.raises((ValidationError, GenerationError)):
        GenerationRunSpecV2.model_validate(raw)


@pytest.mark.parametrize(
    "channels",
    [
        {"mode": "gpt_oss", "effort": "high"},
        {"mode": "qwen3", "enable_thinking": True},
    ],
)
def test_profile_binding_is_validated_but_never_implies_capability(
    channels: dict[str, object],
) -> None:
    raw = raw_spec()
    raw["generation"]["channels"] = channels
    with pytest.raises(ValidationError, match="generation_profile_mismatch"):
        GenerationRunSpecV2.model_validate(raw)
    raw["profile"] = (
        "vllm-0.24-qwen3/v1" if channels["mode"] == "qwen3" else "vllm-0.24-gpt-oss-harmony/v1"
    )
    spec = GenerationRunSpecV2.model_validate(raw)
    assert spec.generation.channels.mode == channels["mode"]


def test_seed_vectors_stable_under_order_run_and_model_condition_changes() -> None:
    raw = raw_spec()
    raw["selection"]["row_limit"] = 3
    spec = GenerationRunSpecV2.model_validate(raw)
    ids = ["row-a", "row|a", "row-\u03b1"]
    planned = list(plan_samples(spec, ids))
    assert [(p.item_id, p.sample_index, p.effective_seed_decimal) for p in planned] == [
        (p["item_id"], p["sample_index"], p["effective_seed_decimal"])
        for p in FIXTURE["seed_vectors"]
    ]
    expected = {(p.item_id, p.sample_index): p.effective_seed_decimal for p in planned}
    raw.update(run_id="another-run", model_ref="fixture/different-model@" + "c" * 40)
    raw["serving"]["concurrency"] = 1
    changed = GenerationRunSpecV2.model_validate(raw)
    assert {
        (p.item_id, p.sample_index): p.effective_seed_decimal
        for p in plan_samples(changed, list(reversed(ids)))
    } == expected
    raw["dataset_ref"] = "fixture/source@" + "d" * 40
    other = GenerationRunSpecV2.model_validate(raw)
    assert all(
        p.effective_seed_decimal != expected[(p.item_id, p.sample_index)]
        for p in plan_samples(other, ids)
    )
    for p in planned:
        request = sample_request(spec, p)
        assert request.effective_seed_decimal == p.effective_seed_decimal
        assert GenerationRequest.model_validate_json(request.model_dump_json()) == request


@pytest.mark.parametrize("samples", [1, 2, 16])
def test_logical_sample_bounds_and_identity(samples: int) -> None:
    raw = raw_spec()
    raw["generation"]["samples"] = samples
    raw["generation"]["seed"] = {"mode": "engine_default"}
    spec = GenerationRunSpecV2.model_validate(raw)
    values = list(plan_samples(spec, ["a", "a|0"]))
    assert len(values) == 2 * samples
    assert [(p.item_id, p.sample_index) for p in values] == [
        (item, i) for item in ("a", "a|0") for i in range(samples)
    ]
    assert all(p.effective_seed_decimal is None for p in values)


@pytest.mark.parametrize("ids", [["same", "same"], [""], ["\ud800"], ["x" * 257], []])
def test_all_ids_validated_before_any_sample_iterator_is_returned(ids: list[str]) -> None:
    with pytest.raises(GenerationError):
        plan_samples(GenerationRunSpecV2.model_validate(raw_spec()), ids)


def test_selection_and_total_sample_cap_refuse_without_a_prefix() -> None:
    raw = raw_spec()
    spec = GenerationRunSpecV2.model_validate(raw)
    with pytest.raises(GenerationError, match="generation_selection_mismatch"):
        plan_samples(spec, ["a", "b", "c"])
    raw["selection"]["row_limit"] = 1_000_000
    raw["generation"]["samples"] = 16
    spec = GenerationRunSpecV2.model_validate(raw)
    with pytest.raises(GenerationError, match="generation_sample_limit_exceeded"):
        plan_samples(spec, ["a"] * 62_501)


def test_mutable_nested_caller_data_and_model_copy_cannot_bypass_validation() -> None:
    spec = GenerationRunSpecV2.model_validate(raw_spec())
    captured = revalidate_generation_spec(spec)
    spec.generation.stop.append("changed")
    assert captured.generation.stop == []
    broken = spec.model_copy(update={"unknown_knob": True})
    with pytest.raises(GenerationError):
        revalidate_generation_spec(broken)
    request = sample_request(captured, next(plan_samples(captured, ["a"])))
    broken_request = request.model_copy(update={"top_k": 0})
    with pytest.raises(GenerationError):
        revalidate_request(broken_request)


@pytest.mark.parametrize("raw", ['{"schema_version":2,"schema_version":2}', "NaN", '"\ud800"'])
def test_bounded_wire_parser_fixed_context_free_refusal(raw: str) -> None:
    with pytest.raises(GenerationError) as caught:
        parse_generation_spec(raw)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_json_schema_alias_and_mutation_roundtrip_stays_on_exact_wire() -> None:
    raw = raw_spec()
    raw["generation"]["output_format"] = {
        "kind": "json_schema",
        "dialect": "strata-json-schema-v1",
        "schema": copy.deepcopy(FIXTURE["schema"]),
    }
    spec = GenerationRunSpecV2.model_validate(raw)
    copied = revalidate_generation_spec(spec)
    assert copied.wire_payload() == raw
    assert copied.digest() == spec.digest()
    assert parse_generation_spec(copied.model_dump_json()).digest() == spec.digest()
    wrong = copy.deepcopy(raw)
    wrong["generation"]["output_format"]["schema_body"] = wrong["generation"]["output_format"].pop(
        "schema"
    )
    with pytest.raises(ValidationError):
        GenerationRunSpecV2.model_validate(wrong)
