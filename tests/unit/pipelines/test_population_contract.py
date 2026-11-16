"""Fictional operational population contracts, without model/data/provider IO."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from strata_forge.pipelines._population import (
    PopulationColumns,
    PopulationError,
    PopulationOutcome,
    population_counts,
    population_members,
    selected_population_digest,
)
from strata_forge.pipelines.evaluation_population_spec import EvaluationPopulationSpecV2
from strata_forge.pipelines.evaluation_spec import EvalSpec
from strata_forge.pipelines.inference_runner import RunSpec
from strata_forge.pipelines.scoring_population_spec import InferencePopulationSpecV2

# Deliberately heterogeneous serialized fixture values exercise data-validation boundaries.
FIXTURE: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures/population_v2.json").read_text()
)


def data(name: str = "inference") -> dict[str, Any]:
    return copy.deepcopy(FIXTURE["jobs"][name])


@pytest.mark.parametrize("name", list(FIXTURE["jobs"]))
def test_wire_presence_golden_digest_and_direct_json_dict_roundtrips(name: str) -> None:
    raw = data(name)
    cls = InferencePopulationSpecV2 if name == "inference" else EvaluationPopulationSpecV2
    value = cls.model_validate(raw)
    expected = FIXTURE["digests"][name]
    assert value.wire_payload() == raw
    assert value.model_dump() == raw
    assert value.digest() == expected
    for again in (
        cls.model_validate(value),
        cls.model_validate(value.model_dump()),
        cls.model_validate_json(value.model_dump_json()),
        cls.model_validate_json(json.dumps(raw)),
    ):
        assert again.wire_payload() == raw
        assert again.digest() == expected
    if name == "inference":
        assert set(cast("dict[str, Any]", value.wire_payload()["job"])["hyperparams"]) == set(
            raw["job"]["hyperparams"]
        )
        assert "temperature" not in value.model_dump_json()
        nested = RunSpec.model_validate(raw["job"])
    else:
        nested = EvalSpec.model_validate(raw["job"])
    with_nested = {**raw, "job": nested}
    assert cls.model_validate(with_nested).wire_payload() == raw


@pytest.mark.parametrize("value", [None, True, False, "2", 2.0, 1, 3])
@pytest.mark.parametrize("name", list(FIXTURE["jobs"]))
def test_version_refuses_coercion_and_other_protocols(name: str, value: object) -> None:
    raw = data(name)
    raw["schema_version"] = value
    cls = InferencePopulationSpecV2 if name == "inference" else EvaluationPopulationSpecV2
    with pytest.raises(ValidationError):
        cls.model_validate(raw)


@pytest.mark.parametrize("value", [None, "legacy", "error", True, 1, {}, []])
@pytest.mark.parametrize("name", list(FIXTURE["jobs"]))
def test_overflow_is_required_drop_only(name: str, value: object) -> None:
    raw = data(name)
    raw["overflow"] = value
    cls = InferencePopulationSpecV2 if name == "inference" else EvaluationPopulationSpecV2
    with pytest.raises(ValidationError):
        cls.model_validate(raw)
    del raw["overflow"]
    with pytest.raises(ValidationError):
        cls.model_validate(raw)


@pytest.mark.parametrize(
    "field",
    [
        "temperature",
        "max_tokens",
        "row_limit",
        "concurrency",
        "tensor_parallel_size",
        "wait_timeout_s",
        "progress_chunk",
    ],
)
def test_inference_forbids_present_generation_defaults_even_null(field: str) -> None:
    raw = data()
    raw["job"]["hyperparams"][field] = None
    with pytest.raises(ValidationError):
        InferencePopulationSpecV2.model_validate(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_model_len", True),
        ("max_model_len", "16"),
        ("scoring_batch_size", "2"),
        ("scoring_batch_size", False),
        ("engine", []),
        ("engine", "vllm_chat"),
    ],
)
def test_inference_critical_knobs_are_strict(field: str, value: object) -> None:
    raw = data()
    raw["job"]["hyperparams"][field] = value
    with pytest.raises(ValidationError):
        InferencePopulationSpecV2.model_validate(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_revision", "main"),
        ("dataset_commit_sha", None),
        ("id_column", None),
        ("dataset_config", ""),
        ("output_repo_id", "example/output"),
        ("progress_path", []),
        ("run_id", "../unsafe"),
    ],
)
def test_inference_pins_ids_and_local_only_output(field: str, value: object) -> None:
    raw = data()
    raw["job"][field] = value
    with pytest.raises(ValidationError):
        InferencePopulationSpecV2.model_validate(raw)


def test_unknown_keys_and_eval_population_coverage_refuse() -> None:
    for name in FIXTURE["jobs"]:
        raw = data(name)
        cls = InferencePopulationSpecV2 if name == "inference" else EvaluationPopulationSpecV2
        raw["unknown"] = None
        with pytest.raises(ValidationError):
            cls.model_validate(raw)
    for populations in (
        [],
        [{"evaluation_id": "unknown", "tier_column": "tier", "within_limits_column": None}],
        data("native_evaluation")["populations"] * 2,
    ):
        raw = data("native_evaluation")
        raw["populations"] = populations
        with pytest.raises(ValidationError):
            EvaluationPopulationSpecV2.model_validate(raw)


def test_direct_nested_job_is_copied_and_stale_model_copy_cannot_bypass_profile() -> None:
    raw = data()
    nested = RunSpec.model_validate(raw["job"])
    value = InferencePopulationSpecV2.model_validate({**raw, "job": nested})
    nested.hyperparams.max_model_len = 99
    assert value.job.hyperparams.max_model_len == 16
    tampered = value.model_copy(update={"overflow": "legacy"})
    with pytest.raises(ValidationError):
        InferencePopulationSpecV2.model_validate(tampered)


def test_population_exact_partition_and_invalid_origin_counts() -> None:
    rows = FIXTURE["accounting"]["rows"]
    columns = PopulationColumns(tier_column="tier", within_limits_column="within_limits")
    members = population_members(rows, id_column="item_id", columns=columns)
    assert len(members) == 10
    assert [row.item_id for row in members if not row.within_limits] == ["a6", "b4"]
    outcomes = tuple(
        PopulationOutcome.model_validate(
            {key: row[key] for key in PopulationOutcome.model_fields if key in row}
            | {
                "index": i,
                "refusal_stage": "render"
                if row["outcome"] == "invalid"
                else "input"
                if row["outcome"] == "overflow"
                else None,
            }
        )
        for i, row in enumerate(rows)
    )
    counts = population_counts(outcomes)
    for key, expected in FIXTURE["accounting"]["total"].items():
        assert getattr(counts, key) == expected
    assert counts.rows_model_sent == 4
    assert counts.invalid_not_sent == 2
    assert selected_population_digest(members) != selected_population_digest(
        tuple(reversed(members))
    )


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "missing_id",
        "null_flag",
        "integer_flag",
        "missing_tier",
        "large_tier",
        "many_tiers",
    ],
)
def test_population_metadata_cannot_silently_drop_or_bucket(change: str) -> None:
    rows: list[dict[str, object]] = copy.deepcopy(FIXTURE["accounting"]["rows"])
    if change == "duplicate":
        rows[-1]["item_id"] = rows[0]["item_id"]
    elif change == "missing_id":
        rows[0].pop("item_id")
    elif change in {"null_flag", "integer_flag"}:
        rows[0]["within_limits"] = None if change == "null_flag" else 1
    elif change == "missing_tier":
        rows[0].pop("tier")
    elif change == "large_tier":
        rows[0]["tier"] = "x" * 129
    else:
        rows = [{"item_id": f"r{i}", "tier": f"tier{i}", "within_limits": True} for i in range(65)]
    with pytest.raises(PopulationError):
        population_members(
            rows,
            id_column="item_id",
            columns=PopulationColumns(tier_column="tier", within_limits_column="within_limits"),
        )


def test_nested_selection_cannot_bypass_strict_integer_fields() -> None:
    from strata_forge.pipelines.evaluation_spec import Selection

    raw = data()
    selection = Selection.model_validate(raw["selection"])
    selection = selection.model_copy(update={"row_limit": True})
    with pytest.raises(ValidationError):
        InferencePopulationSpecV2.model_validate({**raw, "selection": selection})


def test_outer_digest_binds_columns_and_condition_pins_without_default_expansion() -> None:
    raw = data()
    original = InferencePopulationSpecV2.model_validate(raw)
    for changed in (
        {**raw, "population": {**raw["population"], "tier_column": "other_tier"}},
        {**raw, "population": {**raw["population"], "within_limits_column": None}},
        {**raw, "job": {**raw["job"], "model_revision": "c" * 40}},
    ):
        other = InferencePopulationSpecV2.model_validate(changed)
        assert other.digest() != original.digest()
        assert (
            InferencePopulationSpecV2.model_validate_json(other.model_dump_json()).digest()
            == other.digest()
        )
