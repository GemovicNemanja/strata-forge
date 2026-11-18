"""A plugin manifest is data: parsed as JSON, validated closed, refused when it is anything else."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from strata_forge.plugins.manifest import (
    MAX_MANIFEST_BYTES,
    PluginManifestError,
    check_scalar,
    parse_manifest,
)

from .conftest import MANIFEST


def _with(**changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(MANIFEST)
    data.update(changes)
    return data


def _entry(index: int, **changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(MANIFEST)
    data["entry_points"][index].update(changes)
    return data


def test_a_valid_manifest_parses() -> None:
    manifest = parse_manifest(json.dumps(MANIFEST))
    assert manifest.distribution == "toy-prm"
    prm = manifest.entry("prm")
    assert prm is not None
    assert prm.type == "step_scorer"
    assert manifest.entry("nope") is None


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[]",
        json.dumps(_with(schema="strata.plugin/v2")).encode(),
        json.dumps(_with(extra="field")).encode(),
        json.dumps(_with(distribution="Toy_PRM")).encode(),
        json.dumps(_with(version="0.1.0+local")).encode(),
        json.dumps(_with(entry_points=[])).encode(),
        b"x" * (MAX_MANIFEST_BYTES + 1),
    ],
)
def test_malformed_or_undeclared_manifests_are_refused(raw: bytes) -> None:
    with pytest.raises(PluginManifestError):
        parse_manifest(raw)


def test_an_entry_name_cannot_be_an_import_path() -> None:
    with pytest.raises(PluginManifestError):
        parse_manifest(json.dumps(_entry(1, name="toy_prm:job")))
    with pytest.raises(PluginManifestError):
        parse_manifest(json.dumps(_entry(1, name="os.system ls")))


def test_duplicate_names_are_refused() -> None:
    data = copy.deepcopy(MANIFEST)
    data["entry_points"].append(copy.deepcopy(data["entry_points"][1]))
    with pytest.raises(PluginManifestError, match="duplicate entry point"):
        parse_manifest(json.dumps(data))


def test_a_step_scorer_must_read_steps_and_write_scores() -> None:
    with pytest.raises(PluginManifestError, match="step_scorer"):
        parse_manifest(json.dumps(_entry(0, inputs=[{"role": "items"}])))
    with pytest.raises(PluginManifestError, match="step_scorer"):
        parse_manifest(
            json.dumps(_entry(0, inputs=[{"role": "steps", "required_columns": ["step"]}]))
        )
    with pytest.raises(PluginManifestError, match="step_scorer"):
        parse_manifest(json.dumps(_entry(0, outputs=[{"role": "out", "kind": "dataset"}])))


def test_a_step_scorer_declares_its_score_kind() -> None:
    with pytest.raises(PluginManifestError, match="score_kind"):
        parse_manifest(json.dumps(_entry(0, score_kind=None)))
    with pytest.raises(PluginManifestError):
        parse_manifest(json.dumps(_entry(0, score_kind="percent")))
    for kind in ("probability", "logit", "unconstrained"):
        assert (
            parse_manifest(json.dumps(_entry(0, score_kind=kind))).entry_points[0].score_kind
            == kind
        )


def test_scoring_fields_belong_to_a_step_scorer() -> None:
    for field, value in (
        ("score_kind", "logit"),
        ("unscored_reasons", ["unparseable"]),
        ("passthrough_columns", ["text"]),
    ):
        with pytest.raises(PluginManifestError, match="belong to a step_scorer"):
            parse_manifest(json.dumps(_entry(1, **{field: value})))


def test_pass_through_columns_are_required_and_never_forge_s_own() -> None:
    steps = {
        "role": "steps",
        "required_columns": ["item_id", "step_index", "context", "step", "source"],
    }
    ok = _entry(0, inputs=[steps], passthrough_columns=["source"])
    assert parse_manifest(json.dumps(ok)).entry_points[0].passthrough_columns == ("source",)
    with pytest.raises(PluginManifestError, match="must be required columns"):
        parse_manifest(json.dumps(_entry(0, inputs=[steps], passthrough_columns=["other"])))
    for reserved in ("score", "reason", "step_type", "item_id"):
        with pytest.raises(PluginManifestError, match="written by forge"):
            parse_manifest(json.dumps(_entry(0, inputs=[steps], passthrough_columns=[reserved])))
    with pytest.raises(PluginManifestError, match="duplicate"):
        parse_manifest(json.dumps(_entry(0, unscored_reasons=["a", "a"])))


def test_a_model_input_has_no_columns() -> None:
    bad = _entry(1, inputs=[{"role": "m", "kind": "model", "required_columns": ["x"]}])
    with pytest.raises(PluginManifestError, match="no columns"):
        parse_manifest(json.dumps(bad))


@pytest.mark.parametrize(
    "param",
    [
        {"name": "x", "type": "integer", "enum": ["a"]},
        {"name": "x", "type": "string", "minimum": 1},
        {"name": "x", "type": "number", "minimum": 2, "maximum": 1},
        {"name": "x", "type": "number", "required": True, "default": 1},
        {"name": "x", "type": "integer", "default": "1"},
        {"name": "X", "type": "integer"},
    ],
)
def test_inconsistent_parameters_are_refused(param: dict[str, Any]) -> None:
    with pytest.raises(PluginManifestError):
        parse_manifest(json.dumps(_entry(1, config=[param])))


def test_scalars_are_checked_strictly() -> None:
    manifest = parse_manifest(json.dumps(MANIFEST))
    prm = manifest.entry("prm")
    assert prm is not None
    scale, mode = prm.param("scale"), prm.param("mode")
    assert scale is not None
    assert mode is not None
    assert check_scalar(scale, 2) is None
    assert check_scalar(scale, True) == "must be a number"
    assert check_scalar(scale, float("nan")) == "must be a finite number"
    assert check_scalar(scale, 11) == "must be at most 10"
    assert check_scalar(mode, "fast") is None
    assert check_scalar(mode, "medium") == "must be one of ['fast', 'slow']"
    assert check_scalar(mode, 1) == "must be a string"
