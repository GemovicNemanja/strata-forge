"""Inert classifier contract tests require no ML imports or source data."""

from __future__ import annotations

# Malformed JSON fixtures and installed native ML/mock objects are intentionally dynamic.
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from strata_forge.pipelines.classifier_spec import (
    ClassifierTrainSpec,
    ClassifierWorkerError,
    load_classifier_spec,
)


def payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "engine_version": "0.3.0",
        "run_id": "1" * 32,
        "method": "seq_cls",
        "task_type": "SEQ_CLS",
        "base": {
            "ref": "fixture/base@" + "1" * 40,
            "initialization": {"kind": "encoder_random_head"},
        },
        "dataset": {
            "ref": "fixture/data@" + "2" * 40,
            "config": "default",
            "split": "train",
            "format": "classification_text",
            "columns": {"item_id": "id", "text": "text", "label": "label"},
        },
        "training": {
            "labels": ["E", "C", "U"],
            "max_steps": 1,
            "max_length": 16,
            "use_cpu": True,
            "batch_size": 2,
        },
        "progress_path": "progress.jsonl",
    }


def test_roundtrip_and_explicit_objective() -> None:
    spec = ClassifierTrainSpec.model_validate(payload())
    assert load_classifier_spec(spec.canonical()) == spec
    assert spec.training.sdk_config().brier_weight == 0
    p = payload()
    p["training"]["objective_preset"] = "research_ce_brier_0_1"
    other = ClassifierTrainSpec.model_validate(p)
    assert other.training.sdk_config().brier_weight == 0.1
    assert other.digest() != spec.digest()


@pytest.mark.parametrize(
    "key",
    [
        "adapter",
        "merge_adapter",
        "output",
        "output_repo_id",
        "resume_from_checkpoint",
        "class_weights",
        "label_smoothing",
    ],
)
def test_unavailable_top_fields_refuse_even_null(key: Any) -> None:
    p = payload()
    p[key] = None
    with pytest.raises(ValidationError):
        ClassifierTrainSpec.model_validate(p)


@pytest.mark.parametrize(
    "key",
    [
        "class_weights",
        "sample_weight",
        "label_smoothing",
        "source_objective",
        "brier_weight",
        "selection_split",
        "support",
        "resume",
    ],
)
def test_unavailable_training_fields_refuse_even_null(key: Any) -> None:
    p = payload()
    p["training"][key] = None
    with pytest.raises(ValidationError):
        ClassifierTrainSpec.model_validate(p)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":1,"schema_version":1}',
        b'{"private":NaN}',
        b"\xff",
        b"[]",
        b'"private-canary"',
    ],
)
def test_raw_parser_refuses_without_retaining_input(raw: Any) -> None:
    with pytest.raises(ClassifierWorkerError) as error:
        load_classifier_spec(raw)
    assert error.value.__context__ is None
    assert "private" not in str(error.value)


def test_index_map_and_selection_are_explicit() -> None:
    p = payload()
    p["training"]["label_index_map"] = {"0": "E", "1": "C"}
    assert ClassifierTrainSpec.model_validate(p).training.sdk_config().label_index_map == {
        0: "E",
        1: "C",
    }
    p["dataset"]["selection"] = {"mode": "range", "row_start": 1, "row_end": 4, "row_limit": 2}
    with pytest.raises(ValidationError):
        ClassifierTrainSpec.model_validate(p)


@pytest.mark.parametrize("version", [True, False, 1.0, "1"])
def test_schema_version_is_exact_integer(version: Any) -> None:
    p = payload()
    p["schema_version"] = version
    with pytest.raises(ValidationError):
        ClassifierTrainSpec.model_validate(p)


def test_existing_evaluation_wire_digests_are_unchanged() -> None:
    from strata_forge.pipelines.evaluation_spec import EvalSpec

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/evaluation_conformance.json").read_text()
    )
    assert [EvalSpec.model_validate(value).digest() for value in fixture["valid_specs"]] == [
        "14d05b1fd110792ec391d7097c26a81560dfcffdf661ecf9efb1deab9961c40f",
        "3bde362b7b310db3ddd838f795479fd2daaf880c8b8c3cac5ef45c2a62c7e757",
        "d7d10ff09d9c531a71acae5c928f5d7c327789aa127bc08bf4cf5f99c2e7dfda",
    ]


def test_new_worker_modules_import_without_heavy_optional_dependencies() -> None:
    script = """
import builtins
native = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','transformers','numpy','trl','peft','datasets','huggingface_hub'}:
        raise RuntimeError('heavy import requested')
    return native(name,*args,**kwargs)
builtins.__import__ = guarded
import strata_forge.pipelines.classifier_spec
import strata_forge.pipelines.classifier_runner
import strata_forge.pipelines._scoring
"""
    result = subprocess.run(  # noqa: S603 — fixed interpreter and literal import-only script
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "src"),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        },
    )
    assert result.returncode == 0, result.stderr


def test_noncanonical_integer_index_key_refuses() -> None:
    p = payload()
    p["training"]["label_index_map"] = {"01": "E"}
    with pytest.raises(ValidationError):
        ClassifierTrainSpec.model_validate(p)
