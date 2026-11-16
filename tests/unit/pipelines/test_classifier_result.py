"""Closed metadata-only worker records are not arbitrary artifact or remote path readers."""

from __future__ import annotations

import json
from typing import Any

import pytest

from strata_forge.pipelines.classifier_result import (
    MAX_CLASSIFIER_RESULT_BYTES,
    ClassifierLocalResult,
    load_classifier_local_result,
)

# Invalid wire fixtures deliberately include values outside the production schema.


def result(**changes: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "profile": "primary_classifier_hf_worker_v1",
        "run_id": "1" * 32,
        "spec_sha256": "2" * 64,
        "engine_version": "0.3.0",
        "artifact_dir": "training/artifact",
        "receipt_file": "training-receipt.json",
        "receipt_sha256": "3" * 64,
        "steps": 1,
        "trained_rows": 2,
        "overflow_rows": 0,
        "publication": "not_supported",
        "resume": "not_supported",
        **changes,
    }


def test_canonical_bytes_and_text_decode_to_identical_closed_record() -> None:
    value = ClassifierLocalResult.model_validate(result())
    assert load_classifier_local_result(value.canonical()) == value
    assert load_classifier_local_result(value.canonical().decode()) == value
    assert len(value.canonical()) <= MAX_CLASSIFIER_RESULT_BYTES


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"profile": "caller_selected_worker"},
        {"run_id": "../path"},
        {"spec_sha256": "not-a-digest"},
        {"engine_version": "x" * 161},
        {"artifact_dir": "/home/worker/private"},
        {"artifact_dir": "training/../artifact"},
        {"receipt_file": "other.json"},
        {"receipt_sha256": "not-a-digest"},
        {"steps": True},
        {"steps": 0},
        {"steps": 2**53},
        {"trained_rows": 0},
        {"trained_rows": 100_001},
        {"overflow_rows": -1},
        {"overflow_rows": 100_001},
        {"publication": "published"},
        {"resume": "supported"},
        {"rows": ["fictional private canary"]},
        {"worker_credential": "fictional private canary"},
        {"attempt_id": "4" * 32},
    ],
)
def test_result_rejects_paths_private_fields_false_success_and_bad_counts(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match=r"^classifier_result_invalid$") as error:
        load_classifier_local_result(json.dumps(result(**changes)))
    assert error.value.__context__ is None
    assert "private canary" not in str(error.value)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"[]",
        b"\xff",
        b'{"version":1,"version":1}',
        b'{"steps":NaN}',
        b"[" * 2000 + b"]" * 2000,
        b"x" * (MAX_CLASSIFIER_RESULT_BYTES + 1),
        "\ud800",
    ],
)
def test_duplicate_nonfinite_oversize_or_invalid_json_has_fixed_refusal(raw: bytes | str) -> None:
    with pytest.raises(ValueError, match=r"^classifier_result_invalid$") as error:
        load_classifier_local_result(raw)
    assert error.value.__context__ is None
