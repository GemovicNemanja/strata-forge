"""Task inventory boundaries are source declarations, not installed worker eligibility."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from strata_forge.pipelines.capabilities import (
    TaskCapabilityCatalog,
    TaskCapabilityError,
    canonical_task_catalog_bytes,
    task_capabilities,
    task_catalog_sha256,
)
from strata_forge.training.capabilities import (
    assess_training_capability,
    canonical_catalog_bytes,
    training_capabilities,
)

_ROOT = Path(__file__).parents[3]
_GOLDEN = Path(__file__).parent / "fixtures" / "task_capabilities_v1.json"
_SURFACES = {
    "batch_inference_v1": ("inert_worker", "inference_runner", "main"),
    "causal_finetune_v1": ("inert_worker", "finetune_runner", "main"),
    "classification_evaluation_v1": ("inert_worker", "eval_runner", "main"),
    "evaluation_population_v2": ("trusted_python", "eval_runner", "run_population_evaluation"),
    "generation_v2": ("schema_only", "generation_spec", None),
    "primary_classifier_hf_worker_v1": ("inert_worker", "classifier_worker", "main"),
    "primary_classifier_v1": ("trusted_python", "classifier_runner", "run_classifier_training"),
    "scoring_population_v2": ("trusted_python", "inference_runner", "execute_population_scoring"),
}


def test_catalog_matches_frozen_source_projection() -> None:
    catalog = task_capabilities()
    assert canonical_task_catalog_bytes(catalog) == _GOLDEN.read_bytes()
    assert task_catalog_sha256(catalog) == hashlib.sha256(_GOLDEN.read_bytes()).hexdigest()
    assert len(canonical_task_catalog_bytes(catalog)) < 32_768
    assert tuple(row.profile for row in catalog.profiles) == tuple(sorted(_SURFACES))


def test_training_projection_and_legacy_selector_decisions_are_unchanged() -> None:
    assert canonical_catalog_bytes(task_capabilities().training) == canonical_catalog_bytes(
        training_capabilities()
    )
    assert (
        assess_training_capability(
            {"method": "sft", "dataset_format": "text", "adapter": "none"}
        ).code
        == "declared"
    )
    assert (
        assess_training_capability({"method": "grpo", "dataset_format": "text"}).code
        == "method_disabled"
    )
    assert (
        assess_training_capability(
            {"method": "sft", "dataset_format": "text", "initialization": "from_scratch"}
        ).code
        == "initialization_outside_profile"
    )
    assert (
        assess_training_capability(
            {"method": "sft", "dataset_format": "text", "distribution": "fsdp"}
        ).code
        == "distribution_outside_profile"
    )
    assert (
        assess_training_capability(
            {"method": "seq_cls", "dataset_format": "classification_text", "task_type": "SEQ_CLS"}
        ).code
        == "task_outside_profile"
    )


def test_surfaces_correspond_to_actual_source_entrypoints_without_executing_them() -> None:
    for row in task_capabilities().profiles:
        surface, module, function = _SURFACES[row.profile]
        source = _ROOT / "src" / "strata_forge" / "pipelines" / (module + ".py")
        tree = ast.parse(source.read_text())
        functions = {node.name for node in tree.body if isinstance(node, ast.AsyncFunctionDef)}
        assert row.surface == surface
        if function is None:
            assert row.entrypoint is None
            assert row.status == "experimental"
            assert row.code == "no_worker_entrypoint"
            assert "main" not in functions
        else:
            assert function in functions
            assert row.entrypoint == f"strata_forge.pipelines.{module}:{function}"
            assert row.status == "supported"
            assert row.code == (
                "declared_worker" if surface == "inert_worker" else "trusted_python_only"
            )


def test_selectors_come_from_real_schema_literals() -> None:
    rows = {row.profile: row for row in task_capabilities().profiles}
    assert rows["batch_inference_v1"].selectors[0].path == "hyperparams.engine"
    assert rows["batch_inference_v1"].selectors[0].values == (
        "hf_classify",
        "hf_label_logits",
        "vllm_chat",
    )
    assert rows["classification_evaluation_v1"].selectors[0].path == "mode"
    assert rows["classification_evaluation_v1"].selectors[0].values == (
        "classify",
        "score_predictions",
    )
    assert rows["causal_finetune_v1"].selectors == ()


def test_installed_primary_hf_descriptor_preserves_the_separate_sdk_contract() -> None:
    rows = {row.profile: row for row in task_capabilities().profiles}
    worker = rows["primary_classifier_hf_worker_v1"]
    sdk = rows["primary_classifier_v1"]
    assert worker.task_family == sdk.task_family == "training"
    assert worker.spec_type == sdk.spec_type
    assert worker.spec_schema_sha256 == sdk.spec_schema_sha256
    assert worker.primitive_extras == ("finetuning", "storage")
    assert worker.selectors == ()
    assert sdk.surface == "trusted_python"
    assert sdk.entrypoint == "strata_forge.pipelines.classifier_runner:run_classifier_training"
    assert sdk.primitive_extras == ("finetuning",)


def test_no_profile_claims_runtime_availability_or_complete_installation() -> None:
    for row in task_capabilities().profiles:
        assert row.installed == row.environment == row.hardware == "unknown"
        assert row.dependency_profile == "unknown"
        assert row.admitted == "not_assessed"


@pytest.mark.parametrize("field", ["installed", "environment", "hardware"])
def test_unobserved_runtime_state_is_a_closed_literal(field: str) -> None:
    catalog = task_capabilities()
    row = catalog.profiles[0].model_copy(update={field: "available"})
    changed = catalog.model_copy(update={"profiles": (row, *catalog.profiles[1:])})
    with pytest.raises(TaskCapabilityError, match=r"^task_capability_catalog_invalid$"):
        canonical_task_catalog_bytes(changed)


@pytest.mark.parametrize(
    "update",
    [
        {"status": "supported"},
        {"entrypoint": "strata_forge.pipelines.generation_spec:main"},
        {"surface": "inert_worker"},
    ],
)
def test_schema_only_cannot_be_relabelled_as_a_supported_worker(update: dict[str, str]) -> None:
    catalog = task_capabilities()
    rows = tuple(
        row.model_copy(update=update) if row.profile == "generation_v2" else row
        for row in catalog.profiles
    )
    changed = catalog.model_copy(update={"profiles": rows})
    with pytest.raises(TaskCapabilityError, match=r"^task_capability_catalog_invalid$"):
        canonical_task_catalog_bytes(changed)


def test_nested_training_copy_cannot_bypass_revalidation() -> None:
    catalog = task_capabilities()
    training = catalog.training.model_copy(update={"installed": "available"})
    with pytest.raises(TaskCapabilityError, match=r"^task_capability_catalog_invalid$"):
        canonical_task_catalog_bytes(catalog.model_copy(update={"training": training}))


def test_mutating_detached_schema_or_copy_cannot_change_source_catalog() -> None:
    catalog = task_capabilities()
    original = canonical_task_catalog_bytes(catalog)
    schema = TaskCapabilityCatalog.model_json_schema()
    schema.clear()
    copy = catalog.model_copy(update={"engine_version": "999.0.0"})
    assert canonical_task_catalog_bytes(copy) != original
    assert canonical_task_catalog_bytes(task_capabilities()) == original
    with pytest.raises(PydanticValidationError):
        catalog.profiles[0].status = "experimental"


def test_duplicate_or_unsorted_profile_rows_refuse() -> None:
    catalog = task_capabilities()
    for rows in ((catalog.profiles[0], catalog.profiles[0]), tuple(reversed(catalog.profiles))):
        with pytest.raises(TaskCapabilityError, match=r"^task_capability_catalog_invalid$"):
            canonical_task_catalog_bytes(catalog.model_copy(update={"profiles": rows}))


def test_catalog_schema_identity_is_independently_reproducible() -> None:
    schema = TaskCapabilityCatalog.model_json_schema()
    raw = json.dumps(
        schema, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    assert task_capabilities().schema_sha256 == hashlib.sha256(raw).hexdigest()


def test_cold_import_and_inventory_do_not_import_ml_extras() -> None:
    script = r"""
import importlib.abc
import json
import sys
blocked = {"torch", "transformers", "trl", "peft", "datasets", "pyarrow", "numpy", "vllm", "sky", "asyncssh", "langfuse"}
worker = "strata_forge.pipelines.classifier_worker"
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == worker or fullname.split(".")[0] in blocked:
            raise AssertionError("unexpected_optional_import")
sys.meta_path.insert(0, Refuse())
from strata_forge.pipelines.capabilities import canonical_task_catalog_bytes, task_capabilities
value = task_capabilities()
assert len(value.profiles) == 8
assert not blocked.intersection(sys.modules)
assert worker not in sys.modules
print(json.dumps({"profiles": len(value.profiles), "bytes": len(canonical_task_catalog_bytes(value))}))
"""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "LANG", "SYSTEMROOT"}
    }
    environment["PYTHONPATH"] = str(_ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    result = subprocess.run(  # noqa: S603 — fixed trusted script and selected current interpreter
        [sys.executable, "-B", "-c", script],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["profiles"] == 8
