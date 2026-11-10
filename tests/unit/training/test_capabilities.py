"""Closed declaration metadata stays separate from runtime and launch authority."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from strata_forge.training.capabilities import (
    TrainingCapabilityCatalog,
    TrainingCapabilityDecision,
    TrainingCapabilityRequest,
    assess_training_capability,
    canonical_catalog_bytes,
    catalog_sha256,
    training_capabilities,
)
from strata_forge.training.dataset_format import FORMATS
from strata_forge.training.methods import METHODS

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

_ROOT = Path(__file__).resolve().parents[3]
_GOLDEN = Path(__file__).parent / "fixtures" / "training_capabilities_v1.json"
_FORMATS = ("text", "prompt_completion", "conversational", "preference", "unpaired_preference")
_SUPPORTED: dict[str, set[str]] = {
    "sft": {"text", "prompt_completion", "conversational"},
    "dpo": {"preference"},
    "orpo": {"preference"},
    "kto": {"unpaired_preference"},
    "grpo": set(),
}


@pytest.mark.parametrize("method", tuple(_SUPPORTED))
@pytest.mark.parametrize("dataset_format", _FORMATS)
@pytest.mark.parametrize("adapter", ["none", "lora", "qlora"])
@pytest.mark.parametrize("merge", [False, True])
def test_declared_matrix(method: str, dataset_format: str, adapter: str, merge: bool) -> None:
    decision = assess_training_capability(
        {
            "method": method,
            "dataset_format": dataset_format,
            "adapter": adapter,
            "merge_adapter": merge,
        }
    )
    expected = (
        "method_disabled"
        if method == "grpo"
        else "format_unsupported"
        if dataset_format not in _SUPPORTED[method]
        else "merge_requires_adapter"
        if adapter == "none" and merge
        else "declared"
    )
    assert decision.code == expected
    assert decision.status == ("declared" if expected == "declared" else "refused")
    assert (decision.installed, decision.environment, decision.hardware) == ("unknown",) * 3


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("task_type", "SEQ_CLS", "task_outside_profile"),
        ("task_type", "SEQ_2_SEQ_LM", "task_outside_profile"),
        ("task_type", "TOKEN_CLS", "task_outside_profile"),
        ("initialization", "scratch", "initialization_outside_profile"),
        ("initialization", "random", "initialization_outside_profile"),
        ("distribution", "single_gpu", "distribution_outside_profile"),
        ("distribution", "ddp", "distribution_outside_profile"),
        ("distribution", "fsdp", "distribution_outside_profile"),
        ("method", "ppo", "unknown_method"),
        ("dataset_format", "other", "format_unsupported"),
        ("adapter", "ia3", "adapter_unsupported"),
    ],
)
def test_unsupported_selectors_have_fixed_codes(field: str, value: str, code: str) -> None:
    request = {"method": "sft", "dataset_format": "text", field: value}
    decision = assess_training_capability(request)
    assert decision.code == code
    assert decision.status == "refused"
    assert value not in decision.model_dump_json()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "secret-not-a-request",
        [],
        {},
        {"method": "sft", "dataset_format": "text", "secret": "never-echo"},
        {"method": 1, "dataset_format": "text"},
        {"method": "", "dataset_format": "text"},
        {"method": "x" * 65, "dataset_format": "text"},
        {"method": "sft", "dataset_format": "x" * 65},
        {"method": "sft", "dataset_format": "text", "adapter": "x" * 65},
        {"method": "sft", "dataset_format": "text", "task_type": "x" * 65},
        {"method": "sft", "dataset_format": "text", "initialization": "x" * 65},
        {"method": "sft", "dataset_format": "text", "distribution": "x" * 65},
        {"method": "sft", "dataset_format": "text", "merge_adapter": "false"},
        {"method": "sft", "dataset_format": "text", "merge_adapter": 0},
    ],
)
def test_invalid_requests_return_only_fixed_metadata(payload: object) -> None:
    assert assess_training_capability(payload).model_dump() == {
        "schema_version": 1,
        "profile": "inert_finetune_declaration",
        "status": "refused",
        "code": "invalid_request",
        "installed": "unknown",
        "environment": "unknown",
        "hardware": "unknown",
    }


@pytest.mark.parametrize(
    "invalid_update",
    [
        {"merge_adapter": "false"},
        {"method": {"secret": ["never-print-this"]}},
        {"extra": "never-print-this"},
    ],
)
def test_copied_or_constructed_models_are_revalidated(invalid_update: dict[str, object]) -> None:
    request = TrainingCapabilityRequest(method="sft", dataset_format="text")
    assert assess_training_capability(request).code == "declared"
    assert (
        assess_training_capability(request.model_copy(update=invalid_update)).code
        == "invalid_request"
    )


def test_snapshot_roles_and_declined_metadata_match_registry() -> None:
    catalog = training_capabilities()
    assert [item.name for item in catalog.methods] == sorted(METHODS)
    assert [item.name for item in catalog.formats] == sorted(FORMATS)
    for item in catalog.methods:
        source = METHODS[item.name]
        assert item.formats == tuple(sorted(source.formats))
        assert item.tuning_knobs == source.tuning_knobs
        assert item.label == source.label
        assert item.disabled_reason == source.disabled_reason
    for item in catalog.formats:
        source = next(source for source in FORMATS.values() if source.name == item.name)
        assert item.required_roles == source.required
        assert item.optional_roles == source.optional
    grpo = next(item for item in catalog.methods if item.name == "grpo")
    assert (grpo.status, grpo.code) == ("refused", "method_disabled")
    assert "reward" in grpo.disabled_reason


def test_snapshot_and_decision_follow_actual_method_head(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setitem(METHODS, "sft", replace(METHODS["sft"], task_type="SEQ_CLS"))
    assert assess_training_capability({"method": "sft", "dataset_format": "text"}).code == (
        "task_outside_profile"
    )
    item = next(item for item in training_capabilities().methods if item.name == "sft")
    assert (item.status, item.code) == ("refused", "task_outside_profile")


def test_unknown_format_is_not_authorized_by_method_alone(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delitem(FORMATS, "text")
    assert assess_training_capability({"method": "sft", "dataset_format": "text"}).code == (
        "format_unsupported"
    )


def test_nested_snapshot_and_dump_cannot_mutate_registry(monkeypatch: MonkeyPatch) -> None:
    catalog = training_capabilities()
    original = canonical_catalog_bytes(catalog)
    method = catalog.methods[0]
    with pytest.raises(ValidationError):
        method.label = "mutated"
    with pytest.raises(ValidationError):
        catalog.hardware = "unknown"
    dumped = json.loads(original)
    dumped["methods"][0]["formats"].append("injected")
    dumped["formats"][0]["required_roles"].append("injected")
    assert "injected" not in METHODS[method.name].formats
    assert canonical_catalog_bytes(training_capabilities()) == original
    monkeypatch.setitem(METHODS, method.name, replace(METHODS[method.name], label="changed"))
    assert canonical_catalog_bytes(catalog) == original
    assert canonical_catalog_bytes(training_capabilities()) != original


def test_catalog_collection_bounds_are_enforced() -> None:
    catalog = training_capabilities()
    with pytest.raises(ValidationError):
        TrainingCapabilityCatalog.model_validate(
            {**catalog.model_dump(), "methods": catalog.methods * 13}
        )
    with pytest.raises(ValidationError):
        TrainingCapabilityCatalog.model_validate(
            {**catalog.model_dump(), "formats": catalog.formats * 7}
        )


def test_canonical_schema_body_and_digest_match_golden() -> None:
    catalog = training_capabilities()
    expected = json.loads(_GOLDEN.read_text())
    body = canonical_catalog_bytes(catalog)
    assert catalog.model_dump(mode="json") == expected["catalog"]
    schemas = {
        "catalog": TrainingCapabilityCatalog.model_json_schema(),
        "request": TrainingCapabilityRequest.model_json_schema(),
        "decision": TrainingCapabilityDecision.model_json_schema(),
    }
    assert schemas == expected["schemas"]
    assert (
        hashlib.sha256(
            json.dumps(
                schemas, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        == catalog.schema_sha256
    )
    assert (
        body
        == json.dumps(
            expected["catalog"],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    assert catalog_sha256(catalog) == expected["catalog_sha256"]
    assert hashlib.sha256(body).hexdigest() == expected["catalog_sha256"]
    assert TrainingCapabilityCatalog.model_validate_json(body) == catalog
    assert not body.endswith(b"\n")


def test_registry_order_does_not_change_canonical_identity(monkeypatch: MonkeyPatch) -> None:
    expected = canonical_catalog_bytes(training_capabilities())
    monkeypatch.setattr(
        "strata_forge.training.capabilities.METHODS", dict(reversed(METHODS.items()))
    )
    monkeypatch.setattr(
        "strata_forge.training.capabilities.FORMATS", dict(reversed(FORMATS.items()))
    )
    assert canonical_catalog_bytes(training_capabilities()) == expected


def test_declared_extra_exists_and_adapter_literals_match_legacy_spec() -> None:
    from strata_forge.pipelines.finetune_runner import FinetuneSpec

    packaging = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    extras = packaging["project"]["optional-dependencies"]
    catalog = training_capabilities()
    assert catalog.required_extras == ("finetuning",)
    assert all(name in extras for name in catalog.required_extras)
    assert any(requirement.startswith("bitsandbytes>=0.49") for requirement in extras["finetuning"])
    assert (
        tuple(FinetuneSpec.model_json_schema()["properties"]["adapter"]["enum"]) == catalog.adapters
    )
    assert catalog.initialization == "pretrained"
    assert catalog.distribution == "unspecified"
    assert catalog.installed == catalog.environment == catalog.hardware == "unknown"


def test_cold_import_with_denied_optional_modules_and_effects() -> None:
    script = r"""
import importlib.abc
import json
import sys
class Denied(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'transformers', 'trl', 'peft', 'datasets', 'accelerate', 'bitsandbytes',
            'numpy', 'pyarrow', 'asyncssh', 'skypilot',
        }:
            raise AssertionError('optional import: ' + fullname)
sys.meta_path.insert(0, Denied())
effects = []
def audit(event, args):
    if event.startswith(('socket.', 'subprocess.', 'os.system', 'os.exec', 'os.posix_spawn')):
        effects.append(event)
        raise AssertionError('effect: ' + event)
sys.addaudithook(audit)
from strata_forge.training.capabilities import training_capabilities, catalog_sha256, assess_training_capability
catalog = training_capabilities()
assert assess_training_capability({'method':'sft','dataset_format':'text','adapter':'qlora'}).code == 'declared'
assert catalog.installed == catalog.environment == catalog.hardware == 'unknown'
assert effects == [], effects
print(json.dumps({'methods':len(catalog.methods),'catalog_sha256':catalog_sha256(catalog)}))
"""
    env = {
        "HOME": str(Path.home()),
        "PATH": os.defpath,
        "PYTHONPATH": str(_ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHON_DOTENV_DISABLED": "1",
        "LITELLM_MODE": "PRODUCTION",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "OTEL_SDK_DISABLED": "true",
    }
    result = subprocess.run(  # noqa: S603 - fixed interpreter and literal script, no caller code
        [sys.executable, "-B", "-c", script],
        env=env,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "methods": 5,
        "catalog_sha256": catalog_sha256(training_capabilities()),
    }
