"""Immutable declaration metadata, separate from training admission and runtime availability."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from strata_forge.training.dataset_format import FORMATS
from strata_forge.training.methods import METHODS

__all__ = [
    "TrainingCapabilityCatalog",
    "TrainingCapabilityDecision",
    "TrainingCapabilityRequest",
    "assess_training_capability",
    "canonical_catalog_bytes",
    "catalog_sha256",
    "training_capabilities",
]

type _Name = Annotated[str, Field(min_length=1, max_length=64)]
type _Code = Literal[
    "declared",
    "invalid_request",
    "task_outside_profile",
    "initialization_outside_profile",
    "distribution_outside_profile",
    "unknown_method",
    "method_disabled",
    "format_unsupported",
    "adapter_unsupported",
    "merge_requires_adapter",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class _Format(_Frozen):
    name: _Name
    required_roles: Annotated[tuple[_Name, ...], Field(max_length=16)]
    optional_roles: Annotated[tuple[_Name, ...], Field(max_length=16)]


class _Method(_Frozen):
    name: _Name
    label: Annotated[str, Field(min_length=1, max_length=160)]
    task_type: _Name
    formats: Annotated[tuple[_Name, ...], Field(max_length=32)]
    tuning_knobs: Annotated[tuple[_Name, ...], Field(max_length=64)]
    status: Literal["declared", "refused"]
    code: Literal["declared", "method_disabled", "task_outside_profile"]
    disabled_reason: Annotated[str, Field(max_length=1024)]


class TrainingCapabilityCatalog(_Frozen):
    """A detached registry snapshot; a declared cell is not permission to launch."""

    schema_version: Literal[1] = 1
    profile: Literal["inert_finetune_declaration"] = "inert_finetune_declaration"
    schema_sha256: Annotated[str, Field(pattern="^[0-9a-f]{64}$")]
    methods: Annotated[tuple[_Method, ...], Field(max_length=64)]
    formats: Annotated[tuple[_Format, ...], Field(max_length=32)]
    adapters: tuple[Literal["none"], Literal["lora"], Literal["qlora"]] = (
        "none",
        "lora",
        "qlora",
    )
    initialization: Literal["pretrained"] = "pretrained"
    task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"
    distribution: Literal["unspecified"] = "unspecified"
    required_extras: tuple[Literal["finetuning"]] = ("finetuning",)
    installed: Literal["unknown"] = "unknown"
    environment: Literal["unknown"] = "unknown"
    hardware: Literal["unknown"] = "unknown"


class TrainingCapabilityRequest(_Frozen):
    """Bounded selectors for the existing inert declaration path, without execution inputs."""

    method: _Name
    dataset_format: _Name
    adapter: _Name = "lora"
    merge_adapter: bool = False
    task_type: _Name = "CAUSAL_LM"
    initialization: _Name = "pretrained"
    distribution: _Name = "unspecified"


class TrainingCapabilityDecision(_Frozen):
    """Fixed, content-free classification that never asserts runtime eligibility."""

    schema_version: Literal[1] = 1
    profile: Literal["inert_finetune_declaration"] = "inert_finetune_declaration"
    status: Literal["declared", "refused"]
    code: _Code
    installed: Literal["unknown"] = "unknown"
    environment: Literal["unknown"] = "unknown"
    hardware: Literal["unknown"] = "unknown"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _schema_sha256() -> str:
    schemas = {
        "catalog": TrainingCapabilityCatalog.model_json_schema(),
        "request": TrainingCapabilityRequest.model_json_schema(),
        "decision": TrainingCapabilityDecision.model_json_schema(),
    }
    return hashlib.sha256(_canonical(schemas)).hexdigest()


def training_capabilities() -> TrainingCapabilityCatalog:
    """Snapshot existing method and role declarations without inspecting installed extras."""
    methods = tuple(
        _Method(
            name=spec.name,
            label=spec.label,
            task_type=spec.task_type,
            formats=tuple(sorted(spec.formats)),
            tuning_knobs=tuple(spec.tuning_knobs),
            status="declared" if spec.enabled and spec.task_type == "CAUSAL_LM" else "refused",
            code=(
                "method_disabled"
                if not spec.enabled
                else "task_outside_profile"
                if spec.task_type != "CAUSAL_LM"
                else "declared"
            ),
            disabled_reason=spec.disabled_reason,
        )
        for spec in sorted(METHODS.values(), key=lambda item: item.name)
    )
    formats = tuple(
        _Format(
            name=spec.name,
            required_roles=tuple(spec.required),
            optional_roles=tuple(spec.optional),
        )
        for spec in sorted(FORMATS.values(), key=lambda item: item.name)
    )
    return TrainingCapabilityCatalog(
        schema_sha256=_schema_sha256(), methods=methods, formats=formats
    )


def canonical_catalog_bytes(catalog: TrainingCapabilityCatalog) -> bytes:
    """Encode every catalog field as sorted-key compact UTF-8 JSON, without a trailing newline."""
    return _canonical(catalog.model_dump(mode="json"))


def catalog_sha256(catalog: TrainingCapabilityCatalog) -> str:
    """Hash the complete canonical body, including its schema identity."""
    return hashlib.sha256(canonical_catalog_bytes(catalog)).hexdigest()


def _decision(code: _Code) -> TrainingCapabilityDecision:
    return TrainingCapabilityDecision(
        status="declared" if code == "declared" else "refused", code=code
    )


def assess_training_capability(request: object) -> TrainingCapabilityDecision:
    """Classify selectors without raising validation diagnostics or echoing rejected values."""
    if isinstance(request, TrainingCapabilityRequest):
        request = dict(vars(request))
    try:
        parsed = TrainingCapabilityRequest.model_validate(request)
    except ValidationError:
        return _decision("invalid_request")
    if parsed.task_type != "CAUSAL_LM":
        return _decision("task_outside_profile")
    if parsed.initialization != "pretrained":
        return _decision("initialization_outside_profile")
    if parsed.distribution != "unspecified":
        return _decision("distribution_outside_profile")
    spec = METHODS.get(parsed.method)
    if spec is None:
        return _decision("unknown_method")
    if not spec.enabled:
        return _decision("method_disabled")
    if spec.task_type != parsed.task_type:
        return _decision("task_outside_profile")
    if parsed.dataset_format not in spec.formats or parsed.dataset_format not in FORMATS:
        return _decision("format_unsupported")
    if parsed.adapter not in ("none", "lora", "qlora"):
        return _decision("adapter_unsupported")
    if parsed.merge_adapter and parsed.adapter == "none":
        return _decision("merge_requires_adapter")
    return _decision("declared")
