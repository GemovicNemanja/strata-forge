"""Detached task-profile declarations, separate from installed execution and admission."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Self, cast, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strata_forge import __version__
from strata_forge.core.errors import ValidationError
from strata_forge.training.capabilities import (
    TrainingCapabilityCatalog,
    training_capabilities,
)

__all__ = [
    "TaskCapabilityCatalog",
    "TaskCapabilityError",
    "TaskProfileDescriptor",
    "canonical_task_catalog_bytes",
    "task_capabilities",
    "task_catalog_sha256",
]

type _Name = Annotated[str, Field(min_length=1, max_length=128)]
type _Sha = Annotated[str, Field(pattern="^[0-9a-f]{64}$")]
type _Surface = Literal["inert_worker", "trusted_python", "schema_only"]
type _Family = Literal["inference", "training", "evaluation", "generation"]


class TaskCapabilityError(ValidationError):
    """A fixed declaration-source or copied-catalog refusal."""


class _Frozen(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        hide_input_in_errors=True,
        revalidate_instances="always",
    )


class _Choices(_Frozen):
    path: _Name
    values: Annotated[tuple[_Name, ...], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        if tuple(sorted(set(self.values))) != self.values:
            raise ValueError("task_capability_choices_invalid")
        return self


class TaskProfileDescriptor(_Frozen):
    """Support describes only this declared surface, never target readiness or permission."""

    profile: _Name
    task_family: _Family
    surface: _Surface
    status: Literal["supported", "experimental"]
    code: Literal["declared_worker", "trusted_python_only", "no_worker_entrypoint"]
    spec_type: _Name
    spec_schema_sha256: _Sha
    entrypoint: _Name | None
    selectors: Annotated[tuple[_Choices, ...], Field(max_length=8)] = ()
    primitive_extras: Annotated[tuple[_Name, ...], Field(max_length=8)] = ()
    dependency_profile: Literal["unknown"] = "unknown"
    installed: Literal["unknown"] = "unknown"
    environment: Literal["unknown"] = "unknown"
    hardware: Literal["unknown"] = "unknown"
    admitted: Literal["not_assessed"] = "not_assessed"

    @model_validator(mode="after")
    def _boundary(self) -> Self:
        expected = {
            "inert_worker": ("supported", "declared_worker"),
            "trusted_python": ("supported", "trusted_python_only"),
            "schema_only": ("experimental", "no_worker_entrypoint"),
        }[self.surface]
        if (
            (self.status, self.code) != expected
            or (self.entrypoint is None) != (self.surface == "schema_only")
            or len({item.path for item in self.selectors}) != len(self.selectors)
            or tuple(sorted(set(self.primitive_extras))) != self.primitive_extras
        ):
            raise ValueError("task_capability_boundary_invalid")
        return self


class TaskCapabilityCatalog(_Frozen):
    """Source inventory composed from existing schemas and the unchanged training registry."""

    schema_version: Literal[1] = 1
    protocol: Literal["strata.task-capability-inventory.v1"] = "strata.task-capability-inventory.v1"
    engine_version: Annotated[str, Field(min_length=1, max_length=64)]
    schema_sha256: _Sha
    profiles: Annotated[tuple[TaskProfileDescriptor, ...], Field(min_length=1, max_length=32)]
    training: TrainingCapabilityCatalog

    @model_validator(mode="after")
    def _profiles(self) -> Self:
        names = tuple(profile.profile for profile in self.profiles)
        if tuple(sorted(set(names))) != names:
            raise ValueError("task_capability_profiles_invalid")
        return self


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _schema_sha256(model: type[BaseModel]) -> str:
    return hashlib.sha256(_canonical(model.model_json_schema())).hexdigest()


def _choices(model: type[BaseModel], field: str, path: str | None = None) -> _Choices:
    annotation = model.model_fields[field].annotation
    values = cast("tuple[object, ...]", get_args(annotation))
    if (
        get_origin(annotation) is not Literal
        or not values
        or any(type(value) is not str for value in values)
    ):
        raise TaskCapabilityError("task_capability_source_invalid")
    return _Choices(path=path or field, values=tuple(sorted(cast("tuple[str, ...]", values))))


def _profile(
    profile: str,
    family: _Family,
    model: type[BaseModel],
    surface: _Surface,
    entrypoint: str | None,
    extras: tuple[str, ...],
    selectors: tuple[_Choices, ...] = (),
) -> TaskProfileDescriptor:
    return TaskProfileDescriptor(
        profile=profile,
        task_family=family,
        surface=surface,
        status="experimental" if surface == "schema_only" else "supported",
        code=(
            "declared_worker"
            if surface == "inert_worker"
            else "trusted_python_only"
            if surface == "trusted_python"
            else "no_worker_entrypoint"
        ),
        spec_type=model.__module__ + "." + model.__qualname__,
        spec_schema_sha256=_schema_sha256(model),
        entrypoint=entrypoint,
        selectors=selectors,
        primitive_extras=extras,
    )


def task_capabilities() -> TaskCapabilityCatalog:
    """Describe source contracts without probing dependencies, targets, credentials or data."""
    from strata_forge.pipelines.classifier_spec import ClassifierTrainSpec
    from strata_forge.pipelines.evaluation_population_spec import EvaluationPopulationSpecV2
    from strata_forge.pipelines.evaluation_spec import EvalSpec
    from strata_forge.pipelines.finetune_runner import FinetuneSpec
    from strata_forge.pipelines.generation_spec import GenerationRunSpecV2
    from strata_forge.pipelines.inference_runner import Hyperparams, RunSpec
    from strata_forge.pipelines.scoring_population_spec import InferencePopulationSpecV2

    profiles = (
        _profile(
            "batch_inference_v1",
            "inference",
            RunSpec,
            "inert_worker",
            "strata_forge.pipelines.inference_runner:main",
            ("hf", "serving", "storage"),
            (_choices(Hyperparams, "engine", "hyperparams.engine"),),
        ),
        _profile(
            "causal_finetune_v1",
            "training",
            FinetuneSpec,
            "inert_worker",
            "strata_forge.pipelines.finetune_runner:main",
            ("finetuning", "storage"),
        ),
        _profile(
            "classification_evaluation_v1",
            "evaluation",
            EvalSpec,
            "inert_worker",
            "strata_forge.pipelines.eval_runner:main",
            ("hf", "scoring", "serving", "storage"),
            (_choices(EvalSpec, "mode"),),
        ),
        _profile(
            "evaluation_population_v2",
            "evaluation",
            EvaluationPopulationSpecV2,
            "trusted_python",
            "strata_forge.pipelines.eval_runner:run_population_evaluation",
            ("hf", "scoring", "serving", "storage"),
        ),
        _profile(
            "generation_v2",
            "generation",
            GenerationRunSpecV2,
            "schema_only",
            None,
            (),
        ),
        _profile(
            "primary_classifier_hf_worker_v1",
            "training",
            ClassifierTrainSpec,
            "inert_worker",
            "strata_forge.pipelines.classifier_worker:main",
            ("finetuning", "storage"),
        ),
        _profile(
            "primary_classifier_v1",
            "training",
            ClassifierTrainSpec,
            "trusted_python",
            "strata_forge.pipelines.classifier_runner:run_classifier_training",
            ("finetuning",),
        ),
        _profile(
            "scoring_population_v2",
            "inference",
            InferencePopulationSpecV2,
            "trusted_python",
            "strata_forge.pipelines.inference_runner:execute_population_scoring",
            ("hf", "scoring", "serving", "storage"),
        ),
    )
    return TaskCapabilityCatalog(
        engine_version=__version__,
        schema_sha256=_schema_sha256(TaskCapabilityCatalog),
        profiles=profiles,
        training=training_capabilities(),
    )


def canonical_task_catalog_bytes(catalog: TaskCapabilityCatalog) -> bytes:
    """Validate a detached value and encode compact sorted-key JSON without a newline."""
    from pydantic import ValidationError as PydanticValidationError

    try:
        checked = TaskCapabilityCatalog.model_validate(catalog.model_dump(mode="python"))
    except PydanticValidationError:
        raise TaskCapabilityError("task_capability_catalog_invalid") from None
    return _canonical(checked.model_dump(mode="json"))


def task_catalog_sha256(catalog: TaskCapabilityCatalog) -> str:
    """Bind every declared field without turning the identity into execution evidence."""
    return hashlib.sha256(canonical_task_catalog_bytes(catalog)).hexdigest()
