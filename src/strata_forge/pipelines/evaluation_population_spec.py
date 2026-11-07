"""Versioned population execution around the unchanged authoritative evaluation task schema."""

from __future__ import annotations

from typing import Annotated, Literal, cast

from pydantic import BeforeValidator, Field, field_validator, model_serializer, model_validator

from strata_forge.pipelines._population import (
    EvaluationPopulation,
    PopulationData,
    canonical_digest,
)
from strata_forge.pipelines.evaluation_spec import EvalSpec


def _tuple(value: object) -> object:
    return tuple(cast("list[object]", value)) if type(value) is list else value


class EvaluationPopulationSpecV2(PopulationData):
    schema_version: Literal[2]
    protocol: Literal["strata.eval.population.v2"]
    overflow: Literal["drop"]
    job: EvalSpec
    populations: Annotated[
        tuple[EvaluationPopulation, ...],
        BeforeValidator(_tuple),
        Field(min_length=1, max_length=64),
    ]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 2:
            raise ValueError("population_schema_version_invalid")
        return value

    @field_validator("job", mode="before")
    @classmethod
    def _job(cls, value: object) -> EvalSpec:
        if type(value) is EvalSpec:
            return EvalSpec.model_validate(value.wire_payload())
        if type(value) is not dict:
            raise ValueError("population_job_invalid")
        return EvalSpec.model_validate(value)

    @model_validator(mode="after")
    def _profile(self) -> EvaluationPopulationSpecV2:
        if self.job.output_repo_id is not None or any(
            case.truncation_policy != "count_as_invalid" for case in self.job.evaluation_sets
        ):
            raise ValueError("population_job_invalid")
        names = [row.evaluation_id for row in self.populations]
        if len(set(names)) != len(names) or set(names) != {
            case.evaluation_id for case in self.job.evaluation_sets
        }:
            raise ValueError("population_set_invalid")
        return self

    @model_serializer
    def _wire(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "overflow": self.overflow,
            "job": self.job.wire_payload(),
            "populations": [row.model_dump(mode="json") for row in self.populations],
        }

    def wire_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    def digest(self) -> str:
        return canonical_digest(self.wire_payload())

    def input_reads(self) -> tuple[tuple[str, Literal["model", "dataset"]], ...]:
        return self.job.input_reads()
