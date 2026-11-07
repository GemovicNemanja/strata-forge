"""A distinct operational envelope; legacy inference fields retain their presence semantics."""

from __future__ import annotations

import re
from typing import Literal, cast

from pydantic import field_validator, model_serializer

from strata_forge.pipelines._hub_ids import validate_hub_ref
from strata_forge.pipelines._population import PopulationColumns, PopulationData, canonical_digest
from strata_forge.pipelines.evaluation_spec import (
    Selection,
)
from strata_forge.pipelines.inference_runner import RunSpec

_HF_FIELDS = {
    "engine",
    "max_model_len",
    "dtype",
    "pair_template",
    "label_alias",
    "label_tokens",
    "scoring_batch_size",
}


def _inference_job(value: object) -> RunSpec:
    data = (
        cast("dict[str, object]", value.model_dump(mode="python", exclude_unset=True))
        if type(value) is RunSpec
        else value
    )
    if type(data) is not dict:
        raise ValueError("population_job_invalid")
    fields = cast("dict[str, object]", data)
    hp = fields.get("hyperparams")
    if type(hp) is not dict:
        raise ValueError("population_job_invalid")
    knobs = cast("dict[str, object]", hp)
    if (
        not set(knobs) <= _HF_FIELDS
        or type(knobs.get("engine")) is not str
        or knobs.get("engine")
        not in {
            "hf_classify",
            "hf_label_logits",
        }
    ):
        raise ValueError("population_job_invalid")
    for key in ("max_model_len", "scoring_batch_size"):
        number = knobs.get(key)
        if type(number) is not int or number < 1:
            raise ValueError("population_job_invalid")
    for key in (
        "model_id",
        "model_revision",
        "dataset_id",
        "dataset_commit_sha",
        "dataset_config",
        "id_column",
        "run_id",
        "engine_version",
        "split",
        "template",
    ):
        if type(fields.get(key)) is not str:
            raise ValueError("population_job_invalid")
    for key in ("dataset_config", "id_column", "split", "engine_version"):
        if not cast("str", fields[key]):
            raise ValueError("population_job_invalid")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", cast("str", fields["run_id"])) is None:
        raise ValueError("population_job_invalid")
    validate_hub_ref(f"{fields['model_id']}@{fields['model_revision']}")
    validate_hub_ref(f"{fields['dataset_id']}@{fields['dataset_commit_sha']}")
    path = fields.get("progress_path")
    if fields.get("output_repo_id") is not None or (
        path is not None and (type(path) is not str or path != "progress.jsonl")
    ):
        raise ValueError("population_job_invalid")
    for name in ("label_alias", "label_tokens"):
        mapping = knobs.get(name)
        if mapping is not None and (
            type(mapping) is not dict
            or any(
                type(key) is not str or type(child) is not str
                for key, child in cast("dict[object, object]", mapping).items()
            )
        ):
            raise ValueError("population_job_invalid")
    # Reconstruct, rather than retain a mutable legacy instance supplied by the caller.
    return RunSpec.model_validate(fields)


class InferencePopulationSpecV2(PopulationData):
    schema_version: Literal[2]
    protocol: Literal["strata.scoring.population.v2"]
    overflow: Literal["drop"]
    job: RunSpec
    selection: Selection
    population: PopulationColumns

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 2:
            raise ValueError("population_schema_version_invalid")
        return value

    @field_validator("job", mode="before")
    @classmethod
    def _job(cls, value: object) -> RunSpec:
        return _inference_job(value)

    @field_validator("selection", mode="before")
    @classmethod
    def _selection(cls, value: object) -> Selection:
        if type(value) is Selection:
            return Selection.model_validate(value.model_dump(mode="python"))
        if type(value) is not dict:
            raise ValueError("population_selection_invalid")
        return Selection.model_validate(value)

    @model_serializer
    def _wire(self) -> dict[str, object]:
        # Dumping all nested defaults would turn absent generation knobs into present-null
        # fields that this profile correctly refuses. Presence is part of the contract.
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "overflow": self.overflow,
            "job": self.job.model_dump(mode="json", exclude_unset=True),
            "selection": self.selection.model_dump(mode="json"),
            "population": self.population.model_dump(mode="json"),
        }

    def wire_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    def digest(self) -> str:
        return canonical_digest(self.wire_payload())

    def input_reads(self) -> tuple[tuple[str, Literal["model", "dataset"]], ...]:
        return (
            (f"{self.job.model_id}@{self.job.model_revision}", "model"),
            (f"{self.job.dataset_id}@{self.job.dataset_commit_sha}", "dataset"),
        )
