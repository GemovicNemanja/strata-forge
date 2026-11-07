"""Separate exact-pin generation data; no legacy spec, worker or capability activation."""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from strata_forge.llm._generation_schema import GenerationError, parse_json, scalar_text
from strata_forge.llm.generation import (
    MAX_GENERATION_SPEC_BYTES,
    GenerationControls,
    GenerationData,
    GenerationProfile,
    GenerationRequest,
)
from strata_forge.pipelines.evaluation_spec import (  # noqa: TC001 — Pydantic validates these concrete reused types.
    Ref,
    Selection,
)

_PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_]+)\}")


class GenerationServing(GenerationData):
    max_model_len: int = Field(ge=1, le=1_048_576)
    dtype: Literal["auto", "float32", "float16", "bfloat16"]
    tensor_parallel_size: int = Field(ge=1, le=64)
    concurrency: int = Field(ge=1, le=64)
    deadline_seconds: float = Field(gt=0, le=300, allow_inf_nan=False)


class LocalGenerationOutput(GenerationData):
    kind: Literal["local"]


class GenerationRunSpecV2(GenerationData):
    """Finite source membership and explicit controls, without endpoint or authorization data."""

    schema_version: Literal[2]
    protocol: Literal["strata.generation.v2"]
    forge_engine_version: str = Field(min_length=1, max_length=256)
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
    model_ref: Ref
    dataset_ref: Ref
    dataset_config: str = Field(min_length=1, max_length=120)
    split: str = Field(min_length=1, max_length=120)
    id_column: str = Field(min_length=1, max_length=255)
    selection: Selection
    template: str = Field(max_length=200_000)
    column_mapping: dict[str, str]
    profile: GenerationProfile
    serving: GenerationServing
    generation: GenerationControls
    overflow: Literal["error"]
    output: LocalGenerationOutput

    @field_validator("forge_engine_version", "dataset_config", "split", "id_column", "template")
    @classmethod
    def _text(cls, value: str) -> str:
        if not scalar_text(value):
            raise ValueError("generation_text_invalid")
        return value

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("generation_version_invalid")
        return value

    @field_validator("column_mapping")
    @classmethod
    def _mapping(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 64 or any(
            not re.fullmatch(r"[A-Za-z0-9_]+", key)
            or not scalar_text(column)
            or not column
            or len(column) > 255
            for key, column in value.items()
        ):
            raise ValueError("generation_mapping_invalid")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if any(
            match.group(1) not in self.column_mapping
            for match in _PLACEHOLDER.finditer(self.template)
        ):
            raise ValueError("generation_mapping_uncovered")
        # Reuse the single-request profile checks; validation does not establish native support.
        raw = self.generation.wire_payload()
        raw.pop("samples")
        raw.pop("seed")
        GenerationRequest.model_validate(
            {
                **raw,
                "model_ref": self.model_ref,
                "profile": self.profile,
                "effective_seed_decimal": None,
                "deadline_seconds": self.serving.deadline_seconds,
            }
        )
        if self.serving.max_model_len < self.generation.max_tokens:
            raise ValueError("generation_context_budget_invalid")
        return self

    def input_reads(self) -> tuple[tuple[str, Literal["model", "dataset"]], ...]:
        return ((self.model_ref, "model"), (self.dataset_ref, "dataset"))


def parse_generation_spec(raw: str | bytes) -> GenerationRunSpecV2:
    """Decode the bounded wire with duplicate/scalar checks before typed validation."""
    failed = False
    value: GenerationRunSpecV2 | None = None
    try:
        value = GenerationRunSpecV2.model_validate(
            parse_json(raw, max_bytes=MAX_GENERATION_SPEC_BYTES)
        )
    except ValueError, GenerationError:
        failed = True
    if failed or value is None:
        raise GenerationError("generation_spec_invalid")
    return value


def revalidate_generation_spec(value: GenerationRunSpecV2) -> GenerationRunSpecV2:
    """Copy typed source data before future policy/transport awaits; no digest is caller supplied."""
    failed = False
    result: GenerationRunSpecV2 | None = None
    try:
        if type(value) is GenerationRunSpecV2:
            result = GenerationRunSpecV2.model_validate(value)
            result = parse_generation_spec(result.model_dump_json())
    except ValueError, GenerationError:
        failed = True
    if failed or result is None:
        raise GenerationError("generation_spec_invalid")
    return result
