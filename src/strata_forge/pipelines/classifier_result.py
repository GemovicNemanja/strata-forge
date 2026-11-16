"""Bounded content-free local classifier result shared by installed worker and poller."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_CLASSIFIER_RESULT_BYTES = 8192


class ClassifierLocalResult(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        hide_input_in_errors=True,
        revalidate_instances="always",
    )
    version: Literal[1]
    profile: Literal["primary_classifier_hf_worker_v1"]
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    spec_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    engine_version: str = Field(min_length=1, max_length=160)
    artifact_dir: Literal["training/artifact"]
    receipt_file: Literal["training-receipt.json"]
    receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    steps: int = Field(ge=1, le=2**53 - 1)
    trained_rows: int = Field(ge=1, le=100_000)
    overflow_rows: int = Field(ge=0, le=100_000)
    publication: Literal["not_supported"]
    resume: Literal["not_supported"]

    @field_validator("version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("classifier_result_invalid")
        return value

    def canonical(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError("classifier_result_invalid")
        result[key] = value
    return result


def _constant(_: str) -> object:
    raise ValueError("classifier_result_invalid")


def load_classifier_local_result(raw: bytes | str) -> ClassifierLocalResult:
    """Decode the closed record; a Server still binds it to its admitted spec."""
    result: ClassifierLocalResult | None = None
    try:
        if type(raw) is str and len(raw) <= MAX_CLASSIFIER_RESULT_BYTES:
            raw = raw.encode("utf-8")
        if type(raw) is not bytes or not 0 < len(raw) <= MAX_CLASSIFIER_RESULT_BYTES:
            raise ValueError("classifier_result_invalid")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
        result = ClassifierLocalResult.model_validate(value)
    except ValueError, TypeError, RecursionError:
        result = None
    if result is None:
        raise ValueError("classifier_result_invalid")
    return result
