"""Bounded distribution metadata observations, separate from worker eligibility."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
from datetime import UTC, datetime
from importlib import metadata
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_OBSERVATION_BYTES = 16 * 1024
DISTRIBUTIONS = (
    "accelerate",
    "anthropic",
    "anyio",
    "asyncssh",
    "bitsandbytes",
    "datasets",
    "duckdb",
    "litellm",
    "numpy",
    "openai",
    "peft",
    "pyarrow",
    "pydantic",
    "pydantic-settings",
    "strata-forge",
    "torch",
    "transformers",
    "trl",
    "vllm",
)
_VERSION = re.compile(r"[0-9][A-Za-z0-9.!+_-]{0,127}")
_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z")


class _Frozen(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
        revalidate_instances="always",
    )


class _Interpreter(_Frozen):
    implementation: Literal["cpython", "pypy", "other"]
    python_version: Annotated[str, Field(min_length=5, max_length=32)]
    platform: Literal["darwin", "linux", "win32", "other"]
    pointer_bits: Literal[32, 64]

    @field_validator("python_version")
    @classmethod
    def _version(cls, value: str) -> str:
        if re.fullmatch(r"[0-9]{1,4}\.[0-9]{1,4}\.[0-9]{1,4}", value) is None:
            raise ValueError("runtime_environment_invalid")
        return value

    @field_validator("pointer_bits", mode="before")
    @classmethod
    def _bits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("runtime_environment_invalid")
        return value


class _Distribution(_Frozen):
    name: Annotated[str, Field(min_length=1, max_length=64)]
    status: Literal["present", "absent", "unavailable"]
    version: Annotated[str, Field(min_length=1, max_length=128)] | None
    reason: Literal["observed", "metadata_not_found", "metadata_invalid", "metadata_unavailable"]

    @model_validator(mode="after")
    def _observation(self) -> _Distribution:
        if self.name not in DISTRIBUTIONS:
            raise ValueError("runtime_environment_invalid")
        if self.status == "present":
            valid = (
                self.reason == "observed"
                and self.version is not None
                and _VERSION.fullmatch(self.version) is not None
            )
        elif self.status == "absent":
            valid = self.version is None and self.reason == "metadata_not_found"
        else:
            valid = self.version is None and self.reason in {
                "metadata_invalid",
                "metadata_unavailable",
            }
        if not valid:
            raise ValueError("runtime_environment_invalid")
        return self


class RuntimeEnvironmentObservation(_Frozen):
    """A detached observation of metadata; no field asserts installed runtime usability."""

    schema_version: Literal[1] = 1
    profile: Literal["distribution_metadata_v1"] = "distribution_metadata_v1"
    observed_at: Annotated[str, Field(min_length=27, max_length=27)]
    interpreter: _Interpreter
    distributions: Annotated[tuple[_Distribution, ...], Field(min_length=19, max_length=19)]
    source_revision: Literal["not_observed"] = "not_observed"
    image_digest: Literal["not_observed"] = "not_observed"
    lock_digest: Literal["not_observed"] = "not_observed"
    hardware: Literal["not_observed"] = "not_observed"
    importability: Literal["not_observed"] = "not_observed"
    execution: Literal["not_assessed"] = "not_assessed"

    @field_validator("schema_version", mode="before")
    @classmethod
    def _schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("runtime_environment_invalid")
        return value

    @field_validator("observed_at")
    @classmethod
    def _time(cls, value: str) -> str:
        if _TIME.fullmatch(value) is None:
            raise ValueError("runtime_environment_invalid")
        datetime.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _complete_profile(self) -> RuntimeEnvironmentObservation:
        if tuple(item.name for item in self.distributions) != DISTRIBUTIONS:
            raise ValueError("runtime_environment_invalid")
        return self


def _distribution(name: str) -> _Distribution:
    try:
        value = metadata.version(name)
    except metadata.PackageNotFoundError:
        return _Distribution(name=name, status="absent", version=None, reason="metadata_not_found")
    except Exception:
        return _Distribution(
            name=name, status="unavailable", version=None, reason="metadata_unavailable"
        )
    if type(value) is not str or _VERSION.fullmatch(value) is None:
        return _Distribution(
            name=name, status="unavailable", version=None, reason="metadata_invalid"
        )
    return _Distribution(name=name, status="present", version=value, reason="observed")


def capture_runtime_environment() -> RuntimeEnvironmentObservation:
    """Read fixed distribution version metadata, without importing or probing its packages."""
    implementation = sys.implementation.name
    platform = sys.platform
    return RuntimeEnvironmentObservation(
        observed_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        interpreter=_Interpreter.model_validate(
            {
                "implementation": implementation
                if implementation in {"cpython", "pypy"}
                else "other",
                "python_version": ".".join(str(value) for value in sys.version_info[:3]),
                "platform": platform if platform in {"darwin", "linux", "win32"} else "other",
                "pointer_bits": struct.calcsize("P") * 8,
            },
            strict=True,
        ),
        distributions=tuple(_distribution(name) for name in DISTRIBUTIONS),
    )


def canonical_environment_bytes(value: RuntimeEnvironmentObservation) -> bytes:
    """Revalidate all copied observations and encode bounded canonical UTF-8 metadata."""
    raw = None
    try:
        if type(value) is RuntimeEnvironmentObservation:
            current = RuntimeEnvironmentObservation.model_validate(value, strict=True)
            encoded = json.dumps(
                current.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(encoded) <= MAX_OBSERVATION_BYTES:
                raw = encoded
    except Exception:
        raw = None
    if raw is None:
        raise ValueError("runtime_environment_invalid") from None
    return raw


def environment_observation_sha256(value: RuntimeEnvironmentObservation) -> str:
    """Identify exact observation bytes, including capture time; this is not an image pin."""
    return hashlib.sha256(canonical_environment_bytes(value)).hexdigest()
