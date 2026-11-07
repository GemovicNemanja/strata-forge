"""Body-free local attempt metadata; content hashes confer no remote or provider authority."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.prompts.consumption import (
    LocalBuildReference,  # noqa: TC001 - Pydantic runtime schema
)
from strata_forge.prompts.portable import canonical_bytes, portable_sha256

MAX_ATTEMPTS = 64
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_ATTEMPT_METADATA_BYTES = 4096
MAX_INVOCATION_METADATA_BYTES = 256 * 1024
MAX_SAFE_INTEGER = 9_007_199_254_740_991
type LocalId = Annotated[str, Field(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$")]
type Digest = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")]
type Count = Annotated[int, Field(ge=0, le=MAX_SAFE_INTEGER)]
type Gap = Literal[
    "render_projection_unobserved",
    "provider_boundary_unobserved",
    "body_not_retained",
    "request_not_dispatched",
    "response_not_observed",
    "response_oversize",
    "late_return_discarded",
]
type ErrorCode = Literal[
    "capture_invalid",
    "capture_closed",
    "capture_in_progress",
    "capture_scope_invalid",
    "capture_limit",
    "local_transport_timeout",
    "local_transport_failed",
    "capture_response_invalid",
    "capture_response_limit",
    "local_cancelled",
    "local_interrupted",
]
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z\Z")


class _Metadata(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    @field_validator("version", mode="before", check_fields=False)
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("capture_invalid")
        return value

    @field_validator(
        "retained",
        "retrieved",
        "rendered",
        "request_captured",
        "response_observed",
        "closed",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def exact_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("capture_invalid")
        return value

    @field_validator("verified_at", "started_at", "ended_at", "retired_at", check_fields=False)
    @classmethod
    def exact_utc(cls, value: str | None) -> str | None:
        if value is not None:
            if _UTC.fullmatch(value) is None:
                raise ValueError("capture_invalid")
            datetime.fromisoformat(value)
        return value


class AttemptParents(_Metadata):
    """Declared linkage, validated against terminal records of the same live recorder."""

    retry_of: LocalId | None
    fallback_of: LocalId | None
    continuation_of: LocalId | None

    @model_validator(mode="after")
    def one_parent(self) -> Self:
        if (
            sum(
                value is not None
                for value in (self.retry_of, self.fallback_of, self.continuation_of)
            )
            > 1
        ):
            raise ValueError("capture_invalid")
        return self


class DeclaredBodyPin(_Metadata):
    """Caller baseline only; never a claim that request bytes contain a render."""

    sha256: Digest
    size_bytes: int = Field(ge=0, le=MAX_BODY_BYTES)


class ByteObservation(_Metadata):
    """Whole original bytes at this local boundary, distinct from retained/redacted content."""

    profile: Literal["opaque-application-bytes/v1"]
    coverage: Literal["complete", "omitted_oversize"]
    size_bytes: Count
    original_sha256: Digest | None
    redacted_sha256: None
    retained: Literal[False]

    @model_validator(mode="after")
    def exact_coverage(self) -> Self:
        complete = self.coverage == "complete"
        if complete != (self.original_sha256 is not None) or complete != (
            self.size_bytes <= MAX_BODY_BYTES
        ):
            raise ValueError("capture_invalid")
        return self


class AttemptFacts(_Metadata):
    """Accepted live observation facts, not remote retrieval or provider acceptance."""

    retrieved: Literal[True]
    rendered: Literal[True]
    request_captured: bool
    response_observed: bool


class AttemptMetadata(_Metadata):
    """A terminal local observation with explicit unavailable provider/projection facts."""

    domain: Literal["strata-prompt-attempt-v1"]
    version: Literal[1]
    invocation_id: LocalId
    attempt_id: LocalId
    source_kind: Literal["caller_bytes"]
    build_id: Digest
    render_id: Digest
    input_fingerprint: Digest
    parents: AttemptParents
    boundary: Literal["sdk_local_delegate_bytes/v1"]
    observation_source: Literal["instrumented_local"]
    started_at: str
    retired_at: str | None
    ended_at: str
    duration_us: Count
    facts: AttemptFacts
    request: ByteObservation | None
    response: ByteObservation | None
    delegate_status: int | None = Field(ge=0, le=999)
    declared_baseline: DeclaredBodyPin | None
    baseline_relation: Literal["not_supplied", "not_observed", "same_bytes", "changed_bytes"]
    render_relation: Literal["unknown"]
    provider: None
    model: None
    model_revision: None
    usage: None
    cost: None
    outcome: Literal[
        "observed_response", "transport_error", "capture_error", "cancelled", "interrupted"
    ]
    error_code: ErrorCode | None
    retention: Literal["metadata_only"]
    export_status: Literal["not_configured"]
    gaps: tuple[Gap, ...] = Field(max_length=7)

    @model_validator(mode="after")
    def consistent_observation(self) -> Self:
        if self.facts.request_captured != (
            self.request is not None
        ) or self.facts.response_observed != (self.response is not None):
            raise ValueError("capture_invalid")
        if self.request is not None and self.request.coverage != "complete":
            raise ValueError("capture_invalid")
        if self.response is not None and self.request is None:
            raise ValueError("capture_invalid")
        expected_relation = "not_supplied" if self.declared_baseline is None else "not_observed"
        if self.declared_baseline is not None and self.request is not None:
            expected_relation = (
                "same_bytes"
                if (self.declared_baseline.sha256, self.declared_baseline.size_bytes)
                == (self.request.original_sha256, self.request.size_bytes)
                else "changed_bytes"
            )
        if self.baseline_relation != expected_relation:
            raise ValueError("capture_invalid")
        required = {
            "render_projection_unobserved",
            "provider_boundary_unobserved",
            "body_not_retained",
        }
        if self.request is None:
            required.add("request_not_dispatched")
        if self.response is None:
            required.add("response_not_observed")
        if self.response is not None and self.response.coverage == "omitted_oversize":
            required.add("response_oversize")
        if len(set(self.gaps)) != len(self.gaps) or not required <= set(self.gaps):
            raise ValueError("capture_invalid")
        if self.outcome == "observed_response":
            valid = (
                self.response is not None
                and self.response.coverage == "complete"
                and self.error_code is None
                and self.retired_at is None
            )
        elif self.outcome == "capture_error":
            valid = (
                self.error_code == "capture_response_limit"
                and self.response is not None
                and self.response.coverage == "omitted_oversize"
            ) or (self.error_code == "capture_response_invalid" and self.response is None)
        else:
            allowed = {
                "transport_error": {"local_transport_timeout", "local_transport_failed"},
                "cancelled": {"local_cancelled"},
                "interrupted": {"local_interrupted"},
            }
            valid = self.response is None and self.error_code in allowed[self.outcome]
        if not valid or (self.outcome in {"cancelled", "interrupted"}) != (
            self.retired_at is not None
        ):
            raise ValueError("capture_invalid")
        if self.response is None and self.delegate_status is not None:
            raise ValueError("capture_invalid")
        if "late_return_discarded" in self.gaps and self.outcome not in {
            "cancelled",
            "interrupted",
        }:
            raise ValueError("capture_invalid")
        if len(self.canonical()) > MAX_ATTEMPT_METADATA_BYTES:
            raise ValueError("capture_limit")
        return self

    def canonical(self) -> bytes:
        """Encode only this body-free, domain-bearing portable metadata."""
        return canonical_bytes(self.model_dump(mode="json"))

    @property
    def metadata_sha256(self) -> str:
        return portable_sha256(self.model_dump(mode="json"))


class InvocationMetadata(_Metadata):
    """A settled local ledger snapshot, without raw prompt/request/response data."""

    domain: Literal["strata-prompt-invocation-v1"]
    version: Literal[1]
    invocation_id: LocalId
    verified_at: str
    source_kind: Literal["caller_bytes"]
    source_partition: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    source_reference: LocalBuildReference | None
    build_id: Digest
    build_size_bytes: int = Field(ge=1, le=8 * 1024 * 1024)
    render_id: Digest
    input_fingerprint: Digest
    output_kind: Literal["chat", "artifact"]
    output_count: int = Field(ge=1, le=128)
    source_span_count: int = Field(ge=0, le=32768)
    renderer_profile: Literal["strata-modular-compiler/1"]
    portable_profile: Literal["strata-prompt-json-v1"]
    retention: Literal["metadata_only"]
    attempts: tuple[AttemptMetadata, ...] = Field(max_length=MAX_ATTEMPTS)
    closed: bool
    export_status: Literal["not_configured"]

    @model_validator(mode="after")
    def exact_ledger(self) -> Self:
        seen: set[str] = set()
        for attempt in self.attempts:
            if (
                attempt.invocation_id,
                attempt.build_id,
                attempt.render_id,
                attempt.input_fingerprint,
            ) != (
                self.invocation_id,
                self.build_id,
                self.render_id,
                self.input_fingerprint,
            ) or attempt.attempt_id in seen:
                raise ValueError("capture_invalid")
            parents = (
                attempt.parents.retry_of,
                attempt.parents.fallback_of,
                attempt.parents.continuation_of,
            )
            if any(parent is not None and parent not in seen for parent in parents):
                raise ValueError("capture_invalid")
            seen.add(attempt.attempt_id)
        if len(self.canonical()) > MAX_INVOCATION_METADATA_BYTES:
            raise ValueError("capture_limit")
        return self

    def canonical(self) -> bytes:
        return canonical_bytes(self.model_dump(mode="json"))

    @property
    def metadata_sha256(self) -> str:
        return portable_sha256(self.model_dump(mode="json"))
