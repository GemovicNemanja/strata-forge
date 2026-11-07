"""Closed local structured projections; provider/wire identity and authority stay unproved."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.prompts.consumption import LocalBuildReference  # noqa: TC001 - runtime schema
from strata_forge.prompts.portable import canonical_bytes, portable_sha256

MAX_PROVIDER_ATTEMPTS = 64
MAX_PROVIDER_ATTEMPT_BYTES = 4096
MAX_PROVIDER_OBSERVATION_BYTES = 256 * 1024
MAX_PROVIDER_PROJECTION_BYTES = 16 * 1024 * 1024
MAX_PROVIDER_SAFE_INTEGER = 9_007_199_254_740_991

type ProviderMethod = Literal["acompletion", "aresponses"]
type ProviderSlot = Literal["anthropic", "openai", "vertex", "bedrock", "azure", "openai_compat"]
type ProviderId = Annotated[str, Field(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$")]
type ProviderDigest = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")]
type ProviderCount = Annotated[int, Field(ge=0, le=MAX_PROVIDER_SAFE_INTEGER)]
type ProviderCaptureIssue = Literal[
    "input_unsupported",
    "input_limit",
    "response_opaque",
    "response_unsupported",
    "response_limit",
    "identity_unavailable",
    "retired",
]
type ProviderGap = Literal[
    "whole_kwargs_unobserved",
    "auth_merge_unobserved",
    "http_wire_unobserved",
    "provider_acceptance_unobserved",
    "immutable_model_unobserved",
    "render_projection_unobserved",
    "call_relationship_unobserved",
    "usage_cost_uninterpreted",
    "body_not_retained",
    "unselected_fields_omitted",
    "request_projection_unavailable",
    "response_projection_unavailable",
    "identity_unavailable",
    "observer_retired",
]

BASE_PROVIDER_GAPS: tuple[ProviderGap, ...] = (
    "whole_kwargs_unobserved",
    "auth_merge_unobserved",
    "http_wire_unobserved",
    "provider_acceptance_unobserved",
    "immutable_model_unobserved",
    "render_projection_unobserved",
    "call_relationship_unobserved",
    "usage_cost_uninterpreted",
    "body_not_retained",
)
ISSUE_ORDER: tuple[ProviderCaptureIssue, ...] = (
    "input_unsupported",
    "input_limit",
    "response_opaque",
    "response_unsupported",
    "response_limit",
    "identity_unavailable",
    "retired",
)
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z\Z")


class _ProviderMetadata(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    @field_validator("version", mode="before", check_fields=False)
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("observation_invalid")
        return value

    @field_validator(
        "request_entered",
        "return_observed",
        "closed",
        "scope_mismatch_seen",
        "counter_saturated",
        "all_calls_observed",
        "streaming_observed",
        "cache_events_observed",
        "post_close_calls_counted",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def exact_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("observation_invalid")
        return value

    @field_validator("started_at", "ended_at", "created_at", check_fields=False)
    @classmethod
    def exact_time(cls, value: str) -> str:
        if _UTC.fullmatch(value) is None:
            raise ValueError("observation_invalid")
        datetime.fromisoformat(value)
        return value

    def canonical(self) -> bytes:
        return canonical_bytes(self.model_dump(mode="json"))


class _ProviderDomain(_ProviderMetadata):
    @property
    def metadata_sha256(self) -> str:
        return portable_sha256(self.model_dump(mode="json"))


class ProviderProjectionMetadata(_ProviderMetadata):
    profile: Literal["forge-provider-client-input/v1"]
    status: Literal["observed", "opaque", "unsupported", "limit", "not_returned", "retired"]
    canonical_tagged_sha256: ProviderDigest | None
    encoded_size_bytes: int | None = Field(ge=1, le=MAX_PROVIDER_PROJECTION_BYTES)
    selected_present_fields_count: int | None = Field(ge=0, le=12)
    omitted_fields_count: int | None = Field(ge=0, le=4096)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        observed = self.status == "observed"
        if observed != (self.canonical_tagged_sha256 is not None) or observed != (
            self.encoded_size_bytes is not None
        ):
            raise ValueError("observation_invalid")
        if (self.selected_present_fields_count is None) != (self.omitted_fields_count is None):
            raise ValueError("observation_invalid")
        if observed and self.selected_present_fields_count is None:
            raise ValueError("observation_invalid")
        if (
            self.status in {"opaque", "not_returned", "retired"}
            and self.selected_present_fields_count is not None
        ):
            raise ValueError("observation_invalid")
        return self


def unavailable_projection(
    status: Literal["opaque", "unsupported", "limit", "not_returned", "retired"],
) -> ProviderProjectionMetadata:
    return ProviderProjectionMetadata(
        profile="forge-provider-client-input/v1",
        status=status,
        canonical_tagged_sha256=None,
        encoded_size_bytes=None,
        selected_present_fields_count=None,
        omitted_fields_count=None,
    )


class ProviderAttemptMetadata(_ProviderDomain):
    domain: Literal["strata-provider-prompt-attempt-v1"]
    version: Literal[1]
    observation_id: ProviderId
    attempt_id: ProviderId
    entry_index: int = Field(ge=0, lt=MAX_PROVIDER_ATTEMPTS)
    invocation_id: ProviderId
    build_id: ProviderDigest
    render_id: ProviderDigest
    input_fingerprint: ProviderDigest
    source_kind: Literal["caller_bytes"]
    method: ProviderMethod
    declared_provider_slot: ProviderSlot
    observed_provider_model_id: str | None = Field(
        min_length=1, max_length=512, pattern=r"^[\x21-\x7e]+$"
    )
    logical_model: None
    served_model: None
    model_revision: None
    source_relation: Literal["unverified_argument"]
    started_at: str
    ended_at: str
    duration_us: ProviderCount
    call_outcome: Literal["returned", "raised", "cancelled", "interrupted"]
    request_entered: Literal[True]
    return_observed: bool
    error_category: Literal["delegate_error", "cancelled", "interrupted"] | None
    request_projection: ProviderProjectionMetadata
    response_projection: ProviderProjectionMetadata
    capture_status: Literal["recorded", "partial"]
    capture_issues: tuple[ProviderCaptureIssue, ...] = Field(max_length=7)
    parents: None
    relation: Literal["unknown"]
    render_relation: Literal["unknown"]
    usage: None
    cost: None
    retention: Literal["metadata_only"]
    export_status: Literal["not_configured"]
    gaps: tuple[ProviderGap, ...] = Field(max_length=14)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        request, response = self.request_projection, self.response_projection
        if request.status not in {"observed", "unsupported", "limit"}:
            raise ValueError("observation_invalid")
        if self.call_outcome == "returned":
            if (
                self.error_category is not None
                or response.status == "not_returned"
                or self.return_observed != (response.status != "retired")
            ):
                raise ValueError("observation_invalid")
        else:
            expected = {
                "raised": "delegate_error",
                "cancelled": "cancelled",
                "interrupted": "interrupted",
            }[self.call_outcome]
            if (
                self.error_category != expected
                or self.return_observed
                or response.status != "not_returned"
            ):
                raise ValueError("observation_invalid")
        issues: set[ProviderCaptureIssue] = set()
        if request.status == "unsupported":
            issues.add("input_unsupported")
        if request.status == "limit":
            issues.add("input_limit")
        if response.status == "opaque":
            issues.add("response_opaque")
        if response.status == "unsupported":
            issues.add("response_unsupported")
        if response.status == "limit":
            issues.add("response_limit")
        if self.observed_provider_model_id is None:
            issues.add("identity_unavailable")
        if response.status == "retired":
            issues.add("retired")
        if self.capture_issues != tuple(issue for issue in ISSUE_ORDER if issue in issues):
            raise ValueError("observation_invalid")
        if (self.capture_status == "recorded") != (not issues):
            raise ValueError("observation_invalid")
        required = set(BASE_PROVIDER_GAPS)
        if (request.omitted_fields_count or 0) > 0 or (response.omitted_fields_count or 0) > 0:
            required.add("unselected_fields_omitted")
        if request.status != "observed":
            required.add("request_projection_unavailable")
        if response.status in {"opaque", "unsupported", "limit", "retired"}:
            required.add("response_projection_unavailable")
        if self.observed_provider_model_id is None:
            required.add("identity_unavailable")
        if response.status == "retired":
            required.add("observer_retired")
        if (
            len(set(self.gaps)) != len(self.gaps)
            or set(self.gaps) != required
            or len(self.canonical()) > MAX_PROVIDER_ATTEMPT_BYTES
        ):
            raise ValueError("observation_invalid")
        return self


class ProviderObservationLoss(_ProviderMetadata):
    over_capacity_count: ProviderCount
    projection_limited_count: ProviderCount
    metadata_lost_count: ProviderCount
    scope_mismatch_seen: bool
    counter_saturated: bool


class ProviderObservationCoverage(_ProviderMetadata):
    all_calls_observed: Literal[False]
    streaming_observed: Literal[False]
    cache_events_observed: Literal[False]
    post_close_calls_counted: Literal[False]


class ProviderObservationMetadata(_ProviderDomain):
    domain: Literal["strata-provider-prompt-observation-v1"]
    version: Literal[1]
    observation_id: ProviderId
    invocation_id: ProviderId
    source_kind: Literal["caller_bytes"]
    source_partition: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    source_reference: LocalBuildReference | None
    build_id: ProviderDigest
    render_id: ProviderDigest
    input_fingerprint: ProviderDigest
    created_at: str
    closed: bool
    admitted_attempts: int = Field(ge=0, le=MAX_PROVIDER_ATTEMPTS)
    attempts: tuple[ProviderAttemptMetadata, ...] = Field(max_length=MAX_PROVIDER_ATTEMPTS)
    loss: ProviderObservationLoss
    coverage: ProviderObservationCoverage
    retention: Literal["metadata_only"]
    export_status: Literal["not_configured"]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        seen_ids: set[str] = set()
        previous = -1
        for attempt in self.attempts:
            if (
                (
                    attempt.observation_id,
                    attempt.invocation_id,
                    attempt.build_id,
                    attempt.render_id,
                    attempt.input_fingerprint,
                )
                != (
                    self.observation_id,
                    self.invocation_id,
                    self.build_id,
                    self.render_id,
                    self.input_fingerprint,
                )
                or attempt.attempt_id in seen_ids
                or not previous < attempt.entry_index < self.admitted_attempts
            ):
                raise ValueError("observation_invalid")
            seen_ids.add(attempt.attempt_id)
            previous = attempt.entry_index
        if (
            self.admitted_attempts - len(self.attempts) != self.loss.metadata_lost_count
            or len(self.canonical()) > MAX_PROVIDER_OBSERVATION_BYTES
        ):
            raise ValueError("observation_invalid")
        return self
