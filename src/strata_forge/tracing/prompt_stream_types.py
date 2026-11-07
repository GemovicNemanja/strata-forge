"""Closed local native-stream counts/timing metadata; no provider/wire/output claim."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.prompts.consumption import LocalBuildReference  # noqa: TC001 - runtime schema
from strata_forge.prompts.portable import canonical_bytes, portable_sha256
from strata_forge.tracing.prompt_provider_types import (  # noqa: TC001 - runtime schema
    ProviderCount,
    ProviderDigest,
    ProviderId,
    ProviderProjectionMetadata,
    ProviderSlot,
)

MAX_STREAMS = 64
MAX_STREAM_BYTES = 4096
MAX_STREAM_OBSERVATION_BYTES = 256 * 1024
MAX_STREAM_SAFE_INTEGER = 9_007_199_254_740_991

type StreamMethod = Literal["astream", "aresponses_stream"]
type StreamOperation = Literal["anext", "asend", "athrow", "aclose"]
type StreamAction = Literal["next", "send", "throw", "close"]
type StreamOutcome = Literal[
    "item",
    "suspended",
    "exhausted",
    "close_completed",
    "raised",
    "cancelled",
    "interrupted",
    "awaitable_closed",
]
type StreamGap = Literal[
    "http_wire_unobserved",
    "provider_acceptance_unobserved",
    "immutable_model_unobserved",
    "render_projection_unobserved",
    "call_relationship_unobserved",
    "usage_cost_uninterpreted",
    "stream_item_content_unobserved",
    "body_not_retained",
    "factory_transforms_unobserved",
    "input_unobserved",
    "input_projection_unavailable",
    "iterator_end_unconfirmed",
    "observer_retired",
    "identity_unavailable",
]
BASE_STREAM_GAPS: tuple[StreamGap, ...] = (
    "http_wire_unobserved",
    "provider_acceptance_unobserved",
    "immutable_model_unobserved",
    "render_projection_unobserved",
    "call_relationship_unobserved",
    "usage_cost_uninterpreted",
    "stream_item_content_unobserved",
    "body_not_retained",
    "factory_transforms_unobserved",
)
_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z\Z")


class _Metadata(BaseModel):
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
            raise ValueError("stream_observation_invalid")
        return value

    @field_validator(
        "iterator_obtained",
        "counter_saturated",
        "body_retained",
        "closed",
        "scope_mismatch_seen",
        "all_streams_observed",
        "provider_execution_verified",
        "item_content_observed",
        "cache_events_observed",
        "causal_links_verified",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def exact_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("stream_observation_invalid")
        return value

    @field_validator(
        "observed_at",
        "factory_returned_at",
        "first_drive_attempt_at",
        "first_item_at",
        "last_observation_at",
        "closed_observed_at",
        "retired_at",
        "created_at",
        check_fields=False,
    )
    @classmethod
    def exact_time(cls, value: str | None) -> str | None:
        if value is not None:
            if _TIME.fullmatch(value) is None:
                raise ValueError("stream_observation_invalid")
            datetime.fromisoformat(value)
        return value

    def canonical(self) -> bytes:
        return canonical_bytes(self.model_dump(mode="json"))

    @property
    def metadata_sha256(self) -> str:
        return portable_sha256(self.model_dump(mode="json"))


class StreamInputBinding(_Metadata):
    domain: Literal["strata-provider-stream-input-v1"]
    version: Literal[1]
    actual_method: StreamMethod
    normalized_schema: Literal["acompletion", "aresponses"]
    stage: Literal["unobserved", "before_first_drive_attempt"]
    observed_at: str | None
    projection: ProviderProjectionMetadata | None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.normalized_schema != (
            "acompletion" if self.actual_method == "astream" else "aresponses"
        ):
            raise ValueError("stream_observation_invalid")
        observed = self.stage == "before_first_drive_attempt"
        if observed != (self.observed_at is not None) or observed != (self.projection is not None):
            raise ValueError("stream_observation_invalid")
        if self.projection is not None:
            count = self.projection.selected_present_fields_count
            minimum, maximum = (1, 10) if self.actual_method == "astream" else (0, 12)
            if self.projection.status not in {"observed", "unsupported", "limit"} or (
                count is not None and not minimum <= count <= maximum
            ):
                raise ValueError("stream_observation_invalid")
        return self


class ProviderStreamMetadata(_Metadata):
    domain: Literal["strata-provider-stream-v1"]
    version: Literal[1]
    observation_id: ProviderId
    stream_id: ProviderId
    entry_index: int = Field(ge=0, lt=MAX_STREAMS)
    invocation_id: ProviderId
    build_id: ProviderDigest
    render_id: ProviderDigest
    input_fingerprint: ProviderDigest
    source_kind: Literal["caller_bytes"]
    method: StreamMethod
    declared_provider_slot: ProviderSlot
    observed_provider_model_id: str | None = Field(default=None, min_length=1, max_length=512)
    logical_model: None
    served_model: None
    model_revision: None
    usage: None
    cost: None
    render_relation: Literal["unknown"]
    parents: None
    relation: Literal["unknown"]
    input_binding: StreamInputBinding
    factory_returned_at: str
    first_drive_attempt_at: str | None
    first_item_at: str | None
    last_observation_at: str
    closed_observed_at: str | None
    retired_at: str | None
    first_item_drive_latency_us: ProviderCount | None
    capture_state: Literal["observing", "retired", "lost"]
    iterator_obtained: Literal[True]
    end_observation: Literal["exhausted", "aclose_completed"] | None
    last_operation: StreamOperation | None
    last_protocol_action: StreamAction | None
    last_operation_outcome: StreamOutcome | None
    raw_items_observed: ProviderCount
    data_advance_attempts: ProviderCount
    close_attempts: ProviderCount
    operation_errors: ProviderCount
    operation_cancellations: ProviderCount
    counter_saturated: bool
    output_projection: None
    body_retained: Literal[False]
    retention: Literal["metadata_only"]
    export_status: Literal["not_configured"]
    gaps: tuple[StreamGap, ...]

    @field_validator("observed_provider_model_id")
    @classmethod
    def exact_label(cls, value: str | None) -> str | None:
        if value is not None and any(not 0x21 <= ord(char) <= 0x7E for char in value):
            raise ValueError("stream_observation_invalid")
        return value

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.input_binding.actual_method != self.method:
            raise ValueError("stream_observation_invalid")
        driven = self.first_drive_attempt_at is not None
        first_item = self.first_item_at is not None
        retired = self.capture_state == "retired"
        if (
            self.raw_items_observed > self.data_advance_attempts
            or self.operation_cancellations > self.operation_errors
            or driven != (self.data_advance_attempts > 0)
            or driven != (self.input_binding.stage == "before_first_drive_attempt")
            or first_item != (self.raw_items_observed > 0)
            or first_item != (self.first_item_drive_latency_us is not None)
            or (first_item and not driven)
            or retired != (self.retired_at is not None)
            or (self.end_observation is None) != (self.closed_observed_at is None)
            or (self.last_operation is None) != (self.last_protocol_action is None)
            or (self.last_operation is None) != (self.last_operation_outcome is None)
        ):
            raise ValueError("stream_observation_invalid")
        required = set(BASE_STREAM_GAPS)
        if not driven:
            required.add("input_unobserved")
        elif (
            self.input_binding.projection is not None
            and self.input_binding.projection.status != "observed"
        ):
            required.add("input_projection_unavailable")
        if self.end_observation is None:
            required.add("iterator_end_unconfirmed")
        if retired:
            required.add("observer_retired")
        if self.observed_provider_model_id is None:
            required.add("identity_unavailable")
        if len(set(self.gaps)) != len(self.gaps) or set(self.gaps) != required:
            raise ValueError("stream_observation_invalid")
        if len(self.canonical()) > MAX_STREAM_BYTES:
            raise ValueError("stream_observation_invalid")
        return self


class StreamObservationLoss(_Metadata):
    over_capacity_count: ProviderCount
    unsupported_iterator_count: ProviderCount
    factory_failure_count: ProviderCount
    metadata_lost_count: ProviderCount
    scope_mismatch_seen: bool
    counter_saturated: bool


class StreamObservationCoverage(_Metadata):
    all_streams_observed: Literal[False]
    provider_execution_verified: Literal[False]
    item_content_observed: Literal[False]
    cache_events_observed: Literal[False]
    causal_links_verified: Literal[False]


class ProviderStreamObservationMetadata(_Metadata):
    domain: Literal["strata-provider-stream-observation-v1"]
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
    streams: tuple[ProviderStreamMetadata, ...] = Field(max_length=MAX_STREAMS)
    admitted_streams: int = Field(ge=0, le=MAX_STREAMS)
    loss: StreamObservationLoss
    coverage: StreamObservationCoverage
    retention: Literal["metadata_only"]
    export_status: Literal["not_configured"]

    @field_validator("source_partition")
    @classmethod
    def exact_partition(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9._:-]+", value) is None:
            raise ValueError("stream_observation_invalid")
        return value

    @model_validator(mode="after")
    def coherent(self) -> Self:
        seen: set[str] = set()
        previous = -1
        for stream in self.streams:
            if (
                (
                    stream.observation_id,
                    stream.invocation_id,
                    stream.build_id,
                    stream.render_id,
                    stream.input_fingerprint,
                )
                != (
                    self.observation_id,
                    self.invocation_id,
                    self.build_id,
                    self.render_id,
                    self.input_fingerprint,
                )
                or stream.stream_id in seen
                or not previous < stream.entry_index < self.admitted_streams
                or (stream.capture_state == "retired") != self.closed
            ):
                raise ValueError("stream_observation_invalid")
            seen.add(stream.stream_id)
            previous = stream.entry_index
        if (
            self.admitted_streams - len(self.streams) != self.loss.metadata_lost_count
            or len(self.canonical()) > MAX_STREAM_OBSERVATION_BYTES
        ):
            raise ValueError("stream_observation_invalid")
        return self
