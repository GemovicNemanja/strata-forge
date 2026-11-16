"""Private immutable payload pins and descriptive current-policy observations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from strata_forge.core.errors import ValidationError

MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
CHUNK_BYTES = 64 * 1024
MAX_LIVE_READS = 2
READ_SECONDS = 12.0
type Purpose = Literal["evaluation", "inference", "classifier_training", "prompt_render"]
type InputCode = Literal[
    "input_invalid",
    "input_purpose_unsupported",
    "input_authority_unavailable",
    "input_denied",
    "input_changed",
    "input_expired",
    "input_deadline",
    "input_source_unavailable",
    "input_source_invalid",
    "input_integrity",
    "input_limited",
    "input_closed",
    "input_cleanup_failed",
]
type Id = Annotated[str, Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")]
type Digest = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]


class ImmutableInputError(ValidationError):
    """A fixed code contains no rejected identity, path, payload or diagnostic."""


class _Closed(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        hide_input_in_errors=True,
        revalidate_instances="always",
    )


class NativeVersionRef(_Closed):
    """Server-compatible immutable metadata identity without a delivery location."""

    resource_id: Id
    version_id: Id
    manifest_sha256: Digest


class NativeManifestPayload(_Closed):
    """One exact manifest entry; construction does not authorize or retain its bytes."""

    artifact_id: Id
    logical_name: str = Field(min_length=1, max_length=160)
    sha256: Digest
    size_bytes: int = Field(ge=0, le=MAX_PAYLOAD_BYTES)
    media_type: str = Field(min_length=1, max_length=120)
    schema_id: str = Field(min_length=1, max_length=120)

    @field_validator("logical_name")
    @classmethod
    def _logical(cls, value: str) -> str:
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", value, re.ASCII) is None
            or ".." in value
        ):
            raise ValueError("input_invalid")
        return value

    @field_validator("media_type", "schema_id")
    @classmethod
    def _label(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,119}", value, re.ASCII) is None:
            raise ValueError("input_invalid")
        return value


class ImmutableInputRequest(_Closed):
    """The complete requested scope is descriptive until the installed authority checks it."""

    version: Literal[1]
    principal_id: Id
    project_id: Id
    run_id: Id
    attempt_id: Id
    purpose: Purpose
    spec_sha256: Digest
    kind: Literal["model", "dataset", "prompt_build"]
    version_ref: NativeVersionRef
    payload: NativeManifestPayload

    @field_validator("version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("input_invalid")
        return value


class ImmutableInputDecision(_Closed):
    """An installed authority's current observation, never a caller grant or capability."""

    request: ImmutableInputRequest
    effect: Literal["allow", "deny", "unavailable"]
    custody: Literal["unsealed", "sealed", "locked", "fresh", "unknown"]
    policy_identity: Id
    policy_revision: int = Field(ge=1, le=2**53 - 1)
    expires_at_ms: int = Field(ge=0, le=2**53 - 1)


def _fields(value: object, expected: type[BaseModel]) -> dict[str, object]:
    if type(value) is not expected:
        raise ImmutableInputError("input_invalid")
    row = vars(value)
    extra = value.__pydantic_extra__
    if (
        type(row) is not dict
        or len(row) != len(expected.model_fields)
        or (extra is not None and (type(extra) is not dict or len(extra) != 0))
        or set(row) != set(expected.model_fields)
    ):
        raise ImmutableInputError("input_invalid")
    return dict(cast("dict[str, object]", row))


def _scalar_fields(row: dict[str, object], integers: tuple[str, ...] = ()) -> None:
    for name, value in row.items():
        if type(value) is not (int if name in integers else str):
            raise ImmutableInputError("input_invalid")


def checked_request(value: object) -> ImmutableInputRequest:
    """Copy exact bounded fields rather than trusting frozen/model-constructed instances."""
    result = None
    unsupported = False
    try:
        row = _fields(value, ImmutableInputRequest)
        version = _fields(row.pop("version_ref"), NativeVersionRef)
        payload = _fields(row.pop("payload"), NativeManifestPayload)
        _scalar_fields(version)
        _scalar_fields(payload, ("size_bytes",))
        _scalar_fields(row, ("version",))
        if row["purpose"] not in (
            "evaluation",
            "inference",
            "classifier_training",
            "prompt_render",
        ):
            unsupported = True
        else:
            result = ImmutableInputRequest.model_validate(
                {**row, "version_ref": version, "payload": payload}
            )
    except Exception:
        result = None
    if unsupported:
        raise ImmutableInputError("input_purpose_unsupported") from None
    if result is None:
        raise ImmutableInputError("input_invalid") from None
    return result


def checked_decision(value: object) -> ImmutableInputDecision:
    """Revalidate a complete installed observation without reading arbitrary exception text."""
    result = None
    try:
        row = _fields(value, ImmutableInputDecision)
        request = checked_request(row.pop("request"))
        _scalar_fields(row, ("policy_revision", "expires_at_ms"))
        result = ImmutableInputDecision.model_validate({**row, "request": request})
    except Exception:
        result = None
    if result is None:
        raise ImmutableInputError("input_authority_unavailable") from None
    return result


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedInputPayload:
    """Private source output; the source lease retains the original FD until close."""

    request: ImmutableInputRequest
    data: bytes | None = None
    fd: int | None = None


@dataclass(frozen=True, slots=True, repr=False)
class VerifiedInputPayload:
    """Already-read bounded bytes; neither this value nor its digest authorizes another read."""

    request: ImmutableInputRequest
    data: bytes


class HumanRenderInputRequest(_Closed):
    """A live human operation is explicitly distinct from a durable run admission."""

    version: Literal[1]
    admission: Literal["human_render"]
    principal_id: Id
    project_id: Id
    operation_id: Id
    attempt_id: Id
    purpose: Literal["prompt_render"]
    spec_sha256: Digest
    kind: Literal["prompt_build"]
    version_ref: NativeVersionRef
    payload: NativeManifestPayload

    @field_validator("version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("input_invalid")
        return value


class HumanRenderInputDecision(_Closed):
    """An actual installed human authority's observation, never a transferable grant."""

    request: HumanRenderInputRequest
    effect: Literal["allow", "deny", "unavailable"]
    custody: Literal["unsealed", "sealed", "locked", "fresh", "unknown"]
    policy_identity: Id
    policy_revision: int = Field(ge=1, le=2**53 - 1)
    expires_at_ms: int = Field(ge=0, le=2**53 - 1)


def checked_human_render_request(value: object) -> HumanRenderInputRequest:
    """Detach only the closed human variant without relabeling it as a durable run."""
    result = None
    unsupported = False
    try:
        row = _fields(value, HumanRenderInputRequest)
        version = _fields(row.pop("version_ref"), NativeVersionRef)
        payload = _fields(row.pop("payload"), NativeManifestPayload)
        _scalar_fields(version)
        _scalar_fields(payload, ("size_bytes",))
        _scalar_fields(row, ("version",))
        if row["purpose"] != "prompt_render":
            unsupported = True
        else:
            result = HumanRenderInputRequest.model_validate(
                {**row, "version_ref": version, "payload": payload}
            )
    except Exception:
        result = None
    if unsupported:
        raise ImmutableInputError("input_purpose_unsupported") from None
    if result is None:
        raise ImmutableInputError("input_invalid") from None
    return result


def checked_human_render_decision(value: object) -> HumanRenderInputDecision:
    """Revalidate current human scope and fixed observations without exception diagnostics."""
    result = None
    try:
        row = _fields(value, HumanRenderInputDecision)
        request = checked_human_render_request(row.pop("request"))
        _scalar_fields(row, ("policy_revision", "expires_at_ms"))
        result = HumanRenderInputDecision.model_validate({**row, "request": request})
    except Exception:
        result = None
    if result is None:
        raise ImmutableInputError("input_authority_unavailable") from None
    return result


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedHumanRenderPayload:
    """Human operation source output; its lease owns any original FD through cleanup."""

    request: HumanRenderInputRequest
    data: bytes | None = None
    fd: int | None = None


@dataclass(frozen=True, slots=True, repr=False)
class VerifiedHumanRenderPayload:
    """Already disclosed bounded human bytes, without durable-run or reusable authority claims."""

    request: HumanRenderInputRequest
    data: bytes
