"""Private human prompt rendering over an installed immutable-input reader."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from strata_forge.core.errors import ValidationError
from strata_forge.pipelines._immutable_input import HumanRenderInputReader
from strata_forge.pipelines.immutable_input_types import (
    READ_SECONDS,
    HumanRenderInputRequest,
    ImmutableInputError,
    NativeManifestPayload,
    NativeVersionRef,
    checked_human_render_request,
)
from strata_forge.prompts.consumption import (
    ConsumedBuild,
    LocalBuildReference,
    LocalBuildSource,
    consume_build,
    render_consumed,
)
from strata_forge.prompts.portable import PortableValue, canonical_bytes, parse_portable_json

if TYPE_CHECKING:
    from strata_forge.prompts.composition import PromptRender

MAX_NATIVE_VALUES = 128
MAX_NATIVE_VALUES_BYTES = 64 * 1024
MAX_NATIVE_BUILD_BYTES = 8 * 1024 * 1024
MAX_NATIVE_RENDER_BYTES = 8 * 1024 * 1024
_DOMAIN = "strata-human-prompt-render-intent-v1"
_ZERO_DIGEST = "0" * 64
_ISSUER = object()


class NativePromptInputError(ValidationError):
    """A fixed private code excludes rejected content and native identity diagnostics."""


class _RefusalError(Exception):
    def __init__(self, code: str = "native_prompt_invalid") -> None:
        self.code = code


@dataclass(frozen=True, slots=True, repr=False)
class _ConsumedOperation:
    owner: asyncio.Task[object]
    deadline: float
    intent_sha256: str
    values_json: bytes
    issuer: object


@dataclass(frozen=True, slots=True, repr=False)
class NativePromptConsumedCandidate:
    """Internal local bytes awaiting retained-pin validation and final server authorization."""

    request: HumanRenderInputRequest
    surface_ref: NativeVersionRef
    consumed: ConsumedBuild
    values_json: bytes
    _operation: _ConsumedOperation

    @property
    def original_deadline(self) -> float:
        """Return the first stage's retained deadline without admitting a replacement budget."""
        return self._operation.deadline


@dataclass(frozen=True, slots=True, repr=False)
class NativePromptRenderCandidate:
    """Internal rendered content awaiting the server's final current-authorization fence."""

    request: HumanRenderInputRequest
    surface_ref: NativeVersionRef
    render: PromptRender


def _values(value: object) -> dict[str, PortableValue]:
    if type(value) is not dict:
        raise _RefusalError()
    supplied = cast("dict[object, object]", value)
    if len(supplied) > MAX_NATIVE_VALUES:
        raise _RefusalError("native_prompt_values_limited")
    snapshot: dict[str, PortableValue] = {}
    size = 2
    for key, scalar in supplied.items():
        if type(key) is not str or type(scalar) not in (str, int, bool):
            raise _RefusalError()
        if len(key) > MAX_NATIVE_VALUES_BYTES or (
            type(scalar) is str and len(scalar) > MAX_NATIVE_VALUES_BYTES
        ):
            raise _RefusalError("native_prompt_values_limited")
        # Each scalar is immutable; the fresh dictionary cannot change across reader awaits.
        item = cast("PortableValue", scalar)
        size += len(canonical_bytes({key: item})) - 2 + bool(snapshot)
        if size > MAX_NATIVE_VALUES_BYTES:
            raise _RefusalError("native_prompt_values_limited")
        snapshot[key] = item
    return snapshot


def _snapshot(
    request: HumanRenderInputRequest, surface_ref: NativeVersionRef, values: object
) -> tuple[HumanRenderInputRequest, NativeVersionRef, dict[str, PortableValue], str]:
    checked = checked_human_render_request(request)
    surface = checked_human_render_request(
        checked.model_copy(update={"version_ref": surface_ref})
    ).version_ref
    payload = checked.payload
    if (
        payload.logical_name != "build.json"
        or payload.media_type != "application/json"
        or payload.schema_id != "strata.prompt.build.v1"
        or not 1 <= payload.size_bytes <= MAX_NATIVE_BUILD_BYTES
    ):
        raise _RefusalError()
    inputs = _values(values)
    intent = checked.model_dump(mode="json", exclude={"spec_sha256"})
    intent.update(
        {"domain": _DOMAIN, "surface_ref": surface.model_dump(mode="json"), "values": inputs}
    )
    digest = hashlib.sha256(canonical_bytes(intent)).hexdigest()
    return checked, surface, inputs, digest


def native_prompt_intent_digest(
    *,
    principal_id: str,
    project_id: str,
    operation_id: str,
    attempt_id: str,
    surface_ref: NativeVersionRef,
    version_ref: NativeVersionRef,
    payload: NativeManifestPayload,
    values: object,
) -> str:
    """Hash the complete fixed human-render intent, excluding only its own digest field."""
    failure = "native_prompt_invalid"
    try:
        request = HumanRenderInputRequest(
            version=1,
            admission="human_render",
            principal_id=principal_id,
            project_id=project_id,
            operation_id=operation_id,
            attempt_id=attempt_id,
            purpose="prompt_render",
            spec_sha256=_ZERO_DIGEST,
            kind="prompt_build",
            version_ref=version_ref,
            payload=payload,
        )
        return _snapshot(request, surface_ref, values)[3]
    except _RefusalError as error:
        failure = error.code
    except Exception:
        failure = "native_prompt_invalid"
    raise NativePromptInputError(failure) from None


def _current(owner: asyncio.Task[object], deadline: float) -> None:
    if asyncio.current_task() is not owner:
        raise _RefusalError("native_prompt_closed")
    if owner.cancelling():
        raise asyncio.CancelledError
    if time.monotonic() >= deadline:
        raise _RefusalError("native_prompt_deadline")


def _local_source(
    request: HumanRenderInputRequest, surface_ref: NativeVersionRef
) -> LocalBuildSource:
    return LocalBuildSource(
        kind="caller_bytes",
        partition=f"native:{request.project_id}",
        reference=LocalBuildReference(
            project_id=request.project_id,
            surface_resource_id=surface_ref.resource_id,
            resource_id=request.version_ref.resource_id,
            version_id=request.version_ref.version_id,
            manifest_sha256=request.version_ref.manifest_sha256,
        ),
    )


async def consume_native_prompt_input(
    reader: HumanRenderInputReader,
    request: HumanRenderInputRequest,
    *,
    surface_ref: NativeVersionRef,
    values: object,
    deadline: float,
) -> NativePromptConsumedCandidate:
    """Read one bound human intent for the server's actual retained dependency-pin validation."""
    failure = "native_prompt_invalid"
    try:
        if type(reader) is not HumanRenderInputReader or type(deadline) is not float:
            raise _RefusalError()
        if not math.isfinite(deadline):
            raise _RefusalError()
        owner = asyncio.current_task()
        if owner is None:
            raise _RefusalError("native_prompt_closed")
        original_deadline = min(deadline, time.monotonic() + READ_SECONDS)
        _current(owner, original_deadline)
        checked, surface, inputs, digest = _snapshot(request, surface_ref, values)
        if digest != checked.spec_sha256:
            raise _RefusalError("native_prompt_intent_mismatch")
        _current(owner, original_deadline)
        payload = await reader.read(checked, deadline=original_deadline)
        _current(owner, original_deadline)
        if payload.request != checked:
            raise _RefusalError("native_prompt_intent_mismatch")
        consumed = consume_build(
            payload.data,
            expected_build_id=checked.payload.sha256,
            expected_size_bytes=checked.payload.size_bytes,
            source=_local_source(checked, surface),
        )
        _current(owner, original_deadline)
        values_json = canonical_bytes(inputs)
        operation = _ConsumedOperation(owner, original_deadline, digest, values_json, _ISSUER)
        _current(owner, original_deadline)
        return NativePromptConsumedCandidate(checked, surface, consumed, values_json, operation)
    except _RefusalError as error:
        failure = error.code
    except ImmutableInputError:
        raise
    except Exception:
        failure = "native_prompt_render_invalid"
    raise NativePromptInputError(failure) from None


async def render_native_prompt_consumed(
    candidate: NativePromptConsumedCandidate, *, deadline: float
) -> NativePromptRenderCandidate:
    """Render a revalidated same-task local candidate without renewing its original deadline."""
    failure = "native_prompt_invalid"
    try:
        if type(candidate) is not NativePromptConsumedCandidate:
            raise _RefusalError()
        operation = candidate._operation  # pyright: ignore[reportPrivateUsage] -- fixed same-module consumer checks the retained operation state.
        if (
            type(operation) is not _ConsumedOperation
            or operation.issuer is not _ISSUER
            or type(operation.deadline) is not float
            or not math.isfinite(operation.deadline)
            or type(deadline) is not float
            or not math.isfinite(deadline)
        ):
            raise _RefusalError()
        original_deadline = min(deadline, operation.deadline)
        _current(operation.owner, original_deadline)
        if (
            type(candidate.values_json) is not bytes
            or len(candidate.values_json) > MAX_NATIVE_VALUES_BYTES
            or candidate.values_json != operation.values_json
        ):
            raise _RefusalError("native_prompt_intent_mismatch")
        values = parse_portable_json(candidate.values_json)
        checked, surface, inputs, digest = _snapshot(
            candidate.request, candidate.surface_ref, values
        )
        if (
            digest != checked.spec_sha256
            or digest != operation.intent_sha256
            or canonical_bytes(inputs) != candidate.values_json
        ):
            raise _RefusalError("native_prompt_intent_mismatch")
        stored = candidate.consumed
        if (
            type(stored) is not ConsumedBuild
            or set(stored.__dict__) != set(ConsumedBuild.model_fields)
            or stored.__pydantic_extra__
        ):
            raise _RefusalError()
        consumed = consume_build(
            stored.canonical,
            expected_build_id=stored.build_id,
            expected_size_bytes=stored.size_bytes,
            source=stored.source,
        )
        if (
            consumed.build_id != checked.payload.sha256
            or consumed.size_bytes != checked.payload.size_bytes
            or consumed.source != _local_source(checked, surface)
        ):
            raise _RefusalError("native_prompt_intent_mismatch")
        _current(operation.owner, original_deadline)
        rendered = await render_consumed(consumed, inputs)
        _current(operation.owner, original_deadline)
        if len(rendered.canonical) > MAX_NATIVE_RENDER_BYTES:
            raise _RefusalError("native_prompt_render_limited")
        return NativePromptRenderCandidate(checked, surface, rendered)
    except _RefusalError as error:
        failure = error.code
    except ImmutableInputError:
        raise
    except Exception:
        failure = "native_prompt_render_invalid"
    raise NativePromptInputError(failure) from None


async def render_native_prompt_input(
    reader: HumanRenderInputReader,
    request: HumanRenderInputRequest,
    *,
    surface_ref: NativeVersionRef,
    values: object,
    deadline: float,
) -> NativePromptRenderCandidate:
    """Compose both private stages; the server's retained-pin and final fences remain external."""
    candidate = await consume_native_prompt_input(
        reader, request, surface_ref=surface_ref, values=values, deadline=deadline
    )
    return await render_native_prompt_consumed(candidate, deadline=deadline)
