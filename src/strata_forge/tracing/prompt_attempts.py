"""Opt-in local byte-delegate observations, independent of Langfuse and provider authority."""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
import uuid
import weakref
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, NoReturn, Protocol, cast

from pydantic import ValidationError

from strata_forge.core.errors import ForgeError
from strata_forge.prompts.composition import PromptRender
from strata_forge.prompts.consumption import ConsumedBuild, render_consumed
from strata_forge.tracing.prompt_attempt_types import (
    MAX_ATTEMPT_METADATA_BYTES,
    MAX_ATTEMPTS,
    MAX_BODY_BYTES,
    MAX_INVOCATION_METADATA_BYTES,
    AttemptFacts,
    AttemptMetadata,
    AttemptParents,
    ByteObservation,
    DeclaredBodyPin,
    ErrorCode,
    Gap,
    InvocationMetadata,
)

if TYPE_CHECKING:
    from strata_forge.prompts.portable import PortableValue


class PromptCaptureError(ForgeError):
    """A fixed local failure, containing no native error, body or transport configuration."""

    def __init__(self, code: ErrorCode, attempt_id: str | None = None) -> None:
        allowed = {
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
        }
        if type(code) is not str or code not in allowed:
            code = "capture_invalid"
        if attempt_id is not None and (
            type(attempt_id) is not str
            or len(attempt_id) != 32
            or any(c not in "0123456789abcdef" for c in attempt_id)
        ):
            code, attempt_id = "capture_invalid", None
        self.code = code
        self.attempt_id = attempt_id
        super().__init__(code)


def _fail(code: ErrorCode, attempt_id: str | None = None) -> NoReturn:
    raise PromptCaptureError(code, attempt_id)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _identifier() -> str:
    # Execution identifiers are not experiment PRNG state or authority credentials.
    return uuid.uuid4().hex


def _observation(body: bytes) -> ByteObservation:
    return ByteObservation(
        profile="opaque-application-bytes/v1",
        coverage="complete",
        size_bytes=len(body),
        original_sha256=hashlib.sha256(body).hexdigest(),
        redacted_sha256=None,
        retained=False,
    )


class _Opaque:
    __slots__ = ("__weakref__",)

    def __reduce__(self) -> NoReturn:
        _fail("capture_invalid")

    def __copy__(self) -> NoReturn:
        _fail("capture_invalid")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        _fail("capture_invalid")


@dataclass(slots=True, repr=False)
class _Response:
    body: bytes
    status: int | None


_responses: weakref.WeakKeyDictionary[LocalByteResponse, _Response] = weakref.WeakKeyDictionary()


class LocalByteResponse(_Opaque):
    """Opaque working return bytes, exposed only through an explicit body accessor."""

    __slots__ = ()

    def __init__(self, body: bytes, status: int | None) -> None:
        if (
            type(self) is not LocalByteResponse
            or type(body) is not bytes
            or (status is not None and (type(status) is not int or not 0 <= status <= 999))
        ):
            _fail("capture_invalid")
        _responses[self] = _Response(body, status)

    @property
    def status(self) -> int | None:
        return _response(self).status

    def response_body(self) -> bytes:
        """Explicitly access caller-owned working content; the metadata ledger has none."""
        return _response(self).body

    def __repr__(self) -> str:
        value = _responses.get(self)
        return (
            "LocalByteResponse(unissued)"
            if value is None
            else f"LocalByteResponse(size_bytes={len(value.body)}, status={value.status!r})"
        )


def _response(value: object) -> _Response:
    if type(value) is not LocalByteResponse:
        _fail("capture_response_invalid")
    result = _responses.get(value)
    if result is None:
        _fail("capture_response_invalid")
    return result


class LocalByteTransport(Protocol):
    """One application-owned behavior; the recorder neither routes nor authenticates it."""

    async def exchange(self, body: bytes, /) -> LocalByteResponse: ...


@dataclass(slots=True, repr=False)
class _Ticket:
    attempt_id: str
    owner: asyncio.Task[object]
    started_at: str
    start_ns: int
    parents: AttemptParents
    baseline: DeclaredBodyPin | None
    request: ByteObservation
    child: asyncio.Task[object] | None = None
    handed: bool = False
    retired_at: str | None = None
    cancel_sent: bool = False


@dataclass(slots=True, repr=False)
class _Invocation:
    metadata: InvocationMetadata
    render: PromptRender | None
    loop: asyncio.AbstractEventLoop
    thread: int
    active: _Ticket | None = None


_invocations: weakref.WeakKeyDictionary[PreparedInvocation, _Invocation] = (
    weakref.WeakKeyDictionary()
)


def _state(value: object, *, loop: bool = False) -> _Invocation:
    if type(value) is not PreparedInvocation:
        _fail("capture_invalid")
    state = _invocations.get(value)
    if state is None:
        _fail("capture_invalid")
    if state.thread != threading.get_ident() or (
        loop and state.loop is not asyncio.get_running_loop()
    ):
        _fail("capture_scope_invalid")
    return state


class PreparedInvocation(_Opaque):
    """Factory-issued local working material; automatic serialization never exposes its body."""

    __slots__ = ()

    def __init__(self) -> None:
        _fail("capture_invalid")

    def rendered(self) -> PromptRender:
        state = _state(self)
        if state.render is None:
            _fail("capture_closed")
        return PromptRender(canonical=state.render.canonical)

    def snapshot(self) -> InvocationMetadata:
        state = _state(self)
        # A terminal-only schema must not imply an empty/completed history during a live call.
        if state.active is not None:
            _fail("capture_in_progress")
        return InvocationMetadata.model_validate(state.metadata)

    def close(self) -> None:
        state = _state(self)
        if state.active is not None:
            _fail("capture_in_progress")
        if not state.metadata.closed:
            state.render = None
            state.metadata = InvocationMetadata.model_validate(
                {**state.metadata.model_dump(), "closed": True}
            )

    def __repr__(self) -> str:
        state = _invocations.get(self)
        return (
            "PreparedInvocation(unissued)"
            if state is None
            else f"PreparedInvocation(invocation_id={state.metadata.invocation_id!r}, attempts={len(state.metadata.attempts)}, active={state.active is not None}, closed={state.metadata.closed})"
        )


@dataclass(slots=True, repr=False)
class _Exchange:
    response: LocalByteResponse
    metadata: AttemptMetadata


_exchanges: weakref.WeakKeyDictionary[CapturedExchange, _Exchange] = weakref.WeakKeyDictionary()


class CapturedExchange(_Opaque):
    """Opaque successful working response plus a body-free metadata record."""

    __slots__ = ()

    def __init__(self) -> None:
        _fail("capture_invalid")

    @property
    def attempt(self) -> AttemptMetadata:
        return AttemptMetadata.model_validate(_exchange(self).metadata)

    @property
    def status(self) -> int | None:
        return _exchange(self).response.status

    def response_body(self) -> bytes:
        return _exchange(self).response.response_body()

    def __repr__(self) -> str:
        state = _exchanges.get(self)
        return (
            "CapturedExchange(unissued)"
            if state is None
            else f"CapturedExchange(attempt_id={state.metadata.attempt_id!r}, status={state.response.status!r})"
        )


def _exchange(value: object) -> _Exchange:
    if type(value) is not CapturedExchange:
        _fail("capture_invalid")
    state = _exchanges.get(value)
    if state is None:
        _fail("capture_invalid")
    return state


async def prepare_invocation(
    build: ConsumedBuild,
    values: object,
    *,
    retention: Literal["metadata_only"] = "metadata_only",
) -> PreparedInvocation:
    """Revalidate provided bytes and render locally, without fetching or dispatching anything."""
    if type(retention) is not str or retention != "metadata_only":
        _fail("capture_invalid")
    rendered = await render_consumed(build, values)
    wire = rendered.wire_payload()
    output = cast("dict[str, PortableValue]", wire["output"])
    kind = cast("Literal['chat', 'artifact']", output["kind"])
    count = len(cast("list[PortableValue]", output["messages"])) if kind == "chat" else 1
    metadata = InvocationMetadata(
        domain="strata-prompt-invocation-v1",
        version=1,
        invocation_id=_identifier(),
        verified_at=_now(),
        source_kind="caller_bytes",
        source_partition=build.source.partition,
        source_reference=build.source.reference,
        build_id=build.build_id,
        build_size_bytes=build.size_bytes,
        render_id=rendered.render_id,
        input_fingerprint=cast("str", wire["input_fingerprint"]),
        output_kind=kind,
        output_count=count,
        source_span_count=len(cast("list[PortableValue]", wire["source_map"])),
        renderer_profile="strata-modular-compiler/1",
        portable_profile="strata-prompt-json-v1",
        retention="metadata_only",
        attempts=(),
        closed=False,
        export_status="not_configured",
    )
    result = object.__new__(PreparedInvocation)
    _invocations[result] = _Invocation(
        metadata, rendered, asyncio.get_running_loop(), threading.get_ident()
    )
    return result


def _inputs(parents: object, baseline: object) -> tuple[AttemptParents, DeclaredBodyPin | None]:
    checked: AttemptParents | None = None
    pin: DeclaredBodyPin | None = None
    valid = False
    try:
        if type(parents) is AttemptParents and (
            baseline is None or type(baseline) is DeclaredBodyPin
        ):
            checked = AttemptParents.model_validate(parents)
            pin = None if baseline is None else DeclaredBodyPin.model_validate(baseline)
            valid = True
    except ValidationError, ValueError, TypeError:
        valid = False
    if not valid or checked is None:
        _fail("capture_invalid")
    return checked, pin


async def _delegate(ticket: _Ticket, transport: LocalByteTransport, body: bytes) -> object:
    if ticket.retired_at is not None:
        raise asyncio.CancelledError
    ticket.handed = True
    return await transport.exchange(body)


async def _drain(child: asyncio.Task[object]) -> None:
    # Repeat parent cancellation must never cancel the same owned child again or release its slot.
    while not child.done():
        try:
            await asyncio.shield(child)
        except BaseException:
            if child.done():
                break
    # Consume retired errors without logging content; the owner's cancellation still propagates.
    with suppress(BaseException):
        child.result()


def _retire(ticket: _Ticket) -> None:
    if ticket.retired_at is None:
        ticket.retired_at = _now()
    if ticket.child is not None and not ticket.child.done() and not ticket.cancel_sent:
        ticket.cancel_sent = True
        ticket.child.cancel()


def _record(
    state: _Invocation,
    ticket: _Ticket,
    *,
    outcome: Literal[
        "observed_response", "transport_error", "capture_error", "cancelled", "interrupted"
    ],
    error: ErrorCode | None,
    response: ByteObservation | None = None,
    status: int | None = None,
    late: bool = False,
) -> AttemptMetadata:
    if state.active is not ticket or asyncio.current_task() is not ticket.owner:
        _fail("capture_scope_invalid")
    request = ticket.request if ticket.handed else None
    relation: Literal["not_supplied", "not_observed", "same_bytes", "changed_bytes"] = (
        "not_supplied" if ticket.baseline is None else "not_observed"
    )
    if ticket.baseline is not None and request is not None:
        relation = (
            "same_bytes"
            if (ticket.baseline.sha256, ticket.baseline.size_bytes)
            == (request.original_sha256, request.size_bytes)
            else "changed_bytes"
        )
    gaps: list[Gap] = [
        "render_projection_unobserved",
        "provider_boundary_unobserved",
        "body_not_retained",
    ]
    if request is None:
        gaps.append("request_not_dispatched")
    if response is None:
        gaps.append("response_not_observed")
    elif response.coverage == "omitted_oversize":
        gaps.append("response_oversize")
    if late:
        gaps.append("late_return_discarded")
    meta = state.metadata
    record = AttemptMetadata(
        domain="strata-prompt-attempt-v1",
        version=1,
        invocation_id=meta.invocation_id,
        attempt_id=ticket.attempt_id,
        source_kind="caller_bytes",
        build_id=meta.build_id,
        render_id=meta.render_id,
        input_fingerprint=meta.input_fingerprint,
        parents=ticket.parents,
        boundary="sdk_local_delegate_bytes/v1",
        observation_source="instrumented_local",
        started_at=ticket.started_at,
        retired_at=ticket.retired_at,
        ended_at=_now(),
        duration_us=(time.monotonic_ns() - ticket.start_ns) // 1000,
        facts=AttemptFacts(
            retrieved=True,
            rendered=True,
            request_captured=request is not None,
            response_observed=response is not None,
        ),
        request=request,
        response=response,
        delegate_status=status,
        declared_baseline=ticket.baseline,
        baseline_relation=relation,
        render_relation="unknown",
        provider=None,
        model=None,
        model_revision=None,
        usage=None,
        cost=None,
        outcome=outcome,
        error_code=error,
        retention="metadata_only",
        export_status="not_configured",
        gaps=tuple(gaps),
    )
    state.metadata = InvocationMetadata.model_validate(
        {**meta.model_dump(), "attempts": (*meta.attempts, record)}
    )
    state.active = None
    return record


async def capture_once(
    invocation: PreparedInvocation,
    body: bytes,
    transport: LocalByteTransport,
    *,
    parents: AttemptParents,
    declared_baseline: DeclaredBodyPin | None,
) -> CapturedExchange:
    """Observe one local byte call, with no routing, retry, credentials or provider assertion."""
    state = _state(invocation, loop=True)
    if state.metadata.closed:
        _fail("capture_closed")
    if state.active is not None:
        _fail("capture_in_progress")
    owner = cast("asyncio.Task[object] | None", asyncio.current_task())
    if owner is None:
        _fail("capture_scope_invalid")
    if owner.cancelling():
        raise asyncio.CancelledError
    if type(body) is not bytes:
        _fail("capture_invalid")
    if len(body) > MAX_BODY_BYTES:
        _fail("capture_limit")
    checked, baseline = _inputs(parents, declared_baseline)
    known = {item.attempt_id for item in state.metadata.attempts}
    if any(
        parent is not None and parent not in known
        for parent in (checked.retry_of, checked.fallback_of, checked.continuation_of)
    ):
        _fail("capture_invalid")
    if (
        len(known) >= MAX_ATTEMPTS
        or len(state.metadata.canonical()) + MAX_ATTEMPT_METADATA_BYTES
        > MAX_INVOCATION_METADATA_BYTES
    ):
        _fail("capture_limit")
    identifier = _identifier()
    if identifier in known:
        _fail("capture_invalid")
    ticket = _Ticket(
        identifier, owner, _now(), time.monotonic_ns(), checked, baseline, _observation(body)
    )
    state.active = ticket
    # The only child owns the only application-delegate invocation. No transport is installed.
    child = asyncio.create_task(_delegate(ticket, transport, body))
    ticket.child = child
    raw: object = None
    error: ErrorCode | None = None
    interrupted: BaseException | None = None
    cancelled = False
    try:
        raw = await asyncio.shield(child)
    except asyncio.CancelledError:
        cancelled = True
        _retire(ticket)
    except TimeoutError:
        error = "local_transport_timeout"
    except Exception:
        error = "local_transport_failed"
    except BaseException as control:
        interrupted = control
        _retire(ticket)
    if cancelled or interrupted is not None:
        await _drain(child)
        late = child.done() and not child.cancelled() and child.exception() is None
        _record(
            state,
            ticket,
            outcome="cancelled" if cancelled else "interrupted",
            error="local_cancelled" if cancelled else "local_interrupted",
            late=late,
        )
        raw = None
        if cancelled:
            raise asyncio.CancelledError
        if interrupted is not None:
            raise interrupted
    if error is not None:
        _record(state, ticket, outcome="transport_error", error=error)
        _fail(error, identifier)
    response: _Response | None = None
    invalid = False
    try:
        response = _response(raw)
    except PromptCaptureError:
        invalid = True
    if invalid or response is None:
        _record(state, ticket, outcome="capture_error", error="capture_response_invalid")
        _fail("capture_response_invalid", identifier)
    if len(response.body) > MAX_BODY_BYTES:
        oversized = ByteObservation(
            profile="opaque-application-bytes/v1",
            coverage="omitted_oversize",
            size_bytes=len(response.body),
            original_sha256=None,
            redacted_sha256=None,
            retained=False,
        )
        _record(
            state,
            ticket,
            outcome="capture_error",
            error="capture_response_limit",
            response=oversized,
            status=response.status,
        )
        _fail("capture_response_limit", identifier)
    record = _record(
        state,
        ticket,
        outcome="observed_response",
        error=None,
        response=_observation(response.body),
        status=response.status,
    )
    result = object.__new__(CapturedExchange)
    _exchanges[result] = _Exchange(cast("LocalByteResponse", raw), record)
    return result
