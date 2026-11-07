"""Opt-in fail-open ProviderClient observation; execution and credentials remain the delegate's."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NoReturn,
    cast,
)  # Any preserves the existing raw ProviderClient boundary.

from strata_forge.core.errors import ForgeError
from strata_forge.llm.providers.base import ProviderClient
from strata_forge.tracing.prompt_attempts import PreparedInvocation
from strata_forge.tracing.prompt_provider_types import (
    BASE_PROVIDER_GAPS,
    ISSUE_ORDER,
    MAX_PROVIDER_ATTEMPT_BYTES,
    MAX_PROVIDER_ATTEMPTS,
    MAX_PROVIDER_OBSERVATION_BYTES,
    MAX_PROVIDER_SAFE_INTEGER,
    ProviderAttemptMetadata,
    ProviderCaptureIssue,
    ProviderGap,
    ProviderObservationCoverage,
    ProviderObservationLoss,
    ProviderObservationMetadata,
    ProviderProjectionMetadata,
    unavailable_projection,
)
from strata_forge.tracing.prompt_provider_values import (
    project_chat,
    project_response,
    project_responses,
)

if TYPE_CHECKING:
    from _thread import LockType
    from collections.abc import AsyncIterator

    from strata_forge.llm.registry import Model, ProviderName
    from strata_forge.tracing.prompt_provider_types import ProviderMethod, ProviderSlot

_HEADER_RESERVATION = 4096
_SLOTS = frozenset({"anthropic", "openai", "vertex", "bedrock", "azure", "openai_compat"})
type ObservationCode = Literal[
    "observation_invalid",
    "observation_in_progress",
    "observation_closed",
    "observation_scope_invalid",
]
type _Outcome = Literal["returned", "raised", "cancelled", "interrupted"]


class ProviderObservationError(ForgeError):
    """Fixed explicit observer-management failure, never substituted for a provider error."""

    def __init__(self, code: ObservationCode) -> None:
        self.code = (
            code
            if type(code) is str
            and code
            in {
                "observation_invalid",
                "observation_in_progress",
                "observation_closed",
                "observation_scope_invalid",
            }
            else "observation_invalid"
        )
        super().__init__(self.code)


def _fail(code: ObservationCode = "observation_invalid") -> NoReturn:
    raise ProviderObservationError(code)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _id() -> str:
    return uuid.uuid4().hex


@dataclass(slots=True, repr=False)
class _Ticket:
    identifier: str
    index: int
    owner: asyncio.Task[object]
    cancelling: int
    method: ProviderMethod
    provider_slot: ProviderSlot
    model_label: str | None
    started_at: str
    start_ns: int
    request: ProviderProjectionMetadata | None = None
    projection_failed: bool = False


@dataclass(slots=True, repr=False)
class _State:
    initial: ProviderObservationMetadata
    loop: asyncio.AbstractEventLoop
    thread: int
    lock: LockType = field(default_factory=threading.Lock)
    active: dict[int, _Ticket] = field(default_factory=lambda: dict[int, _Ticket]())
    finished: dict[int, ProviderAttemptMetadata] = field(
        default_factory=lambda: dict[int, ProviderAttemptMetadata]()
    )
    admitted: int = 0
    finished_bytes: int = 0
    closed: bool = False
    over_capacity_count: int = 0
    projection_limited_count: int = 0
    metadata_lost_count: int = 0
    scope_mismatch_seen: bool = False
    counter_saturated: bool = False


_states: weakref.WeakKeyDictionary[ProviderObservation, _State] = weakref.WeakKeyDictionary()


def _state(value: object, *, explicit: bool = False) -> _State:
    if type(value) is not ProviderObservation:
        _fail()
    result = _states.get(value)
    if result is None:
        _fail()
    if explicit and result.thread != threading.get_ident():
        _fail("observation_scope_invalid")
    return result


class ProviderObservation:
    """Issued body-free local observer; source labels confer no provider or native authority."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        _fail()

    def snapshot(self) -> ProviderObservationMetadata:
        state = _state(self, explicit=True)
        busy = False
        with state.lock:
            if state.active:
                busy = True
                fields: dict[str, object] = {}
            else:
                fields = {
                    **state.initial.model_dump(),
                    "closed": state.closed,
                    "admitted_attempts": state.admitted,
                    "attempts": tuple(state.finished[index] for index in sorted(state.finished)),
                    "loss": {
                        "over_capacity_count": state.over_capacity_count,
                        "projection_limited_count": state.projection_limited_count,
                        "metadata_lost_count": state.metadata_lost_count,
                        "scope_mismatch_seen": state.scope_mismatch_seen,
                        "counter_saturated": state.counter_saturated,
                    },
                }
        if busy:
            _fail("observation_in_progress")
        result: ProviderObservationMetadata | None = None
        try:
            result = ProviderObservationMetadata.model_validate(fields)
        except Exception:
            result = None
        if result is None:
            _fail()
        return result

    def close(self) -> None:
        state = _state(self, explicit=True)
        with state.lock:
            busy = bool(state.active)
            if not busy:
                state.closed = True
        if busy:
            _fail("observation_in_progress")

    def __repr__(self) -> str:
        return (
            "ProviderObservation(local_metadata_only)"
            if self in _states
            else "ProviderObservation(unissued)"
        )

    def __reduce__(self) -> NoReturn:
        _fail()

    def __copy__(self) -> NoReturn:
        _fail()

    def __deepcopy__(self, _memo: object) -> NoReturn:
        _fail()


async def prepare_provider_observation(invocation: PreparedInvocation) -> ProviderObservation:
    """Use the exact issued public handle API, without trusting a parsed DTO or reading content."""
    snapshot = None
    try:
        if type(invocation) is PreparedInvocation:
            snapshot = PreparedInvocation.snapshot(invocation)
    except Exception:
        snapshot = None
    if snapshot is None:
        _fail()
    if snapshot.closed:
        _fail("observation_closed")
    initial = ProviderObservationMetadata(
        domain="strata-provider-prompt-observation-v1",
        version=1,
        observation_id=_id(),
        invocation_id=snapshot.invocation_id,
        source_kind="caller_bytes",
        source_partition=snapshot.source_partition,
        source_reference=snapshot.source_reference,
        build_id=snapshot.build_id,
        render_id=snapshot.render_id,
        input_fingerprint=snapshot.input_fingerprint,
        created_at=_now(),
        closed=False,
        admitted_attempts=0,
        attempts=(),
        loss=ProviderObservationLoss(
            over_capacity_count=0,
            projection_limited_count=0,
            metadata_lost_count=0,
            scope_mismatch_seen=False,
            counter_saturated=False,
        ),
        coverage=ProviderObservationCoverage(
            all_calls_observed=False,
            streaming_observed=False,
            cache_events_observed=False,
            post_close_calls_counted=False,
        ),
        retention="metadata_only",
        export_status="not_configured",
    )
    if len(initial.canonical()) > _HEADER_RESERVATION:
        _fail()
    result = object.__new__(ProviderObservation)
    _states[result] = _State(initial, asyncio.get_running_loop(), threading.get_ident())
    return result


def _increment(
    state: _State,
    name: Literal["over_capacity_count", "projection_limited_count", "metadata_lost_count"],
) -> None:
    value = getattr(state, name)
    if value == MAX_PROVIDER_SAFE_INTEGER:
        state.counter_saturated = True
    else:
        setattr(state, name, value + 1)


def _label(model: object) -> str | None:
    return (
        model
        if type(model) is str
        and 1 <= len(model) <= 512
        and all(0x21 <= ord(char) <= 0x7E for char in model)
        else None
    )


def _begin(
    state: _State, method: ProviderMethod, slot: ProviderSlot, model: object
) -> _Ticket | None:
    owner = cast("asyncio.Task[object] | None", asyncio.current_task())
    with state.lock:
        if state.closed:
            return None
        if (
            state.thread != threading.get_ident()
            or state.loop is not asyncio.get_running_loop()
            or owner is None
        ):
            state.scope_mismatch_seen = True
            return None
        if (
            state.admitted >= MAX_PROVIDER_ATTEMPTS
            or _HEADER_RESERVATION
            + state.finished_bytes
            + (len(state.active) + 1) * MAX_PROVIDER_ATTEMPT_BYTES
            > MAX_PROVIDER_OBSERVATION_BYTES
        ):
            _increment(state, "over_capacity_count")
            return None
        ticket = _Ticket(
            _id(),
            state.admitted,
            owner,
            owner.cancelling(),
            method,
            slot,
            _label(model),
            _now(),
            time.monotonic_ns(),
        )
        state.active[ticket.index] = ticket
        state.admitted += 1
        return ticket


def _start_loss(state: _State) -> None:
    with state.lock:
        if state.closed:
            return
        if state.thread != threading.get_ident() or state.loop is not asyncio.get_running_loop():
            state.scope_mismatch_seen = True
        elif state.admitted < MAX_PROVIDER_ATTEMPTS:
            state.admitted += 1
            _increment(state, "metadata_lost_count")
        else:
            _increment(state, "over_capacity_count")


def _prepare(
    ticket: _Ticket | None, method: ProviderMethod, model: object, body: object, parameters: object
) -> None:
    if ticket is None:
        return
    try:
        ticket.request = (
            project_chat(model, body, parameters)
            if method == "acompletion"
            else project_responses(model, body)
        )
    except Exception:
        ticket.projection_failed = True


def _metadata(
    state: _State, ticket: _Ticket, outcome: _Outcome, raw: object
) -> ProviderAttemptMetadata:
    if ticket.projection_failed or ticket.request is None:
        _fail()
    retired = outcome == "returned" and ticket.owner.cancelling() > ticket.cancelling
    response = (
        unavailable_projection("not_returned")
        if outcome != "returned"
        else unavailable_projection("retired")
        if retired
        else project_response(ticket.method, raw)
    )
    request = ticket.request
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
    if ticket.model_label is None:
        issues.add("identity_unavailable")
    if retired:
        issues.add("retired")
    gaps: list[ProviderGap] = list(BASE_PROVIDER_GAPS)
    if (request.omitted_fields_count or 0) > 0 or (response.omitted_fields_count or 0) > 0:
        gaps.append("unselected_fields_omitted")
    if request.status != "observed":
        gaps.append("request_projection_unavailable")
    if response.status in {"opaque", "unsupported", "limit", "retired"}:
        gaps.append("response_projection_unavailable")
    if ticket.model_label is None:
        gaps.append("identity_unavailable")
    if retired:
        gaps.append("observer_retired")
    initial = state.initial
    return ProviderAttemptMetadata(
        domain="strata-provider-prompt-attempt-v1",
        version=1,
        observation_id=initial.observation_id,
        attempt_id=ticket.identifier,
        entry_index=ticket.index,
        invocation_id=initial.invocation_id,
        build_id=initial.build_id,
        render_id=initial.render_id,
        input_fingerprint=initial.input_fingerprint,
        source_kind="caller_bytes",
        method=ticket.method,
        declared_provider_slot=ticket.provider_slot,
        observed_provider_model_id=ticket.model_label,
        logical_model=None,
        served_model=None,
        model_revision=None,
        source_relation="unverified_argument",
        started_at=ticket.started_at,
        ended_at=_now(),
        duration_us=(time.monotonic_ns() - ticket.start_ns) // 1000,
        call_outcome=outcome,
        request_entered=True,
        return_observed=outcome == "returned" and not retired,
        error_category=None
        if outcome == "returned"
        else "delegate_error"
        if outcome == "raised"
        else "cancelled"
        if outcome == "cancelled"
        else "interrupted",
        request_projection=request,
        response_projection=response,
        capture_status="partial" if issues else "recorded",
        capture_issues=tuple(issue for issue in ISSUE_ORDER if issue in issues),
        parents=None,
        relation="unknown",
        render_relation="unknown",
        usage=None,
        cost=None,
        retention="metadata_only",
        export_status="not_configured",
        gaps=tuple(gaps),
    )


def _finish(state: _State, ticket: _Ticket | None, outcome: _Outcome, raw: object = None) -> None:
    if ticket is None:
        return
    record: ProviderAttemptMetadata | None = None
    size = 0
    try:
        if asyncio.current_task() is ticket.owner:
            record = _metadata(state, ticket, outcome, raw)
            size = len(record.canonical()) + 1
    except Exception:
        record = None
    with state.lock:
        if state.active.get(ticket.index) is not ticket:
            return
        del state.active[ticket.index]
        if record is None:
            _increment(state, "metadata_lost_count")
        else:
            state.finished[ticket.index] = record
            state.finished_bytes += size
            if record.request_projection.status == "limit":
                _increment(state, "projection_limited_count")
            if record.response_projection.status == "limit":
                _increment(state, "projection_limited_count")


def _finish_safely(
    state: _State, ticket: _Ticket | None, outcome: _Outcome, raw: object = None
) -> None:
    try:
        _finish(state, ticket, outcome, raw)
    except Exception:
        # Terminal observer failure cannot veto the original result or exception.
        if ticket is not None:
            with state.lock:
                if state.active.get(ticket.index) is ticket:
                    del state.active[ticket.index]
                    _increment(state, "metadata_lost_count")


@dataclass(slots=True, repr=False)
class _Wrapper:
    delegate: ProviderClient
    state: _State
    provider_slot: ProviderSlot


_wrappers: weakref.WeakKeyDictionary[_ObservedProvider, _Wrapper] = weakref.WeakKeyDictionary()


def _wrapper(value: _ObservedProvider) -> _Wrapper:
    if type(value) is not _ObservedProvider:
        _fail()
    result = _wrappers.get(value)
    if result is None:
        _fail()
    return result


class _ObservedProvider(ProviderClient):
    """Fixed named-method facade; never a secret config clone or generic attribute proxy."""

    def __init__(self) -> None:
        _fail()

    def auth_kwargs(self) -> dict[str, Any]:
        return _wrapper(self).delegate.auth_kwargs()

    def responses_auth_kwargs(self) -> dict[str, Any]:
        return _wrapper(self).delegate.responses_auth_kwargs()

    def litellm_model(self, provider_model_id: str) -> str:
        return _wrapper(self).delegate.litellm_model(provider_model_id)

    def describe_to_litellm(self, provider_model_id: str, model: Model) -> None:
        _wrapper(self).delegate.describe_to_litellm(provider_model_id, model)

    def astream(
        self, *, provider_model_id: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[Any]:
        return _wrapper(self).delegate.astream(
            provider_model_id=provider_model_id, messages=messages, **kwargs
        )

    def aresponses_stream(
        self, *, provider_model_id: str, request: dict[str, Any]
    ) -> AsyncIterator[Any]:
        return _wrapper(self).delegate.aresponses_stream(
            provider_model_id=provider_model_id, request=request
        )

    async def acompletion(
        self, *, provider_model_id: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        wrapper = _wrapper(self)
        operation = wrapper.delegate.acompletion
        ticket: _Ticket | None = None
        try:
            if callable(operation):
                ticket = _begin(
                    wrapper.state, "acompletion", wrapper.provider_slot, provider_model_id
                )
            _prepare(ticket, "acompletion", provider_model_id, messages, kwargs)
        except Exception:
            if ticket is not None:
                ticket.projection_failed = True
            else:
                _start_loss(wrapper.state)
        try:
            raw = await operation(provider_model_id=provider_model_id, messages=messages, **kwargs)
        except asyncio.CancelledError:
            _finish_safely(wrapper.state, ticket, "cancelled")
            raise
        except Exception:
            _finish_safely(wrapper.state, ticket, "raised")
            raise
        except BaseException:
            _finish_safely(wrapper.state, ticket, "interrupted")
            raise
        _finish_safely(wrapper.state, ticket, "returned", raw)
        return raw

    async def aresponses(self, *, provider_model_id: str, request: dict[str, Any]) -> Any:
        wrapper = _wrapper(self)
        operation = wrapper.delegate.aresponses
        ticket: _Ticket | None = None
        try:
            if callable(operation):
                ticket = _begin(
                    wrapper.state, "aresponses", wrapper.provider_slot, provider_model_id
                )
            _prepare(ticket, "aresponses", provider_model_id, request, None)
        except Exception:
            if ticket is not None:
                ticket.projection_failed = True
            else:
                _start_loss(wrapper.state)
        try:
            raw = await operation(provider_model_id=provider_model_id, request=request)
        except asyncio.CancelledError:
            _finish_safely(wrapper.state, ticket, "cancelled")
            raise
        except Exception:
            _finish_safely(wrapper.state, ticket, "raised")
            raise
        except BaseException:
            _finish_safely(wrapper.state, ticket, "interrupted")
            raise
        _finish_safely(wrapper.state, ticket, "returned", raw)
        return raw

    def __repr__(self) -> str:
        return "ObservedProvider(local_projection_only)"

    def __reduce__(self) -> NoReturn:
        _fail()

    def __copy__(self) -> NoReturn:
        _fail()

    def __deepcopy__(self, _memo: object) -> NoReturn:
        _fail()


def _valid_delegate(value: object) -> bool:
    return isinstance(value, ProviderClient)


def wrap_provider(
    delegate: ProviderClient, *, observation: ProviderObservation, provider_slot: ProviderName
) -> ProviderClient:
    """Explicitly inject a fixed facade; do not inspect or copy delegate configuration."""
    state = _state(observation, explicit=True)
    if (
        not _valid_delegate(delegate)
        or type(provider_slot) is not str
        or provider_slot not in _SLOTS
    ):
        _fail()
    with state.lock:
        closed = state.closed
    if closed:
        _fail("observation_closed")
    result = object.__new__(_ObservedProvider)
    # ProviderClient annotates name as ClassVar; this facade's value is only the declared slot.
    object.__setattr__(result, "name", provider_slot)
    _wrappers[result] = _Wrapper(delegate, state, cast("ProviderSlot", provider_slot))
    return result
