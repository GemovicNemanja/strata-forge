"""Explicit native-generator protocol observation; no source driving or item inspection."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
import weakref
from collections.abc import Coroutine
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import AsyncGeneratorType
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Protocol, Self, cast

from strata_forge.core.errors import ForgeError
from strata_forge.llm.providers.base import ProviderClient
from strata_forge.tracing.prompt_attempts import PreparedInvocation
from strata_forge.tracing.prompt_provider_values import project_chat, project_responses
from strata_forge.tracing.prompt_stream_types import (
    BASE_STREAM_GAPS,
    MAX_STREAM_BYTES,
    MAX_STREAM_OBSERVATION_BYTES,
    MAX_STREAM_SAFE_INTEGER,
    MAX_STREAMS,
    ProviderStreamMetadata,
    ProviderStreamObservationMetadata,
    StreamInputBinding,
    StreamObservationCoverage,
    StreamObservationLoss,
)

if TYPE_CHECKING:
    from _thread import LockType
    from collections.abc import AsyncGenerator, AsyncIterator, Generator

    from strata_forge.llm.registry import Model, ProviderName
    from strata_forge.tracing.prompt_provider_types import ProviderSlot
    from strata_forge.tracing.prompt_stream_types import (
        StreamAction,
        StreamGap,
        StreamMethod,
        StreamOperation,
        StreamOutcome,
    )

# Any forwards the existing raw ProviderClient interface/native throw argument forms only.
type StreamErrorCode = Literal["stream_observation_invalid", "stream_observation_scope_invalid"]
_SLOTS = frozenset({"anthropic", "openai", "vertex", "bedrock", "azure", "openai_compat"})
_HEADER = 4096


class StreamObservationError(ForgeError):
    """Fixed explicit local management error, never a replacement native outcome."""

    def __init__(self, code: StreamErrorCode = "stream_observation_invalid") -> None:
        self.code = (
            code if code == "stream_observation_scope_invalid" else "stream_observation_invalid"
        )
        super().__init__(self.code)


def _fail(code: StreamErrorCode = "stream_observation_invalid") -> NoReturn:
    raise StreamObservationError(code)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _gaps(fields: dict[str, object]) -> tuple[StreamGap, ...]:
    values: list[StreamGap] = list(BASE_STREAM_GAPS)
    binding = cast("StreamInputBinding", fields["input_binding"])
    if binding.stage == "unobserved":
        values.append("input_unobserved")
    elif binding.projection is not None and binding.projection.status != "observed":
        values.append("input_projection_unavailable")
    if fields["end_observation"] is None:
        values.append("iterator_end_unconfirmed")
    if fields["capture_state"] == "retired":
        values.append("observer_retired")
    if fields["observed_provider_model_id"] is None:
        values.append("identity_unavailable")
    return tuple(values)


def _fields(record: ProviderStreamMetadata) -> dict[str, object]:
    result = record.model_dump()
    # Keep the exact closed model available to fixed gap derivation, not a caller object.
    result["input_binding"] = record.input_binding
    return result


@dataclass(slots=True, repr=False)
class _ObservationState:
    initial: ProviderStreamObservationMetadata
    loop: weakref.ReferenceType[asyncio.AbstractEventLoop]
    thread: int
    lock: LockType = field(default_factory=threading.Lock)
    records: dict[int, ProviderStreamMetadata] = field(
        default_factory=dict[int, ProviderStreamMetadata]
    )
    views: list[weakref.ReferenceType[_IteratorView]] = field(
        default_factory=lambda: list[weakref.ReferenceType[_IteratorView]]()
    )
    admitted: int = 0
    closed: bool = False
    over_capacity_count: int = 0
    unsupported_iterator_count: int = 0
    factory_failure_count: int = 0
    metadata_lost_count: int = 0
    scope_mismatch_seen: bool = False
    counter_saturated: bool = False


_observations: weakref.WeakSet[ProviderStreamObservation] = weakref.WeakSet()
_iterators: weakref.WeakSet[_IteratorView] = weakref.WeakSet()
_advances: weakref.WeakSet[_AdvanceView] = weakref.WeakSet()
_wrappers: weakref.WeakSet[_StreamingProvider] = weakref.WeakSet()


def _observation(value: object, *, explicit: bool = False) -> _ObservationState:
    if type(value) is not ProviderStreamObservation or value not in _observations:
        _fail()
    state = cast("_ObservationState", object.__getattribute__(value, "_state"))
    if explicit and state.thread != threading.get_ident():
        _fail("stream_observation_scope_invalid")
    return state


def _scope(state: _ObservationState) -> bool:
    if state.thread != threading.get_ident():
        return False
    try:
        return state.loop() is asyncio.get_running_loop()
    except RuntimeError:
        return False


def _increment(state: _ObservationState, name: str) -> None:
    value = cast("int", getattr(state, name))
    if value == MAX_STREAM_SAFE_INTEGER:
        state.counter_saturated = True
    else:
        setattr(state, name, value + 1)


def _loss(state: _ObservationState, name: str) -> None:
    with state.lock:
        if not state.closed:
            if _scope(state):
                _increment(state, name)
            else:
                state.scope_mismatch_seen = True


def _safe_loss(state: _ObservationState, name: str) -> None:
    # Fixed optional metadata loss cannot veto the original factory/protocol call.
    with suppress(Exception):
        _loss(state, name)


class ProviderStreamObservation:
    """Issued partial metadata owner; close retires capture without closing native work."""

    __slots__ = ("__weakref__", "_state")

    def __init__(self) -> None:
        _fail()

    def snapshot(self) -> ProviderStreamObservationMetadata:
        state = _observation(self, explicit=True)
        with state.lock:
            fields = {
                **state.initial.model_dump(),
                "closed": state.closed,
                "admitted_streams": state.admitted,
                "streams": tuple(state.records[index] for index in sorted(state.records)),
                "loss": {
                    "over_capacity_count": state.over_capacity_count,
                    "unsupported_iterator_count": state.unsupported_iterator_count,
                    "factory_failure_count": state.factory_failure_count,
                    "metadata_lost_count": state.metadata_lost_count,
                    "scope_mismatch_seen": state.scope_mismatch_seen,
                    "counter_saturated": state.counter_saturated,
                },
            }
        result = None
        try:
            result = ProviderStreamObservationMetadata.model_validate(fields)
        except Exception:
            result = None
        if result is None:
            _fail()
        return result

    def close(self) -> None:
        state = _observation(self, explicit=True)
        with state.lock:
            if state.closed:
                return
            state.closed = True
            records = tuple(state.records.items())
            references = tuple(state.views)
        for reference in references:
            view = reference()
            if type(view) is _IteratorView and view in _iterators:
                work = cast("_IteratorWork", object.__getattribute__(view, "_work"))
                object.__setattr__(work, "arguments", None)
        for index, record in records:
            try:
                fields = _fields(record)
                fields.update(
                    capture_state="retired", retired_at=_now(), last_observation_at=_now()
                )
                fields["gaps"] = _gaps(fields)
                retired = ProviderStreamMetadata.model_validate(fields)
            except Exception:
                with state.lock:
                    if state.records.pop(index, None) is not None:
                        _increment(state, "metadata_lost_count")
            else:
                with state.lock:
                    state.records[index] = retired

    def __repr__(self) -> str:
        return "ProviderStreamObservation(local_counts_only)"

    def __reduce__(self) -> NoReturn:
        _fail()

    def __copy__(self) -> NoReturn:
        _fail()

    def __deepcopy__(self, _memo: object) -> NoReturn:
        _fail()


async def prepare_stream_observation(invocation: PreparedInvocation) -> ProviderStreamObservation:
    snapshot = None
    try:
        if type(invocation) is PreparedInvocation:
            snapshot = PreparedInvocation.snapshot(invocation)
    except Exception:
        snapshot = None
    if snapshot is None or snapshot.closed:
        _fail()
    initial = ProviderStreamObservationMetadata(
        domain="strata-provider-stream-observation-v1",
        version=1,
        observation_id=uuid.uuid4().hex,
        invocation_id=snapshot.invocation_id,
        source_kind="caller_bytes",
        source_partition=snapshot.source_partition,
        source_reference=snapshot.source_reference,
        build_id=snapshot.build_id,
        render_id=snapshot.render_id,
        input_fingerprint=snapshot.input_fingerprint,
        created_at=_now(),
        closed=False,
        streams=(),
        admitted_streams=0,
        loss=StreamObservationLoss(
            over_capacity_count=0,
            unsupported_iterator_count=0,
            factory_failure_count=0,
            metadata_lost_count=0,
            scope_mismatch_seen=False,
            counter_saturated=False,
        ),
        coverage=StreamObservationCoverage(
            all_streams_observed=False,
            provider_execution_verified=False,
            item_content_observed=False,
            cache_events_observed=False,
            causal_links_verified=False,
        ),
        retention="metadata_only",
        export_status="not_configured",
    )
    if len(initial.canonical()) > _HEADER:
        _fail()
    handle = object.__new__(ProviderStreamObservation)
    object.__setattr__(
        handle,
        "_state",
        _ObservationState(
            initial,
            weakref.ref(asyncio.get_running_loop()),
            threading.get_ident(),
        ),
    )
    _observations.add(handle)
    return handle


@dataclass(slots=True, repr=False)
class _IteratorWork:
    native: AsyncGenerator[object, object]
    observation: _ObservationState
    index: int
    # Extra references exist only until the first projection attempt or retirement.
    arguments: tuple[object, object, object] | None


def _iterator(value: _IteratorView) -> _IteratorWork:
    if type(value) is not _IteratorView or value not in _iterators:
        _fail()
    return cast("_IteratorWork", object.__getattribute__(value, "_work"))


def _record(work: _IteratorWork) -> ProviderStreamMetadata | None:
    state = work.observation
    with state.lock:
        if state.closed:
            return None
        if not _scope(state):
            state.scope_mismatch_seen = True
            return None
        return state.records.get(work.index)


def _replace(
    work: _IteratorWork, record: ProviderStreamMetadata, fields: dict[str, object]
) -> None:
    fields["gaps"] = _gaps(fields)
    new = ProviderStreamMetadata.model_validate(fields)
    state = work.observation
    with state.lock:
        if not state.closed and state.records.get(work.index) is record:
            state.records[work.index] = new


def _lost(work: _IteratorWork) -> None:
    state = work.observation
    with state.lock:
        if not state.closed and state.records.pop(work.index, None) is not None:
            _increment(state, "metadata_lost_count")
    work.arguments = None


def _count(fields: dict[str, object], name: str) -> None:
    old = cast("int", fields[name])
    if old < MAX_STREAM_SAFE_INTEGER:
        fields[name] = old + 1
    else:
        fields["counter_saturated"] = True


class _NativeAdvance(Protocol):
    """Only an operation just returned by an exact native generator is stored here."""

    def __await__(self) -> object: ...
    def __iter__(self) -> object: ...
    def __next__(self) -> object: ...
    def send(self, value: object) -> object: ...
    def throw(self, *args: Any) -> object: ...
    def close(self) -> object: ...


@dataclass(slots=True, repr=False)
class _AdvanceWork:
    native: _NativeAdvance
    iterator: _IteratorView
    operation: StreamOperation
    started: bool = False
    start_ns: int = 0


def _advance(value: _AdvanceView) -> _AdvanceWork:
    if type(value) is not _AdvanceView or value not in _advances:
        _fail()
    return cast("_AdvanceWork", object.__getattribute__(value, "_work"))


def _note_start(advance: _AdvanceWork, action: StreamAction) -> None:
    if advance.started or action == "close":
        return
    advance.started = True
    advance.start_ns = time.monotonic_ns()
    work = _iterator(advance.iterator)
    record = _record(work)
    if record is None:
        work.arguments = None
        return
    fields = _fields(record)
    if advance.operation == "aclose":
        _count(fields, "close_attempts")
    else:
        _count(fields, "data_advance_attempts")
        if record.first_drive_attempt_at is None:
            now = _now()
            arguments = work.arguments
            try:
                if arguments is None:
                    _fail()
                model, body, parameters = arguments
                projection = (
                    project_chat(model, body, parameters)
                    if record.method == "astream"
                    else project_responses(model, body)
                )
                fields["input_binding"] = StreamInputBinding(
                    domain="strata-provider-stream-input-v1",
                    version=1,
                    actual_method=record.method,
                    normalized_schema="acompletion" if record.method == "astream" else "aresponses",
                    stage="before_first_drive_attempt",
                    observed_at=now,
                    projection=projection,
                )
                fields["first_drive_attempt_at"] = now
            finally:
                work.arguments = None
    _replace(work, record, fields)


def _note_result(advance: _AdvanceWork, action: StreamAction, outcome: StreamOutcome) -> None:
    work = _iterator(advance.iterator)
    record = _record(work)
    if record is None:
        work.arguments = None
        return
    fields = _fields(record)
    now = _now()
    fields.update(
        last_operation=advance.operation,
        last_protocol_action=action,
        last_operation_outcome=outcome,
        last_observation_at=now,
    )
    if outcome == "item":
        _count(fields, "raw_items_observed")
        if record.first_item_at is None:
            fields["first_item_at"] = now
            elapsed = (time.monotonic_ns() - advance.start_ns) // 1000
            fields["first_item_drive_latency_us"] = min(MAX_STREAM_SAFE_INTEGER, elapsed)
            if elapsed > MAX_STREAM_SAFE_INTEGER:
                fields["counter_saturated"] = True
    if outcome in {"exhausted", "close_completed"} and record.end_observation is None:
        fields["end_observation"] = "exhausted" if outcome == "exhausted" else "aclose_completed"
        fields["closed_observed_at"] = now
    if outcome in {"raised", "cancelled", "interrupted"}:
        _count(fields, "operation_errors")
    if outcome == "cancelled":
        _count(fields, "operation_cancellations")
    _replace(work, record, fields)


def _safe_start(advance: _AdvanceWork, action: StreamAction) -> None:
    try:
        _note_start(advance, action)
    except Exception:
        _lost(_iterator(advance.iterator))


def _safe_result(advance: _AdvanceWork, action: StreamAction, outcome: StreamOutcome) -> None:
    try:
        _note_result(advance, action, outcome)
    except Exception:
        _lost(_iterator(advance.iterator))


class _AdvanceView(Coroutine[object, object, object]):
    __slots__ = ("__weakref__", "_work")

    def __init__(self) -> None:
        _fail()

    def __await__(self) -> Generator[object, None, object]:
        _advance(self).native.__await__()
        return cast("Generator[object, None, object]", self)

    def __iter__(self) -> Self:
        _advance(self).native.__iter__()
        return self

    def __next__(self) -> object:
        return self._perform("next", None, ())

    def send(self, value: object) -> object:
        return self._perform("send", value, ())

    def throw(self, *args: Any) -> object:
        return self._perform("throw", None, args)

    def close(self) -> None:
        self._perform("close", None, ())

    def _perform(self, action: StreamAction, value: object, args: tuple[object, ...]) -> object:
        work = _advance(self)
        _safe_start(work, action)
        try:
            if action == "next":
                result = next(work.native)
            elif action == "send":
                result = work.native.send(value)
            elif action == "throw":
                result = work.native.throw(*args)
            else:
                result = work.native.close()
        except StopIteration:
            _safe_result(
                work,
                action,
                "awaitable_closed"
                if action == "close"
                else "close_completed"
                if work.operation == "aclose"
                else "item",
            )
            raise
        except StopAsyncIteration:
            _safe_result(work, action, "exhausted")
            raise
        except asyncio.CancelledError:
            _safe_result(work, action, "cancelled")
            raise
        except Exception:
            _safe_result(work, action, "raised")
            raise
        except BaseException:
            _safe_result(work, action, "interrupted")
            raise
        _safe_result(work, action, "awaitable_closed" if action == "close" else "suspended")
        return result

    def __repr__(self) -> str:
        return "NativeStreamAdvance(local_counts_only)"

    def __reduce__(self) -> NoReturn:
        _fail()

    def __copy__(self) -> NoReturn:
        _fail()

    def __deepcopy__(self, _memo: object) -> NoReturn:
        _fail()


class _IteratorView:
    __slots__ = ("__weakref__", "_work")

    def __init__(self) -> None:
        _fail()

    def __aiter__(self) -> Self:
        _iterator(self)
        return self

    def __anext__(self) -> _AdvanceView:
        work = _iterator(self)
        return self._issued(work.native.__anext__(), "anext")

    def asend(self, value: object) -> _AdvanceView:
        return self._issued(_iterator(self).native.asend(value), "asend")

    def athrow(self, *args: Any) -> _AdvanceView:
        return self._issued(_iterator(self).native.athrow(*args), "athrow")

    def aclose(self) -> _AdvanceView:
        return self._issued(_iterator(self).native.aclose(), "aclose")

    def _issued(self, native: object, operation: StreamOperation) -> _AdvanceView:
        view = object.__new__(_AdvanceView)
        object.__setattr__(
            view, "_work", _AdvanceWork(cast("_NativeAdvance", native), self, operation)
        )
        _advances.add(view)
        return view

    def __repr__(self) -> str:
        return "NativeStreamIterator(local_counts_only)"

    def __reduce__(self) -> NoReturn:
        _fail()

    def __copy__(self) -> NoReturn:
        _fail()

    def __deepcopy__(self, _memo: object) -> NoReturn:
        _fail()


def _admit(
    state: _ObservationState,
    native: object,
    method: StreamMethod,
    slot: ProviderSlot,
    model: object,
    body: object,
    parameters: object,
) -> object:
    with state.lock:
        if state.closed:
            return native
        if not _scope(state):
            state.scope_mismatch_seen = True
            return native
        if type(native) is not AsyncGeneratorType:
            _increment(state, "unsupported_iterator_count")
            return native
        if (
            state.admitted == MAX_STREAMS
            or _HEADER + (state.admitted + 1) * MAX_STREAM_BYTES > MAX_STREAM_OBSERVATION_BYTES
        ):
            _increment(state, "over_capacity_count")
            return cast("object", native)
        index = state.admitted
        state.admitted += 1
    try:
        source = state.initial
        label = (
            model
            if type(model) is str
            and 1 <= len(model) <= 512
            and all(0x21 <= ord(c) <= 0x7E for c in model)
            else None
        )
        now = _now()
        fields: dict[str, object] = {
            "domain": "strata-provider-stream-v1",
            "version": 1,
            "observation_id": source.observation_id,
            "stream_id": uuid.uuid4().hex,
            "entry_index": index,
            "invocation_id": source.invocation_id,
            "build_id": source.build_id,
            "render_id": source.render_id,
            "input_fingerprint": source.input_fingerprint,
            "source_kind": "caller_bytes",
            "method": method,
            "declared_provider_slot": slot,
            "observed_provider_model_id": label,
            "logical_model": None,
            "served_model": None,
            "model_revision": None,
            "usage": None,
            "cost": None,
            "render_relation": "unknown",
            "parents": None,
            "relation": "unknown",
            "input_binding": StreamInputBinding(
                domain="strata-provider-stream-input-v1",
                version=1,
                actual_method=method,
                normalized_schema="acompletion" if method == "astream" else "aresponses",
                stage="unobserved",
                observed_at=None,
                projection=None,
            ),
            "factory_returned_at": now,
            "first_drive_attempt_at": None,
            "first_item_at": None,
            "last_observation_at": now,
            "closed_observed_at": None,
            "retired_at": None,
            "first_item_drive_latency_us": None,
            "capture_state": "observing",
            "iterator_obtained": True,
            "end_observation": None,
            "last_operation": None,
            "last_protocol_action": None,
            "last_operation_outcome": None,
            "raw_items_observed": 0,
            "data_advance_attempts": 0,
            "close_attempts": 0,
            "operation_errors": 0,
            "operation_cancellations": 0,
            "counter_saturated": False,
            "output_projection": None,
            "body_retained": False,
            "retention": "metadata_only",
            "export_status": "not_configured",
        }
        fields["gaps"] = _gaps(fields)
        record = ProviderStreamMetadata.model_validate(fields)
        view = object.__new__(_IteratorView)
        object.__setattr__(
            view,
            "_work",
            _IteratorWork(
                cast("AsyncGenerator[object, object]", native),
                state,
                index,
                (model, body, parameters),
            ),
        )
        _iterators.add(view)
        reference = weakref.ref(view)
        with state.lock:
            if state.closed:
                _increment(state, "metadata_lost_count")
                return cast("object", native)
            state.views.append(reference)
            state.records[index] = record
        return view
    except Exception:
        with state.lock:
            # A reserved entry remains lost even if publication partly succeeded.
            state.records.pop(index, None)
            _increment(state, "metadata_lost_count")
        return cast("object", native)


@dataclass(slots=True, repr=False)
class _ProviderWork:
    delegate: ProviderClient
    observation: _ObservationState
    slot: ProviderSlot


def _provider(value: _StreamingProvider) -> _ProviderWork:
    if type(value) is not _StreamingProvider or value not in _wrappers:
        _fail()
    return cast("_ProviderWork", object.__getattribute__(value, "_work"))


class _StreamingProvider(ProviderClient):
    __slots__ = ("_work",)

    def __init__(self) -> None:
        _fail()

    def auth_kwargs(self) -> dict[str, Any]:
        return _provider(self).delegate.auth_kwargs()

    def responses_auth_kwargs(self) -> dict[str, Any]:
        return _provider(self).delegate.responses_auth_kwargs()

    def litellm_model(self, provider_model_id: str) -> str:
        return _provider(self).delegate.litellm_model(provider_model_id)

    def describe_to_litellm(self, provider_model_id: str, model: Model) -> None:
        _provider(self).delegate.describe_to_litellm(provider_model_id, model)

    async def acompletion(
        self, *, provider_model_id: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        return await _provider(self).delegate.acompletion(
            provider_model_id=provider_model_id, messages=messages, **kwargs
        )

    async def aresponses(self, *, provider_model_id: str, request: dict[str, Any]) -> Any:
        return await _provider(self).delegate.aresponses(
            provider_model_id=provider_model_id, request=request
        )

    def astream(
        self, *, provider_model_id: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[Any]:
        work = _provider(self)
        try:
            native = work.delegate.astream(
                provider_model_id=provider_model_id, messages=messages, **kwargs
            )
        except BaseException:
            _safe_loss(work.observation, "factory_failure_count")
            raise
        try:
            return cast(
                "AsyncIterator[Any]",
                _admit(
                    work.observation,
                    native,
                    "astream",
                    work.slot,
                    provider_model_id,
                    messages,
                    kwargs,
                ),
            )
        except Exception:
            return native

    def aresponses_stream(
        self, *, provider_model_id: str, request: dict[str, Any]
    ) -> AsyncIterator[Any]:
        work = _provider(self)
        try:
            native = work.delegate.aresponses_stream(
                provider_model_id=provider_model_id, request=request
            )
        except BaseException:
            _safe_loss(work.observation, "factory_failure_count")
            raise
        try:
            return cast(
                "AsyncIterator[Any]",
                _admit(
                    work.observation,
                    native,
                    "aresponses_stream",
                    work.slot,
                    provider_model_id,
                    request,
                    None,
                ),
            )
        except Exception:
            return native

    def __repr__(self) -> str:
        return "StreamingProvider(local_counts_only)"

    def __reduce__(self) -> NoReturn:
        _fail()

    def __copy__(self) -> NoReturn:
        _fail()

    def __deepcopy__(self, _memo: object) -> NoReturn:
        _fail()


def _valid_delegate(value: object) -> bool:
    return isinstance(value, ProviderClient)


def wrap_streaming_provider(
    delegate: ProviderClient,
    *,
    observation: ProviderStreamObservation,
    provider_slot: ProviderName,
) -> ProviderClient:
    state = _observation(observation, explicit=True)
    if (
        not _valid_delegate(delegate)
        or type(provider_slot) is not str
        or provider_slot not in _SLOTS
    ):
        _fail()
    handle = object.__new__(_StreamingProvider)
    object.__setattr__(handle, "name", provider_slot)
    object.__setattr__(
        handle, "_work", _ProviderWork(delegate, state, cast("ProviderSlot", provider_slot))
    )
    _wrappers.add(handle)
    return handle
