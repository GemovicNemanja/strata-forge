"""Bounded current-authority reads from installed operational source leases only."""

# Private call and reader owners cooperate inside this module; none is a public field API.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import stat
import threading
import time
from contextlib import suppress
from contextvars import Context
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn, Protocol, cast

from strata_forge.pipelines.immutable_input_types import (
    CHUNK_BYTES,
    MAX_LIVE_READS,
    MAX_PAYLOAD_BYTES,
    READ_SECONDS,
    HumanRenderInputDecision,
    HumanRenderInputRequest,
    ImmutableInputDecision,
    ImmutableInputError,
    ImmutableInputRequest,
    InputCode,
    ResolvedHumanRenderPayload,
    ResolvedInputPayload,
    VerifiedHumanRenderPayload,
    VerifiedInputPayload,
    checked_decision,
    checked_human_render_decision,
    checked_human_render_request,
    checked_request,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable


class ImmutableInputAuthority(Protocol):
    """Installed producer of actual current rights/custody, never selected by an inert spec."""

    async def decide(self, request: ImmutableInputRequest, /) -> ImmutableInputDecision: ...


class ImmutableInputLease(Protocol):
    """Operational ownership survives delayed resolution; close is idempotent and owns its I/O."""

    async def resolve(self) -> ResolvedInputPayload: ...
    async def aclose(self) -> None: ...


class ImmutableInputSource(Protocol):
    """Construct a fresh operational lease without I/O, paths or serialized policy selectors."""

    def lease(self, request: ImmutableInputRequest, /) -> ImmutableInputLease | None: ...


class HumanRenderInputAuthority(Protocol):
    """Installed human-operation authority; no durable-run grant is inferred."""

    async def decide(self, request: HumanRenderInputRequest, /) -> HumanRenderInputDecision: ...


class HumanRenderInputLease(Protocol):
    """A live human source owns all delayed resolution and original descriptors."""

    async def resolve(self) -> ResolvedHumanRenderPayload: ...
    async def aclose(self) -> None: ...


class HumanRenderInputSource(Protocol):
    """Construct only a fresh human operational lease, with no I/O or permission fallback."""

    def lease(self, request: HumanRenderInputRequest, /) -> HumanRenderInputLease | None: ...


type _InputRequest = ImmutableInputRequest | HumanRenderInputRequest
type _InputDecision = ImmutableInputDecision | HumanRenderInputDecision
type _InputLease = ImmutableInputLease | HumanRenderInputLease
type _InputDelivery = ResolvedInputPayload | ResolvedHumanRenderPayload


@dataclass(frozen=True, slots=True, repr=False)
class _ReadValue:
    request: _InputRequest
    data: bytes


class _UnwindError(Exception):
    def __init__(self, code: InputCode) -> None:
        self.code: InputCode = code
        super().__init__(code)


@dataclass(frozen=True, slots=True, repr=False)
class _Outcome:
    value: object = None
    failed: bool = False
    cancelled: bool = False
    control: type[KeyboardInterrupt] | type[SystemExit] | None = None


@dataclass(slots=True, repr=False)
class _Descriptor:
    fd: int | None = None
    deadline: float = 0.0
    stop: threading.Event = field(default_factory=threading.Event)


async def _capture(work: Awaitable[object]) -> _Outcome:
    try:
        return _Outcome(await work)
    except asyncio.CancelledError:
        return _Outcome(cancelled=True)
    except (KeyboardInterrupt, SystemExit) as error:
        return _Outcome(control=type(error))
    except BaseException:
        return _Outcome(failed=True)


async def _drain(task: asyncio.Task[_Outcome]) -> tuple[_Outcome, bool]:
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted = True
    return task.result(), interrupted


def _control(kind: type[KeyboardInterrupt] | type[SystemExit]) -> NoReturn:
    error = kind()
    try:
        raise error from None
    finally:
        error.__context__ = error.__cause__ = None


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _duplicate(original: int, owned: _Descriptor, size: int) -> tuple[int, int, int, int, int]:
    if owned.stop.is_set() or time.monotonic() >= owned.deadline:
        raise ImmutableInputError("input_deadline")
    owned.fd = os.dup(original)
    info = os.fstat(owned.fd)
    if not stat.S_ISREG(info.st_mode) or info.st_size != size:
        raise ImmutableInputError("input_source_invalid")
    return _identity(info)


def _chunk(
    owned: _Descriptor, identity: tuple[int, int, int, int, int], offset: int, size: int
) -> bytes:
    if owned.stop.is_set() or time.monotonic() >= owned.deadline:
        raise ImmutableInputError("input_deadline")
    if owned.fd is None or _identity(os.fstat(owned.fd)) != identity:
        raise ImmutableInputError("input_source_invalid")
    if owned.stop.is_set() or time.monotonic() >= owned.deadline:
        raise ImmutableInputError("input_deadline")
    return os.pread(owned.fd, size, offset)


def _close_descriptor(owned: _Descriptor) -> None:
    fd, owned.fd = owned.fd, None
    if fd is not None:
        # A failed close can have consumed its FD number. Never retry it after reuse.
        os.close(fd)


@dataclass(eq=False, slots=True, repr=False)
class _Call:
    reader: _InputReaderCore
    owner: asyncio.Task[object]
    request: _InputRequest
    deadline: float
    done: asyncio.Event = field(default_factory=asyncio.Event)
    descriptor: _Descriptor = field(default_factory=_Descriptor)
    lease: _InputLease | None = None
    policy: tuple[str, int] | None = None
    cancelled: bool = False
    control: type[KeyboardInterrupt] | type[SystemExit] | None = None
    failure: InputCode | None = None

    def check(self) -> None:
        if self.owner is not asyncio.current_task():
            raise _UnwindError("input_closed")
        if self.control is not None:
            raise _UnwindError("input_source_unavailable")
        if self.cancelled or self.owner.cancelling():
            self.cancelled = True
            raise _UnwindError("input_closed")
        if self.reader._closed or self.reader._poisoned:
            raise _UnwindError("input_closed")
        if time.monotonic() >= self.deadline:
            raise _UnwindError("input_deadline")

    async def current(self) -> None:
        self.check()
        decision = None
        try:
            # Actual domain checks stay in the parent task; no proof enters I/O children.
            async with asyncio.timeout(max(0.0, self.deadline - time.monotonic())):
                decision = await self.reader._decision(self.request)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise _UnwindError("input_deadline") from None
        except Exception:
            raise _UnwindError("input_authority_unavailable") from None
        self.check()
        if decision.request != self.request:
            raise _UnwindError("input_changed")
        if decision.effect != "allow" or decision.custody != "unsealed":
            raise _UnwindError("input_denied")
        identity = decision.policy_identity, decision.policy_revision
        if self.policy is not None and self.policy != identity:
            raise _UnwindError("input_changed")
        self.policy = identity
        remaining = (decision.expires_at_ms - time.time_ns() // 1_000_000) / 1000
        if remaining <= 0:
            raise _UnwindError("input_expired")
        self.deadline = min(self.deadline, time.monotonic() + remaining)
        self.check()

    async def owned(self, work: Awaitable[object], *, cleanup: bool = False) -> object:
        task = asyncio.create_task(_capture(work), context=Context())
        self.reader._children.add(task)
        try:
            try:
                if cleanup:
                    outcome, interrupted = await _drain(task)
                    self.cancelled |= interrupted
                else:
                    try:
                        async with asyncio.timeout(max(0.0, self.deadline - time.monotonic())):
                            await asyncio.shield(task)
                    except asyncio.CancelledError:
                        self.cancelled = True
                        self.descriptor.stop.set()
                    except TimeoutError:
                        self.failure = "input_deadline"
                        self.descriptor.stop.set()
                    finally:
                        # Do not cancel a resolver/thread and mistake its await for physical completion.
                        outcome, interrupted = await _drain(task)
                        self.cancelled |= interrupted
            except BaseException:
                outcome, interrupted = await _drain(task)
                self.cancelled |= interrupted
                raise
            self.control = self.control or outcome.control
            self.cancelled |= outcome.cancelled
            if outcome.failed or (cleanup and (outcome.cancelled or outcome.control is not None)):
                raise _UnwindError(
                    "input_cleanup_failed" if cleanup else "input_source_unavailable"
                )
            if not cleanup:
                self.check()
                if self.failure is not None:
                    raise _UnwindError(self.failure)
            return outcome.value
        finally:
            self.reader._children.discard(task)

    async def materialize(self, delivery: _InputDelivery) -> bytes:
        valid = (
            type(self.request) is ImmutableInputRequest
            and type(delivery) is ResolvedInputPayload
            and checked_request(delivery.request) == self.request
        ) or (
            type(self.request) is HumanRenderInputRequest
            and type(delivery) is ResolvedHumanRenderPayload
            and checked_human_render_request(delivery.request) == self.request
        )
        if not valid:
            raise _UnwindError("input_changed")
        size = self.request.payload.size_bytes
        if (delivery.data is None) == (delivery.fd is None):
            raise _UnwindError("input_source_invalid")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        if delivery.data is not None:
            if (
                type(delivery.data) is not bytes
                or len(delivery.data) != size
                or len(delivery.data) > MAX_PAYLOAD_BYTES
            ):
                raise _UnwindError("input_source_invalid")
            for offset in range(0, size, CHUNK_BYTES):
                await self.current()
                chunk = delivery.data[offset : offset + CHUNK_BYTES]
                digest.update(chunk)
                chunks.append(chunk)
        else:
            original = delivery.fd
            if type(original) is not int or not 0 <= original <= 1_000_000:
                raise _UnwindError("input_source_invalid")
            await self.current()
            self.descriptor.deadline = self.deadline
            identity = cast(
                "tuple[int, int, int, int, int]",
                await self.owned(asyncio.to_thread(_duplicate, original, self.descriptor, size)),
            )
            for offset in range(0, size, CHUNK_BYTES):
                await self.current()
                self.descriptor.deadline = self.deadline
                wanted = min(CHUNK_BYTES, size - offset)
                chunk = await self.owned(
                    asyncio.to_thread(_chunk, self.descriptor, identity, offset, wanted)
                )
                await self.current()
                if type(chunk) is not bytes or len(chunk) != wanted:
                    raise _UnwindError("input_integrity")
                digest.update(chunk)
                chunks.append(chunk)
            await self.current()
            self.descriptor.deadline = self.deadline
            tail = await self.owned(asyncio.to_thread(_chunk, self.descriptor, identity, size, 1))
            if tail != b"":
                raise _UnwindError("input_integrity")
        if digest.hexdigest() != self.request.payload.sha256:
            raise _UnwindError("input_integrity")
        return b"".join(chunks)

    async def cleanup(self) -> None:
        if self.descriptor.fd is not None:
            try:
                await self.owned(
                    asyncio.to_thread(_close_descriptor, self.descriptor), cleanup=True
                )
            except BaseException:
                self.reader._unknown_fd_close = self.reader._poisoned = True
                self.failure = "input_cleanup_failed"
        lease = self.lease
        if lease is not None:

            async def close() -> None:
                await lease.aclose()

            try:
                await self.owned(close(), cleanup=True)
            except BaseException:
                self.reader._retained.append(lease)
                self.reader._poisoned = True
                self.failure = "input_cleanup_failed"
            else:
                self.reader._leases.discard(id(lease))


class _InputReaderCore:
    """The physical byte/FD owner is shared without relabeling one admission as another."""

    def __init__(self) -> None:
        loop = None
        with suppress(RuntimeError):
            loop = asyncio.get_running_loop()
        if loop is None:
            raise ImmutableInputError("input_closed")
        self._loop = loop
        self._calls: set[_Call] = set()
        self._children: set[asyncio.Task[_Outcome]] = set()
        self._leases: set[int] = set()
        self._retained: list[_InputLease] = []
        self._closed = self._poisoned = self._unknown_fd_close = self._closing = False

    def _require_ports(self) -> None:
        raise ImmutableInputError("input_authority_unavailable")

    def _checked(self, value: object) -> _InputRequest:
        raise ImmutableInputError("input_invalid")

    async def _decision(self, request: _InputRequest) -> _InputDecision:
        raise ImmutableInputError("input_authority_unavailable")

    def _source_lease(self, request: _InputRequest) -> _InputLease | None:
        raise ImmutableInputError("input_source_unavailable")

    async def _read(self, request: _InputRequest, *, deadline: float | None = None) -> _ReadValue:
        if asyncio.get_running_loop() is not self._loop or self._closed or self._poisoned:
            raise ImmutableInputError("input_closed")
        self._require_ports()
        request = self._checked(request)
        owner = asyncio.current_task()
        if owner is None or owner.cancelling():
            raise asyncio.CancelledError
        if len(self._calls) >= MAX_LIVE_READS or any(call.owner is owner for call in self._calls):
            raise ImmutableInputError("input_limited")
        now = time.monotonic()
        if deadline is not None and (type(deadline) is not float or not math.isfinite(deadline)):
            raise ImmutableInputError("input_invalid")
        call = _Call(
            self,
            owner,
            request,
            min(now + READ_SECONDS, deadline if deadline is not None else now + READ_SECONDS),
        )
        data = None
        self._calls.add(call)
        try:
            try:
                await call.current()
                lease = self._source_lease(request)
                if lease is None or id(lease) in self._leases:
                    raise _UnwindError("input_source_invalid")
                call.lease = lease
                self._leases.add(id(lease))
                stop, original_deadline = call.descriptor.stop, call.deadline

                async def resolve() -> _InputDelivery:
                    if (
                        stop.is_set()
                        or owner.done()
                        or owner.cancelling()
                        or time.monotonic() >= original_deadline
                    ):
                        raise ImmutableInputError("input_deadline")
                    return await lease.resolve()

                delivery = await call.owned(resolve())
                await call.current()
                data = await call.materialize(cast("_InputDelivery", delivery))
            except _UnwindError as error:
                call.failure = call.failure or error.code
            except asyncio.CancelledError:
                call.cancelled = True
            except (KeyboardInterrupt, SystemExit) as error:
                call.control = call.control or type(error)
            except BaseException:
                call.failure = "input_source_unavailable"
            await call.cleanup()
            if call.failure is None and not call.cancelled and call.control is None:
                try:
                    await call.current()
                    call.check()
                except _UnwindError as error:
                    call.failure = error.code
                except asyncio.CancelledError:
                    call.cancelled = True
                except (KeyboardInterrupt, SystemExit) as error:
                    call.control = type(error)
                except BaseException:
                    call.failure = "input_authority_unavailable"
        finally:
            self._calls.discard(call)
            call.done.set()
        if call.control is not None:
            _control(call.control)
        if call.cancelled or owner.cancelling():
            raise asyncio.CancelledError from None
        if call.failure is not None or data is None:
            raise ImmutableInputError(call.failure or "input_source_unavailable") from None
        return _ReadValue(self._checked(request), data)

    async def aclose(self) -> None:
        if (
            asyncio.get_running_loop() is not self._loop
            or self._closing
            or asyncio.current_task() in self._children
            or any(call.owner is asyncio.current_task() for call in self._calls)
        ):
            raise ImmutableInputError("input_closed")
        self._closed = self._closing = True
        owner = asyncio.current_task()
        cancelled = owner is not None and owner.cancelling() > 0
        control = None
        failed = self._unknown_fd_close
        try:
            calls = tuple(self._calls)
            for call in calls:
                call.descriptor.stop.set()
                if not call.owner.done() and not call.owner.cancelling():
                    call.owner.cancel()

            async def wait() -> None:
                for call in calls:
                    await call.done.wait()

            outcome, interrupted = await _drain(
                asyncio.create_task(_capture(wait()), context=Context())
            )
            cancelled |= interrupted or outcome.cancelled
            control = outcome.control
            failed |= outcome.failed
            for lease in tuple(self._retained):

                async def close(target: _InputLease = lease) -> None:
                    await target.aclose()

                outcome, interrupted = await _drain(
                    asyncio.create_task(_capture(close()), context=Context())
                )
                cancelled |= interrupted or outcome.cancelled
                control = control or outcome.control
                if outcome.failed or outcome.cancelled or outcome.control is not None:
                    failed = True
                else:
                    self._retained.remove(lease)
                    self._leases.discard(id(lease))
        finally:
            self._closing = False
        if control is not None:
            _control(control)
        if cancelled:
            raise asyncio.CancelledError from None
        if failed or self._retained:
            raise ImmutableInputError("input_cleanup_failed") from None


class ImmutableInputReader(_InputReaderCore):
    """Exact run-bound reads; live human operations cannot enter this variant."""

    def __init__(
        self, authority: ImmutableInputAuthority | None, source: ImmutableInputSource | None
    ) -> None:
        super().__init__()
        self._authority, self._source = authority, source

    def _require_ports(self) -> None:
        if self._authority is None:
            raise ImmutableInputError("input_authority_unavailable")
        if self._source is None:
            raise ImmutableInputError("input_source_unavailable")

    def _checked(self, value: object) -> ImmutableInputRequest:
        return checked_request(value)

    async def _decision(self, request: _InputRequest) -> ImmutableInputDecision:
        self._require_ports()
        if self._authority is None:
            raise ImmutableInputError("input_authority_unavailable")
        return checked_decision(await self._authority.decide(checked_request(request)))

    def _source_lease(self, request: _InputRequest) -> ImmutableInputLease | None:
        if self._source is None:
            raise ImmutableInputError("input_source_unavailable")
        return self._source.lease(checked_request(request))

    async def read(
        self, request: ImmutableInputRequest, *, deadline: float | None = None
    ) -> VerifiedInputPayload:
        value = await self._read(request, deadline=deadline)
        return VerifiedInputPayload(checked_request(value.request), value.data)


class HumanRenderInputReader(_InputReaderCore):
    """Only explicitly admitted live human prompt-render reads, with no durable run claim."""

    def __init__(
        self, authority: HumanRenderInputAuthority | None, source: HumanRenderInputSource | None
    ) -> None:
        super().__init__()
        self._authority, self._source = authority, source

    def _require_ports(self) -> None:
        if self._authority is None:
            raise ImmutableInputError("input_authority_unavailable")
        if self._source is None:
            raise ImmutableInputError("input_source_unavailable")

    def _checked(self, value: object) -> HumanRenderInputRequest:
        return checked_human_render_request(value)

    async def _decision(self, request: _InputRequest) -> HumanRenderInputDecision:
        self._require_ports()
        if self._authority is None:
            raise ImmutableInputError("input_authority_unavailable")
        return checked_human_render_decision(
            await self._authority.decide(checked_human_render_request(request))
        )

    def _source_lease(self, request: _InputRequest) -> HumanRenderInputLease | None:
        if self._source is None:
            raise ImmutableInputError("input_source_unavailable")
        return self._source.lease(checked_human_render_request(request))

    async def read(
        self, request: HumanRenderInputRequest, *, deadline: float | None = None
    ) -> VerifiedHumanRenderPayload:
        value = await self._read(request, deadline=deadline)
        return VerifiedHumanRenderPayload(checked_human_render_request(value.request), value.data)
