"""Private policy fixtures and actual temporary FDs; no server grant or model execution."""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Literal

import pytest

from strata_forge.pipelines import _immutable_input as implementation
from strata_forge.pipelines._immutable_input import ImmutableInputReader
from strata_forge.pipelines.immutable_input_types import (
    CHUNK_BYTES,
    MAX_PAYLOAD_BYTES,
    ImmutableInputDecision,
    ImmutableInputError,
    ImmutableInputRequest,
    InputCode,
    NativeManifestPayload,
    NativeVersionRef,
    ResolvedInputPayload,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

# Assertions inspect private operation owners; dynamic values are deliberate invalid fixtures.
# pyright: reportPrivateUsage=false
type Effect = Literal["allow", "deny", "unavailable"]
type Custody = Literal["unsealed", "sealed", "locked", "fresh", "unknown"]
pytestmark = pytest.mark.asyncio
SCOPE = ContextVar("fictional_immutable_input_context", default="empty")
DATA = b'{"item_id":"fictional-1","text":"synthetic payload"}\n'


def request(data: bytes = DATA) -> ImmutableInputRequest:
    return ImmutableInputRequest(
        version=1,
        principal_id="1" * 32,
        project_id="2" * 32,
        run_id="3" * 32,
        attempt_id="4" * 32,
        purpose="evaluation",
        spec_sha256="a" * 64,
        kind="dataset",
        version_ref=NativeVersionRef(
            resource_id="5" * 32, version_id="6" * 32, manifest_sha256="b" * 64
        ),
        payload=NativeManifestPayload(
            artifact_id="7" * 32,
            logical_name="part.jsonl",
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            media_type="application/jsonl",
            schema_id="fictional.rows/v1",
        ),
    )


class Policy:
    def __init__(self) -> None:
        self.effect: Effect = "allow"
        self.custody: Custody = "unsealed"
        self.identity, self.revision = "8" * 32, 1
        self.expiry = time.time_ns() // 1_000_000 + 60_000
        self.calls = 0
        self.owner: asyncio.Task[object] | None = None
        self.change: Callable[[Policy, ImmutableInputRequest], None] | None = None
        self.events: list[str] = []

    async def decide(self, selected: ImmutableInputRequest) -> ImmutableInputDecision:
        self.calls += 1
        current = asyncio.current_task()
        if self.owner is None:
            self.owner = current
        assert current is self.owner
        self.events.append("policy")
        if self.change is not None:
            self.change(self, selected)
        return ImmutableInputDecision(
            request=selected,
            effect=self.effect,
            custody=self.custody,
            policy_identity=self.identity,
            policy_revision=self.revision,
            expires_at_ms=self.expiry,
        )


class Lease:
    def __init__(
        self, selected: ImmutableInputRequest, *, data: bytes | None = DATA, fd: int | None = None
    ) -> None:
        self.selected, self.data, self.fd = selected, data, fd
        self.resolved = self.closed = False
        self.close_calls = 0
        self.resolve_gate: asyncio.Event | None = None
        self.close_gate: asyncio.Event | None = None
        self.on_resolve: Callable[[], Awaitable[None]] | None = None
        self.on_close: Callable[[], Awaitable[None]] | None = None
        self.entered = asyncio.Event()

    async def resolve(self) -> ResolvedInputPayload:
        assert SCOPE.get() == "empty"
        self.entered.set()
        if self.resolve_gate is not None:
            await self.resolve_gate.wait()
        if self.on_resolve is not None:
            await self.on_resolve()
        self.resolved = True
        return ResolvedInputPayload(self.selected, self.data, self.fd)

    async def aclose(self) -> None:
        assert SCOPE.get() == "empty"
        self.close_calls += 1
        if self.close_gate is not None:
            await self.close_gate.wait()
        if self.on_close is not None:
            await self.on_close()
        self.closed = True


class Source:
    def __init__(self, *, data: bytes | None = DATA, fd: int | None = None) -> None:
        self.data, self.fd = data, fd
        self.leases: list[Lease] = []
        self.configure: Callable[[Lease], None] | None = None
        self.two_created = asyncio.Event()

    def lease(self, selected: ImmutableInputRequest) -> Lease:
        lease = Lease(selected, data=self.data, fd=self.fd)
        if self.configure is not None:
            self.configure(lease)
        self.leases.append(lease)
        if len(self.leases) == 2:
            self.two_created.set()
        return lease


async def refused(
    reader: ImmutableInputReader,
    selected: ImmutableInputRequest,
    code: InputCode | None = None,
    **kwargs: Any,
) -> None:
    with pytest.raises(ImmutableInputError) as error:
        await reader.read(selected, **kwargs)
    if code is not None:
        assert str(error.value) == code
    assert "PRIVATE" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.parametrize(
    "purpose", ["evaluation", "inference", "classifier_training", "prompt_render"]
)
@pytest.mark.parametrize("kind", ["dataset", "model", "prompt_build"])
async def test_exact_scopes_return_bounded_bytes_after_source_cleanup(
    purpose: str, kind: str
) -> None:
    selected = request().model_copy(update={"purpose": purpose, "kind": kind})
    policy, source = Policy(), Source()

    async def closed() -> None:
        policy.events.append("source-closed")

    source.configure = lambda lease: setattr(lease, "on_close", closed)
    reader = ImmutableInputReader(policy, source)
    try:
        value = await reader.read(selected)
        assert value.request == selected
        assert value.data == DATA
        assert value.request is not selected
        assert DATA.decode().strip() not in repr(value)
        assert source.leases[0].closed
        assert policy.events[-2:] == ["source-closed", "policy"]
        assert not reader._calls
    finally:
        await reader.aclose()


@pytest.mark.parametrize("authority", [None, {"authorized": True}])
async def test_no_inert_flag_or_missing_authority_can_invoke_source(authority: Any) -> None:
    source = Source()
    reader = ImmutableInputReader(authority, source)
    try:
        await refused(reader, request(), "input_authority_unavailable")
        assert not source.leases
    finally:
        await reader.aclose()


@pytest.mark.parametrize(
    "purpose", ["read", "training", "temperature_calibration_fit", "paired_cluster_uncertainty"]
)
async def test_unsupported_purpose_never_becomes_training_or_fit_permission(purpose: str) -> None:
    policy, source = Policy(), Source()
    reader = ImmutableInputReader(policy, source)
    try:
        await refused(
            reader, request().model_copy(update={"purpose": purpose}), "input_purpose_unsupported"
        )
        assert policy.calls == 0
        assert not source.leases
    finally:
        await reader.aclose()


@pytest.mark.parametrize("custody", ["sealed", "locked", "fresh", "unknown"])
async def test_positive_or_unknown_custody_refuses_before_source(custody: Custody) -> None:
    policy, source = Policy(), Source()
    policy.custody = custody
    reader = ImmutableInputReader(policy, source)
    try:
        await refused(reader, request(), "input_denied")
        assert not source.leases
    finally:
        await reader.aclose()


@pytest.mark.parametrize("effect", ["deny", "unavailable"])
async def test_current_refusal_is_not_url_or_digest_permission(effect: Effect) -> None:
    policy, source = Policy(), Source()
    policy.effect = effect
    reader = ImmutableInputReader(policy, source)
    try:
        await refused(reader, request(), "input_denied")
        assert not source.leases
    finally:
        await reader.aclose()


@pytest.mark.parametrize(
    "field",
    [
        "principal_id",
        "project_id",
        "run_id",
        "attempt_id",
        "purpose",
        "spec_sha256",
        "kind",
        "resource_id",
        "version_id",
        "manifest_sha256",
        "artifact_id",
        "logical_name",
        "sha256",
        "size_bytes",
        "media_type",
        "schema_id",
    ],
)
async def test_every_scope_ref_and_manifest_field_is_bound_before_source(field: str) -> None:
    class Mismatch(Policy):
        async def decide(self, selected: ImmutableInputRequest) -> ImmutableInputDecision:
            decision = await super().decide(selected)
            if field in {"resource_id", "version_id", "manifest_sha256"}:
                updated = selected.version_ref.model_copy(
                    update={field: "c" * (64 if field.endswith("sha256") else 32)}
                )
                selected = selected.model_copy(update={"version_ref": updated})
            elif field in {
                "artifact_id",
                "logical_name",
                "sha256",
                "size_bytes",
                "media_type",
                "schema_id",
            }:
                values = {
                    "artifact_id": "c" * 32,
                    "logical_name": "other.jsonl",
                    "sha256": "c" * 64,
                    "size_bytes": 1,
                    "media_type": "text/plain",
                    "schema_id": "other/v1",
                }
                selected = selected.model_copy(
                    update={"payload": selected.payload.model_copy(update={field: values[field]})}
                )
            else:
                values = {"purpose": "inference", "kind": "model", "spec_sha256": "c" * 64}
                selected = selected.model_copy(update={field: values.get(field, "c" * 32)})
            return decision.model_copy(update={"request": selected})

    source = Source()
    reader = ImmutableInputReader(Mismatch(), source)
    try:
        await refused(reader, request(), "input_changed")
        assert not source.leases
    finally:
        await reader.aclose()


async def test_expired_policy_refuses_before_source() -> None:
    policy, source = Policy(), Source()
    policy.expiry = time.time_ns() // 1_000_000 - 1
    reader = ImmutableInputReader(policy, source)
    try:
        await refused(reader, request(), "input_expired")
        assert not source.leases
    finally:
        await reader.aclose()


@pytest.mark.parametrize("change", ["revoked", "revision", "identity"])
async def test_current_change_during_delayed_resolution_discards_result_and_closes(
    change: str,
) -> None:
    policy, source = Policy(), Source()

    async def alter() -> None:
        if change == "revoked":
            policy.effect = "deny"
        elif change == "revision":
            policy.revision = 2
        else:
            policy.identity = "9" * 32

    source.configure = lambda lease: setattr(lease, "on_resolve", alter)
    reader = ImmutableInputReader(policy, source)
    try:
        await refused(reader, request(), "input_denied" if change == "revoked" else "input_changed")
        assert source.leases[0].closed
    finally:
        await reader.aclose()


async def test_current_recheck_after_actual_source_close_prevents_late_disclosure() -> None:
    policy, source = Policy(), Source()

    async def revoke() -> None:
        policy.effect = "deny"

    source.configure = lambda lease: setattr(lease, "on_close", revoke)
    reader = ImmutableInputReader(policy, source)
    try:
        await refused(reader, request(), "input_denied")
        assert source.leases[0].closed
    finally:
        await reader.aclose()


async def test_actual_fd_duplicate_is_closed_and_original_offset_and_owner_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fictional.dat"
    path.write_bytes(DATA)
    policy = Policy()
    duplicates: list[int] = []
    actual = os.dup

    def duplicate(fd: int) -> int:
        assert SCOPE.get() == "empty"
        value = actual(fd)
        duplicates.append(value)
        return value

    monkeypatch.setattr(os, "dup", duplicate)
    with path.open("rb") as original:
        original.seek(3)
        source = Source(data=None, fd=original.fileno())
        reader = ImmutableInputReader(policy, source)
        token = SCOPE.set("parent-domain-context")
        try:
            assert (await reader.read(request())).data == DATA
            assert original.tell() == 3
            assert len(duplicates) == 1
            with pytest.raises(OSError, match=r"[Bb]ad file descriptor"):
                os.fstat(duplicates[0])
            assert os.fstat(original.fileno()).st_size == len(DATA)
            assert source.leases[0].closed
        finally:
            SCOPE.reset(token)
            await reader.aclose()


@pytest.mark.parametrize("invalid", ["size", "digest", "delivery_binding", "nonregular"])
async def test_real_descriptor_refusals_never_return_unverified_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    path = tmp_path / "fictional.dat"
    path.write_bytes(DATA)
    original = os.open(path, os.O_RDONLY)
    extra = None
    if invalid == "nonregular":
        os.close(original)
        original, extra = os.pipe()
    source = Source(data=None, fd=original)
    selected = request()
    reads: list[int] = []
    if invalid == "size":
        selected = selected.model_copy(
            update={"payload": selected.payload.model_copy(update={"size_bytes": 1})}
        )
    if invalid == "digest":
        selected = selected.model_copy(
            update={"payload": selected.payload.model_copy(update={"sha256": "0" * 64})}
        )
    if invalid == "delivery_binding":
        source.configure = lambda lease: setattr(
            lease, "selected", lease.selected.model_copy(update={"project_id": "f" * 32})
        )
    actual = os.pread

    def read(fd: int, size: int, offset: int) -> bytes:
        reads.append(size)
        return actual(fd, size, offset)

    monkeypatch.setattr(os, "pread", read)
    reader = ImmutableInputReader(Policy(), source)
    try:
        await refused(reader, selected)
        if invalid != "digest":
            assert not reads
        assert source.leases[0].closed
    finally:
        await reader.aclose()
        os.close(original)
        if extra is not None:
            os.close(extra)


async def test_file_mutation_between_owned_chunks_refuses(tmp_path: Path) -> None:
    data = b"x" * (CHUNK_BYTES + 1)
    path = tmp_path / "fictional.dat"
    path.write_bytes(data)
    with path.open("r+b") as stream:
        policy = Policy()

        def mutate(current: Policy, _selected: ImmutableInputRequest) -> None:
            if current.calls == 6:
                os.pwrite(stream.fileno(), b"y", 0)

        policy.change = mutate
        source = Source(data=None, fd=stream.fileno())
        reader = ImmutableInputReader(policy, source)
        try:
            await refused(reader, request(data))
            assert source.leases[0].closed
        finally:
            await reader.aclose()


async def test_delayed_real_resolver_thread_and_shutdown_remain_owned_through_cancellation() -> (
    None
):
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    source = Source()

    async def physical() -> None:
        def resolve() -> None:
            try:
                assert SCOPE.get() == "empty"
                entered.set()
                assert release.wait(3.0)
            finally:
                exited.set()

        await asyncio.to_thread(resolve)

    async def close() -> None:
        assert exited.is_set()

    def configure(lease: Lease) -> None:
        lease.on_resolve = physical
        lease.on_close = close

    source.configure = configure
    reader = ImmutableInputReader(Policy(), source)
    task = asyncio.create_task(reader.read(request()))
    shutdown = None
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        task.cancel()
        task.cancel()
        shutdown = asyncio.create_task(reader.aclose())
        await asyncio.sleep(0)
        assert not task.done()
        assert not shutdown.done()
        assert len(reader._calls) == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await shutdown
        assert exited.is_set()
        assert source.leases[0].closed
        assert not reader._calls
        assert not reader._children
    finally:
        release.set()
        await asyncio.gather(
            task, *([] if shutdown is None else [shutdown]), return_exceptions=True
        )
        await reader.aclose()


async def test_one_original_deadline_includes_resolver_and_held_cleanup() -> None:
    resolve_gate, close_gate = asyncio.Event(), asyncio.Event()
    source = Source()

    def configure(lease: Lease) -> None:
        lease.resolve_gate = resolve_gate
        lease.close_gate = close_gate

    source.configure = configure
    reader = ImmutableInputReader(Policy(), source)
    task = asyncio.create_task(reader.read(request(), deadline=time.monotonic() + 0.05))
    try:
        await asyncio.sleep(0.03)
        resolve_gate.set()
        await asyncio.sleep(0.04)
        assert not task.done()
        close_gate.set()
        with pytest.raises(ImmutableInputError, match="input_deadline"):
            await task
        assert source.leases[0].closed
    finally:
        resolve_gate.set()
        close_gate.set()
        await asyncio.gather(task, return_exceptions=True)
        await reader.aclose()


async def test_two_live_reads_are_admitted_without_a_third_resolver() -> None:
    gate = asyncio.Event()
    source = Source()
    source.configure = lambda lease: setattr(lease, "resolve_gate", gate)

    # Each admitted call has its own same-task policy observation, as a real installed producer does.
    class Concurrent(Policy):
        async def decide(self, selected: ImmutableInputRequest) -> ImmutableInputDecision:
            return ImmutableInputDecision(
                request=selected,
                effect="allow",
                custody="unsealed",
                policy_identity="8" * 32,
                policy_revision=1,
                expires_at_ms=self.expiry,
            )

    reader = ImmutableInputReader(Concurrent(), source)
    tasks = [asyncio.create_task(reader.read(request())) for _ in range(2)]
    try:
        await asyncio.wait_for(source.two_created.wait(), 1)
        await refused(reader, request(), "input_limited")
        assert len(source.leases) == 2
        gate.set()
        values = await asyncio.gather(*tasks)
        assert all(value.data == DATA for value in values)
    finally:
        gate.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await reader.aclose()


async def test_oversized_byte_delivery_and_copied_extra_fields_refuse_before_copy() -> None:
    source = Source(data=b"x" * (MAX_PAYLOAD_BYTES + 1))
    reader = ImmutableInputReader(Policy(), source)
    try:
        await refused(reader, request())
        assert source.leases[0].closed
        malformed = request()
        vars(malformed)["unexpected"] = True
        await refused(reader, malformed, "input_invalid")
        assert len(source.leases) == 1
    finally:
        await reader.aclose()


async def test_untrusted_decision_diagnostics_are_fixed_before_source() -> None:
    class Broken(Policy):
        async def decide(self, selected: ImmutableInputRequest) -> ImmutableInputDecision:
            raise RuntimeError("PRIVATE secret/path/value")

    source = Source()
    reader = ImmutableInputReader(Broken(), source)
    try:
        await refused(reader, request(), "input_authority_unavailable")
        assert not source.leases
    finally:
        await reader.aclose()


async def test_mutating_callback_copy_cannot_change_private_requested_identity() -> None:
    policy = Policy()
    source = Source()

    def mutate(_policy: Policy, selected: ImmutableInputRequest) -> None:
        vars(selected)["project_id"] = "f" * 32

    policy.change = mutate
    reader = ImmutableInputReader(policy, source)
    try:
        await refused(reader, request(), "input_changed")
        assert not source.leases
    finally:
        await reader.aclose()


async def test_operational_child_cannot_close_its_own_reader() -> None:
    source = Source()
    reader = ImmutableInputReader(Policy(), source)

    async def attempt() -> None:
        with pytest.raises(ImmutableInputError, match="input_closed"):
            await reader.aclose()
        assert not reader._closed

    source.configure = lambda lease: setattr(lease, "on_resolve", attempt)
    try:
        assert (await reader.read(request())).data == DATA
    finally:
        await reader.aclose()


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
async def test_child_controls_are_sanitized_only_after_source_cleanup(
    control: type[KeyboardInterrupt] | type[SystemExit],
) -> None:
    source = Source()

    async def interrupted() -> None:
        raise control("PRIVATE control argument")

    source.configure = lambda lease: setattr(lease, "on_resolve", interrupted)
    reader = ImmutableInputReader(Policy(), source)
    try:
        with pytest.raises(control) as error:
            await reader.read(request())
        assert error.value.args == ()
        assert error.value.__context__ is None
        assert source.leases[0].closed
    finally:
        await reader.aclose()


async def test_reused_source_lease_does_not_close_another_active_read() -> None:
    gate = asyncio.Event()
    source = Source()
    reader = None

    class Concurrent(Policy):
        async def decide(self, selected: ImmutableInputRequest) -> ImmutableInputDecision:
            return ImmutableInputDecision(
                request=selected,
                effect="allow",
                custody="unsealed",
                policy_identity="8" * 32,
                policy_revision=1,
                expires_at_ms=self.expiry,
            )

    shared = Lease(request())
    shared.resolve_gate = gate
    source.lease = lambda selected: shared
    reader = ImmutableInputReader(Concurrent(), source)
    task = asyncio.create_task(reader.read(request()))
    try:
        await asyncio.wait_for(shared.entered.wait(), 1)
        await refused(reader, request(), "input_source_invalid")
        assert shared.close_calls == 0
        gate.set()
        assert (await task).data == DATA
    finally:
        gate.set()
        await asyncio.gather(task, return_exceptions=True)
        await reader.aclose()


async def test_failed_source_close_is_quarantined_and_actual_idempotent_close_retried(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fictional.dat"
    path.write_bytes(DATA)
    with path.open("rb") as original:
        owned: int | None = os.dup(original.fileno())
        source = Source(data=None, fd=owned)

        async def close() -> None:
            nonlocal owned
            descriptor, owned = owned, None
            if descriptor is not None:
                os.close(descriptor)
                raise RuntimeError("PRIVATE lost close acknowledgement")

        source.configure = lambda lease: setattr(lease, "on_close", close)
        reader = ImmutableInputReader(Policy(), source)
        await refused(reader, request(), "input_cleanup_failed")
        assert owned is None
        assert reader._poisoned
        await reader.aclose()
        assert source.leases[0].close_calls == 2
        assert not reader._retained
        assert os.fstat(original.fileno()).st_size == len(DATA)


async def test_uncertain_fd_close_never_retries_a_reused_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fictional.dat"
    path.write_bytes(DATA)
    actual = implementation._close_descriptor

    def lost(owned: implementation._Descriptor) -> None:
        actual(owned)
        raise OSError("PRIVATE close acknowledgement")

    monkeypatch.setattr(implementation, "_close_descriptor", lost)
    with path.open("rb") as original:
        source = Source(data=None, fd=original.fileno())
        reader = ImmutableInputReader(Policy(), source)
        await refused(reader, request(), "input_cleanup_failed")
        other = os.dup(original.fileno())
        try:
            with pytest.raises(ImmutableInputError, match="input_cleanup_failed"):
                await reader.aclose()
            assert os.fstat(other).st_size == len(DATA)
            assert source.leases[0].closed
        finally:
            os.close(other)


@pytest.mark.parametrize("operation", ["duplicate", "chunk"])
async def test_queued_real_fd_worker_checks_expiry_before_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    path = tmp_path / "fictional.dat"
    path.write_bytes(DATA)
    entered, release = threading.Event(), threading.Event()
    reads: list[int] = []
    duplicates: list[int] = []
    actual_pread, actual_dup = os.pread, os.dup

    def pread(fd: int, size: int, offset: int) -> bytes:
        reads.append(size)
        return actual_pread(fd, size, offset)

    def duplicate(fd: int) -> int:
        duplicates.append(fd)
        return actual_dup(fd)

    monkeypatch.setattr(os, "pread", pread)
    monkeypatch.setattr(os, "dup", duplicate)
    original_operation = getattr(
        implementation, "_duplicate" if operation == "duplicate" else "_chunk"
    )

    def delayed(*arguments: Any) -> Any:
        entered.set()
        assert release.wait(3)
        return original_operation(*arguments)

    monkeypatch.setattr(
        implementation, "_duplicate" if operation == "duplicate" else "_chunk", delayed
    )
    with path.open("rb") as stream:
        source = Source(data=None, fd=stream.fileno())
        reader = ImmutableInputReader(Policy(), source)
        task = asyncio.create_task(reader.read(request(), deadline=time.monotonic() + 0.05))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            await asyncio.sleep(0.07)
            assert not task.done()
            assert reader._calls
            release.set()
            with pytest.raises(ImmutableInputError, match="input_deadline"):
                await task
            assert not reads
            assert len(duplicates) == (operation == "chunk")
            assert source.leases[0].closed
            assert not reader._calls
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await reader.aclose()


async def test_empty_exact_payload_checks_hash_and_cleanup() -> None:
    source = Source(data=b"")
    reader = ImmutableInputReader(Policy(), source)
    try:
        assert (await reader.read(request(b""))).data == b""
        assert source.leases[0].closed
    finally:
        await reader.aclose()


@pytest.mark.parametrize(
    "field", ["policy_revision", "expires_at_ms", "effect", "custody", "policy_identity"]
)
async def test_malformed_current_decision_refuses_before_source(field: str) -> None:
    class Malformed(Policy):
        async def decide(self, selected: ImmutableInputRequest) -> ImmutableInputDecision:
            decision = await super().decide(selected)
            return decision.model_copy(update={field: True})

    source = Source()
    reader = ImmutableInputReader(Malformed(), source)
    try:
        await refused(reader, request(), "input_authority_unavailable")
        assert not source.leases
    finally:
        await reader.aclose()


@pytest.mark.parametrize("deadline", [True, 1, float("inf"), float("nan")])
async def test_invalid_outer_deadline_refuses_before_policy(deadline: Any) -> None:
    policy, source = Policy(), Source()
    reader = ImmutableInputReader(policy, source)
    try:
        await refused(reader, request(), "input_invalid", deadline=deadline)
        assert policy.calls == 0
        assert not source.leases
    finally:
        await reader.aclose()


async def test_missing_source_and_closed_reader_refuse_without_policy() -> None:
    policy = Policy()
    reader = ImmutableInputReader(policy, None)
    await refused(reader, request(), "input_source_unavailable")
    await reader.aclose()
    await refused(reader, request(), "input_closed")
    assert policy.calls == 0
