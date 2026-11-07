"""Local fake-byte calls, owned cancellation and body-free evidence; no provider executes."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from strata_forge.prompts.consumption import LocalBuildSource, consume_build
from strata_forge.tracing import prompt_attempts as runtime
from strata_forge.tracing.prompt_attempt_types import AttemptParents, DeclaredBodyPin
from strata_forge.tracing.prompt_attempts import (
    LocalByteResponse,
    LocalByteTransport,
    PreparedInvocation,
    PromptCaptureError,
    capture_once,
    prepare_invocation,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine

ROOT = Path(__file__).parents[3]
CORPUS = json.loads((ROOT / "tests/fixtures/prompts/composition-renders-v1.json").read_text())
CASE = next(case for case in CORPUS["cases"] if case["expected_output"]["kind"] == "chat")
NO_PARENT = AttemptParents(retry_of=None, fallback_of=None, continuation_of=None)


async def prepared() -> PreparedInvocation:
    raw = bytes.fromhex(CASE["build_canonical_utf8_hex"])
    build = consume_build(
        raw,
        expected_build_id=CASE["build_id"],
        expected_size_bytes=len(raw),
        source=LocalBuildSource(kind="caller_bytes", partition="fixture:attempts", reference=None),
    )
    return await prepare_invocation(build, CASE["values"])


class Fake:
    def __init__(self, body: bytes = b"BODY CANARY", status: int | None = 200) -> None:
        self.calls: list[bytes] = []
        self.body, self.status = body, status

    @property
    def credentials(self) -> str:
        raise AssertionError("CONFIG CANARY")

    async def exchange(self, body: bytes) -> LocalByteResponse:
        self.calls.append(body)
        return LocalByteResponse(self.body, self.status)


async def test_render_preparation_and_explicit_call_are_distinct_body_free_facts() -> None:
    invocation = await prepared()
    initial = invocation.snapshot()
    assert initial.source_kind == "caller_bytes"
    assert initial.attempts == ()
    assert invocation.rendered().render_id == CASE["render_id"]
    body = b'{"settings":{"temperature":0.7},"message":"REQUEST CANARY"}'
    baseline = DeclaredBodyPin(sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body))
    fake = Fake()
    result = await capture_once(
        invocation, body, fake, parents=NO_PARENT, declared_baseline=baseline
    )
    record = result.attempt
    assert fake.calls == [body]
    assert result.response_body() == b"BODY CANARY"
    assert result.status == 200
    assert record.facts.request_captured
    assert record.facts.response_observed
    assert record.baseline_relation == "same_bytes"
    assert record.render_relation == "unknown"
    assert record.source_kind == "caller_bytes"
    assert record.provider is None
    assert record.cost is None
    assert record.request is not None
    assert record.request.original_sha256 == hashlib.sha256(body).hexdigest()
    for value in (invocation, result, LocalByteResponse(b"BODY CANARY", 200)):
        assert "CANARY" not in repr(value)
        with pytest.raises(PromptCaptureError, match="capture_invalid"):
            pickle.dumps(value)
        with pytest.raises(PromptCaptureError, match="capture_invalid"):
            copy.copy(value)
        with pytest.raises(TypeError):
            vars(value)
    assert "CANARY" not in invocation.snapshot().canonical().decode()
    invocation.close()
    invocation.close()
    assert invocation.snapshot().closed
    with pytest.raises(PromptCaptureError, match="capture_closed"):
        invocation.rendered()


async def test_empty_body_and_zero_status_are_observed_not_missing() -> None:
    invocation = await prepared()
    result = await capture_once(
        invocation, b"", Fake(b"", 0), parents=NO_PARENT, declared_baseline=None
    )
    assert result.status == 0
    assert result.attempt.response is not None
    assert result.attempt.response.size_bytes == 0
    assert result.attempt.response.original_sha256 == hashlib.sha256(b"").hexdigest()
    assert result.attempt.facts.response_observed
    assert result.attempt.baseline_relation == "not_supplied"


async def test_modified_baseline_and_explicit_retry_links_preserve_original_render() -> None:
    invocation = await prepared()
    baseline = DeclaredBodyPin(sha256=hashlib.sha256(b"original").hexdigest(), size_bytes=8)
    first = await capture_once(
        invocation, b"original", Fake(status=429), parents=NO_PARENT, declared_baseline=baseline
    )
    second = await capture_once(
        invocation,
        b"modified",
        Fake(),
        parents=AttemptParents(
            retry_of=first.attempt.attempt_id, fallback_of=None, continuation_of=None
        ),
        declared_baseline=baseline,
    )
    assert first.attempt.outcome == "observed_response"
    assert first.status == 429
    assert first.attempt.attempt_id != second.attempt.attempt_id
    assert first.attempt.render_id == second.attempt.render_id
    assert second.attempt.baseline_relation == "changed_bytes"
    assert len(invocation.snapshot().attempts) == 2
    foreign = await prepared()
    fake = Fake()
    with pytest.raises(PromptCaptureError, match="capture_invalid"):
        await capture_once(
            foreign,
            b"x",
            fake,
            parents=AttemptParents(
                retry_of=first.attempt.attempt_id, fallback_of=None, continuation_of=None
            ),
            declared_baseline=None,
        )
    assert fake.calls == []


@pytest.mark.parametrize("failure", [RuntimeError("ERROR CANARY"), TimeoutError("ERROR CANARY")])
async def test_native_failure_is_fixed_without_context_or_body(failure: Exception) -> None:
    class Broken:
        async def exchange(self, _body: bytes) -> LocalByteResponse:
            raise failure

    invocation = await prepared()
    with pytest.raises(PromptCaptureError) as caught:
        await capture_once(
            invocation, b"REQUEST CANARY", Broken(), parents=NO_PARENT, declared_baseline=None
        )
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "CANARY" not in str(caught.value)
    record = invocation.snapshot().attempts[0]
    assert record.outcome == "transport_error"
    assert record.response is None
    assert not record.facts.response_observed
    assert "CANARY" not in record.canonical().decode()


async def test_owned_child_is_cancelled_once_and_slot_held_through_repeated_parent_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, released = asyncio.Event(), asyncio.Event()
    children: list[asyncio.Task[object]] = []
    original = asyncio.create_task

    def creating(coroutine: Coroutine[object, object, object]) -> asyncio.Task[object]:
        child = original(coroutine)
        children.append(child)
        return child

    class Held:
        calls = 0
        cancellations = 0

        async def exchange(self, _body: bytes) -> LocalByteResponse:
            self.calls += 1
            entered.set()
            try:
                await released.wait()
            except asyncio.CancelledError:
                self.cancellations += 1
                await released.wait()
            return LocalByteResponse(b"LATE BODY CANARY", 200)

    invocation = await prepared()
    monkeypatch.setattr(runtime.asyncio, "create_task", creating)
    fake = Held()
    task = asyncio.get_running_loop().create_task(
        capture_once(invocation, b"x", fake, parents=NO_PARENT, declared_baseline=None)
    )
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert len(children) == 1
    assert fake.cancellations == 1
    with pytest.raises(PromptCaptureError, match="capture_in_progress"):
        invocation.close()
    with pytest.raises(PromptCaptureError, match="capture_in_progress"):
        invocation.snapshot()
    refused = Fake()
    with pytest.raises(PromptCaptureError, match="capture_in_progress"):
        await capture_once(
            invocation, b"another", refused, parents=NO_PARENT, declared_baseline=None
        )
    assert refused.calls == []
    released.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    record = invocation.snapshot().attempts[0]
    assert record.outcome == "cancelled"
    assert record.retired_at is not None
    assert record.response is None
    assert not record.facts.response_observed
    assert "late_return_discarded" in record.gaps
    result = await capture_once(
        invocation, b"after drain", Fake(), parents=NO_PARENT, declared_baseline=None
    )
    assert result.attempt.outcome == "observed_response"
    assert len(children) == 2


async def test_cancel_before_handoff_keeps_declared_baseline_unobserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runtime._delegate  # pyright: ignore[reportPrivateUsage] - pause the owned pre-handoff child

    async def parked(*args: object) -> object:
        await asyncio.Event().wait()
        return await original(*args)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(runtime, "_delegate", parked)
    invocation = await prepared()
    fake = Fake()
    pin = DeclaredBodyPin(sha256=hashlib.sha256(b"x").hexdigest(), size_bytes=1)
    task = asyncio.create_task(
        capture_once(invocation, b"x", fake, parents=NO_PARENT, declared_baseline=pin)
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    record = invocation.snapshot().attempts[0]
    assert fake.calls == []
    assert record.request is None
    assert not record.facts.request_captured
    assert record.baseline_relation == "not_observed"
    assert record.declared_baseline == pin


async def test_caps_and_forged_handles_refuse_before_dispatch() -> None:
    invocation = await prepared()
    fake = Fake()
    with pytest.raises(PromptCaptureError, match="capture_limit"):
        await capture_once(
            invocation,
            b"x" * (16 * 1024 * 1024 + 1),
            fake,
            parents=NO_PARENT,
            declared_baseline=None,
        )
    assert fake.calls == []
    assert invocation.snapshot().attempts == ()
    forged = object.__new__(PreparedInvocation)
    assert "CANARY" not in repr(forged)
    with pytest.raises(PromptCaptureError, match="capture_invalid"):
        await capture_once(forged, b"x", fake, parents=NO_PARENT, declared_baseline=None)
    with pytest.raises(PromptCaptureError, match="capture_response_limit"):
        await capture_once(
            invocation,
            b"x",
            Fake(b"x" * (16 * 1024 * 1024 + 1)),
            parents=NO_PARENT,
            declared_baseline=None,
        )
    record = invocation.snapshot().attempts[0]
    assert record.response is not None
    assert record.response.coverage == "omitted_oversize"
    assert record.response.original_sha256 is None


async def test_child_can_neither_reenter_nor_close_its_active_invocation() -> None:
    invocation = await prepared()

    class Reentrant:
        async def exchange(self, _body: bytes) -> LocalByteResponse:
            with pytest.raises(PromptCaptureError, match="capture_in_progress"):
                invocation.close()
            with pytest.raises(PromptCaptureError, match="capture_in_progress"):
                await capture_once(
                    invocation, b"nested", Fake(), parents=NO_PARENT, declared_baseline=None
                )
            return LocalByteResponse(b"ok", 200)

    await capture_once(invocation, b"x", Reentrant(), parents=NO_PARENT, declared_baseline=None)
    assert len(invocation.snapshot().attempts) == 1


@pytest.mark.parametrize("shape", ["missing", "noncallable", "raising"])
async def test_invalid_delegate_refuses_without_ticket_or_raw_error(shape: str) -> None:
    class Noncallable:
        exchange = 42

    class Raising:
        @property
        def exchange(self) -> object:
            raise RuntimeError("LOOKUP CANARY")

    value = (
        object() if shape == "missing" else Noncallable() if shape == "noncallable" else Raising()
    )
    invocation = await prepared()
    with pytest.raises(PromptCaptureError, match=r"^capture_invalid$") as caught:
        await capture_once(
            invocation,
            b"fictional",
            cast("LocalByteTransport", value),
            parents=NO_PARENT,
            declared_baseline=None,
        )
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert caught.value.attempt_id is None
    assert invocation.snapshot().attempts == ()


async def test_declared_exchange_is_resolved_once_and_frozen_before_child_entry() -> None:
    class Once:
        lookups = 0
        calls = 0

        @property
        def exchange(self) -> object:
            self.lookups += 1
            assert self.lookups == 1
            return self.perform

        async def perform(self, body: bytes) -> LocalByteResponse:
            self.calls += 1
            assert body == b"fictional"
            return LocalByteResponse(b"response", 200)

    delegate = Once()
    invocation = await prepared()
    result = await capture_once(
        invocation,
        b"fictional",
        cast("LocalByteTransport", delegate),
        parents=NO_PARENT,
        declared_baseline=None,
    )
    assert delegate.lookups == delegate.calls == 1
    assert result.attempt.facts.request_captured


async def test_real_queued_owner_cancel_before_child_entry_never_dispatches() -> None:
    invocation = await prepared()
    fake = Fake()
    pin = DeclaredBodyPin(sha256=hashlib.sha256(b"fictional").hexdigest(), size_bytes=9)

    async def worker() -> None:
        owner = asyncio.current_task()
        assert owner is not None
        asyncio.get_running_loop().call_soon(owner.cancel)
        await capture_once(invocation, b"fictional", fake, parents=NO_PARENT, declared_baseline=pin)

    task = asyncio.create_task(worker())
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.calls == []
    record = invocation.snapshot().attempts[0]
    assert record.outcome == "cancelled"
    assert record.request is None
    assert not record.facts.request_captured
    assert not record.facts.response_observed
    assert record.baseline_relation == "not_observed"
    assert "request_not_dispatched" in record.gaps


@pytest.mark.parametrize(
    "extra", ["request_not_dispatched", "response_not_observed", "response_oversize"]
)
async def test_reported_metadata_refuses_contradictory_extra_gap(extra: str) -> None:
    from pydantic import ValidationError

    from strata_forge.tracing.prompt_attempt_types import AttemptMetadata

    invocation = await prepared()
    result = await capture_once(invocation, b"x", Fake(), parents=NO_PARENT, declared_baseline=None)
    record = result.attempt
    with pytest.raises(ValidationError):
        AttemptMetadata.model_validate({**record.model_dump(), "gaps": (*record.gaps, extra)})


@pytest.mark.parametrize(
    ("outcome", "error"),
    [("transport_error", "local_transport_failed"), ("capture_error", "capture_response_invalid")],
)
async def test_reported_failure_cannot_assert_handoff_failure_without_a_request(
    outcome: str, error: str
) -> None:
    from pydantic import ValidationError

    from strata_forge.tracing.prompt_attempt_types import AttemptMetadata

    invocation = await prepared()
    result = await capture_once(invocation, b"x", Fake(), parents=NO_PARENT, declared_baseline=None)
    value = {
        **result.attempt.model_dump(),
        "request": None,
        "response": None,
        "delegate_status": None,
        "facts": {
            "retrieved": True,
            "rendered": True,
            "request_captured": False,
            "response_observed": False,
        },
        "outcome": outcome,
        "error_code": error,
        "gaps": (
            "render_projection_unobserved",
            "provider_boundary_unobserved",
            "body_not_retained",
            "request_not_dispatched",
            "response_not_observed",
        ),
    }
    with pytest.raises(ValidationError):
        AttemptMetadata.model_validate(value)
