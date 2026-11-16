"""Actual cross-language fake-byte corpus, caps and cold import without external work."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal, TypedDict, cast

import pytest

from strata_forge.prompts.consumption import LocalBuildSource, consume_build
from strata_forge.tracing import prompt_attempts as runtime
from strata_forge.tracing.prompt_attempt_types import (
    AttemptMetadata,
    AttemptParents,
    DeclaredBodyPin,
)
from strata_forge.tracing.prompt_attempts import (
    LocalByteResponse,
    PromptCaptureError,
    capture_once,
    prepare_invocation,
)


class Parent(TypedDict):
    kind: Literal["retry_of", "fallback_of", "continuation_of"]
    index: int


class Step(TypedDict):
    name: str
    request_utf8_hex: str
    response_utf8_hex: str | None
    returned_status: int | None
    parent: Parent | None
    behavior: Literal["response", "timeout", "cancel"]
    declared_baseline: dict[str, object] | None
    attempt: dict[str, object]
    attempt_canonical_utf8_hex: str
    attempt_sha256: str


class Fixture(TypedDict):
    source_partition: str
    build_id: str
    build_canonical_utf8_hex: str
    values: object
    render_id: str
    render_canonical_utf8_hex: str
    steps: list[Step]
    invocation: dict[str, object]
    invocation_canonical_utf8_hex: str
    invocation_sha256: str


ROOT = Path(__file__).parents[3]
FIXTURE_PATH = ROOT / "tests/fixtures/prompts/local-attempts-v1.json"
FIXTURE = cast("Fixture", json.loads(FIXTURE_PATH.read_text()))
NO_PARENT = AttemptParents(retry_of=None, fallback_of=None, continuation_of=None)


async def prepared() -> runtime.PreparedInvocation:
    raw = bytes.fromhex(FIXTURE["build_canonical_utf8_hex"])
    build = consume_build(
        raw,
        expected_build_id=FIXTURE["build_id"],
        expected_size_bytes=len(raw),
        source=LocalBuildSource(
            kind="caller_bytes", partition=FIXTURE["source_partition"], reference=None
        ),
    )
    return await prepare_invocation(build, FIXTURE["values"])


async def test_actual_fake_delegate_metadata_and_bytes_match_shared_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = iter(f"{index:032x}" for index in range(1, 100))
    monkeypatch.setattr(runtime, "_identifier", lambda: next(ids))
    monkeypatch.setattr(runtime, "_now", lambda: "2026-11-07T00:00:00.000000Z")
    monkeypatch.setattr(runtime.time, "monotonic_ns", lambda: 1000)
    invocation = await prepared()
    assert invocation.rendered().canonical.hex() == FIXTURE["render_canonical_utf8_hex"]
    assert invocation.rendered().render_id == FIXTURE["render_id"]
    previous: list[str] = []
    for step in FIXTURE["steps"]:
        entered = asyncio.Event()
        request = bytes.fromhex(step["request_utf8_hex"])
        response_hex = step["response_utf8_hex"]
        response = None if response_hex is None else bytes.fromhex(response_hex)
        calls: list[bytes] = []

        class Fake:
            def __init__(
                self,
                current: Step,
                sent: bytes,
                received: bytes | None,
                history: list[bytes],
                event: asyncio.Event,
            ) -> None:
                self.step, self.request, self.response = current, sent, received
                self.calls, self.entered = history, event

            @property
            def credentials(self) -> str:
                raise AssertionError("CONFIG CANARY")

            async def exchange(self, body: bytes) -> LocalByteResponse:
                assert body == self.request
                self.calls.append(body)
                self.entered.set()
                if self.step["behavior"] == "timeout":
                    raise TimeoutError("ERROR CANARY")
                if self.step["behavior"] == "cancel":
                    await asyncio.Event().wait()
                assert self.response is not None
                return LocalByteResponse(self.response, self.step["returned_status"])

        parent = step["parent"]
        parent_id = None if parent is None else previous[parent["index"]]
        linkage = AttemptParents(
            retry_of=parent_id if parent is not None and parent["kind"] == "retry_of" else None,
            fallback_of=parent_id
            if parent is not None and parent["kind"] == "fallback_of"
            else None,
            continuation_of=parent_id
            if parent is not None and parent["kind"] == "continuation_of"
            else None,
        )
        pin = (
            None
            if step["declared_baseline"] is None
            else DeclaredBodyPin.model_validate(step["declared_baseline"])
        )
        task = asyncio.create_task(
            capture_once(
                invocation,
                request,
                Fake(step, request, response, calls, entered),
                parents=linkage,
                declared_baseline=pin,
            )
        )
        if step["behavior"] == "cancel":
            await entered.wait()
            task.cancel()
        if step["behavior"] == "response":
            result = await task
            assert result.response_body() == response
        else:
            with pytest.raises(
                PromptCaptureError if step["behavior"] == "timeout" else asyncio.CancelledError
            ):
                await task
        assert calls == [request]
        record = invocation.snapshot().attempts[-1]
        assert record.model_dump(mode="json") == step["attempt"]
        assert record.canonical().hex() == step["attempt_canonical_utf8_hex"]
        assert record.metadata_sha256 == step["attempt_sha256"]
        previous.append(record.attempt_id)
    ledger = invocation.snapshot()
    assert ledger.model_dump(mode="json") == FIXTURE["invocation"]
    assert ledger.canonical().hex() == FIXTURE["invocation_canonical_utf8_hex"]
    assert ledger.metadata_sha256 == FIXTURE["invocation_sha256"]
    first_request = json.loads(bytes.fromhex(FIXTURE["steps"][0]["request_utf8_hex"]))
    rendered_output = cast("dict[str, object]", invocation.rendered().wire_payload()["output"])
    assert first_request["messages"] == rendered_output["messages"]
    assert first_request["settings"]["temperature"] == 0.7
    assert first_request["tools"][0]["function"]["parameters"]["properties"] == {
        "value": {"type": "string"}
    }


async def test_exact_body_limit_and_count_limit_preserve_real_zero_and_whole_hashes() -> None:
    invocation = await prepared()
    body = b"x" * (16 * 1024 * 1024)

    class Fake:
        calls = 0

        async def exchange(self, body: bytes) -> LocalByteResponse:
            self.calls += 1
            return LocalByteResponse(body, 0)

    delegate = Fake()
    result = await capture_once(
        invocation, body, delegate, parents=NO_PARENT, declared_baseline=None
    )
    assert result.attempt.response is not None
    assert result.attempt.response.size_bytes == len(body)
    assert result.attempt.response.original_sha256 == hashlib.sha256(body).hexdigest()
    for _ in range(63):
        await capture_once(invocation, b"", delegate, parents=NO_PARENT, declared_baseline=None)
    assert len(invocation.snapshot().attempts) == delegate.calls == 64
    with pytest.raises(PromptCaptureError, match="capture_limit"):
        await capture_once(invocation, b"", delegate, parents=NO_PARENT, declared_baseline=None)
    assert delegate.calls == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("duration_us", 0.7),
        ("provider", "CANARY"),
        ("started_at", "2026-02-30T00:00:00.000000Z"),
        ("attempt_id", "a" * 32 + "\n"),
        ("retired_at", "2026-11-07T00:00:00.000000Z"),
    ],
)
def test_golden_parser_rejects_unavailable_or_nonportable_facts(field: str, value: object) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as caught:
        AttemptMetadata.model_validate({**FIXTURE["steps"][0]["attempt"], field: value})
    assert "CANARY" not in str(caught.value)


def test_cold_import_and_actual_local_capture_have_no_network_export_or_heavy_import() -> None:
    program = r"""
import asyncio, importlib.abc, json, pathlib, socket, sys
fixture = json.loads(pathlib.Path(sys.argv[1]).read_text())
attempts = []
def denied(*args, **kwargs):
    attempts.append("network_or_asset")
    raise AssertionError("forbidden")
socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied
socket.getaddrinfo = denied
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch","transformers","trl","peft","numpy","langfuse","pyarrow","skypilot"}:
            attempts.append(fullname)
            raise AssertionError("optional_import_forbidden")
sys.meta_path.insert(0, Guard())
import tiktoken
tiktoken.get_encoding = denied
tiktoken.encoding_for_model = denied
from strata_forge.prompts import LocalBuildSource, consume_build
from strata_forge.tracing import AttemptParents, LocalByteResponse, prepare_invocation, capture_once
async def main():
    raw = bytes.fromhex(fixture["build_canonical_utf8_hex"])
    build = consume_build(raw, expected_build_id=fixture["build_id"], expected_size_bytes=len(raw), source=LocalBuildSource(kind="caller_bytes", partition="fixture:local", reference=None))
    invocation = await prepare_invocation(build, fixture["values"])
    class Fake:
        async def exchange(self, body):
            return LocalByteResponse(body, 200)
    result = await capture_once(invocation, b"fictional", Fake(), parents=AttemptParents(retry_of=None, fallback_of=None, continuation_of=None), declared_baseline=None)
    assert result.attempt.facts.response_observed
asyncio.run(main())
assert attempts == [], attempts
print("local-only")
"""
    result = subprocess.run(  # noqa: S603 - fixed local program/synthetic bytes only
        [sys.executable, "-c", program, str(FIXTURE_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "local-only"


async def test_metadata_budget_is_reserved_before_calling_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocation = await prepared()
    monkeypatch.setattr(
        runtime, "MAX_INVOCATION_METADATA_BYTES", len(invocation.snapshot().canonical()) + 4095
    )

    class Forbidden:
        async def exchange(self, _body: bytes) -> LocalByteResponse:
            pytest.fail("budget must refuse before dispatch")

    with pytest.raises(PromptCaptureError, match="capture_limit"):
        await capture_once(invocation, b"x", Forbidden(), parents=NO_PARENT, declared_baseline=None)
    assert invocation.snapshot().attempts == ()


async def test_cancelled_late_native_error_is_drained_without_unhandled_diagnostics() -> None:
    invocation = await prepared()
    entered, release = asyncio.Event(), asyncio.Event()
    diagnostics: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))

    class Held:
        async def exchange(self, _body: bytes) -> LocalByteResponse:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            raise RuntimeError("LATE ERROR CANARY")

    try:
        task = asyncio.create_task(
            capture_once(invocation, b"x", Held(), parents=NO_PARENT, declared_baseline=None)
        )
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        record = invocation.snapshot().attempts[0]
        assert record.response is None
        assert record.outcome == "cancelled"
        assert "late_return_discarded" not in record.gaps
        assert diagnostics == []
    finally:
        loop.set_exception_handler(previous)
