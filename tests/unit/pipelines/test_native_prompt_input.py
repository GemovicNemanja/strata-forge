"""Synthetic native prompt candidates reuse real immutable reads and modular rendering."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import FrozenInstanceError, replace
from typing import TYPE_CHECKING, Literal

import pytest

from strata_forge.pipelines import _native_prompt_input as implementation
from strata_forge.pipelines._immutable_input import HumanRenderInputReader
from strata_forge.pipelines._native_prompt_input import (
    MAX_NATIVE_RENDER_BYTES,
    MAX_NATIVE_VALUES_BYTES,
    NativePromptInputError,
    consume_native_prompt_input,
    native_prompt_intent_digest,
    render_native_prompt_consumed,
    render_native_prompt_input,
)
from strata_forge.pipelines.immutable_input_types import (
    HumanRenderInputDecision,
    HumanRenderInputRequest,
    ImmutableInputDecision,
    ImmutableInputError,
    ImmutableInputRequest,
    NativeManifestPayload,
    NativeVersionRef,
    ResolvedHumanRenderPayload,
    checked_decision,
    checked_human_render_decision,
    checked_human_render_request,
    checked_request,
)
from strata_forge.prompts.composition import (
    CompositionRegistry,
    PromptBuild,
    PromptRender,
    compile_surface,
    render_build,
)
from strata_forge.prompts.consumption import LocalBuildSource, consume_build, render_consumed
from strata_forge.prompts.portable import canonical_bytes

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

pytestmark = pytest.mark.asyncio
PRIVATE = "synthetic-private-canary"


def surface_ref() -> NativeVersionRef:
    return NativeVersionRef(resource_id="a" * 32, version_id="b" * 32, manifest_sha256="c" * 64)


async def build(
    *,
    artifact: bool = False,
    repeats: int = 1,
    value_type: Literal["string", "integer", "boolean"] = "string",
) -> PromptBuild:
    parts = [{"kind": "variable", "name": "s"} for _ in range(repeats)]
    target: dict[str, object]
    if artifact:
        target = {
            "kind": "artifact",
            "purpose": "tool_description",
            "parts": parts,
            "separator": "",
            "descriptor": {"name": "synthetic", "description": "literal {s}"},
        }
    else:
        target = {
            "kind": "chat",
            "messages": [{"role": "system", "parts": parts, "separator": ""}],
        }
    registry = CompositionRegistry()
    await registry.put(
        {
            "kind": "surface",
            "name": "synthetic",
            "revision": 1,
            "variables": [{"name": "s", "type": value_type}],
            "target": target,
        }
    )
    return await compile_surface(registry, "synthetic@1")


def request(raw: bytes, values: object) -> HumanRenderInputRequest:
    ref = NativeVersionRef(resource_id="5" * 32, version_id="6" * 32, manifest_sha256="d" * 64)
    payload = NativeManifestPayload(
        artifact_id="7" * 32,
        logical_name="build.json",
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        media_type="application/json",
        schema_id="strata.prompt.build.v1",
    )
    digest = native_prompt_intent_digest(
        principal_id="1" * 32,
        project_id="2" * 32,
        operation_id="3" * 32,
        attempt_id="4" * 32,
        surface_ref=surface_ref(),
        version_ref=ref,
        payload=payload,
        values=values,
    )
    return HumanRenderInputRequest(
        version=1,
        admission="human_render",
        principal_id="1" * 32,
        project_id="2" * 32,
        operation_id="3" * 32,
        attempt_id="4" * 32,
        purpose="prompt_render",
        spec_sha256=digest,
        kind="prompt_build",
        version_ref=ref,
        payload=payload,
    )


class Authority:
    def __init__(self, *, effect: Literal["allow", "deny"] = "allow") -> None:
        self.effect: Literal["allow", "deny"] = effect
        self.calls = 0
        self.owner: asyncio.Task[object] | None = None

    async def decide(self, selected: HumanRenderInputRequest) -> HumanRenderInputDecision:
        self.calls += 1
        if self.owner is None:
            self.owner = asyncio.current_task()
        assert self.owner is asyncio.current_task()
        return HumanRenderInputDecision(
            request=selected,
            effect=self.effect,
            custody="unsealed",
            policy_identity="8" * 32,
            policy_revision=1,
            expires_at_ms=time.time_ns() // 1_000_000 + 60_000,
        )


class Lease:
    def __init__(
        self,
        selected: HumanRenderInputRequest,
        raw: bytes | None,
        fd: int | None,
        on_resolve: Callable[[], Awaitable[None]] | None,
    ) -> None:
        self.selected, self.raw, self.fd = selected, raw, fd
        self.on_resolve = on_resolve
        self.closed = False

    async def resolve(self) -> ResolvedHumanRenderPayload:
        if self.on_resolve is not None:
            await self.on_resolve()
        return ResolvedHumanRenderPayload(self.selected, self.raw, self.fd)

    async def aclose(self) -> None:
        self.closed = True


class Source:
    def __init__(self, raw: bytes | None, *, fd: int | None = None) -> None:
        self.raw, self.fd = raw, fd
        self.leases: list[Lease] = []
        self.on_resolve: Callable[[], Awaitable[None]] | None = None

    def lease(self, selected: HumanRenderInputRequest) -> Lease:
        lease = Lease(selected, self.raw, self.fd, self.on_resolve)
        self.leases.append(lease)
        return lease


async def refused(
    reader: HumanRenderInputReader,
    selected: HumanRenderInputRequest,
    values: object,
    *,
    surface: NativeVersionRef | None = None,
    deadline: float | None = None,
    code: str | None = None,
) -> None:
    with pytest.raises((NativePromptInputError, ImmutableInputError)) as caught:
        await render_native_prompt_input(
            reader,
            selected,
            surface_ref=surface_ref() if surface is None else surface,
            values=values,
            deadline=time.monotonic() + 12.0 if deadline is None else deadline,
        )
    if code is not None:
        assert str(caught.value) == code
    assert PRIVATE not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("artifact", [False, True])
@pytest.mark.parametrize("held_fd", [False, True])
async def test_real_reader_candidate_matches_existing_local_render(
    artifact: bool, held_fd: bool, tmp_path: Path
) -> None:
    compiled = await build(artifact=artifact)
    raw = compiled.canonical
    values = {"s": "literal {{s}} éé\r\n"}
    selected = request(raw, values)
    path = tmp_path / "synthetic-build.json"
    path.write_bytes(raw)
    fd = os.open(path, os.O_RDONLY) if held_fd else None
    source = Source(None if held_fd else raw, fd=fd)
    authority = Authority()
    reader = HumanRenderInputReader(authority, source)
    try:
        candidate = await render_native_prompt_input(
            reader,
            selected,
            surface_ref=surface_ref(),
            values=values,
            deadline=time.monotonic() + 12.0,
        )
        local = consume_build(
            raw,
            expected_build_id=compiled.build_id,
            expected_size_bytes=len(raw),
            source=LocalBuildSource(
                kind="caller_bytes", partition="synthetic:local", reference=None
            ),
        )
        direct = await render_consumed(local, values)
        assert candidate.render.canonical == direct.canonical
        assert candidate.render.canonical == (await render_build(compiled, values)).canonical
        assert candidate.request == selected
        assert candidate.request is not selected
        assert candidate.surface_ref == surface_ref()
        assert values["s"] not in repr(candidate)
        assert "canonical=" not in repr(candidate)
        assert source.leases[0].closed
        assert authority.calls >= 3
        if fd is not None:
            assert os.pread(fd, len(raw), 0) == raw
    finally:
        await reader.aclose()
        if fd is not None:
            os.close(fd)


async def test_intent_is_complete_portable_canonical_human_variant() -> None:
    values = {"s": "é\r\n{literal}"}
    selected = request((await build()).canonical, values)
    wire = selected.model_dump(mode="json", exclude={"spec_sha256"})
    wire.update(
        {
            "domain": "strata-human-prompt-render-intent-v1",
            "surface_ref": surface_ref().model_dump(mode="json"),
            "values": values,
        }
    )
    assert selected.spec_sha256 == hashlib.sha256(canonical_bytes(wire)).hexdigest()
    assert wire["admission"] == "human_render"
    assert "run_id" not in wire
    assert wire["operation_id"] == "3" * 32


async def test_human_and_durable_run_admissions_and_decisions_cannot_be_relabelled() -> None:
    human = request((await build()).canonical, {"s": PRIVATE})
    fields = human.model_dump(exclude={"admission", "operation_id"})
    legacy = ImmutableInputRequest.model_validate({**fields, "run_id": human.operation_id})
    with pytest.raises(ImmutableInputError, match="input_invalid"):
        checked_request(human)
    with pytest.raises(ImmutableInputError, match="input_invalid"):
        checked_human_render_request(legacy)
    human_decision = await Authority().decide(human)
    legacy_decision = ImmutableInputDecision.model_validate(
        {**human_decision.model_dump(exclude={"request"}), "request": legacy}
    )
    with pytest.raises(ImmutableInputError, match="input_authority_unavailable"):
        checked_human_render_decision(legacy_decision)
    with pytest.raises(ImmutableInputError, match="input_authority_unavailable"):
        checked_decision(human_decision)


@pytest.mark.parametrize(
    ("value_type", "value"), [("integer", -9007199254740991), ("boolean", False)]
)
async def test_portable_integer_and_boolean_values_keep_existing_scalar_rendering(
    value_type: Literal["integer", "boolean"],
    value: int | bool,
) -> None:
    compiled = await build(value_type=value_type)
    values = {"s": value}
    reader = HumanRenderInputReader(Authority(), Source(compiled.canonical))
    try:
        candidate = await render_native_prompt_input(
            reader,
            request(compiled.canonical, values),
            surface_ref=surface_ref(),
            values=values,
            deadline=time.monotonic() + 12.0,
        )
        assert candidate.render.canonical == (await render_build(compiled, values)).canonical
    finally:
        await reader.aclose()


@pytest.mark.parametrize(
    "field",
    [
        "principal_id",
        "project_id",
        "operation_id",
        "attempt_id",
        "spec_sha256",
        "resource_id",
        "version_id",
        "manifest_sha256",
        "artifact_id",
        "sha256",
        "size_bytes",
    ],
)
async def test_all_request_and_payload_fields_bind_before_source(field: str) -> None:
    compiled = await build()
    selected = request(compiled.canonical, {"s": PRIVATE})
    if field in {"resource_id", "version_id", "manifest_sha256"}:
        width = 64 if field == "manifest_sha256" else 32
        selected = selected.model_copy(
            update={"version_ref": selected.version_ref.model_copy(update={field: "9" * width})}
        )
    elif field in {"artifact_id", "sha256", "size_bytes"}:
        replacement = (
            selected.payload.size_bytes + 1
            if field == "size_bytes"
            else "9" * (64 if field == "sha256" else 32)
        )
        selected = selected.model_copy(
            update={"payload": selected.payload.model_copy(update={field: replacement})}
        )
    else:
        selected = selected.model_copy(update={field: "9" * (64 if field == "spec_sha256" else 32)})
    source, authority = Source(compiled.canonical), Authority()
    reader = HumanRenderInputReader(authority, source)
    try:
        await refused(reader, selected, {"s": PRIVATE}, code="native_prompt_intent_mismatch")
        assert not source.leases
        assert authority.calls == 0
    finally:
        await reader.aclose()


@pytest.mark.parametrize("field", ["resource_id", "version_id", "manifest_sha256"])
async def test_full_surface_reference_is_bound_before_source(field: str) -> None:
    raw = (await build()).canonical
    selected = request(raw, {"s": PRIVATE})
    changed = surface_ref().model_copy(
        update={field: "9" * (64 if field == "manifest_sha256" else 32)}
    )
    source = Source(raw)
    reader = HumanRenderInputReader(Authority(), source)
    try:
        await refused(reader, selected, {"s": PRIVATE}, surface=changed)
        assert not source.leases
    finally:
        await reader.aclose()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("admission", "run"),
        ("purpose", "evaluation"),
        ("kind", "dataset"),
        ("operation_id", PRIVATE),
    ],
)
async def test_wrong_admission_purpose_and_constructed_fields_refuse(
    field: str, value: object
) -> None:
    raw = (await build()).canonical
    selected = request(raw, {"s": PRIVATE}).model_copy(update={field: value})
    source, authority = Source(raw), Authority()
    reader = HumanRenderInputReader(authority, source)
    try:
        await refused(reader, selected, {"s": PRIVATE})
        assert not source.leases
        assert authority.calls == 0
    finally:
        await reader.aclose()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("logical_name", "other.json"),
        ("media_type", "text/plain"),
        ("schema_id", "synthetic.other.v1"),
        ("size_bytes", 0),
        ("size_bytes", 8 * 1024 * 1024 + 1),
    ],
)
async def test_only_exact_retained_build_descriptor_is_admitted(field: str, value: object) -> None:
    raw = (await build()).canonical
    selected = request(raw, {"s": PRIVATE})
    changed = selected.model_copy(
        update={"payload": selected.payload.model_copy(update={field: value})}
    )
    source = Source(raw)
    reader = HumanRenderInputReader(Authority(), source)
    try:
        await refused(reader, changed, {"s": PRIVATE})
        assert not source.leases
    finally:
        await reader.aclose()


@pytest.mark.parametrize(
    "values",
    [
        {"s": None},
        {"s": 1.0},
        {"s": []},
        {"s": {}},
        {"s": 2**53},
        {"s": "\ud800"},
        {"s": "x" * MAX_NATIVE_VALUES_BYTES},
        {str(index): "x" for index in range(129)},
        {1: PRIVATE},
        [],
    ],
)
async def test_non_scalar_nonportable_or_oversized_values_refuse_without_read(
    values: object,
) -> None:
    raw = (await build()).canonical
    selected = request(raw, {"s": PRIVATE})
    source = Source(raw)
    reader = HumanRenderInputReader(Authority(), source)
    try:
        await refused(reader, selected, values)
        assert not source.leases
    finally:
        await reader.aclose()


async def test_exact_value_byte_and_key_limits_are_inclusive() -> None:
    raw = (await build()).canonical
    value = "x" * (MAX_NATIVE_VALUES_BYTES - len(canonical_bytes({"s": ""})))
    assert len(canonical_bytes({"s": value})) == MAX_NATIVE_VALUES_BYTES
    assert len(request(raw, {"s": value}).spec_sha256) == 64
    assert len(request(raw, {str(index): False for index in range(128)}).spec_sha256) == 64


async def test_caller_mutation_during_resolution_cannot_change_rendered_values() -> None:
    compiled = await build()
    values = {"s": "original literal {s}"}
    selected = request(compiled.canonical, values)
    source = Source(compiled.canonical)

    async def mutate() -> None:
        values["s"] = PRIVATE

    source.on_resolve = mutate
    reader = HumanRenderInputReader(Authority(), source)
    try:
        candidate = await render_native_prompt_input(
            reader,
            selected,
            surface_ref=surface_ref(),
            values=values,
            deadline=time.monotonic() + 12.0,
        )
        expected = await render_build(compiled, {"s": "original literal {s}"})
        assert candidate.render.canonical == expected.canonical
        assert PRIVATE.encode() not in candidate.render.canonical
    finally:
        await reader.aclose()


async def test_current_denial_is_not_overridden_by_a_matching_intent_digest() -> None:
    raw = (await build()).canonical
    source = Source(raw)
    reader = HumanRenderInputReader(Authority(effect="deny"), source)
    try:
        await refused(reader, request(raw, {"s": PRIVATE}), {"s": PRIVATE}, code="input_denied")
        assert not source.leases
    finally:
        await reader.aclose()


@pytest.mark.parametrize("deadline", [float("inf"), float("nan"), -1.0])
async def test_invalid_or_elapsed_original_deadline_cannot_begin_source(deadline: float) -> None:
    raw = (await build()).canonical
    source = Source(raw)
    reader = HumanRenderInputReader(Authority(), source)
    try:
        await refused(reader, request(raw, {"s": PRIVATE}), {"s": PRIVATE}, deadline=deadline)
        assert not source.leases
    finally:
        await reader.aclose()


async def test_deadline_after_synchronous_render_refuses_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = (await build()).canonical
    source = Source(raw)
    reader = HumanRenderInputReader(Authority(), source)
    deadline = time.monotonic() + 12.0
    original = implementation.render_consumed

    class ExpiredClock:
        @staticmethod
        def monotonic() -> float:
            return deadline

    async def expired_after_render(*args: object) -> PromptRender:
        result = await original(*args)  # pyright: ignore[reportArgumentType] -- fixed render interception exercises the post-render deadline.
        monkeypatch.setattr(implementation, "time", ExpiredClock)
        return result

    monkeypatch.setattr(implementation, "render_consumed", expired_after_render)
    try:
        await refused(
            reader,
            request(raw, {"s": PRIVATE}),
            {"s": PRIVATE},
            deadline=deadline,
            code="native_prompt_deadline",
        )
        assert source.leases[0].closed
    finally:
        await reader.aclose()


async def test_cancellation_after_render_cannot_return_late_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = (await build()).canonical
    reader = HumanRenderInputReader(Authority(), Source(raw))
    original = implementation.render_consumed

    async def cancel_after_render(*args: object) -> PromptRender:
        result = await original(*args)  # pyright: ignore[reportArgumentType] -- fixed render interception exercises pending owner cancellation.
        owner = asyncio.current_task()
        assert owner is not None
        owner.cancel()
        return result

    monkeypatch.setattr(implementation, "render_consumed", cancel_after_render)
    task = asyncio.create_task(
        render_native_prompt_input(
            reader,
            request(raw, {"s": PRIVATE}),
            surface_ref=surface_ref(),
            values={"s": PRIVATE},
            deadline=time.monotonic() + 12.0,
        )
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await reader.aclose()


async def test_cancelled_source_is_drained_and_closed_before_refusal() -> None:
    raw = (await build()).canonical
    source = Source(raw)
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed() -> None:
        entered.set()
        await release.wait()

    source.on_resolve = delayed
    reader = HumanRenderInputReader(Authority(), source)
    task = asyncio.create_task(
        render_native_prompt_input(
            reader,
            request(raw, {"s": PRIVATE}),
            surface_ref=surface_ref(),
            values={"s": PRIVATE},
            deadline=time.monotonic() + 12.0,
        )
    )
    try:
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert source.leases[0].closed
    finally:
        release.set()
        await reader.aclose()


@pytest.mark.parametrize("values", [{}, {"s": True}, {"s": "x", "extra": False}])
async def test_existing_exact_variable_semantics_remain_authoritative(values: object) -> None:
    raw = (await build()).canonical
    source = Source(raw)
    reader = HumanRenderInputReader(Authority(), source)
    try:
        await refused(reader, request(raw, values), values, code="native_prompt_render_invalid")
        assert source.leases[0].closed
    finally:
        await reader.aclose()


async def test_complete_canonical_output_cap_includes_source_map_overhead() -> None:
    compiled = await build(repeats=128)
    value = "x" * (MAX_NATIVE_VALUES_BYTES - len(canonical_bytes({"s": ""})))
    values = {"s": value}
    direct = await render_build(compiled, values)
    assert len(value.encode()) * 128 < MAX_NATIVE_RENDER_BYTES
    assert len(direct.canonical) > MAX_NATIVE_RENDER_BYTES
    source = Source(compiled.canonical)
    reader = HumanRenderInputReader(Authority(), source)
    try:
        await refused(
            reader, request(compiled.canonical, values), values, code="native_prompt_render_limited"
        )
        assert source.leases[0].closed
    finally:
        await reader.aclose()


async def test_schema_invalid_matching_bytes_never_leak_renderer_diagnostics() -> None:
    raw = canonical_bytes({"private": PRIVATE})
    source = Source(raw)
    reader = HumanRenderInputReader(Authority(), source)
    try:
        await refused(
            reader,
            request(raw, {"s": PRIVATE}),
            {"s": PRIVATE},
            code="native_prompt_render_invalid",
        )
        assert source.leases[0].closed
    finally:
        await reader.aclose()


async def test_staged_candidate_reuses_one_read_and_matches_direct_render() -> None:
    compiled = await build(artifact=True)
    values = {"s": "frozen literal {s}"}
    source = Source(compiled.canonical)
    reader = HumanRenderInputReader(Authority(), source)
    try:
        candidate = await consume_native_prompt_input(
            reader,
            request(compiled.canonical, values),
            surface_ref=surface_ref(),
            values=values,
            deadline=time.monotonic() + 12.0,
        )
        assert candidate.consumed.canonical == compiled.canonical
        assert candidate.values_json == canonical_bytes(values)
        assert len(source.leases) == 1
        assert source.leases[0].closed
        assert values["s"] not in repr(candidate)
        # Retained native pins are checked by the server here, before the separate render stage.
        values["s"] = PRIVATE
        rendered = await render_native_prompt_consumed(candidate, deadline=time.monotonic() + 120.0)
        assert len(source.leases) == 1
        assert (
            rendered.render.canonical
            == (await render_build(compiled, {"s": "frozen literal {s}"})).canonical
        )
    finally:
        await reader.aclose()


async def test_modified_staged_values_refuse_even_with_recomputed_request_digest() -> None:
    compiled = await build()
    values = {"s": "original"}
    reader = HumanRenderInputReader(Authority(), Source(compiled.canonical))
    try:
        candidate = await consume_native_prompt_input(
            reader,
            request(compiled.canonical, values),
            surface_ref=surface_ref(),
            values=values,
            deadline=time.monotonic() + 12.0,
        )
        modified = replace(
            candidate,
            values_json=canonical_bytes({"s": PRIVATE}),
            request=request(compiled.canonical, {"s": PRIVATE}),
        )
        with pytest.raises(NativePromptInputError, match="native_prompt_intent_mismatch") as caught:
            await render_native_prompt_consumed(modified, deadline=time.monotonic() + 12.0)
        assert PRIVATE not in str(caught.value)
        assert caught.value.__context__ is None
    finally:
        await reader.aclose()


async def test_original_stage_deadline_is_read_only_and_cannot_be_renewed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = await build()
    source = Source(compiled.canonical)
    reader = HumanRenderInputReader(Authority(), source)
    try:
        candidate = await consume_native_prompt_input(
            reader,
            request(compiled.canonical, {"s": PRIVATE}),
            surface_ref=surface_ref(),
            values={"s": PRIVATE},
            deadline=time.monotonic() + 12.0,
        )
        with pytest.raises(FrozenInstanceError):
            candidate.original_deadline = candidate.original_deadline + 120.0  # pyright: ignore[reportAttributeAccessIssue] -- the read-only budget is intentionally challenged.

        class ExpiredClock:
            @staticmethod
            def monotonic() -> float:
                return candidate.original_deadline

        monkeypatch.setattr(implementation, "time", ExpiredClock)
        with pytest.raises(NativePromptInputError, match="native_prompt_deadline"):
            await render_native_prompt_consumed(
                candidate, deadline=candidate.original_deadline + 120.0
            )
        assert len(source.leases) == 1
        assert source.leases[0].closed
    finally:
        await reader.aclose()


async def test_staged_candidate_cannot_move_to_another_owner_task() -> None:
    compiled = await build()
    reader = HumanRenderInputReader(Authority(), Source(compiled.canonical))
    try:
        candidate = await consume_native_prompt_input(
            reader,
            request(compiled.canonical, {"s": PRIVATE}),
            surface_ref=surface_ref(),
            values={"s": PRIVATE},
            deadline=time.monotonic() + 12.0,
        )
        task = asyncio.create_task(
            render_native_prompt_consumed(candidate, deadline=time.monotonic() + 12.0)
        )
        with pytest.raises(NativePromptInputError, match="native_prompt_closed"):
            await task
    finally:
        await reader.aclose()


@pytest.mark.parametrize("field", ["build_id", "source"])
async def test_staged_consumed_metadata_is_revalidated_before_render(field: str) -> None:
    compiled = await build()
    reader = HumanRenderInputReader(Authority(), Source(compiled.canonical))
    try:
        candidate = await consume_native_prompt_input(
            reader,
            request(compiled.canonical, {"s": PRIVATE}),
            surface_ref=surface_ref(),
            values={"s": PRIVATE},
            deadline=time.monotonic() + 12.0,
        )
        value: object = "0" * 64
        if field == "source":
            value = candidate.consumed.source.model_copy(update={"partition": "changed"})
        modified = replace(candidate, consumed=candidate.consumed.model_copy(update={field: value}))
        with pytest.raises(NativePromptInputError) as caught:
            await render_native_prompt_consumed(modified, deadline=time.monotonic() + 12.0)
        assert PRIVATE not in str(caught.value)
        assert caught.value.__context__ is None
    finally:
        await reader.aclose()
