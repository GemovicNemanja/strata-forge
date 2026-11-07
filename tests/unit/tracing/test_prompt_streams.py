"""Synthetic native protocol controls; no provider request, model or network execution."""

from __future__ import annotations

import asyncio
import copy
import gc
import pickle
import weakref
from typing import (  # Raw ProviderClient interface and adversarial fake values only.
    TYPE_CHECKING,
    Any,
    cast,
)

import pytest
from pydantic import ValidationError

from strata_forge.tracing.prompt_provider_values import project_chat
from strata_forge.tracing.prompt_stream_types import ProviderStreamMetadata, StreamInputBinding
from strata_forge.tracing.prompt_streams import (
    ProviderStreamObservation,
    StreamObservationError,
    prepare_stream_observation,
    wrap_streaming_provider,
)
from tests.unit.tracing.test_prompt_provider import FakeProvider, issued_invocation

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from strata_forge.llm.providers.base import ProviderClient
    from strata_forge.llm.registry import Model
    from strata_forge.tracing.prompt_attempts import PreparedInvocation


class StreamFake(FakeProvider):
    def __init__(self, native: object) -> None:
        super().__init__()
        self.native = native
        self.factory_calls = 0

    def astream(
        self, *, provider_model_id: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[Any]:
        self.factory_calls += 1
        return cast("AsyncIterator[Any]", self.native)

    def aresponses_stream(
        self, *, provider_model_id: str, request: dict[str, Any]
    ) -> AsyncIterator[Any]:
        self.factory_calls += 1
        return cast("AsyncIterator[Any]", self.native)


async def observed(native: object) -> tuple[ProviderStreamObservation, StreamFake, Any]:
    observation = await prepare_stream_observation(await issued_invocation())
    fake = StreamFake(native)
    wrapper = wrap_streaming_provider(fake, observation=observation, provider_slot="openai_compat")
    return observation, fake, wrapper


async def sequence(items: list[object]) -> AsyncGenerator[object, object]:
    for item in items:
        yield item


def first(observation: ProviderStreamObservation) -> ProviderStreamMetadata:
    return observation.snapshot().streams[0]


async def test_lazy_construction_and_before_first_drive_input_snapshot() -> None:
    entered: list[bool] = []
    marker = object()

    async def native() -> AsyncGenerator[object, object]:
        entered.append(True)
        yield marker

    observation, fake, wrapper = await observed(native())
    messages = [{"role": "user", "content": "original fictional text"}]
    view = wrapper.astream(provider_model_id="fixture", messages=messages, temperature=1.0)
    assert view.__aiter__() is view
    advance = view.__anext__()
    assert advance.__await__() is advance
    assert iter(advance) is advance
    assert fake.factory_calls == 1
    assert entered == []
    before = first(observation)
    assert before.input_binding.stage == "unobserved"
    assert before.first_drive_attempt_at is None
    messages[0]["content"] = "changed fictional text"
    with pytest.raises(StopIteration) as done:
        next(advance)
    assert done.value.value is marker
    assert entered == [True]
    record = first(observation)
    assert record.input_binding.projection == project_chat(
        "fixture", messages, {"temperature": 1.0}
    )
    assert record.raw_items_observed == record.data_advance_attempts == 1
    assert record.first_item_drive_latency_us is not None
    assert "changed fictional text" not in record.canonical().decode()
    await view.aclose()


@pytest.mark.parametrize("value", [None, False, 0, "", b""])
async def test_falsey_raw_items_are_real_items_returned_by_identity(value: object) -> None:
    observation, _, wrapper = await observed(sequence([value]))
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    assert await anext(view) is value
    assert first(observation).raw_items_observed == 1
    with pytest.raises(StopAsyncIteration):
        await anext(view)
    assert first(observation).end_observation == "exhausted"


async def test_asend_athrow_and_original_error_identity() -> None:
    marker, sent, thrown = object(), object(), ValueError("fictional error")

    async def native() -> AsyncGenerator[object, object]:
        value = yield marker
        try:
            yield value
        except ValueError as error:
            yield error
        raise thrown

    observation, _, wrapper = await observed(native())
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    assert await anext(view) is marker
    assert await view.asend(sent) is sent
    assert await view.athrow(thrown) is thrown
    with pytest.raises(ValueError, match="fictional") as caught:
        await anext(view)
    assert caught.value is thrown
    record = first(observation)
    assert record.raw_items_observed == 3
    assert record.data_advance_attempts == 4
    assert record.operation_errors == 1
    assert record.end_observation is None


async def test_aclose_before_data_drive_does_not_project_or_start_body() -> None:
    entered: list[bool] = []

    async def native() -> AsyncGenerator[object, object]:
        entered.append(True)
        yield None

    observation, _, wrapper = await observed(native())
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    close = view.aclose()
    assert first(observation).close_attempts == 0
    await close
    record = first(observation)
    assert entered == []
    assert record.input_binding.stage == "unobserved"
    assert record.close_attempts == 1
    assert record.end_observation == "aclose_completed"
    assert record.first_item_at is None


async def test_suspension_is_native_backpressure_without_observer_tasks_or_item_count() -> None:
    release = asyncio.Event()
    marker = object()

    async def native() -> AsyncGenerator[object, object]:
        await release.wait()
        yield marker

    observation, _, wrapper = await observed(native())
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    advance = view.__anext__()
    existing = asyncio.all_tasks()
    work = asyncio.create_task(advance)
    await asyncio.sleep(0)
    assert asyncio.all_tasks() == existing | {work}
    record = first(observation)
    assert record.raw_items_observed == 0
    assert record.last_operation_outcome == "suspended"
    assert not work.done()
    release.set()
    assert await work is marker
    assert first(observation).raw_items_observed == 1
    await view.aclose()


async def test_reused_and_overlapping_advances_keep_native_refusals_and_live_source() -> None:
    release = asyncio.Event()

    async def native() -> AsyncGenerator[object, object]:
        await release.wait()
        yield None
        yield False

    observation, _, wrapper = await observed(native())
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    advance = view.__anext__()
    work = asyncio.create_task(advance)
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="already running"):
        await anext(view)
    assert first(observation).end_observation is None
    release.set()
    assert await work is None
    with pytest.raises(RuntimeError, match="cannot reuse"):
        await advance
    assert await anext(view) is False
    assert first(observation).raw_items_observed == 2
    await view.aclose()


async def test_throw_cancel_and_observation_retirement_preserve_original_item_and_frozen_metadata() -> (
    None
):
    entered, release = asyncio.Event(), asyncio.Event()
    observed_cancel: list[BaseException] = []
    marker = object()

    async def native() -> AsyncGenerator[object, object]:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError as error:
            observed_cancel.append(error)
            raise
        yield marker

    observation, _, wrapper = await observed(native())
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    work = asyncio.create_task(view.__anext__())
    await entered.wait()
    work.cancel("fictional cancellation")
    with pytest.raises(asyncio.CancelledError) as caught:
        await work
    assert caught.value is observed_cancel[0]
    record = first(observation)
    assert record.operation_cancellations == 1
    assert record.end_observation is None
    observation.close()
    frozen = observation.snapshot().canonical()
    with pytest.raises(StopAsyncIteration):
        await anext(view)
    assert observation.snapshot().canonical() == frozen


async def test_retirement_during_held_advance_does_not_cancel_or_capture_late_item() -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    marker = object()

    async def native() -> AsyncGenerator[object, object]:
        entered.set()
        await release.wait()
        yield marker

    observation, _, wrapper = await observed(native())
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    work = asyncio.create_task(view.__anext__())
    await entered.wait()
    observation.close()
    frozen = observation.snapshot().canonical()
    assert not work.done()
    release.set()
    assert await work is marker
    assert observation.snapshot().canonical() == frozen
    await view.aclose()


async def test_unsupported_factory_return_is_untouched_loss_only() -> None:
    class Unsupported:
        def __aiter__(self) -> object:
            raise AssertionError("no iterator probing")

    native = Unsupported()
    observation, fake, wrapper = await observed(native)
    assert wrapper.astream(provider_model_id="fixture", messages=[]) is native
    snapshot = observation.snapshot()
    assert fake.factory_calls == 1
    assert snapshot.admitted_streams == 0
    assert snapshot.streams == ()
    assert snapshot.loss.unsupported_iterator_count == 1


async def test_safe_working_inspection_and_unissued_handles_refuse() -> None:
    observation, _, wrapper = await observed(sequence([]))
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    advance = view.__anext__()
    assert vars(wrapper) == {"name": "openai_compat"}
    for value in (observation, wrapper, view, advance):
        assert "fictional" not in repr(value)
        for operation in (copy.copy, copy.deepcopy, pickle.dumps):
            with pytest.raises(StreamObservationError):
                operation(value)
    for value in (observation, view, advance):
        with pytest.raises(TypeError):
            vars(value)
    advance.close()
    await view.aclose()
    with pytest.raises(StreamObservationError):
        ProviderStreamObservation()


@pytest.mark.parametrize(
    "change",
    [
        {"iterator_obtained": 1},
        {"served_model": "fictional"},
        {"body_retained": True},
        {"raw_items_observed": True},
        {"data_advance_attempts": 1},
        {"raw_items_observed": 1},
        {"retired_at": "2026-11-07T00:00:00.000000Z"},
        {"gaps": ()},
        {"end_observation": "exhausted"},
        {"first_item_at": "2026-11-07T00:00:00.000000Z"},
    ],
)
async def test_reported_impossible_metadata_is_not_an_issued_fact(
    change: dict[str, object],
) -> None:
    observation, _, wrapper = await observed(sequence([]))
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    with pytest.raises(ValidationError) as caught:
        ProviderStreamMetadata.model_validate({**first(observation).model_dump(), **change})
    assert "fictional" not in str(caught.value)
    await view.aclose()


async def test_binding_requires_matching_normalized_schema_and_owning_count_table() -> None:
    observation, _, wrapper = await observed(sequence([None]))
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    await anext(view)
    fields = first(observation).input_binding.model_dump()
    with pytest.raises(ValidationError):
        StreamInputBinding.model_validate({**fields, "normalized_schema": "aresponses"})
    projection = dict(cast("dict[str, object]", fields["projection"]))
    for count in (0, 11, 12):
        with pytest.raises(ValidationError):
            StreamInputBinding.model_validate(
                {**fields, "projection": {**projection, "selected_present_fields_count": count}}
            )
    await view.aclose()


async def test_retained_metadata_does_not_root_native_argument_view_cycle() -> None:
    class Holder:
        view: object = None

    holder = Holder()
    finalized = asyncio.Event()

    async def native(argument: Holder) -> AsyncGenerator[object, object]:
        try:
            yield argument
        finally:
            finalized.set()

    original = native(holder)
    observation, fake, wrapper = await observed(original)
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    holder.view = view
    assert await anext(view) is holder
    view_ref, native_ref = weakref.ref(view), weakref.ref(original)
    del holder, view, original, fake, wrapper
    gc.collect()
    await asyncio.wait_for(finalized.wait(), 1)
    assert view_ref() is None
    assert native_ref() is None
    assert first(observation).end_observation is None


async def test_factory_exception_identity_and_no_admitted_iterator_claim() -> None:
    observation, fake, wrapper = await observed(None)
    error = ValueError("fictional factory failure")
    calls = 0

    def failure(**_kwargs: object) -> Any:
        nonlocal calls
        calls += 1
        raise error

    fake.astream = failure
    with pytest.raises(ValueError, match="fictional") as caught:
        wrapper.astream(provider_model_id="fixture", messages=[])
    assert caught.value is error
    assert calls == 1
    snapshot = observation.snapshot()
    assert snapshot.admitted_streams == 0
    assert snapshot.streams == ()
    assert snapshot.loss.factory_failure_count == 1


async def test_projection_failure_loses_metadata_without_vetoing_the_native_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from strata_forge.tracing import prompt_streams as runtime

    marker = object()
    observation, _, wrapper = await observed(sequence([marker]))
    view = wrapper.astream(provider_model_id="fixture", messages=[])

    def failure(*_args: object) -> Any:
        raise ValueError("fictional observer failure")

    monkeypatch.setattr(runtime, "project_chat", failure)
    assert await anext(view) is marker
    snapshot = observation.snapshot()
    assert snapshot.admitted_streams == 1
    assert snapshot.streams == ()
    assert snapshot.loss.metadata_lost_count == 1
    await view.aclose()


@pytest.mark.parametrize("retire", [False, True])
async def test_extra_input_references_release_after_projection_or_retirement(retire: bool) -> None:
    class Extra:
        pass

    extra = Extra()
    reference = weakref.ref(extra)
    observation, _, wrapper = await observed(sequence([None]))
    view = wrapper.astream(provider_model_id="fixture", messages=[], unselected=extra)
    del extra
    gc.collect()
    assert reference() is not None
    if retire:
        observation.close()
    else:
        assert await anext(view) is None
    gc.collect()
    assert reference() is None
    await view.aclose()


async def test_retained_metadata_does_not_root_error_advance_backrefs() -> None:
    error = ValueError("fictional native error")

    async def native(failure: ValueError) -> AsyncGenerator[object, object]:
        raise failure
        yield None

    original = native(error)
    observation, fake, wrapper = await observed(original)
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    advance = view.__anext__()
    object.__setattr__(error, "advance", advance)
    with pytest.raises(ValueError, match="fictional") as caught:
        await advance
    assert caught.value is error
    view_ref, advance_ref = weakref.ref(view), weakref.ref(advance)
    del error, caught, advance, view, original, fake, wrapper
    gc.collect()
    assert view_ref() is None
    assert advance_ref() is None
    assert first(observation).operation_errors == 1
    assert first(observation).end_observation is None


async def test_budget_is_reserved_before_forwarding_and_entry_numbers_are_not_reused() -> None:
    observation, fake, wrapper = await observed(None)
    views: list[Any] = []
    for _ in range(65):
        fake.native = sequence([])
        views.append(wrapper.astream(provider_model_id="fixture", messages=[]))
    snapshot = observation.snapshot()
    assert 0 < snapshot.admitted_streams <= 64
    assert snapshot.admitted_streams + snapshot.loss.over_capacity_count == 65
    assert len(snapshot.streams) == snapshot.admitted_streams
    for view in views:
        await view.aclose()
    before = observation.snapshot()
    fake.native = sequence([])
    raw = wrapper.astream(provider_model_id="fixture", messages=[])
    assert raw is fake.native
    assert observation.snapshot().admitted_streams == before.admitted_streams
    await raw.aclose()


@pytest.mark.parametrize("change", [{"raw_items_observed": 2}, {"operation_cancellations": 1}])
async def test_reported_counters_preserve_native_advance_and_error_relationships(
    change: dict[str, object],
) -> None:
    observation, _, wrapper = await observed(sequence([None]))
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    await anext(view)
    with pytest.raises(ValidationError):
        ProviderStreamMetadata.model_validate({**first(observation).model_dump(), **change})
    await view.aclose()


@pytest.mark.parametrize("failed_step", ["retirement_index", "record_publication"])
async def test_admission_allocation_failure_keeps_reserved_loss_accounting_coherent(
    failed_step: str,
) -> None:
    class FailedRetirementIndex(list[object]):
        def append(self, _value: object) -> None:
            raise MemoryError("fictional weak-index allocation failure")

    class FailedRecordPublication(dict[int, ProviderStreamMetadata]):
        def __setitem__(self, key: int, value: ProviderStreamMetadata) -> None:
            super().__setitem__(key, value)
            raise MemoryError("fictional post-publication allocation failure")

    native = sequence([None])
    observation, fake, wrapper = await observed(native)
    # Only this synthetic internal allocation fault replaces fixed private bookkeeping.
    state = object.__getattribute__(observation, "_state")
    if failed_step == "retirement_index":
        state.views = FailedRetirementIndex()
    else:
        state.records = FailedRecordPublication()
    returned = wrapper.astream(provider_model_id="fixture", messages=[])
    try:
        assert returned is native
        assert fake.factory_calls == 1
        snapshot = observation.snapshot()
        assert snapshot.admitted_streams == 1
        assert snapshot.streams == ()
        assert snapshot.loss.metadata_lost_count == 1
    finally:
        await returned.aclose()


@pytest.mark.parametrize("index", range(4))
def test_shared_recorded_metadata_fixture_preserves_canonical_bytes(index: int) -> None:
    import json
    from pathlib import Path

    from strata_forge.tracing.prompt_stream_types import ProviderStreamObservationMetadata

    fixture = json.loads(
        (Path(__file__).parents[3] / "tests/fixtures/prompts/provider-stream-v1.json").read_text()
    )
    vector = fixture["vectors"][index]
    metadata = ProviderStreamObservationMetadata.model_validate_json(json.dumps(vector["metadata"]))
    assert metadata.canonical().hex() == vector["canonical_utf8_hex"]
    assert metadata.metadata_sha256 == vector["metadata_sha256"]
    assert metadata.source_kind == "caller_bytes"
    assert all(
        stream.output_projection is None and stream.served_model is None
        for stream in metadata.streams
    )


async def test_fixed_forwarding_keeps_auth_route_and_nonstreaming_objects_unobserved() -> None:
    original = StreamFake(sequence([]))
    auth = {"fictional": object()}
    described: list[tuple[str, object]] = []
    model = object()
    original.auth_kwargs = lambda: auth
    original.responses_auth_kwargs = lambda: auth
    original.litellm_model = lambda value: f"fixture/{value}"

    def describe(identifier: str, actual: Model) -> None:
        described.append((identifier, actual))

    original.describe_to_litellm = describe
    observation = await prepare_stream_observation(await issued_invocation())
    wrapper = wrap_streaming_provider(
        original, observation=observation, provider_slot="openai_compat"
    )
    assert wrapper.auth_kwargs() is auth
    assert wrapper.responses_auth_kwargs() is auth
    assert wrapper.litellm_model("fictional") == "fixture/fictional"
    wrapper.describe_to_litellm("fictional", cast("Model", model))
    assert described == [("fictional", model)]
    assert await wrapper.acompletion(provider_model_id="fixture", messages=[]) is original.returned
    assert await wrapper.aresponses(provider_model_id="fixture", request={}) is original.returned
    assert original.owner is asyncio.current_task()
    assert observation.snapshot().admitted_streams == 0
    await cast("AsyncGenerator[object, object]", original.native).aclose()


async def test_unissued_working_objects_and_source_claims_refuse_with_fixed_management_errors() -> (
    None
):
    invocation = await issued_invocation()
    for source in (object(), invocation.snapshot()):
        with pytest.raises(StreamObservationError, match="stream_observation_invalid"):
            await prepare_stream_observation(cast("PreparedInvocation", source))
    observation, _, wrapper = await observed(sequence([]))
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    advance = view.__anext__()
    for value in (observation, wrapper, view, advance):
        with pytest.raises(StreamObservationError, match="stream_observation_invalid"):
            type(value)()
    with pytest.raises(StreamObservationError, match="stream_observation_invalid"):
        object.__new__(type(wrapper)).astream(provider_model_id="fixture", messages=[])
    with pytest.raises(StreamObservationError, match="stream_observation_invalid"):
        wrap_streaming_provider(
            cast("ProviderClient", object()), observation=observation, provider_slot="openai_compat"
        )
    advance.close()
    await view.aclose()


async def test_foreign_thread_factory_forwards_original_and_explicit_snapshot_refuses() -> None:
    native = sequence([])
    observation, _, wrapper = await observed(native)
    returned = await asyncio.to_thread(wrapper.astream, provider_model_id="fixture", messages=[])
    assert returned is native
    snapshot = observation.snapshot()
    assert snapshot.loss.scope_mismatch_seen is True
    assert snapshot.admitted_streams == 0
    with pytest.raises(StreamObservationError, match="stream_observation_scope_invalid"):
        await asyncio.to_thread(observation.snapshot)
    await native.aclose()


async def test_foreign_thread_native_drive_is_forwarded_without_argument_projection() -> None:
    marker = object()
    observation, _, wrapper = await observed(sequence([marker]))
    view = wrapper.astream(provider_model_id="fixture", messages=[])

    def drive() -> object:
        try:
            next(view.__anext__())
        except StopIteration as done:
            return done.value
        raise AssertionError("native immediate item missing")

    assert await asyncio.to_thread(drive) is marker
    snapshot = observation.snapshot()
    assert snapshot.loss.scope_mismatch_seen is True
    assert snapshot.streams[0].input_binding.stage == "unobserved"
    assert snapshot.streams[0].raw_items_observed == 0
    await view.aclose()


def test_factory_without_running_loop_forwards_native_without_loop_retention() -> None:
    loop = asyncio.new_event_loop()
    native = sequence([])
    try:
        observation, _, wrapper = loop.run_until_complete(observed(native))
        assert wrapper.astream(provider_model_id="fixture", messages=[]) is native
        assert observation.snapshot().loss.scope_mismatch_seen is True
        loop.run_until_complete(native.aclose())
    finally:
        loop.close()


async def test_management_snapshot_failure_is_fixed_without_affecting_native_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from strata_forge.tracing import prompt_streams as runtime

    marker = object()
    observation, _, wrapper = await observed(sequence([marker]))
    view = wrapper.astream(provider_model_id="fixture", messages=[])

    def failure(*_args: object, **_kwargs: object) -> Any:
        raise ValueError("fictional unreported metadata cause")

    with monkeypatch.context() as local:
        local.setattr(runtime.ProviderStreamObservationMetadata, "model_validate", failure)
        with pytest.raises(StreamObservationError) as caught:
            observation.snapshot()
    assert caught.value.__context__ is None
    assert "unreported" not in str(caught.value)
    assert await anext(view) is marker
    await view.aclose()


async def test_retirement_allocation_failure_preserves_native_work_and_closed_loss_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from strata_forge.tracing import prompt_streams as runtime

    marker = object()
    observation, _, wrapper = await observed(sequence([marker]))
    view = wrapper.astream(provider_model_id="fixture", messages=[])

    def failure(*_args: object, **_kwargs: object) -> Any:
        raise MemoryError("fictional retirement allocation")

    with monkeypatch.context() as local:
        local.setattr(runtime.ProviderStreamMetadata, "model_validate", failure)
        observation.close()
    observation.close()
    assert await anext(view) is marker
    snapshot = observation.snapshot()
    assert snapshot.closed is True
    assert snapshot.streams == ()
    assert snapshot.loss.metadata_lost_count == 1
    await view.aclose()


async def test_terminal_metadata_failure_cannot_replace_original_raw_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from strata_forge.tracing import prompt_streams as runtime

    marker = object()
    observation, _, wrapper = await observed(sequence([marker]))
    view = wrapper.astream(provider_model_id="fixture", messages=[])

    def failure(*_args: object) -> None:
        raise MemoryError("fictional terminal allocation")

    monkeypatch.setattr(runtime, "_note_result", failure)
    assert await anext(view) is marker
    snapshot = observation.snapshot()
    assert snapshot.streams == ()
    assert snapshot.loss.metadata_lost_count == 1
    await view.aclose()


async def test_saturating_counter_flag_does_not_change_original_item_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from strata_forge.tracing import prompt_streams as runtime

    # This bounded test-only ceiling exercises saturation without fabricating trillions
    # of calls or adding a runtime limit/configuration selector.
    monkeypatch.setattr(runtime, "MAX_STREAM_SAFE_INTEGER", 1)
    marker = object()
    observation, _, wrapper = await observed(sequence([marker, marker]))
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    assert await anext(view) is marker
    assert await anext(view) is marker
    record = first(observation)
    assert record.raw_items_observed == record.data_advance_attempts == 1
    assert record.counter_saturated is True
    await view.aclose()
