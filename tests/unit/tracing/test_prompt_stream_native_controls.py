"""Independent native-vs-observed controls; fictional values and caller-owned work only."""

from __future__ import annotations

import asyncio
import gc
import warnings
import weakref
from typing import TYPE_CHECKING, Any  # Raw native/custom advance protocol fixture only.

import pytest

from tests.unit.tracing.test_prompt_streams import observed

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine, Generator

    from strata_forge.tracing.prompt_streams import ProviderStreamObservation


def fixture_view(
    native: AsyncGenerator[object, object],
    wrapped: bool,
) -> Coroutine[object, object, tuple[ProviderStreamObservation | None, Any]]:
    async def create() -> tuple[ProviderStreamObservation | None, Any]:
        if not wrapped:
            return None, native
        observation, _, wrapper = await observed(native)
        return observation, wrapper.astream(provider_model_id="fixture", messages=[])

    return create()


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_deprecated_throw_form_preserves_native_warning_and_injected_identity(
    wrapped: bool,
) -> None:
    item = object()
    injected = ValueError("fictional native throw")

    async def native() -> AsyncGenerator[object, object]:
        yield item

    observation, view = await fixture_view(native(), wrapped)
    try:
        assert await anext(view) is item
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            with pytest.raises(ValueError) as caught:  # noqa: PT011 - preserve exact independent error-identity witness
                await view.athrow(ValueError, injected, None)
        assert caught.value is injected
        assert len(recorded) == 1 and recorded[0].category is DeprecationWarning  # noqa: PT018 - preserve exact independent assertion
        if observation is not None:
            record = observation.snapshot().streams[0]
            assert record.operation_errors == 1
            assert record.end_observation is None
    finally:
        await view.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_manual_native_throw_swallowed_by_generator_yields_original_item(
    wrapped: bool,
) -> None:
    suspension, item = object(), object()
    control = asyncio.CancelledError("fictional manual control")
    received: list[BaseException] = []

    class AwaitMarker:
        def __await__(self) -> Generator[object, object]:
            yield suspension

    async def native() -> AsyncGenerator[object, object]:
        try:
            await AwaitMarker()
        except asyncio.CancelledError as error:
            received.append(error)
        yield item

    observation, view = await fixture_view(native(), wrapped)
    try:
        advance = view.__anext__()
        assert next(advance) is suspension
        with pytest.raises(StopIteration) as done:
            advance.throw(control)
        assert done.value.value is item
        assert received == [control] and received[0] is control  # noqa: PT018 - preserve exact independent assertion
        if observation is not None:
            record = observation.snapshot().streams[0]
            assert record.raw_items_observed == record.data_advance_attempts == 1
            assert record.operation_cancellations == record.operation_errors == 0
    finally:
        await view.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_held_aclose_keeps_native_swallowed_repeated_cancellation(wrapped: bool) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    controls: list[BaseException] = []
    item = object()

    async def native() -> AsyncGenerator[object, object]:
        try:
            yield item
        finally:
            entered.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError as error:
                    controls.append(error)

    observation, view = await fixture_view(native(), wrapped)
    assert await anext(view) is item
    existing = asyncio.all_tasks()
    close = asyncio.create_task(view.aclose())
    try:
        await entered.wait()
        for label in ("fictional one", "fictional two"):
            close.cancel(label)
            await asyncio.sleep(0)
        assert len(controls) == 2
        assert not close.done()
        assert asyncio.all_tasks() == existing | {close}
        if observation is not None:
            record = observation.snapshot().streams[0]
            assert record.close_attempts == 1
            assert record.end_observation is None
            assert record.closed_observed_at is None
        release.set()
        assert await close is None
        assert not close.cancelled()
        if observation is not None:
            record = observation.snapshot().streams[0]
            assert record.end_observation == "aclose_completed"
            assert record.raw_items_observed == 1
            assert record.operation_errors == record.operation_cancellations == 0
    finally:
        release.set()
        await close


@pytest.mark.asyncio
async def test_manual_first_item_latency_includes_drive_pause_without_construction_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from strata_forge.tracing import prompt_streams as runtime

    suspension, item = object(), object()

    class AwaitMarker:
        def __await__(self) -> Generator[object, object]:
            yield suspension

    async def native() -> AsyncGenerator[object, object]:
        await AwaitMarker()
        yield item

    observation, _, wrapper = await observed(native())
    clock = [1_000_000]
    calls: list[bool] = []

    def nano() -> int:
        calls.append(True)
        return clock[0]

    monkeypatch.setattr(runtime.time, "monotonic_ns", nano)
    view = wrapper.astream(provider_model_id="fixture", messages=[])
    advance = view.__anext__()
    assert calls == []
    try:
        assert next(advance) is suspension
        assert calls == [True]
        clock[0] = 4_000_000
        with pytest.raises(StopIteration) as done:
            next(advance)
        assert done.value.value is item
        assert observation.snapshot().streams[0].first_item_drive_latency_us == 3000
    finally:
        await view.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_closing_an_advance_does_not_claim_observed_generator_close(wrapped: bool) -> None:
    entered: list[bool] = []
    item = object()

    async def native() -> AsyncGenerator[object, object]:
        entered.append(True)
        yield item

    observation, view = await fixture_view(native(), wrapped)
    advance = view.__anext__()
    try:
        assert advance.close() is None
        assert entered == []
        if observation is not None:
            record = observation.snapshot().streams[0]
            assert record.first_drive_attempt_at is None
            assert record.close_attempts == 0
            assert record.end_observation is None
        with pytest.raises(RuntimeError):
            await advance
        with pytest.raises(StopAsyncIteration):
            await anext(view)
        assert entered == []
        if observation is not None:
            record = observation.snapshot().streams[0]
            assert record.end_observation == "exhausted"
            assert record.raw_items_observed == record.close_attempts == 0
    finally:
        await view.aclose()


def test_retained_observation_does_not_root_completed_callback_task_loop_cycle() -> None:
    loop = asyncio.new_event_loop()
    refs: list[weakref.ReferenceType[object]] = []

    async def make() -> ProviderStreamObservation:
        async def native() -> AsyncGenerator[object, object]:
            yield None

        observation, fake, wrapper = await observed(native())
        view = wrapper.astream(provider_model_id="fixture", messages=[])
        await anext(view)
        await view.aclose()
        completed = asyncio.create_task(asyncio.sleep(0, result=wrapper))
        assert await completed is wrapper
        fake.callback_task = completed  # pyright: ignore[reportAttributeAccessIssue] -- deliberate ordinary application backref
        fake.callback_loop = loop  # pyright: ignore[reportAttributeAccessIssue] -- deliberate ordinary application backref  # noqa: F821 - fixture closure executes before the outer loop reference is deleted
        refs.extend([weakref.ref(fake), weakref.ref(wrapper), weakref.ref(completed)])
        return observation

    observation = loop.run_until_complete(make())
    loop_ref = weakref.ref(loop)
    loop.close()
    del loop
    gc.collect()
    assert loop_ref() is None
    assert all(reference() is None for reference in refs)
    assert observation.snapshot().streams[0].end_observation == "aclose_completed"
