"""Retained body-free metadata must not root abandoned application working cycles."""

from __future__ import annotations

import asyncio
import copy
import gc
import pickle
import weakref
from typing import TYPE_CHECKING

import pytest

from strata_forge.llm.client import LLMClient
from strata_forge.tracing.prompt_provider import (
    ProviderObservationError,
    prepare_provider_observation,
    wrap_provider,
)
from tests.unit.tracing.test_prompt_provider import FakeProvider, issued_invocation, observed

if TYPE_CHECKING:
    from strata_forge.tracing.prompt_provider import ProviderObservation


async def test_metadata_owner_does_not_root_delegate_wrapper_cycle() -> None:
    observation, delegate, wrapper = await observed()
    object.__setattr__(delegate, "owner_wrapper", wrapper)
    delegate_ref, wrapper_ref = weakref.ref(delegate), weakref.ref(wrapper)
    del delegate, wrapper
    gc.collect()
    assert delegate_ref() is None
    assert wrapper_ref() is None
    assert observation.snapshot().admitted_attempts == 0


async def test_metadata_owner_does_not_root_delegate_client_wrapper_cycle() -> None:
    observation, delegate, wrapper = await observed()
    client = LLMClient("openai/gpt-4.1-mini", provider_clients={"openai_compat": wrapper})
    object.__setattr__(delegate, "owner_client", client)
    refs = weakref.ref(delegate), weakref.ref(wrapper), weakref.ref(client)
    del delegate, wrapper, client
    gc.collect()
    assert all(reference() is None for reference in refs)
    assert observation.snapshot().admitted_attempts == 0


async def test_acyclic_wrapper_collection_preserves_metadata_and_safe_inspection() -> None:
    observation, delegate, wrapper = await observed()
    assert vars(wrapper) == {"name": "openai_compat"}
    for value in (observation, wrapper):
        assert "CANARY" not in repr(value)
        for operation in (copy.copy, copy.deepcopy, pickle.dumps):
            with pytest.raises(ProviderObservationError, match="observation_invalid"):
                operation(value)
    del value
    with pytest.raises(TypeError):
        vars(observation)
    delegate_ref, wrapper_ref = weakref.ref(delegate), weakref.ref(wrapper)
    del delegate, wrapper
    gc.collect()
    assert delegate_ref() is None
    assert wrapper_ref() is None
    assert observation.snapshot().attempts == ()


@pytest.mark.parametrize("keep_metadata", [False, True])
def test_callback_completed_task_and_loop_cycles_collect_without_implicit_close(
    keep_metadata: bool,
) -> None:
    loop = asyncio.new_event_loop()

    async def make() -> ProviderObservation:
        return await prepare_provider_observation(await issued_invocation())

    task = loop.create_task(make())
    observation = loop.run_until_complete(task)
    # A normal loop callback can retain the task and its observation result. Keeping the
    # observation must not turn that application cycle into globally rooted working state.
    loop.call_soon(lambda held=task: held.result())
    loop_ref, task_ref, observation_ref = (
        weakref.ref(loop),
        weakref.ref(task),
        weakref.ref(observation),
    )
    held_metadata = observation if keep_metadata else None
    del task, observation, loop
    gc.collect()
    try:
        assert loop_ref() is None
        assert task_ref() is None
        if held_metadata is not None:
            assert held_metadata.snapshot().closed is False
            assert held_metadata.snapshot().admitted_attempts == 0
        else:
            assert observation_ref() is None
    finally:
        # Clean the historical failing implementation's retained fictional loop, without
        # hiding the failed collection assertion or adding any observer-driven cleanup.
        retained = loop_ref()
        if retained is not None:
            retained.close()


def test_unwrapped_callback_task_loop_cycle_is_collectible() -> None:
    loop = asyncio.new_event_loop()

    async def make() -> object:
        return object()

    task = loop.create_task(make())
    result = loop.run_until_complete(task)
    loop.call_soon(lambda held=task: held.result())
    loop_ref, task_ref = weakref.ref(loop), weakref.ref(task)
    del loop, task
    gc.collect()
    assert loop_ref() is None
    assert task_ref() is None
    assert result is not None


@pytest.mark.parametrize("wrapped", [False, True])
def test_abandoned_native_task_cycle_is_not_rooted_by_retained_metadata(wrapped: bool) -> None:
    loop = asyncio.new_event_loop()
    diagnostics: list[object] = []
    loop.set_exception_handler(lambda _loop, context: diagnostics.append(context.get("message")))

    async def start() -> tuple[
        ProviderObservation | None,
        weakref.ReferenceType[asyncio.Task[object]],
        weakref.ReferenceType[object],
        weakref.ReferenceType[object],
    ]:
        observation = (
            await prepare_provider_observation(await issued_invocation()) if wrapped else None
        )
        delegate = FakeProvider()
        wrapper = (
            wrap_provider(delegate, observation=observation, provider_slot="openai_compat")
            if observation is not None
            else delegate
        )
        future: asyncio.Future[object] = loop.create_future()

        async def hold(**_kwargs: object) -> object:
            # Ordinary application callbacks/frames may form a cycle through the wrapper.
            object.__setattr__(delegate, "owner_wrapper", wrapper)
            return await future

        delegate.acompletion = hold
        task = loop.create_task(
            wrapper.acompletion(provider_model_id="fixture:model", messages=[])
            if wrapped
            else hold()
        )
        await asyncio.sleep(0)
        return observation, weakref.ref(task), weakref.ref(delegate), weakref.ref(wrapper)

    observation, task_ref, delegate_ref, wrapper_ref = loop.run_until_complete(start())
    gc.collect()
    try:
        assert task_ref() is None
        assert delegate_ref() is None
        assert wrapper_ref() is None
        assert diagnostics == ["Task was destroyed but it is pending!"]
        # Native GC can close coroutine frames; it is not evidence of a returned result.
        if observation is not None:
            assert observation.snapshot().attempts == ()
    finally:
        retained = task_ref()
        if retained is not None:
            retained.cancel()
            loop.run_until_complete(asyncio.gather(retained, return_exceptions=True))
        loop.close()
