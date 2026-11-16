"""Closed reported metadata is not an issued capture handle or provider assertion."""

from __future__ import annotations

import asyncio
import copy
import json
import pickle
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    NoReturn,
    cast,
)  # The unchanged ProviderClient interface intentionally carries raw SDK values.

import pytest
from pydantic import ValidationError

from strata_forge.llm.providers.base import ProviderClient
from strata_forge.prompts.consumption import LocalBuildSource, consume_build
from strata_forge.tracing import prompt_provider as runtime
from strata_forge.tracing.prompt_attempts import PreparedInvocation, prepare_invocation
from strata_forge.tracing.prompt_provider import (
    ProviderObservation,
    ProviderObservationError,
    prepare_provider_observation,
    wrap_provider,
)
from strata_forge.tracing.prompt_provider_types import (
    BASE_PROVIDER_GAPS,
    ProviderAttemptMetadata,
    ProviderObservationMetadata,
)
from strata_forge.tracing.prompt_provider_values import project_chat, project_response

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from strata_forge.llm.registry import Model, ProviderName


def reported_attempt(**changes: object) -> ProviderAttemptMetadata:
    fields: dict[str, object] = {
        "domain": "strata-provider-prompt-attempt-v1",
        "version": 1,
        "observation_id": "a" * 32,
        "attempt_id": "b" * 32,
        "entry_index": 0,
        "invocation_id": "c" * 32,
        "build_id": "d" * 64,
        "render_id": "e" * 64,
        "input_fingerprint": "f" * 64,
        "source_kind": "caller_bytes",
        "method": "acompletion",
        "declared_provider_slot": "openai_compat",
        "observed_provider_model_id": "fixture:model",
        "logical_model": None,
        "served_model": None,
        "model_revision": None,
        "source_relation": "unverified_argument",
        "started_at": "2026-11-07T00:00:00.000000Z",
        "ended_at": "2026-11-07T00:00:00.000001Z",
        "duration_us": 1,
        "call_outcome": "returned",
        "request_entered": True,
        "return_observed": True,
        "error_category": None,
        "request_projection": project_chat("fixture:model", [], {}),
        "response_projection": project_response("acompletion", {"choices": []}),
        "capture_status": "recorded",
        "capture_issues": (),
        "parents": None,
        "relation": "unknown",
        "render_relation": "unknown",
        "usage": None,
        "cost": None,
        "retention": "metadata_only",
        "export_status": "not_configured",
        "gaps": BASE_PROVIDER_GAPS,
    }
    return ProviderAttemptMetadata.model_validate({**fields, **changes})


def reported_observation(**changes: object) -> ProviderObservationMetadata:
    attempt = reported_attempt()
    fields: dict[str, object] = {
        "domain": "strata-provider-prompt-observation-v1",
        "version": 1,
        "observation_id": attempt.observation_id,
        "invocation_id": attempt.invocation_id,
        "source_kind": "caller_bytes",
        "source_partition": "fixture:metadata",
        "source_reference": None,
        "build_id": attempt.build_id,
        "render_id": attempt.render_id,
        "input_fingerprint": attempt.input_fingerprint,
        "created_at": "2026-11-07T00:00:00.000000Z",
        "closed": False,
        "admitted_attempts": 1,
        "attempts": (attempt,),
        "loss": {
            "over_capacity_count": 0,
            "projection_limited_count": 0,
            "metadata_lost_count": 0,
            "scope_mismatch_seen": False,
            "counter_saturated": False,
        },
        "coverage": {
            "all_calls_observed": False,
            "streaming_observed": False,
            "cache_events_observed": False,
            "post_close_calls_counted": False,
        },
        "retention": "metadata_only",
        "export_status": "not_configured",
    }
    return ProviderObservationMetadata.model_validate({**fields, **changes})


def test_reported_metadata_hashes_are_domain_separated_and_have_no_provider_claim() -> None:
    attempt = reported_attempt()
    observation = reported_observation()
    assert len(attempt.metadata_sha256) == len(observation.metadata_sha256) == 64
    assert attempt.metadata_sha256 != observation.metadata_sha256
    assert attempt.logical_model is attempt.served_model is attempt.model_revision is None
    assert attempt.parents is None
    assert observation.coverage.all_calls_observed is False
    assert observation.source_kind == "caller_bytes"


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"domain": "wrong"},
        {"request_entered": False},
        {"return_observed": 1},
        {"entry_index": 64},
        {"duration_us": 0.7},
        {"declared_provider_slot": "verified_vendor"},
        {"logical_model": "secret"},
        {"served_model": "secret"},
        {"model_revision": "a" * 40},
        {"parents": {"retry_of": "a" * 32}},
        {"relation": "retry"},
        {"render_relation": "exact"},
        {"usage": 0},
        {"cost": 0},
        {"error_category": "CANARY"},
        {"extra": "CANARY"},
        {"capture_issues": ("response_opaque",)},
        {"capture_status": "partial"},
        {"gaps": ()},
        {"gaps": (*BASE_PROVIDER_GAPS, "response_projection_unavailable")},
        {"call_outcome": "raised"},
        {"return_observed": False},
        {"started_at": "2026-02-30T00:00:00.000000Z"},
        {"attempt_id": "a" * 32 + "\n"},
    ],
)
def test_forged_reported_facts_refuse_with_content_free_errors(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError) as caught:
        reported_attempt(**changes)
    assert "CANARY" not in str(caught.value)


@pytest.mark.parametrize("ending", ["\n", "\r", "\u2028", "\u2029"])
def test_observed_label_has_complete_ascii_grammar(ending: str) -> None:
    with pytest.raises(ValidationError):
        reported_attempt(observed_provider_model_id="fixture" + ending)


def test_unknown_response_projection_is_partial_not_a_failed_provider_call() -> None:
    attempt = reported_attempt(
        response_projection=project_response("acompletion", object()),
        capture_status="partial",
        capture_issues=("response_opaque",),
        gaps=(*BASE_PROVIDER_GAPS, "response_projection_unavailable"),
    )
    assert attempt.call_outcome == "returned"
    assert attempt.return_observed
    assert attempt.response_projection.canonical_tagged_sha256 is None
    assert attempt.error_category is None


def test_honest_metadata_loss_keeps_admission_holes_and_rejects_reuse_or_false_counts() -> None:
    observation = reported_observation()
    fields = observation.model_dump()
    loss = cast("dict[str, object]", fields["loss"])
    honest = reported_observation(attempts=(), loss={**loss, "metadata_lost_count": 1})
    assert honest.admitted_attempts == 1
    with pytest.raises(ValidationError):
        reported_observation(attempts=())
    with pytest.raises(ValidationError):
        reported_observation(attempts=(reported_attempt(), reported_attempt()), admitted_attempts=2)
    with pytest.raises(ValidationError):
        reported_observation(
            coverage={
                "all_calls_observed": True,
                "streaming_observed": False,
                "cache_events_observed": False,
                "post_close_calls_counted": False,
            }
        )


@pytest.mark.parametrize(
    ("side", "count"),
    [
        ("request_projection", 0),
        ("request_projection", 11),
        ("request_projection", 12),
        ("response_projection", 9),
        ("response_projection", 12),
    ],
)
def test_reported_projection_counts_match_owning_method_and_side(side: str, count: int) -> None:
    base = reported_attempt().model_dump()
    projection = cast("dict[str, object]", base[side])
    with pytest.raises(ValidationError):
        reported_attempt(**{side: {**projection, "selected_present_fields_count": count}})


async def issued_invocation() -> PreparedInvocation:
    path = Path(__file__).parents[3] / "tests/fixtures/prompts/local-attempts-v1.json"
    fixture = cast("dict[str, object]", json.loads(path.read_text()))
    raw = bytes.fromhex(cast("str", fixture["build_canonical_utf8_hex"]))
    consumed = consume_build(
        raw,
        expected_build_id=cast("str", fixture["build_id"]),
        expected_size_bytes=len(raw),
        source=LocalBuildSource(kind="caller_bytes", partition="fixture:provider", reference=None),
    )
    return await prepare_invocation(consumed, fixture["values"])


class FakeProvider(ProviderClient):
    name: ClassVar[ProviderName] = "openai_compat"
    litellm_prefix = "fixture/"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.returned: object = {"choices": []}
        self.failure: BaseException | None = None
        self.owner: asyncio.Task[object] | None = None

    def __getattribute__(self, name: str) -> Any:
        if name == "config":
            raise AssertionError("CONFIG CANARY")
        return object.__getattribute__(self, name)

    def auth_kwargs(self) -> dict[str, Any]:
        raise AssertionError("AUTH CANARY")

    async def acompletion(
        self, *, provider_model_id: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        self.owner = asyncio.current_task()
        self.calls.append(
            (
                "acompletion",
                {"provider_model_id": provider_model_id, "messages": messages, **kwargs},
            )
        )
        if self.failure is not None:
            raise self.failure
        return self.returned

    async def aresponses(self, *, provider_model_id: str, request: dict[str, Any]) -> Any:
        self.owner = asyncio.current_task()
        self.calls.append(
            ("aresponses", {"provider_model_id": provider_model_id, "request": request})
        )
        if self.failure is not None:
            raise self.failure
        return self.returned


async def observed() -> tuple[ProviderObservation, FakeProvider, ProviderClient]:
    observation = await prepare_provider_observation(await issued_invocation())
    fake = FakeProvider()
    return (
        observation,
        fake,
        wrap_provider(fake, observation=observation, provider_slot="openai_compat"),
    )


async def test_fixed_facade_preserves_raw_return_argument_identity_and_current_task() -> None:
    observation, fake, wrapped = await observed()
    messages: list[dict[str, object]] = [{"role": "user", "content": "synthetic {{literal}}\r\n😀"}]
    marker = object()
    result = await wrapped.acompletion(
        provider_model_id="fixture:model", messages=messages, temperature=0.7, api_key=marker
    )
    assert result is fake.returned
    assert fake.owner is asyncio.current_task()
    assert fake.calls[0][1]["messages"] is messages
    assert fake.calls[0][1]["api_key"] is marker
    first = observation.snapshot().attempts[0]
    assert first.call_outcome == "returned"
    assert first.capture_status == "recorded"
    assert first.request_projection.omitted_fields_count == 1
    assert first.served_model is None
    assert first.logical_model is None
    request: dict[str, Any] = {"input": [], "store": False, "include": [], "headers": marker}
    assert (
        await wrapped.aresponses(provider_model_id="fixture:model", request=request)
        is fake.returned
    )
    assert fake.calls[-1][1]["request"] is request
    assert observation.snapshot().attempts[-1].method == "aresponses"
    for value in [observation, wrapped]:
        assert "CANARY" not in repr(value)
        with pytest.raises(ProviderObservationError):
            pickle.dumps(value)
        with pytest.raises(ProviderObservationError):
            copy.copy(value)
    assert vars(wrapped) == {"name": "openai_compat"}
    assert "CANARY" not in observation.snapshot().canonical().decode()


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("ORIGINAL CANARY"),
        TimeoutError("ORIGINAL CANARY"),
        asyncio.CancelledError("ORIGINAL CANARY"),
    ],
)
async def test_original_exception_and_control_identity_are_preserved(
    failure: BaseException,
) -> None:
    observation, fake, wrapped = await observed()
    fake.failure = failure
    with pytest.raises(type(failure)) as caught:
        await wrapped.acompletion(provider_model_id="fixture", messages=[])
    assert caught.value is failure
    assert caught.value.__cause__ is None
    record = observation.snapshot().attempts[0]
    assert record.call_outcome == (
        "cancelled" if isinstance(failure, asyncio.CancelledError) else "raised"
    )
    assert record.response_projection.status == "not_returned"
    assert "CANARY" not in record.canonical().decode()


@pytest.mark.parametrize("phase", ["projection", "construction", "finalization"])
async def test_observer_failure_never_replaces_or_repeats_the_model_call(
    phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation, fake, wrapped = await observed()

    def broken(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("OBSERVER CANARY")

    monkeypatch.setattr(
        runtime,
        {"projection": "project_chat", "construction": "_Ticket", "finalization": "_finish"}[phase],
        broken,
    )
    assert await wrapped.acompletion(provider_model_id="fixture", messages=[]) is fake.returned
    assert len(fake.calls) == 1
    snapshot = observation.snapshot()
    assert snapshot.admitted_attempts == snapshot.loss.metadata_lost_count == 1
    assert snapshot.attempts == ()
    observation.close()
    assert observation.snapshot().closed


async def test_unsupported_and_overbudget_values_still_delegate_once_without_rewriting() -> None:
    observation, fake, wrapped = await observed()
    custom = object()
    assert (
        await wrapped.acompletion(provider_model_id="fixture", messages=[], temperature=custom)
        is fake.returned
    )
    assert fake.calls[-1][1]["temperature"] is custom
    assert observation.snapshot().attempts[0].request_projection.status == "unsupported"
    assert (
        await wrapped.acompletion(
            provider_model_id="fixture", messages=[], temperature=float("nan")
        )
        is fake.returned
    )
    assert (
        await wrapped.acompletion(provider_model_id="fixture", messages=[], tools=[None] * 65_536)
        is fake.returned
    )
    snapshot = observation.snapshot()
    assert snapshot.attempts[-1].request_projection.status == "limit"
    assert snapshot.loss.projection_limited_count == 1
    assert len(fake.calls) == 3


async def test_parallel_calls_are_not_serialized_and_complete_in_entry_order() -> None:
    observation = await prepare_provider_observation(await issued_invocation())
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]

    class Held(FakeProvider):
        async def acompletion(
            self, *, provider_model_id: str, messages: list[dict[str, Any]], **kwargs: Any
        ) -> Any:
            index = int(provider_model_id)
            entered[index].set()
            await release[index].wait()
            return self.returned

    fake = Held()
    wrapped = wrap_provider(fake, observation=observation, provider_slot="openai_compat")
    tasks = [
        asyncio.create_task(wrapped.acompletion(provider_model_id=str(index), messages=[]))
        for index in range(2)
    ]
    await asyncio.gather(*(item.wait() for item in entered))
    with pytest.raises(ProviderObservationError, match="observation_in_progress"):
        observation.snapshot()
    with pytest.raises(ProviderObservationError, match="observation_in_progress"):
        observation.close()
    release[1].set()
    assert await tasks[1] is fake.returned
    release[0].set()
    assert await tasks[0] is fake.returned
    assert [item.entry_index for item in observation.snapshot().attempts] == [0, 1]


async def test_capacity_and_closed_scope_drop_observation_without_skipping_calls() -> None:
    observation, fake, wrapped = await observed()
    for _ in range(65):
        assert await wrapped.acompletion(provider_model_id="fixture", messages=[]) is fake.returned
    snapshot = observation.snapshot()
    assert snapshot.admitted_attempts == len(snapshot.attempts) == 64
    assert snapshot.loss.over_capacity_count == 1
    assert len(fake.calls) == 65
    observation.close()
    assert await wrapped.acompletion(provider_model_id="fixture", messages=[]) is fake.returned
    assert observation.snapshot().loss.over_capacity_count == 1
    assert len(fake.calls) == 66


async def test_forged_and_closed_sources_cannot_issue_observation_handles() -> None:
    with pytest.raises(ProviderObservationError):
        await prepare_provider_observation(object.__new__(PreparedInvocation))
    with pytest.raises(ProviderObservationError):
        await prepare_provider_observation(cast("PreparedInvocation", reported_observation()))
    invocation = await issued_invocation()
    invocation.close()
    with pytest.raises(ProviderObservationError, match="observation_closed"):
        await prepare_provider_observation(invocation)
    with pytest.raises(ProviderObservationError):
        wrap_provider(
            FakeProvider(),
            observation=object.__new__(ProviderObservation),
            provider_slot="openai_compat",
        )


async def test_actual_task_cancellation_is_the_delegates_original_control_without_new_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = await prepare_provider_observation(await issued_invocation())
    entered = asyncio.Event()

    class Held(FakeProvider):
        original: asyncio.CancelledError | None = None

        async def acompletion(
            self, *, provider_model_id: str, messages: list[dict[str, Any]], **kwargs: Any
        ) -> Any:
            self.owner = asyncio.current_task()
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as error:
                self.original = error
                raise

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("observer must not create a child task")

    fake = Held()
    wrapped = wrap_provider(fake, observation=observation, provider_slot="openai_compat")
    monkeypatch.setattr(runtime.asyncio, "create_task", forbidden)
    task = asyncio.get_running_loop().create_task(
        wrapped.acompletion(provider_model_id="fixture", messages=[])
    )
    await entered.wait()
    assert fake.owner is task
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value is fake.original
    record = observation.snapshot().attempts[0]
    assert record.call_outcome == "cancelled"
    assert not record.return_observed
    assert record.response_projection.status == "not_returned"


async def test_swallowed_cancellation_keeps_original_late_return_without_projecting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = await prepare_provider_observation(await issued_invocation())
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Held(FakeProvider):
        async def acompletion(
            self, *, provider_model_id: str, messages: list[dict[str, Any]], **kwargs: Any
        ) -> Any:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            return self.returned

    def forbidden(*_args: object) -> NoReturn:
        pytest.fail("retired response must not be projected")

    fake = Held()
    wrapped = wrap_provider(fake, observation=observation, provider_slot="openai_compat")
    task = asyncio.create_task(wrapped.acompletion(provider_model_id="fixture", messages=[]))
    await entered.wait()
    task.cancel()
    await cancelled.wait()
    monkeypatch.setattr(runtime, "project_response", forbidden)
    release.set()
    assert await task is fake.returned
    record = observation.snapshot().attempts[0]
    assert record.call_outcome == "returned"
    assert not record.return_observed
    assert record.response_projection.status == "retired"
    assert "observer_retired" in record.gaps


async def test_stream_auth_description_and_route_methods_forward_without_observation() -> None:
    observation = await prepare_provider_observation(await issued_invocation())

    async def stream() -> AsyncGenerator[object]:
        yield object()

    class Forward(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.auth: dict[str, Any] = {"api_key": object()}
            self.responses_auth: dict[str, Any] = {"api_key": object()}
            self.chat_stream, self.response_stream = stream(), stream()
            self.described: tuple[str, Model] | None = None

        def auth_kwargs(self) -> dict[str, Any]:
            return self.auth

        def responses_auth_kwargs(self) -> dict[str, Any]:
            return self.responses_auth

        def describe_to_litellm(self, provider_model_id: str, model: Model) -> None:
            self.described = provider_model_id, model

        def astream(
            self, *, provider_model_id: str, messages: list[dict[str, Any]], **kwargs: Any
        ) -> AsyncIterator[Any]:
            return self.chat_stream

        def aresponses_stream(
            self, *, provider_model_id: str, request: dict[str, Any]
        ) -> AsyncIterator[Any]:
            return self.response_stream

    fake = Forward()
    wrapped = wrap_provider(fake, observation=observation, provider_slot="openai_compat")
    model = cast("Model", object())
    assert wrapped.auth_kwargs() is fake.auth
    assert wrapped.responses_auth_kwargs() is fake.responses_auth
    assert wrapped.litellm_model("fixture") == "fixture/fixture"
    wrapped.describe_to_litellm("fixture", model)
    assert fake.described == ("fixture", model)
    assert wrapped.astream(provider_model_id="fixture", messages=[]) is fake.chat_stream
    assert (
        wrapped.aresponses_stream(provider_model_id="fixture", request={}) is fake.response_stream
    )
    await fake.chat_stream.aclose()
    await fake.response_stream.aclose()
    assert observation.snapshot().admitted_attempts == 0


async def test_foreign_loop_calls_forward_without_content_projection_and_mark_scope_loss() -> None:
    observation, fake, wrapped = await observed()

    async def foreign() -> object:
        return await wrapped.acompletion(provider_model_id="fixture", messages=[])

    assert await asyncio.to_thread(lambda: asyncio.run(foreign())) is fake.returned
    snapshot = observation.snapshot()
    assert snapshot.admitted_attempts == 0
    assert snapshot.loss.scope_mismatch_seen
    assert len(fake.calls) == 1
