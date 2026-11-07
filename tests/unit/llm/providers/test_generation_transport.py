"""Actual installed SDK with bounded fictional HTTP streams and unavailable native capability."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from pydantic import SecretStr

from strata_forge.llm.client import LLMClient
from strata_forge.llm.generation import GenerationError
from strata_forge.llm.messages import UserMessage
from strata_forge.llm.providers import openai_compat as provider
from strata_forge.llm.providers.config import OpenAICompatConfig
from tests.unit.llm.test_generation_protocol import MODEL, request, response

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


_Endpoint = provider._ObservedGenerationEndpoint  # pyright: ignore[reportPrivateUsage] — privately issued fixture observation only.


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], wait: asyncio.Event | None = None):
        self.chunks = chunks
        self.wait = wait
        self.entered = asyncio.Event()
        self.reads = self.closed = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.entered.set()
        if self.wait is not None:
            await self.wait.wait()
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed += 1


def client() -> LLMClient:
    # This legacy config is deliberately different; the new operation must ignore it entirely.
    legacy = provider.OpenAICompatProvider(
        OpenAICompatConfig(
            base_url="http://unused.invalid/v1", api_key=SecretStr("FICTIONAL_UNUSED_KEY")
        )
    )
    return LLMClient(
        model=MODEL, provider="openai_compat", provider_clients={"openai_compat": legacy}
    )


def install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
    stream: Stream | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {"requests": []}
    if stream is None:
        stream = Stream([json.dumps(response()).encode()])
    state["stream"] = stream

    async def handler(raw: httpx.Request) -> httpx.Response:
        state["requests"].append((str(raw.url), dict(raw.headers), json.loads(await raw.aread())))
        return httpx.Response(
            status, headers={"content-type": "application/json", **(headers or {})}, stream=stream
        )

    def factory(**kwargs: object) -> httpx.MockTransport:
        return httpx.MockTransport(handler)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", factory)

    async def observe(_provider: object, current: Any) -> Any:
        return _Endpoint(  # pyright: ignore[reportPrivateUsage] — fixed synthetic observation only.
            "http://127.0.0.1:54321/v1",
            MODEL,
            current.model_ref,
            current.profile,
            "2.36.0",
            "0.24.0",
        )  # pyright: ignore[reportPrivateUsage] — fixed synthetic observation only.

    monkeypatch.setattr(provider, "_observe_generation_endpoint", observe)
    return state


async def test_default_observer_denies_without_constructing_any_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(**kwargs: object) -> None:
        raise AssertionError("no transport construction")

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", refuse)
    with pytest.raises(GenerationError, match="generation_profile_unavailable"):
        await client().generate_one([UserMessage(content="Synthetic")], request=request())


async def test_actual_sdk_exact_body_and_explicit_placeholder_ignores_ambient_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = install(monkeypatch)
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_ADMIN_KEY",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT_ID",
        "OPENAI_WEBHOOK_SECRET",
    ):
        monkeypatch.setenv(name, "FICTIONAL_AMBIENT_VALUE")
    result = await client().generate_one([UserMessage(content="Synthetic")], request=request())
    assert result.final_text == "<think>literal text</think>{unchanged}"
    assert len(state["requests"]) == 1
    url, headers, body = state["requests"][0]
    assert url == "http://127.0.0.1:54321/v1/chat/completions"
    assert headers["authorization"] == "Bearer EMPTY"
    assert headers.get("openai-organization", "") == ""
    assert headers.get("openai-project", "") == ""
    assert "FICTIONAL_AMBIENT_VALUE" not in str(headers)
    assert body["n"] == 1
    assert body["top_k"] == 32
    assert body["min_p"] == 0.05
    effective_seed = request().effective_seed_decimal
    assert effective_seed is not None
    assert body["seed"] == int(effective_seed)
    assert body["max_completion_tokens"] == 1024
    assert "max_tokens" not in body
    assert "response_format" not in body
    assert state["stream"].closed >= 1


async def test_sdk_schema_translation_is_exact_and_final_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = response()
    raw["choices"][0]["message"].update(
        reasoning="Synthetic separate channel.", content='{"answer":"yes","note":null}'
    )
    state = install(monkeypatch, stream=Stream([json.dumps(raw).encode()]))
    value = request(profile="qwen3", json_output=True)
    result = await client().generate_one([UserMessage(content="Synthetic")], request=value)
    body = state["requests"][0][2]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "strata_output",
            "strict": True,
            "schema": value.output_format.wire_payload()["schema"],
        },
    }
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in body
    assert result.reasoning_text == "Synthetic separate channel."
    assert result.status == "completed"


@pytest.mark.parametrize(
    ("status", "headers", "expected"),
    [
        (307, {"location": "https://fixture.invalid/redirect"}, "generation_redirect_refused"),
        (500, {}, "generation_http_error"),
        (200, {"content-length": str(2 * 1024 * 1024 + 1)}, "generation_response_limit_exceeded"),
        (200, {"content-encoding": "gzip"}, "generation_response_encoding_unsupported"),
    ],
)
async def test_headers_and_sdk_error_status_refuse_before_any_body_read(
    monkeypatch: pytest.MonkeyPatch, status: int, headers: dict[str, str], expected: str
) -> None:
    stream = Stream([b"FICTIONAL_PRIVATE_ERROR_BODY"] * 100)
    state = install(monkeypatch, status=status, headers=headers, stream=stream)
    with pytest.raises(GenerationError, match=expected) as caught:
        await client().generate_one([UserMessage(content="Synthetic")], request=request())
    assert stream.reads == 0
    assert stream.closed >= 1
    assert len(state["requests"]) == 1
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "FICTIONAL_PRIVATE_ERROR_BODY" not in str(caught.value)


async def test_unknown_length_stream_stops_at_first_overflow_chunk_not_full_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = Stream([b" " * 65536] * 40)
    state = install(monkeypatch, stream=stream)
    with pytest.raises(GenerationError, match="generation_response_limit_exceeded"):
        await client().generate_one([UserMessage(content="Synthetic")], request=request())
    assert stream.reads == 33
    assert stream.closed >= 1
    assert len(state["requests"]) == 1


async def test_custom_sdk_header_environment_refuses_without_reading_or_sending_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = install(monkeypatch)
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", '{"X-Fictional":"FICTIONAL_HEADER_VALUE"}')
    with pytest.raises(GenerationError, match="generation_sdk_configuration_unsupported"):
        await client().generate_one([UserMessage(content="Synthetic")], request=request())
    assert state["requests"] == []


async def test_cancellation_closes_owned_stream_and_cannot_return_later_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = asyncio.Event()
    stream = Stream([json.dumps(response()).encode()], wait=release)
    state = install(monkeypatch, stream=stream)
    task = asyncio.create_task(
        client().generate_one([UserMessage(content="Synthetic")], request=request())
    )
    await asyncio.wait_for(stream.entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.sleep(0)
    assert task.cancelled()
    assert stream.closed >= 1
    assert len(state["requests"]) == 1


async def test_native_observation_await_has_the_same_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = install(monkeypatch)
    closed = asyncio.Event()

    async def held(_provider: object, _request: object) -> Any:
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(provider, "_observe_generation_endpoint", held)
    value = request().model_copy(update={"deadline_seconds": 0.02})
    with pytest.raises(GenerationError, match="generation_timeout"):
        await client().generate_one([UserMessage(content="Synthetic")], request=value)
    assert closed.is_set()
    assert state["requests"] == []


async def test_original_request_and_message_mutation_during_observer_await_is_not_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = install(monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()

    async def held(_provider: object, current: Any) -> Any:
        entered.set()
        await release.wait()
        return _Endpoint(
            "http://127.0.0.1:54321/v1",
            MODEL,
            current.model_ref,
            current.profile,
            "2.36.0",
            "0.24.0",
        )  # pyright: ignore[reportPrivateUsage] — controlled synthetic wait.

    monkeypatch.setattr(provider, "_observe_generation_endpoint", held)
    original = request()
    message = UserMessage(content="original synthetic")
    task = asyncio.create_task(client().generate_one([message], request=original))
    await asyncio.wait_for(entered.wait(), 5)
    original.stop.append("changed")
    object.__setattr__(message, "content", "changed synthetic")
    release.set()
    assert (await task).status == "completed"
    body = state["requests"][0][2]
    assert body["stop"] == []
    assert body["messages"] == [{"role": "user", "content": "original synthetic"}]


async def test_engine_default_omits_seed_and_harmony_maps_only_declared_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = install(monkeypatch)
    value = request(profile="gpt_oss").model_copy(update={"effective_seed_decimal": None})
    await client().generate_one([UserMessage(content="Synthetic")], request=value)
    body = state["requests"][0][2]
    assert "seed" not in body
    assert body["reasoning_effort"] == "high"
    assert "chat_template_kwargs" not in body
    assert body["n"] == 1


@pytest.mark.parametrize(
    "change", ["profile", "model_ref", "sdk_version", "vllm_version", "base_url"]
)
async def test_mismatched_observation_never_constructs_a_request(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    state = install(monkeypatch)

    async def invalid(_provider: object, current: Any) -> Any:
        from dataclasses import replace

        original = _Endpoint(
            "http://127.0.0.1:54321/v1",
            MODEL,
            current.model_ref,
            current.profile,
            "2.36.0",
            "0.24.0",
        )  # pyright: ignore[reportPrivateUsage] — intentionally mismatched observation.
        changed = {
            "profile": "vllm-0.24-qwen3/v1",
            "model_ref": "fixture/other@" + "c" * 40,
            "sdk_version": "0.0.0",
            "vllm_version": "0.0.0",
            "base_url": "https://fixture.invalid/v1",
        }[change]
        return replace(original, **{change: changed})

    monkeypatch.setattr(provider, "_observe_generation_endpoint", invalid)
    with pytest.raises(GenerationError, match="generation_profile_unavailable"):
        await client().generate_one([UserMessage(content="Synthetic")], request=request())
    assert state["requests"] == []


async def test_sdk_custom_header_guard_observes_only_variable_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import MutableMapping

    class Environment(MutableMapping[str, str]):
        value_reads = 0

        def __iter__(self):
            return iter(("OPENAI_CUSTOM_HEADERS",))

        def __len__(self) -> int:
            return 1

        def __getitem__(self, key: str) -> str:
            if key == "OPENAI_CUSTOM_HEADERS":
                self.value_reads += 1
                raise AssertionError("header value must not be read")
            raise KeyError(key)

        def __setitem__(self, key: str, value: str) -> None:
            raise AssertionError("no environment mutation")

        def __delitem__(self, key: str) -> None:
            raise AssertionError("no environment mutation")

    caller = client()
    state = install(monkeypatch)
    environment = Environment()
    with monkeypatch.context() as scoped:
        scoped.setattr(provider.os, "environ", environment)
        with pytest.raises(GenerationError, match="generation_sdk_configuration_unsupported"):
            await caller.generate_one([UserMessage(content="Synthetic")], request=request())
    assert environment.value_reads == 0
    assert state["requests"] == []


@pytest.mark.parametrize("phase", ["response", "refusal", "transport"])
async def test_repeated_cancellation_drains_owned_close_before_return(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    release, started, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class DelayedCloseStream(Stream):
        async def aclose(self) -> None:
            started.set()
            await release.wait()
            self.closed += 1
            closed.set()

    stream = (
        Stream([json.dumps(response()).encode()])
        if phase == "transport"
        else DelayedCloseStream([], asyncio.Event() if phase == "response" else None)
    )
    install(monkeypatch, stream=stream, status=500 if phase == "refusal" else 200)
    if phase == "transport":
        # The body closes normally; hold the actual SDK-owned HTTP pool instead.
        original_factory = httpx.AsyncHTTPTransport

        def factory(**kwargs: Any) -> httpx.MockTransport:
            inner = cast("httpx.MockTransport", original_factory(**kwargs))
            original_close = inner.aclose

            async def close() -> None:
                started.set()
                await release.wait()
                await original_close()
                closed.set()

            monkeypatch.setattr(inner, "aclose", close)
            return inner

        monkeypatch.setattr(httpx, "AsyncHTTPTransport", factory)
    task = asyncio.create_task(
        client().generate_one([UserMessage(content="Synthetic")], request=request())
    )
    try:
        if phase == "response":
            await asyncio.wait_for(stream.entered.wait(), 5)
            task.cancel()
        await asyncio.wait_for(started.wait(), 5)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert closed.is_set()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_expired_deadline_after_synchronous_validation_never_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = install(monkeypatch)
    original = provider._matching_generation_endpoint  # pyright: ignore[reportPrivateUsage] — fixed observer validation seam under adverse timing.
    loop = asyncio.get_running_loop()
    clock = loop.time
    offset = 0.0

    def delayed(*args: Any) -> bool:
        nonlocal offset
        matched = original(*args)
        offset += 1.0
        return matched

    with monkeypatch.context() as scoped:
        scoped.setattr(loop, "time", lambda: clock() + offset)
        scoped.setattr(provider, "_matching_generation_endpoint", delayed)
        value = request().model_copy(update={"deadline_seconds": 0.5})
        with pytest.raises(GenerationError, match="generation_timeout"):
            await client().generate_one([UserMessage(content="Synthetic")], request=value)
    assert state["requests"] == []


async def test_expired_deadline_at_transport_dispatch_never_calls_inner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = install(monkeypatch)
    transport = provider._GenerationTransport(0.0)  # pyright: ignore[reportPrivateUsage] — actual final dispatch guard, not a native capability claim.
    try:
        with pytest.raises(TimeoutError):
            await transport.handle_async_request(httpx.Request("POST", "http://127.0.0.1:54321/v1"))
    finally:
        await transport.aclose()
    assert state["requests"] == []


async def test_expired_deadline_after_synchronous_response_parse_refuses_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from strata_forge.llm import generation

    state = install(monkeypatch)
    original = generation.parse_generation_response
    loop = asyncio.get_running_loop()
    clock = loop.time
    offset = 0.0

    def delayed(*args: Any, **kwargs: Any) -> generation.GenerationResponse:
        nonlocal offset
        result = original(*args, **kwargs)
        offset += 1.0
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(loop, "time", lambda: clock() + offset)
        scoped.setattr(generation, "parse_generation_response", delayed)
        value = request().model_copy(update={"deadline_seconds": 0.5})
        with pytest.raises(GenerationError, match="generation_timeout"):
            await client().generate_one([UserMessage(content="Synthetic")], request=value)
    assert len(state["requests"]) == 1
    assert state["stream"].closed >= 1
