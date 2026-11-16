"""Actual unchanged LLMClient parsers/loops over injected fictional native streams only."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast  # Existing raw ProviderClient/SDK fixture seam.

import pytest
from pydantic import BaseModel

from strata_forge.core.errors import ProviderError, ProviderServerError
from strata_forge.core.errors import ValidationError as ForgeValidationError
from strata_forge.llm.cache import InMemoryCache
from strata_forge.llm.client import LLMClient
from strata_forge.llm.fallback import ModelFallback
from strata_forge.llm.loop_events import Done, LoopError, PendingToolCalls, TextDelta, ToolResult
from strata_forge.llm.messages import Message, ToolResultMessage
from strata_forge.llm.tools import ToolDeclaration, tool
from strata_forge.tracing.prompt_provider import prepare_provider_observation, wrap_provider
from strata_forge.tracing.prompt_streams import prepare_stream_observation, wrap_streaming_provider
from tests.unit.llm.test_client import (
    _rs_text_events,  # pyright: ignore[reportPrivateUsage] -- fixed synthetic parser fixture
    _rs_tool_events,  # pyright: ignore[reportPrivateUsage] -- fixed synthetic parser fixture
)
from tests.unit.tracing.test_prompt_provider import FakeProvider, issued_invocation

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from strata_forge.llm.registry import Model
    from strata_forge.tracing.prompt_streams import ProviderStreamObservation


class EchoArgs(BaseModel):
    label: str


def chat_text(text: str, *, finish: str | None = "stop") -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None), finish_reason=finish
            )
        ],
        usage=None,
    )


def chat_tool(name: str, arguments: str) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="fixture-call",
                            function=SimpleNamespace(name=name, arguments=arguments),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
    )


class NativeTurns(FakeProvider):
    def __init__(self, turns: list[list[object]]) -> None:
        super().__init__()
        self.turns = list(turns)
        self.described: list[tuple[str, Model]] = []

    def describe_to_litellm(self, provider_model_id: str, model: Model) -> None:
        self.described.append((provider_model_id, model))

    async def astream(
        self,
        *,
        provider_model_id: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        self.calls.append(
            ("astream", {"provider_model_id": provider_model_id, "messages": messages, **kwargs})
        )
        for item in self.turns.pop(0):
            if isinstance(item, BaseException):
                raise item
            yield item

    async def aresponses_stream(
        self,
        *,
        provider_model_id: str,
        request: dict[str, Any],
    ) -> AsyncIterator[Any]:
        self.calls.append(
            ("aresponses_stream", {"provider_model_id": provider_model_id, "request": request})
        )
        for item in self.turns.pop(0):
            if isinstance(item, BaseException):
                raise item
            yield item


async def client_for(
    turns: list[list[object]],
    *,
    responses: bool = False,
    cache: InMemoryCache | None = None,
) -> tuple[ProviderStreamObservation, NativeTurns, LLMClient]:
    observer = await prepare_stream_observation(await issued_invocation())
    original = NativeTurns(turns)
    slot = "openai" if responses else "openai_compat"
    wrapped = wrap_streaming_provider(original, observation=observer, provider_slot=slot)
    client = LLMClient(
        "gpt-6.1-sol" if responses else "fixture-model",
        provider=None if responses else "openai_compat",
        provider_clients={slot: wrapped},
        cache=cache,
        retry_max_attempts=1,
    )
    return observer, original, client


def turn_text(text: str, responses: bool) -> list[object]:
    return list(_rs_text_events(text)) if responses else [chat_text(text)]


def turn_tool(name: str, arguments: str, responses: bool) -> list[object]:
    return (
        list(_rs_tool_events([("fixture-call", name, arguments)]))
        if responses
        else [chat_tool(name, arguments)]
    )


@pytest.mark.parametrize("responses", [False, True])
async def test_actual_client_stream_preserves_normalized_chunks_and_uninterpreted_raw_counts(
    responses: bool,
) -> None:
    raw = turn_text("fictional 雪😀", responses)
    observer, original, client = await client_for([raw], responses=responses)
    stream = await client.stream(
        [Message.system("fictional instruction"), Message.user("fictional input")]
    )
    assert observer.snapshot().admitted_streams == 0
    chunks = [chunk async for chunk in stream]
    assert "".join(chunk.delta_text for chunk in chunks) == "fictional 雪😀"
    assert chunks[-1].finish_reason == "stop"
    assert len(original.calls) == 1
    record = observer.snapshot().streams[0]
    assert record.raw_items_observed == len(raw)
    assert record.end_observation == "exhausted"
    assert record.method == ("aresponses_stream" if responses else "astream")
    assert record.input_binding.normalized_schema == ("aresponses" if responses else "acompletion")
    assert record.usage is record.cost is record.served_model is record.output_projection is None
    assert "fictional input" not in record.canonical().decode()
    if responses:
        assert original.described
        request = cast("dict[str, object]", original.calls[0][1]["request"])
        assert request["store"] is False
        assert request["stream"] is True
        assert record.raw_items_observed > len(chunks)


async def test_actual_responses_missing_terminal_remains_client_error_after_local_exhaustion() -> (
    None
):
    raw = list(_rs_text_events("fictional"))[:-1]
    observer, original, client = await client_for([list(raw)], responses=True)
    with pytest.raises(ProviderServerError, match="before the response completed"):
        _ = [chunk async for chunk in await client.stream([Message.user("fictional")])]
    record = observer.snapshot().streams[0]
    assert record.end_observation == "exhausted"
    assert record.raw_items_observed == len(raw)
    assert record.served_model is None
    assert len(original.calls) == 1


async def test_actual_stream_bypasses_cache_retry_and_fallback_without_extra_dispatch() -> None:
    class ForbiddenCache(InMemoryCache):
        async def get(self, key: str) -> Any:
            pytest.fail("stream must not read cache")

    observer = await prepare_stream_observation(await issued_invocation())
    failure = RuntimeError("fictional transport error")
    original = NativeTurns([[failure], [chat_text("must not be used")]])
    wrapped = wrap_streaming_provider(original, observation=observer, provider_slot="openai_compat")
    client = LLMClient(
        chain=[
            ModelFallback("fixture-first", providers=("openai_compat",)),
            ModelFallback("fixture-second", providers=("openai_compat",)),
        ],
        provider_clients={"openai_compat": wrapped},
        cache=ForbiddenCache(),
        retry_max_attempts=3,
    )
    with pytest.raises(ProviderError, match="Unmapped LiteLLM exception") as caught:
        _ = [chunk async for chunk in await client.stream([Message.user("fictional")])]
    assert type(caught.value) is ProviderError
    assert caught.value.__cause__ is failure
    assert len(original.calls) == 1
    assert original.calls[0][1]["provider_model_id"] == "fixture-first"
    record = observer.snapshot().streams[0]
    assert record.operation_errors == 1
    assert record.end_observation is None


@pytest.mark.parametrize("responses", [False, True])
async def test_actual_tool_execution_and_followup_preserve_events_and_unknown_parents(
    responses: bool,
) -> None:
    executed: list[str] = []

    @tool
    async def echo(args: EchoArgs) -> str:
        """Return a fictional label."""
        executed.append(args.label)
        return args.label

    observer, original, client = await client_for(
        [
            turn_tool("echo", '{"label":"fictional-tool"}', responses),
            turn_text("fictional answer", responses),
        ],
        responses=responses,
    )
    events = [
        event async for event in client.stream_tool_loop([Message.user("fictional")], tools=[echo])
    ]
    expected = ["IterationStart", "ToolCallStarted", "ToolResult", "IterationStart"]
    expected.extend(["TextDelta", "TextDelta"] if responses else ["TextDelta"])
    expected.append("Done")
    assert [type(event).__name__ for event in events] == expected
    assert executed == ["fictional-tool"]
    result = next(event for event in events if isinstance(event, ToolResult))
    assert result.content == "fictional-tool"
    assert isinstance(events[-1], Done)
    assert len(original.calls) == 2
    snapshot = observer.snapshot()
    assert len(snapshot.streams) == 2
    assert all(
        record.parents is None and record.relation == "unknown" for record in snapshot.streams
    )
    if responses:
        request = cast("dict[str, object]", original.calls[1][1]["request"])
        body = cast("list[dict[str, object]]", request["input"])
        assert any(item.get("output") == "fictional-tool" for item in body)
    else:
        body = cast("list[dict[str, object]]", original.calls[1][1]["messages"])
        assert body[-1]["content"] == "fictional-tool"


@pytest.mark.parametrize("responses", [False, True])
async def test_actual_pending_declaration_explicit_resume_does_not_reinvoke_a_tool(
    responses: bool,
) -> None:
    declared = ToolDeclaration(
        name="fixture_action",
        description="Fictional caller action",
        parameters={"type": "object"},
    )
    observer, original, client = await client_for(
        [
            turn_tool("fixture_action", '{"value":"fictional"}', responses),
            turn_text("fictional resumed", responses),
        ],
        responses=responses,
    )
    prompt = [Message.user("fictional")]
    events = [event async for event in client.stream_tool_loop(prompt, tools=[declared])]
    pending = events[-1]
    assert isinstance(pending, PendingToolCalls)
    assert pending.calls[0].arguments == {"value": "fictional"}
    assert not any(isinstance(event, ToolResult) for event in events)
    assert len(original.calls) == 1
    resumed = [
        *prompt,
        *pending.messages,
        ToolResultMessage(tool_call_id="fixture-call", content="fictional caller result"),
    ]
    after = [event async for event in client.stream_tool_loop(resumed, tools=[declared])]
    assert isinstance(after[-1], Done)
    assert (
        "".join(event.text for event in after if isinstance(event, TextDelta))
        == "fictional resumed"
    )
    assert len(original.calls) == 2
    assert observer.snapshot().admitted_streams == 2
    assert all(record.parents is None for record in observer.snapshot().streams)


@pytest.mark.parametrize("failure", ["transport", "malformed", "iterations"])
async def test_actual_loop_error_and_iteration_limit_stay_upstream(failure: str) -> None:
    @tool
    async def echo(args: EchoArgs) -> str:
        """Return a fictional label."""
        return args.label

    turn: list[object] = (
        [RuntimeError("fictional transport")]
        if failure == "transport"
        else [
            chat_tool("echo", "malformed" if failure == "malformed" else '{"label":"fictional"}'),
        ]
    )
    observer, original, client = await client_for([turn])
    events = [
        event
        async for event in client.stream_tool_loop(
            [Message.user("fictional")],
            tools=[echo],
            max_iterations=1,
        )
    ]
    assert isinstance(events[-1], LoopError)
    assert events[-1].exceeded_max_iterations == (failure == "iterations")
    assert len(original.calls) == 1
    assert observer.snapshot().admitted_streams == 1
    assert observer.snapshot().streams[0].parents is None


async def test_actual_client_preflight_refuses_without_admitting_a_provider_stream() -> None:
    observer, original, client = await client_for([])
    with pytest.raises(ForgeValidationError):
        await client.stream([ToolResultMessage(tool_call_id="fictional", content="fictional")])
    assert original.calls == []
    assert observer.snapshot().admitted_streams == 0


async def test_actual_client_cancellation_stays_caller_owned_and_not_exhaustion() -> None:
    entered = asyncio.Event()
    cancellation: list[BaseException] = []

    class Held(NativeTurns):
        async def astream(
            self,
            *,
            provider_model_id: str,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> AsyncIterator[Any]:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as error:
                cancellation.append(error)
                raise
            yield None

    observer = await prepare_stream_observation(await issued_invocation())
    original = Held([])
    wrapped = wrap_streaming_provider(original, observation=observer, provider_slot="openai_compat")
    client = LLMClient(
        "fixture-model", provider="openai_compat", provider_clients={"openai_compat": wrapped}
    )

    async def read() -> None:
        _ = [chunk async for chunk in await client.stream([Message.user("fictional")])]

    work = asyncio.create_task(read())
    await entered.wait()
    work.cancel("fictional cancellation")
    with pytest.raises(asyncio.CancelledError) as caught:
        await work
    assert caught.value is cancellation[0]
    record = observer.snapshot().streams[0]
    assert record.operation_cancellations == 1
    assert record.end_observation is None


async def test_default_identity_and_explicit_profile_composition_remain_separate() -> None:
    invocation = await issued_invocation()
    nonstreaming = await prepare_provider_observation(invocation)
    streaming = await prepare_stream_observation(invocation)
    original = NativeTurns([[chat_text("fictional")]])
    default = wrap_provider(original, observation=nonstreaming, provider_slot="openai_compat")
    wrapped = wrap_streaming_provider(default, observation=streaming, provider_slot="openai_compat")
    client = LLMClient(
        "fixture", provider="openai_compat", provider_clients={"openai_compat": wrapped}
    )
    chunks = [chunk async for chunk in await client.stream([Message.user("fictional")])]
    assert chunks[0].delta_text == "fictional"
    assert nonstreaming.snapshot().admitted_attempts == 0
    assert streaming.snapshot().admitted_streams == 1


async def test_default_nonstreaming_profile_keeps_actual_native_iterator_identity() -> None:
    from tests.unit.tracing.test_prompt_streams import StreamFake, sequence

    native = sequence([])
    original = StreamFake(native)
    observer = await prepare_provider_observation(await issued_invocation())
    default = wrap_provider(original, observation=observer, provider_slot="openai_compat")
    returned = default.astream(provider_model_id="fixture", messages=[])
    assert returned is native
    assert observer.snapshot().admitted_attempts == 0
    await native.aclose()


def test_cold_explicit_stream_facade_does_not_reach_network_assets_or_exporter() -> None:
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    program = r"""
import asyncio,importlib.abc,json,pathlib,socket,sys
attempts=[]
def refused(*args,**kwargs):
 attempts.append("network_or_asset");raise AssertionError("forbidden")
socket.socket.connect=refused;socket.socket.connect_ex=refused
socket.create_connection=refused;socket.getaddrinfo=refused
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in {"torch","transformers","trl","peft","numpy","langfuse","pyarrow","skypilot"}:
   attempts.append(fullname);raise AssertionError("optional_import_forbidden")
sys.meta_path.insert(0,Guard())
sys.path.insert(0,sys.argv[2])
import tiktoken
tiktoken.get_encoding=refused;tiktoken.encoding_for_model=refused
from strata_forge.prompts import LocalBuildSource,consume_build
from strata_forge.llm.providers.base import ProviderClient
from strata_forge.tracing import prepare_invocation,prepare_stream_observation,wrap_streaming_provider
fixture=json.loads(pathlib.Path(sys.argv[1]).read_text())
class Fake(ProviderClient):
 name="openai_compat"
 litellm_prefix="fixture/"
 def __init__(self):pass
 def auth_kwargs(self):raise AssertionError("auth_unread")
 async def astream(self,**kwargs):yield None
async def main():
 raw=bytes.fromhex(fixture["build_canonical_utf8_hex"])
 consumed=consume_build(raw,expected_build_id=fixture["build_id"],expected_size_bytes=len(raw),source=LocalBuildSource(kind="caller_bytes",partition="fixture:cold-stream",reference=None))
 prepared=await prepare_invocation(consumed,fixture["values"])
 observation=await prepare_stream_observation(prepared)
 wrapped=wrap_streaming_provider(Fake(),observation=observation,provider_slot="openai_compat")
 view=wrapped.astream(provider_model_id="fixture",messages=[])
 assert await anext(view) is None
 try:await anext(view)
 except StopAsyncIteration:pass
 else:raise AssertionError("native exhaustion missing")
 record=observation.snapshot().streams[0]
 assert record.raw_items_observed==1 and record.end_observation=="exhausted"
 assert record.source_kind=="caller_bytes" and record.served_model is None
 observation.close()
asyncio.run(main())
assert attempts==[],attempts
print(json.dumps({"forbidden_attempts":len(attempts),"scope":"local_fake_stream_only"}))
"""
    root = Path(__file__).parents[3]
    with tempfile.TemporaryDirectory(prefix="strata-stream-cold-") as temporary:
        result = subprocess.run(  # noqa: S603 - fixed local synthetic program/source/fixture only
            [
                sys.executable,
                "-I",
                "-c",
                program,
                str(root / "tests/fixtures/prompts/local-attempts-v1.json"),
                str(root / "src"),
            ],
            cwd=temporary,
            capture_output=True,
            text=True,
            check=False,
            env={
                "HOME": os.environ["HOME"],
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"forbidden_attempts": 0, "scope": "local_fake_stream_only"}
