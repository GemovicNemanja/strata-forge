"""Actual unchanged LLMClient paths with injected synthetic providers, never HTTP or model jobs."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import (  # Raw SDK objects are the existing provider interface.
    TYPE_CHECKING,
    Any,
    ClassVar,
    NoReturn,
    cast,
)

import litellm
import pytest
from pydantic import BaseModel

from strata_forge.core.budget import BudgetContext
from strata_forge.core.errors import BudgetExceededError, FallbackExhaustedError, RegistryError
from strata_forge.llm.cache import InMemoryCache
from strata_forge.llm.client import LLMClient
from strata_forge.llm.fallback import ModelFallback
from strata_forge.llm.messages import Message
from strata_forge.llm.providers.base import ProviderClient
from strata_forge.llm.tools import tool
from strata_forge.tracing.prompt_provider import prepare_provider_observation, wrap_provider
from tests.unit.tracing.test_prompt_provider import issued_invocation

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from strata_forge.llm.registry import Model, ProviderName
    from strata_forge.tracing.prompt_provider import ProviderObservation


def chat(
    text: str = "synthetic", *, tool_calls: list[dict[str, Any]] | None = None
) -> litellm.ModelResponse:
    return litellm.ModelResponse(
        id="fixture-response",
        model="fixture:reported",
        choices=[
            {
                "message": {"role": "assistant", "content": text, "tool_calls": tool_calls},
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    )


def response(text: str = "synthetic") -> dict[str, Any]:
    return {
        "id": "fixture-response",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_fixture",
                "role": "assistant",
                "status": "completed",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    }


class SequenceProvider(ProviderClient):
    name: ClassVar[ProviderName] = "openai_compat"
    litellm_prefix = "fixture/"

    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.described: list[tuple[str, Model]] = []

    def auth_kwargs(self) -> dict[str, Any]:
        raise AssertionError("AUTH MUST REMAIN UNREAD")

    def describe_to_litellm(self, provider_model_id: str, model: Model) -> None:
        self.described.append((provider_model_id, model))

    def result(self) -> object:
        value = self.outputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    async def acompletion(
        self, *, provider_model_id: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        self.calls.append(
            (
                "acompletion",
                {"provider_model_id": provider_model_id, "messages": messages, **kwargs},
            )
        )
        return self.result()

    async def aresponses(self, *, provider_model_id: str, request: dict[str, Any]) -> Any:
        self.calls.append(
            ("aresponses", {"provider_model_id": provider_model_id, "request": request})
        )
        return self.result()


async def capturing(
    delegate: ProviderClient, slot: ProviderName = "openai_compat"
) -> tuple[ProviderObservation, ProviderClient]:
    observation = await prepare_provider_observation(await issued_invocation())
    return observation, wrap_provider(delegate, observation=observation, provider_slot=slot)


async def test_actual_chat_client_cache_and_rich_response_keep_upstream_behavior() -> None:
    raw = chat()
    provider = SequenceProvider([raw])
    observation, wrapped = await capturing(provider)
    client = LLMClient(
        "fixture-model",
        provider="openai_compat",
        provider_clients={"openai_compat": wrapped},
        cache=InMemoryCache(),
    )
    messages = [Message.user("synthetic {{literal}}\r\n😀")]
    first = await client.complete(messages, temperature=0.7, top_p=1.0)
    second = await client.complete(messages, temperature=0.7, top_p=1.0)
    assert first.text == second.text == "synthetic"
    assert not first.cache_hit
    assert second.cache_hit
    assert second.cost_usd == 0.0
    assert len(provider.calls) == 1
    assert provider.calls[0][1]["temperature"] == 0.7
    record = observation.snapshot().attempts[0]
    assert record.call_outcome == "returned"
    assert record.response_projection.status == "opaque"
    assert record.usage is record.cost is record.served_model is None
    assert (
        raw.model == "fixture:reported"
    )  # The recorder did not promote this reported label to served identity.


async def test_actual_responses_builder_description_parse_and_usage_stay_upstream() -> None:
    provider = SequenceProvider([response()])
    observation, wrapped = await capturing(provider, "openai")
    client = LLMClient("gpt-6.1-sol", provider_clients={"openai": wrapped})
    result = await client.complete(
        [Message.system("fictional instruction"), Message.user("synthetic")],
        max_tokens=8,
        provider_extras={"openai": {"reasoning_effort": "none", "temperature": 0.7}},
    )
    assert result.text == "synthetic"
    assert result.route.wire_api == "responses"
    assert result.usage.input_tokens == 5
    request = provider.calls[0][1]["request"]
    assert request["instructions"] == "fictional instruction"
    assert request["store"] is False
    assert request["max_output_tokens"] == 8
    assert request["temperature"] == 0.7
    assert provider.described[0][0] == "gpt-6.1-sol"
    record = observation.snapshot().attempts[0]
    assert record.method == "aresponses"
    assert record.response_projection.status == "observed"
    assert record.usage is record.cost is None  # Existing LLMClient normalized usage independently.


class Answer(BaseModel):
    value: int


async def test_actual_structured_reprompt_creates_distinct_unrelated_observer_entries() -> None:
    provider = SequenceProvider([chat("invalid json"), chat('{"value":7}')])
    observation, wrapped = await capturing(provider)
    client = LLMClient(
        "fixture-model", provider="openai_compat", provider_clients={"openai_compat": wrapped}
    )
    result = await client.complete_structured(
        [Message.user("synthetic")], schema=Answer, max_reprompt_attempts=1
    )
    assert result.parsed.value == 7
    assert len(provider.calls) == 2
    assert len(provider.calls[1][1]["messages"]) > len(provider.calls[0][1]["messages"])
    first, second = observation.snapshot().attempts
    assert first.attempt_id != second.attempt_id
    assert (
        first.request_projection.canonical_tagged_sha256
        != second.request_projection.canonical_tagged_sha256
    )
    assert first.parents is second.parents is None
    assert first.relation == second.relation == "unknown"


class EchoArgs(BaseModel):
    text: str


async def test_actual_nonstreaming_tool_loop_preserves_schemas_results_and_call_count() -> None:
    @tool
    async def fixture_echo(args: EchoArgs) -> str:
        """Return synthetic text without I/O."""
        return args.text

    call = {
        "id": "call_fixture",
        "type": "function",
        "function": {"name": "fixture_echo", "arguments": json.dumps({"text": "synthetic result"})},
    }
    provider = SequenceProvider([chat("", tool_calls=[call]), chat("done")])
    observation, wrapped = await capturing(provider)
    client = LLMClient(
        "fixture-model", provider="openai_compat", provider_clients={"openai_compat": wrapped}
    )
    result = await client.run_tool_loop(
        [Message.user("synthetic")], tools=[fixture_echo], max_iterations=2
    )
    assert result.text == "done"
    assert len(provider.calls) == 2
    assert provider.calls[0][1]["tools"][0]["function"]["name"] == "fixture_echo"
    assert provider.calls[1][1]["messages"][-1]["content"] == "synthetic result"
    assert len(observation.snapshot().attempts) == 2
    assert all(
        item.parents is None and item.relation == "unknown"
        for item in observation.snapshot().attempts
    )


async def test_actual_retry_provider_and_model_fallback_remain_the_clients_policy() -> None:
    from litellm.exceptions import AuthenticationError, RateLimitError

    first = SequenceProvider(
        [
            RateLimitError(message="synthetic", model="fixture", llm_provider="anthropic"),
            RateLimitError(message="synthetic", model="fixture", llm_provider="anthropic"),
        ]
    )
    second = SequenceProvider(
        [AuthenticationError(message="synthetic", model="fixture", llm_provider="bedrock")]
    )
    third = SequenceProvider([chat("fallback")])
    observation = await prepare_provider_observation(await issued_invocation())
    providers: dict[ProviderName, ProviderClient] = {
        slot: wrap_provider(provider, observation=observation, provider_slot=slot)
        for slot, provider in cast(
            "list[tuple[ProviderName, SequenceProvider]]",
            [("anthropic", first), ("bedrock", second), ("openai_compat", third)],
        )
    }
    client = LLMClient(
        chain=[
            ModelFallback(model="claude-opus-4-7", providers=("anthropic", "bedrock")),
            ModelFallback(model="fixture-next", providers=("openai_compat",)),
        ],
        provider_clients=providers,
        retry_max_attempts=2,
        retry_initial_wait=0,
        retry_max_wait=0,
    )
    result = await client.complete([Message.user("synthetic")])
    assert result.text == "fallback"
    assert [len(item.calls) for item in [first, second, third]] == [2, 1, 1]
    records = observation.snapshot().attempts
    assert [item.declared_provider_slot for item in records] == [
        "anthropic",
        "anthropic",
        "bedrock",
        "openai_compat",
    ]
    assert [item.call_outcome for item in records] == ["raised", "raised", "raised", "returned"]
    assert all(item.parents is None and item.relation == "unknown" for item in records)


async def test_actual_budget_rejection_is_upstream_not_a_recorded_provider_failure() -> None:
    provider = SequenceProvider([chat()])
    observation, wrapped = await capturing(provider)
    client = LLMClient(
        "fixture-model", provider="openai_compat", provider_clients={"openai_compat": wrapped}
    )
    async with BudgetContext(max_tokens=1) as budget:
        with pytest.raises(BudgetExceededError):
            await client.complete([Message.user("synthetic")])
        assert budget.spent_tokens == 0
    assert len(provider.calls) == 1
    assert observation.snapshot().attempts[0].call_outcome == "returned"


async def test_original_raw_error_remains_upstream_cause_and_preflight_has_no_provider_fact() -> (
    None
):
    failure = ValueError("FICTIONAL ORIGINAL ERROR")
    provider = SequenceProvider([failure])
    observation, wrapped = await capturing(provider)
    client = LLMClient(
        "fixture-model",
        provider="openai_compat",
        provider_clients={"openai_compat": wrapped},
        retry_max_attempts=1,
    )
    with pytest.raises(FallbackExhaustedError) as caught:
        await client.complete([Message.user("synthetic")])
    assert caught.value.causes[0][2].__cause__ is failure
    assert len(provider.calls) == 1
    assert observation.snapshot().attempts[0].error_category == "delegate_error"
    before = observation.snapshot()
    strict = LLMClient(
        "fixture-model",
        provider="openai_compat",
        provider_clients={"openai_compat": wrapped},
        require_tool_support=True,
    )
    from strata_forge.llm.tools import ToolDeclaration

    with pytest.raises(RegistryError):
        await strict.complete(
            [Message.user("synthetic")],
            tools=[
                ToolDeclaration(
                    name="fixture", description="synthetic", parameters={"type": "object"}
                )
            ],
        )
    assert observation.snapshot() == before
    assert len(provider.calls) == 1


async def test_response_parse_failure_does_not_turn_returned_fact_into_model_success() -> None:
    raw: dict[str, object] = {
        "id": "fixture",
        "status": "failed",
        "error": {"code": "invalid_request_error", "message": "fictional refusal"},
        "output": [],
    }
    provider = SequenceProvider([raw])
    observation, wrapped = await capturing(provider, "openai")
    client = LLMClient("gpt-6.1-sol", provider_clients={"openai": wrapped}, retry_max_attempts=1)
    with pytest.raises(FallbackExhaustedError):
        await client.complete([Message.user("synthetic")])
    assert len(provider.calls) == 1
    record = observation.snapshot().attempts[0]
    assert record.call_outcome == "returned"
    assert record.return_observed
    assert record.response_projection.status == "observed"
    assert record.usage is record.cost is None


async def test_observer_finalization_loss_preserves_actual_client_normalization_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from strata_forge.tracing import prompt_provider as observer

    provider = SequenceProvider([chat("still returned")])
    observation, wrapped = await capturing(provider)

    def broken(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("OBSERVER CANARY")

    monkeypatch.setattr(observer, "_metadata", broken)
    client = LLMClient(
        "fixture-model",
        provider="openai_compat",
        provider_clients={"openai_compat": wrapped},
        cache=InMemoryCache(),
    )
    first = await client.complete([Message.user("synthetic")])
    second = await client.complete([Message.user("synthetic")])
    assert first.text == second.text == "still returned"
    assert second.cache_hit
    assert len(provider.calls) == 1
    snapshot = observation.snapshot()
    assert snapshot.admitted_attempts == snapshot.loss.metadata_lost_count == 1
    assert snapshot.attempts == ()


async def test_actual_client_streaming_stays_unobserved_and_bypasses_cache() -> None:
    class Streaming(SequenceProvider):
        async def astream(
            self, *, provider_model_id: str, messages: list[dict[str, Any]], **kwargs: Any
        ) -> AsyncIterator[Any]:
            self.calls.append(("astream", {"messages": messages}))
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="synthetic", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )

    class ForbiddenCache(InMemoryCache):
        async def get(self, key: str) -> Any:
            pytest.fail("stream cache must remain bypassed")

    provider = Streaming([])
    observation, wrapped = await capturing(provider)
    client = LLMClient(
        "fixture-model",
        provider="openai_compat",
        provider_clients={"openai_compat": wrapped},
        cache=ForbiddenCache(),
    )
    chunks = [chunk async for chunk in await client.stream([Message.user("synthetic")])]
    assert chunks[0].delta_text == "synthetic"
    assert len(provider.calls) == 1
    assert observation.snapshot().admitted_attempts == 0


async def test_existing_exact_type_generation_admission_is_preserved_not_spoofed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from strata_forge.llm.generation import GenerationError, GenerationRequest
    from strata_forge.llm.providers.openai_compat import OpenAICompatProvider

    # This test isolates admission, not a generation transport/model claim.
    original = object.__new__(OpenAICompatProvider)
    sentinel = object()
    calls: list[object] = []

    async def fake_generation(_self: object, _text: str, request: GenerationRequest) -> Any:
        calls.append(request)
        return sentinel

    monkeypatch.setattr(OpenAICompatProvider, "agenerate_one", fake_generation)
    request = cast("GenerationRequest", object())
    direct = LLMClient(
        "fixture-model", provider="openai_compat", provider_clients={"openai_compat": original}
    )
    assert await direct.generate_one([Message.user("synthetic")], request=request) is sentinel
    observation, wrapped = await capturing(original)
    observed_client = LLMClient(
        "fixture-model", provider="openai_compat", provider_clients={"openai_compat": wrapped}
    )
    with pytest.raises(GenerationError, match="generation_client_configuration_unsupported"):
        await observed_client.generate_one([Message.user("synthetic")], request=request)
    assert calls == [request]
    assert observation.snapshot().admitted_attempts == 0


async def test_actual_client_cancellation_keeps_original_delegate_control_and_one_entry() -> None:
    import asyncio

    entered = asyncio.Event()

    class Held(SequenceProvider):
        cancelled: asyncio.CancelledError | None = None

        async def acompletion(
            self, *, provider_model_id: str, messages: list[dict[str, Any]], **kwargs: Any
        ) -> Any:
            self.calls.append(("acompletion", {"messages": messages}))
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as error:
                self.cancelled = error
                raise

    provider = Held([])
    observation, wrapped = await capturing(provider)
    client = LLMClient(
        "fixture-model", provider="openai_compat", provider_clients={"openai_compat": wrapped}
    )
    task = asyncio.create_task(client.complete([Message.user("synthetic")]))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value is provider.cancelled
    assert len(provider.calls) == 1
    assert observation.snapshot().attempts[0].call_outcome == "cancelled"


def test_cold_explicit_facade_import_and_fake_call_do_not_reach_network_assets_or_exporter() -> (
    None
):
    import subprocess
    import sys
    from pathlib import Path

    program = r"""
import asyncio,importlib.abc,json,pathlib,socket,sys
fixture=json.loads(pathlib.Path(sys.argv[1]).read_text())
attempts=[]
def denied(*args,**kwargs):
    attempts.append("network_or_asset")
    raise AssertionError("forbidden")
socket.socket.connect=denied
socket.socket.connect_ex=denied
socket.create_connection=denied
socket.getaddrinfo=denied
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in {"torch","transformers","trl","peft","numpy","langfuse","pyarrow","skypilot"}:
            attempts.append(fullname)
            raise AssertionError("optional_import_forbidden")
sys.meta_path.insert(0,Guard())
import tiktoken
tiktoken.get_encoding=denied
tiktoken.encoding_for_model=denied
from strata_forge.prompts import LocalBuildSource,consume_build
from strata_forge.llm.providers.base import ProviderClient
from strata_forge.tracing import prepare_invocation,prepare_provider_observation,wrap_provider
class Fake(ProviderClient):
    name="openai_compat"
    litellm_prefix="fixture/"
    def __init__(self):pass
    def auth_kwargs(self):raise AssertionError("auth_unread")
    async def acompletion(self,**kwargs):return {"choices":[]}
async def main():
    raw=bytes.fromhex(fixture["build_canonical_utf8_hex"])
    consumed=consume_build(raw,expected_build_id=fixture["build_id"],expected_size_bytes=len(raw),source=LocalBuildSource(kind="caller_bytes",partition="fixture:cold",reference=None))
    prepared=await prepare_invocation(consumed,fixture["values"])
    observation=await prepare_provider_observation(prepared)
    wrapped=wrap_provider(Fake(),observation=observation,provider_slot="openai_compat")
    assert await wrapped.acompletion(provider_model_id="fixture",messages=[])=={"choices":[]}
    assert observation.snapshot().attempts[0].call_outcome=="returned"
asyncio.run(main())
assert attempts==[],attempts
print("provider-local-only")
"""
    fixture = Path(__file__).parents[3] / "tests/fixtures/prompts/local-attempts-v1.json"
    result = subprocess.run(  # noqa: S603 - fixed local fake program and synthetic fixture only
        [sys.executable, "-c", program, str(fixture)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "provider-local-only"
