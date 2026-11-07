"""Exact local generation wire translation and channel parsing without model/provider execution."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from strata_forge.llm._generation_schema import GenerationError
from strata_forge.llm.generation import (
    GenerationRequest,
    parse_generation_response,
    request_body,
)

# These are heterogeneous literal protocol objects, never provider responses or research data.
FIXTURE: dict[str, Any] = json.loads(
    (Path(__file__).parents[1] / "pipelines/fixtures/generation_v2.json").read_text()
)
MODEL = "strata-fictional"


def request(*, profile: str = "plain", json_output: bool = False) -> GenerationRequest:
    raw = copy.deepcopy(FIXTURE["generation_spec"]["generation"])
    raw.pop("samples")
    raw.pop("seed")
    if json_output:
        raw["output_format"] = {
            "kind": "json_schema",
            "dialect": "strata-json-schema-v1",
            "schema": copy.deepcopy(FIXTURE["schema"]),
        }
    if profile == "qwen3":
        raw["channels"] = {"mode": "qwen3", "enable_thinking": True}
    elif profile == "gpt_oss":
        raw["channels"] = {"mode": "gpt_oss", "effort": "high"}
    raw.update(
        model_ref="fixture/model@" + "b" * 40,
        profile={
            "plain": "vllm-0.24-chat-text/v1",
            "qwen3": "vllm-0.24-qwen3/v1",
            "gpt_oss": "vllm-0.24-gpt-oss-harmony/v1",
        }[profile],
        effective_seed_decimal=FIXTURE["seed_vectors"][0]["effective_seed_decimal"],
        deadline_seconds=30.0,
    )
    return GenerationRequest.model_validate(raw)


def response() -> dict[str, Any]:
    return {
        "id": "fictional-id",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "<think>literal text</think>{unchanged}",
                    "reasoning": None,
                    "refusal": None,
                },
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
        "system_fingerprint": None,
    }


def parse(raw: dict[str, Any], value: GenerationRequest | None = None):
    return parse_generation_response(
        value or request(), json.dumps(raw).encode(), served_model=MODEL
    )


def test_single_choice_wire_preserves_all_controls_and_exact_schema() -> None:
    value = request(json_output=True)
    text = "Synthetic \r\n{literal}\u03b1"
    body = request_body(value, user_text=text, served_model=MODEL)
    assert body == {
        **FIXTURE["one_sample_engine_request"],
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "response_format": FIXTURE["json_response_format"],
    }
    assert "max_tokens" not in body
    assert "guided_json" not in body
    assert "structured_outputs" not in body
    assert "reasoning_effort" not in body
    assert "chat_template_kwargs" not in body


def test_only_closed_qwen_and_harmony_native_controls_are_mapped() -> None:
    qwen = request_body(request(profile="qwen3"), user_text="text", served_model=MODEL)
    assert qwen["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in qwen
    harmony = request_body(request(profile="gpt_oss"), user_text="text", served_model=MODEL)
    assert harmony["reasoning_effort"] == "high"
    assert "chat_template_kwargs" not in harmony


def test_literal_tags_and_absent_null_metadata_are_not_repaired() -> None:
    raw = response()
    value = parse(raw)
    assert value.final_text == raw["choices"][0]["message"]["content"]
    assert value.reasoning_present
    assert value.reasoning_text is None
    assert value.refusal_present
    assert value.refusal_text is None
    assert value.response_id_present
    assert value.response_id == "fictional-id"
    assert value.system_fingerprint_present
    assert value.system_fingerprint is None
    del raw["id"]
    del raw["system_fingerprint"]
    value = parse(raw)
    assert value.response_id_present is False
    assert value.response_id is None
    assert value.system_fingerprint_present is False
    assert value.system_fingerprint is None
    raw.update(id=None, system_fingerprint="fp-fictional")
    value = parse(raw)
    assert value.response_id_present
    assert value.response_id is None
    assert value.system_fingerprint == "fp-fictional"


def test_explicit_reasoning_and_final_json_are_independent() -> None:
    raw = response()
    raw["choices"][0]["message"].update(
        reasoning="Synthetic separate thought.", content='{"answer":"yes","note":null}'
    )
    raw["usage"]["completion_tokens_details"] = {"reasoning_tokens": 3}
    value = parse(raw, request(profile="qwen3", json_output=True))
    assert value.final_text == '{"answer":"yes","note":null}'
    assert value.reasoning_text == "Synthetic separate thought."
    assert value.parsed_json == {"answer": "yes", "note": None}
    assert value.status == "completed"
    assert value.usage.generated_tokens == 7
    assert value.usage.reasoning_tokens == 3
    assert value.usage.final_tokens == 4


@pytest.mark.parametrize("final", ["[]", "null", '"a"', '{"answer":"bad","note":null}'])
def test_valid_json_wrong_schema_retained_as_invalid_without_coercion(final: str) -> None:
    raw = response()
    raw["choices"][0]["message"]["content"] = final
    value = parse(raw, request(json_output=True))
    assert value.json_status == "valid"
    assert value.schema_status == "invalid"
    assert value.status == "invalid_schema"
    assert value.parsed_json == json.loads(final)


@pytest.mark.parametrize(
    "final",
    [
        '```json\n{"answer":"yes","note":null}\n```',
        '{"answer":"yes","answer":"no","note":null}',
        "NaN",
        "{broken",
    ],
)
def test_malformed_final_not_repaired_or_extracted_from_reasoning(final: str) -> None:
    raw = response()
    raw["choices"][0]["message"].update(content=final, reasoning='{"answer":"yes","note":null}')
    value = parse(raw, request(profile="gpt_oss", json_output=True))
    assert value.status == "invalid_json"
    assert value.final_text == final
    assert value.parsed_json is None
    assert value.schema_status == "not_evaluated"


def test_length_end_keeps_json_validity_without_success_and_analysis_is_never_final() -> None:
    raw = response()
    raw["choices"][0].update(finish_reason="length")
    raw["choices"][0]["message"].update(
        content='{"answer":"yes","note":null}', reasoning="Synthetic thinking."
    )
    value = parse(raw, request(profile="gpt_oss", json_output=True))
    assert value.status == "incomplete_length"
    assert value.schema_status == "valid"
    raw["choices"][0]["message"]["content"] = None
    value = parse(raw, request(profile="gpt_oss", json_output=True))
    assert value.final_text is None
    assert value.reasoning_text == "Synthetic thinking."
    assert value.status == "incomplete_length"
    assert value.json_status == "invalid"


@pytest.mark.parametrize(
    "change",
    [
        "wrong_model",
        "extra_top",
        "extra_message",
        "two_choices",
        "wrong_index",
        "bool_index",
        "unknown_finish",
        "missing_final",
        "reasoning_alias",
        "unexpected_reasoning",
        "non_text",
        "tool",
        "null_choices",
        "id_bool",
        "id_long",
        "fingerprint_list",
        "fingerprint_control",
        "usage_bool",
        "usage_negative",
        "usage_mismatch",
    ],
)
def test_malformed_protocol_fixed_context_free_refusal(change: str) -> None:
    raw = response()
    message = raw["choices"][0]["message"]
    if change == "wrong_model":
        raw["model"] = "other"
    elif change == "extra_top":
        raw["extra"] = True
    elif change == "extra_message":
        message["extra"] = True
    elif change == "two_choices":
        raw["choices"].append(copy.deepcopy(raw["choices"][0]))
    elif change == "wrong_index":
        raw["choices"][0]["index"] = 1
    elif change == "bool_index":
        raw["choices"][0]["index"] = False
    elif change == "unknown_finish":
        raw["choices"][0]["finish_reason"] = "future"
    elif change == "missing_final":
        del message["content"]
    elif change == "reasoning_alias":
        message["reasoning_content"] = "alias"
    elif change == "unexpected_reasoning":
        message["reasoning"] = "not plain"
    elif change == "non_text":
        message["content"] = [{"type": "text", "text": "mixed"}]
    elif change == "tool":
        message["tool_calls"] = [{"id": "fictional"}]
    elif change == "null_choices":
        raw["choices"] = None
    elif change == "id_bool":
        raw["id"] = True
    elif change == "id_long":
        raw["id"] = "x" * 513
    elif change == "fingerprint_list":
        raw["system_fingerprint"] = ["bad"]
    elif change == "fingerprint_control":
        raw["system_fingerprint"] = "x\ny"
    elif change == "usage_bool":
        raw["usage"]["completion_tokens"] = True
    elif change == "usage_negative":
        raw["usage"]["completion_tokens"] = -1
    else:
        raw["usage"]["total_tokens"] = 999
    with pytest.raises(GenerationError) as caught:
        parse(raw)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert caught.value.args == ("generation_response_invalid",)
