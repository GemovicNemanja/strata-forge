"""Closed generation controls and responses independent of native-engine capability or execution."""

from __future__ import annotations

import hashlib
import re
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.llm._generation_schema import (
    MAX_SAFE_INTEGER,
    GenerationError,
    JsonValue,
    canonical_json,
    conforms,
    parse_json,
    scalar_text,
    validate_schema,
)

type GenerationProfile = Literal[
    "vllm-0.24-chat-text/v1", "vllm-0.24-qwen3/v1", "vllm-0.24-gpt-oss-harmony/v1"
]

_REPO_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?@[0-9a-f]{40}")
MAX_GENERATION_SPEC_BYTES = 1024 * 1024
MAX_GENERATION_CHANNEL_BYTES = 1024 * 1024
MAX_GENERATION_RESPONSE_BYTES = 2 * 1024 * 1024


class GenerationData(BaseModel):
    """Strict inert values; operations revalidate copies before treating them as effective data."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        serialize_by_alias=True,
    )

    def wire_payload(self) -> dict[str, JsonValue]:
        return cast("dict[str, JsonValue]", self.model_dump(mode="json", by_alias=True))

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.wire_payload())).hexdigest()


class TextOutput(GenerationData):
    kind: Literal["text"]


class JsonSchemaOutput(GenerationData):
    kind: Literal["json_schema"]
    dialect: Literal["strata-json-schema-v1"]
    schema_body: dict[str, JsonValue] = Field(alias="schema")

    @field_validator("schema_body", mode="before")
    @classmethod
    def _schema(cls, value: object) -> dict[str, JsonValue]:
        return validate_schema(value)


type OutputFormat = Annotated[TextOutput | JsonSchemaOutput, Field(discriminator="kind")]


class PlainChannels(GenerationData):
    mode: Literal["plain"]


class QwenChannels(GenerationData):
    mode: Literal["qwen3"]
    enable_thinking: bool


class HarmonyChannels(GenerationData):
    mode: Literal["gpt_oss"]
    effort: Literal["low", "medium", "high"]


type ChannelPolicy = Annotated[
    PlainChannels | QwenChannels | HarmonyChannels, Field(discriminator="mode")
]


class DerivedSeed(GenerationData):
    mode: Literal["per_item_sample_v1"]
    base_seed: int = Field(ge=0, le=MAX_SAFE_INTEGER)


class EngineDefaultSeed(GenerationData):
    mode: Literal["engine_default"]


type SeedPolicy = Annotated[DerivedSeed | EngineDefaultSeed, Field(discriminator="mode")]


class GenerationOptions(GenerationData):
    temperature: float = Field(ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(gt=0, le=1, allow_inf_nan=False)
    top_k: int
    min_p: float = Field(ge=0, le=1, allow_inf_nan=False)
    max_tokens: int = Field(ge=1, le=32768)
    stop: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(max_length=8)
    output_format: OutputFormat
    channels: ChannelPolicy

    @field_validator("output_format", mode="before")
    @classmethod
    def _output_snapshot(cls, value: object) -> object:
        # Pydantic instance revalidation otherwise feeds the Python alias name back as wire data.
        if type(value) is JsonSchemaOutput:
            if (
                set(value.__dict__) != set(JsonSchemaOutput.model_fields)
                or value.__pydantic_extra__
            ):
                raise ValueError("generation_controls_invalid")
            return value.wire_payload()
        return value

    @field_validator("top_k")
    @classmethod
    def _top_k(cls, value: int) -> int:
        if value != -1 and not 1 <= value <= 1024:
            raise ValueError("generation_controls_invalid")
        return value

    @field_validator("stop")
    @classmethod
    def _stop(cls, value: list[str]) -> list[str]:
        if (
            len(set(value)) != len(value)
            or any(
                not scalar_text(item) or not item or len(item.encode("utf-8")) > 256
                for item in value
            )
            or sum(len(item.encode("utf-8")) for item in value) > 1024
        ):
            raise ValueError("generation_controls_invalid")
        return value


class GenerationControls(GenerationOptions):
    samples: int = Field(ge=1, le=16)
    seed: SeedPolicy


class GenerationRequest(GenerationOptions):
    """One logical sample request; endpoint identity is supplied only by the fixed observer."""

    model_ref: str = Field(max_length=512)
    profile: GenerationProfile
    effective_seed_decimal: str | None
    deadline_seconds: float = Field(gt=0, le=300, allow_inf_nan=False)

    @field_validator("model_ref")
    @classmethod
    def _model_ref(cls, value: str) -> str:
        if not _REPO_REF.fullmatch(value) or ".." in value.split("@", 1)[0]:
            raise ValueError("generation_model_ref_invalid")
        return value

    @field_validator("effective_seed_decimal")
    @classmethod
    def _seed(cls, value: str | None) -> str | None:
        if value is not None and (
            not re.fullmatch(r"0|[1-9][0-9]{0,18}", value) or int(value) > 2**63 - 1
        ):
            raise ValueError("generation_seed_invalid")
        return value

    @model_validator(mode="after")
    def _profile(self) -> Self:
        expected = {
            "vllm-0.24-chat-text/v1": "plain",
            "vllm-0.24-qwen3/v1": "qwen3",
            "vllm-0.24-gpt-oss-harmony/v1": "gpt_oss",
        }[self.profile]
        if self.channels.mode != expected or (expected == "gpt_oss" and self.stop):
            raise ValueError("generation_profile_mismatch")
        return self


class GenerationUsage(GenerationData):
    input_tokens: int | None = Field(ge=0)
    generated_tokens: int | None = Field(ge=0)
    reasoning_tokens: int | None = Field(ge=0)
    final_tokens: int | None = Field(ge=0)

    @model_validator(mode="after")
    def _partition(self) -> Self:
        if self.generated_tokens is not None and any(
            value is not None and value > self.generated_tokens
            for value in (self.reasoning_tokens, self.final_tokens)
        ):
            raise ValueError("generation_usage_invalid")
        if all(
            value is not None
            for value in (self.generated_tokens, self.reasoning_tokens, self.final_tokens)
        ) and self.generated_tokens != cast("int", self.reasoning_tokens) + cast(
            "int", self.final_tokens
        ):
            raise ValueError("generation_usage_invalid")
        return self


class GenerationResponse(GenerationData):
    """Preserved protocol channels; validity and completion state remain separate facts."""

    response_id_present: bool
    response_id: str | None = Field(max_length=512)
    system_fingerprint_present: bool
    system_fingerprint: str | None = Field(max_length=512)
    status: Literal[
        "completed", "incomplete_length", "provider_refusal", "invalid_json", "invalid_schema"
    ]
    finish_reason: Literal["stop", "length", "content_filter"]
    final_present: bool
    final_text: str | None = Field(max_length=MAX_GENERATION_CHANNEL_BYTES)
    reasoning_present: bool
    reasoning_text: str | None = Field(max_length=MAX_GENERATION_CHANNEL_BYTES)
    refusal_present: bool
    refusal_text: str | None = Field(max_length=MAX_GENERATION_CHANNEL_BYTES)
    json_status: Literal["not_requested", "valid", "invalid"]
    schema_status: Literal["not_requested", "not_evaluated", "valid", "invalid"]
    parsed_json: JsonValue
    usage: GenerationUsage

    @model_validator(mode="after")
    def _channels(self) -> Self:
        if any(
            not present and text is not None
            for present, text in (
                (self.final_present, self.final_text),
                (self.reasoning_present, self.reasoning_text),
                (self.refusal_present, self.refusal_text),
                (self.response_id_present, self.response_id),
                (self.system_fingerprint_present, self.system_fingerprint),
            )
        ):
            raise ValueError("generation_channels_invalid")
        for metadata in (self.response_id, self.system_fingerprint):
            if metadata is not None and (
                not scalar_text(metadata)
                or len(metadata.encode("utf-8")) > 512
                or any(ord(char) < 32 or ord(char) == 127 for char in metadata)
            ):
                raise ValueError("generation_response_invalid")
        canonical_json(self.parsed_json)
        texts = [
            value
            for value in (self.final_text, self.reasoning_text, self.refusal_text)
            if value is not None
        ]
        if (
            any(not scalar_text(value) for value in texts)
            or sum(len(value.encode("utf-8")) for value in texts) > MAX_GENERATION_CHANNEL_BYTES
        ):
            raise ValueError("generation_channels_invalid")
        if self.status == "completed" and (
            self.finish_reason != "stop"
            or self.final_text is None
            or self.refusal_text is not None
            or self.json_status == "invalid"
            or self.schema_status == "invalid"
        ):
            raise ValueError("generation_response_invalid")
        if (
            (self.status == "incomplete_length" and self.finish_reason != "length")
            or (
                self.status == "provider_refusal"
                and self.refusal_text is None
                and self.finish_reason != "content_filter"
            )
            or (
                self.status == "invalid_json"
                and (self.finish_reason != "stop" or self.json_status != "invalid")
            )
            or (
                self.status == "invalid_schema"
                and (
                    self.finish_reason != "stop"
                    or self.json_status != "valid"
                    or self.schema_status != "invalid"
                )
            )
            or (
                self.json_status == "not_requested"
                and (self.schema_status != "not_requested" or self.parsed_json is not None)
            )
            or (
                self.json_status == "invalid"
                and (self.schema_status != "not_evaluated" or self.parsed_json is not None)
            )
            or (self.json_status == "valid" and (self.schema_status not in ("valid", "invalid")))
        ):
            raise ValueError("generation_response_invalid")
        if self.schema_status == "valid" and type(self.parsed_json) is not dict:
            raise ValueError("generation_response_invalid")
        return self


def revalidate_request(value: GenerationRequest) -> GenerationRequest:
    """Freeze nested caller data before capability checks or transport awaits."""
    failed = False
    result: GenerationRequest | None = None
    try:
        if type(value) is GenerationRequest:
            checked = GenerationRequest.model_validate(value)
            copied = canonical_json(checked.wire_payload())
            result = GenerationRequest.model_validate_json(copied)
    except ValueError, GenerationError:
        failed = True
    if failed or result is None:
        raise GenerationError("generation_request_invalid")
    return result


def request_body(
    request: GenerationRequest, *, user_text: str, served_model: str
) -> dict[str, JsonValue]:
    """Translate fixed controls without granting an endpoint or native profile capability."""
    current = revalidate_request(request)
    if (
        not scalar_text(user_text)
        or len(user_text) > 200_000
        or not scalar_text(served_model)
        or not served_model
        or len(served_model) > 512
    ):
        raise GenerationError("generation_request_invalid")
    body: dict[str, JsonValue] = {
        "model": served_model,
        "messages": [{"role": "user", "content": user_text}],
        "temperature": current.temperature,
        "top_p": current.top_p,
        "top_k": current.top_k,
        "min_p": current.min_p,
        "max_completion_tokens": current.max_tokens,
        "stop": list(current.stop),
        "n": 1,
        "stream": False,
    }
    if current.effective_seed_decimal is not None:
        body["seed"] = int(current.effective_seed_decimal)
    if isinstance(current.channels, QwenChannels):
        body["chat_template_kwargs"] = {"enable_thinking": current.channels.enable_thinking}
    elif isinstance(current.channels, HarmonyChannels):
        body["reasoning_effort"] = current.channels.effort
    if isinstance(current.output_format, JsonSchemaOutput):
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "strata_output",
                "strict": True,
                "schema": current.output_format.schema_body,
            },
        }
    return body


def _protocol_object(value: JsonValue) -> dict[str, JsonValue]:
    if type(value) is not dict:
        raise GenerationError("generation_response_invalid")
    return value


def _nullable_text(value: JsonValue) -> str | None:
    if value is not None and type(value) is not str:
        raise GenerationError("generation_response_invalid")
    return value


def _usage(value: JsonValue) -> GenerationUsage:
    if value is None:
        return GenerationUsage(
            input_tokens=None, generated_tokens=None, reasoning_tokens=None, final_tokens=None
        )
    raw = _protocol_object(value)
    if set(raw) - {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
    }:
        raise GenerationError("generation_response_invalid")

    def count(field: str) -> int | None:
        result = raw.get(field)
        if result is not None and (type(result) is not int or result < 0):
            raise GenerationError("generation_response_invalid")
        return result

    generated, prompt = count("completion_tokens"), count("prompt_tokens")
    total = count("total_tokens")
    if (
        total is not None
        and generated is not None
        and prompt is not None
        and total != generated + prompt
    ):
        raise GenerationError("generation_response_invalid")
    detail = raw.get("completion_tokens_details")
    reasoning: int | None = None
    if detail is not None:
        details = _protocol_object(detail)
        if set(details) - {
            "reasoning_tokens",
            "accepted_prediction_tokens",
            "rejected_prediction_tokens",
            "audio_tokens",
        } or any(
            item is not None and (type(item) is not int or item < 0) for item in details.values()
        ):
            raise GenerationError("generation_response_invalid")
        reasoning_value = details.get("reasoning_tokens")
        if reasoning_value is not None and (
            type(reasoning_value) is not int or reasoning_value < 0
        ):
            raise GenerationError("generation_response_invalid")
        reasoning = reasoning_value
    final = generated - reasoning if generated is not None and reasoning is not None else None
    return GenerationUsage(
        input_tokens=prompt,
        generated_tokens=generated,
        reasoning_tokens=reasoning,
        final_tokens=final,
    )


def _response(
    request: GenerationRequest, raw: dict[str, JsonValue], served_model: str
) -> GenerationResponse:
    top_fields = {
        "id",
        "object",
        "created",
        "model",
        "choices",
        "usage",
        "system_fingerprint",
        "service_tier",
        "prompt_logprobs",
        "prompt_token_ids",
        "prompt_text",
        "kv_transfer_params",
    }
    if (
        set(raw) - top_fields
        or raw.get("object") != "chat.completion"
        or raw.get("model") != served_model
    ):
        raise GenerationError("generation_response_invalid")
    if "created" in raw and (type(raw["created"]) is not int or raw["created"] < 0):
        raise GenerationError("generation_response_invalid")
    if any(
        raw.get(field) is not None
        for field in ("prompt_logprobs", "prompt_token_ids", "prompt_text", "kv_transfer_params")
    ):
        raise GenerationError("generation_response_invalid")
    choices = raw.get("choices")
    if type(choices) is not list or len(choices) != 1:
        raise GenerationError("generation_response_invalid")
    choice = _protocol_object(choices[0])
    if (
        set(choice)
        - {
            "index",
            "message",
            "logprobs",
            "finish_reason",
            "stop_reason",
            "token_ids",
            "routed_experts",
        }
        or type(choice.get("index")) is not int
        or choice["index"] != 0
        or any(
            choice.get(field) is not None for field in ("logprobs", "token_ids", "routed_experts")
        )
    ):
        raise GenerationError("generation_response_invalid")
    finish = choice.get("finish_reason")
    if finish not in ("stop", "length", "content_filter"):
        raise GenerationError("generation_response_invalid")
    message = _protocol_object(choice.get("message"))
    if (
        set(message)
        - {
            "role",
            "content",
            "reasoning",
            "refusal",
            "annotations",
            "audio",
            "function_call",
            "tool_calls",
        }
        or message.get("role") != "assistant"
        or any(
            message.get(field) is not None for field in ("annotations", "audio", "function_call")
        )
        or message.get("tool_calls") not in (None, [])
    ):
        raise GenerationError("generation_response_invalid")
    final = _nullable_text(message.get("content"))
    reasoning = _nullable_text(message.get("reasoning"))
    refusal = _nullable_text(message.get("refusal"))
    if isinstance(request.channels, PlainChannels) and reasoning is not None:
        raise GenerationError("generation_response_invalid")
    if not isinstance(request.channels, PlainChannels) and "reasoning" not in message:
        raise GenerationError("generation_response_invalid")
    if (
        isinstance(request.channels, QwenChannels)
        and not request.channels.enable_thinking
        and reasoning not in (None, "")
    ):
        raise GenerationError("generation_response_invalid")
    if finish == "stop" and final is None and refusal is None:
        raise GenerationError("generation_response_invalid")
    json_status: Literal["not_requested", "valid", "invalid"] = "not_requested"
    schema_status: Literal["not_requested", "not_evaluated", "valid", "invalid"] = "not_requested"
    parsed: JsonValue = None
    if isinstance(request.output_format, JsonSchemaOutput):
        json_status, schema_status = "invalid", "not_evaluated"
        if final is not None:
            failed = False
            try:
                parsed = parse_json(final, max_bytes=MAX_GENERATION_CHANNEL_BYTES)
            except GenerationError:
                failed = True
            if not failed:
                json_status = "valid"
                schema_status = (
                    "valid" if conforms(request.output_format.schema_body, parsed) else "invalid"
                )
    status: Literal[
        "completed", "incomplete_length", "provider_refusal", "invalid_json", "invalid_schema"
    ]
    if refusal is not None or finish == "content_filter":
        status = "provider_refusal"
    elif finish == "length":
        status = "incomplete_length"
    elif json_status == "invalid":
        status = "invalid_json"
    elif schema_status == "invalid":
        status = "invalid_schema"
    else:
        status = "completed"
    return GenerationResponse(
        response_id_present="id" in raw,
        response_id=_nullable_text(raw.get("id")),
        system_fingerprint_present="system_fingerprint" in raw,
        system_fingerprint=_nullable_text(raw.get("system_fingerprint")),
        status=status,
        finish_reason=finish,
        final_present="content" in message,
        final_text=final,
        reasoning_present="reasoning" in message,
        reasoning_text=reasoning,
        refusal_present="refusal" in message,
        refusal_text=refusal,
        json_status=json_status,
        schema_status=schema_status,
        parsed_json=parsed,
        usage=_usage(raw.get("usage")),
    )


def parse_generation_response(
    request: GenerationRequest, raw: bytes, *, served_model: str
) -> GenerationResponse:
    """Read exact protocol fields; no tag splitting, first-object extraction or repair."""
    failed = False
    result: GenerationResponse | None = None
    try:
        current = revalidate_request(request)
        result = _response(
            current,
            _protocol_object(parse_json(raw, max_bytes=MAX_GENERATION_RESPONSE_BYTES)),
            served_model,
        )
    except ValueError, GenerationError:
        failed = True
    if failed or result is None:
        raise GenerationError("generation_response_invalid")
    return result
