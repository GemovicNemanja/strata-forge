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
    parsed_json: dict[str, JsonValue] | None
    usage: GenerationUsage

    @model_validator(mode="after")
    def _channels(self) -> Self:
        if any(
            not present and text is not None
            for present, text in (
                (self.final_present, self.final_text),
                (self.reasoning_present, self.reasoning_text),
                (self.refusal_present, self.refusal_text),
            )
        ):
            raise ValueError("generation_channels_invalid")
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
            or (
                self.json_status == "valid"
                and (self.parsed_json is None or self.schema_status not in ("valid", "invalid"))
            )
        ):
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
