"""Inert verifier rendering metadata shared by artifact writers and scoring workers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["RenderedVerifierInput", "VerifierConfig", "render_verifier_input"]

_TOKEN = re.compile(r"\{(premises|hypothesis)\}")
MAX_RENDERED_CHARS = 200_000


class RenderedVerifierInput(BaseModel):
    """Rendered input or a content-free refusal; never a clipped hypothesis."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    text: str | None = None
    text_pair: str | None = None
    error_code: (
        Literal["input_shape_invalid", "premise_limit_exceeded", "rendered_input_too_long"] | None
    ) = None


class VerifierConfig(BaseModel):
    """Versioned metadata for the bounded premise/hypothesis renderer."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)
    schema_version: Literal[1] = 1
    labels: tuple[str, ...] = Field(min_length=2, max_length=16)
    template: str = Field(min_length=1, max_length=MAX_RENDERED_CHARS)
    pair_template: str | None = Field(default=None, max_length=MAX_RENDERED_CHARS)
    premise_format: Literal["numbered_lines", "json"]
    max_length: int = Field(strict=True, ge=1, le=8192)
    max_premises: int = Field(strict=True, ge=1, le=20)
    label_tokens: dict[str, str] | None = None
    output_capabilities: tuple[
        Literal["classification", "inconsistency", "support", "abstention"], ...
    ] = ("classification",)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("verifier_schema_version_invalid")
        return value

    @model_validator(mode="after")
    def _validate(self) -> VerifierConfig:
        if len(set(self.labels)) != len(self.labels) or any(
            not s.strip() or len(s) > 64 for s in self.labels
        ):
            raise ValueError("verifier_labels_invalid")
        if "classification" not in self.output_capabilities or len(
            set(self.output_capabilities)
        ) != len(self.output_capabilities):
            raise ValueError("verifier_capabilities_invalid")
        placeholders: set[str] = set()
        for template in (self.template, self.pair_template or ""):
            placeholders.update(_TOKEN.findall(template))
            remainder = _TOKEN.sub("", template)
            if "{" in remainder or "}" in remainder:
                raise ValueError("verifier_template_invalid")
        if placeholders != {"premises", "hypothesis"}:
            raise ValueError("verifier_placeholders_missing")
        if self.label_tokens is not None and (
            set(self.label_tokens) != set(self.labels)
            or any(not v or len(v) > 64 for v in self.label_tokens.values())
        ):
            raise ValueError("verifier_verbalizers_invalid")
        return self

    def digest(self) -> str:
        """Hash the complete canonical metadata, excluding no effective fields."""
        raw = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def render_verifier_input(
    config: VerifierConfig, premises: object, hypothesis: object
) -> RenderedVerifierInput:
    """Render the supported list-of-text premise/hypothesis shape with declared limits."""
    if not isinstance(premises, list | tuple) or not isinstance(hypothesis, str):
        return RenderedVerifierInput(error_code="input_shape_invalid")
    values_untyped = cast("Sequence[object]", premises)
    if any(not isinstance(p, str) for p in values_untyped):
        return RenderedVerifierInput(error_code="input_shape_invalid")
    texts = cast("Sequence[str]", values_untyped)
    if len(texts) > config.max_premises:
        return RenderedVerifierInput(error_code="premise_limit_exceeded")
    if len(hypothesis) + sum(len(p) for p in texts) > MAX_RENDERED_CHARS:
        return RenderedVerifierInput(error_code="rendered_input_too_long")
    premise_text = (
        "\n".join(f"{index}. {text}" for index, text in enumerate(texts, 1))
        if config.premise_format == "numbered_lines"
        else json.dumps(texts, ensure_ascii=False, separators=(",", ":"))
    )
    values = {"premises": premise_text, "hypothesis": hypothesis}
    templates = (config.template, config.pair_template or "")
    rendered_length = sum(len(template) for template in templates)
    rendered_length += sum(
        len(values[match.group(1)]) - len(match.group())
        for template in templates
        for match in _TOKEN.finditer(template)
    )
    if rendered_length > MAX_RENDERED_CHARS:
        return RenderedVerifierInput(error_code="rendered_input_too_long")
    text = _TOKEN.sub(lambda match: values[match.group(1)], config.template)
    pair = (
        _TOKEN.sub(lambda match: values[match.group(1)], config.pair_template)
        if config.pair_template is not None
        else None
    )
    if len(text) + len(pair or "") > MAX_RENDERED_CHARS:
        return RenderedVerifierInput(error_code="rendered_input_too_long")
    return RenderedVerifierInput(text=text, text_pair=pair)
