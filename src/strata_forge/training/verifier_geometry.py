"""Exact PID rendering and sequence-aware token geometry, without native model imports."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.core.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

_MAX_CHARS = 200_000
_TOKEN = re.compile(r"\{(premises|hypothesis)\}")


class VerifierGeometryError(ValidationError):
    """A fixed content-free verifier geometry or artifact refusal."""

    def __init__(self, code: str = "verifier_geometry_invalid") -> None:
        allowed = {
            "verifier_geometry_invalid",
            "verifier_markers_invalid",
            "verifier_overflow",
            "verifier_tokenizer_unsupported",
            "verifier_model_unsupported",
            "verifier_artifact_invalid",
            "verifier_initialization_invalid",
            "verifier_batch_invalid",
            "verifier_output_invalid",
        }
        super().__init__(code if code in allowed else "verifier_geometry_invalid")


class _Data(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )


class VerifierPremise(_Data):
    """One literal premise with its retained row-local identifier."""

    pid: str = Field(min_length=2, max_length=64, pattern=r"^p[1-9][0-9]*$")
    text: str = Field(max_length=_MAX_CHARS)

    @model_validator(mode="after")
    def _exact(self) -> VerifierPremise:
        if re.fullmatch(r"p[1-9][0-9]*", self.pid) is None:
            raise ValueError("verifier_pid_invalid")
        self.text.encode("utf-8")
        return self


class VerifierExample(_Data):
    """An explicit input; optional support targets validate IDs but never enter model text."""

    item_id: str = Field(min_length=1, max_length=1024)
    premises: tuple[VerifierPremise, ...] = Field(max_length=20)
    hypothesis: str = Field(max_length=_MAX_CHARS)
    support_set: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _exact(self) -> VerifierExample:
        ids = tuple(p.pid for p in self.premises)
        if len(set(ids)) != len(ids) or not self.item_id.strip():
            raise ValueError("verifier_ids_invalid")
        if self.support_set is not None and (
            not ids
            or len(set(self.support_set)) != len(self.support_set)
            or not set(self.support_set) <= set(ids)
        ):
            raise ValueError("verifier_support_ids_invalid")
        self.item_id.encode("utf-8")
        self.hypothesis.encode("utf-8")
        return self


class VerifierGeometryConfig(_Data):
    """A distinct rendering ABI; existing version-one verifier metadata is unchanged."""

    schema_version: Literal[2] = 2
    labels: tuple[str, ...] = Field(min_length=2, max_length=16)
    premise_format: Literal["ids"] = "ids"
    template: str = Field(min_length=1, max_length=_MAX_CHARS)
    pair_template: str | None = Field(default=None, max_length=_MAX_CHARS)
    max_length: int = Field(ge=1, le=8192)
    max_premises: int = Field(default=20, ge=1, le=20)
    overflow: Literal["error"]
    pooling: Literal["cls", "answer_position"]
    pid_marker: Literal["first_token_overlapping_pid_start"] = "first_token_overlapping_pid_start"
    outputs: tuple[Literal["label", "inconsistent", "support_set"], ...]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 2:
            raise ValueError("verifier_geometry_invalid")
        return value

    @model_validator(mode="after")
    def _exact(self) -> VerifierGeometryConfig:
        if type(self.schema_version) is not int or len(set(self.labels)) != len(self.labels):
            raise ValueError("verifier_geometry_invalid")
        if any(not label.strip() or len(label) > 64 for label in self.labels):
            raise ValueError("verifier_labels_invalid")
        if (
            not self.outputs
            or self.outputs[0] != "label"
            or len(set(self.outputs)) != len(self.outputs)
        ):
            raise ValueError("verifier_outputs_invalid")
        if self.pooling == "cls":
            if self.template != "{premises}" or self.pair_template != "{hypothesis}":
                raise ValueError("verifier_encoder_rendering_invalid")
        elif (
            self.pair_template is not None
            or self.template.count("{premises}") != 1
            or self.template.count("{hypothesis}") != 1
            or self.template.index("{premises}") > self.template.index("{hypothesis}")
            or "{" in _TOKEN.sub("", self.template)
            or "}" in _TOKEN.sub("", self.template)
        ):
            raise ValueError("verifier_decoder_rendering_invalid")
        for label in self.labels:
            label.encode("utf-8")
        self.template.encode("utf-8")
        return self

    def digest(self) -> str:
        """Hash every declared field without changing the existing rendering digest ABI."""
        raw = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, repr=False)
class RenderedVerifierGeometry:
    """Literal model inputs and exact character positions before native tokenization."""

    text: str
    text_pair: str | None
    premise_ids: tuple[str, ...]
    pid_starts: tuple[int, ...]
    premise_span: tuple[int, int]


def _render_verifier_geometry(
    config: VerifierGeometryConfig, example: VerifierExample
) -> RenderedVerifierGeometry:
    """Render explicit IDs once; neither premise text nor hypothesis is parsed as a template."""
    if len(example.premises) > config.max_premises:
        raise VerifierGeometryError()
    if example.support_set is not None and "support_set" not in config.outputs:
        raise VerifierGeometryError()
    starts: list[int] = []
    lines: list[str] = []
    length = 0
    for premise in example.premises:
        starts.append(length)
        line = premise.pid + ": " + premise.text
        lines.append(line)
        length += len(line) + 1
    premises = "\n".join(lines)
    if len(premises) + len(example.hypothesis) > _MAX_CHARS:
        raise VerifierGeometryError()
    if config.pooling == "cls":
        text, pair, start = premises, example.hypothesis, 0
    else:
        start = config.template.index("{premises}")
        text = _TOKEN.sub(
            lambda m: premises if m.group(1) == "premises" else example.hypothesis, config.template
        )
        pair = None
    if len(text) + len(pair or "") > _MAX_CHARS:
        raise VerifierGeometryError()
    return RenderedVerifierGeometry(
        text,
        pair,
        tuple(p.pid for p in example.premises),
        tuple(start + pos for pos in starts),
        (start, start + len(premises)),
    )


def render_verifier_geometry(
    config: VerifierGeometryConfig, example: VerifierExample
) -> RenderedVerifierGeometry:
    """Validate and render explicit inputs without persisting rejected caller values."""
    result = None
    try:
        config = VerifierGeometryConfig.model_validate(config)
        example = VerifierExample.model_validate(example)
        result = _render_verifier_geometry(config, example)
    except Exception:
        result = None
    if result is None:
        raise VerifierGeometryError()
    return result


def _sequence(value: object) -> tuple[object, ...]:
    if type(value) not in (list, tuple):
        raise ValueError()
    return tuple(cast("Sequence[object]", value))


def resolve_pid_markers(
    rendered: RenderedVerifierGeometry,
    *,
    offsets: object,
    sequence_ids: object,
    special_tokens_mask: object,
    attention_mask: object,
) -> tuple[int, ...]:
    """Resolve the first attended nonspecial premise token overlapping each PID's first char."""
    result: tuple[int, ...] | None = None
    try:
        offsets = _sequence(offsets)
        sequence_ids = _sequence(sequence_ids)
        special_tokens_mask = _sequence(special_tokens_mask)
        attention_mask = _sequence(attention_mask)
        if not (
            len(offsets) == len(sequence_ids) == len(special_tokens_mask) == len(attention_mask)
        ):
            raise ValueError()
        candidates: list[tuple[int, int, int]] = []
        for index, (span, sequence, special, attended) in enumerate(
            zip(offsets, sequence_ids, special_tokens_mask, attention_mask, strict=True)
        ):
            span = _sequence(span)
            if len(span) != 2:
                raise ValueError()
            begin, end = span
            if type(begin) is not int or type(end) is not int:
                raise ValueError()
            if (
                type(special) is not int
                or special not in (0, 1)
                or type(attended) is not int
                or attended not in (0, 1)
            ):
                raise ValueError()
            if sequence is not None and (type(sequence) is not int or sequence not in (0, 1)):
                raise ValueError()
            if begin < 0 or end < begin:
                raise ValueError()
            if not attended or special:
                continue
            target = (
                rendered.text if sequence == 0 else rendered.text_pair if sequence == 1 else None
            )
            if target is None or end > len(target):
                raise ValueError()
            if sequence == 0:
                candidates.append((index, begin, end))
        markers: list[int] = []
        for start in rendered.pid_starts:
            found = next(
                ((index, begin, end) for index, begin, end in candidates if begin <= start < end),
                None,
            )
            if (
                found is None
                or found[1] < rendered.premise_span[0]
                or found[2] > rendered.premise_span[1]
                or found[0] in markers
            ):
                raise ValueError()
            markers.append(found[0])
        result = tuple(markers)
    except Exception:
        result = None
    if result is None:
        raise VerifierGeometryError("verifier_markers_invalid")
    return result
