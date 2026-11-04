"""Strict inert primary-classifier inputs; authority and source code never enter the spec."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Self, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from strata_forge.core.errors import ValidationError
from strata_forge.pipelines._hub_ids import validate_hub_ref
from strata_forge.training.classifier import ClassifierConfig
from strata_forge.training.verifier_config import (
    VerifierConfig,  # noqa: TC001 — Pydantic resolves this field at runtime
)

MAX_SPEC_BYTES = 32_768
MAX_TRAINING_ROWS = 100_000
# Prototype worker memory/work bounds, independent of scientific population requirements.
MAX_ROW_BYTES = 2 * 1024 * 1024
MAX_INPUT_BYTES = 64 * 1024 * 1024


class ClassifierWorkerError(ValidationError):
    """A fixed content-free worker refusal."""


def _tuple(value: object) -> object:
    return tuple(cast("list[object]", value)) if type(value) is list else value


type Items[T] = Annotated[tuple[T, ...], BeforeValidator(_tuple)]
type Ref = Annotated[str, Field(max_length=256), AfterValidator(validate_hub_ref)]
type Name = Annotated[str, Field(min_length=1, max_length=120)]


class _Data(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        hide_input_in_errors=True,
        revalidate_instances="always",
        protected_namespaces=(),
    )


class EncoderInitialization(_Data):
    kind: Literal["encoder_random_head"]


class DecoderInitialization(_Data):
    kind: Literal["decoder_logit_head"]
    label_tokens: dict[str, str] = Field(min_length=2, max_length=16)

    @model_validator(mode="after")
    def _strings(self) -> Self:
        if any(not token or len(token) > 1024 for token in self.label_tokens.values()):
            raise ValueError("classifier_verbalizers_invalid")
        return self


class ClassifierBase(_Data):
    ref: Ref
    initialization: Annotated[
        EncoderInitialization | DecoderInitialization, Field(discriminator="kind")
    ]


class ClassifierSelection(_Data):
    mode: Literal["all", "range"] = "all"
    row_start: int = Field(default=0, ge=0, le=MAX_TRAINING_ROWS)
    row_end: int | None = Field(default=None, ge=1, le=2 * MAX_TRAINING_ROWS)
    row_limit: int = Field(default=MAX_TRAINING_ROWS, ge=1, le=MAX_TRAINING_ROWS)

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.mode == "all" and (self.row_start != 0 or self.row_end is not None):
            raise ValueError("classifier_selection_invalid")
        if self.mode == "range" and (
            self.row_end is None or not 0 < self.row_end - self.row_start <= self.row_limit
        ):
            raise ValueError("classifier_selection_invalid")
        return self


class ClassifierDataset(_Data):
    ref: Ref
    config: Name
    split: Name
    format: Literal["classification_text", "nli_pair"]
    columns: dict[str, str] = Field(min_length=3, max_length=6)
    selection: ClassifierSelection = Field(default_factory=ClassifierSelection)

    @model_validator(mode="after")
    def _mapping(self) -> Self:
        required = (
            {"item_id", "text", "label"}
            if self.format == "classification_text"
            else {"item_id", "premises", "hypothesis", "label"}
        )
        optional = (
            {"label_dist", "text_pair"} if self.format == "classification_text" else {"label_dist"}
        )
        if (
            not required <= self.columns.keys()
            or not self.columns.keys() <= required | optional
            or len(set(self.columns.values())) != len(self.columns)
            or any(not name or len(name) > 255 for name in self.columns.values())
        ):
            raise ValueError("classifier_mapping_invalid")
        return self


class ClassifierTrainingSettings(_Data):
    labels: Items[str] = Field(min_length=2, max_length=16)
    objective_preset: Literal["cross_entropy", "research_ce_brier_0_1"] = "cross_entropy"
    label_aliases: dict[str, str] = Field(default_factory=dict)
    label_index_map: dict[str, str] = Field(default_factory=dict)
    max_length: int = Field(default=512, ge=1, le=8192)
    overflow: Literal["error", "drop"] = "error"
    epochs: float = Field(default=1.0, gt=0, le=100, allow_inf_nan=False)
    max_steps: int = Field(default=-1, ge=-1, le=1_000_000)
    learning_rate: float = Field(default=5e-5, gt=0, le=1, allow_inf_nan=False)
    batch_size: int = Field(default=8, ge=1, le=1024)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    precision: Literal["fp32", "bf16"] = "fp32"
    gradient_checkpointing: bool = False
    use_cpu: bool = False
    scheduler: Literal["linear", "constant", "cosine"] = "linear"
    warmup_steps: int = Field(default=0, ge=0, le=1_000_000)
    checkpoint_selection: Literal["final"] = "final"
    rendering: VerifierConfig | None = None

    def sdk_config(self) -> ClassifierConfig:
        values = self.model_dump(exclude={"objective_preset", "label_index_map"})
        values["label_index_map"] = {int(key): value for key, value in self.label_index_map.items()}
        if self.objective_preset == "research_ce_brier_0_1":
            return ClassifierConfig.research_ce_brier(**values)
        return ClassifierConfig(**values)

    @model_validator(mode="after")
    def _sdk(self) -> Self:
        if any(
            not key.isascii() or not key.isdecimal() or str(int(key)) != key or len(key) > 10
            for key in self.label_index_map
        ):
            raise ValueError("classifier_index_map_invalid")
        self.sdk_config()
        return self


class ClassifierTrainSpec(_Data):
    schema_version: Literal[1]
    engine_version: str = Field(min_length=1, max_length=128)
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    method: Literal["seq_cls"]
    task_type: Literal["SEQ_CLS"]
    base: ClassifierBase
    dataset: ClassifierDataset
    training: ClassifierTrainingSettings
    progress_path: Literal["progress.jsonl"]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("classifier_schema_version_invalid")
        return value

    @model_validator(mode="after")
    def _contract(self) -> Self:
        init, rendering = self.base.initialization, self.training.rendering
        if isinstance(init, DecoderInitialization) and set(init.label_tokens) != set(
            self.training.labels
        ):
            raise ValueError("classifier_verbalizer_labels_mismatch")
        if self.dataset.format == "nli_pair" and rendering is None:
            raise ValueError("classifier_rendering_required")
        if self.dataset.format != "nli_pair" and rendering is not None:
            raise ValueError("classifier_rendering_unexpected")
        if rendering is not None and rendering.max_length != self.training.max_length:
            raise ValueError("classifier_rendering_length_mismatch")
        if len(self.canonical()) > MAX_SPEC_BYTES:
            raise ValueError("classifier_spec_too_large")
        return self

    def canonical(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json", warnings=False),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical()).hexdigest()


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError("classifier_json_invalid")
        result[key] = value
    return result


def parse_json(raw: bytes, maximum: int) -> object:
    result: object = None
    failed = False
    try:
        if type(raw) is not bytes or len(raw) > maximum:
            raise ValueError("classifier_json_invalid")
        result = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except Exception:
        failed = True
    if failed:
        raise ClassifierWorkerError("classifier_json_invalid")
    return result


def load_classifier_spec(raw: bytes) -> ClassifierTrainSpec:
    result: ClassifierTrainSpec | None = None
    try:
        value = parse_json(raw, MAX_SPEC_BYTES)
        result = ClassifierTrainSpec.model_validate(value)
    except Exception:
        result = None
    if result is None:
        raise ClassifierWorkerError("classifier_spec_invalid")
    return result
