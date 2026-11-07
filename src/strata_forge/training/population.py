"""Strict local population-v2 plans and private receipts, without dispatch or read grants."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.core.errors import ValidationError
from strata_forge.training.classifier import (
    ClassifierConfig,  # noqa: TC001 — runtime Pydantic field type
)
from strata_forge.training.verifier_geometry import (
    VerifierGeometryConfig,  # noqa: TC001 — runtime Pydantic field type
)

if TYPE_CHECKING:
    from pathlib import Path

MAX_POPULATION_ROWS = 600_000
MAX_POPULATION_ROW_BYTES = 2 * 1024 * 1024
MAX_POPULATION_SOURCE_BYTES = 8 * 1024**3
MAX_POPULATION_DISK_BYTES = 64 * 1024**3
POPULATION_READ_BYTES = 65_536
POPULATION_BATCH_ROWS = 32
POPULATION_BATCH_BYTES = 8 * 1024**2
POPULATION_DISK_RESERVE = 8 * 1024**2


class PopulationError(ValidationError):
    """A fixed refusal; private receipts are read separately, never attached to this message."""

    def __init__(self, code: str = "population_invalid") -> None:
        codes = {
            "population_invalid",
            "population_source_invalid",
            "population_limit",
            "population_overflow",
            "population_authority_denied",
            "population_attempt_unavailable",
            "population_cache_invalid",
            "population_tokenizer_invalid",
            "population_training_unsupported",
            "population_training_failed",
            "population_cancelled",
        }
        super().__init__(code if code in codes else "population_invalid")


class _Data(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )


class PopulationIdentity(_Data):
    """Declared immutable input identity, not proof that a Python object came from that ref."""

    repo_type: Literal["model", "dataset"]
    ref: str = Field(min_length=43, max_length=256)

    @field_validator("ref")
    @classmethod
    def _ref(cls, value: str) -> str:
        repo, separator, sha = value.rpartition("@")
        if (
            separator != "@"
            or re.fullmatch(r"[a-f0-9]{40}", sha) is None
            or len(repo) > 215
            or len(repo.split("/")) != 2
            or ".." in repo
            or "--" in repo
            or repo.endswith(".git")
            or any(
                re.fullmatch(r"[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?", part) is None
                for part in repo.split("/")
            )
        ):
            raise ValueError("population_identity_invalid")
        return value


class PopulationSourceIdentity(PopulationIdentity):
    repo_type: Literal["model", "dataset"] = "dataset"

    @field_validator("repo_type")
    @classmethod
    def _dataset(cls, value: str) -> str:
        if value != "dataset":
            raise ValueError("population_source_invalid")
        return value

    config: str = Field(min_length=1, max_length=120)
    split: str = Field(min_length=1, max_length=120)
    expected_rows: int = Field(ge=1, le=MAX_POPULATION_ROWS)

    @field_validator("config", "split")
    @classmethod
    def _name(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
            raise ValueError("population_source_invalid")
        return value


class CompletePopulation(_Data):
    mode: Literal["complete"]


class PopulationRange(_Data):
    mode: Literal["range"]
    start: int = Field(ge=0)
    end: int = Field(ge=1, le=MAX_POPULATION_ROWS)


class PopulationLimits(_Data):
    """Explicit admission ceilings; no disk space is preallocated by a limit value."""

    serialized_bytes: int = Field(ge=1, le=MAX_POPULATION_SOURCE_BYTES)
    disk_bytes: int = Field(ge=2 * POPULATION_DISK_RESERVE, le=MAX_POPULATION_DISK_BYTES)


class PopulationOptimizerSettings(_Data):
    """The supported effective native Trainer optimizer profile, with no free passthrough."""

    optimizer: Literal["adamw_torch_fused"] = "adamw_torch_fused"
    learning_rate: float = Field(default=5e-5, gt=0, le=1, allow_inf_nan=False)
    scheduler: Literal["linear", "constant", "cosine"] = "linear"
    warmup_steps: int = Field(default=0, ge=0, le=1_000_000)


class PrimaryPopulationTask(_Data):
    codec: Literal["primary_classifier_v1"]
    config: ClassifierConfig

    @model_validator(mode="after")
    def _supported(self) -> PrimaryPopulationTask:
        if (
            self.config.overflow != "error"
            or self.config.epochs != 1
            or self.config.batch_size != 32
            or self.config.max_steps != -1
            or self.config.precision != "fp32"
            or self.config.gradient_checkpointing
            or self.config.checkpoint_selection != "final"
            or self.config.selection_split is not None
        ):
            raise ValueError("population_training_unsupported")
        return self


class AuxiliaryPopulationTask(_Data):
    codec: Literal["auxiliary_verifier_v1"]
    geometry: VerifierGeometryConfig
    class_order: tuple[Literal["E", "C", "U"], Literal["E", "C", "U"], Literal["E", "C", "U"]]
    optimizer: PopulationOptimizerSettings = Field(default_factory=PopulationOptimizerSettings)

    @model_validator(mode="after")
    def _classes(self) -> AuxiliaryPopulationTask:
        if len(self.geometry.labels) != 3 or set(self.class_order) != {"E", "C", "U"}:
            raise ValueError("population_class_order_invalid")
        return self


class TrainingPopulationPlanV2(_Data):
    """An inert local plan; no authority object, filesystem path or executable can be serialized."""

    schema_version: Literal[2]
    profile: Literal["complete_classifier_population_v2"] = "complete_classifier_population_v2"
    source: PopulationSourceIdentity
    model: PopulationIdentity
    tokenizer: PopulationIdentity
    family: Literal["bert", "gpt2", "modernbert-decoder"]
    selection: Annotated[CompletePopulation | PopulationRange, Field(discriminator="mode")]
    task: Annotated[PrimaryPopulationTask | AuxiliaryPopulationTask, Field(discriminator="codec")]
    limits: PopulationLimits
    seed: int = Field(ge=0, le=2**32 - 1)
    overflow: Literal["error"]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 2:
            raise ValueError("population_version_invalid")
        return value

    @model_validator(mode="after")
    def _contract(self) -> TrainingPopulationPlanV2:
        if self.model.repo_type != "model" or self.tokenizer.repo_type != "model":
            raise ValueError("population_model_identity_invalid")
        if (
            isinstance(self.selection, PopulationRange)
            and not 0 <= self.selection.start < self.selection.end <= self.source.expected_rows
        ):
            raise ValueError("population_range_invalid")
        if isinstance(self.task, PrimaryPopulationTask):
            if self.task.config.seed != self.seed or self.task.config.warmup_steps > self.steps:
                raise ValueError("population_training_unsupported")
        elif (self.task.geometry.pooling == "cls") != (
            self.family == "bert"
        ) or self.task.optimizer.warmup_steps > self.steps:
            raise ValueError("population_training_unsupported")
        if len(self.canonical()) > 65_536:
            raise ValueError("population_plan_limit")
        return self

    @property
    def selected_rows(self) -> int:
        return (
            self.selection.end - self.selection.start
            if isinstance(self.selection, PopulationRange)
            else self.source.expected_rows
        )

    @property
    def source_start(self) -> int:
        return self.selection.start if isinstance(self.selection, PopulationRange) else 0

    @property
    def steps(self) -> int:
        return (self.selected_rows + 31) // 32

    @property
    def max_length(self) -> int:
        return (
            self.task.config.max_length
            if isinstance(self.task, PrimaryPopulationTask)
            else self.task.geometry.max_length
        )

    def canonical(self) -> bytes:
        """Serialize every declared effective field under a separate versioned profile."""
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical()).hexdigest()


@dataclass(frozen=True, repr=False)
class ResolvedPopulationTokenizer:
    """Trusted resolver metadata and native object; neither field attests external origin/custody."""

    identity: PopulationIdentity
    tokenizer: object


class TrustedPopulationSource(Protocol):
    """A trusted Python bounded row-framed stream; not a serialized builder or remote adapter."""

    identity: PopulationSourceIdentity

    async def read(self, maximum: int) -> bytes: ...
    async def aclose(self) -> None: ...


class PopulationAuthority(Protocol):
    """Trusted caller policy seam; return is not a credential or custody attestation."""

    async def check(
        self,
        plan: TrainingPopulationPlanV2,
        *,
        phase: Literal["before_read", "after_read", "before_train", "after_train"],
    ) -> None: ...


class PopulationPreparationReceipt(_Data):
    version: Literal[2] = 2
    status: Literal["prepared_complete", "overflow", "invalid", "cancelled"]
    plan_sha256: str
    tokenizer_sha256: str
    scan_complete: bool
    expected_source_rows: int
    source_rows_observed: int
    selected_rows_observed: int
    source_overflow_rows: int
    selected_overflow_rows: int
    first_overflow_item_id: str | None
    serialized_bytes: int
    source_stream_sha256: str | None
    canonical_source_sha256: str | None
    selected_population_sha256: str | None
    cache_sha256: str | None
    cache_bytes: int
    disk_high_water_bytes: int
    max_buffered_rows: int
    max_buffered_bytes: int
    authority: Literal["trusted_caller_seam_only"] = "trusted_caller_seam_only"


@dataclass(frozen=True, repr=False, eq=False)
class PreparedTrainingPopulation:
    """Issued in-process cache handle; it cannot be adopted by deserializing a path or receipt."""

    directory: Path
    plan: TrainingPopulationPlanV2
    receipt: PopulationPreparationReceipt
    receipt_sha256: str


class PopulationEpochReceipt(_Data):
    version: Literal[2] = 2
    status: Literal["complete", "failed", "cancelled", "authority_denied"]
    plan_sha256: str
    preparation_sha256: str
    selected_rows: int
    confirmed_applied_rows: int
    confirmed_optimizer_steps: int
    expected_optimizer_steps: int
    order_algorithm: Literal["torch_cpu_randperm_v1"] = "torch_cpu_randperm_v1"
    planned_order_sha256: str
    applied_order_sha256: str
    optimizer_in_doubt: bool
    optimizer_profile: dict[str, str | float | int]
    native_versions: dict[str, str]
    cache_unchanged: bool
