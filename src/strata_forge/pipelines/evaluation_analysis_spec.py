"""Closed local temperature-fit data; declarations never grant access or fit eligibility."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.core.errors import ValidationError
from strata_forge.metrics import (  # noqa: TC001 — nested Pydantic fields resolve at runtime
    MetricValue,
    TemperatureCalibration,
)
from strata_forge.metrics.temperature_fit import (  # noqa: TC001 — nested Pydantic fields resolve at runtime
    TemperatureFitBounds,
    TemperatureFitSolution,
)
from strata_forge.pipelines._hub_ids import validate_hub_ref

MAX_ANALYSIS_ROWS = 100_000
MAX_ANALYSIS_RECORD_BYTES = 16 * 1024
MAX_ANALYSIS_SOURCE_BYTES = 64 * 1024**2
MAX_ANALYSIS_WORK_BYTES = 256 * 1024**2
MAX_ANALYSIS_RECEIPT_BYTES = 64 * 1024

_CODES = frozenset(
    {
        "analysis_invalid",
        "analysis_unavailable",
        "analysis_denied",
        "analysis_limit",
        "analysis_input_invalid",
        "analysis_population_mismatch",
        "analysis_source_changed",
        "analysis_deadline",
        "analysis_failed",
    }
)


class EvaluationAnalysisError(ValidationError):
    """A fixed content-free refusal; partial temperatures are never returned."""

    def __init__(self, code: str = "analysis_invalid") -> None:
        super().__init__(code if code in _CODES else "analysis_invalid")


def _text(value: str) -> str:
    if not value or len(value) > 255 or len(value.encode("utf-8")) > 255:
        raise ValueError("analysis_invalid")
    return value


def _sha(value: str) -> str:
    if re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("analysis_invalid")
    return value


type Text = Annotated[str, AfterValidator(_text)]
type Sha = Annotated[str, AfterValidator(_sha)]


class _Data(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("analysis_invalid")
        return value

    def canonical(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical()).hexdigest()


class AnalysisRef(_Data):
    """An exact typed declared ref, not a resolver or custody decision."""

    repo_type: Literal["dataset", "model"]
    ref: str = Field(min_length=1, max_length=256)
    _ref = field_validator("ref")(validate_hub_ref)


class GoldIndex(_Data):
    """An explicit integer-to-label mapping; no column order is inferred."""

    index: int = Field(ge=-(2**53 - 1), le=2**53 - 1)
    label: Text


class TemperatureFitSourceMetadata(_Data):
    """The named source and raw-logit condition the trusted adapter must actually supply."""

    schema_version: Literal[1]
    dataset: AnalysisRef
    model: AnalysisRef
    tokenizer: AnalysisRef
    predictions: AnalysisRef | None
    config: Text | None
    split: Text
    id_column: Text
    gold_column: Text
    labels: tuple[Text, ...] = Field(min_length=2, max_length=16)
    gold_index_map: tuple[GoldIndex, ...] = Field(max_length=16)
    expected_rows: int = Field(ge=1, le=MAX_ANALYSIS_ROWS)
    records_sha256: Sha
    source_receipt_sha256: Sha
    rendering_sha256: Sha
    head_sha256: Sha
    input_kind: Literal["exact_class_logits", "exact_restricted_label_logits"]
    projection: Literal["identity"]
    selection: Literal["complete_named_split"]
    row_codec: Literal["strata.analysis.fit.rows.v1"]

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if (
            self.dataset.repo_type != "dataset"
            or self.model.repo_type != "model"
            or self.tokenizer.repo_type != "model"
            or (self.predictions is not None and self.predictions.repo_type != "dataset")
            or len(set(self.labels)) != len(self.labels)
            or any(
                not label.strip() or len(label) > 64 or len(label.encode("utf-8")) > 64
                for label in self.labels
            )
            or len({row.index for row in self.gold_index_map}) != len(self.gold_index_map)
            or any(row.label not in self.labels for row in self.gold_index_map)
        ):
            raise ValueError("analysis_invalid")
        return self

    def input_refs(self) -> tuple[AnalysisRef, ...]:
        values = (self.dataset, self.model, self.tokenizer) + (
            () if self.predictions is None else (self.predictions,)
        )
        return tuple({(value.repo_type, value.ref): value for value in values}.values())

    @property
    def probability_kind(self) -> Literal["class_probabilities", "restricted_softmax"]:
        return (
            "class_probabilities"
            if self.input_kind == "exact_class_logits"
            else "restricted_softmax"
        )


class TemperatureFitPlanV1(_Data):
    """An explicit fit request with no registry, source import, credential or output path."""

    schema_version: Literal[1]
    protocol: Literal["strata.eval.analysis.local.v1"]
    mode: Literal["temperature_fit"]
    source: TemperatureFitSourceMetadata
    registration_id: Text
    registration_revision: Text
    bounds: TemperatureFitBounds
    soft_budget_ms: int = Field(ge=1, le=3_600_000)


class TemperatureFitRow(_Data):
    """One complete exact raw-logit row; no implicit probability inversion or target repair."""

    item_id: str = Field(min_length=1, max_length=1024)
    gold: str | int
    logits: tuple[float | int, ...] = Field(min_length=2, max_length=16)

    @field_validator("item_id")
    @classmethod
    def _item_id(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 1024:
            raise ValueError("analysis_invalid")
        return value


class FitPolicyRequest(_Data):
    """A metadata-only request for a fresh installed policy decision."""

    schema_version: Literal[1]
    purpose: Literal["temperature_calibration_fit"]
    plan_sha256: Sha
    source: TemperatureFitSourceMetadata
    registration_id: Text
    registration_revision: Text


class CanonicalAnalysisRead(_Data):
    """A trusted typed resolution result; positive and unknown custody never permit reads."""

    requested: AnalysisRef
    canonical_ref: AnalysisRef
    custody: Literal["unsealed", "protected", "unknown"]


class FitRegistration(_Data):
    """Trusted registry metadata, not a field a serialized fit plan may use as authority."""

    registration_id: Text
    registration_revision: Text
    metadata_sha256: Sha
    canonical_dataset: AnalysisRef
    canonical_config: Text | None
    canonical_split: Text
    split_aliases: tuple[Text, ...] = Field(min_length=1, max_length=32)
    roles: tuple[
        Literal["calibration_fit", "monitoring", "selection", "held_out", "gate", "test"], ...
    ] = Field(min_length=1, max_length=6)
    role_coverage: Literal["known", "unknown"]
    expected_rows: int = Field(ge=1, le=MAX_ANALYSIS_ROWS)
    records_sha256: Sha


class RegisteredFitDecision(_Data):
    """Issued by trusted Python policy code, never accepted from a request or environment."""

    schema_version: Literal[1]
    request_sha256: Sha
    status: Literal["allow_unsealed", "deny", "unknown"]
    policy_revision: Text
    reads: tuple[CanonicalAnalysisRead, ...] = Field(max_length=4)
    registration: FitRegistration | None


class FitPointMetrics(_Data):
    """Existing clipped point NLL and multiclass-sum Brier with explicit denominators."""

    nll: MetricValue
    brier: MetricValue


def calibration_component_digest(component: TemperatureCalibration) -> str:
    """Hash component content without claiming persisted identity or authorization."""
    return hashlib.sha256(
        json.dumps(
            component.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class TemperatureFitReceipt(_Data):
    """A frozen in-memory fit receipt, without authority or persisted-artifact attestation."""

    schema_version: Literal[1]
    status: Literal["complete"]
    plan: TemperatureFitPlanV1
    plan_sha256: Sha
    source_sha256: Sha
    observed_rows: int = Field(ge=1, le=MAX_ANALYSIS_ROWS)
    observed_bytes: int = Field(ge=1, le=MAX_ANALYSIS_SOURCE_BYTES)
    records_sha256: Sha
    solution: TemperatureFitSolution
    component: TemperatureCalibration
    component_sha256: Sha
    before: FitPointMetrics
    after: FitPointMetrics
    policy_checks: int = Field(ge=1)
    elapsed_to_final_policy_ms: float = Field(ge=0, allow_inf_nan=False)
    decision_chain_sha256: Sha
    initial_decision_sha256: Sha
    final_decision_sha256: Sha
    initial_decision: RegisteredFitDecision
    final_decision: RegisteredFitDecision
    python_version: Text
    numpy_version: Text
    installed_source_evidence: Literal["unattested_local_sdk"] = "unattested_local_sdk"
    input_ownership: Literal["copied_owned_numeric_buffers"] = "copied_owned_numeric_buffers"

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        source = self.plan.source
        request = FitPolicyRequest(
            schema_version=1,
            purpose="temperature_calibration_fit",
            plan_sha256=self.plan_sha256,
            source=source,
            registration_id=self.plan.registration_id,
            registration_revision=self.plan.registration_revision,
        )
        if (
            self.initial_decision.digest() != self.initial_decision_sha256
            or self.final_decision.digest() != self.final_decision_sha256
            or self.initial_decision.request_sha256 != request.digest()
            or self.final_decision.request_sha256 != request.digest()
            or self.initial_decision.status != "allow_unsealed"
            or self.final_decision.status != "allow_unsealed"
            or self.initial_decision.registration != self.final_decision.registration
        ):
            raise ValueError("analysis_invalid")
        if (
            self.elapsed_to_final_policy_ms > self.plan.soft_budget_ms
            or self.plan.digest() != self.plan_sha256
            or source.digest() != self.source_sha256
            or self.observed_rows != source.expected_rows
            or self.records_sha256 != source.records_sha256
            or self.solution.n != self.observed_rows
            or self.solution.k != len(source.labels)
            or self.solution.bounds != self.plan.bounds
            or type(self.component.schema_version) is not int
            or self.component.schema_version != 1
            or self.component.method != "temperature"
            or self.component.temperature != self.solution.temperature
            or self.component.labels != source.labels
            or self.component.n != self.observed_rows
            or self.component.fit_ref != source.dataset.ref
            or self.component.config != source.config
            or self.component.split != source.split
            or self.component.probability_kind != source.probability_kind
            or self.component_sha256 != calibration_component_digest(self.component)
            or self.component.before is None
            or self.component.after is None
            or self.component.before.nll != self.before.nll.value
            or self.component.before.brier != self.before.brier.value
            or self.component.after.nll != self.after.nll.value
            or self.component.after.brier != self.after.brier.value
            or self.component.nll_clip_epsilon != self.before.nll.clip_epsilon
            or self.component.nll_clip_epsilon != self.after.nll.clip_epsilon
        ):
            raise ValueError("analysis_invalid")
        for metrics in (self.before, self.after):
            for metric in (metrics.nll, metrics.brier):
                if (
                    metric.n != self.observed_rows
                    or metric.excluded != 0
                    or metric.reason is not None
                    or metric.probability_kind != source.probability_kind
                ):
                    raise ValueError("analysis_invalid")
        return self
