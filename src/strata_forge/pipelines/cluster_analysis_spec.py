"""Closed local paired-cluster values; source declarations are never authority."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Literal, Self, cast

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.core.errors import ValidationError
from strata_forge.metrics.classification import MetricValue
from strata_forge.metrics.cluster_bootstrap import (
    BootstrapMetric,
    ClusterBootstrapResult,
    PairedClusterProfile,
    PercentileInterval,
)
from strata_forge.pipelines._hub_ids import validate_hub_ref
from strata_forge.pipelines._population import PopulationCounts

MAX_RECORD_BYTES = 16 * 1024
MAX_SOURCE_BYTES = 64 * 1024**2
MAX_METADATA_BYTES = 64 * 1024
_CODES = frozenset(
    {
        "cluster_analysis_invalid",
        "cluster_analysis_unavailable",
        "cluster_analysis_denied",
        "cluster_analysis_limit",
        "cluster_analysis_input_invalid",
        "cluster_analysis_population_mismatch",
        "cluster_analysis_source_changed",
        "cluster_analysis_deadline",
        "cluster_analysis_failed",
    }
)


class ClusterAnalysisError(ValidationError):
    def __init__(self, code: str = "cluster_analysis_invalid") -> None:
        super().__init__(
            code
            if type(code) is str and len(code) <= max(map(len, _CODES)) and code in _CODES
            else "cluster_analysis_invalid"
        )


def _text(value: str) -> str:
    if type(value) is not str or not value or len(value) > 255 or len(value.encode("utf-8")) > 255:
        raise ValueError("cluster_analysis_invalid")
    return value


def _sha(value: str) -> str:
    if type(value) is not str or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("cluster_analysis_invalid")
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
            raise ValueError("cluster_analysis_invalid")
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


class ClusterAnalysisRef(_Data):
    repo_type: Literal["model", "dataset"]
    ref: str = Field(min_length=1, max_length=256)
    _ref = field_validator("ref")(validate_hub_ref)


class ClusterGoldIndex(_Data):
    index: int = Field(ge=-(2**53 - 1), le=2**53 - 1)
    label: Text


class PairedClusterTarget(_Data):
    dataset: ClusterAnalysisRef
    config: Text | None
    split: Text
    id_column: Text
    gold_column: Text
    gold_index_map: tuple[ClusterGoldIndex, ...] = Field(max_length=16)
    cluster_column: Literal["base_problem_id"]
    labels: tuple[Text, ...] = Field(min_length=2, max_length=16)
    label_space: Text
    projection: Literal["identity"]
    selection: Literal["complete_metric_population"]
    slice_sha256: Sha
    target_identity_sha256: Sha
    selected_population_sha256: Sha
    metric_population_sha256: Sha
    expected_metric_rows: int = Field(ge=0, le=100_000)

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if (
            self.dataset.repo_type != "dataset"
            or len(set(self.labels)) != len(self.labels)
            or any(not label.strip() or len(label.encode()) > 64 for label in self.labels)
            or len({entry.index for entry in self.gold_index_map}) != len(self.gold_index_map)
            or any(entry.label not in self.labels for entry in self.gold_index_map)
        ):
            raise ValueError("cluster_analysis_invalid")
        return self


class PairedClusterCondition(_Data):
    condition_id: Text
    model: ClusterAnalysisRef
    tokenizer: ClusterAnalysisRef
    predictions: ClusterAnalysisRef | None
    source_receipt_sha256: Sha
    source_projection_sha256: Sha
    rendering_sha256: Sha
    head_sha256: Sha
    probability_kind: Literal["class_probabilities", "restricted_softmax"] | None
    calibration_state: Literal["raw", "declared_applied"]
    calibration_receipt_sha256: Sha | None
    records_sha256: Sha
    counts: PopulationCounts

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if (
            self.model.repo_type != "model"
            or self.tokenizer.repo_type != "model"
            or (self.predictions is not None and self.predictions.repo_type != "dataset")
            or (self.calibration_state == "raw") != (self.calibration_receipt_sha256 is None)
        ):
            raise ValueError("cluster_analysis_invalid")
        return self


class PairedClusterSourceMetadata(_Data):
    schema_version: Literal[1]
    target: PairedClusterTarget
    condition: PairedClusterCondition
    row_codec: Literal["strata.analysis.cluster.arm.rows.v1"]

    @model_validator(mode="after")
    def _count(self) -> Self:
        if self.condition.counts.rows_metric_population != self.target.expected_metric_rows:
            raise ValueError("cluster_analysis_invalid")
        return self

    def input_refs(self) -> tuple[ClusterAnalysisRef, ...]:
        condition = self.condition
        values = (self.target.dataset, condition.model, condition.tokenizer) + (
            () if condition.predictions is None else (condition.predictions,)
        )
        return tuple({(ref.repo_type, ref.ref): ref for ref in values}.values())


class PairedClusterArmRow(_Data):
    item_id: str = Field(min_length=1, max_length=1024)
    gold: str | int
    base_problem_id: str = Field(min_length=1, max_length=1024)
    outcome: Literal["scored", "invalid"]
    metric_eligible: Literal[True]
    prediction_status: Literal["valid", "invalid", "missing", "refused"]
    pred: str | None
    probabilities: tuple[float | int, ...] | None = Field(max_length=16)
    model_sent: bool | None

    @field_validator("metric_eligible", mode="before")
    @classmethod
    def _eligible(cls, value: object) -> object:
        if type(value) is not bool or value is not True:
            raise ValueError("cluster_analysis_invalid")
        return value

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        for value in (self.item_id, self.base_problem_id):
            if len(value.encode()) > 1024:
                raise ValueError("cluster_analysis_invalid")
        if (self.outcome == "scored") != (self.prediction_status == "valid") or (
            self.prediction_status == "valid"
        ) != (self.pred is not None):
            raise ValueError("cluster_analysis_invalid")
        if type(self.gold) is int and abs(self.gold) > 2**53 - 1:
            raise ValueError("cluster_analysis_invalid")
        if type(self.gold) is str and (not self.gold or len(self.gold.encode()) > 64):
            raise ValueError("cluster_analysis_invalid")
        if self.pred is not None and (not self.pred or len(self.pred.encode()) > 64):
            raise ValueError("cluster_analysis_invalid")
        if self.probabilities is not None and any(
            type(value) not in (int, float) or not 0 <= value <= 1 for value in self.probabilities
        ):
            raise ValueError("cluster_analysis_invalid")
        return self


class PairedClusterPlanV1(_Data):
    schema_version: Literal[1]
    protocol: Literal["strata.eval.cluster-analysis.local.v1"]
    mode: Literal["paired_cluster_percentile"]
    baseline: PairedClusterSourceMetadata
    candidate: PairedClusterSourceMetadata
    profile: PairedClusterProfile
    soft_budget_ms: int = Field(ge=1, le=3_600_000)

    @model_validator(mode="after")
    def _paired(self) -> Self:
        if (
            self.baseline.target != self.candidate.target
            or self.baseline.condition.condition_id == self.candidate.condition.condition_id
            or self.profile.labels != self.baseline.target.labels
        ):
            raise ValueError("cluster_analysis_invalid")
        if self.profile.probability_kind is not None and any(
            arm.condition.probability_kind != self.profile.probability_kind
            for arm in (self.baseline, self.candidate)
        ):
            raise ValueError("cluster_analysis_invalid")
        return self


class ClusterPolicyRequest(_Data):
    schema_version: Literal[1]
    purpose: Literal["paired_cluster_uncertainty"]
    plan_sha256: Sha
    baseline: PairedClusterSourceMetadata
    candidate: PairedClusterSourceMetadata

    def input_refs(self) -> tuple[ClusterAnalysisRef, ...]:
        return tuple(
            {
                (ref.repo_type, ref.ref): ref
                for source in (self.baseline, self.candidate)
                for ref in source.input_refs()
            }.values()
        )


class ClusterCanonicalRead(_Data):
    requested: ClusterAnalysisRef
    canonical_ref: ClusterAnalysisRef
    custody: Literal["unsealed", "protected", "unknown"]


class ClusterPopulationRegistration(_Data):
    registration_id: Text
    revision: Text
    metadata_sha256: Sha
    canonical_target: PairedClusterTarget
    baseline_source_sha256: Sha
    candidate_source_sha256: Sha
    baseline_source_receipt_sha256: Sha
    candidate_source_receipt_sha256: Sha
    baseline_records_sha256: Sha
    candidate_records_sha256: Sha
    coverage: Literal["known", "unknown"]
    purpose: Literal["paired_cluster_uncertainty"]


class CurrentClusterDecision(_Data):
    schema_version: Literal[1]
    request_sha256: Sha
    status: Literal["allow_unsealed", "deny", "unknown"]
    policy_revision: Text
    reads: tuple[ClusterCanonicalRead, ...] = Field(max_length=8)
    registration: ClusterPopulationRegistration | None


def cluster_decision_identity(
    request: ClusterPolicyRequest, decision: CurrentClusterDecision
) -> bytes | None:
    """Check metadata consistency only; the installed caller owns decision authenticity."""
    registration = decision.registration
    if (
        decision.status != "allow_unsealed"
        or decision.request_sha256 != request.digest()
        or registration is None
        or registration.coverage != "known"
    ):
        return None
    expected = {(ref.repo_type, ref.ref) for ref in request.input_refs()}
    if (
        len(decision.reads) != len(expected)
        or {(read.requested.repo_type, read.requested.ref) for read in decision.reads} != expected
    ):
        return None
    canonical_dataset = None
    for read in decision.reads:
        if (
            read.custody != "unsealed"
            or read.requested.repo_type != read.canonical_ref.repo_type
            or read.requested.ref.rsplit("@", 1)[1] != read.canonical_ref.ref.rsplit("@", 1)[1]
        ):
            return None
        if read.requested == request.baseline.target.dataset:
            canonical_dataset = read.canonical_ref
    if registration.canonical_target != request.baseline.target.model_copy(
        update={"dataset": canonical_dataset}
    ):
        return None
    for prefix, metadata in (("baseline", request.baseline), ("candidate", request.candidate)):
        for suffix, value in (
            ("source_sha256", metadata.digest()),
            ("source_receipt_sha256", metadata.condition.source_receipt_sha256),
            ("records_sha256", metadata.condition.records_sha256),
        ):
            if getattr(registration, f"{prefix}_{suffix}") != value:
                return None
    return json.dumps(
        {
            "registration": registration.model_dump(mode="json"),
            "reads": [
                read.model_dump(mode="json")
                for read in sorted(
                    decision.reads, key=lambda read: (read.requested.repo_type, read.requested.ref)
                )
            ],
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


class PairedClusterReceipt(_Data):
    schema_version: Literal[1]
    status: Literal["complete", "undefined"]
    plan: PairedClusterPlanV1
    plan_sha256: Sha
    baseline_source_sha256: Sha
    candidate_source_sha256: Sha
    baseline_observed_rows: int = Field(ge=0, le=100_000)
    candidate_observed_rows: int = Field(ge=0, le=100_000)
    baseline_observed_bytes: int = Field(ge=0, le=MAX_SOURCE_BYTES)
    candidate_observed_bytes: int = Field(ge=0, le=MAX_SOURCE_BYTES)
    baseline_records_sha256: Sha
    candidate_records_sha256: Sha
    baseline_probability_rows: int = Field(ge=0, le=100_000)
    candidate_probability_rows: int = Field(ge=0, le=100_000)
    result: ClusterBootstrapResult
    initial_decision: CurrentClusterDecision
    final_decision: CurrentClusterDecision
    initial_decision_sha256: Sha
    final_decision_sha256: Sha
    decision_chain_sha256: Sha
    policy_checks: int = Field(ge=2)
    elapsed_to_final_policy_ms: float = Field(ge=0, allow_inf_nan=False)
    python_version: Text
    numpy_version: Text
    implementation_attestation: Literal["unattested_local_sdk"]

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        rows = self.plan.baseline.target.expected_metric_rows
        if (
            self.plan_sha256 != self.plan.digest()
            or self.baseline_source_sha256 != self.plan.baseline.digest()
            or self.candidate_source_sha256 != self.plan.candidate.digest()
            or self.status != self.result.status
            or self.result.profile != self.plan.profile
            or self.result.paired_rows != rows
            or self.baseline_observed_rows != rows
            or self.candidate_observed_rows != rows
            or self.baseline_observed_bytes + self.candidate_observed_bytes > MAX_SOURCE_BYTES
            or self.baseline_records_sha256 != self.plan.baseline.condition.records_sha256
            or self.candidate_records_sha256 != self.plan.candidate.condition.records_sha256
            or self.initial_decision_sha256 != self.initial_decision.digest()
            or self.final_decision_sha256 != self.final_decision.digest()
        ):
            raise ValueError("cluster_analysis_invalid")
        if any(
            n > rows for n in (self.baseline_probability_rows, self.candidate_probability_rows)
        ) or (
            self.plan.profile.probability_kind is not None
            and (self.baseline_probability_rows != rows or self.candidate_probability_rows != rows)
        ):
            raise ValueError("cluster_analysis_invalid")
        request = ClusterPolicyRequest(
            schema_version=1,
            purpose="paired_cluster_uncertainty",
            plan_sha256=self.plan_sha256,
            baseline=self.plan.baseline,
            candidate=self.plan.candidate,
        )
        initial = cluster_decision_identity(request, self.initial_decision)
        final = cluster_decision_identity(request, self.final_decision)
        if (
            initial is None
            or final != initial
            or self.elapsed_to_final_policy_ms > self.plan.soft_budget_ms
        ):
            raise ValueError("cluster_analysis_invalid")
        return self


_MODELS = (
    ClusterAnalysisRef,
    ClusterGoldIndex,
    PairedClusterTarget,
    PairedClusterCondition,
    PairedClusterSourceMetadata,
    PairedClusterArmRow,
    PairedClusterPlanV1,
    ClusterPolicyRequest,
    ClusterCanonicalRead,
    ClusterPopulationRegistration,
    CurrentClusterDecision,
    PairedClusterReceipt,
    PopulationCounts,
    PairedClusterProfile,
    ClusterBootstrapResult,
    BootstrapMetric,
    PercentileInterval,
    MetricValue,
)


def revalidate_cluster_value[T: BaseModel](value: T) -> T:
    def raw(item: object, depth: int = 0) -> object:
        if depth > 20:
            raise ClusterAnalysisError()
        if any(type(item) is model_type for model_type in _MODELS):
            model = cast("BaseModel", item)
            storage: object = object.__getattribute__(model, "__dict__")
            if type(storage) is not dict or model.__pydantic_extra__ is not None:
                raise ClusterAnalysisError()
            data = cast("dict[object,object]", storage).copy()
            fields = type(model).model_fields
            if any(type(key) is not str for key in data) or set(data) != set(fields):
                raise ClusterAnalysisError()
            return {key: raw(data[key], depth + 1) for key in fields}
        if type(item) is tuple:
            values = cast("tuple[object,...]", item)
            if len(values) > 256:
                raise ClusterAnalysisError()
            return tuple(raw(child, depth + 1) for child in values)
        if item is None or any(
            type(item) is native_type for native_type in (str, int, float, bool)
        ):
            return item
        raise ClusterAnalysisError()

    if not any(type(value) is model_type for model_type in _MODELS):
        raise ClusterAnalysisError()
    return type(value).model_validate(raw(value))
