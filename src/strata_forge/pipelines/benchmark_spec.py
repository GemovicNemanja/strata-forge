"""Closed local CPU benchmark values; no executable, path, credential or loading contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.core.errors import ValidationError
from strata_forge.pipelines._hub_ids import validate_hub_ref

MAX_BENCHMARK_ROWS = 128
MAX_BENCHMARK_ROW_BYTES = 256 * 1024
MAX_BENCHMARK_TEXT_BYTES = 8 * 1024**2
MAX_BENCHMARK_TENSOR_BYTES = 32 * 1024**2
MAX_BENCHMARK_LENGTH = 8192
MAX_BENCHMARK_APPLICATIONS = 100_000
MAX_DURATION_NS = 2**53 - 1

type BenchmarkBoundary = Literal[
    "hf_scoring_sdk", "verifier_preencoded_sdk", "verifier_encode_score_sdk"
]
type BenchmarkReason = Literal[
    "benchmark_invalid",
    "benchmark_target_unsupported",
    "benchmark_input_invalid",
    "benchmark_overflow",
    "benchmark_failed",
    "benchmark_deadline",
    "benchmark_timer_invalid",
]
_CODES = frozenset(
    {
        "benchmark_invalid",
        "benchmark_target_unsupported",
        "benchmark_input_invalid",
        "benchmark_overflow",
        "benchmark_failed",
        "benchmark_deadline",
        "benchmark_timer_invalid",
    }
)


class BenchmarkError(ValidationError):
    """A fixed content-free admission refusal; native errors are never retained as context."""

    def __init__(self, code: str = "benchmark_invalid") -> None:
        super().__init__(code if code in _CODES else "benchmark_invalid")


class _Data(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    @field_validator("version", "concurrency", mode="before", check_fields=False)
    @classmethod
    def _exact_literal_types(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("benchmark_invalid")
        return value

    @field_validator("clock_monotonic", mode="before", check_fields=False)
    @classmethod
    def _monotonic_bool(cls, value: object) -> object:
        if value is not True:
            raise ValueError("benchmark_invalid")
        return value


class ClassifierBenchmarkPlanV1(_Data):
    """Explicit local measurement choices, without a model loader or mutable runtime options."""

    version: Literal[1]
    boundary: BenchmarkBoundary
    device: Literal["cpu"]
    precision: Literal["fp32"]
    concurrency: Literal[1]
    batch_size: int = Field(ge=1, le=MAX_BENCHMARK_ROWS)
    warmup_calls: int = Field(ge=0, le=50)
    measured_calls: int = Field(ge=1, le=1000)
    max_elapsed_ms: int = Field(ge=1, le=300_000)

    @field_validator("version", "concurrency", mode="before")
    @classmethod
    def _literal_integer(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("benchmark_invalid")
        return value

    def canonical(self) -> bytes:
        """Return bounded deterministic data, not an execution or authority attestation."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()

    def digest(self) -> str:
        """Identify the declared measurement choices."""
        return hashlib.sha256(self.canonical()).hexdigest()


@dataclass(frozen=True, repr=False)
class ResolvedBenchmarkTarget:
    """Trusted native objects and declared refs; neither attests origin or authorizes access."""

    model_ref: str
    tokenizer_ref: str
    scorer: object | None = None
    verifier: object | None = None
    tokenizer: object | None = None


def _sha(value: str) -> str:
    if re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("benchmark_invalid")
    return value


class BenchmarkTargetDescriptor(_Data):
    """Bounded actual runtime metadata, with source pins explicitly marked as declarations."""

    model_ref: str
    tokenizer_ref: str
    boundary: BenchmarkBoundary
    engine: Literal["hf_classify", "hf_label_logits", "forge_auxiliary_verifier_v1"]
    family: Literal["bert", "gpt2", "modernbert-decoder"]
    probability_family: Literal["full_class_softmax", "restricted_label_token_softmax"]
    label_count: int = Field(ge=2, le=16)
    ordered_labels_sha256: str
    tokenizer_sha256: str
    geometry_sha256: str | None
    outputs: tuple[Literal["label", "inconsistent", "support_set"], ...]
    rows_per_call: int = Field(ge=1, le=MAX_BENCHMARK_ROWS)
    max_length: int = Field(ge=1, le=MAX_BENCHMARK_LENGTH)
    native_context: int = Field(ge=1)
    input_shape: Literal["single", "pair", "preencoded"]
    mode_policy: Literal["scorer_preexisting_eval", "verifier_save_eval_restore"]
    source_evidence: Literal["caller_resolved_unattested"] = "caller_resolved_unattested"
    exact_token_lengths: tuple[int, ...] | None
    token_lengths_reason: Literal["sdk_has_no_length_receipt"] | None
    padding_side: Literal["left", "right"]

    _refs = field_validator("model_ref", "tokenizer_ref")(validate_hub_ref)
    _hashes = field_validator("ordered_labels_sha256", "tokenizer_sha256")(_sha)

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.geometry_sha256 is not None:
            _sha(self.geometry_sha256)
        if (
            self.max_length > self.native_context
            or not self.outputs
            or self.outputs[0] != "label"
            or len(set(self.outputs)) != len(self.outputs)
        ):
            raise ValueError("benchmark_invalid")
        if (self.exact_token_lengths is None) != (self.token_lengths_reason is not None):
            raise ValueError("benchmark_invalid")
        if self.exact_token_lengths is not None and (
            len(self.exact_token_lengths) != self.rows_per_call
            or any(
                type(length) is not int or not 1 <= length <= self.max_length
                for length in self.exact_token_lengths
            )
        ):
            raise ValueError("benchmark_invalid")
        return self

    def digest(self) -> str:
        """Identify metadata without proving native object provenance."""
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class BenchmarkEnvironment(_Data):
    """An allowlist with no hostname, environment variables, model repr or raw paths."""

    python: str = Field(min_length=1, max_length=128)
    torch: str = Field(min_length=1, max_length=128)
    transformers: str = Field(min_length=1, max_length=128)
    operating_system: str = Field(min_length=1, max_length=128)
    architecture: str = Field(min_length=1, max_length=128)
    device: Literal["cpu"] = "cpu"
    precision: Literal["fp32"] = "fp32"
    concurrency: Literal[1] = 1
    intraop_threads: int = Field(ge=1)
    interop_threads: int = Field(ge=1)
    attention: Literal["eager", "sdpa"] | None
    clock: Literal["perf_counter_ns"] = "perf_counter_ns"
    clock_implementation: str = Field(min_length=1, max_length=256)
    clock_resolution_ns: float = Field(gt=0, allow_inf_nan=False)
    clock_monotonic: Literal[True] = True


class BenchmarkStatistics(_Data):
    """Nearest-rank call latencies and two explicitly different throughput denominators."""

    quantile_method: Literal["nearest_rank_v1"] = "nearest_rank_v1"
    sample_count: int = Field(ge=1, le=1000)
    p50_rank: int = Field(ge=1)
    p95_rank: int = Field(ge=1)
    p99_rank: int = Field(ge=1)
    p50_ns: int = Field(ge=0)
    p95_ns: int = Field(ge=0)
    p99_ns: int = Field(ge=0)
    min_ns: int = Field(ge=0)
    max_ns: int = Field(ge=0)
    mean_ns: float = Field(ge=0, allow_inf_nan=False)
    summed_sdk_ns: int = Field(ge=0)
    empirical_cdf_step: float = Field(gt=0, le=1, allow_inf_nan=False)
    rows_per_second: float | None = Field(ge=0, allow_inf_nan=False)
    rows_rate_reason: Literal["zero_elapsed"] | None
    sdk_active_rows_per_second: float | None = Field(ge=0, allow_inf_nan=False)
    sdk_rate_reason: Literal["zero_elapsed"] | None
    amortized_sdk_ns_per_row: float = Field(ge=0, allow_inf_nan=False)


def _statistics(durations: tuple[int, ...], window_ns: int, rows: int) -> BenchmarkStatistics:
    values = sorted(durations)
    n, total = len(values), sum(values)
    ranks = tuple((percent * n + 99) // 100 for percent in (50, 95, 99))
    return BenchmarkStatistics(
        sample_count=n,
        p50_rank=ranks[0],
        p95_rank=ranks[1],
        p99_rank=ranks[2],
        p50_ns=values[ranks[0] - 1],
        p95_ns=values[ranks[1] - 1],
        p99_ns=values[ranks[2] - 1],
        min_ns=values[0],
        max_ns=values[-1],
        mean_ns=total / n,
        summed_sdk_ns=total,
        empirical_cdf_step=1 / n,
        rows_per_second=None if window_ns == 0 else rows * 1e9 / window_ns,
        rows_rate_reason="zero_elapsed" if window_ns == 0 else None,
        sdk_active_rows_per_second=None if total == 0 else rows * 1e9 / total,
        sdk_rate_reason="zero_elapsed" if total == 0 else None,
        amortized_sdk_ns_per_row=total / rows,
    )


class ClassifierBenchmarkResult(_Data):
    """Private caller result; consistent arithmetic is not a measurement/issuer attestation."""

    version: Literal[1] = 1
    status: Literal["complete", "failed", "deadline"]
    reason: BenchmarkReason | None
    plan: ClassifierBenchmarkPlanV1
    plan_sha256: str
    target: BenchmarkTargetDescriptor
    target_sha256: str
    workload_sha256: str
    environment: BenchmarkEnvironment
    warmup_completed: int = Field(ge=0)
    measured_attempted: int = Field(ge=0)
    measured_completed: int = Field(ge=0)
    measured_rows: int = Field(ge=0)
    warmup_durations_ns: tuple[int, ...] = Field(max_length=50)
    measured_durations_ns: tuple[int, ...] = Field(max_length=1000)
    failed_phase: Literal["warmup", "measured"] | None
    failed_call_index: int | None = Field(ge=0)
    failed_call_duration_ns: int | None = Field(ge=0)
    measurement_window_ns: int | None = Field(ge=0)
    total_elapsed_ns: int = Field(ge=0)
    statistics: BenchmarkStatistics | None
    cold_start_reason: Literal["already_resolved_target"] = "already_resolved_target"
    forward_only_reason: Literal["sdk_boundary_includes_postprocessing"] = (
        "sdk_boundary_includes_postprocessing"
    )
    serving_reason: Literal["local_sdk_only"] = "local_sdk_only"
    authority: Literal["none_local_trusted_objects_only"] = "none_local_trusted_objects_only"

    _hashes = field_validator("plan_sha256", "target_sha256", "workload_sha256")(_sha)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (
            self.plan.digest() != self.plan_sha256
            or self.target.digest() != self.target_sha256
            or self.plan.boundary != self.target.boundary
            or self.warmup_completed != len(self.warmup_durations_ns)
            or self.measured_completed != len(self.measured_durations_ns)
            or self.measured_rows != self.target.rows_per_call * self.measured_completed
            or not self.measured_completed <= self.measured_attempted <= self.plan.measured_calls
            or self.warmup_completed > self.plan.warmup_calls
        ):
            raise ValueError("benchmark_invalid")
        times = (
            *self.warmup_durations_ns,
            *self.measured_durations_ns,
            *(() if self.failed_call_duration_ns is None else (self.failed_call_duration_ns,)),
        )
        if any(type(value) is not int or not 0 <= value <= MAX_DURATION_NS for value in times) or (
            self.measurement_window_ns is not None
            and self.measurement_window_ns < sum(self.measured_durations_ns)
        ):
            raise ValueError("benchmark_invalid")
        if (
            self.measurement_window_ns is not None
            and self.total_elapsed_ns < self.measurement_window_ns + sum(self.warmup_durations_ns)
        ):
            raise ValueError("benchmark_invalid")
        if (self.failed_phase is None) != (self.failed_call_index is None):
            raise ValueError("benchmark_invalid")
        if self.status == "deadline" and self.reason != "benchmark_deadline":
            raise ValueError("benchmark_invalid")
        if self.status == "complete":
            if (
                self.total_elapsed_ns > self.plan.max_elapsed_ms * 1_000_000
                or self.reason is not None
                or self.failed_phase is not None
                or self.failed_call_index is not None
                or self.failed_call_duration_ns is not None
                or self.warmup_completed != self.plan.warmup_calls
                or self.measured_completed != self.plan.measured_calls
                or self.measurement_window_ns is None
                or self.statistics
                != _statistics(
                    self.measured_durations_ns, self.measurement_window_ns, self.measured_rows
                )
            ):
                raise ValueError("benchmark_invalid")
        elif self.reason is None or self.statistics is not None:
            raise ValueError("benchmark_invalid")
        if self.statistics is not None and not math.isfinite(self.statistics.mean_ns):
            raise ValueError("benchmark_invalid")
        return self
