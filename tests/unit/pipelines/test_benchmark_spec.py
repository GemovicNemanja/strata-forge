"""Closed benchmark choices and malformed frozen-result counterexamples."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from strata_forge.pipelines.benchmark_spec import (
    ClassifierBenchmarkPlanV1,
    ClassifierBenchmarkResult,
)
from tests.unit.pipelines.test_classifier_benchmark import Clock, Operation, measure, plan


@pytest.mark.parametrize(
    "update",
    [
        {"version": True},
        {"version": 1.0},
        {"version": 2},
        {"device": "cuda"},
        {"device": "mps"},
        {"precision": "bf16"},
        {"concurrency": 2},
        {"concurrency": True},
        {"batch_size": 0},
        {"batch_size": 129},
        {"batch_size": True},
        {"warmup_calls": -1},
        {"warmup_calls": 51},
        {"measured_calls": 0},
        {"measured_calls": 1001},
        {"max_elapsed_ms": 0},
        {"max_elapsed_ms": 300001},
        {"callback": "package:execute"},
        {"model_path": "/private/fictional"},
        {"provider": "external"},
        {"boundary": "forward_only"},
        {"boundary": "cold_start"},
        {"boundary": "served_http"},
    ],
)
def test_measurement_plan_refuses_unsupported_values_without_passthrough(
    update: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        plan(**update)


@pytest.mark.asyncio
async def test_public_result_rejects_inconsistent_counts_statistics_and_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await measure(plan(), Operation(Clock()), monkeypatch)
    raw = result.model_dump()
    variants = [
        {"measured_rows": 19},
        {"measured_attempted": 0},
        {"warmup_completed": 0},
        {"plan_sha256": "0" * 64},
        {"target_sha256": "0" * 64},
        {"workload_sha256": "f" * 64 + "\n"},
        {"status": "failed"},
        {"version": True},
        {"measurement_window_ns": 19_000_000},
        {"total_elapsed_ns": 0},
    ]
    for values in variants:
        with pytest.raises(ValidationError):
            ClassifierBenchmarkResult.model_validate({**raw, **values})
    statistics = dict(raw["statistics"])
    statistics["rows_per_second"] = 1000.0
    with pytest.raises(ValidationError):
        ClassifierBenchmarkResult.model_validate({**raw, "statistics": statistics})
    assert ClassifierBenchmarkResult.model_validate_json(result.model_dump_json()) == result


def test_declared_plan_canonicalization_is_stable_and_version_specific() -> None:
    left = plan()
    right = ClassifierBenchmarkPlanV1.model_validate_json(left.canonical())
    assert left.digest() == right.digest()
    assert left.canonical() == right.canonical()
    assert plan(batch_size=8).digest() != left.digest()
