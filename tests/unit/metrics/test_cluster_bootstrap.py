"""Fictional cluster controls independent of the production count accumulation."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from fractions import Fraction
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError as ModelValidationError

from strata_forge.metrics import brier, classification_report, nll
from strata_forge.metrics.cluster_bootstrap import (
    ClusterBootstrapError,
    ClusterBootstrapResult,
    PairedClusterObservation,
    PairedClusterProfile,
    _ClusterBootstrapState,  # pyright: ignore[reportPrivateUsage] — bounded partial-state arithmetic controls
    paired_cluster_percentile,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "cluster_bootstrap_v1.json").read_text())


def profile(*, probabilities: bool = True) -> PairedClusterProfile:
    return PairedClusterProfile(
        schema_version=1,
        labels=("A", "B", "C"),
        metrics=("accuracy", "macro_f1", "invalid_rate", "brier", "nll")
        if probabilities
        else ("accuracy", "macro_f1", "invalid_rate"),
        probability_kind="class_probabilities" if probabilities else None,
        resamples=10000,
        seed=20261104,
        confidence_level="0.95",
        weighting="row_pooled",
        interval="percentile_nearest_rank_v1",
        rng_profile="numpy_generator_pcg64_int64_v1",
    )


def observations(*, probabilities: bool = True) -> tuple[PairedClusterObservation, ...]:
    result: list[PairedClusterObservation] = []
    labels = ("A", "B", "C")
    for row in FIXTURE["rows"]:
        result.append(
            PairedClusterObservation(
                item_id=row["item_id"],
                base_problem_id=row["base_problem_id"],
                gold_index=labels.index(row["gold"]),
                baseline_pred_index=None
                if row["baseline_pred"] is None
                else labels.index(row["baseline_pred"]),
                candidate_pred_index=None
                if row["candidate_pred"] is None
                else labels.index(row["candidate_pred"]),
                baseline_probabilities=tuple(row["baseline_probabilities"])
                if probabilities
                else None,
                candidate_probabilities=tuple(row["candidate_probabilities"])
                if probabilities
                else None,
            )
        )
    return tuple(result)


@pytest.fixture(scope="module")
def golden() -> ClusterBootstrapResult:
    return paired_cluster_percentile(observations(), profile=profile())


def test_actual_10000_paired_draws_have_independent_pcg64_digest_and_denominators(
    golden: ClusterBootstrapResult,
) -> None:
    rng = np.random.Generator(np.random.PCG64(20261104))
    digest = hashlib.sha256(b"strata.analysis.cluster.draws.v1\0" + struct.pack("<QQ", 10000, 2))
    sizes: list[int] = []
    for _ in range(10000):
        draw = rng.integers(0, 2, size=2, dtype=np.int64, endpoint=False)
        digest.update(draw.astype("<i8", copy=False).tobytes())
        sizes.append(sum((1, 3)[int(i)] for i in draw))
    assert (
        digest.hexdigest()
        == golden.draws_sha256
        == "a140fd9940d9c907776e888073704cba8a8323fa5fd77824a09e899def735e4f"
    )
    assert golden.resamples_completed == 10000
    assert golden.draw_work_units == golden.planned_draw_work_units == 20000
    assert (
        (golden.minimum_resample_rows, golden.maximum_resample_rows)
        == (min(sizes), max(sizes))
        == (2, 6)
    )
    assert (
        golden.replicates_sha256
        == "e2bcbf54cee517c65cfeaa9b04c4e6b2d9114abb7ff34570dc0fcd7c1214a1e9"
    )


def test_exact_fraction_and_expanded_existing_point_controls() -> None:
    state = _ClusterBootstrapState(observations(), profile())
    rows = FIXTURE["rows"]
    groups = [
        [row for row in rows if row["base_problem_id"] == name] for name in FIXTURE["cluster_order"]
    ]
    for case in FIXTURE["explicit_draw_controls"]:
        weights = tuple(case["draw"].count(index) for index in range(2))
        actual = state._metrics(weights)  # pyright: ignore[reportPrivateUsage] — compare fixed weighted draws before completion
        expanded = [row for i in case["draw"] for row in groups[i]]
        gold = [row["gold"] for row in expanded]
        assert len(expanded) == case["n"]
        for arm_index, arm in enumerate(("baseline", "candidate")):
            report = classification_report(
                gold, [row[arm + "_pred"] for row in expanded], labels=("A", "B", "C")
            )
            for metric_index, name in enumerate(("accuracy", "macro_f1", "invalid_rate", "brier")):
                value = actual[arm_index][metric_index]
                assert value.value is not None
                assert value.n == len(expanded)
                assert math.isclose(
                    value.value, float(Fraction(case[arm][name])), rel_tol=0, abs_tol=1e-15
                )
                if name != "brier":
                    assert value == getattr(report, name)
            vectors = [row[arm + "_probabilities"] for row in expanded]
            for index, function in ((3, brier), (4, nll)):
                value = function(
                    gold, vectors, labels=("A", "B", "C"), probability_kind="class_probabilities"
                )
                assert value.value is not None
                assert actual[arm_index][index].value is not None
                assert math.isclose(
                    value.value,
                    cast("float", actual[arm_index][index].value),
                    abs_tol=1e-12,
                    rel_tol=1e-12,
                )
        assert actual[0][0].value != float(Fraction(2, 3)) or case["draw"] != [0, 1]
    with pytest.raises(ClusterBootstrapError):
        state.result()


def test_percentile_endpoints_are_paired_not_marginal_subtraction(
    golden: ClusterBootstrapResult,
) -> None:
    acc = golden.metrics[0]
    assert acc.delta == 0
    assert acc.baseline_interval is not None
    assert acc.candidate_interval is not None
    assert acc.delta_interval is not None
    assert (acc.delta_interval.lower, acc.delta_interval.upper) == (-1, 1 / 3)
    macro = golden.metrics[1]
    assert macro.delta_interval is not None
    assert macro.candidate_interval is not None
    assert macro.baseline_interval is not None
    assert (
        macro.delta_interval.lower != macro.candidate_interval.lower - macro.baseline_interval.upper
    )
    assert golden.metrics[1].delta_interval is not None
    assert math.isclose(golden.metrics[1].delta_interval.upper, 7 / 18, rel_tol=0, abs_tol=1e-15)
    assert golden.metrics[3].delta_interval is not None
    assert (golden.metrics[3].delta_interval.lower, golden.metrics[3].delta_interval.upper) == (
        -1 / 3,
        1,
    )


def test_canonical_order_is_independent_of_input_permutation(
    golden: ClusterBootstrapResult,
) -> None:
    actual = paired_cluster_percentile(tuple(reversed(observations())), profile=profile())
    assert actual == golden


@pytest.mark.parametrize("count", [0, 1])
def test_zero_rows_and_one_cluster_do_not_fabricate_interval(count: int) -> None:
    result = paired_cluster_percentile(
        observations(probabilities=False)[:count], profile=profile(probabilities=False)
    )
    assert result.status == "undefined"
    assert result.resamples_completed == 0
    assert result.draws_sha256 is result.replicates_sha256 is result.numpy_version is None
    assert result.draw_work_units == result.planned_draw_work_units == 0
    for metric in result.metrics:
        assert metric.baseline_interval is metric.delta_interval is None
        assert metric.interval_reason == (
            "insufficient_clusters" if count else "no_metric_population"
        )
        assert (metric.baseline.value is not None) == bool(count)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gold_index", True),
        ("gold_index", 3),
        ("gold_index", -1),
        ("baseline_pred_index", 4),
        ("item_id", "\ud800"),
        ("item_id", ""),
        ("base_problem_id", ""),
        ("baseline_probabilities", None),
        ("baseline_probabilities", (1, 0)),
        ("baseline_probabilities", (float("nan"), 0, 1)),
        ("baseline_probabilities", (1, 1, 1)),
        ("baseline_probabilities", (True, 0, 0)),
        ("baseline_probabilities", (10**1000, 0, 0)),
    ],
)
def test_forged_malformed_rows_refuse_without_raw_context(field: str, value: object) -> None:
    rows = observations()
    bad = rows[0].model_copy(update={field: value})
    with pytest.raises(ClusterBootstrapError) as error:
        paired_cluster_percentile((bad, *rows[1:]), profile=profile())
    assert error.value.__cause__ is error.value.__context__ is None


@pytest.mark.parametrize(
    "patch",
    [
        {"schema_version": True},
        {"resamples": 9999},
        {"seed": 1},
        {"confidence_level": ".95"},
        {"metrics": ("accuracy", "accuracy")},
        {"labels": ("A", "A")},
        {"probability_kind": None},
        {"weighting": "equal_clusters"},
    ],
)
def test_preconstructed_profiles_revalidate_all_fields(patch: dict[str, object]) -> None:
    with pytest.raises(ClusterBootstrapError):
        paired_cluster_percentile(observations(), profile=profile().model_copy(update=patch))


def test_no_unrequested_probability_or_duplicate_id_is_silently_ignored() -> None:
    with pytest.raises(ClusterBootstrapError):
        paired_cluster_percentile(observations(), profile=profile(probabilities=False))
    with pytest.raises(ClusterBootstrapError):
        paired_cluster_percentile((observations()[0], observations()[0]), profile=profile())


def test_memory_and_work_bounds_refuse_before_distribution_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import strata_forge.metrics.cluster_bootstrap as module

    def no_allocation(*args: object, **kwargs: object) -> object:
        raise AssertionError("must not allocate after denied planning")

    monkeypatch.setattr(np, "empty", no_allocation)
    monkeypatch.setattr(module, "MAX_WORK", 1)
    with pytest.raises(ClusterBootstrapError):
        paired_cluster_percentile(observations(), profile=profile())
    monkeypatch.setattr(module, "MAX_WORK", 100000000)
    monkeypatch.setattr(module, "MAX_ARRAY_BYTES", 1)
    with pytest.raises(ClusterBootstrapError):
        paired_cluster_percentile(observations(), profile=profile())


def test_arrays_never_have_resample_by_row_or_cluster_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    original = np.empty
    allocated: list[tuple[int, int, int]] = []

    def empty(
        shape: tuple[int, int, int], *, dtype: type[np.float64]
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        allocated.append(shape)
        return original(shape, dtype=dtype)

    monkeypatch.setattr(np, "empty", empty)
    state = _ClusterBootstrapState(observations(), profile())
    assert allocated == [(10000, 5, 3)]
    assert state.planned < 256 * 1024**2


@pytest.mark.parametrize(
    "patch",
    [
        {"resamples_completed": False},
        {"status": "undefined"},
        {"clusters": 0},
        {"paired_rows": 3},
        {"draw_work_units": 0},
        {"maximum_resample_rows": 1000000000},
        {"metrics": ()},
        {"draws_sha256": None},
    ],
)
def test_mismatched_result_fields_are_not_a_valid_frozen_receipt(
    golden: ClusterBootstrapResult, patch: dict[str, object]
) -> None:
    data = golden.model_dump(mode="python")
    data.update(patch)
    with pytest.raises(ModelValidationError):
        ClusterBootstrapResult.model_validate(data)


def test_nearest_rank_exact_indexes_with_distinct_sorted_values() -> None:
    # Isolate rank selection; this injected private state is not an RNG conformance run.
    state = _ClusterBootstrapState(observations(probabilities=False), profile(probabilities=False))
    values = cast("np.ndarray[tuple[int,int,int],np.dtype[np.float64]]", state.values)
    values[:, :, 0] = np.arange(10000, dtype=np.float64)[:, None] / 9999
    values[:, :, 1] = values[:, :, 0]
    values[:, :, 2] = 0
    state.completed = 10000
    state.low_n = state.high_n = 4
    result = state.result()
    for metric in result.metrics:
        assert metric.baseline_interval is not None
        assert metric.baseline_interval.lower == 249 / 9999
        assert metric.baseline_interval.upper == 9749 / 9999
        assert metric.baseline_interval.lower_rank == 250
        assert metric.baseline_interval.upper_rank == 9750


def test_probability_clipping_and_invalids_retain_variable_population() -> None:
    rows = tuple(
        row.model_copy(
            update={"baseline_probabilities": (0, 1, 0), "candidate_probabilities": (0, 1, 0)}
        )
        for row in observations()
    )
    state = _ClusterBootstrapState(rows, profile())
    actual = state._metrics((2, 0))  # pyright: ignore[reportPrivateUsage] — exact small repeated-cluster control
    assert actual[0][4].n == actual[0][4].clipped == 2
    assert actual[0][4].value == -math.log(1e-15)
    actual = state._metrics((0, 2))  # pyright: ignore[reportPrivateUsage] — invalid outputs retain independent probabilities
    assert actual[0][4].n == 6
    assert actual[0][4].clipped == 2
    assert actual[0][2].value == 1 / 3


def test_row_and_cluster_caps_do_not_silently_use_a_prefix() -> None:
    row = observations()[0]
    with pytest.raises(ClusterBootstrapError):
        paired_cluster_percentile((row,) * 100001, profile=profile())
    rows = tuple(
        row.model_copy(update={"item_id": str(i), "base_problem_id": str(i)}) for i in range(10001)
    )
    with pytest.raises(ClusterBootstrapError):
        paired_cluster_percentile(rows, profile=profile())


def test_hostile_subclass_values_and_storage_do_not_run_hooks() -> None:
    calls: list[str] = []

    class Text(str):
        def __hash__(self) -> int:
            calls.append("hash")
            raise AssertionError("no subclass hashing")

    class Number(int):
        def __lt__(self, other: object) -> bool:
            calls.append("lt")
            raise AssertionError("no subclass ordering")

    class Model(PairedClusterObservation):
        def __getattribute__(self, name: str) -> object:
            calls.append("getattr")
            raise AssertionError("no subclass access")

    row = observations()[0]
    for bad in (
        row.model_copy(update={"item_id": Text("x")}),
        row.model_copy(update={"gold_index": Number(0)}),
        object.__new__(Model),
    ):
        with pytest.raises(ClusterBootstrapError):
            paired_cluster_percentile((bad,), profile=profile())
    assert calls == []


def test_unknown_metaclass_equality_is_not_run_during_exact_value_copy() -> None:
    calls: list[str] = []

    class Meta(type):
        def __eq__(cls, other: object) -> bool:
            calls.append("eq")
            return False

        __hash__ = type.__hash__

    class Foreign(metaclass=Meta):
        pass

    row = observations()[0].model_copy(update={"gold_index": Foreign()})
    with pytest.raises(ClusterBootstrapError):
        paired_cluster_percentile((row,), profile=profile())
    assert calls == []
