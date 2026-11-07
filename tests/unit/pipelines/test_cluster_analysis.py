"""Trusted fictional adapters; no source metadata flag itself grants row access."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import pytest
from pydantic import ValidationError as ModelValidationError

from strata_forge.metrics.cluster_bootstrap import PairedClusterProfile
from strata_forge.metrics.cluster_bootstrap import (
    _ClusterBootstrapState as BootstrapState,  # pyright: ignore[reportPrivateUsage] — held numerical chunks test owned cancellation
)
from strata_forge.pipelines._population import PopulationCounts
from strata_forge.pipelines.cluster_analysis import (
    paired_cluster_row_bytes,
    paired_registered_cluster_uncertainty,
)
from strata_forge.pipelines.cluster_analysis_spec import (
    ClusterAnalysisError,
    ClusterAnalysisRef,
    ClusterCanonicalRead,
    ClusterPolicyRequest,
    ClusterPopulationRegistration,
    CurrentClusterDecision,
    PairedClusterArmRow,
    PairedClusterCondition,
    PairedClusterPlanV1,
    PairedClusterReceipt,
    PairedClusterSourceMetadata,
    PairedClusterTarget,
)

SHA = "a" * 64
REF = "fictional/shared@" + "b" * 40
type Custody = Literal["unsealed", "protected", "unknown"]


def profile(probabilities: bool = False) -> PairedClusterProfile:
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


class Stream:
    def __init__(self, source: Source) -> None:
        self.source = source
        self.index = 0

    def __aiter__(self) -> Stream:
        return self

    async def __anext__(self) -> PairedClusterArmRow:
        if self.source.read_hold is not None:
            self.source.entered.set()
            await self.source.read_hold.wait()
        if self.index == len(self.source.values):
            raise StopAsyncIteration
        row = self.source.values[self.index]
        self.index += 1
        self.source.reads += 1
        return row

    async def aclose(self) -> None:
        self.source.closed += 1
        if self.source.close_error:
            raise RuntimeError("fictional sensitive source diagnostic")


@dataclass
class Source:
    meta: PairedClusterSourceMetadata
    values: tuple[PairedClusterArmRow, ...]
    opens: int = 0
    reads: int = 0
    closed: int = 0
    metadata_calls: int = 0
    mutate_metadata_at: int | None = None
    read_hold: asyncio.Event | None = None
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    close_error: bool = False

    async def metadata(self) -> PairedClusterSourceMetadata:
        self.metadata_calls += 1
        if self.mutate_metadata_at == self.metadata_calls:
            return self.meta.model_copy(
                update={
                    "condition": self.meta.condition.model_copy(update={"head_sha256": "c" * 64})
                }
            )
        return self.meta

    def rows(self) -> Stream:
        self.opens += 1
        return Stream(self)


class Policy:
    def __init__(self) -> None:
        self.calls = 0
        self.deny_at: int | None = None
        self.protection: Custody | None = None
        self.unknown_registration = False
        self.omit_model = False
        self.mutate_registration_at: int | None = None
        self.hold: asyncio.Event | None = None
        self.entered = asyncio.Event()

    async def decide(self, request: ClusterPolicyRequest) -> CurrentClusterDecision:
        self.calls += 1
        if self.hold is not None:
            self.entered.set()
            await self.hold.wait()
        reads = tuple(
            ClusterCanonicalRead(
                requested=ref,
                canonical_ref=ref,
                custody=self.protection or "unsealed",
            )
            for ref in request.input_refs()
        )
        if self.omit_model:
            reads = tuple(read for read in reads if read.requested.repo_type != "model")
        registration = ClusterPopulationRegistration(
            registration_id="fictional-pair",
            revision="v1" if self.mutate_registration_at != self.calls else "v2",
            metadata_sha256=SHA,
            canonical_target=request.baseline.target,
            baseline_source_sha256=request.baseline.digest(),
            candidate_source_sha256=request.candidate.digest(),
            baseline_source_receipt_sha256=request.baseline.condition.source_receipt_sha256,
            candidate_source_receipt_sha256=request.candidate.condition.source_receipt_sha256,
            baseline_records_sha256=request.baseline.condition.records_sha256,
            candidate_records_sha256=request.candidate.condition.records_sha256,
            coverage="unknown" if self.unknown_registration else "known",
            purpose="paired_cluster_uncertainty",
        )
        return CurrentClusterDecision(
            schema_version=1,
            request_sha256=request.digest(),
            status="deny" if self.deny_at == self.calls else "allow_unsealed",
            policy_revision="fixture-v1",
            reads=reads,
            registration=registration,
        )


def setup(
    *, count: int = 1, probabilities: bool = False
) -> tuple[PairedClusterPlanV1, Source, Source, Policy]:
    f = json.loads(
        (
            Path(__file__).parents[1] / "metrics" / "fixtures" / "cluster_bootstrap_v1.json"
        ).read_text()
    )
    target = PairedClusterTarget(
        dataset=ClusterAnalysisRef(repo_type="dataset", ref=REF),
        config=None,
        split="fictional-eval",
        id_column="id",
        gold_column="gold",
        gold_index_map=(),
        cluster_column="base_problem_id",
        labels=("A", "B", "C"),
        label_space="nominal-v1",
        projection="identity",
        selection="complete_metric_population",
        slice_sha256=SHA,
        target_identity_sha256=SHA,
        selected_population_sha256=SHA,
        metric_population_sha256=SHA,
        expected_metric_rows=count,
    )
    sources: list[Source] = []
    for arm in ("baseline", "candidate"):
        rows = tuple(
            PairedClusterArmRow(
                item_id=row["item_id"],
                gold=row["gold"],
                base_problem_id=row["base_problem_id"],
                outcome="scored" if row[arm + "_pred"] is not None else "invalid",
                metric_eligible=True,
                prediction_status="valid" if row[arm + "_pred"] is not None else "missing",
                pred=row[arm + "_pred"],
                probabilities=tuple(row[arm + "_probabilities"]) if probabilities else None,
                model_sent=True,
            )
            for row in f["rows"][:count]
        )
        failures = sum(row.outcome == "invalid" for row in rows)
        counts = PopulationCounts(
            rows_in=count + 2,
            rows_suite_excluded=1,
            rows_overflow_dropped=1,
            rows_scored_success=count - failures,
            rows_failed_or_invalid=failures,
            rows_metric_population=count,
            rows_model_sent=count,
            invalid_sent=failures,
            invalid_not_sent=0,
            invalid_origin_unknown=0,
        )
        condition = PairedClusterCondition(
            condition_id=arm,
            model=ClusterAnalysisRef(repo_type="model", ref=REF),
            tokenizer=ClusterAnalysisRef(repo_type="model", ref=REF),
            predictions=None,
            source_receipt_sha256=SHA,
            source_projection_sha256=SHA,
            rendering_sha256=SHA,
            head_sha256=SHA,
            probability_kind="class_probabilities" if probabilities else None,
            calibration_state="raw",
            calibration_receipt_sha256=None,
            records_sha256=hashlib.sha256(
                b"".join(paired_cluster_row_bytes(row) for row in rows)
            ).hexdigest(),
            counts=counts,
        )
        metadata = PairedClusterSourceMetadata(
            schema_version=1,
            target=target,
            condition=condition,
            row_codec="strata.analysis.cluster.arm.rows.v1",
        )
        sources.append(Source(metadata, rows))
    a, b = sources
    plan = PairedClusterPlanV1(
        schema_version=1,
        protocol="strata.eval.cluster-analysis.local.v1",
        mode="paired_cluster_percentile",
        baseline=a.meta,
        candidate=b.meta,
        profile=profile(probabilities),
        soft_budget_ms=30000,
    )
    return plan, a, b, Policy()


async def run(parts: tuple[PairedClusterPlanV1, Source, Source, Policy]) -> PairedClusterReceipt:
    plan, a, b, policy = parts
    return await paired_registered_cluster_uncertainty(plan, baseline=a, candidate=b, policy=policy)


async def test_actual_registered_10000_resample_receipt_keeps_counts_and_no_private_rows() -> None:
    parts = setup(count=4, probabilities=True)
    receipt = await run(parts)
    assert receipt.status == "complete"
    assert receipt.result.resamples_completed == 10000
    assert (
        receipt.result.draws_sha256
        == "a140fd9940d9c907776e888073704cba8a8323fa5fd77824a09e899def735e4f"
    )
    assert receipt.result.minimum_resample_rows == 2
    assert receipt.result.maximum_resample_rows == 6
    assert (
        receipt.plan.baseline.condition.counts.rows_in == 6
    )  # E/D evidence is retained, not bootstrapped.
    assert receipt.result.paired_rows == 4
    assert receipt.baseline_probability_rows == receipt.candidate_probability_rows == 4
    assert parts[1].closed == parts[2].closed == 1
    assert receipt.policy_checks >= 100
    assert len(receipt.canonical()) < 65536
    raw = json.loads(receipt.canonical())
    assert "rows" not in raw
    assert "observations" not in raw
    assert "replicates" not in raw["result"]
    assert PairedClusterReceipt.model_validate(receipt.model_dump(mode="python")) == receipt


@pytest.mark.parametrize("count", [0, 1])
async def test_complete_empty_and_single_cluster_sources_have_honest_undefined_status(
    count: int,
) -> None:
    parts = setup(count=count)
    receipt = await run(parts)
    assert receipt.status == "undefined"
    assert receipt.result.resamples_completed == 0
    assert parts[1].reads == parts[2].reads == count
    assert parts[1].closed == parts[2].closed == 1


@pytest.mark.parametrize("reason", ["deny", "protected", "unknown", "registration", "typed_ref"])
async def test_initial_policy_denial_precedes_rows_and_numpy(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    import strata_forge.pipelines.cluster_analysis as module

    parts = setup()
    policy = parts[3]
    if reason == "deny":
        policy.deny_at = 1
    elif reason in ("protected", "unknown"):
        policy.protection = cast("Custody", reason)
    elif reason == "registration":
        policy.unknown_registration = True
    else:
        policy.omit_model = True  # Same textual dataset pin cannot grant the model pin.

    def no_numpy() -> str:
        raise AssertionError("optional numerical execution must not begin")

    monkeypatch.setattr(module, "_numpy_version", no_numpy)
    with pytest.raises(ClusterAnalysisError, match=r"^cluster_analysis_denied$") as caught:
        await run(parts)
    assert caught.value.__cause__ is caught.value.__context__ is None
    assert parts[1].opens == parts[2].opens == 0


async def test_missing_trusted_policy_is_unavailable_before_metadata() -> None:
    plan, a, b, _ = setup()
    with pytest.raises(ClusterAnalysisError, match="unavailable"):
        await paired_registered_cluster_uncertainty(plan, baseline=a, candidate=b, policy=None)
    assert a.metadata_calls == b.metadata_calls == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "gold",
        "group",
        "id",
        "duplicate",
        "extra",
        "digest",
        "invalid_counts",
        "probability_missing",
        "probability_sum",
    ],
)
async def test_exact_pair_and_whole_probability_population_refuse(mutation: str) -> None:
    plan, a, b, policy = setup(count=4, probabilities=True)
    values = list(b.values)
    if mutation == "gold":
        values[0] = values[0].model_copy(update={"gold": "C"})
    elif mutation == "group":
        values[0] = values[0].model_copy(update={"base_problem_id": "other"})
    elif mutation == "id":
        values[0] = values[0].model_copy(update={"item_id": "other"})
    elif mutation == "duplicate":
        values[1] = values[0]
    elif mutation == "extra":
        values.append(values[0])
    elif mutation == "probability_missing":
        values[0] = values[0].model_copy(update={"probabilities": None})
    elif mutation == "probability_sum":
        values[0] = values[0].model_copy(update={"probabilities": (0.1, 0.1, 0.1)})
    b.values = tuple(values)
    if mutation not in ("digest", "invalid_counts"):
        digest = hashlib.sha256(
            b"".join(paired_cluster_row_bytes(row) for row in b.values)
        ).hexdigest()
        b.meta = b.meta.model_copy(
            update={"condition": b.meta.condition.model_copy(update={"records_sha256": digest})}
        )
        plan = plan.model_copy(update={"candidate": b.meta})
    elif mutation == "digest":
        b.meta = b.meta.model_copy(
            update={"condition": b.meta.condition.model_copy(update={"records_sha256": "d" * 64})}
        )
        plan = plan.model_copy(update={"candidate": b.meta})
    else:
        changed = b.meta.condition.counts.model_copy(
            update={"invalid_sent": 0, "invalid_origin_unknown": 1, "rows_model_sent": None}
        )
        b.meta = b.meta.model_copy(
            update={"condition": b.meta.condition.model_copy(update={"counts": changed})}
        )
        plan = plan.model_copy(update={"candidate": b.meta})
    with pytest.raises(ClusterAnalysisError):
        await run((plan, a, b, policy))
    assert b.closed == 1


@pytest.mark.parametrize("when", [3, 4, 5])
async def test_current_revocation_never_emits_an_undefined_receipt(when: int) -> None:
    parts = setup()
    parts[3].deny_at = when
    with pytest.raises(ClusterAnalysisError, match="denied"):
        await run(parts)


async def test_metadata_and_registration_changes_refuse_after_reads() -> None:
    parts = setup()
    parts[1].mutate_metadata_at = 2
    with pytest.raises(ClusterAnalysisError, match="source_changed"):
        await run(parts)
    parts = setup()
    parts[3].mutate_registration_at = 3
    with pytest.raises(ClusterAnalysisError, match="source_changed"):
        await run(parts)


async def test_source_byte_limit_and_forged_plan_fail_before_additional_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import strata_forge.pipelines.cluster_analysis as module

    parts = setup()
    monkeypatch.setattr(module, "MAX_SOURCE_BYTES", 1)
    with pytest.raises(ClusterAnalysisError, match="limit"):
        await run(parts)
    assert parts[1].reads == 1
    assert parts[1].closed == 1
    assert parts[2].opens == 0
    parts = setup()
    bad = parts[0].model_copy(update={"schema_version": True})
    with pytest.raises(ClusterAnalysisError):
        await run((bad, *parts[1:]))
    assert parts[1].metadata_calls == 0


async def test_repeated_cancellation_drains_read_and_cleanup_fault_cannot_replace_it() -> None:
    parts = setup()
    source = parts[1]
    source.read_hold = asyncio.Event()
    source.close_error = True
    task = asyncio.create_task(run(parts))
    await source.entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    source.read_hold.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert source.closed == 1
    assert parts[2].opens == 0


async def test_policy_cancellation_drains_without_opening_sources() -> None:
    parts = setup()
    policy = parts[3]
    policy.hold = asyncio.Event()
    task = asyncio.create_task(run(parts))
    await policy.entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    policy.hold.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert parts[1].opens == parts[2].opens == 0


async def test_native_chunk_cancellation_drains_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    parts = setup(count=4)
    entered, release = threading.Event(), threading.Event()
    original = BootstrapState.advance

    def held(self: BootstrapState) -> None:
        entered.set()
        release.wait(5)
        original(self)

    monkeypatch.setattr(BootstrapState, "advance", held)
    task = asyncio.create_task(run(parts))
    assert await asyncio.to_thread(entered.wait, 3)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert parts[1].closed == parts[2].closed == 1


async def test_soft_budget_overrun_after_one_chunk_is_not_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import strata_forge.pipelines.cluster_analysis as module

    parts = setup(count=4)
    clock = [0.0]
    monkeypatch.setattr(module, "_CLOCK", lambda: clock[0])
    original = BootstrapState.advance

    def overrun(self: BootstrapState) -> None:
        original(self)
        clock[0] = 31.0

    monkeypatch.setattr(BootstrapState, "advance", overrun)
    with pytest.raises(ClusterAnalysisError, match="deadline"):
        await run(parts)


@pytest.mark.parametrize("value", [{}, [], "unknown", True, None])
def test_error_selector_is_content_free_and_has_no_coercion(value: object) -> None:
    assert str(ClusterAnalysisError(cast("str", value))) == "cluster_analysis_invalid"


async def test_hostile_exact_error_args_do_not_execute_string_hooks() -> None:
    calls: list[str] = []

    class Raw:
        def __str__(self) -> str:
            calls.append("str")
            raise AssertionError("no raw diagnostics")

    class BadPolicy(Policy):
        async def decide(self, request: ClusterPolicyRequest) -> CurrentClusterDecision:
            error = ClusterAnalysisError()
            error.args = (Raw(),)
            raise error

    plan, a, b, _ = setup()
    with pytest.raises(ClusterAnalysisError, match="failed") as caught:
        await run((plan, a, b, BadPolicy()))
    assert not calls
    assert caught.value.__context__ is caught.value.__cause__ is None


@pytest.mark.parametrize(
    "patch",
    [
        {"plan_sha256": "f" * 64},
        {"baseline_observed_rows": 0},
        {"baseline_probability_rows": 0},
        {"status": "complete"},
    ],
)
async def test_receipt_mutations_do_not_validate(patch: dict[str, object]) -> None:
    receipt = await run(setup())
    data = receipt.model_dump(mode="python")
    data.update(patch)
    if "baseline_probability_rows" in patch:
        data["baseline_probability_rows"] = 2
    with pytest.raises(ModelValidationError):
        PairedClusterReceipt.model_validate(data)


async def test_policy_rechecks_during_full_arm_read_before_candidate_open() -> None:
    plan, a, b, policy = setup()
    values = tuple(a.values[0].model_copy(update={"item_id": f"row-{i}"}) for i in range(1025))
    target = plan.baseline.target.model_copy(update={"expected_metric_rows": 1025})
    counts = plan.baseline.condition.counts.model_copy(
        update={
            "rows_in": 1027,
            "rows_scored_success": 1025,
            "rows_metric_population": 1025,
            "rows_model_sent": 1025,
        }
    )
    digest = hashlib.sha256(b"".join(paired_cluster_row_bytes(row) for row in values)).hexdigest()
    for source in (a, b):
        source.values = values
        source.meta = source.meta.model_copy(
            update={
                "target": target,
                "condition": source.meta.condition.model_copy(
                    update={"counts": counts, "records_sha256": digest}
                ),
            }
        )
    plan = plan.model_copy(update={"baseline": a.meta, "candidate": b.meta})
    policy.deny_at = 3
    with pytest.raises(ClusterAnalysisError, match="denied"):
        await run((plan, a, b, policy))
    assert a.reads == 1024
    assert a.closed == 1
    assert b.opens == 0


async def test_revoke_after_a_real_100_replicate_chunk_cannot_return_prefix() -> None:
    parts = setup(count=4)
    parts[3].deny_at = 6
    with pytest.raises(ClusterAnalysisError, match="denied"):
        await run(parts)
    assert parts[1].closed == parts[2].closed == 1


async def test_explicit_integer_gold_mapping_can_pair_with_string_gold() -> None:
    from strata_forge.pipelines.cluster_analysis_spec import ClusterGoldIndex

    plan, a, b, policy = setup()
    target = plan.baseline.target.model_copy(
        update={"gold_index_map": (ClusterGoldIndex(index=0, label="A"),)}
    )
    a.values = (a.values[0].model_copy(update={"gold": 0}),)
    for source in (a, b):
        digest = hashlib.sha256(
            b"".join(paired_cluster_row_bytes(row) for row in source.values)
        ).hexdigest()
        source.meta = source.meta.model_copy(
            update={
                "target": target,
                "condition": source.meta.condition.model_copy(update={"records_sha256": digest}),
            }
        )
    receipt = await run(
        (plan.model_copy(update={"baseline": a.meta, "candidate": b.meta}), a, b, policy)
    )
    assert receipt.result.paired_rows == 1
    assert receipt.baseline_records_sha256 != receipt.candidate_records_sha256


async def test_partial_probability_coverage_is_only_permitted_for_hard_metrics() -> None:
    plan, a, b, policy = setup(count=4, probabilities=True)
    b.values = (b.values[0].model_copy(update={"probabilities": None}), *b.values[1:])
    digest = hashlib.sha256(b"".join(paired_cluster_row_bytes(row) for row in b.values)).hexdigest()
    b.meta = b.meta.model_copy(
        update={
            "condition": b.meta.condition.model_copy(
                update={
                    "records_sha256": digest,
                    "probability_kind": "restricted_softmax",
                    "calibration_state": "declared_applied",
                    "calibration_receipt_sha256": SHA,
                }
            )
        }
    )
    plan = plan.model_copy(update={"candidate": b.meta, "profile": profile(False)})
    receipt = await run((plan, a, b, policy))
    assert receipt.baseline_probability_rows == 4
    assert receipt.candidate_probability_rows == 3
    assert tuple(metric.metric for metric in receipt.result.metrics) == (
        "accuracy",
        "macro_f1",
        "invalid_rate",
    )
    assert receipt.plan.candidate.condition.calibration_state == "declared_applied"


async def test_incompatible_probability_family_is_refused_before_metadata() -> None:
    plan, a, b, policy = setup(probabilities=True)
    b.meta = b.meta.model_copy(
        update={
            "condition": b.meta.condition.model_copy(
                update={"probability_kind": "restricted_softmax"}
            )
        }
    )
    with pytest.raises(ClusterAnalysisError):
        await run((plan.model_copy(update={"candidate": b.meta}), a, b, policy))
    assert a.metadata_calls == b.metadata_calls == 0


def test_error_selector_hostile_hash_eq_string_and_length_are_never_used() -> None:
    calls: list[str] = []

    class Hostile(str):
        def __hash__(self) -> int:
            calls.append("hash")
            raise AssertionError("no raw callbacks")

        def __eq__(self, other: object) -> bool:
            calls.append("eq")
            raise AssertionError("no raw callbacks")

        def __len__(self) -> int:
            calls.append("len")
            raise AssertionError("no raw callbacks")

    assert (
        str(ClusterAnalysisError(Hostile("cluster_analysis_denied"))) == "cluster_analysis_invalid"
    )
    assert calls == []


async def test_receipt_policy_registration_mutation_cannot_preserve_old_bindings() -> None:
    receipt = await run(setup())
    data = receipt.model_dump(mode="python")
    final = receipt.final_decision
    assert final.registration is not None
    altered = final.model_copy(
        update={
            "registration": final.registration.model_copy(
                update={"candidate_source_sha256": "d" * 64}
            )
        }
    )
    data["final_decision"] = altered
    data["final_decision_sha256"] = altered.digest()
    with pytest.raises(ModelValidationError):
        PairedClusterReceipt.model_validate(data)


async def test_policy_cannot_change_the_private_request_anchor() -> None:
    class MutatesRequest(Policy):
        async def decide(self, request: ClusterPolicyRequest) -> CurrentClusterDecision:
            object.__setattr__(request, "plan_sha256", "f" * 64)
            return await super().decide(request)

    plan, a, b, _ = setup()
    with pytest.raises(ClusterAnalysisError, match="denied"):
        await run((plan, a, b, MutatesRequest()))
    assert a.opens == b.opens == 0


async def test_unknown_metaclass_does_not_run_equality_in_plan_reconstruction() -> None:
    calls: list[str] = []

    class Meta(type):
        def __eq__(cls, other: object) -> bool:
            calls.append("eq")
            return False

        __hash__ = type.__hash__

    class Foreign(metaclass=Meta):
        pass

    plan, a, b, policy = setup()
    with pytest.raises(ClusterAnalysisError):
        await run((plan.model_copy(update={"profile": Foreign()}), a, b, policy))
    assert calls == []
    assert a.metadata_calls == b.metadata_calls == 0
