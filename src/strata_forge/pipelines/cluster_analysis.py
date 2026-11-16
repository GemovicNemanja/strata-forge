"""Local current-policy paired cluster analysis; no issuer, model call, files or worker."""

from __future__ import annotations

import asyncio
import hashlib
import math
import platform
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Protocol

from strata_forge.metrics.cluster_bootstrap import (
    PairedClusterObservation,
    _ClusterBootstrapState,  # pyright: ignore[reportPrivateUsage] — fixed incremental state owned by the numerical leaf
)
from strata_forge.pipelines.cluster_analysis_spec import (
    MAX_METADATA_BYTES,
    MAX_RECORD_BYTES,
    MAX_SOURCE_BYTES,
    ClusterAnalysisError,
    ClusterPolicyRequest,
    CurrentClusterDecision,
    PairedClusterArmRow,
    PairedClusterPlanV1,
    PairedClusterReceipt,
    PairedClusterSourceMetadata,
    cluster_decision_identity,
    revalidate_cluster_value,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

_CLOCK = time.perf_counter


class PairedClusterStream(Protocol):
    def __aiter__(self) -> PairedClusterStream: ...
    async def __anext__(self) -> PairedClusterArmRow: ...
    async def aclose(self) -> None: ...


class PairedClusterSource(Protocol):
    """Trusted installed adapter; metadata may not materialize rows or run a model."""

    async def metadata(self) -> PairedClusterSourceMetadata: ...
    def rows(self) -> PairedClusterStream: ...


class PairedClusterPolicy(Protocol):
    """Trusted local current resolver/registry, never serialized caller authority."""

    async def decide(self, request: ClusterPolicyRequest) -> CurrentClusterDecision: ...


async def _owned[T](awaitable: Awaitable[T]) -> T:
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.done() and not task.cancelled():
            task.exception()
        raise


def _row_bytes(row: PairedClusterArmRow) -> bytes:
    if type(row) is not PairedClusterArmRow:
        raise ClusterAnalysisError("cluster_analysis_input_invalid")
    row = revalidate_cluster_value(row)
    normalized = row.model_copy(
        update={
            "probabilities": None
            if row.probabilities is None
            else tuple(float(value) for value in row.probabilities)
        }
    )
    content = normalized.canonical() + b"\n"
    if len(content) > MAX_RECORD_BYTES:
        raise ClusterAnalysisError("cluster_analysis_limit")
    return content


def paired_cluster_row_bytes(row: PairedClusterArmRow) -> bytes:
    """Fixed normalized source codec; hashes never grant access."""
    result = None
    with suppress(Exception):
        result = _row_bytes(row)
    if result is None:
        raise ClusterAnalysisError("cluster_analysis_input_invalid")
    return result


class _Run:
    def __init__(
        self,
        plan: PairedClusterPlanV1,
        baseline: PairedClusterSource,
        candidate: PairedClusterSource,
        policy: PairedClusterPolicy,
    ) -> None:
        self.plan, self.sources, self.policy = plan, (baseline, candidate), policy
        self.start = self.last = _CLOCK()
        if type(self.start) not in (int, float) or not math.isfinite(self.start):
            raise ClusterAnalysisError("cluster_analysis_failed")
        self.request = ClusterPolicyRequest(
            schema_version=1,
            purpose="paired_cluster_uncertainty",
            plan_sha256=plan.digest(),
            baseline=plan.baseline,
            candidate=plan.candidate,
        )
        self.checks = 0
        self.initial: CurrentClusterDecision | None = None
        self.final: CurrentClusterDecision | None = None
        self.identity: bytes | None = None
        self.chain = hashlib.sha256(b"strata.analysis.cluster.decisions.v1\0")
        self.total_bytes = 0
        self.total_rows = 0

    def budget(self) -> None:
        now = _CLOCK()
        if type(now) not in (int, float) or not math.isfinite(now) or now < self.last:
            raise ClusterAnalysisError("cluster_analysis_failed")
        self.last = now
        if (now - self.start) * 1000 > self.plan.soft_budget_ms:
            raise ClusterAnalysisError("cluster_analysis_deadline")

    async def metadata(self) -> None:
        for source, expected in zip(
            self.sources, (self.plan.baseline, self.plan.candidate), strict=True
        ):
            actual = await _owned(source.metadata())
            if (
                type(actual) is not PairedClusterSourceMetadata
                or revalidate_cluster_value(actual) != expected
            ):
                raise ClusterAnalysisError("cluster_analysis_source_changed")
            self.budget()

    async def current(self) -> None:
        self.budget()
        value = await _owned(self.policy.decide(revalidate_cluster_value(self.request)))
        if type(value) is not CurrentClusterDecision:
            raise ClusterAnalysisError("cluster_analysis_denied")
        decision = revalidate_cluster_value(value)
        identity = cluster_decision_identity(self.request, decision)
        if identity is None:
            raise ClusterAnalysisError("cluster_analysis_denied")
        if self.identity is not None and identity != self.identity:
            raise ClusterAnalysisError("cluster_analysis_source_changed")
        self.identity = identity
        self.checks += 1
        self.chain.update(bytes.fromhex(decision.digest()))
        if self.initial is None:
            self.initial = decision
        self.final = decision
        self.budget()


async def _read_arm(
    run: _Run, source: PairedClusterSource, metadata: PairedClusterSourceMetadata
) -> tuple[dict[str, tuple[PairedClusterArmRow, str]], int, int]:
    rows: dict[str, tuple[PairedClusterArmRow, str]] = {}
    size = probabilities = 0
    digest = hashlib.sha256()
    counts = metadata.condition.counts
    failures = sent = invalid_sent = invalid_not_sent = invalid_unknown = 0
    any_unknown = False
    indexes = {entry.index: entry.label for entry in metadata.target.gold_index_map}
    stream = source.rows()
    cancelled = False
    try:
        while True:
            run.budget()
            try:
                row = await _owned(anext(stream))
            except StopAsyncIteration:
                break
            if len(rows) >= metadata.target.expected_metric_rows:
                raise ClusterAnalysisError("cluster_analysis_population_mismatch")
            if type(row) is not PairedClusterArmRow:
                raise ClusterAnalysisError("cluster_analysis_input_invalid")
            row = revalidate_cluster_value(row)
            content = _row_bytes(row)
            if run.total_bytes + len(content) > MAX_SOURCE_BYTES:
                raise ClusterAnalysisError("cluster_analysis_limit")
            label = row.gold if isinstance(row.gold, str) else indexes.get(row.gold)
            if (
                row.item_id in rows
                or label not in metadata.target.labels
                or (row.pred is not None and row.pred not in metadata.target.labels)
            ):
                raise ClusterAnalysisError("cluster_analysis_input_invalid")
            vector = row.probabilities
            if vector is not None:
                if (
                    metadata.condition.probability_kind is None
                    or len(vector) != len(metadata.target.labels)
                    or abs(math.fsum(vector) - 1) > 1e-6
                ):
                    raise ClusterAnalysisError("cluster_analysis_input_invalid")
                probabilities += 1
            elif run.plan.profile.probability_kind is not None:
                raise ClusterAnalysisError("cluster_analysis_input_invalid")
            failures += row.outcome == "invalid"
            sent += row.model_sent is True
            any_unknown |= row.model_sent is None
            invalid_sent += row.outcome == "invalid" and row.model_sent is True
            invalid_not_sent += row.outcome == "invalid" and row.model_sent is False
            invalid_unknown += row.outcome == "invalid" and row.model_sent is None
            rows[row.item_id] = (row, label)
            size += len(content)
            run.total_bytes += len(content)
            digest.update(content)
            run.total_rows += 1
            if run.total_rows % 1024 == 0:
                await run.current()
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        if cancelled:
            with suppress(Exception):
                await _owned(stream.aclose())
        else:
            await _owned(stream.aclose())
    if (
        len(rows) != metadata.target.expected_metric_rows
        or digest.hexdigest() != metadata.condition.records_sha256
        or failures != counts.rows_failed_or_invalid
        or len(rows) - failures != counts.rows_scored_success
        or (None if any_unknown else sent) != counts.rows_model_sent
        or (invalid_sent, invalid_not_sent, invalid_unknown)
        != (counts.invalid_sent, counts.invalid_not_sent, counts.invalid_origin_unknown)
    ):
        raise ClusterAnalysisError("cluster_analysis_population_mismatch")
    return rows, size, probabilities


def _join(
    plan: PairedClusterPlanV1,
    baseline: dict[str, tuple[PairedClusterArmRow, str]],
    candidate: dict[str, tuple[PairedClusterArmRow, str]],
) -> tuple[PairedClusterObservation, ...]:
    if set(baseline) != set(candidate):
        raise ClusterAnalysisError("cluster_analysis_population_mismatch")
    result: list[PairedClusterObservation] = []
    labels = plan.profile.labels
    for key in sorted(baseline, key=lambda key: key.encode("utf-8")):
        a, ag = baseline[key]
        b, bg = candidate[key]
        if ag != bg or a.base_problem_id != b.base_problem_id:
            raise ClusterAnalysisError("cluster_analysis_population_mismatch")
        result.append(
            PairedClusterObservation(
                item_id=key,
                base_problem_id=a.base_problem_id,
                gold_index=labels.index(ag),
                baseline_pred_index=None if a.pred is None else labels.index(a.pred),
                candidate_pred_index=None if b.pred is None else labels.index(b.pred),
                baseline_probabilities=a.probabilities
                if plan.profile.probability_kind is not None
                else None,
                candidate_probabilities=b.probabilities
                if plan.profile.probability_kind is not None
                else None,
            )
        )
    return tuple(result)


def _numpy_version() -> str:
    import numpy as np

    return np.__version__


async def _analyse(
    plan: PairedClusterPlanV1,
    baseline: PairedClusterSource,
    candidate: PairedClusterSource,
    policy: PairedClusterPolicy,
) -> PairedClusterReceipt:
    if any(
        len(value.canonical()) > MAX_METADATA_BYTES
        for value in (plan, plan.baseline, plan.candidate)
    ):
        raise ClusterAnalysisError("cluster_analysis_limit")
    run = _Run(plan, baseline, candidate, policy)
    await run.metadata()
    await run.current()
    version = await _owned(asyncio.to_thread(_numpy_version))
    await run.current()
    arms: list[tuple[dict[str, tuple[PairedClusterArmRow, str]], int, int]] = []
    for source, metadata in zip(run.sources, (plan.baseline, plan.candidate), strict=True):
        arms.append(await _read_arm(run, source, metadata))
    await run.current()
    observations = await _owned(asyncio.to_thread(_join, plan, arms[0][0], arms[1][0]))
    state = await _owned(asyncio.to_thread(_ClusterBootstrapState, observations, plan.profile))
    await run.current()
    while not state.done:
        await run.current()
        await _owned(asyncio.to_thread(state.advance))
    result = await _owned(asyncio.to_thread(state.result))
    result = revalidate_cluster_value(result)
    await run.metadata()
    await run.current()
    if run.initial is None or run.final is None:
        raise ClusterAnalysisError("cluster_analysis_failed")
    receipt = PairedClusterReceipt(
        schema_version=1,
        status=result.status,
        plan=plan,
        plan_sha256=plan.digest(),
        baseline_source_sha256=plan.baseline.digest(),
        candidate_source_sha256=plan.candidate.digest(),
        baseline_observed_rows=len(arms[0][0]),
        candidate_observed_rows=len(arms[1][0]),
        baseline_observed_bytes=arms[0][1],
        candidate_observed_bytes=arms[1][1],
        baseline_records_sha256=plan.baseline.condition.records_sha256,
        candidate_records_sha256=plan.candidate.condition.records_sha256,
        baseline_probability_rows=arms[0][2],
        candidate_probability_rows=arms[1][2],
        result=result,
        initial_decision=run.initial,
        final_decision=run.final,
        initial_decision_sha256=run.initial.digest(),
        final_decision_sha256=run.final.digest(),
        decision_chain_sha256=run.chain.hexdigest(),
        policy_checks=run.checks,
        elapsed_to_final_policy_ms=(run.last - run.start) * 1000,
        python_version=platform.python_version(),
        numpy_version=version,
        implementation_attestation="unattested_local_sdk",
    )
    if len(receipt.canonical()) > MAX_METADATA_BYTES:
        raise ClusterAnalysisError("cluster_analysis_limit")
    run.budget()
    return revalidate_cluster_value(receipt)


async def paired_registered_cluster_uncertainty(
    plan: PairedClusterPlanV1,
    *,
    baseline: PairedClusterSource | None,
    candidate: PairedClusterSource | None,
    policy: PairedClusterPolicy | None,
) -> PairedClusterReceipt:
    """Analyse only an exact registered unsealed pair through explicit trusted local adapters."""
    result = None
    code = "cluster_analysis_failed"
    missing = False
    try:
        if (
            type(plan) is not PairedClusterPlanV1
            or baseline is None
            or candidate is None
            or policy is None
        ):
            raise ClusterAnalysisError("cluster_analysis_unavailable")
        result = await _analyse(revalidate_cluster_value(plan), baseline, candidate, policy)
    except asyncio.CancelledError:
        raise
    except ImportError:
        missing = True
    except Exception as error:
        if (
            type(error) is ClusterAnalysisError
            and type(error.args) is tuple
            and len(error.args) == 1
            and type(error.args[0]) is str
        ):
            code = error.args[0]
    if missing:
        raise ImportError("Install 'strata-forge[scoring]' for paired cluster uncertainty.")
    if result is None:
        raise ClusterAnalysisError(code)
    return result
