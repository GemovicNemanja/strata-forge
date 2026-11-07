"""Local registered-split temperature fitting with no model, loader, writer or issuer."""

from __future__ import annotations

import asyncio
import hashlib
import math
import platform
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic import BaseModel

from strata_forge.metrics import (
    CalibrationScores,
    MetricValue,
    TemperatureCalibration,
    brier,
    nll,
    temperature_scale,
)
from strata_forge.metrics.temperature_fit import (
    MAX_LOGIT_ABS,
    _FitState,  # pyright: ignore[reportPrivateUsage] — fixed bounded state owned by fit core
)
from strata_forge.pipelines.evaluation_analysis_spec import (
    MAX_ANALYSIS_RECEIPT_BYTES,
    MAX_ANALYSIS_RECORD_BYTES,
    MAX_ANALYSIS_SOURCE_BYTES,
    MAX_ANALYSIS_WORK_BYTES,
    EvaluationAnalysisError,
    FitPointMetrics,
    FitPolicyRequest,
    RegisteredFitDecision,
    TemperatureFitPlanV1,
    TemperatureFitReceipt,
    TemperatureFitRow,
    TemperatureFitSourceMetadata,
    calibration_component_digest,
)

_CLOCK = time.perf_counter
if TYPE_CHECKING:
    from collections.abc import Awaitable


class TemperatureFitStream(Protocol):
    """Trusted typed row stream; cancellation closes it after its outstanding read drains."""

    def __aiter__(self) -> TemperatureFitStream: ...
    async def __anext__(self) -> TemperatureFitRow: ...
    async def aclose(self) -> None: ...


class TemperatureFitSource(Protocol):
    """Installed source adapter; metadata cannot materialize rows or perform a model call."""

    async def metadata(self) -> TemperatureFitSourceMetadata: ...
    def rows(self) -> TemperatureFitStream: ...


class TemperatureFitPolicy(Protocol):
    """Installed current read-and-registry policy, not a caller-supplied serialized grant."""

    async def decide(self, request: FitPolicyRequest) -> RegisteredFitDecision: ...


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


def _copy[Model: BaseModel](value: Model) -> Model:
    def raw(item: object, depth: int = 0) -> object:
        if depth > 16:
            raise EvaluationAnalysisError()
        if isinstance(item, BaseModel):
            if item.model_extra or set(item.__dict__) != set(type(item).model_fields):
                raise EvaluationAnalysisError()
            return {key: raw(getattr(item, key), depth + 1) for key in type(item).model_fields}
        if type(item) is tuple:
            values = cast("tuple[object, ...]", item)
            if len(values) > 256:
                raise EvaluationAnalysisError()
            return tuple(raw(child, depth + 1) for child in values)
        if item is None or type(item) in (str, int, float, bool):
            return item
        raise EvaluationAnalysisError()

    return type(value).model_validate(raw(value))


def _row_bytes(row: TemperatureFitRow) -> bytes:
    if type(row) is not TemperatureFitRow:
        raise EvaluationAnalysisError("analysis_input_invalid")
    row = _copy(row)
    if (
        (
            type(row.gold) is str
            and (not row.gold or len(row.gold) > 64 or len(row.gold.encode("utf-8")) > 64)
        )
        or (type(row.gold) is int and abs(row.gold) > 2**53 - 1)
        or any(
            type(value) not in (int, float) or not -MAX_LOGIT_ABS <= value <= MAX_LOGIT_ABS
            for value in row.logits
        )
    ):
        raise EvaluationAnalysisError("analysis_input_invalid")
    normalized = TemperatureFitRow(
        item_id=row.item_id, gold=row.gold, logits=tuple(float(v) for v in row.logits)
    )
    content = normalized.canonical() + b"\n"
    if len(content) > MAX_ANALYSIS_RECORD_BYTES:
        raise EvaluationAnalysisError("analysis_limit")
    return content


def temperature_fit_row_bytes(row: TemperatureFitRow) -> bytes:
    """Return the fixed normalized row-codec bytes; this is not source authorization."""
    result = None
    with suppress(Exception):
        result = _row_bytes(row)
    if result is None:
        raise EvaluationAnalysisError("analysis_input_invalid")
    return result


class _Run:
    def __init__(
        self, plan: TemperatureFitPlanV1, source: TemperatureFitSource, policy: TemperatureFitPolicy
    ) -> None:
        self.plan, self.source, self.policy = plan, source, policy
        self.start = self.last = _CLOCK()
        if type(self.start) not in (int, float) or not math.isfinite(self.start):
            raise EvaluationAnalysisError("analysis_failed")
        self.request = FitPolicyRequest(
            schema_version=1,
            purpose="temperature_calibration_fit",
            plan_sha256=plan.digest(),
            source=plan.source,
            registration_id=plan.registration_id,
            registration_revision=plan.registration_revision,
        )
        self.checks = 0
        self.chain = hashlib.sha256(b"strata-analysis-decisions-v1\0")
        self.initial = self.final = ""
        self.first_decision: RegisteredFitDecision | None = None
        self.last_decision: RegisteredFitDecision | None = None
        self.identity: bytes | None = None

    def budget(self) -> None:
        now = _CLOCK()
        if type(now) not in (int, float) or not math.isfinite(now) or now < self.last:
            raise EvaluationAnalysisError("analysis_failed")
        self.last = now
        if (now - self.start) * 1000 > self.plan.soft_budget_ms:
            raise EvaluationAnalysisError("analysis_deadline")

    async def metadata(self) -> None:
        value = await _owned(self.source.metadata())
        if type(value) is not TemperatureFitSourceMetadata or _copy(value) != self.plan.source:
            raise EvaluationAnalysisError("analysis_source_changed")
        self.budget()

    async def current(self) -> None:
        self.budget()
        decision = await _owned(self.policy.decide(self.request))
        if type(decision) is not RegisteredFitDecision:
            raise EvaluationAnalysisError("analysis_denied")
        decision = _copy(decision)
        registration = decision.registration
        if (
            decision.request_sha256 != self.request.digest()
            or decision.status != "allow_unsealed"
            or registration is None
            or registration.role_coverage != "known"
            or registration.registration_id != self.plan.registration_id
            or registration.registration_revision != self.plan.registration_revision
            or registration.expected_rows != self.plan.source.expected_rows
            or registration.records_sha256 != self.plan.source.records_sha256
            or registration.canonical_config != self.plan.source.config
            or len(set(registration.roles)) != len(registration.roles)
            or "calibration_fit" not in registration.roles
            or any(role in registration.roles for role in ("held_out", "gate", "test"))
            or len(set(registration.split_aliases)) != len(registration.split_aliases)
            or self.plan.source.split not in registration.split_aliases
            or registration.canonical_split not in registration.split_aliases
            or any(
                name.casefold() == "ho_dev"
                for name in (
                    self.plan.source.split,
                    registration.canonical_split,
                    *registration.split_aliases,
                )
            )
        ):
            raise EvaluationAnalysisError("analysis_denied")
        expected = {(ref.repo_type, ref.ref) for ref in self.plan.source.input_refs()}
        if (
            len(decision.reads) != len(expected)
            or {(read.requested.repo_type, read.requested.ref) for read in decision.reads}
            != expected
        ):
            raise EvaluationAnalysisError("analysis_denied")
        dataset = None
        for read in decision.reads:
            if (
                read.custody != "unsealed"
                or read.requested.repo_type != read.canonical_ref.repo_type
                or read.requested.ref.rsplit("@", 1)[1] != read.canonical_ref.ref.rsplit("@", 1)[1]
            ):
                raise EvaluationAnalysisError("analysis_denied")
            if read.requested == self.plan.source.dataset:
                dataset = read.canonical_ref
        if (
            dataset != registration.canonical_dataset
            or registration.canonical_dataset.repo_type != "dataset"
        ):
            raise EvaluationAnalysisError("analysis_denied")
        identity = b"\0".join(
            [registration.canonical(), *(read.canonical_ref.canonical() for read in decision.reads)]
        )
        if self.identity is not None and identity != self.identity:
            raise EvaluationAnalysisError("analysis_source_changed")
        self.identity = identity
        digest = decision.digest()
        self.checks += 1
        self.chain.update(bytes.fromhex(digest))
        if not self.initial:
            self.initial = digest
            self.first_decision = decision
        self.final = digest
        self.last_decision = decision
        self.budget()


async def _read(
    run: _Run,
) -> tuple[tuple[tuple[object, ...], ...], tuple[int, ...], tuple[str, ...], int]:
    source = run.plan.source
    stream = run.source.rows()
    logits: list[tuple[object, ...]] = []
    gold_indexes: list[int] = []
    gold_labels: list[str] = []
    ids: set[str] = set()
    digest = hashlib.sha256()
    size = 0
    index_map = {item.index: item.label for item in source.gold_index_map}
    cancelled = False
    try:
        while True:
            run.budget()
            try:
                row = await _owned(anext(stream))
            except StopAsyncIteration:
                break
            if len(logits) >= source.expected_rows:
                raise EvaluationAnalysisError("analysis_population_mismatch")
            if type(row) is not TemperatureFitRow:
                raise EvaluationAnalysisError("analysis_input_invalid")
            row = _copy(row)
            content = _row_bytes(row)
            if size + len(content) > MAX_ANALYSIS_SOURCE_BYTES:
                raise EvaluationAnalysisError("analysis_limit")
            if len(row.logits) != len(source.labels) or row.item_id in ids:
                raise EvaluationAnalysisError("analysis_input_invalid")
            label = row.gold if isinstance(row.gold, str) else index_map.get(row.gold)
            if label is None or label not in source.labels:
                raise EvaluationAnalysisError("analysis_input_invalid")
            ids.add(row.item_id)
            size += len(content)
            digest.update(content)
            logits.append(tuple(float(value) for value in row.logits))
            gold_indexes.append(source.labels.index(label))
            gold_labels.append(label)
            if len(logits) % 1024 == 0:
                await run.current()
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        if cancelled:
            # An adapter cleanup fault must not turn a cancelled fit into an ordinary failure.
            with suppress(Exception):
                await _owned(stream.aclose())
        else:
            await _owned(stream.aclose())
    if len(logits) != source.expected_rows or digest.hexdigest() != source.records_sha256:
        raise EvaluationAnalysisError("analysis_population_mismatch")
    # Fixed native point helpers may allocate several bounded arrays per chunk, never per replicate.
    if len(logits) * (len(source.labels) + 1) * 8 * 4 > MAX_ANALYSIS_WORK_BYTES:
        raise EvaluationAnalysisError("analysis_limit")
    return tuple(logits), tuple(gold_indexes), tuple(gold_labels), size


def _point(
    logits: tuple[tuple[object, ...], ...],
    gold: tuple[str, ...],
    plan: TemperatureFitPlanV1,
    temperature: float,
) -> FitPointMetrics:
    probabilities = temperature_scale(logits, temperature=temperature)
    return FitPointMetrics(
        nll=nll(
            gold,
            probabilities,
            labels=plan.source.labels,
            probability_kind=plan.source.probability_kind,
        ),
        brier=brier(
            gold,
            probabilities,
            labels=plan.source.labels,
            probability_kind=plan.source.probability_kind,
        ),
    )


async def _points(
    run: _Run, logits: tuple[tuple[object, ...], ...], gold: tuple[str, ...], temperature: float
) -> FitPointMetrics:
    chunks: list[FitPointMetrics] = []
    for start in range(0, len(logits), 1024):
        await run.current()
        chunk = await _owned(
            asyncio.to_thread(
                _point,
                logits[start : start + 1024],
                gold[start : start + 1024],
                run.plan,
                temperature,
            )
        )
        if type(chunk) is not FitPointMetrics:
            raise EvaluationAnalysisError("analysis_failed")
        chunk = _copy(chunk)
        for metric in (chunk.nll, chunk.brier):
            if (
                metric.value is None
                or metric.n != len(logits[start : start + 1024])
                or metric.excluded != 0
                or metric.probability_kind != run.plan.source.probability_kind
            ):
                raise EvaluationAnalysisError("analysis_failed")
        chunks.append(chunk)

    def combined(name: Literal["nll", "brier"]) -> MetricValue:
        values = [chunk.nll if name == "nll" else chunk.brier for chunk in chunks]
        return MetricValue(
            value=math.fsum(cast("float", value.value) * value.n for value in values) / len(logits),
            n=len(logits),
            probability_kind=run.plan.source.probability_kind,
            clip_epsilon=values[0].clip_epsilon,
            clipped=sum(value.clipped for value in values),
            normalization_tolerance=values[0].normalization_tolerance,
        )

    return FitPointMetrics(nll=combined("nll"), brier=combined("brier"))


def _numerical_version() -> str:
    from importlib import import_module
    from importlib.metadata import version

    import_module("numpy")
    return version("numpy")


async def _fit(
    plan: TemperatureFitPlanV1, source: TemperatureFitSource, policy: TemperatureFitPolicy
) -> TemperatureFitReceipt:
    run = _Run(plan, source, policy)
    await run.metadata()
    await run.current()
    numpy_version = await _owned(asyncio.to_thread(_numerical_version))
    await run.current()
    logits, gold, names, size = await _read(run)
    await run.current()
    state = await _owned(asyncio.to_thread(_FitState, logits, gold, plan.bounds))
    while not state.done:
        await run.current()
        await _owned(asyncio.to_thread(state.advance))
    solution = state.solution()
    before = await _points(run, logits, names, 1.0)
    after = await _points(run, logits, names, solution.temperature)
    await run.metadata()
    await run.current()
    component = TemperatureCalibration(
        labels=plan.source.labels,
        temperature=solution.temperature,
        fit_ref=plan.source.dataset.ref,
        config=plan.source.config,
        split=plan.source.split,
        n=len(logits),
        probability_kind=plan.source.probability_kind,
        before=CalibrationScores(nll=before.nll.value, brier=before.brier.value),
        after=CalibrationScores(nll=after.nll.value, brier=after.brier.value),
    )
    if run.first_decision is None or run.last_decision is None:
        raise EvaluationAnalysisError("analysis_failed")
    receipt = TemperatureFitReceipt(
        schema_version=1,
        status="complete",
        plan=plan,
        plan_sha256=plan.digest(),
        source_sha256=plan.source.digest(),
        observed_rows=len(logits),
        observed_bytes=size,
        records_sha256=plan.source.records_sha256,
        solution=solution,
        component=component,
        component_sha256=calibration_component_digest(component),
        before=before,
        after=after,
        policy_checks=run.checks,
        elapsed_to_final_policy_ms=(run.last - run.start) * 1000,
        decision_chain_sha256=run.chain.hexdigest(),
        initial_decision_sha256=run.initial,
        final_decision_sha256=run.final,
        initial_decision=run.first_decision,
        final_decision=run.last_decision,
        python_version=platform.python_version(),
        numpy_version=numpy_version,
    )
    if len(receipt.canonical()) > MAX_ANALYSIS_RECEIPT_BYTES:
        raise EvaluationAnalysisError("analysis_limit")
    run.budget()
    return receipt


async def fit_registered_temperature(
    plan: TemperatureFitPlanV1,
    *,
    source: TemperatureFitSource | None,
    policy: TemperatureFitPolicy | None,
) -> TemperatureFitReceipt:
    """Fit only the exact eligible registered named source through trusted local adapters."""
    result = None
    code = "analysis_invalid"
    missing = False
    try:
        if type(plan) is not TemperatureFitPlanV1 or source is None or policy is None:
            raise EvaluationAnalysisError("analysis_unavailable")
        result = await _fit(_copy(plan), source, policy)
    except asyncio.CancelledError:
        raise
    except ImportError:
        missing = True
    except Exception as error:
        code = str(error) if type(error) is EvaluationAnalysisError else "analysis_failed"
    if missing:
        raise ImportError("Install 'strata-forge[scoring]' for registered temperature fitting.")
    if result is None:
        raise EvaluationAnalysisError(code)
    return result
