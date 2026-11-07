"""Fictional named registry/source boundaries; no installed issuer or scientific data."""

from __future__ import annotations

import asyncio
import hashlib
import math
import threading
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import pytest

from strata_forge.metrics.temperature_fit import TemperatureFitBounds
from strata_forge.pipelines import evaluation_analysis as sdk
from strata_forge.pipelines.evaluation_analysis_spec import (
    AnalysisRef,
    CanonicalAnalysisRead,
    EvaluationAnalysisError,
    FitPolicyRequest,
    FitRegistration,
    RegisteredFitDecision,
    TemperatureFitPlanV1,
    TemperatureFitReceipt,
    TemperatureFitRow,
    TemperatureFitSourceMetadata,
)


def rows() -> tuple[TemperatureFitRow, ...]:
    return tuple(
        TemperatureFitRow(
            item_id=f"PRIVATE_ROW_{i}", gold="A" if i < 3 else "B", logits=(2 * math.log(3), 0.0)
        )
        for i in range(4)
    )


def metadata(data: tuple[TemperatureFitRow, ...] | None = None) -> TemperatureFitSourceMetadata:
    data = rows() if data is None else data
    return TemperatureFitSourceMetadata(
        schema_version=1,
        dataset=AnalysisRef(repo_type="dataset", ref="fictional/fit@" + "a" * 40),
        model=AnalysisRef(repo_type="model", ref="fictional/model@" + "b" * 40),
        tokenizer=AnalysisRef(repo_type="model", ref="fictional/tokenizer@" + "c" * 40),
        predictions=None,
        config=None,
        split="dev",
        id_column="id",
        gold_column="gold",
        labels=("A", "B"),
        gold_index_map=(),
        expected_rows=len(data),
        records_sha256=hashlib.sha256(
            b"".join(sdk.temperature_fit_row_bytes(r) for r in data)
        ).hexdigest(),
        source_receipt_sha256="d" * 64,
        rendering_sha256="e" * 64,
        head_sha256="f" * 64,
        input_kind="exact_class_logits",
        projection="identity",
        selection="complete_named_split",
        row_codec="strata.analysis.fit.rows.v1",
    )


def plan(meta: TemperatureFitSourceMetadata | None = None) -> TemperatureFitPlanV1:
    return TemperatureFitPlanV1(
        schema_version=1,
        protocol="strata.eval.analysis.local.v1",
        mode="temperature_fit",
        source=meta or metadata(),
        registration_id="fictional/fit-v1",
        registration_revision="1",
        bounds=TemperatureFitBounds(minimum=0.5, maximum=4.0),
        soft_budget_ms=10000,
    )


class Source:
    def __init__(
        self, meta: TemperatureFitSourceMetadata, data: tuple[TemperatureFitRow, ...] | None = None
    ) -> None:
        self.meta = meta
        self.data = rows() if data is None else data
        self.started = self.closed = self.read = self.metadata_calls = 0

    async def metadata(self) -> TemperatureFitSourceMetadata:
        self.metadata_calls += 1
        return self.meta

    async def _stream(self) -> AsyncGenerator[TemperatureFitRow]:
        try:
            for row in self.data:
                self.read += 1
                yield row
        finally:
            self.closed += 1

    def rows(self) -> sdk.TemperatureFitStream:
        self.started += 1
        return cast("sdk.TemperatureFitStream", self._stream())


class Policy:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, request: FitPolicyRequest) -> RegisteredFitDecision:
        meta = request.source
        return RegisteredFitDecision(
            schema_version=1,
            request_sha256=request.digest(),
            status="allow_unsealed",
            policy_revision="fictional-v1",
            reads=tuple(
                CanonicalAnalysisRead(requested=ref, canonical_ref=ref, custody="unsealed")
                for ref in meta.input_refs()
            ),
            registration=FitRegistration(
                registration_id=request.registration_id,
                registration_revision=request.registration_revision,
                metadata_sha256="1" * 64,
                canonical_dataset=meta.dataset,
                canonical_config=meta.config,
                canonical_split=meta.split,
                split_aliases=(meta.split,),
                roles=("calibration_fit", "monitoring"),
                role_coverage="known",
                expected_rows=meta.expected_rows,
                records_sha256=meta.records_sha256,
            ),
        )

    async def decide(self, request: FitPolicyRequest) -> RegisteredFitDecision:
        self.calls += 1
        return self.answer(request)


async def fitted() -> TemperatureFitReceipt:
    p = plan()
    source = Source(p.source)
    result = await sdk.fit_registered_temperature(p, source=source, policy=Policy())
    assert source.started == source.closed == 1
    assert source.read == 4
    return result


async def test_registered_analytic_fit_has_exact_binding_and_private_immutable_receipt() -> None:
    result = await fitted()
    assert result.solution.temperature == pytest.approx(2.0, abs=1e-7)
    assert result.after.brier.value == pytest.approx(0.375, abs=1e-12)
    assert result.before.brier.value == pytest.approx(0.42, abs=1e-12)
    assert result.after.nll.n == result.observed_rows == 4
    assert result.before.nll.clip_epsilon == 1e-15
    assert result.policy_checks >= 6
    assert result.plan.digest() == result.plan_sha256
    assert TemperatureFitReceipt.model_validate_json(result.model_dump_json()) == result
    assert "PRIVATE_ROW" not in result.model_dump_json()
    with pytest.raises(ValueError, match="frozen_instance"):
        result.observed_rows = 8


@pytest.mark.parametrize(
    "change",
    [
        "deny",
        "unknown",
        "request",
        "role",
        "role_unknown",
        "alias",
        "canonical_ho_dev",
        "revision",
        "count",
        "records",
        "private",
        "read_missing",
        "read_domain",
    ],
)
async def test_registry_or_read_refusal_precedes_iterator(change: str) -> None:
    p = plan()
    source = Source(p.source)

    class Refusal(Policy):
        def answer(self, request: FitPolicyRequest) -> RegisteredFitDecision:
            decision = super().answer(request)
            registration = decision.registration
            assert registration is not None
            if change in ("deny", "unknown"):
                return decision.model_copy(update={"status": change})
            if change == "request":
                return decision.model_copy(update={"request_sha256": "0" * 64})
            if change == "role":
                registration = registration.model_copy(
                    update={"roles": ("calibration_fit", "test")}
                )
            if change == "role_unknown":
                registration = registration.model_copy(update={"role_coverage": "unknown"})
            if change == "alias":
                registration = registration.model_copy(update={"split_aliases": ("dev", "ho_dev")})
            if change == "canonical_ho_dev":
                registration = registration.model_copy(
                    update={"canonical_split": "ho_dev", "split_aliases": ("dev", "ho_dev")}
                )
            if change == "revision":
                registration = registration.model_copy(update={"registration_revision": "2"})
            if change == "count":
                registration = registration.model_copy(update={"expected_rows": 5})
            if change == "records":
                registration = registration.model_copy(update={"records_sha256": "0" * 64})
            reads = decision.reads
            if change == "private":
                reads = (reads[0].model_copy(update={"custody": "protected"}), *reads[1:])
            if change == "read_missing":
                reads = reads[:-1]
            if change == "read_domain":
                reads = (
                    reads[0].model_copy(
                        update={
                            "canonical_ref": AnalysisRef(
                                repo_type="model", ref=reads[0].canonical_ref.ref
                            )
                        }
                    ),
                    *reads[1:],
                )
            return decision.model_copy(update={"registration": registration, "reads": reads})

    with pytest.raises(EvaluationAnalysisError, match=r"^analysis_denied$") as caught:
        await sdk.fit_registered_temperature(p, source=source, policy=Refusal())
    assert source.started == source.read == 0
    assert caught.value.__cause__ is caught.value.__context__ is None


async def test_literal_ho_dev_never_gets_fit_authority_even_from_incorrect_fake_policy() -> None:
    meta = metadata().model_copy(update={"split": "ho_dev"})
    p = plan(meta)
    source = Source(meta)
    with pytest.raises(EvaluationAnalysisError, match=r"^analysis_denied$"):
        await sdk.fit_registered_temperature(p, source=source, policy=Policy())
    assert source.started == 0


@pytest.mark.parametrize("change", ["short", "extra", "duplicate", "gold", "width", "digest"])
async def test_complete_population_and_exact_hash_are_required(change: str) -> None:
    p = plan()
    data = rows()
    if change == "short":
        data = data[:-1]
    if change == "extra":
        data = (*data, data[0].model_copy(update={"item_id": "extra"}))
    if change == "duplicate":
        data = (*data[:-1], data[0])
    if change == "gold":
        data = (*data[:-1], data[-1].model_copy(update={"gold": "unknown"}))
    if change == "width":
        data = (*data[:-1], data[-1].model_copy(update={"logits": (1.0, 2.0, 3.0)}))
    if change == "digest":
        data = (*data[:-1], data[-1].model_copy(update={"logits": (1.0, 0.0)}))
    source = Source(p.source, data)
    with pytest.raises(EvaluationAnalysisError):
        await sdk.fit_registered_temperature(p, source=source, policy=Policy())
    assert source.closed == 1


async def test_policy_is_current_after_math_and_before_return() -> None:
    p = plan()
    source = Source(p.source)

    class Revoked(Policy):
        async def decide(self, request: FitPolicyRequest) -> RegisteredFitDecision:
            result = await super().decide(request)
            return result if self.calls <= 4 else result.model_copy(update={"status": "deny"})

    policy = Revoked()
    with pytest.raises(EvaluationAnalysisError, match=r"^analysis_denied$"):
        await sdk.fit_registered_temperature(p, source=source, policy=policy)
    assert source.read == 4
    assert source.closed == 1
    assert policy.calls == 5


async def test_clipped_point_nll_and_unclipped_fit_objective_remain_distinct() -> None:
    data = (TemperatureFitRow(item_id="x", gold="B", logits=(1_000_000.0, -1_000_000.0)),)
    p = plan(metadata(data))
    p = p.model_copy(update={"bounds": TemperatureFitBounds(minimum=0.001, maximum=1000.0)})
    result = await sdk.fit_registered_temperature(p, source=Source(p.source, data), policy=Policy())
    assert result.solution.objective_after == 2000.0
    assert result.after.nll.value == pytest.approx(-math.log(1e-15))
    assert result.after.nll.clipped == 1


async def test_repeated_cancel_drains_a_pending_source_read_and_closes_without_math() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    p = plan()

    class Held(Source):
        async def _stream(self) -> AsyncGenerator[TemperatureFitRow]:
            try:
                entered.set()
                await release.wait()
                yield self.data[0]
            finally:
                self.closed += 1

    source = Held(p.source)
    task = asyncio.create_task(sdk.fit_registered_temperature(p, source=source, policy=Policy()))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert source.closed == 0
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert source.closed == 1


async def test_repeated_cancel_drains_owned_math_without_continuing_to_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = plan()
    entered = threading.Event()
    release = threading.Event()
    advanced = []
    original = sdk._FitState.advance  # pyright: ignore[reportPrivateUsage] — controlled native-step boundary

    def held(self: object, passes: int = 8) -> None:
        entered.set()
        release.wait()
        advanced.append(True)
        original(cast("sdk._FitState", self), passes)  # pyright: ignore[reportPrivateUsage] — exact held state type

    monkeypatch.setattr(sdk._FitState, "advance", held)  # pyright: ignore[reportPrivateUsage] — no serialized callback seam
    task = asyncio.create_task(
        sdk.fit_registered_temperature(p, source=Source(p.source), policy=Policy())
    )
    assert await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert advanced == [True]


async def test_missing_policy_has_no_default_or_iterator() -> None:
    p = plan()
    source = Source(p.source)
    with pytest.raises(EvaluationAnalysisError, match=r"^analysis_unavailable$"):
        await sdk.fit_registered_temperature(p, source=source, policy=None)
    assert source.started == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "paired_cluster_percentile"),
        ("schema_version", True),
        ("mode", "bootstrap"),
        ("soft_budget_ms", 0),
        ("soft_budget_ms", 3_600_001),
        ("callback", "unsafe"),
        ("output_repo_id", "fictional/output"),
    ],
)
def test_no_unimplemented_mode_or_execution_knob_enters_plan(field: str, value: object) -> None:
    raw = plan().model_dump(mode="python")
    raw[field] = value
    with pytest.raises(ValueError, match="validation error"):
        TemperatureFitPlanV1.model_validate(raw)


async def test_tampered_frozen_nested_fields_refuse_before_metadata() -> None:
    p = plan()
    source = Source(p.source)
    broken = p.source.model_copy(update={"unknown": "PRIVATE_TAMPER"})
    p = p.model_copy(update={"source": broken})
    with pytest.raises(EvaluationAnalysisError):
        await sdk.fit_registered_temperature(p, source=source, policy=Policy())
    assert source.started == 0


async def test_raw_probability_input_and_nonidentity_projection_are_not_logit_fits() -> None:
    for patch in ({"input_kind": "probabilities"}, {"projection": "entail_vs_not"}):
        raw = metadata().model_dump(mode="python")
        raw.update(patch)
        with pytest.raises(ValueError, match="validation error"):
            TemperatureFitSourceMetadata.model_validate(raw)


async def test_bounds_and_source_byte_refusal_occurs_before_numerical_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = plan()
    source = Source(p.source)
    monkeypatch.setattr(sdk, "MAX_ANALYSIS_SOURCE_BYTES", 1)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("fit math cannot run after a source-byte refusal")

    monkeypatch.setattr(sdk, "_FitState", forbidden)
    with pytest.raises(EvaluationAnalysisError, match=r"^analysis_limit$"):
        await sdk.fit_registered_temperature(p, source=source, policy=Policy())
    assert source.read == source.closed == 1


async def test_current_policy_denial_at_input_chunk_stops_later_rows() -> None:
    data = tuple(
        TemperatureFitRow(item_id=str(i), gold="A" if i % 4 else "B", logits=(2 * math.log(3), 0.0))
        for i in range(1025)
    )
    p = plan(metadata(data))
    source = Source(p.source, data)

    class Revoke(Policy):
        async def decide(self, request: FitPolicyRequest) -> RegisteredFitDecision:
            result = await super().decide(request)
            return result if self.calls <= 2 else result.model_copy(update={"status": "deny"})

    with pytest.raises(EvaluationAnalysisError, match=r"^analysis_denied$"):
        await sdk.fit_registered_temperature(p, source=source, policy=Revoke())
    assert source.read == 1024
    assert source.closed == 1


async def test_soft_budget_overrun_does_not_return_a_partially_fitted_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    p = plan().model_copy(update={"soft_budget_ms": 1})
    monkeypatch.setattr(sdk, "_CLOCK", lambda: now[0])
    original = sdk._FitState.advance  # pyright: ignore[reportPrivateUsage] — fixed compute-step timer fixture

    def overrun(self: object, passes: int = 8) -> None:
        original(cast("sdk._FitState", self), passes)  # pyright: ignore[reportPrivateUsage] — exact state under test
        now[0] = 0.002

    monkeypatch.setattr(sdk._FitState, "advance", overrun)  # pyright: ignore[reportPrivateUsage] — no serialized timer seam
    with pytest.raises(EvaluationAnalysisError, match=r"^analysis_deadline$"):
        await sdk.fit_registered_temperature(p, source=Source(p.source), policy=Policy())


async def test_adapter_error_is_content_free_without_retained_context() -> None:
    p = plan()
    source = Source(p.source)

    class Broken(Policy):
        async def decide(self, request: FitPolicyRequest) -> RegisteredFitDecision:
            raise RuntimeError("PRIVATE_AUTHORITY_CANARY")

    with pytest.raises(EvaluationAnalysisError, match=r"^analysis_failed$") as caught:
        await sdk.fit_registered_temperature(p, source=source, policy=Broken())
    assert caught.value.__context__ is caught.value.__cause__ is None
    assert source.started == 0


async def test_cleanup_failure_does_not_replace_cancellation() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    p = plan()

    class BrokenClose(Source):
        async def _stream(self) -> AsyncGenerator[TemperatureFitRow]:
            try:
                entered.set()
                await release.wait()
                yield self.data[0]
            finally:
                self.closed += 1
                raise RuntimeError("PRIVATE_CLOSE_CANARY")

    source = BrokenClose(p.source)
    task = asyncio.create_task(sdk.fit_registered_temperature(p, source=source, policy=Policy()))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert source.closed == 1


async def test_integer_gold_requires_exact_declared_mapping() -> None:
    from strata_forge.pipelines.evaluation_analysis_spec import GoldIndex

    data = tuple(row.model_copy(update={"gold": 9 if row.gold == "A" else 4}) for row in rows())
    meta = metadata(data)
    p = plan(meta)
    with pytest.raises(EvaluationAnalysisError, match=r"^analysis_input_invalid$"):
        await sdk.fit_registered_temperature(p, source=Source(meta, data), policy=Policy())
    meta = meta.model_copy(
        update={"gold_index_map": (GoldIndex(index=9, label="A"), GoldIndex(index=4, label="B"))}
    )
    result = await sdk.fit_registered_temperature(
        plan(meta), source=Source(meta, data), policy=Policy()
    )
    assert result.solution.temperature == pytest.approx(2.0, abs=1e-7)


@pytest.mark.parametrize(
    "field", ["plan_sha256", "source_sha256", "records_sha256", "component_sha256", "observed_rows"]
)
async def test_receipt_revalidates_explicit_component_population_and_source_bindings(
    field: str,
) -> None:
    result = await fitted()
    raw = result.model_dump(mode="python")
    raw[field] = 8 if field == "observed_rows" else "0" * 64
    with pytest.raises(ValueError, match="analysis_invalid"):
        TemperatureFitReceipt.model_validate(raw)


async def test_restricted_logit_fit_preserves_conditional_probability_identity() -> None:
    meta = metadata().model_copy(update={"input_kind": "exact_restricted_label_logits"})
    result = await sdk.fit_registered_temperature(plan(meta), source=Source(meta), policy=Policy())
    assert result.component.probability_kind == "restricted_softmax"
    assert result.before.nll.probability_kind == "restricted_softmax"
    assert result.after.brier.probability_kind == "restricted_softmax"


async def test_unusable_point_chunk_is_not_repaired_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from strata_forge.metrics import MetricValue
    from strata_forge.pipelines.evaluation_analysis_spec import FitPointMetrics

    missing = MetricValue(
        value=None,
        n=0,
        excluded=4,
        reason="no_probabilities",
        probability_kind="class_probabilities",
    )

    def invalid_point(*args: object) -> FitPointMetrics:
        return FitPointMetrics(nll=missing, brier=missing)

    monkeypatch.setattr(sdk, "_point", invalid_point)
    p = plan()
    with pytest.raises(EvaluationAnalysisError, match=r"^analysis_failed$"):
        await sdk.fit_registered_temperature(p, source=Source(p.source), policy=Policy())


async def test_final_registry_check_can_revoke_after_all_point_computation() -> None:
    p = plan()
    source = Source(p.source)

    class FinalDeny(Policy):
        async def decide(self, request: FitPolicyRequest) -> RegisteredFitDecision:
            decision = await super().decide(request)
            return (
                decision
                if source.metadata_calls < 2
                else decision.model_copy(update={"status": "deny"})
            )

    with pytest.raises(EvaluationAnalysisError, match=r"^analysis_denied$"):
        await sdk.fit_registered_temperature(p, source=source, policy=FinalDeny())
    assert source.metadata_calls == 2
    assert source.closed == 1


async def test_repeated_cancel_drains_final_policy_without_returning_receipt() -> None:
    p = plan()
    source = Source(p.source)
    entered = asyncio.Event()
    release = asyncio.Event()

    class FinalHeld(Policy):
        async def decide(self, request: FitPolicyRequest) -> RegisteredFitDecision:
            decision = await super().decide(request)
            if source.metadata_calls == 2:
                entered.set()
                await release.wait()
            return decision

    task = asyncio.create_task(sdk.fit_registered_temperature(p, source=source, policy=FinalHeld()))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert source.metadata_calls == 2
    assert source.closed == 1
