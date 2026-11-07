"""Exact fake-clock measurement definitions and tiny random CPU SDK integration controls."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, cast

import pytest

from strata_forge.pipelines import classifier_benchmark as bench
from strata_forge.pipelines.benchmark_spec import (
    BenchmarkEnvironment,
    BenchmarkError,
    BenchmarkTargetDescriptor,
    ClassifierBenchmarkPlanV1,
    ClassifierBenchmarkResult,
)


def plan(**updates: Any) -> ClassifierBenchmarkPlanV1:
    return ClassifierBenchmarkPlanV1.model_validate(
        {
            "version": 1,
            "boundary": "hf_scoring_sdk",
            "device": "cpu",
            "precision": "fp32",
            "concurrency": 1,
            "batch_size": 4,
            "warmup_calls": 2,
            "measured_calls": 5,
            "max_elapsed_ms": 1000,
            **updates,
        }
    )


class Clock:
    def __init__(self) -> None:
        self.time = 0

    def now(self) -> int:
        return self.time

    def advance(self, duration: int) -> None:
        self.time += duration


class Operation:
    def __init__(
        self,
        clock: Clock,
        *,
        durations: tuple[int, ...] = (
            100_000_000,
            200_000_000,
            1_000_000,
            2_000_000,
            3_000_000,
            4_000_000,
            10_000_000,
        ),
        gap: int = 1_000_000,
        fail: int | None = None,
    ) -> None:
        self.clock, self.durations, self.gap, self.fail = clock, durations, gap, fail
        self.stopped = threading.Event()
        self.calls = 0
        self.validated = 0
        self.descriptor = BenchmarkTargetDescriptor(
            model_ref="fixture/model@" + "a" * 40,
            tokenizer_ref="fixture/tokenizer@" + "b" * 40,
            boundary="hf_scoring_sdk",
            engine="hf_classify",
            family="bert",
            probability_family="full_class_softmax",
            label_count=2,
            ordered_labels_sha256="c" * 64,
            tokenizer_sha256="d" * 64,
            geometry_sha256=None,
            outputs=("label",),
            rows_per_call=4,
            max_length=16,
            native_context=128,
            input_shape="pair",
            mode_policy="scorer_preexisting_eval",
            exact_token_lengths=None,
            token_lengths_reason="sdk_has_no_length_receipt",
            padding_side="right",
        )
        self.environment = BenchmarkEnvironment(
            python="fake",
            torch="fake",
            transformers="fake",
            operating_system="synthetic",
            architecture="synthetic",
            intraop_threads=1,
            interop_threads=1,
            attention="eager",
            clock_implementation="fixture",
            clock_resolution_ns=1.0,
        )

    def fingerprint(self) -> str:
        return "f" * 64

    async def call(self) -> object:
        index = self.calls
        self.calls += 1
        self.clock.advance(self.durations[index])
        if index == self.fail:
            raise RuntimeError("PRIVATE_BENCHMARK_FAILURE_CANARY")
        return 4

    def validate(self, value: object) -> None:
        assert value == 4
        self.validated += 1
        self.clock.advance(self.gap)


async def measure(
    p: ClassifierBenchmarkPlanV1, operation: Operation, monkeypatch: pytest.MonkeyPatch
) -> ClassifierBenchmarkResult:
    monkeypatch.setattr(bench, "_CLOCK", operation.clock.now)
    clock = bench._Clock()  # pyright: ignore[reportPrivateUsage] — independent deterministic clock witness
    start = clock.now()
    return await bench._measure(p, operation, clock, start)  # pyright: ignore[reportPrivateUsage] — exercise the exact production loop


@pytest.mark.asyncio
async def test_exact_clock_fixture_warmup_ranks_and_two_rate_denominators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    operation = Operation(clock)
    result = await measure(plan(), operation, monkeypatch)
    assert result.status == "complete", (
        result.reason,
        result.failed_phase,
        result.measured_completed,
    )
    assert operation.calls == 7
    assert result.measured_durations_ns == (1_000_000, 2_000_000, 3_000_000, 4_000_000, 10_000_000)
    assert result.measurement_window_ns == 25_000_000
    assert result.measured_rows == 20
    assert result.warmup_durations_ns == (100_000_000, 200_000_000)
    stats = result.statistics
    assert stats is not None
    assert (stats.p50_rank, stats.p95_rank, stats.p99_rank) == (3, 5, 5)
    assert (stats.p50_ns, stats.p95_ns, stats.p99_ns) == (3_000_000, 10_000_000, 10_000_000)
    assert stats.mean_ns == 4_000_000
    assert stats.rows_per_second == 800
    assert stats.sdk_active_rows_per_second == 1000
    assert stats.amortized_sdk_ns_per_row == 1_000_000
    assert stats.empirical_cdf_step == 0.2
    assert "PRIVATE" not in result.model_dump_json()


@pytest.mark.parametrize("failure", [0, 3])
@pytest.mark.asyncio
async def test_failed_call_is_counted_and_never_gives_survivor_headlines(
    monkeypatch: pytest.MonkeyPatch, failure: int
) -> None:
    operation = Operation(Clock(), fail=failure)
    result = await measure(plan(), operation, monkeypatch)
    assert result.status == "failed"
    assert result.reason == "benchmark_failed"
    assert result.statistics is None
    assert operation.calls == failure + 1
    assert result.failed_call_duration_ns == operation.durations[failure]
    assert result.failed_phase == ("warmup" if failure < 2 else "measured")
    assert result.measured_completed == max(0, failure - 2)
    assert result.measured_attempted == max(0, failure - 1)
    assert result.total_elapsed_ns == operation.clock.time
    assert "PRIVATE" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_one_native_overrun_is_incomplete_not_a_hard_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = Operation(Clock(), durations=(2_000_000,), gap=0)
    result = await measure(
        plan(warmup_calls=0, measured_calls=5, max_elapsed_ms=1), operation, monkeypatch
    )
    assert operation.calls == 1
    assert result.status == "deadline"
    assert result.reason == "benchmark_deadline"
    assert result.measured_completed == 1
    assert result.statistics is None
    assert result.total_elapsed_ns == 2_000_000


@pytest.mark.parametrize("gap", [0, 1_000_000])
@pytest.mark.asyncio
async def test_zero_timing_never_creates_infinite_rate(
    monkeypatch: pytest.MonkeyPatch, gap: int
) -> None:
    operation = Operation(Clock(), durations=(0,), gap=gap)
    result = await measure(plan(warmup_calls=0, measured_calls=1), operation, monkeypatch)
    assert result.statistics is not None
    assert result.statistics.p50_ns == result.statistics.p95_ns == result.statistics.p99_ns == 0
    assert result.statistics.sdk_active_rows_per_second is None
    assert result.statistics.sdk_rate_reason == "zero_elapsed"
    assert result.statistics.rows_per_second == (None if gap == 0 else 4000)


@pytest.mark.asyncio
async def test_cancel_held_sdk_task_drains_before_return_and_stops_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class Held(Operation):
        async def call(self) -> object:
            self.calls += 1
            entered.set()
            await release.wait()
            return 4

    operation = Held(Clock())
    task = asyncio.create_task(measure(plan(warmup_calls=0), operation, monkeypatch))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert operation.calls == 1
    assert operation.validated == 0


@pytest.mark.asyncio
async def test_changed_workload_has_no_complete_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class Changed(Operation):
        def fingerprint(self) -> str:
            return ("f" if self.calls == 0 else "e") * 64

    result = await measure(
        plan(warmup_calls=0, measured_calls=1),
        Changed(Clock(), durations=(1_000_000,)),
        monkeypatch,
    )
    assert result.status == "failed"
    assert result.reason == "benchmark_input_invalid"
    assert result.statistics is None


def scorer_target(engine: str = "hf_classify") -> tuple[Any, Any, Any]:
    from strata_forge.pipelines._scoring import Scorer
    from strata_forge.pipelines.benchmark_spec import ResolvedBenchmarkTarget
    from strata_forge.training._load_guard import LoadingAudit, PaddingPolicy
    from tests.unit.training.test_classifier import stack
    from tests.unit.training.test_verifier_classifier import tokenizer as tiny_tokenizer

    torch, transformers = stack()
    if torch is None:
        return None, None, None
    torch.set_num_threads(1)
    tokenizer = tiny_tokenizer(engine == "hf_classify")
    if engine == "hf_classify":
        model = transformers.BertForSequenceClassification(
            transformers.BertConfig(
                vocab_size=32,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                max_position_embeddings=128,
                num_labels=2,
                id2label={0: "yes", 1: "no"},
                label2id={"yes": 0, "no": 1},
                pad_token_id=0,
            )
        )
    else:
        model = transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=32,
                n_embd=8,
                n_layer=1,
                n_head=2,
                n_positions=128,
                pad_token_id=0,
                eos_token_id=0,
            )
        )
    model.eval()
    scorer = Scorer(
        engine=cast("Any", engine),
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        labels=["yes", "no"],
        token_ids=[] if engine == "hf_classify" else [7, 8],
        max_length=64,
        device="cpu",
        loading_audit=LoadingAudit(status="accepted"),
        padding_policy=PaddingPolicy(
            source="tokenizer_pad", pad_token_id=0, original_model_pad_token_id=0
        ),
    )
    return (
        torch,
        ResolvedBenchmarkTarget(
            model_ref="fixture/model@" + "a" * 40,
            tokenizer_ref="fixture/tokenizer@" + "b" * 40,
            scorer=scorer,
        ),
        model,
    )


@pytest.mark.parametrize("engine", ["hf_classify", "hf_label_logits"])
@pytest.mark.asyncio
async def test_tiny_native_scorer_counts_every_call_and_partial_batch(engine: str) -> None:
    from strata_forge.pipelines._scoring import ScoringInput

    torch, target, model = scorer_target(engine)
    if torch is None:
        return
    sizes = []

    def observed(_module: Any, args: Any, kwargs: Any) -> None:
        sizes.append(int(kwargs["input_ids"].shape[0]))

    hook = model.register_forward_pre_hook(observed, with_kwargs=True)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    rows = tuple(
        ScoringInput(text="a b", text_pair="q" if engine == "hf_classify" else None)
        for _ in range(3)
    )
    p = plan(batch_size=2, warmup_calls=1, measured_calls=2, max_elapsed_ms=10000)
    result = await bench.benchmark_scorer(p, target=target, rows=rows)
    hook.remove()
    assert result.status == "complete", (
        result.reason,
        result.failed_phase,
        result.measured_completed,
    )
    assert result.measured_rows == 6
    assert sizes == [2, 1] * 3
    assert result.target.probability_family == (
        "full_class_softmax" if engine == "hf_classify" else "restricted_label_token_softmax"
    )
    assert result.statistics is not None
    assert result.statistics.sample_count == 2
    assert all(value > 0 for value in result.measured_durations_ns)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert not any(module.training for module in model.modules())


@pytest.mark.parametrize("family", ["bert", "gpt2", "modernbert-decoder"])
@pytest.mark.parametrize("boundary", ["verifier_preencoded_sdk", "verifier_encode_score_sdk"])
@pytest.mark.asyncio
async def test_tiny_native_verifier_boundaries_and_mode_cleanup(family: str, boundary: str) -> None:
    from strata_forge.pipelines.benchmark_spec import ResolvedBenchmarkTarget
    from strata_forge.training.verifier_classifier import encode_verifier_batch
    from tests.unit.training.test_verifier_classifier import examples, initialized

    torch, _, verifier, tokenizer = await initialized(family)
    if torch is None:
        return
    source = examples()
    encoded = await encode_verifier_batch(tokenizer, source, geometry=verifier.geometry)
    target = ResolvedBenchmarkTarget(
        model_ref="fixture/model@" + "a" * 40,
        tokenizer_ref="fixture/tokenizer@" + "b" * 40,
        verifier=verifier,
        tokenizer=tokenizer,
    )
    modes = tuple(module.training for module in verifier.model.modules())
    sizes = []

    def observed(_module: Any, args: Any, kwargs: Any) -> None:
        sizes.append(int(kwargs["input_ids"].shape[0]))

    hook = verifier.model.register_forward_pre_hook(observed, with_kwargs=True)
    p = plan(
        boundary=boundary, batch_size=2, warmup_calls=1, measured_calls=2, max_elapsed_ms=10000
    )
    result = await bench.benchmark_verifier(
        p,
        target=target,
        batch=encoded if boundary == "verifier_preencoded_sdk" else None,
        examples=source if boundary == "verifier_encode_score_sdk" else None,
    )
    hook.remove()
    assert result.status == "complete", (
        result.reason,
        result.failed_phase,
        result.measured_completed,
    )
    assert result.target.family == family
    assert result.measured_rows == 4
    assert sizes == [2, 2, 2]
    assert tuple(module.training for module in verifier.model.modules()) == modes
    assert result.target.outputs == ("label", "inconsistent", "support_set")
    assert (
        result.target.exact_token_lengths is not None
        if boundary == "verifier_preencoded_sdk"
        else result.target.exact_token_lengths is None
    )
    assert "é" not in result.model_dump_json()
    assert "p1" not in result.model_dump_json()


@pytest.mark.parametrize(
    "change", ["half", "left", "context", "vocab", "unknown_family", "not_eval", "compiled"]
)
@pytest.mark.asyncio
async def test_native_refusals_happen_before_any_forward(change: str) -> None:
    from strata_forge.pipelines._scoring import ScoringInput

    torch, target, model = scorer_target()
    if torch is None:
        return
    if change == "half":
        model.half()
    elif change == "left":
        target.scorer._tokenizer.padding_side = "left"
    elif change == "context":
        target.scorer._max_length = 129
    elif change == "vocab":
        model.bert.embeddings.word_embeddings = torch.nn.Embedding(4, 8)
    elif change == "unknown_family":
        model.config.model_type = "unsupported_family"
    elif change == "compiled":

        def compiled(*args: Any, **kwargs: Any) -> None:
            pytest.fail("compiled code cannot run")

        model._compiled_call_impl = compiled
    else:
        model.train()
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        raise AssertionError("no forward for invalid admission")

    handle = model.register_forward_pre_hook(forbidden, with_kwargs=True)
    with pytest.raises(BenchmarkError, match=r"^benchmark_target_unsupported$") as caught:
        await bench.benchmark_scorer(
            plan(), target=target, rows=(ScoringInput(text="a", text_pair="q"),)
        )
    handle.remove()
    assert calls == []
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "change",
    ["too_many_rows", "too_many_applications", "oversized_row", "mixed_shape", "tampered_row"],
)
@pytest.mark.asyncio
async def test_bounded_admission_has_no_forward(change: str) -> None:
    from strata_forge.pipelines._scoring import ScoringInput

    torch, target, model = scorer_target()
    if torch is None:
        return
    p = plan()
    rows = (ScoringInput(text="a", text_pair="q"),)
    if change == "too_many_rows":
        rows = rows * 129
    elif change == "too_many_applications":
        rows = rows * 128
        p = plan(warmup_calls=50, measured_calls=1000)
    elif change == "oversized_row":
        rows = (ScoringInput(text="a" * 262144, text_pair="q"),)
    elif change == "mixed_shape":
        rows = (*rows, ScoringInput(text="a"))
    else:
        rows = (rows[0].model_copy(update={"text": 123}),)
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        raise AssertionError("no forward")

    handle = model.register_forward_pre_hook(forbidden, with_kwargs=True)
    with pytest.raises(BenchmarkError):
        await bench.benchmark_scorer(p, target=target, rows=rows)
    handle.remove()
    assert calls == []


@pytest.mark.asyncio
async def test_zero_warmups_are_not_replaced_by_a_hidden_probe() -> None:
    from strata_forge.pipelines._scoring import ScoringInput

    torch, target, model = scorer_target()
    if torch is None:
        return
    calls = []

    def count(*args: Any, **kwargs: Any) -> None:
        calls.append(True)

    handle = model.register_forward_pre_hook(count, with_kwargs=True)
    result = await bench.benchmark_scorer(
        plan(warmup_calls=0, measured_calls=1),
        target=target,
        rows=(ScoringInput(text="a", text_pair="q"),),
    )
    handle.remove()
    assert result.status == "complete"
    assert result.warmup_completed == 0
    assert result.measured_completed == 1
    assert calls == [True]


@pytest.mark.asyncio
async def test_alias_claim_is_held_through_cancellation_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from strata_forge.pipelines._scoring import Scorer, ScoringInput
    from strata_forge.pipelines.benchmark_spec import ResolvedBenchmarkTarget

    torch, target, _model = scorer_target()
    if torch is None:
        return
    entered, release = asyncio.Event(), asyncio.Event()
    original = Scorer.score

    async def held(self: Any, *args: Any, **kwargs: Any) -> Any:
        entered.set()
        await release.wait()
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(Scorer, "score", held)
    rows = (ScoringInput(text="a", text_pair="q"),)
    p = plan(warmup_calls=0, measured_calls=1)
    task = asyncio.create_task(bench.benchmark_scorer(p, target=target, rows=rows))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    alias = ResolvedBenchmarkTarget(
        model_ref=target.model_ref, tokenizer_ref=target.tokenizer_ref, scorer=target.scorer
    )
    with pytest.raises(BenchmarkError, match=r"^benchmark_target_unsupported$"):
        await bench.benchmark_scorer(p, target=alias, rows=rows)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(Scorer, "score", original)
    result = await bench.benchmark_scorer(p, target=alias, rows=rows)
    assert result.status == "complete"


@pytest.mark.asyncio
async def test_semantic_tokenizer_change_is_not_hidden_by_runtime_settings_filter() -> None:
    from strata_forge.pipelines._scoring import ScoringInput

    torch, target, model = scorer_target()
    if torch is None:
        return
    calls = []

    def changed(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        target.scorer._tokenizer.add_tokens(["new_semantic_token"])

    handle = model.register_forward_pre_hook(changed, with_kwargs=True)
    result = await bench.benchmark_scorer(
        plan(warmup_calls=0, measured_calls=1),
        target=target,
        rows=(ScoringInput(text="a", text_pair="q"),),
    )
    handle.remove()
    assert result.status == "failed"
    assert result.reason == "benchmark_input_invalid"
    assert result.statistics is None
    assert calls == [True]


@pytest.mark.parametrize("change", ["shape", "id_range", "mask", "marker", "geometry"])
@pytest.mark.asyncio
async def test_invalid_preencoded_inputs_refuse_before_native_call(change: str) -> None:
    from strata_forge.pipelines.benchmark_spec import ResolvedBenchmarkTarget
    from strata_forge.training.verifier_classifier import VerifierBatch, encode_verifier_batch
    from tests.unit.training.test_verifier_classifier import examples, initialized

    torch, _, verifier, tokenizer = await initialized("bert")
    if torch is None:
        return
    encoded = await encode_verifier_batch(tokenizer, examples(), geometry=verifier.geometry)
    if change == "shape":
        encoded.model_inputs["attention_mask"] = encoded.model_inputs["attention_mask"][:1]
    elif change == "id_range":
        encoded.model_inputs["input_ids"][0, 1] = 1000
    elif change == "mask":
        encoded.model_inputs["attention_mask"][0, 1] = 2
    elif change == "marker":
        encoded = VerifierBatch(
            encoded.geometry_sha256,
            encoded.item_ids,
            encoded.model_inputs,
            ((1000, 1000), encoded.marker_positions[1]),
            encoded.premise_ids,
        )
    else:
        encoded = VerifierBatch(
            "0" * 64,
            encoded.item_ids,
            encoded.model_inputs,
            encoded.marker_positions,
            encoded.premise_ids,
        )
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        raise AssertionError("invalid batch cannot forward")

    handle = verifier.model.register_forward_pre_hook(forbidden, with_kwargs=True)
    target = ResolvedBenchmarkTarget(
        model_ref="fixture/model@" + "a" * 40,
        tokenizer_ref="fixture/tokenizer@" + "b" * 40,
        verifier=verifier,
        tokenizer=tokenizer,
    )
    with pytest.raises(BenchmarkError, match=r"^benchmark_input_invalid$"):
        await bench.benchmark_verifier(
            plan(boundary="verifier_preencoded_sdk", batch_size=2), target=target, batch=encoded
        )
    handle.remove()
    assert calls == []


@pytest.mark.asyncio
async def test_cancellation_after_encoding_does_not_start_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from strata_forge.pipelines.benchmark_spec import ResolvedBenchmarkTarget
    from strata_forge.training import verifier_classifier as sdk
    from tests.unit.training.test_verifier_classifier import examples, initialized

    torch, _, verifier, tokenizer = await initialized("bert")
    if torch is None:
        return
    entered, release = asyncio.Event(), asyncio.Event()
    original = sdk.encode_verifier_batch

    async def held(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(sdk, "encode_verifier_batch", held)
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        raise AssertionError("no scoring after cancellation during encoding")

    handle = verifier.model.register_forward_pre_hook(forbidden, with_kwargs=True)
    target = ResolvedBenchmarkTarget(
        model_ref="fixture/model@" + "a" * 40,
        tokenizer_ref="fixture/tokenizer@" + "b" * 40,
        verifier=verifier,
        tokenizer=tokenizer,
    )
    task = asyncio.create_task(
        bench.benchmark_verifier(
            plan(boundary="verifier_encode_score_sdk", batch_size=2),
            target=target,
            examples=examples(),
        )
    )
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    handle.remove()
    assert calls == []


@pytest.mark.parametrize("invalid", [True, 1.5, "native", None])
@pytest.mark.asyncio
async def test_invalid_clock_values_fail_without_model_calls(
    monkeypatch: pytest.MonkeyPatch, invalid: Any
) -> None:
    operation = Operation(Clock())
    monkeypatch.setattr(bench, "_CLOCK", lambda: invalid)
    clock = bench._Clock()  # pyright: ignore[reportPrivateUsage] — malformed private clock fixture
    with pytest.raises(BenchmarkError, match=r"^benchmark_timer_invalid$"):
        await bench._measure(plan(), operation, clock, clock.now())  # pyright: ignore[reportPrivateUsage] — direct production-loop control
    assert operation.calls == 0


@pytest.mark.asyncio
async def test_backwards_clock_has_no_false_fast_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = Operation(Clock(), durations=(-1,), gap=0)
    result = await measure(plan(warmup_calls=0, measured_calls=1), operation, monkeypatch)
    assert result.status == "failed"
    assert result.reason == "benchmark_timer_invalid"
    assert result.statistics is None


@pytest.mark.parametrize(
    "change",
    ["missing", "order", "truncated", "overflow", "error", "nonfinite", "wrong_labels", "boolean"],
)
@pytest.mark.asyncio
async def test_scorer_bad_outputs_never_improve_headline_rates(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from strata_forge.pipelines._scoring import ScoredRow, Scorer, ScoringInput, ScoringResult

    torch, target, _ = scorer_target()
    if torch is None:
        return
    items = [
        ScoredRow(index=0, probabilities={"yes": 0.6, "no": 0.4}, pred="yes"),
        ScoredRow(index=1, probabilities={"yes": 0.6, "no": 0.4}, pred="yes"),
    ]
    if change == "missing":
        items.pop()
    elif change == "order":
        items[1] = items[1].model_copy(update={"index": 0})
    elif change in ("truncated", "overflow"):
        items[0] = items[0].model_copy(update={change: True})
    elif change == "error":
        items[0] = items[0].model_copy(update={"error": "PRIVATE_OUTPUT_CANARY"})
    elif change == "wrong_labels":
        items[0] = items[0].model_copy(update={"probabilities": {"other": 1.0}})
    elif change == "nonfinite":
        items[0] = items[0].model_copy(update={"probabilities": {"yes": float("nan"), "no": 0.4}})
    else:
        items[0] = items[0].model_copy(update={"probabilities": {"yes": True, "no": 0.0}})
    count = 0

    async def fake(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal count
        count += 1
        return ScoringResult(rows=items, failed_rows=0, overflow_rows=0, truncated_rows=0)

    monkeypatch.setattr(Scorer, "score", fake)
    result = await bench.benchmark_scorer(
        plan(warmup_calls=0, measured_calls=3),
        target=target,
        rows=(ScoringInput(text="a", text_pair="q"),) * 2,
    )
    assert result.status == "failed"
    assert result.statistics is None
    assert result.measured_rows == 0
    assert result.measured_attempted == 1
    assert count == 1
    assert "PRIVATE" not in result.model_dump_json()


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "wrong_id",
        "flag_missing",
        "flag_boolean",
        "support_missing",
        "support_nonfinite",
        "support_wrong_ids",
    ],
)
@pytest.mark.asyncio
async def test_auxiliary_head_output_coverage_is_explicit(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from strata_forge.pipelines.benchmark_spec import ResolvedBenchmarkTarget
    from strata_forge.training import verifier_classifier as sdk
    from tests.unit.training.test_verifier_classifier import examples, initialized

    torch, _, verifier, tokenizer = await initialized("bert")
    if torch is None:
        return
    source = examples()
    encoded = await sdk.encode_verifier_batch(tokenizer, source, geometry=verifier.geometry)
    rows: list[dict[str, object]] = [
        {
            "item_id": row.item_id,
            "probabilities": dict(zip(verifier.geometry.labels, (0.6, 0.3, 0.1), strict=True)),
            "p_inconsistent": 0.2,
            "support_scores": {premise.pid: -1.0 for premise in row.premises},
        }
        for row in source
    ]
    if change == "missing":
        rows.pop()
    elif change == "wrong_id":
        rows[0]["item_id"] = "PRIVATE_UNEXPECTED_ID"
    elif change == "flag_missing":
        rows[0]["p_inconsistent"] = None
    elif change == "flag_boolean":
        rows[0]["p_inconsistent"] = False
    elif change == "support_missing":
        rows[0]["support_scores"] = None
    elif change == "support_nonfinite":
        rows[0]["support_scores"] = {"p1": float("inf"), "p2": -1.0}
    else:
        rows[0]["support_scores"] = {"wrong": -1.0}
    count = 0

    async def fake(*args: Any, **kwargs: Any) -> Any:
        nonlocal count
        count += 1
        return tuple(rows)

    monkeypatch.setattr(sdk, "score_verifier_batch", fake)
    target = ResolvedBenchmarkTarget(
        model_ref="fixture/model@" + "a" * 40,
        tokenizer_ref="fixture/tokenizer@" + "b" * 40,
        verifier=verifier,
        tokenizer=tokenizer,
    )
    result = await bench.benchmark_verifier(
        plan(boundary="verifier_preencoded_sdk", batch_size=2, warmup_calls=0, measured_calls=3),
        target=target,
        batch=encoded,
    )
    assert result.status == "failed"
    assert result.statistics is None
    assert result.measured_rows == 0
    assert count == 1
    assert "PRIVATE" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_total_workload_byte_limit_has_no_silent_prefix() -> None:
    from strata_forge.pipelines._scoring import ScoringInput

    torch, target, model = scorer_target()
    if torch is None:
        return
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append(True)

    hook = model.register_forward_pre_hook(forbidden, with_kwargs=True)
    with pytest.raises(BenchmarkError):
        await bench.benchmark_scorer(
            plan(warmup_calls=0, measured_calls=1),
            target=target,
            rows=(ScoringInput(text="a" * 100000, text_pair="q"),) * 128,
        )
    hook.remove()
    assert calls == []


@pytest.mark.asyncio
async def test_library_does_not_load_or_resolve_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    import huggingface_hub

    from strata_forge.pipelines._scoring import ScoringInput

    torch, target, _ = scorer_target()
    if torch is None:
        return
    import transformers

    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        raise AssertionError("benchmark cannot load assets or use network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    for name in (
        "AutoTokenizer",
        "AutoConfig",
        "AutoModelForSequenceClassification",
        "AutoModelForCausalLM",
    ):
        monkeypatch.setattr(getattr(transformers, name), "from_pretrained", forbidden)
    result = await bench.benchmark_scorer(
        plan(warmup_calls=0, measured_calls=1),
        target=target,
        rows=(ScoringInput(text="a", text_pair="q"),),
    )
    assert result.status == "complete"
    assert calls == []


@pytest.mark.asyncio
async def test_executable_plan_extra_is_rejected_without_calling_it() -> None:
    from strata_forge.pipelines._scoring import ScoringInput

    calls = []

    def callback() -> None:
        calls.append(True)

    p = plan().model_copy(update={"callback": callback})
    torch, target, _ = scorer_target()
    if torch is None:
        return
    with pytest.raises(BenchmarkError, match=r"^benchmark_invalid$"):
        await bench.benchmark_scorer(
            p, target=target, rows=(ScoringInput(text="a", text_pair="q"),)
        )
    assert calls == []
