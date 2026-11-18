"""The latency benchmark runner: its plan, its measurement loop, its spec and its outputs.

The pure planning and the measurement loop run against a fake clock and a fake engine step, so
warmup exclusion, the concurrency bound, the wall-clock cap and failure handling are tested
exactly. The orchestration runs the real ``main`` with the heavy externals replaced (the split,
the model download, the classifier, the model server and the Hub). The real transformers path is
covered by the clean-install smoke check (``check_benchmark``), on a tiny CPU classifier.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import types
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from pydantic import ValidationError

from strata_forge.pipelines import SPEC_VERSION
from strata_forge.pipelines import _benchmark as bench
from strata_forge.pipelines import _environment as env
from strata_forge.pipelines import benchmark_runner as br
from strata_forge.pipelines import inference_runner as ir
from strata_forge.pipelines._common import LEGACY_TOKEN_ENV, RunError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence
    from pathlib import Path

_TOKEN = "hf_secretwritetoken1234567890"
_SHA = "0123456789abcdef0123456789abcdef01234567"
_DATA_SHA = "89abcdef0123456789abcdef0123456789abcdef"
_SECRET_ROW_TEXT = "premise-text-that-must-never-leave-the-vm"

# ------------------------------ planning ------------------------------


def test_configurations_are_the_grid_batch_size_major() -> None:
    grid = bench.configurations([1, 8], [1, 4])
    assert [(c.index, c.batch_size, c.concurrency) for c in grid] == [
        (0, 1, 1),
        (1, 1, 4),
        (2, 8, 1),
        (3, 8, 4),
    ]


def test_admission_counts_each_exclusion_by_reason() -> None:
    lengths = [10, None, 600, 300, 40, 50]
    items = [3, 2, 1, 1, 25, None]
    eligible, excluded = bench.admit(
        lengths, items, model_limit=512, max_input_tokens=200, max_items=20
    )
    assert eligible == [0]
    assert excluded == {
        "unmeasurable": 2,  # a None length and a None item count
        "over_model_limit": 1,
        "over_max_input_tokens": 1,
        "over_max_items": 1,
    }


def test_admission_without_caps_keeps_every_measurable_row() -> None:
    eligible, excluded = bench.admit(
        [5, 6], None, model_limit=None, max_input_tokens=None, max_items=None
    )
    assert eligible == [0, 1]
    assert sum(excluded.values()) == 0


def test_the_sample_is_seeded_in_row_order_and_never_padded() -> None:
    eligible = list(range(100))
    first = bench.draw_sample(eligible, 10, seed=42)
    assert first == bench.draw_sample(eligible, 10, seed=42)
    assert first == sorted(first)
    assert first != bench.draw_sample(eligible, 10, seed=43)
    assert bench.draw_sample(eligible, None, seed=1) == eligible
    with pytest.raises(RunError, match="only 100 eligible rows"):
        bench.draw_sample(eligible, 101, seed=0)


def test_each_pass_is_a_shuffle_seeded_with_seed_plus_pass_index() -> None:
    sample = list(range(20))
    import random

    expected = list(sample)
    random.Random(7 + 3).shuffle(expected)  # noqa: S311 - the documented derivation
    assert bench.pass_order(sample, 7, 3) == expected
    assert bench.pass_order(sample, 7, 0) != bench.pass_order(sample, 7, 1)
    assert sorted(bench.pass_order(sample, 7, 2)) == sample


def test_batches_are_full_and_the_remainder_is_counted() -> None:
    batches, remainder = bench.full_batches([5, 4, 3, 2, 1], 2)
    assert batches == [[5, 4], [3, 2]]
    assert remainder == 1


def test_warmup_draws_from_the_pass_order_and_wraps() -> None:
    assert bench.warmup_batches([1, 2, 3], 2, 2) == [[1, 2], [3, 1]]
    assert bench.warmup_batches([], 1, 5) == []


def test_bucket_labels_use_inclusive_upper_edges() -> None:
    edges = [512, 2048, 4096, 8192]
    assert bench.bucket_label(1, edges) == "0-512"
    assert bench.bucket_label(512, edges) == "0-512"
    assert bench.bucket_label(513, edges) == "513-2048"
    assert bench.bucket_label(8192, edges) == "4097-8192"
    assert bench.bucket_label(9000, edges) == ">8192"


def test_union_seconds_counts_overlap_once() -> None:
    ms = 1_000_000
    assert bench.union_seconds([(0, 10 * ms), (5 * ms, 20 * ms), (30 * ms, 40 * ms)]) == 0.03
    assert bench.union_seconds([]) == 0.0


# ------------------------------ measurement ------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


def _sequential_step(
    clock: _Clock, seen: list[list[int]], *, fail_on: int | None = None
) -> bench.Step:
    async def _step(rows: list[int]) -> bench.StepTiming:
        seen.append(rows)
        if fail_on is not None and len(seen) == fail_on:
            raise RuntimeError("CUDA out of memory")
        start = clock.now
        # The first call is slow, like a cold kernel; every later one costs 1 ms per row.
        clock.now += (500 if len(seen) == 1 else 1) * 1_000_000 * len(rows)
        return bench.StepTiming(start_ns=start, end_ns=clock.now, forward_ms=0.6 * len(rows))

    return _step


async def test_warmup_runs_before_every_pass_and_is_never_measured() -> None:
    clock = _Clock()
    seen: list[list[int]] = []
    config = bench.Configuration(index=0, batch_size=1, concurrency=1)
    outcome = await bench.measure_configuration(
        config,
        sample=list(range(10)),
        passes=3,
        warmup=2,
        order_seed=5,
        warmup_seed=1005,
        step=_sequential_step(clock, seen),
        clock_ns=clock,
        deadline_ns=10**18,
    )
    assert outcome.status == "measured"
    assert outcome.passes_completed == 3
    assert len(seen) == 3 * (2 + 10)
    assert len(outcome.iterations) == 30
    # The 500 ms cold first call was a warmup: it is the recorded first iteration, not a sample.
    assert outcome.first_iteration_ms == 500.0
    assert all(i.end_ns - i.start_ns == 1_000_000 for i in outcome.iterations)
    # Each pass serves its own order_seed shuffle; its warmups are drawn from the workload with
    # warmup_seed + pass, not from the start of the timed order.
    for pass_index in range(3):
        order = bench.pass_order(list(range(10)), 5, pass_index)
        warm = bench.pass_order(list(range(10)), 1005, pass_index)
        start = pass_index * 12
        assert seen[start : start + 2] == [[warm[0]], [warm[1]]]
        timed = [i.rows[0] for i in outcome.iterations if i.pass_index == pass_index]
        assert timed == order


async def test_concurrency_bounds_the_iterations_in_flight() -> None:
    in_flight = 0
    peak = 0
    tick = 0

    async def _step(rows: list[int]) -> bench.StepTiming:
        nonlocal in_flight, peak, tick
        in_flight += 1
        peak = max(peak, in_flight)
        start = tick
        tick += 1
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        in_flight -= 1
        return bench.StepTiming(start_ns=start * 1_000_000, end_ns=(start + 4) * 1_000_000)

    outcome = await bench.measure_configuration(
        bench.Configuration(index=0, batch_size=1, concurrency=3),
        sample=list(range(12)),
        passes=1,
        warmup=1,
        order_seed=0,
        warmup_seed=1000,
        step=_step,
        clock_ns=lambda: 0,
        deadline_ns=1,
    )
    assert peak == 3
    assert len(outcome.iterations) == 12


async def test_the_wall_clock_cap_stops_measurement_and_marks_it() -> None:
    clock = _Clock()
    seen: list[list[int]] = []
    outcome = await bench.measure_configuration(
        bench.Configuration(index=0, batch_size=1, concurrency=1),
        sample=list(range(10)),
        passes=5,
        warmup=1,
        order_seed=0,
        warmup_seed=1000,
        step=_sequential_step(clock, seen),
        clock_ns=clock,
        deadline_ns=505 * 1_000_000,
    )
    assert outcome.status == "wall_clock_cap"
    assert outcome.passes_completed == 0
    assert len(outcome.iterations) == 5


async def test_a_pass_that_ends_exactly_at_the_cap_is_complete() -> None:
    clock = _Clock()
    outcome = await bench.measure_configuration(
        bench.Configuration(index=0, batch_size=1, concurrency=1),
        sample=list(range(10)),
        passes=1,
        warmup=1,
        order_seed=0,
        warmup_seed=1000,
        step=_sequential_step(clock, []),
        clock_ns=clock,
        deadline_ns=510 * 1_000_000,
    )
    assert outcome.status == "measured"
    assert outcome.passes_completed == 1


async def test_a_failed_iteration_stops_its_configuration() -> None:
    clock = _Clock()
    seen: list[list[int]] = []
    outcome = await bench.measure_configuration(
        bench.Configuration(index=0, batch_size=2, concurrency=1),
        sample=list(range(10)),
        passes=2,
        warmup=1,
        order_seed=0,
        warmup_seed=1000,
        step=_sequential_step(clock, seen, fail_on=4),
        clock_ns=clock,
        deadline_ns=10**18,
    )
    assert outcome.status == "stopped_on_failure"
    assert outcome.failed == 1
    assert outcome.error == "RuntimeError: CUDA out of memory"
    assert len(outcome.iterations) == 2
    assert len(seen) == 4


async def test_a_sample_smaller_than_a_batch_is_not_measured() -> None:
    outcome = await bench.measure_configuration(
        bench.Configuration(index=0, batch_size=8, concurrency=1),
        sample=[1, 2, 3],
        passes=1,
        warmup=1,
        order_seed=0,
        warmup_seed=1000,
        step=_sequential_step(_Clock(), []),
        clock_ns=lambda: 0,
        deadline_ns=10**18,
    )
    assert outcome.status == "not_measured"
    assert "does not fill one batch of 8" in (outcome.error or "")


# ------------------------------ summary ------------------------------


async def _outcome(batch: int, passes: int = 2) -> bench.ConfigurationOutcome:
    clock = _Clock()
    return await bench.measure_configuration(
        bench.Configuration(index=0, batch_size=batch, concurrency=1),
        sample=list(range(8)),
        passes=passes,
        warmup=1,
        order_seed=0,
        warmup_seed=1000,
        step=_sequential_step(clock, []),
        clock_ns=clock,
        deadline_ns=10**18,
    )


async def test_summary_at_batch_one_reports_per_request_latency_and_buckets() -> None:
    outcome = await _outcome(1)
    tokens = [100, 200, 600, 700, 100, 200, 600, 700]
    result = bench.summarize(
        outcome,
        boundaries=["model_only", "end_to_end"],
        tokens=tokens,
        length_of=lambda row: bench.bucket_label(tokens[row], [512]),
        items_of=lambda _row: None,
        bootstrap=None,
    )
    assert result.latency_unit == "ms_per_request"
    model, e2e = result.boundaries
    assert model.boundary == "model_only"
    assert model.latency.pooled["p95"] == pytest.approx(0.6)
    assert e2e.latency.pooled["p95"] == 1.0
    assert e2e.latency.n == 16
    # 16 one-millisecond iterations: 16 items in 16 ms.
    assert e2e.throughput.items_per_s == pytest.approx(1000.0)
    assert e2e.throughput.input_tokens_per_s == pytest.approx(sum(tokens) * 2 / 0.016)
    assert model.throughput.items_per_s == pytest.approx(16 / 0.0096)
    assert e2e.bootstrap is None
    assert e2e.by_length is not None
    assert [row.bucket for row in e2e.by_length] == ["0-512", ">512"]
    assert [row.n for row in e2e.by_length] == [8, 8]
    assert e2e.by_items is None


async def test_summary_above_batch_one_is_per_batch_and_has_no_buckets() -> None:
    outcome = await _outcome(4)
    result = bench.summarize(
        outcome,
        boundaries=["end_to_end"],
        tokens=[1] * 8,
        length_of=lambda _row: "0-512",
        items_of=lambda _row: None,
        bootstrap=(1, 50),
    )
    assert result.latency_unit == "ms_per_batch"
    (e2e,) = result.boundaries
    assert e2e.latency.pooled["p50"] == 4.0
    assert e2e.throughput.items_per_s == pytest.approx(1000.0)
    assert e2e.by_length is None
    assert e2e.bootstrap is not None
    assert e2e.bootstrap.resamples == 50


async def test_iteration_rows_carry_indices_and_numbers_only() -> None:
    outcome = await _outcome(2, passes=1)
    rows = bench.iteration_rows(
        [outcome],
        boundaries=["end_to_end"],
        tokens=[10] * 8,
        items=[3] * 8,
        length_of=lambda _row: "0-512",
        items_of=lambda _row: "0-5",
    )
    assert len(rows) == 4
    first = rows[0]
    assert set(first) == {
        "configuration",
        "batch_size",
        "concurrency",
        "pass",
        "iteration",
        "row_indices",
        "input_tokens",
        "max_input_tokens",
        "items",
        "length_bucket",
        "items_bucket",
        "pass_completed",
        "total_ms",
        "forward_ms",
        "output_tokens",
    }
    assert first["input_tokens"] == 20
    assert first["items"] == 6
    assert first["total_ms"] == pytest.approx(2.0)
    assert first["forward_ms"] == pytest.approx(1.2)
    assert first["pass_completed"] is True
    # A batch spans rows of several lengths, so it has no bucket of its own.
    assert first["length_bucket"] is None


# ------------------------------ spec ------------------------------

_CLASSIFY: dict[str, Any] = {
    "engine": "hf_classify",
    "pair_template": "{h}",
    "dtype": "bfloat16",
    "label_alias": {"ENTAILMENT": "entailed", "NOT_ENTAILMENT": "not_entailed"},
}
_CHAT: dict[str, Any] = {"engine": "vllm_chat", "max_tokens": 16, "temperature": 0.0}


def _bench_hp(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "engine_settings": _CLASSIFY,
        "boundaries": ["end_to_end", "model_only"],
        "batch_sizes": [1, 2],
        "warmup": 2,
        "passes": 2,
        "sample_seed": 11,
        "order_seed": 100,
        "warmup_seed": 200,
    }
    return base | overrides


def _spec_dict(**hp: Any) -> dict[str, Any]:
    return {
        "engine_version": SPEC_VERSION,
        "model_id": "org/nli",
        "model_revision": _SHA,
        "dataset_id": "org/workload",
        "dataset_commit_sha": _DATA_SHA,
        "split": "test",
        "column_mapping": {"p": "premises", "h": "hypothesis"},
        "template": "{p}",
        "hyperparams": _bench_hp(**hp),
        "output_repo_id": "org/bench-out",
        "progress_path": "progress.jsonl",
        "run_id": "run1",
    }


def test_the_spec_orders_boundaries_and_defaults_the_protocol() -> None:
    spec = br.BenchmarkSpec.model_validate(_spec_dict(warmup=50, passes=5))
    assert spec.hyperparams.boundaries == ["model_only", "end_to_end"]
    assert spec.hyperparams.concurrency == [1]
    assert spec.hyperparams.bootstrap_resamples is None
    assert spec.batch_spec().hyperparams.engine == "hf_classify"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"engine_settings": _CHAT, "batch_sizes": [1]}, "end_to_end only"),
        ({"engine_settings": _CHAT, "boundaries": ["end_to_end"]}, "batch_sizes must be"),
        (
            {
                "engine_settings": {"engine": "vllm_chat"},
                "boundaries": ["end_to_end"],
                "batch_sizes": [1],
            },
            "needs engine_settings.max_tokens",
        ),
        ({"concurrency": [1, 2]}, "concurrency must be"),
        ({"engine_settings": _CLASSIFY | {"scoring_batch_size": 4}}, "may not set"),
        ({"engine_settings": _CLASSIFY | {"slice_column": "tier"}}, "may not set"),
        (
            {
                "engine_settings": _CHAT | {"n": 4, "temperature": 0.7},
                "boundaries": ["end_to_end"],
                "batch_sizes": [1],
            },
            "may not set n",
        ),
        (
            {"engine_settings": {"engine": "hf_label_logits"}, "boundaries": ["end_to_end"]},
            "hf_label_logits",
        ),
        ({"batch_sizes": [1, 1]}, "lists a value twice"),
        ({"batch_sizes": [0]}, "from 1 to 1024"),
        ({"max_items": 20}, "need item_count_column"),
        ({"length_buckets": [512, 256]}, "strictly ascending"),
        ({"warmup": 0}, "greater than or equal to 1"),
        ({"require_gpu": "H100; rm -rf"}, "GPU name fragment"),
        ({"sample_size": 100_000, "passes": 20, "batch_sizes": [1, 2]}, "limit"),
        ({"engine_settings": {"engine": "hf_classify"}}, "engine 'hf_classify' needs"),
        ({"boundaries": ["model_only"]}, "must include end_to_end"),
        ({"order_seed": 5, "warmup_seed": 6, "passes": 5}, "at least `passes` apart"),
        ({"sample_size": 50, "item_cap": 10}, "cannot exceed item_cap"),
        ({"prefix_caching": True}, "'vllm_chat' only"),
    ],
)
def test_the_spec_refuses_what_it_cannot_measure(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        br.BenchmarkSpec.model_validate(_spec_dict(**overrides))


@pytest.mark.parametrize("field", ["model_revision", "dataset_commit_sha"])
def test_both_pins_are_required(field: str) -> None:
    data = _spec_dict()
    del data[field]
    with pytest.raises(ValidationError, match=field):
        br.BenchmarkSpec.model_validate(data)


def test_a_chat_benchmark_with_concurrency_is_accepted() -> None:
    spec = br.BenchmarkSpec.model_validate(
        _spec_dict(
            engine_settings=_CHAT, boundaries=["end_to_end"], batch_sizes=[1], concurrency=[1, 4]
        )
    )
    assert [c.concurrency for c in bench.configurations([1], spec.hyperparams.concurrency)] == [
        1,
        4,
    ]


# ------------------------------ environment ------------------------------


def test_gpu_rows_are_parsed_and_unknowns_stay_unknown() -> None:
    raw = (
        "0, NVIDIA H100 80GB HBM3, 81559, 550.54.15, Enabled, 1980, 1980, 700.00, 00000000:18:00.0\n"
        "1, NVIDIA H100 80GB HBM3, [N/A], 550.54.15, Disabled, [N/A], 1980, [N/A], 00000000:2A:00.0\n"
        "garbage line\n"
    )
    gpus = env.parse_gpus(raw)
    assert len(gpus) == 2
    assert gpus[0]["name"] == "NVIDIA H100 80GB HBM3"
    assert gpus[0]["memory_total_mib"] == 81559
    assert gpus[0]["power_limit_w"] == 700.0
    assert gpus[1]["memory_total_mib"] is None
    assert gpus[1]["clocks_sm_mhz"] is None
    full = {"gpus": gpus, "torch": None}
    assert env.gpu_matches(full, "h100")
    assert not env.gpu_matches(full, "A100")
    assert not env.gpu_matches({"gpus": [], "torch": None}, "H100")
    assert env.short_receipt(full)["count"] == 2


# ------------------------------ orchestration ------------------------------

_ROWS = [
    {"premises": [_SECRET_ROW_TEXT, "b"][: 1 + i % 2], "hypothesis": f"h{i}", "uid": f"u{i}"}
    for i in range(10)
] + [{"premises": ["x"] * 30, "hypothesis": "long", "uid": "u10"}]


class _FakeClassifier:
    """A loaded classifier that times itself on its own counter: 1 ms per row, 0.6 ms forward."""

    def __init__(self) -> None:
        self.labels = ["entailed", "not_entailed"]
        self.max_length = 32
        self.max_length_source = "model"
        self.dtype = "bfloat16"
        self.tf32 = {"matmul": True, "cudnn": True}
        self.device = "cpu"
        self.pad_token = "tokenizer"
        self.attention = "sdpa"
        self.now = 0
        self.calls: list[list[str]] = []

    def lengths(self, texts: Sequence[str], pairs: Sequence[str]) -> list[int | None]:
        del pairs
        return [len(t.split("\n")) + 3 for t in texts]

    def forward_timer(self) -> Any:
        return types.SimpleNamespace(method="perf_counter_ns_cpu")

    def timed_forward(
        self, texts: Sequence[str], pairs: Sequence[str], *, clock: Any
    ) -> tuple[int, int, float]:
        del clock, pairs
        self.calls.append(list(texts))
        start = self.now
        self.now += 1_000_000 * len(texts)
        return start, self.now, 0.6 * len(texts)


class _Hub:
    files: ClassVar[tuple[str, ...]] = (".gitattributes",)
    calls: ClassVar[list[tuple[str, Any]]] = []
    uploaded: ClassVar[dict[str, bytes]] = {}

    def __init__(self, *, token: Any) -> None:
        del token

    async def create_repo(self, repo_id: str, **_kw: Any) -> None:
        _Hub.calls.append(("create_repo", repo_id))

    async def is_private(self, repo_id: str, **_kw: Any) -> bool:
        return True

    async def list_repo_files(self, repo_id: str, **_kw: Any) -> tuple[str, ...]:
        return _Hub.files

    async def upload_files(self, files: dict[str, Path], repo_id: str, **kwargs: Any) -> None:
        _Hub.calls.append(("upload_files", (sorted(files), repo_id, kwargs["commit_message"])))
        _Hub.uploaded = {name: path.read_bytes() for name, path in files.items()}


def _environment(name: str = "NVIDIA H100 80GB HBM3") -> dict[str, Any]:
    return {
        "nvidia_smi": True,
        "gpus": [{"index": 0, "name": name, "memory_total_mib": 81559, "driver_version": "550"}],
        "compute_processes_before": 0,
        "cuda_visible_devices": "0",
        "torch": {"available": True, "cuda_runtime": "12.8", "device_name": name},
        "packages": {"torch": "2.9.0"},
    }


@pytest.fixture
def bench_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Wire `main` for a benchmark run; return what the run touched."""
    record: dict[str, Any] = {"classifier": _FakeClassifier(), "rows": list(_ROWS)}
    monkeypatch.delenv(LEGACY_TOKEN_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    secrets = tmp_path / ".secrets.json"
    secrets.write_text(json.dumps({"HF_TOKEN": _TOKEN}))
    secrets.chmod(0o600)
    monkeypatch.setenv("FORGE_SECRETS_FILE", str(secrets))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    _Hub.files = (".gitattributes",)
    _Hub.calls = []
    _Hub.uploaded = {}
    monkeypatch.setattr(ir, "HFHubClient", _Hub)
    monkeypatch.setattr(br, "HFHubClient", _Hub)
    record["environment"] = _environment()

    def _rows(spec: Any, token: Any) -> list[dict[str, Any]]:
        record["token"] = token
        record["loaded_at"] = spec.dataset_commit_sha
        return record["rows"]

    async def _fetch(spec: Any, token: Any) -> Path:
        snapshot = tmp_path / "models--org--nli" / "snapshots" / _SHA
        snapshot.mkdir(parents=True, exist_ok=True)
        return snapshot

    def _load(snapshot: Path, **kwargs: Any) -> _FakeClassifier:
        record["load"] = kwargs
        return record["classifier"]

    def _write(rows: list[dict[str, Any]], outdir: Path) -> Path:
        record["timings"] = rows
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / br.TIMINGS_FILENAME
        path.write_bytes(b"parquet")
        return path

    monkeypatch.setattr(br, "_load_rows", _rows)
    monkeypatch.setattr(br, "_fetch_model", _fetch)
    monkeypatch.setattr(br, "load_classifier", _load)
    monkeypatch.setattr(br, "_write_timings", _write)

    def _receipt(*, use_torch: bool) -> dict[str, Any]:
        del use_torch
        return record["environment"]

    monkeypatch.setattr(br, "receipt", _receipt)
    # An idle GPU by default: nvidia-smi lists only this runner's own process.
    record["pids"] = lambda: {os.getpid()}
    record["probes"] = 0

    def _pids() -> set[int] | None:
        record["probes"] += 1
        return record["pids"]()

    monkeypatch.setattr(br, "compute_pids", _pids)
    monkeypatch.setattr(
        br,
        "gpu_state",
        lambda: [
            {
                "index": "0",
                "clocks_sm_mhz": 1980.0,
                "clocks_mem_mhz": 2619.0,
                "power_limit_w": 700.0,
                "power_draw_w": 120.0,
                "temperature_c": 41.0,
                "persistence_mode": "Enabled",
            }
        ],
    )
    record["progress"] = tmp_path / "progress.jsonl"
    record["results_dir"] = tmp_path / "strata-benchmark-results" / "run1"
    record["workdir"] = tmp_path
    return record


def _launch(monkeypatch: pytest.MonkeyPatch, record: dict[str, Any], **hp: Any) -> None:
    spec = _spec_dict(**hp) | {"progress_path": str(record["progress"])}
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(spec))


def _events(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in record["progress"].read_text().splitlines() if line]


async def test_a_classifier_benchmark_measures_the_grid_and_pushes_one_commit(
    monkeypatch: pytest.MonkeyPatch, bench_run: dict[str, Any]
) -> None:
    _launch(
        monkeypatch,
        bench_run,
        max_input_tokens=20,
        item_count_column="premises",
        max_items=20,
        length_buckets=[4, 8],
        item_buckets=[5, 20],
        bootstrap_resamples=100,
        require_gpu="H100",
    )
    assert await br.main() == 0

    # The long row (30 premises) is excluded and counted, never timed.
    classifier = bench_run["classifier"]
    timed_texts = {text for texts in classifier.calls for text in texts}
    assert all(len(t.split("\n")) <= 2 for t in timed_texts)
    # Two passes over 10 rows at batch 1 and 2, with 2 warmups before each pass.
    assert len(classifier.calls) == 2 * (2 + 10) + 2 * (2 + 5)
    assert bench_run["loaded_at"] == _DATA_SHA
    assert bench_run["token"] == _TOKEN

    uploads = [call for call in _Hub.calls if call[0] == "upload_files"]
    assert uploads == [
        (
            "upload_files",
            (
                ["manifest.json", "summary.json", "timings.parquet"],
                "org/bench-out",
                "benchmark measurements",
            ),
        )
    ]
    summary = json.loads(_Hub.uploaded["summary.json"])
    workdir_copy = json.loads((bench_run["workdir"] / br.WORKDIR_SUMMARY_FILENAME).read_text())
    assert workdir_copy == summary
    # Validated from its text, as the control plane reads it.
    parsed = bench.BenchmarkSummary.model_validate_json(_Hub.uploaded["summary.json"])
    assert parsed.complete
    assert parsed.sample == {"rows": 11, "eligible": 10, "sampled": 10}
    assert parsed.excluded["over_max_input_tokens"] == 0
    assert parsed.excluded["over_model_limit"] == 1
    assert parsed.gpu["name"] == "NVIDIA H100 80GB HBM3"
    one, two = parsed.configurations
    assert (one.batch_size, one.latency_unit, one.iterations_measured) == (1, "ms_per_request", 20)
    assert (two.batch_size, two.latency_unit, two.iterations_measured) == (2, "ms_per_batch", 10)
    model, e2e = one.boundaries
    assert model.latency.pooled["p95"] == pytest.approx(0.6)
    assert e2e.latency.pooled["p95"] == pytest.approx(1.0)
    assert e2e.bootstrap is not None
    assert e2e.by_items is not None
    assert [row.bucket for row in e2e.by_length or ()] == ["0-4", "5-8"]
    assert parsed.cold_start["first_iteration_ms"] == pytest.approx(1.0)
    # The headline: the median of the per-pass end-to-end p95, with min and max, readable.
    assert parsed.headline_readable
    assert parsed.headline_unreadable_reason is None
    assert parsed.model_path == "reference_automodel"
    assert parsed.gpu_idle_check == "checked"
    assert (one.headline.p95_median_ms, one.headline.passes) == (pytest.approx(1.0), 2)
    assert one.headline.unit == "ms_per_request"
    assert two.headline.unit == "ms_per_batch"
    assert (model.role, e2e.role) == ("diagnostic", "headline")
    # Clocks, power and temperature at the start and end of every pass of every configuration.
    assert len(parsed.gpu_telemetry) == 2 * 2 * 2
    assert {t.at for t in parsed.gpu_telemetry} == {"start", "end"}
    assert parsed.gpu_telemetry[0].gpus[0].temperature_c == 41.0

    manifest = json.loads(_Hub.uploaded["manifest.json"])
    assert manifest["kind"] == "benchmark"
    assert manifest["model"] == {"id": "org/nli", "requested_revision": _SHA, "commit": _SHA}
    assert manifest["dataset"]["commit"] == _DATA_SHA
    protocol = manifest["benchmark"]
    assert protocol["warmup"]["scope"].startswith("before_every_timed_pass")
    assert set(protocol["boundaries"]) == {"model_only", "end_to_end"}
    assert "CUDA events" in protocol["boundaries"]["model_only"]
    assert "ONE torch.cuda.synchronize()" in protocol["boundaries"]["end_to_end"]
    assert protocol["model_path"] == "reference_automodel"
    assert protocol["seeds"] == {
        "sample_seed": 11,
        "order_seed": 100,
        "warmup_seed": 200,
        "bootstrap_seed": 0,
    }
    assert protocol["item_cap"] == 100_000
    assert manifest["headline_readable"] is True
    assert manifest["engine_facts"]["model_path"] == "reference_automodel"
    assert "third-party reference" in manifest["engine_facts"]["label"]
    assert manifest["environment"]["gpus"][0]["name"] == "NVIDIA H100 80GB HBM3"
    assert manifest["engine_facts"]["attention"] == "sdpa"
    assert manifest["compute"]["gpu_count"] == 1
    assert "pair_template" not in manifest["hyperparams"]["engine_settings"]

    events = _events(bench_run)
    end = next(e for e in events if e["kind"] == "end")
    assert end["message"] == "org/bench-out"
    assert end["metrics"]["iterations_measured"] == 30.0
    assert end["metrics"]["complete"] == 1.0
    assert end["metrics"]["gpu_count"] == 1.0
    assert "gpu_seconds" in end["metrics"]
    start = next(e for e in events if e["kind"] == "start")
    assert start["total_steps"] == 30
    # No telemetry probe while timing: step events carry no GPU counters.
    steps = [e for e in events if e["kind"] == "step"]
    assert steps
    assert all(not any(k.startswith("gpu_") for k in e["metrics"]) for e in steps)

    # No row text and no token anywhere the run wrote.
    written = (
        json.dumps(bench_run["timings"])
        + bench_run["progress"].read_text()
        + "".join(b.decode() for name, b in _Hub.uploaded.items() if name != "timings.parquet")
    )
    assert _SECRET_ROW_TEXT not in written
    assert _TOKEN not in written


async def test_a_failed_iteration_is_reported_redacted_and_the_run_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch, bench_run: dict[str, Any]
) -> None:
    classifier = bench_run["classifier"]
    real = classifier.timed_forward

    def _forward(texts: Sequence[str], pairs: Sequence[str], **kw: Any) -> Any:
        if len(texts) == 2:
            raise RuntimeError(f"CUDA out of memory while holding {_TOKEN}")
        return real(texts, pairs, **kw)

    classifier.timed_forward = _forward
    _launch(monkeypatch, bench_run)
    assert await br.main() == 0
    summary = json.loads(_Hub.uploaded["summary.json"])
    assert summary["complete"] is False
    assert "stopped on a failed iteration" in summary["incomplete_reason"]
    assert summary["incomplete_codes"] == ["stopped_on_failure"]
    assert summary["headline_readable"] is False
    one, two = summary["configurations"]
    assert one["status"] == "measured"
    assert two["status"] == "stopped_on_failure"
    assert two["iterations_failed"] == 1
    assert "CUDA out of memory" in two["error"]
    assert _TOKEN not in json.dumps(summary)


async def test_a_gpu_requirement_the_machine_does_not_meet_is_refused(
    monkeypatch: pytest.MonkeyPatch, bench_run: dict[str, Any]
) -> None:
    bench_run["environment"] = _environment("NVIDIA A10G")
    _launch(monkeypatch, bench_run, require_gpu="H100")
    assert await br.main() == 1
    error = next(e for e in _events(bench_run) if e["kind"] == "error")
    assert "requires a GPU named like 'H100'" in error["message"]
    # Refused from nvidia-smi's answer, before the workload or the model was downloaded.
    assert "token" not in bench_run
    assert bench_run["classifier"].calls == []


async def test_an_output_repo_holding_files_is_refused_before_anything_loads(
    monkeypatch: pytest.MonkeyPatch, bench_run: dict[str, Any]
) -> None:
    _Hub.files = (".gitattributes", "manifest.json")
    _launch(monkeypatch, bench_run)
    assert await br.main() == 1
    assert "token" not in bench_run
    assert (
        "already holds 1 file"
        in next(e for e in _events(bench_run) if e["kind"] == "error")["message"]
    )


async def test_the_cap_marks_the_results_incomplete(
    monkeypatch: pytest.MonkeyPatch, bench_run: dict[str, Any]
) -> None:
    _launch(monkeypatch, bench_run, wall_clock_cap_s=1e-9)
    assert await br.main() == 1  # nothing measured at all is a failed run
    summary = json.loads((bench_run["results_dir"] / br.SUMMARY_FILENAME).read_text())
    assert summary["complete"] is False
    assert "wall-clock cap" in summary["incomplete_reason"]
    assert [c["status"] for c in summary["configurations"]] == ["wall_clock_cap", "not_measured"]
    assert summary["incomplete_codes"] == ["wall_clock_cap", "not_measured"]
    assert "wall-clock cap" in summary["configurations"][1]["error"]


async def test_without_a_write_token_the_measurements_stay_on_the_vm(
    monkeypatch: pytest.MonkeyPatch, bench_run: dict[str, Any], tmp_path: Path
) -> None:
    (tmp_path / ".secrets.json").write_text("{}")
    _launch(monkeypatch, bench_run)
    assert await br.main() == 0
    assert not [c for c in _Hub.calls if c[0] == "upload_files"]
    end = next(e for e in _events(bench_run) if e["kind"] == "end")
    assert end["message"] == str(bench_run["results_dir"])
    assert (bench_run["results_dir"] / "manifest.json").is_file()


async def test_a_chat_benchmark_serves_the_model_and_times_requests(
    monkeypatch: pytest.MonkeyPatch, bench_run: dict[str, Any]
) -> None:
    served: dict[str, Any] = {}

    @contextlib.asynccontextmanager
    async def _serving(_backend: Any, task: Any, **_kw: Any) -> AsyncGenerator[Any]:
        served["task"] = task

        async def _alive() -> bool:
            return True

        yield types.SimpleNamespace(base_url="http://127.0.0.1:8000/v1", is_alive=_alive)

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            served["client"] = kwargs
            served["requests"] = 0

        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            served["requests"] += 1
            served["sampling"] = kwargs
            return types.SimpleNamespace(usage=types.SimpleNamespace(output_tokens=7))

    async def _config(spec: Any, token: Any) -> tuple[dict[str, Any], str]:
        return {"model_type": "qwen3"}, _SHA

    async def _no_len(*_a: Any) -> None:
        return None

    monkeypatch.setattr(br, "serving_endpoint", _serving)
    monkeypatch.setattr(br, "LLMClient", _Client)
    monkeypatch.setattr(br, "_read_model_config", _config)

    def _lengths(_snapshot: Path, prompts: Sequence[str], kwargs: Any = None) -> list[int | None]:
        served["template_kwargs"] = kwargs
        return [5] * len(prompts)

    monkeypatch.setattr(br, "_chat_lengths", _lengths)
    monkeypatch.setattr(br, "_served_max_model_len", _no_len)
    _launch(
        monkeypatch,
        bench_run,
        engine_settings=_CHAT,
        boundaries=["end_to_end"],
        batch_sizes=[1],
        concurrency=[1, 2],
        passes=1,
        warmup=1,
    )
    assert await br.main() == 0
    assert served["client"]["retry_max_attempts"] == 1
    assert "cache" not in served["client"]
    assert served["sampling"]["max_tokens"] == 16
    assert served["task"].env["HF_HUB_OFFLINE"] == "1"
    # Every pass replays the same prompts: the server's prefix cache is off unless declared.
    assert "--no-enable-prefix-caching" in served["task"].run
    assert "HF_TOKEN" not in served["task"].env
    # 11 rows, two configurations, one warmup each.
    assert served["requests"] == 2 * (1 + 11)
    summary = json.loads(_Hub.uploaded["summary.json"])
    assert summary["engine"] == "vllm_chat"
    assert summary["boundaries"] == ["end_to_end"]
    first = summary["configurations"][0]["boundaries"][0]
    assert first["throughput"]["output_tokens_per_s"] is not None
    manifest = json.loads(_Hub.uploaded["manifest.json"])
    assert manifest["engine_facts"]["prefix_caching"] is False
    assert summary["model_path"] == "vllm_chat"


def test_item_counts_read_list_lengths_and_refuse_a_missing_column() -> None:
    rows = [{"p": ["a", "b"]}, {"p": None}, {"p": "text"}]
    assert br._item_counts(rows, "p") == [2, None, None]  # pyright: ignore[reportPrivateUsage]
    assert br._item_counts(rows, None) is None  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RunError, match="not a column"):
        br._item_counts(rows, "q")  # pyright: ignore[reportPrivateUsage]


# ------------------------------ rulings: headline, GPU, workload ------------------------------


async def test_a_truncated_pass_never_enters_the_headline_or_the_percentiles() -> None:
    clock = _Clock()
    seen: list[list[int]] = []
    # Pass 0: 1 warmup (500 ms cold) + 10 timed at 1 ms; pass 1 is cut by the cap mid-way.
    outcome = await bench.measure_configuration(
        bench.Configuration(index=0, batch_size=1, concurrency=1),
        sample=list(range(10)),
        passes=3,
        warmup=1,
        order_seed=0,
        warmup_seed=1000,
        step=_sequential_step(clock, seen),
        clock_ns=clock,
        deadline_ns=(500 + 10 + 1 + 4) * 1_000_000,
    )
    assert outcome.status == "wall_clock_cap"
    assert outcome.passes_completed == 1
    assert {i.pass_index for i in outcome.iterations} == {0, 1}
    result = bench.summarize(
        outcome,
        boundaries=["end_to_end"],
        tokens=[1] * 10,
        length_of=lambda _row: None,
        items_of=lambda _row: None,
        bootstrap=None,
    )
    assert result.passes_completed == 1
    assert result.iterations_summarized == 10
    assert result.iterations_measured > 10
    assert result.headline.passes == 1
    (e2e,) = result.boundaries
    assert e2e.latency.n == 10
    assert [p.pass_index for p in e2e.latency.passes] == [0]
    rows = bench.iteration_rows(
        [outcome],
        boundaries=["end_to_end"],
        tokens=[1] * 10,
        items=None,
        length_of=lambda _row: None,
        items_of=lambda _row: None,
    )
    assert {r["pass"] for r in rows if not r["pass_completed"]} == {1}


async def test_the_headline_is_the_median_of_per_pass_p95_with_min_and_max() -> None:
    passes = [[1.0] * 19 + [10.0], [1.0] * 19 + [20.0], [1.0] * 19 + [40.0]]
    iterations = [
        bench.Iteration(
            configuration=0,
            pass_index=p,
            iteration=i,
            rows=(i,),
            start_ns=0,
            end_ns=int(value * 1_000_000),
            forward_ms=None,
            output_tokens=None,
        )
        for p, values in enumerate(passes)
        for i, value in enumerate(values)
    ]
    outcome = bench.ConfigurationOutcome(
        configuration=bench.Configuration(index=0, batch_size=1, concurrency=1),
        status="measured",
        passes_planned=3,
        passes_completed=3,
        iterations=iterations,
    )
    result = bench.summarize(
        outcome,
        boundaries=["end_to_end"],
        tokens=[1] * 20,
        length_of=lambda _row: None,
        items_of=lambda _row: None,
        bootstrap=None,
    )
    p95s = [env_percentile(v) for v in passes]
    assert result.headline.p95_median_ms == pytest.approx(sorted(p95s)[1])
    assert result.headline.p95_min_ms == pytest.approx(min(p95s))
    assert result.headline.p95_max_ms == pytest.approx(max(p95s))


def env_percentile(values: list[float]) -> float:
    numpy: Any = pytest.importorskip("numpy")
    return float(numpy.percentile(values, 95))


def test_a_summary_refuses_nan_and_infinity() -> None:
    with pytest.raises(ValidationError):
        bench.Headline(
            p95_median_ms=math.nan, p95_min_ms=1.0, p95_max_ms=1.0, passes=1, unit="ms_per_request"
        )
    with pytest.raises(ValidationError):
        bench.GpuState(temperature_c=math.inf)


async def test_a_busy_gpu_at_the_start_refuses_the_run_before_anything_loads(
    monkeypatch: pytest.MonkeyPatch, bench_run: dict[str, Any]
) -> None:
    bench_run["pids"] = lambda: {os.getpid(), 999_999}
    _launch(monkeypatch, bench_run)
    assert await br.main() == 1
    error = next(e for e in _events(bench_run) if e["kind"] == "error")
    assert "needs the GPU to itself" in error["message"]
    assert "token" not in bench_run
    assert bench_run["classifier"].calls == []


async def test_a_gpu_taken_between_passes_stops_the_run_as_incomplete_gpu_busy(
    monkeypatch: pytest.MonkeyPatch, bench_run: dict[str, Any]
) -> None:
    # Idle at the start and before pass 0; another process appears before pass 1.
    def _pids() -> set[int]:
        return {os.getpid()} if bench_run["probes"] <= 2 else {os.getpid(), 4242}

    bench_run["pids"] = _pids
    _launch(monkeypatch, bench_run)
    assert await br.main() == 0
    summary = bench.BenchmarkSummary.model_validate_json(_Hub.uploaded["summary.json"])
    assert summary.complete is False
    assert summary.headline_readable is False
    assert summary.headline_unreadable_reason == "an incomplete run has no T13 read"
    assert summary.incomplete_codes == ("gpu_busy", "not_measured")
    assert "gpu_busy" in (summary.incomplete_reason or "")
    one, two = summary.configurations
    assert (one.status, one.passes_completed) == ("gpu_busy", 1)
    assert two.status == "not_measured"
    assert "gpu_busy" in (two.error or "")


async def test_without_nvidia_smi_the_headline_is_not_a_t13_read(
    monkeypatch: pytest.MonkeyPatch, bench_run: dict[str, Any]
) -> None:
    bench_run["pids"] = lambda: None
    _launch(monkeypatch, bench_run)
    assert await br.main() == 0
    summary = bench.BenchmarkSummary.model_validate_json(_Hub.uploaded["summary.json"])
    assert summary.complete is True
    assert summary.gpu_idle_check == "unavailable"
    assert summary.headline_readable is False
    assert "could not be checked" in (summary.headline_unreadable_reason or "")


async def test_every_row_is_read_and_a_split_past_item_cap_is_refused(
    monkeypatch: pytest.MonkeyPatch, bench_run: dict[str, Any]
) -> None:
    limits: list[Any] = []
    real = br._load_rows  # pyright: ignore[reportPrivateUsage]

    def _rows(spec: Any, token: Any) -> list[dict[str, Any]]:
        limits.append(spec.hyperparams.row_limit)
        return real(spec, token)

    monkeypatch.setattr(br, "_load_rows", _rows)
    _launch(monkeypatch, bench_run, item_cap=10)
    assert await br.main() == 1
    assert limits == [11]
    error = next(e for e in _events(bench_run) if e["kind"] == "error")
    assert "more than item_cap (10) rows" in error["message"]
    assert bench_run["classifier"].calls == []


async def test_a_trained_verifier_is_timed_through_its_own_request(
    monkeypatch: pytest.MonkeyPatch, bench_run: dict[str, Any]
) -> None:
    verifier = types.SimpleNamespace(
        max_length=32,
        max_length_source="artifact",
        dtype="bfloat16",
        tf32={"matmul": False, "cudnn": False},
        device="cpu",
        pad_token="tokenizer",
        attention="sdpa",
        labels=["entailed", "contradicted", "neutral"],
        verifier_sha256="ab" * 32,
        heads=["flag", "support"],
        rendering=types.SimpleNamespace(decoder=False),
        temperature=None,
        forward_timer=lambda: types.SimpleNamespace(method="perf_counter_ns_cpu"),
    )
    loaded: dict[str, Any] = {}
    timed: list[list[Any]] = []

    def _load(snapshot: Path, **kwargs: Any) -> Any:
        loaded.update(kwargs)
        return verifier

    class _Lengths:
        def __init__(self, scorer: Any) -> None:
            assert scorer is verifier

        def lengths(self, inputs: Any) -> list[int | None]:
            return [len(c[0]) + 3 for c in inputs.cells]

    now = [0]

    def _request(scorer: Any, cells: list[Any], *, clock: Any) -> tuple[int, int, float]:
        assert scorer is verifier
        timed.append(cells)
        start = now[0]
        now[0] += 2_000_000
        return start, now[0], 1.5

    monkeypatch.setattr(br, "load_verifier", _load)
    monkeypatch.setattr(br, "VerifierRunScorer", _Lengths)
    monkeypatch.setattr(br, "timed_verifier_request", _request)
    spec = _spec_dict(
        engine_settings={
            "engine": "hf_classify",
            "artifact": "strata_verifier",
            "pair_template": "{hypothesis}",
            "dtype": "bfloat16",
        },
        batch_sizes=[1],
    ) | {
        "template": "{premises}",
        "column_mapping": {"premises": "premises", "hypothesis": "hypothesis"},
        "progress_path": str(bench_run["progress"]),
    }
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(spec))
    assert await br.main() == 0
    assert loaded["dtype"] == "bfloat16"
    assert loaded["apply_calibration"] is False
    # The raw cells reach the timed request: rendering is inside the boundary.
    assert all(isinstance(c[0][0], list) for c in timed)
    summary = bench.BenchmarkSummary.model_validate_json(_Hub.uploaded["summary.json"])
    assert summary.model_path == "strata_verifier"
    (config,) = summary.configurations
    model, e2e = config.boundaries
    assert e2e.latency.pooled["p95"] == pytest.approx(2.0)
    assert model.latency.pooled["p95"] == pytest.approx(1.5)
    manifest = json.loads(_Hub.uploaded["manifest.json"])
    facts = manifest["engine_facts"]
    assert (facts["model_path"], facts["dtype"], facts["attention"]) == (
        "strata_verifier",
        "bfloat16",
        "sdpa",
    )
    assert "3-way softmax" in manifest["benchmark"]["boundaries"]["end_to_end"]


async def test_a_model_server_that_dies_mid_run_still_writes_what_was_measured(
    monkeypatch: pytest.MonkeyPatch, bench_run: dict[str, Any]
) -> None:
    state: dict[str, Any] = {"alive": True, "requests": 0}

    @contextlib.asynccontextmanager
    async def _serving(_backend: Any, task: Any, **_kw: Any) -> AsyncGenerator[Any]:
        async def _alive() -> bool:
            return state["alive"]

        yield types.SimpleNamespace(base_url="http://127.0.0.1:8000/v1", is_alive=_alive)

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            state["requests"] += 1
            if state["requests"] > 12:
                state["alive"] = False
                raise ConnectionError("connection refused")
            return types.SimpleNamespace(usage=types.SimpleNamespace(output_tokens=3))

    async def _config(spec: Any, token: Any) -> tuple[dict[str, Any], str]:
        return {"model_type": "qwen3"}, _SHA

    async def _no_len(*_a: Any) -> None:
        return None

    monkeypatch.setattr(br, "serving_endpoint", _serving)
    monkeypatch.setattr(br, "LLMClient", _Client)
    monkeypatch.setattr(br, "_read_model_config", _config)

    def _lengths(_snapshot: Path, prompts: Sequence[str], kwargs: Any = None) -> list[int | None]:
        del kwargs
        return [5] * len(prompts)

    monkeypatch.setattr(br, "_chat_lengths", _lengths)
    monkeypatch.setattr(br, "_served_max_model_len", _no_len)
    _launch(
        monkeypatch,
        bench_run,
        engine_settings=_CHAT,
        boundaries=["end_to_end"],
        batch_sizes=[1],
        concurrency=[1, 2, 4],
        passes=1,
        warmup=1,
    )
    assert await br.main() == 1
    error = next(e for e in _events(bench_run) if e["kind"] == "error")
    assert "model server died" in error["message"]
    summary = bench.BenchmarkSummary.model_validate_json(_Hub.uploaded["summary.json"])
    assert [c.status for c in summary.configurations] == [
        "measured",
        "stopped_on_failure",
        "not_measured",
    ]
    assert "server_died" in summary.incomplete_codes
    assert summary.headline_readable is False


def test_visible_devices_follow_cuda_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    assert env.visible_tokens(None) is None
    assert env.visible_tokens("") == []
    assert env.visible_tokens("1, GPU-abc,x,2") == ["1", "GPU-abc"]
    answers = {
        "--query-gpu=index,uuid": "0, GPU-aaa\n1, GPU-bbb\n2, GPU-ccc\n",
        "--query-compute-apps=gpu_uuid,pid": "GPU-aaa, 100\nGPU-bbb, 200\nGPU-ccc, 300\n",
    }

    def _smi(args: list[str]) -> str | None:
        return answers.get(args[0])

    monkeypatch.setattr(env, "_nvidia_smi", _smi)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,GPU-cc")
    assert env.visible_indices() == {"1", "2"}
    # A process on a GPU this run cannot see is not competing with it.
    assert env.compute_pids() == {200, 300}
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert env.compute_pids() == {100, 200, 300}


def test_own_processes_include_the_runner_itself() -> None:
    assert os.getpid() in env.own_processes(os.getpid())


async def test_a_summary_must_say_complete_only_when_every_configuration_was_measured(
    monkeypatch: pytest.MonkeyPatch, bench_run: dict[str, Any]
) -> None:
    _launch(monkeypatch, bench_run, wall_clock_cap_s=1e-9)
    assert await br.main() == 1
    text = (bench_run["results_dir"] / br.SUMMARY_FILENAME).read_text()
    data = json.loads(text)
    bench.BenchmarkSummary.model_validate_json(text)
    with pytest.raises(ValidationError, match="complete must say"):
        bench.BenchmarkSummary.model_validate_json(json.dumps(data | {"complete": True}))
    with pytest.raises(ValidationError, match="readable only"):
        bench.BenchmarkSummary.model_validate_json(json.dumps(data | {"headline_readable": True}))
    nan = text.replace('"p95_median_ms": null', '"p95_median_ms": NaN', 1)
    assert nan != text
    with pytest.raises(ValidationError):
        bench.BenchmarkSummary.model_validate_json(nan)
