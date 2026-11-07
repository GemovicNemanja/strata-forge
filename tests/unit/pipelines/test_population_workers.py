"""Operational native/import population workers with actual local Parquet and numerical metrics."""

from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path
from typing import Any, Literal

import pytest

from strata_forge.pipelines import SPEC_VERSION
from strata_forge.pipelines._evaluation_access import ReadDecision
from strata_forge.pipelines._population import PopulationError
from strata_forge.pipelines.eval_runner import run_population_evaluation
from strata_forge.pipelines.evaluation_population_spec import EvaluationPopulationSpecV2
from strata_forge.pipelines.inference_runner import execute_population_scoring
from strata_forge.pipelines.scoring_population_spec import InferencePopulationSpecV2
from tests.unit.pipelines.test_scoring import fake_hf as _fake_hf

hf_fixture = _fake_hf


@pytest.fixture
def fake_hf(hf_fixture: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(
        sys.modules["torch"],
        "cuda",
        types.SimpleNamespace(is_available=lambda: False),
        raising=False,
    )
    return hf_fixture


# Serialized synthetic jobs intentionally contain heterogeneous source-column/model settings.
FIXTURE: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures/population_v2.json").read_text()
)
ALIASES = {"contradiction": "E", "entailment": "C", "neutral": "U"}


def job(name: str) -> Any:
    raw = copy.deepcopy(FIXTURE["jobs"][name])
    raw["job"]["engine_version"] = SPEC_VERSION
    if name == "inference":
        raw["job"]["hyperparams"].update(max_model_len=7, label_alias=ALIASES)
        raw["selection"]["row_limit"] = 5
        return InferencePopulationSpecV2.model_validate(raw)
    if name == "native_evaluation":
        raw["job"]["label_aliases"] = ALIASES
        raw["job"]["rendering"]["max_length"] = 7
    raw["job"]["evaluation_sets"][0]["selection"]["row_limit"] = 5
    return EvaluationPopulationSpecV2.model_validate(raw)


def rows() -> tuple[dict[str, object], ...]:
    result = []
    for item_id, tier, premise, premises, gold, eligible in [
        ("short", "a", "first", ["first"], "E", True),
        ("exact", "a", "first first first", ["first first"], "C", True),
        ("overflow", "a", "first " * 12, ["first " * 12], "E", True),
        ("excluded", "b", "EXCLUDED" + "x" * 200001, 42, "C", False),
        ("invalid", "b", "x" * 200001, 42, "U", True),
    ]:
        result.append(
            {
                "item_id": item_id,
                "tier": tier,
                "within_limits": eligible,
                "premise_text": premise,
                "premises": premises,
                "hypothesis": "second",
                "gold": gold,
            }
        )
    return tuple(result)


class Policy:
    def __init__(
        self,
        *,
        wrong_digest: str | None = None,
        custody: Literal["unsealed", "unknown"] = "unsealed",
    ):
        self.calls: list[tuple[str, str, str]] = []
        self.wrong_digest = wrong_digest
        self.custody = custody

    async def decide(
        self, ref: str, *, repo_type: Literal["model", "dataset"], spec_sha256: str
    ) -> ReadDecision:
        self.calls.append((ref, repo_type, spec_sha256))
        return ReadDecision(
            ref=ref,
            repo_type=repo_type,
            spec_sha256=self.wrong_digest or spec_sha256,
            authorized=self.custody == "unsealed",
            custody=self.custody,
            policy_version="fictional-v2",
        )


class Reader:
    def __init__(
        self,
        gold: tuple[dict[str, object], ...] | None = None,
        predictions: tuple[dict[str, object], ...] = (),
    ):
        self.gold = rows() if gold is None else gold
        self.predictions = predictions
        self.calls: list[tuple[str, str]] = []
        self.metadata: dict[tuple[str, str], dict[str, object]] = {}

    async def prepare(self, ref: str, *, model: bool = False) -> None:
        self.calls.append(("prepare", ref))

    async def rows(
        self, ref: str, config: str, split: str, selection: object, columns: tuple[str, ...]
    ) -> tuple[dict[str, object], ...]:
        self.calls.append(("rows", ref))
        return self.predictions if "predictions" in ref else self.gold

    async def json(
        self, ref: str, path: str, *, repo_type: Literal["model", "dataset"] = "dataset"
    ) -> dict[str, object]:
        self.calls.append(("json", ref))
        return self.metadata[(ref, path)]


@pytest.mark.parametrize("name", ["inference", "native_evaluation"])
async def test_native_worker_exact_partition_and_own_outer_authority(
    name: str, fake_hf: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow.parquet as pq

    value = job(name)
    source = Reader()
    policy = Policy()
    encoded = []
    original = fake_hf.tokenizer.encode

    def observe(text: str, *, add_special_tokens: bool) -> list[int]:
        encoded.append(text)
        return original(text, add_special_tokens=add_special_tokens)

    monkeypatch.setattr(fake_hf.tokenizer, "encode", observe)
    output = tmp_path / name
    if name == "inference":
        result = await execute_population_scoring(
            value, reader=source, policy=policy, output_dir=output
        )
        path = output / "results.parquet"
    else:
        result = await run_population_evaluation(
            value, reader=source, policy=policy, output_dir=output
        )
        path = output / "evaluations/threeway/items.parquet"
    assert {digest for _, _, digest in policy.calls} == {value.digest()}
    assert result.spec_sha256 == value.digest()
    table = pq.read_table(path).to_pylist()
    assert [row["item_id"] for row in table] == [
        "short",
        "exact",
        "overflow",
        "excluded",
        "invalid",
    ]
    assert [row["outcome"] for row in table] == [
        "scored",
        "scored",
        "overflow",
        "suite_excluded",
        "invalid",
    ]
    assert [row["metric_eligible"] for row in table] == [True, True, False, False, True]
    assert table[2]["token_count"] > table[2]["model_limit"] == 7
    assert table[3]["token_count"] is None
    assert all("EXCLUDED" not in text for text in encoded)
    assert fake_hf.models[0].forward_calls == 1
    assert result.counts.rows_in == 5
    assert result.counts.rows_scored_success == 2
    assert result.counts.rows_failed_or_invalid == 1
    assert result.counts.rows_metric_population == 3
    assert result.counts.rows_suite_excluded == result.counts.rows_overflow_dropped == 1
    assert all(not call["truncation"] for call in fake_hf.tokenizer.calls)
    manifest = json.loads(result.report_path.read_text())
    assert manifest["spec_sha256"] == value.digest()
    assert manifest["execution_status"] == "completed"
    assert not (output / ".population-pending").exists()
    if name == "native_evaluation":
        report = manifest["evaluations"][0]
        assert report["classification"]["accuracy"]["n"] == 3
        assert report["classification"]["invalid_rate"]["value"] == pytest.approx(1 / 3)
        assert table[2]["gold"] is None
        assert table[3]["gold"] is None


@pytest.mark.parametrize("name", ["inference", "native_evaluation"])
@pytest.mark.parametrize("denial", ["missing", "unknown", "inner_only"])
async def test_denial_precedes_any_source_model_or_artifact(
    name: str, denial: str, fake_hf: Any, tmp_path: Path
) -> None:
    value = job(name)
    source = Reader()
    inner_digest = value.job.digest() if name != "inference" else "0" * 64
    policy = (
        None
        if denial == "missing"
        else Policy(custody="unknown")
        if denial == "unknown"
        else Policy(wrong_digest=inner_digest)
    )
    output = tmp_path / "denied"
    call = execute_population_scoring if name == "inference" else run_population_evaluation
    with pytest.raises(PopulationError):
        await call(value, reader=source, policy=policy, output_dir=output)
    assert source.calls == []
    assert fake_hf.calls == []
    assert not output.exists()


@pytest.mark.parametrize("name", ["inference", "native_evaluation"])
async def test_all_native_overflow_or_excluded_has_no_metric_denominator(
    name: str, fake_hf: Any, tmp_path: Path
) -> None:
    import pyarrow.parquet as pq

    value = job(name)
    source = Reader(gold=(rows()[2], rows()[3]))
    call = execute_population_scoring if name == "inference" else run_population_evaluation
    result = await call(value, reader=source, policy=Policy(), output_dir=tmp_path / "empty")
    assert result.counts.rows_metric_population == 0
    assert result.counts.rows_scored_success == 0
    if name == "native_evaluation":
        report = json.loads(result.report_path.read_text())["evaluations"][0]
        assert report["population_status"] == "no_metric_population"
        assert report["classification"]["accuracy"]["value"] is None
        assert report["metrics"]["brier_raw"] == {
            "status": "not_produced",
            "reason": "no_metric_population",
            "result": None,
        }
        assert len(pq.read_table(result.directory / "evaluations/threeway/items.parquet")) == 2
    else:
        table = pq.read_table(result.directory / "results.parquet")
        assert len(table) == 2
        assert all(name in table.column_names for name in ("p_E", "p_C", "p_U"))
        assert all(row["p_E"] is None for row in table.to_pylist())
    assert fake_hf.models[0].forward_calls == 0


async def test_external_invalid_hard_label_preserves_independent_vector_and_no_flag_drop(
    tmp_path: Path,
) -> None:
    import pyarrow.parquet as pq

    value = job("external_prediction_evaluation")
    gold = tuple(dict(row) for row in rows()[:3])
    predictions = (
        {
            "item_id": "short",
            "pred": "E",
            "p_E": 0.8,
            "p_C": 0.1,
            "p_U": 0.1,
            "overflow": True,
            "token_count": 999,
            "model_limit": 1,
        },
        {"item_id": "exact", "pred": "bad", "p_E": 0.1, "p_C": 0.8, "p_U": 0.1},
    )
    result = await run_population_evaluation(
        value, reader=Reader(gold, predictions), policy=Policy(), output_dir=tmp_path / "external"
    )
    report = json.loads(result.report_path.read_text())["evaluations"][0]
    assert result.counts.rows_metric_population == 3
    assert result.counts.rows_overflow_dropped == 0
    assert result.counts.rows_failed_or_invalid == 2
    assert result.counts.rows_model_sent is None
    assert report["metrics"]["brier_raw"]["result"]["n"] == 2
    table = pq.read_table(result.directory / "evaluations/threeway/items.parquet").to_pylist()
    assert table[1]["outcome"] == "invalid"
    assert table[1]["p_C"] == 0.8
    assert all(row["token_count"] is None for row in table)


class ComparisonReader(Reader):
    def __init__(self, baseline: Path, gold: tuple[dict[str, object], ...] | None = None):
        super().__init__(gold)
        self.baseline = baseline

    async def json(
        self, ref: str, path: str, *, repo_type: Literal["model", "dataset"] = "dataset"
    ) -> dict[str, object]:
        if "previous" in ref:
            return json.loads((self.baseline / path).read_text())
        return await super().json(ref, path, repo_type=repo_type)

    async def rows(
        self, ref: str, config: str, split: str, selection: object, columns: tuple[str, ...]
    ) -> tuple[dict[str, object], ...]:
        if "previous" in ref:
            import pyarrow.parquet as pq

            return tuple(
                pq.read_table(
                    self.baseline / f"evaluations/{config}/items.parquet", columns=list(columns)
                ).to_pylist()
            )
        return await super().rows(ref, config, split, selection, columns)


@pytest.mark.parametrize("different_drop", [False, True])
async def test_comparison_allows_different_models_rendering_and_limits_but_refuses_different_population(
    fake_hf: Any, tmp_path: Path, different_drop: bool
) -> None:
    baseline = job("native_evaluation")
    await run_population_evaluation(
        baseline, reader=Reader(), policy=Policy(), output_dir=tmp_path / "baseline"
    )
    raw = baseline.wire_payload()
    raw["job"]["model_id"] = "example/another-model"
    raw["job"]["model_revision"] = "d" * 40
    raw["job"]["rendering"]["template"] = "condition {premises}"
    raw["job"]["rendering"]["max_length"] = 4 if different_drop else 8
    raw["job"]["evaluation_sets"][0]["compare_to"] = [
        {
            "comparison_id": "previous",
            "baseline_ref": "example/previous@" + "c" * 40,
            "baseline_evaluation_id": "threeway",
            "sampling_unit": "independent_items",
        }
    ]
    raw["job"]["holm_families"] = [{"family_id": "declared", "members": ["threeway/previous"]}]
    value = EvaluationPopulationSpecV2.model_validate(raw)
    reader = ComparisonReader(tmp_path / "baseline")
    if different_drop:
        with pytest.raises(PopulationError, match="comparison_population_mismatch"):
            await run_population_evaluation(
                value, reader=reader, policy=Policy(), output_dir=tmp_path / "candidate"
            )
        assert not (tmp_path / "candidate/metrics.json").exists()
    else:
        result = await run_population_evaluation(
            value, reader=reader, policy=Policy(), output_dir=tmp_path / "candidate"
        )
        report = json.loads(result.report_path.read_text())
        comparison = report["evaluations"][0]["comparisons"][0]
        assert comparison["n_pairs"] == 3
        assert comparison["inference"]["status"] == "computed"
        assert report["holm_families"][0]["adjustment"]["status"] == "computed"


async def test_native_binary_projection_can_compare_with_different_source_label_realization(
    fake_hf: Any, tmp_path: Path
) -> None:
    gold = tuple(
        {**row, "gold": "E" if row["item_id"] == "short" else "not_E"} for row in rows()[:2]
    )
    baseline_raw = job("external_prediction_evaluation").wire_payload()
    case = baseline_raw["job"]["evaluation_sets"][0]
    case.update(
        label_space="2way_entail_vs_not",
        labels=["E", "not_E"],
        source_labels=["E", "not_E"],
        projection=None,
    )
    case["parse_config"]["probability_columns"] = {"E": "p_E", "not_E": "p_not_E"}
    predictions = (
        {"item_id": "short", "pred": "E", "p_E": 0.8, "p_not_E": 0.2},
        {"item_id": "exact", "pred": "not_E", "p_E": 0.2, "p_not_E": 0.8},
    )
    baseline = EvaluationPopulationSpecV2.model_validate(baseline_raw)
    await run_population_evaluation(
        baseline,
        reader=Reader(gold, predictions),
        policy=Policy(),
        output_dir=tmp_path / "baseline",
    )
    candidate_raw = job("native_evaluation").wire_payload()
    case = candidate_raw["job"]["evaluation_sets"][0]
    case.update(
        label_space="2way_entail_vs_not",
        labels=["E", "not_E"],
        projection={"E": "E", "C": "not_E", "U": "not_E"},
        compare_to=[
            {
                "comparison_id": "previous",
                "baseline_ref": "example/previous@" + "c" * 40,
                "baseline_evaluation_id": "threeway",
                "sampling_unit": "independent_items",
            }
        ],
    )
    candidate = EvaluationPopulationSpecV2.model_validate(candidate_raw)
    result = await run_population_evaluation(
        candidate,
        reader=ComparisonReader(tmp_path / "baseline", gold),
        policy=Policy(),
        output_dir=tmp_path / "candidate",
    )
    comparison = json.loads(result.report_path.read_text())["evaluations"][0]["comparisons"][0]
    assert comparison["n_pairs"] == 2
    assert comparison["inference"]["status"] == "computed"


async def test_imported_v2_manifest_cannot_authorize_a_drop(tmp_path: Path) -> None:
    raw = job("external_prediction_evaluation").wire_payload()
    case = raw["job"]["evaluation_sets"][0]
    case["prediction_manifest_path"] = "scoring.json"
    value = EvaluationPopulationSpecV2.model_validate(raw)
    reader = Reader()
    reader.metadata[(case["predictions_ref"], "scoring.json")] = {
        "schema_version": "strata.scoring.population.v2",
        "counts": {"overflow": 3},
    }
    with pytest.raises(PopulationError, match="prediction_overflow_evidence_unavailable"):
        await run_population_evaluation(
            value, reader=reader, policy=Policy(), output_dir=tmp_path / "refused"
        )
    assert not (tmp_path / "refused/metrics.json").exists()


@pytest.mark.parametrize("name", ["inference", "native_evaluation"])
@pytest.mark.parametrize("stage", ["native", "write"])
async def test_cancelled_owned_thread_cannot_publish_complete_manifest_or_end(
    name: str, stage: str, fake_hf: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import threading

    from strata_forge.pipelines import _population_report, _scoring
    from strata_forge.training.progress import JsonlProgressWriter

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    if stage == "native":
        original = _scoring.Scorer._score_outcomes

        def native(self: object, *args: Any, **kwargs: Any) -> Any:
            entered.set()
            assert release.wait(5)
            try:
                return original(self, *args, **kwargs)
            finally:
                finished.set()

        monkeypatch.setattr(_scoring.Scorer, "_score_outcomes", native)
    else:
        method = "write_scoring_attempt" if name == "inference" else "write_evaluation_attempt"
        original = getattr(_population_report, method)

        def write(*args: Any, **kwargs: Any) -> None:
            entered.set()
            assert release.wait(5)
            try:
                original(*args, **kwargs)
            finally:
                finished.set()

        monkeypatch.setattr(_population_report, method, write)
    value = job(name)
    call = execute_population_scoring if name == "inference" else run_population_evaluation
    directory = tmp_path / "abandoned"
    with JsonlProgressWriter(tmp_path / "progress.jsonl") as writer:
        task = asyncio.create_task(
            call(value, reader=Reader(), policy=Policy(), output_dir=directory, writer=writer)
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            assert await asyncio.to_thread(finished.wait, 5)
        finally:
            release.set()
            await asyncio.to_thread(finished.wait, 5)
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    assert not (directory / "scoring.json").exists()
    assert not (directory / "metrics.json").exists()
    frames = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert not any(frame["kind"] == "end" for frame in frames)


@pytest.mark.parametrize("name", ["inference", "native_evaluation"])
async def test_raw_upstream_failure_has_no_retained_context_or_source(
    name: str, tmp_path: Path
) -> None:
    value = job(name)

    async def refuse(*args: object, **kwargs: object) -> object:
        raise RuntimeError("FICTIONAL_PRIVATE_SOURCE_CANARY")

    call = execute_population_scoring if name == "inference" else run_population_evaluation
    with pytest.raises(PopulationError, match="population_execution_failed") as caught:
        await call(
            value,
            reader=Reader(),
            policy=Policy(),
            output_dir=tmp_path / "failed",
            scorer_loader=refuse,
        )
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "FICTIONAL_PRIVATE_SOURCE_CANARY" not in str(caught.value)


async def test_missing_imported_predictions_stay_all_wrong_not_an_empty_population(
    tmp_path: Path,
) -> None:
    value = job("external_prediction_evaluation")
    result = await run_population_evaluation(
        value,
        reader=Reader(tuple(dict(row) for row in rows()[:3])),
        policy=Policy(),
        output_dir=tmp_path / "missing",
    )
    report = json.loads(result.report_path.read_text())["evaluations"][0]
    assert result.counts.rows_metric_population == 3
    assert result.counts.rows_failed_or_invalid == 3
    assert result.counts.rows_scored_success == 0
    assert report["classification"]["accuracy"]["value"] == 0
    assert report["classification"]["accuracy"]["n"] == 3
    assert report["metrics"]["brier_raw"]["result"]["coverage"] == 0


async def test_native_ten_membership_metric_and_per_tier_reference(
    fake_hf: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import math

    from tests.unit.pipelines.test_scoring import Model, Tensor

    expected_vectors = {
        10: [0.8, 0.1, 0.1],
        11: [0.7, 0.2, 0.1],
        12: [0.1, 0.1, 0.8],
        13: [0.1, 0.8, 0.1],
    }
    fake_hf.tokenizer.vocab.update({"a1": 10, "a2": 11, "a3": 12, "b1": 13})

    def forward(self: Any, **features: Any) -> Any:
        assert self.evaluating
        self.forward_calls += 1
        ids = features["input_ids"].tolist()
        vectors = [
            expected_vectors[next(token for token in row if token in expected_vectors)]
            for row in ids
        ]
        return types.SimpleNamespace(
            logits=Tensor([[math.log(p) for p in vector] for vector in vectors])
        )

    monkeypatch.setattr(Model, "__call__", forward)
    fixture = FIXTURE["accounting"]
    source = []
    for row in fixture["rows"]:
        text = (
            "a " * 12
            if row["outcome"] == "overflow"
            else "x" * 200001
            if row["outcome"] == "invalid"
            else row["item_id"]
        )
        source.append(
            {
                "item_id": row["item_id"],
                "tier": row["tier"],
                "within_limits": row["within_limits"],
                "premises": [text],
                "hypothesis": "second",
                "gold": row["gold"],
            }
        )
    raw = job("native_evaluation").wire_payload()
    raw["job"]["evaluation_sets"][0]["selection"]["row_limit"] = 10
    value = EvaluationPopulationSpecV2.model_validate(raw)
    result = await run_population_evaluation(
        value, reader=Reader(tuple(source)), policy=Policy(), output_dir=tmp_path / "reference"
    )
    report = json.loads(result.report_path.read_text())["evaluations"][0]
    assert {key: report["population"]["counts"][key] for key in fixture["total"]} == fixture[
        "total"
    ]
    assert {
        entry["tier"]: {key: entry["counts"][key] for key in fixture["per_tier"][entry["tier"]]}
        for entry in report["population"]["tiers"]
    } == fixture["per_tier"]
    expected = fixture["metrics_expected"]
    for name in ("accuracy", "macro_f1", "invalid_rate"):
        assert report["classification"][name]["value"] == pytest.approx(expected[name], abs=1e-6)
    for name in ("brier_raw", "nll_raw"):
        assert report["metrics"][name]["result"]["value"] == pytest.approx(expected[name], abs=1e-6)
        assert report["metrics"][name]["result"]["n"] == 4


@pytest.mark.parametrize("name", ["inference", "native_evaluation"])
async def test_reader_cannot_silently_expand_the_declared_selection(
    name: str, fake_hf: Any, tmp_path: Path
) -> None:
    raw = job(name).wire_payload()
    selection = (
        raw["selection"] if name == "inference" else raw["job"]["evaluation_sets"][0]["selection"]
    )
    selection["row_limit"] = 1
    cls = InferencePopulationSpecV2 if name == "inference" else EvaluationPopulationSpecV2
    value = cls.model_validate(raw)
    call = execute_population_scoring if name == "inference" else run_population_evaluation
    with pytest.raises(PopulationError, match="population_selection_mismatch"):
        await call(value, reader=Reader(), policy=Policy(), output_dir=tmp_path / "expanded")
    assert fake_hf.models[0].forward_calls == 0
