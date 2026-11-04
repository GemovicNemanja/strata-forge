"""Synthetic evaluation conformance with real metrics/Parquet and fake policy/model/Hub."""

from __future__ import annotations

import asyncio
import copy
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pytest
from pydantic import ValidationError

from strata_forge.pipelines import SPEC_VERSION
from strata_forge.pipelines import eval_runner as er
from strata_forge.pipelines._evaluation_access import EvaluationError, ReadDecision, authorize_reads
from strata_forge.pipelines._evaluation_items import gold_items, parse_predictions
from strata_forge.pipelines._evaluation_reader import (
    HubEvaluationReader,
    _files,  # pyright: ignore[reportPrivateUsage] — direct worker boundary regression
    _parquet_rows,  # pyright: ignore[reportPrivateUsage] — direct worker boundary regression
    _safe_path,  # pyright: ignore[reportPrivateUsage] — direct worker boundary regression
)
from strata_forge.pipelines._scoring import ScoredRow, ScoringResult
from strata_forge.pipelines.evaluation_spec import EvalSpec, PredictionSet, Selection
from strata_forge.training._load_guard import LoadingAudit, ModelLoadingError, PaddingPolicy
from strata_forge.training.progress import JsonlProgressWriter

if TYPE_CHECKING:
    from strata_forge.pipelines._evaluation_reader import Row
    from strata_forge.pipelines._scoring import Scorer, ScoringInput
    from strata_forge.storage import HFHubClient

# This fixture intentionally holds heterogeneous valid and invalid serialized JSON shapes.
_FIXTURE: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures/evaluation_conformance.json").read_text()
)


def spec_data(mode: str = "score_predictions") -> dict[str, Any]:
    result = copy.deepcopy(_FIXTURE["valid_specs"][0 if mode == "classify" else 1])
    result["output_repo_id"] = None
    result["engine_version"] = SPEC_VERSION
    return result


class Policy:
    def __init__(
        self,
        events: list[str] | None = None,
        *,
        custody: Literal["unsealed", "sealed", "locked", "unknown"] = "unsealed",
    ) -> None:
        self.events = events if events is not None else []
        self.custody: Literal["unsealed", "sealed", "locked", "unknown"] = custody

    async def decide(
        self, ref: str, *, repo_type: Literal["model", "dataset"], spec_sha256: str
    ) -> ReadDecision:
        self.events.append("authorize:" + ref)
        return ReadDecision(
            ref=ref,
            repo_type=repo_type,
            spec_sha256=spec_sha256,
            authorized=True,
            custody=self.custody,
            policy_version="synthetic-v1",
        )


class Reader:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.gold: dict[str, tuple[Row, ...]] = {}
        self.predictions: dict[str, tuple[Row, ...]] = {}
        self.metadata: dict[tuple[str, str], dict[str, object]] = {}
        for population in _FIXTURE["populations"]:
            self.gold[population["evaluation_id"]] = tuple(
                dict(row, premises=["synthetic premise"], hypothesis="synthetic hypothesis")
                for row in copy.deepcopy(population["gold"])
            )
            self.predictions[population["evaluation_id"]] = tuple(
                copy.deepcopy(population["predictions"])
            )

    async def prepare(self, ref: str, *, model: bool = False) -> None:
        self.events.append("prepare:" + ref)

    async def rows(
        self,
        ref: str,
        config: str,
        split: str,
        selection: Selection | None,
        columns: tuple[str, ...],
    ) -> tuple[Row, ...]:
        self.events.append("rows:" + ref)
        rows = (
            self.gold[config]
            if ref.startswith("example/synthetic-gold@")
            else self.predictions[config]
        )
        if selection is not None:
            rows = rows[selection.row_start : selection.row_end][: selection.row_limit]
        return rows

    async def json(
        self, ref: str, path: str, *, repo_type: Literal["model", "dataset"] = "dataset"
    ) -> dict[str, object]:
        self.events.append("json:" + ref)
        return self.metadata[(ref, path)]


class ScorerFake:
    labels = ("E", "C", "U")
    loading_audit = LoadingAudit(status="accepted")
    padding_policy = PaddingPolicy(
        source="eos_fallback",
        pad_token_id=4,
        original_model_pad_token_id=None,
        model_config_aligned=True,
    )

    async def score(
        self, rows: tuple[ScoringInput, ...] | list[ScoringInput], *, batch_size: int = 8
    ) -> ScoringResult:
        return ScoringResult(
            rows=[
                ScoredRow(index=i, probabilities={"E": 0.8, "C": 0.1, "U": 0.1}, pred="E")
                for i, _ in enumerate(rows)
            ],
            overflow_rows=0,
            truncated_rows=0,
            failed_rows=0,
        )


@pytest.mark.parametrize("data", _FIXTURE["valid_specs"])
def test_valid_spec_and_export(data: dict[str, Any]) -> None:
    spec = EvalSpec.model_validate(data)
    assert len(spec.evaluation_sets) == 2
    assert EvalSpec.model_validate_json(json.dumps(spec.wire_payload())) == spec
    assert EvalSpec.model_validate(spec.wire_payload()).digest() == spec.digest()
    if spec.mode == "score_predictions":
        assert "model_id" not in spec.wire_payload()
        invalid = {**spec.wire_payload(), "model_id": None}
        with pytest.raises(ValidationError):
            EvalSpec.model_validate(invalid)
    assert EvalSpec.model_json_schema()["additionalProperties"] is False


@pytest.mark.parametrize(
    "case",
    [*_FIXTURE["invalid_specs"], *_FIXTURE["invalid_semantics"]],
    ids=lambda case: case["case"],
)
def test_invalid_spec_fails_cheaply(case: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EvalSpec.model_validate(case["spec"])


@pytest.mark.parametrize(
    "repo",
    [
        "org/a..b",
        "org/a--b",
        "org/model.git",
        "org/.model",
        "org/model-",
        "org/" + "x" * 97,
        "org/repo\n",
    ],
)
@pytest.mark.parametrize("location", ["gold", "predictions", "baseline"])
def test_invalid_hub_id_all_read_positions(repo: str, location: str) -> None:
    data = spec_data()
    if location == "gold":
        data["dataset_id"] = repo
    elif location == "predictions":
        data["evaluation_sets"][0]["predictions_ref"] = repo + "@" + "a" * 40
    else:
        data["evaluation_sets"][0]["compare_to"] = [
            {
                "comparison_id": "x",
                "baseline_ref": repo + "@" + "a" * 40,
                "baseline_evaluation_id": "x",
                "sampling_unit": "independent_items",
            }
        ]
    with pytest.raises(ValidationError):
        EvalSpec.model_validate(data)


@pytest.mark.parametrize("custody", ["unknown", "sealed", "locked"])
@pytest.mark.asyncio
async def test_denied_custody_precedes_every_read_and_model(
    custody: Literal["unknown", "sealed", "locked"], tmp_path: Path
) -> None:
    events: list[str] = []

    async def loader(*args: object, **kwargs: object) -> Scorer:
        events.append("model")
        raise AssertionError

    with pytest.raises(EvaluationError, match=r"^custody_denied$"):
        await er.run_evaluation(
            EvalSpec.model_validate(spec_data("classify")),
            policy=Policy(events, custody=custody),
            reader=Reader(events),
            scorer_loader=loader,
            output_dir=tmp_path,
        )
    assert all(event.startswith("authorize:") for event in events)
    assert await asyncio.to_thread(lambda: list(tmp_path.iterdir())) == []


@pytest.mark.asyncio
async def test_no_policy_and_tampered_decision_fail_closed() -> None:
    with pytest.raises(EvaluationError, match="custody_policy_unavailable"):
        await authorize_reads(
            (("example/gold@" + "a" * 40, "dataset"),), None, spec_sha256="f" * 64
        )

    class Mismatch:
        async def decide(
            self, ref: str, *, repo_type: Literal["model", "dataset"], spec_sha256: str
        ) -> ReadDecision:
            return ReadDecision(
                ref="example/different@" + "a" * 40,
                repo_type=repo_type,
                spec_sha256=spec_sha256,
                authorized=True,
                custody="unsealed",
                policy_version="x",
            )

    with pytest.raises(EvaluationError, match="custody_ref_mismatch"):
        await authorize_reads(
            (("example/gold@" + "a" * 40, "dataset"),), Mismatch(), spec_sha256="f" * 64
        )

    class Malformed:
        async def decide(
            self, ref: str, *, repo_type: Literal["model", "dataset"], spec_sha256: str
        ) -> ReadDecision:
            return ReadDecision.model_construct(
                ref=ref,
                repo_type=repo_type,
                spec_sha256=spec_sha256,
                authorized=1,
                custody="unsealed",
                policy_version="x",
            )

    with pytest.raises(EvaluationError, match="custody_resolution_failed"):
        await authorize_reads(
            (("example/gold@" + "a" * 40, "dataset"),), Malformed(), spec_sha256="f" * 64
        )


@pytest.mark.asyncio
async def test_same_name_model_and_dataset_do_not_share_a_grant() -> None:
    ref = "example/same-name@" + "a" * 40

    class DatasetOnly:
        async def decide(
            self, ref: str, *, repo_type: Literal["model", "dataset"], spec_sha256: str
        ) -> ReadDecision:
            return ReadDecision(
                ref=ref,
                repo_type="dataset",
                spec_sha256=spec_sha256,
                authorized=True,
                custody="unsealed",
                policy_version="synthetic",
            )

    with pytest.raises(EvaluationError, match="custody_ref_mismatch"):
        await authorize_reads(
            ((ref, "dataset"), (ref, "model")), DatasetOnly(), spec_sha256="f" * 64
        )
    data = spec_data("classify")
    data["model_id"] = data["dataset_id"]
    data["model_revision"] = data["dataset_commit_sha"]
    spec = EvalSpec.model_validate(data)
    assert len(spec.input_reads()) == 2
    assert spec.input_reads()[0][0] == spec.input_reads()[1][0]
    assert spec.input_reads()[0][1] != spec.input_reads()[1][1]


@pytest.mark.asyncio
async def test_changed_spec_digest_cannot_reuse_a_prior_approval() -> None:
    class PriorSpec:
        async def decide(
            self, ref: str, *, repo_type: Literal["model", "dataset"], spec_sha256: str
        ) -> ReadDecision:
            return ReadDecision(
                ref=ref,
                repo_type=repo_type,
                spec_sha256="a" * 64,
                authorized=True,
                custody="unsealed",
                policy_version="synthetic",
            )

    with pytest.raises(EvaluationError, match="custody_spec_mismatch"):
        await authorize_reads(
            (("example/gold@" + "a" * 40, "dataset"),), PriorSpec(), spec_sha256="b" * 64
        )


def test_digest_binds_config_split_selection_and_rejects_boolean_version() -> None:
    data = spec_data()
    first = EvalSpec.model_validate(data).digest()
    data["evaluation_sets"][0]["split"] = "other"
    assert EvalSpec.model_validate(data).digest() != first
    data["schema_version"] = True
    with pytest.raises(ValidationError):
        EvalSpec.model_validate(data)


@pytest.mark.asyncio
async def test_existing_artifact_directory_is_not_rewritten(tmp_path: Path) -> None:
    original = tmp_path / "metrics.json"
    original.write_text("immutable earlier attempt")
    reader = Reader()
    with pytest.raises(EvaluationError, match="artifact_directory_not_empty"):
        await er.run_evaluation(
            EvalSpec.model_validate(spec_data()),
            policy=Policy(),
            reader=reader,
            output_dir=tmp_path,
        )
    assert original.read_text() == "immutable earlier attempt"
    assert reader.events == []


@pytest.mark.asyncio
async def test_prediction_modes_preserve_population_probability_family_and_arrow_schema(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    events: list[str] = []
    path = tmp_path / "progress.jsonl"
    with JsonlProgressWriter(path) as writer:
        result = await er.run_evaluation(
            EvalSpec.model_validate(spec_data()),
            policy=Policy(events),
            reader=Reader(events),
            writer=writer,
            output_dir=tmp_path / "results",
        )
    report = json.loads(result.metrics_path.read_text())
    assert "accuracy" not in report
    assert report["counts"] == {
        "evaluation_sets": 2,
        "eligible": 8,
        "valid_predictions": 6,
        "invalid_predictions": 2,
        "missing_predictions": 2,
        "overflow": 0,
        "truncated": 0,
    }
    for entry, population in zip(report["evaluations"], _FIXTURE["populations"], strict=True):
        expected = population["expected"]
        assert entry["classification"]["accuracy"]["value"] == expected["accuracy"]
        assert math.isclose(
            entry["classification"]["macro_f1"]["value"], expected["macro_f1"], abs_tol=1e-12
        )
        assert math.isclose(
            entry["metrics"]["brier_raw"]["result"]["value"], expected["brier"], abs_tol=1e-12
        )
        assert math.isclose(
            entry["metrics"]["nll_raw"]["result"]["value"],
            -math.log(expected["nll_gold_product"]) / 3,
            abs_tol=1e-12,
        )
        assert entry["metrics"]["brier_raw"]["result"]["coverage"] == 0.75
        assert entry["metrics"]["brier_calibrated"]["status"] == "not_produced"
    assert report["evaluations"][1]["probability_context"]["kind"] == "restricted_softmax"
    slices = report["evaluations"][0]["slices"]
    assert {s["value"]["value"]: s["n"] for s in slices if s["column"] == "tags"} == {
        "base": 2,
        "alpha": 1,
        "stress": 2,
    }
    table = cast("pa.Table", pq.read_table(tmp_path / "results/evaluations/threeway/items.parquet"))  # pyright: ignore[reportUnknownMemberType] — Arrow read_table lacks complete stubs
    assert table.schema.field("p_E").type == pa.float64()
    assert table.schema.field("p_cal_E").type == pa.float64()
    assert table.column("p_cal_E").null_count == 4
    assert table.column("pred").to_pylist() == ["E", "E", None, "U"]
    assert not {"premises", "hypothesis", "response", "reasoning"} & set(table.column_names)
    assert "metrics.json" not in (tmp_path / "results/README.md").read_text().split("---")[1]
    assert all(s.startswith("authorize:") for s in events[:2])
    progress = [json.loads(line) for line in path.read_text().splitlines()]
    assert progress[-1]["kind"] == "end"
    assert all(len(event.get("metrics", {})) <= 64 for event in progress)
    assert result.output_ref is None


@pytest.mark.parametrize("case", _FIXTURE["join_refusals"], ids=lambda case: case["case"])
def test_prediction_join_refusals(case: dict[str, Any]) -> None:
    spec = EvalSpec.model_validate(spec_data())
    population = spec.evaluation_sets[0]
    assert isinstance(population, PredictionSet)
    gold = gold_items(Reader().gold["threeway"], population)
    with pytest.raises(EvaluationError, match=case["code"]):
        parse_predictions(gold, case["rows"], population)


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 1.1, True, None, 10**1000])
def test_malformed_probability_is_not_renormalized(value: object) -> None:
    case = cast("PredictionSet", EvalSpec.model_validate(spec_data()).evaluation_sets[0])
    gold = gold_items(Reader().gold["threeway"], case)
    rows = [{"item_id": "a", "pred": "E", "p_E": value, "p_C": 0.1, "p_U": 0.1}]
    with pytest.raises(EvaluationError, match=r"^probability_invalid$"):
        parse_predictions(gold, rows, case)


@pytest.mark.asyncio
async def test_missing_predictions_are_complete_wrong_result(tmp_path: Path) -> None:
    reader = Reader()
    reader.predictions = {"threeway": (), "binary": ()}
    result = await er.run_evaluation(
        EvalSpec.model_validate(spec_data()), policy=Policy(), reader=reader, output_dir=tmp_path
    )
    report = json.loads(result.metrics_path.read_text())
    for entry in report["evaluations"]:
        assert entry["classification"]["accuracy"]["value"] == 0
        assert entry["classification"]["invalid_rate"]["value"] == 1
        assert entry["metrics"]["brier_raw"]["result"]["value"] is None
        assert entry["metrics"]["brier_raw"]["result"]["excluded"] == 4
        assert entry["metrics"]["brier_raw"]["result"]["coverage"] == 0


@pytest.mark.asyncio
async def test_classifier_load_guard_precedes_rows_and_saves_safe_audit(tmp_path: Path) -> None:
    events: list[str] = []

    async def loader(*args: object, **kwargs: object) -> Scorer:
        events.append("model")
        raise ModelLoadingError(
            LoadingAudit(
                status="refused",
                missing_keys=("encoder.weight",),
                missing_count=1,
                codes=("missing_weights",),
            )
        )

    with pytest.raises(EvaluationError, match=r"^model_loading_refused$"):
        await er.run_evaluation(
            EvalSpec.model_validate(spec_data("classify")),
            policy=Policy(events),
            reader=Reader(events),
            scorer_loader=loader,
            output_dir=tmp_path,
        )
    assert not any(event.startswith("rows:") for event in events)
    evidence = json.loads((tmp_path / "loading-refusal.json").read_text())
    assert evidence["loading_info"]["missing_keys"] == ["encoder.weight"]
    assert not (tmp_path / "metrics.json").exists()


@pytest.mark.asyncio
async def test_classifier_render_refusal_first_row_preserves_vectors_and_load_audit(
    tmp_path: Path,
) -> None:
    reader = Reader()
    reader.gold["threeway"][0]["premises"] = ["synthetic"] * 21

    async def loader(*args: object, **kwargs: object) -> Scorer:
        return cast("Scorer", ScorerFake())

    result = await er.run_evaluation(
        EvalSpec.model_validate(spec_data("classify")),
        policy=Policy(),
        reader=reader,
        scorer_loader=loader,
        output_dir=tmp_path,
    )
    report = json.loads(result.metrics_path.read_text())
    first = report["evaluations"][0]
    assert first["counts"]["invalid_predictions"] == 1
    assert first["source"]["loading_info"]["status"] == "accepted"
    assert first["source"]["padding_policy"]["source"] == "eos_fallback"
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = cast("pa.Table", pq.read_table(tmp_path / "evaluations/threeway/items.parquet"))  # pyright: ignore[reportUnknownMemberType] — Arrow read_table lacks complete stubs
    rows = cast("list[Row]", table.to_pylist())
    assert rows[0]["p_E"] is None
    assert rows[1]["p_E"] == 0.8
    assert rows[0]["error_code"] == "premise_limit_exceeded"


@pytest.mark.asyncio
async def test_cancel_forward_does_not_publish_or_end(tmp_path: Path) -> None:
    entered = asyncio.Event()

    class Blocked(ScorerFake):
        async def score(
            self, rows: tuple[ScoringInput, ...] | list[ScoringInput], *, batch_size: int = 8
        ) -> ScoringResult:
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError

    async def loader(*args: object, **kwargs: object) -> Scorer:
        return cast("Scorer", Blocked())

    task = asyncio.create_task(
        er.run_evaluation(
            EvalSpec.model_validate(spec_data("classify")),
            policy=Policy(),
            reader=Reader(),
            scorer_loader=loader,
            output_dir=tmp_path,
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not (tmp_path / "metrics.json").exists()


@pytest.mark.parametrize(
    "path",
    ["../x.parquet", "/x.parquet", "https://other/x.parquet", "hf://other/x", "a\\b.parquet", ""],
)
def test_reader_external_and_traversal_dependencies_refuse(path: str) -> None:
    with pytest.raises(EvaluationError, match="dataset_dependency_unsupported"):
        _safe_path(path)


def test_native_card_inventory_and_real_parquet_selection(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    card = "---\nconfigs:\n- config_name: fixture\n  data_files:\n  - split: test\n    path: data/*.parquet\n---\n"
    assert _files(card, ("data/a.parquet", "README.md"), "fixture", "test") == ("data/a.parquet",)
    path = tmp_path / "tiny.parquet"
    pq.write_table(  # pyright: ignore[reportUnknownMemberType] — Arrow writer lacks complete stubs
        pa.Table.from_pylist([{"id": str(i), "unused": "private"} for i in range(8)]), path
    )
    selected = _parquet_rows((path,), Selection(row_start=2, row_end=7, row_limit=3), ("id",))
    assert selected == ({"id": "2"}, {"id": "3"}, {"id": "4"})
    with pytest.raises(EvaluationError, match="dataset_column_missing"):
        _parquet_rows((path,), None, ("missing",))


def scoring_manifest(data: dict[str, Any], case_index: int = 0) -> dict[str, object]:
    case = data["evaluation_sets"][case_index]
    return {
        "id_semantics": {
            "kind": "stable_string",
            "source_column": "item_id",
            "artifact_column": "item_id",
            "unique": True,
        },
        "input_refs": {"dataset": data["dataset_id"] + "@" + data["dataset_commit_sha"]},
        "dataset_config": case["dataset_config"],
        "split": case["split"],
        "labels": case["source_labels"],
        "row_limit": 4,
        "engine": "hf_classify",
    }


@pytest.mark.parametrize("policy", ["retain_reported", "count_as_invalid"])
@pytest.mark.asyncio
async def test_scoring_manifest_identity_and_truncation_policy(policy: str, tmp_path: Path) -> None:
    data = spec_data()
    data["evaluation_sets"] = data["evaluation_sets"][:1]
    case = data["evaluation_sets"][0]
    case["prediction_manifest_path"] = "scoring.json"
    case["truncation_policy"] = policy
    reader = Reader()
    reader.metadata[(case["predictions_ref"], "scoring.json")] = scoring_manifest(data)
    reader.predictions["threeway"] = tuple(
        dict(row, error=None, truncated=True, overflow=True)
        for row in reader.predictions["threeway"]
    )
    result = await er.run_evaluation(
        EvalSpec.model_validate(data), policy=Policy(), reader=reader, output_dir=tmp_path
    )
    report = json.loads(result.metrics_path.read_text())["evaluations"][0]
    assert report["counts"]["truncated"] == 3
    assert report["counts"]["overflow"] == 3
    assert report["source"]["manifest_status"] == "strata_scoring_verified"
    assert report["classification"]["accuracy"]["value"] == (
        0.25 if policy == "retain_reported" else 0
    )
    assert report["metrics"]["brier_raw"]["result"]["coverage"] == (
        0.75 if policy == "retain_reported" else 0
    )


@pytest.mark.parametrize("change", ["ordinal", "pin", "config", "labels", "limit", "family"])
@pytest.mark.asyncio
async def test_mismatched_producer_manifest_refuses_without_reading_prediction_rows(
    change: str, tmp_path: Path
) -> None:
    data = spec_data()
    data["evaluation_sets"] = data["evaluation_sets"][:1]
    case = data["evaluation_sets"][0]
    case["prediction_manifest_path"] = "scoring.json"
    manifest = scoring_manifest(data)
    if change == "ordinal":
        manifest["id_semantics"] = {"kind": "ordinal", "artifact_column": "custom_id"}
    if change == "pin":
        manifest["input_refs"] = {"dataset": "example/other@" + "a" * 40}
    if change == "config":
        manifest["dataset_config"] = "other"
    if change == "labels":
        manifest["labels"] = ["U", "C", "E"]
    if change == "limit":
        manifest["row_limit"] = None
    if change == "family":
        manifest["engine"] = "hf_label_logits"
    reader = Reader()
    reader.metadata[(case["predictions_ref"], "scoring.json")] = manifest
    with pytest.raises(EvaluationError, match="prediction_manifest_mismatch"):
        await er.run_evaluation(
            EvalSpec.model_validate(data), policy=Policy(), reader=reader, output_dir=tmp_path
        )
    assert "rows:" + case["predictions_ref"] not in reader.events


@pytest.mark.asyncio
async def test_json_field_extracts_hard_labels_and_keeps_invalid_wrong_without_probabilities(
    tmp_path: Path,
) -> None:
    data = copy.deepcopy(_FIXTURE["valid_specs"][2])
    data["output_repo_id"] = None
    data["engine_version"] = SPEC_VERSION
    data["evaluation_sets"] = data["evaluation_sets"][:1]
    reader = Reader()
    reader.predictions["threeway"] = (
        {"item_id": "a", "response": '{"label":"E"}'},
        {"item_id": "b", "response": '{"label":"C"}'},
        {"item_id": "c", "response": '{"label":"U", "bad":NaN}'},
        {"item_id": "d", "response": "[broken"},
    )
    result = await er.run_evaluation(
        EvalSpec.model_validate(data), policy=Policy(), reader=reader, output_dir=tmp_path
    )
    report = json.loads(result.metrics_path.read_text())["evaluations"][0]
    assert report["classification"]["accuracy"]["value"] == 0.5
    assert report["classification"]["invalid_rate"]["value"] == 0.5
    assert report["metrics"]["brier_raw"]["status"] == "not_produced"
    assert report["probability_context"]["kind"] is None


def add_baseline(data: dict[str, Any], reader: Reader) -> None:
    case = data["evaluation_sets"][0]
    ref = "example/baseline@" + "d" * 40
    case["compare_to"] = [
        {
            "comparison_id": "prior",
            "baseline_ref": ref,
            "baseline_evaluation_id": "baseline3",
            "sampling_unit": "independent_items",
        }
    ]
    data["holm_families"] = [{"family_id": "explicit-family", "members": ["threeway/prior"]}]
    reader.metadata[(ref, "metrics.json")] = {
        "schema_version": "strata.eval.metrics.v1",
        "input_refs": [
            {"role": "gold", "ref": data["dataset_id"] + "@" + data["dataset_commit_sha"]}
        ],
        "evaluations": [
            {
                "evaluation_id": "baseline3",
                "dataset_config": "threeway",
                "split": "test",
                "selection": case["selection"],
                "labels": case["labels"],
                "projection": None,
            }
        ],
    }
    reader.predictions["baseline3"] = tuple(
        {
            "item_id": row["item_id"],
            "gold": row["gold"],
            "pred": row["gold"],
            "prediction_status": "valid",
        }
        for row in reader.gold["threeway"]
    )


@pytest.mark.asyncio
async def test_exact_paired_population_delta_and_explicit_holm_family(tmp_path: Path) -> None:
    data = spec_data()
    reader = Reader()
    add_baseline(data, reader)
    result = await er.run_evaluation(
        EvalSpec.model_validate(data), policy=Policy(), reader=reader, output_dir=tmp_path
    )
    report = json.loads(result.metrics_path.read_text())
    statistic = report["evaluations"][0]["comparisons"][0]["mcnemar"]["result"]
    assert statistic["n"] == 4
    assert statistic["p_value"] == 0.25
    assert statistic["accuracy_delta"] == -0.75
    assert report["holm_families"][0]["adjustment"]["status"] == "computed"


@pytest.mark.asyncio
async def test_repeated_base_problem_groups_do_not_become_independent_pairs(tmp_path: Path) -> None:
    data = spec_data()
    reader = Reader()
    add_baseline(data, reader)
    for row in reader.gold["threeway"]:
        row["base_problem_id"] = "same-group"
    result = await er.run_evaluation(
        EvalSpec.model_validate(data), policy=Policy(), reader=reader, output_dir=tmp_path
    )
    report = json.loads(result.metrics_path.read_text())
    assert report["evaluations"][0]["comparisons"][0]["mcnemar"] == {
        "status": "not_produced",
        "reason": "requires_cluster_method",
        "result": None,
    }
    assert report["holm_families"][0]["adjustment"]["status"] == "not_produced"


@pytest.mark.asyncio
async def test_paired_missing_id_refuses_instead_of_inner_joining(tmp_path: Path) -> None:
    data = spec_data()
    reader = Reader()
    add_baseline(data, reader)
    reader.predictions["baseline3"] = reader.predictions["baseline3"][:-1]
    with pytest.raises(EvaluationError, match="comparison_population_mismatch"):
        await er.run_evaluation(
            EvalSpec.model_validate(data), policy=Policy(), reader=reader, output_dir=tmp_path
        )


@pytest.mark.asyncio
async def test_unfenced_publication_is_unavailable_before_any_reads(tmp_path: Path) -> None:
    data = spec_data()
    data["output_repo_id"] = "example/synthetic-output"
    events: list[str] = []
    with pytest.raises(EvaluationError, match="publication_policy_unavailable"):
        await er.run_evaluation(
            EvalSpec.model_validate(data),
            policy=Policy(events),
            reader=Reader(events),
            output_dir=tmp_path,
        )
    assert events == []


@pytest.mark.asyncio
async def test_real_native_pinned_reader_uses_only_declared_parquet_and_exact_revision(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    assets = tmp_path / "assets"
    assets.mkdir()
    files: dict[tuple[str, str], Path] = {}
    fixture = Reader()
    for repo, rows_by_config, split in (
        ("example/synthetic-gold", fixture.gold, "test"),
        ("example/synthetic-predictions", fixture.predictions, "train"),
    ):
        sections: list[str] = []
        for config, rows in rows_by_config.items():
            path = assets / f"{repo.rsplit('/', 1)[1]}-{config}.parquet"
            pq.write_table(pa.Table.from_pylist(list(rows)), path)  # pyright: ignore[reportUnknownMemberType] — untyped Arrow I/O boundary
            files[(repo, config + ".parquet")] = path
            sections.append(
                f"- config_name: {config}\n  data_files:\n  - split: {split}\n    path: {config}.parquet\n"
            )
        card = assets / (repo.rsplit("/", 1)[1] + ".md")
        card.write_text("---\nconfigs:\n" + "".join(sections) + "---\n")
        files[(repo, "README.md")] = card
    calls: list[tuple[str, str, str | None]] = []

    class PinnedHub:
        async def resolve_revision(self, repo: str, *, revision: str, repo_type: str) -> str:
            calls.append(("resolve", repo, revision))
            return revision

        async def list_repo_files(
            self, repo: str, *, revision: str, repo_type: str
        ) -> tuple[str, ...]:
            calls.append(("list", repo, revision))
            return tuple(name for owner, name in files if owner == repo)

        async def download_file(
            self, repo: str, filename: str, *, revision: str, repo_type: str
        ) -> Path:
            calls.append(("file", repo, revision))
            return files[(repo, filename)]

    reader = HubEvaluationReader(False)
    reader.hub = cast("HFHubClient", PinnedHub())
    result = await er.run_evaluation(
        EvalSpec.model_validate(spec_data()),
        policy=Policy(),
        reader=reader,
        output_dir=tmp_path / "result",
    )
    report = json.loads(result.metrics_path.read_text())
    assert report["evaluations"][0]["classification"]["accuracy"]["value"] == 0.25
    assert report["evaluations"][1]["classification"]["accuracy"]["value"] == 0.5
    assert all(
        revision == ("a" * 40 if repo.endswith("gold") else "c" * 40) for _, repo, revision in calls
    )
    assert len([call for call in calls if call[0] == "resolve"]) == 2


@pytest.mark.parametrize("kind", ["wrong_pin", "adapter_only", "no_native_weights"])
@pytest.mark.asyncio
async def test_reader_prepares_model_pin_without_implicit_base_or_non_native_fallback(
    kind: str,
) -> None:
    class MetadataHub:
        async def resolve_revision(self, repo: str, *, revision: str, repo_type: str) -> str:
            return "b" * 40 if kind == "wrong_pin" else revision

        async def list_repo_files(
            self, repo: str, *, revision: str, repo_type: str
        ) -> tuple[str, ...]:
            return (
                ("adapter_config.json", "model.safetensors")
                if kind == "adapter_only"
                else ("config.json", "pytorch_model.bin")
            )

    reader = HubEvaluationReader(False)
    reader.hub = cast("HFHubClient", MetadataHub())
    with pytest.raises(EvaluationError, match=r"input_ref_mismatch|model_checkpoint_unsupported"):
        await reader.prepare("example/model@" + "a" * 40, model=True)


@pytest.mark.asyncio
async def test_pinned_model_rendering_metadata_overrides_record_both_digests(
    tmp_path: Path,
) -> None:
    data = spec_data("classify")
    data["rendering"] = {
        "source": "model_artifact",
        "file": "strata_verifier.json",
        "overrides": {"max_length": 64},
    }
    reader = Reader()
    reader.metadata[(data["model_id"] + "@" + data["model_revision"], "strata_verifier.json")] = {
        "schema_version": 1,
        "labels": ["E", "C", "U"],
        "template": "{premises}",
        "pair_template": "{hypothesis}",
        "premise_format": "numbered_lines",
        "max_length": 128,
        "max_premises": 20,
    }

    async def loader(*args: object, **kwargs: object) -> Scorer:
        assert kwargs["max_length"] == 64
        return cast("Scorer", ScorerFake())

    result = await er.run_evaluation(
        EvalSpec.model_validate(data),
        policy=Policy(),
        reader=reader,
        scorer_loader=loader,
        output_dir=tmp_path,
    )
    audit = json.loads(result.metrics_path.read_text())["evaluations"][0]["source"]["rendering"]
    assert audit["max_length"] == 64
    assert audit["original_sha256"] != audit["effective_sha256"]


@pytest.mark.asyncio
async def test_own_versioned_artifact_can_be_rescored_with_verified_population(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = spec_data()
    data["evaluation_sets"] = data["evaluation_sets"][:1]
    first = await er.run_evaluation(
        EvalSpec.model_validate(data),
        policy=Policy(),
        reader=Reader(),
        output_dir=tmp_path / "first",
    )
    manifest = json.loads(first.metrics_path.read_text())
    table = cast("pa.Table", pq.read_table(tmp_path / "first/evaluations/threeway/items.parquet"))  # pyright: ignore[reportUnknownMemberType] — untyped Arrow read boundary
    rows = tuple(cast("list[Row]", table.to_pylist()))
    reader = Reader()
    reader.predictions["threeway"] = rows
    ref = "example/first-eval@" + "d" * 40
    reader.metadata[(ref, "metrics.json")] = manifest
    case = data["evaluation_sets"][0]
    case.update(predictions_ref=ref, prediction_manifest_path="metrics.json")
    second = await er.run_evaluation(
        EvalSpec.model_validate(data),
        policy=Policy(),
        reader=reader,
        output_dir=tmp_path / "second",
    )
    report = json.loads(second.metrics_path.read_text())["evaluations"][0]
    assert report["source"]["manifest_status"] == "strata_evaluation_verified"
    assert report["classification"]["accuracy"]["value"] == 0.25
    assert report["counts"]["missing_predictions"] == 1


@pytest.mark.asyncio
async def test_decoder_token_map_uses_explicit_label_order_not_json_object_order(
    tmp_path: Path,
) -> None:
    data = spec_data("classify")
    data["engine"] = "hf_label_logits"
    data["label_tokens"] = {"U": "C", "C": "B", "E": "A"}
    data["rendering"].update(template="{premises}\n{hypothesis}", pair_template=None)

    async def loader(*args: object, **kwargs: object) -> Scorer:
        tokens = cast("dict[str, str]", kwargs["label_tokens"])
        assert list(tokens) == ["E", "C", "U"]
        assert kwargs["label_alias"] is None
        return cast("Scorer", ScorerFake())

    result = await er.run_evaluation(
        EvalSpec.model_validate(data),
        policy=Policy(),
        reader=Reader(),
        scorer_loader=loader,
        output_dir=tmp_path,
    )
    report = json.loads(result.metrics_path.read_text())
    assert all(
        entry["probability_context"]["kind"] == "restricted_softmax"
        for entry in report["evaluations"]
    )
