"""Scoring dispatch and real Parquet persistence with synthetic rows and fake HF models."""

from __future__ import annotations

import json
import types
from typing import TYPE_CHECKING, Any, cast

import pytest

from strata_forge.pipelines import SPEC_VERSION
from strata_forge.pipelines import inference_runner as ir
from strata_forge.pipelines._scoring import ScoredRow, ScoringInput, ScoringResult
from strata_forge.storage import CommitRef
from strata_forge.training._load_guard import LoadingAudit, ModelLoadingError, PaddingPolicy
from strata_forge.training.progress import ProgressEvent

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_MODEL_SHA = "a" * 40
_DATASET_SHA = "b" * 40
_OUTPUT_SHA = "c" * 40
_TOKEN = "hf_synthetic_scoring_fixture_credential"


def _spec(*, decoder: bool = False, **overrides: Any) -> dict[str, Any]:
    hyperparams: dict[str, Any] = {
        "engine": "hf_label_logits" if decoder else "hf_classify",
        "max_model_len": 128,
        "scoring_batch_size": 2,
    }
    hyperparams.update(
        {"label_tokens": {"yes": "A", "no": "B"}}
        if decoder
        else {
            "pair_template": "Hypothesis: {h}",
            "label_alias": {"LABEL_0": "yes", "LABEL_1": "no"},
        }
    )
    out = {
        "engine_version": SPEC_VERSION,
        "model_id": "org/model",
        "model_revision": "release",
        "dataset_id": "org/data",
        "dataset_commit_sha": "data-release",
        "dataset_config": "pairs",
        "split": "test",
        "template": "Premise: {p}",
        "column_mapping": {"p": "premise", "h": "hypothesis"},
        "hyperparams": hyperparams,
        "run_id": "score-fixture",
    }
    out.update(overrides)
    return out


@pytest.fixture(autouse=True)
def _no_inherited_token(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction] — pytest autouse fixture
    monkeypatch.delenv("FORGE_SECRETS_FILE", raising=False)
    monkeypatch.delenv("HF_WRITE_TOKEN", raising=False)


@pytest.fixture
def run_fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> types.SimpleNamespace:
    state = types.SimpleNamespace(
        hub_tokens=[],
        revisions=[],
        loads=[],
        loaded_specs=[],
        commits=[],
        inputs=[],
        rows=[{"premise": "one", "hypothesis": "two"}, {"premise": "three", "hypothesis": "four"}],
        push_error=False,
        all_failed=False,
        loading_refusal=False,
        results_dir=tmp_path / "persisted",
    )

    class Hub:
        def __init__(self, *, token: str | bool) -> None:
            state.hub_tokens.append(token)

        async def resolve_revision(self, repo_id: str, **kwargs: Any) -> str:
            state.revisions.append((repo_id, kwargs))
            return _MODEL_SHA if repo_id == "org/model" else _DATASET_SHA

        async def create_repo(self, *args: Any, **kwargs: Any) -> str:
            assert kwargs["private"] is True
            return "org/out"

        async def commit_folder(self, directory: Path, repo_id: str, **kwargs: Any) -> CommitRef:
            state.commits.append((directory, repo_id, kwargs))
            if state.push_error:
                raise RuntimeError(f"denied {_TOKEN}")
            return CommitRef(
                sha=_OUTPUT_SHA,
                url=f"https://huggingface.co/datasets/{repo_id}/commit/{_OUTPUT_SHA}",
            )

    class Scorer:
        labels = ("yes", "no")
        loading_audit = LoadingAudit(status="accepted")
        padding_policy = PaddingPolicy(
            source="tokenizer_pad", pad_token_id=0, original_model_pad_token_id=None
        )

        async def score(
            self,
            inputs: list[ScoringInput],
            *,
            batch_size: int,
            on_progress: Callable[[ProgressEvent], None],
        ) -> ScoringResult:
            assert batch_size == 2
            state.inputs.extend(inputs)
            scored = [ScoredRow(index=0, overflow=True, error="hypothesis_too_long")]
            scored.append(
                ScoredRow(index=1, error="hypothesis_too_long", overflow=True)
                if state.all_failed
                else ScoredRow(
                    index=1, probabilities={"yes": 0.75, "no": 0.25}, pred="yes", mass_outside=0.1
                )
            )
            on_progress(
                ProgressEvent(kind="step", stage="run", step=len(inputs), total_steps=len(inputs))
            )
            return ScoringResult(
                rows=scored,
                overflow_rows=2 if state.all_failed else 1,
                truncated_rows=0,
                failed_rows=2 if state.all_failed else 1,
            )

    async def load(model_id: str, **kwargs: Any) -> Scorer:
        state.loads.append((model_id, kwargs))
        if state.loading_refusal:
            raise ModelLoadingError(
                LoadingAudit(
                    status="refused",
                    missing_keys=("encoder.embeddings.weight",),
                    missing_count=1,
                    codes=("missing_weights",),
                )
            )
        return Scorer()

    def rows(spec: ir.RunSpec, token: str | bool) -> list[dict[str, str]]:
        state.loaded_specs.append((spec, token))
        return state.rows

    def no_vllm(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("HF scoring must not start or download a vLLM server")

    def result_directory(run_id: str | None) -> Path:
        return state.results_dir

    monkeypatch.setattr(ir, "HFHubClient", Hub)
    monkeypatch.setattr(ir, "load_scorer", load)
    monkeypatch.setattr(ir, "_load_rows", rows)
    monkeypatch.setattr(ir, "_local_results_dir", result_directory)
    monkeypatch.setattr(ir, "_fetch_model", no_vllm)
    monkeypatch.setattr(ir, "serving_endpoint", no_vllm)
    return state


async def test_classifier_dispatch_pins_inputs_and_retains_probability_columns_after_refusal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_fakes: types.SimpleNamespace,
) -> None:
    # Real Arrow serialization must preserve float columns after a refused first row.
    parquet = pytest.importorskip("pyarrow.parquet")
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(_spec(progress_path=str(progress))))
    assert await ir.main() == 0
    loaded_spec, credential = run_fakes.loaded_specs[0]
    assert loaded_spec.model_revision == _MODEL_SHA
    assert loaded_spec.dataset_commit_sha == _DATASET_SHA
    assert credential is False
    assert run_fakes.loads[0][1]["revision"] == _MODEL_SHA
    assert run_fakes.loads[0][1]["token"] is False
    assert run_fakes.loads[0][1]["device"] == "auto"
    assert run_fakes.inputs[0].text_pair == "Hypothesis: two"
    path = run_fakes.results_dir / "results.parquet"
    persisted = parquet.read_table(path).to_pylist()
    assert persisted[0]["p_yes"] is None
    assert persisted[1]["p_yes"] == 0.75
    assert persisted[1]["output"] == persisted[1]["pred"] == "yes"
    assert [row["custom_id"] for row in persisted] == ["row-0", "row-1"]
    summary = json.loads((path.parent / "scoring.json").read_text())
    assert summary["input_refs"] == {
        "model": f"org/model@{_MODEL_SHA}",
        "dataset": f"org/data@{_DATASET_SHA}",
    }
    assert "Premise:" not in json.dumps(summary)
    assert "Hypothesis:" not in json.dumps(summary)
    assert summary["loading_info"]["status"] == "accepted"
    assert summary["padding_policy"]["pad_token_id"] == 0
    assert "results.parquet" in (path.parent / "README.md").read_text()
    end = json.loads(progress.read_text().splitlines()[-1])
    assert end["input_refs"] == summary["input_refs"]
    assert end["output_ref"] is None
    assert end["metrics"]["failed"] == 1
    assert end["metrics"]["overflow_rows"] == 1


async def test_incomplete_model_is_refused_before_dataset_materialization(
    monkeypatch: pytest.MonkeyPatch, run_fakes: types.SimpleNamespace
) -> None:
    run_fakes.loading_refusal = True
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(_spec()))
    assert await ir.main() != 0
    assert run_fakes.loaded_specs == []
    assert run_fakes.inputs == []
    summary = json.loads((run_fakes.results_dir / "scoring.json").read_text())
    assert summary["status"] == "loading_refused"
    assert summary["loading_info"]["missing_keys"] == ["encoder.embeddings.weight"]
    assert summary["input_refs"]["model"] == f"org/model@{_MODEL_SHA}"
    assert not (run_fakes.results_dir / "results.parquet").exists()


async def test_decoder_upload_uses_actual_commit_and_no_credential_in_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_fakes: types.SimpleNamespace,
) -> None:
    secret = tmp_path / ".secrets.json"
    secret.write_text(json.dumps({"HF_TOKEN": _TOKEN}))
    secret.chmod(0o600)
    monkeypatch.setenv("FORGE_SECRETS_FILE", str(secret))
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv(
        "STRATA_RUN_CONFIG",
        json.dumps(_spec(decoder=True, output_repo_id="org/out", progress_path=str(progress))),
    )
    assert await ir.main() == 0
    assert run_fakes.hub_tokens == [_TOKEN]
    assert run_fakes.loads[0][1]["token"] == _TOKEN
    assert run_fakes.inputs[0].text_pair is None
    end = json.loads(progress.read_text().splitlines()[-1])
    assert end["output_ref"] == f"org/out@{_OUTPUT_SHA}"
    assert end["message"] == "org/out"
    assert _TOKEN not in progress.read_text()
    assert _TOKEN not in (run_fakes.results_dir / "scoring.json").read_text()
    assert set(run_fakes.commits[0][2]["allow_patterns"]) == {
        "results.parquet",
        "scoring.json",
        "README.md",
    }
    assert not secret.exists()


async def test_all_failed_scores_keep_artifacts_and_exit_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_fakes: types.SimpleNamespace,
) -> None:
    run_fakes.all_failed = True
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(_spec(progress_path=str(progress))))
    assert await ir.main() == 1
    assert (run_fakes.results_dir / "results.parquet").exists()
    assert (run_fakes.results_dir / "scoring.json").exists()
    assert "all 2 rows failed" in progress.read_text()


async def test_push_failure_preserves_local_scores_and_scrubs_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_fakes: types.SimpleNamespace,
) -> None:
    run_fakes.push_error = True
    secret = tmp_path / ".secrets.json"
    secret.write_text(json.dumps({"HF_TOKEN": _TOKEN}))
    secret.chmod(0o600)
    monkeypatch.setenv("FORGE_SECRETS_FILE", str(secret))
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv(
        "STRATA_RUN_CONFIG",
        json.dumps(_spec(output_repo_id="org/out", progress_path=str(progress))),
    )
    assert await ir.main() == 1
    assert (run_fakes.results_dir / "results.parquet").exists()
    assert "results are on the VM" in progress.read_text()
    assert _TOKEN not in progress.read_text()


async def test_oversized_hypothesis_refuses_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
    run_fakes: types.SimpleNamespace,
) -> None:
    run_fakes.rows[0]["hypothesis"] = "secret-input" * 30_000
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(_spec()))
    with pytest.raises(ir.RunError, match="no text was truncated"):
        await ir._execute(ir.load_spec(), ir.RunSecrets(), None)  # pyright: ignore[reportPrivateUsage]
    assert len(run_fakes.loads) == 1
    assert run_fakes.inputs == []


async def test_empty_split_refuses_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
    run_fakes: types.SimpleNamespace,
) -> None:
    run_fakes.rows = []
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(_spec()))
    with pytest.raises(ir.RunError, match="split was empty"):
        await ir._execute(ir.load_spec(), ir.RunSecrets(), None)  # pyright: ignore[reportPrivateUsage]
    assert len(run_fakes.loads) == 1
    assert run_fakes.inputs == []


def test_pair_template_placeholders_are_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _spec()
    payload["hyperparams"]["pair_template"] = "{missing}"
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(payload))
    with pytest.raises(ir.RunError, match="missing"):
        ir.load_spec()


@pytest.mark.parametrize(
    "overrides",
    [
        {"engine": "unknown"},
        {"engine": "hf_classify"},
        {"engine": "hf_classify", "pair_template": "h"},
        {"engine": "hf_label_logits", "max_model_len": 128},
        {"engine": "vllm_chat", "pair_template": "h"},
        {"engine": "hf_classify", "pair_template": "h", "max_model_len": 128, "temperature": 0},
        {
            "engine": "hf_classify",
            "pair_template": "h",
            "max_model_len": 128,
            "tensor_parallel_size": 2,
        },
        {"engine": "hf_classify", "pair_template": "h", "max_model_len": 128, "dtype": "pickle"},
        {
            "engine": "hf_classify",
            "pair_template": "h",
            "max_model_len": 128,
            "label_tokens": {"yes": "A"},
        },
        {
            "engine": "hf_label_logits",
            "max_model_len": 128,
            "label_tokens": {"yes": "A"},
            "pair_template": "h",
        },
    ],
)
def test_incompatible_engine_knobs_fail_spec_validation(overrides: dict[str, Any]) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ir.Hyperparams(**overrides)


async def test_stable_ids_preserve_legacy_custom_ids_and_declared_source_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_fakes: types.SimpleNamespace,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    run_fakes.rows[0]["record_id"] = "alpha"
    run_fakes.rows[1]["record_id"] = "beta"
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(_spec(id_column="record_id")))
    assert await ir.main() == 0
    path = run_fakes.results_dir / "results.parquet"
    table = cast("pa.Table", pq.read_table(path))  # pyright: ignore[reportUnknownMemberType] — Arrow reader lacks complete stubs
    rows = cast("list[dict[str, object]]", table.to_pylist())
    assert [row["item_id"] for row in rows] == ["alpha", "beta"]
    assert [row["custom_id"] for row in rows] == ["row-0", "row-1"]
    manifest = json.loads((path.parent / "scoring.json").read_text())
    assert manifest["id_semantics"] == {
        "kind": "stable_string",
        "source_column": "record_id",
        "artifact_column": "item_id",
        "unique": True,
    }
    assert rows[0]["p_yes"] is None
    assert table.schema.field("p_yes").type == pa.float64()


@pytest.mark.parametrize("ids", [["duplicate", "duplicate"], ["valid", 1], ["valid", ""]])
async def test_bad_stable_ids_refuse_before_forward(
    ids: list[object],
    monkeypatch: pytest.MonkeyPatch,
    run_fakes: types.SimpleNamespace,
) -> None:
    for row, value in zip(run_fakes.rows, ids, strict=True):
        row["record_id"] = value
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(_spec(id_column="record_id")))
    assert await ir.main() == 1
    assert run_fakes.inputs == []
    assert not (run_fakes.results_dir / "results.parquet").exists()
