"""An evaluation over a trained verifier's own files, exactly as the trainer and the scorer write them.

A tiny synthetic verifier is trained by ``seq_cls`` on CPU, with a calibration split, so its
artifact carries the trainer's ``calibration.json`` (``strata.calibration/v1``) beside its
``strata_verifier.json``. The scoring runner then scores a split with ``artifact:
strata_verifier`` and writes its results parquet, and the evaluation runner reads that parquet as
its ``predictions`` and that ``calibration.json`` as the system's ``calibration_ref``, both under
the names they were written with. Only the Hub is stood in for. Run it with::

    uv sync --extra finetuning --extra scoring --extra hf --extra storage
    uv run pytest tests/unit/pipelines/test_eval_consumes_verifier.py
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from strata_forge.metrics.calibration import temperature_scale
from strata_forge.metrics.classification import brier
from strata_forge.pipelines import eval_runner
from strata_forge.pipelines import inference_runner as ir
from strata_forge.pipelines._common import RunSecrets

torch: Any = pytest.importorskip("torch")
datasets: Any = pytest.importorskip("datasets")
pq: Any = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("safetensors.torch")

from strata_forge.training.seq_cls import SeqClsConfig, SeqClsRunner  # noqa: E402
from tests.unit.training import tiny_classifier as tiny  # noqa: E402 - after the import skips

if TYPE_CHECKING:
    from pathlib import Path

_MODEL_SHA = "a" * 40
_GOLD_SHA = "b" * 40
_SCORES_SHA = "c" * 40
LABELS = list(tiny.LABELS)


class _Probe:
    def reset(self) -> None:
        return None

    def peak_bytes(self) -> int | None:
        return 10

    def total_bytes(self) -> int | None:
        return 1000


def _three_way(n: int, *, seed: int) -> list[dict[str, Any]]:
    """Generated 3-way rows (the evaluation's label space) with a stable id and a cluster."""
    rows = [r for r in tiny.rows(n, seed=seed, two_way_every=10**6) if r["label_space"] == "3way"]
    return [{**row, "uid": f"u{i}", "base": f"b{i // 2}"} for i, row in enumerate(rows)]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A trained encoder verifier, with the trainer's own calibration.json."""
    root = tmp_path_factory.mktemp("verifier")
    config = SeqClsConfig.model_validate(
        {
            "model_id": "synthetic/tiny",
            "output_dir": str(root / "checkpoints"),
            "labels": tiny.LABELS,
            **tiny.TWO_WAY,
            "max_length": 64,
            "per_device_batch_size": 8,
            "max_steps": 2,
        }
    )
    out = root / "artifact"
    SeqClsRunner(config, memory_probe=_Probe()).run(
        checkpoint=tiny.encoder(root / "base"),
        artifact_dir=out,
        train_dataset=datasets.Dataset.from_list(tiny.rows(16)),
        calibration_dataset=datasets.Dataset.from_list(tiny.rows(8, seed=2)),
    )
    return out


def _score(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, artifact: Path, rows: Any) -> Path:
    """Run the scoring runner with ``artifact: strata_verifier``; return its results parquet."""
    spec = ir.RunSpec.model_validate(
        {
            "model_id": "synthetic/verifier",
            "model_revision": _MODEL_SHA,
            "dataset_id": "synthetic/gold",
            "dataset_commit_sha": _GOLD_SHA,
            "split": "test",
            "template": "{premises}",
            "column_mapping": {"premises": "premises", "hypothesis": "hypothesis"},
            "hyperparams": {
                "engine": "hf_classify",
                "artifact": "strata_verifier",
                "pair_template": "{hypothesis}",
                "dtype": "float32",
                "id_column": "uid",
            },
            "run_id": "scores",
        }
    )

    async def commit(_spec: Any, _token: Any) -> str:
        return _GOLD_SHA

    async def fetch(_spec: Any, _token: Any) -> Path:
        return artifact

    def load_rows(_spec: Any, _token: Any) -> list[dict[str, Any]]:
        return rows

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(ir, "_dataset_commit", commit)
    monkeypatch.setattr(ir, "_fetch_model", fetch)
    monkeypatch.setattr(ir, "_load_rows", load_rows)
    asyncio.run(ir._execute_scoring(spec, RunSecrets(), None))  # pyright: ignore[reportPrivateUsage]
    return tmp_path / "strata-inference-results" / "scores" / "results.parquet"


class _LocalHub:
    files: ClassVar[dict[tuple[str, str, str], Path]] = {}

    def __init__(self, token: Any = None) -> None:
        del token

    async def download_file(self, repo_id: str, path: str, **kwargs: Any) -> Path:
        return _LocalHub.files[(repo_id, kwargs["revision"], path)]


def test_the_trainer_s_calibration_file_is_the_schema_the_evaluation_reads(artifact: Path) -> None:
    body = json.loads((artifact / "calibration.json").read_text())
    assert body["schema"] == "strata.calibration/v1"
    assert body["fit"]["temperature"] > 0
    assert body["labels"] == LABELS
    verifier = json.loads((artifact / "strata_verifier.json").read_text())
    assert verifier["labels"] == LABELS


@pytest.mark.parametrize("with_labels", [True, False])
def test_an_evaluation_reads_the_verifier_s_scores_and_calibration_as_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, artifact: Path, with_labels: bool
) -> None:
    rows = _three_way(24, seed=7)
    results = _score(monkeypatch, tmp_path, artifact, rows)
    written = pq.read_table(str(results)).to_pylist()
    # The verifier's file: its labels' logits and probabilities, plus the auxiliary heads.
    assert {f"logit_{label}" for label in LABELS} <= set(written[0])
    assert {"item_id", "pred", "overflow"} <= set(written[0])

    calibration = artifact / "calibration.json"
    if not with_labels:
        # A calibration file from a trainer that recorded its labels only in strata_verifier.json.
        body = json.loads(calibration.read_text())
        del body["labels"]
        calibration = tmp_path / "calibration-without-labels.json"
        calibration.write_text(json.dumps(body))
    _LocalHub.files = {
        ("synthetic/scores", _SCORES_SHA, "results.parquet"): results,
        ("synthetic/verifier", _MODEL_SHA, "calibration.json"): calibration,
        ("synthetic/verifier", _MODEL_SHA, "strata_verifier.json"): artifact
        / "strata_verifier.json",
    }
    gold = {key: [row[key] for row in rows] for key in ("uid", "label", "base", "premises")}

    def load_gold(_spec: Any, _token: Any) -> dict[str, list[Any]]:
        return gold

    spec = eval_runner.EvalSpec.model_validate(
        {
            "model_id": "synthetic/verifier",
            "dataset_id": "synthetic/gold",
            "dataset_commit_sha": _GOLD_SHA,
            "split": "test",
            "run_id": "eval",
            "evaluation": {
                "labels": LABELS,
                "predictions": {
                    "name": "verifier",
                    "repo_id": "synthetic/scores",
                    "commit_sha": _SCORES_SHA,
                    "id_column": "item_id",
                    "prediction_column": "pred",
                    "overflow_column": "overflow",
                    # The trainer's file as written; without its own labels the evaluation
                    # takes them from the strata_verifier.json beside it at the same commit.
                    "calibration_ref": {
                        "repo_id": "synthetic/verifier",
                        "commit_sha": _MODEL_SHA,
                        "path": "calibration.json",
                    },
                },
                "id_column": "uid",
                "gold_column": "label",
                "cluster_column": "base",
                "premises_column": "premises",
                "seed": 20261104,
                "resamples": 200,
                "min_slice_rows": 1,
                "calibration": {"apply_calibration": "auto"},
            },
        }
    )
    monkeypatch.setattr(eval_runner, "HFHubClient", _LocalHub)
    monkeypatch.setattr(eval_runner, "_load_gold", load_gold)
    asyncio.run(eval_runner._execute(spec, RunSecrets(), None))  # pyright: ignore[reportPrivateUsage]
    report = json.loads((tmp_path / "strata-eval-results" / "eval" / "metrics.json").read_text())

    assert report["systems"][0]["counts"]["scored"] == len(rows)
    assert report["systems"][0]["counts"]["missing"] == 0
    own = report["calibration"]["systems"]["verifier"]["own"]
    trained = json.loads((artifact / "calibration.json").read_text())["fit"]["temperature"]
    assert own["temperature"] == trained
    assert own["labels_from"] == ("calibration.json" if with_labels else "strata_verifier.json")
    overall = report["slices"][0]["systems"]["verifier"]
    assert overall["raw"]["coverage"] == 1.0
    assert overall["applied"]["n"] == len(rows)
    # The applied read is the trainer's temperature on the scorer's raw logits, nothing else.
    by_id = {row["item_id"]: row for row in written}
    logits = [[by_id[row["uid"]][f"logit_{label}"] for label in LABELS] for row in rows]
    expected = brier(
        [row["label"] for row in rows],
        temperature_scale(logits, temperature=trained),
        labels=LABELS,
        probability_kind="class_probabilities",
    ).value
    assert overall["applied"]["brier"]["point"] == pytest.approx(expected, abs=1e-12)
