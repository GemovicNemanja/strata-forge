"""A classifier run through the VM runner, on tiny synthetic checkpoints (CPU).

The acceptance smoke for the whole path: spec -> pinned inputs -> every check -> training -> the
artifact with its manifest, pushed in one commit. Run it with::

    uv sync --extra finetuning
    uv run pytest tests/unit/pipelines/test_finetune_seq_cls.py -k smoke

The Hub is never contacted: the split and the checkpoint are generated and handed in.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from strata_forge.pipelines import SPEC_VERSION
from strata_forge.pipelines import finetune_runner as fr
from strata_forge.pipelines._common import RunError, RunSecrets
from strata_forge.training.progress import JsonlProgressWriter

torch: Any = pytest.importorskip("torch")
datasets: Any = pytest.importorskip("datasets")

from tests.unit.training import tiny_classifier as tiny  # noqa: E402 - after the import skips

if TYPE_CHECKING:
    from pathlib import Path

_SHA = "b" * 40
_TOKEN = "hf_secretwritetoken1234567890"
_MAPPING = {
    "premises": "ctx",
    "hypothesis": "claim",
    "label": "gold",
    "label_space": "space",
    "inconsistent_premises": "flag",
    "support_set": "support",
}


def _source_rows(n: int) -> list[dict[str, Any]]:
    """Generated rows under dataset-style column names, plus one column the run must ignore."""
    renamed = {v: k for k, v in _MAPPING.items()}
    return [
        {**{col: row[role] for col, role in renamed.items()}, "unrelated": i}
        for i, row in enumerate(tiny.rows(n))
    ]


def _spec(**overrides: Any) -> fr.FinetuneSpec:
    base: dict[str, Any] = {
        "engine_version": SPEC_VERSION,
        "method": "seq_cls",
        "model_id": "synthetic/tiny-decoder",
        "model_revision": _SHA,
        "dataset_id": "synthetic/rows",
        "dataset_commit_sha": _SHA,
        "split": "train",
        "eval_split": "dev",
        "dataset_format": "nli_pair",
        "column_mapping": _MAPPING,
        "adapter": "none",
        "output_repo_id": "synthetic/out",
        "run_id": "smoke",
        "hyperparams": {
            "labels": list(tiny.LABELS),
            "label_aliases": {"not_entailment": "not_entailed"},
            "head_init": "decoder_lm_head",
            "verbalizers": dict(tiny.VERBALIZERS),
            "max_length": 64,
            "per_device_batch_size": 4,
            "max_steps": 2,
            "precision": "fp32",
            "brier_weight": 0.1,
            "gradient_checkpointing": True,
        },
    }
    for key, value in overrides.items():
        if key == "hyperparams":
            base["hyperparams"] = {**base["hyperparams"], **value}
        else:
            base[key] = value
    return fr.FinetuneSpec.model_validate(base)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny.decoder(tmp_path_factory.mktemp("decoder"))


def _drive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint: Path, spec: fr.FinetuneSpec
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    pushed: dict[str, Any] = {}

    def _open(_spec: Any, split: str, _token: Any) -> Any:
        return datasets.Dataset.from_list(_source_rows(16 if split == "train" else 8))

    async def _fetch(_spec: Any, _token: Any) -> Path:
        return checkpoint

    async def _private(*_a: Any, **_k: Any) -> None:
        return None

    def _record(directory: Path) -> None:
        pushed["files"] = sorted(p.name for p in directory.iterdir())
        pushed["manifest"] = json.loads((directory / "manifest.json").read_text())

    async def _push(_spec: Any, directory: Path, _token: str) -> str:
        await asyncio.to_thread(_record, directory)
        return "synthetic/out"

    def _dir(_run_id: str | None, *, name: str) -> Path:
        return tmp_path / name

    monkeypatch.setattr(fr, "_open_split", _open)
    monkeypatch.setattr(fr, "_fetch_checkpoint", _fetch)
    monkeypatch.setattr(fr, "ensure_private_repo", _private)
    monkeypatch.setattr(fr, "_push_artifact", _push)
    monkeypatch.setattr(fr, "results_dir", _dir)
    progress = tmp_path / "progress.jsonl"
    writer = JsonlProgressWriter(progress)
    secrets = RunSecrets(hf_token=SecretStr(_TOKEN))
    try:
        destination = asyncio.run(fr._execute(spec, secrets, writer))  # pyright: ignore[reportPrivateUsage]
    finally:
        writer.close()
    events = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    return destination, events, pushed


class TestAcceptanceSmoke:
    def test_smoke_decoder_run_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint: Path
    ) -> None:
        destination, events, pushed = _drive(monkeypatch, tmp_path, checkpoint, _spec())
        assert destination == "synthetic/out"

        # Every check reported itself before training began, in order.
        checks = [
            e["message"].split(":")[0]
            for e in events
            if e["kind"] == "phase" and "check" in e["message"].split(":")[0]
        ]
        assert checks == [
            "Length check passed",
            "Load check passed",
            "Step-0 check passed",
            "Memory check ran without a GPU to measure (activation checkpointing on)",
        ]
        step0 = next(e for e in events if e["message"].startswith("Step-0 check passed"))
        assert step0["metrics"]["step0_max_abs_logit_diff"] < 1e-5
        assert events[-1]["kind"] == "end"
        assert events[-1]["metrics"]["gpu_seconds"] == 0.0

        # The artifact and its manifest went up together; the checkpoints did not.
        assert {"manifest.json", "model.safetensors", "strata_verifier.json"} <= set(
            pushed["files"]
        )
        assert not any(name.startswith("checkpoint-") for name in pushed["files"])
        manifest = pushed["manifest"]
        assert manifest["kind"] == "finetune"
        assert manifest["outcome"] == "succeeded"
        assert manifest["model"] == {
            "id": "synthetic/tiny-decoder",
            "requested_revision": _SHA,
            "commit": _SHA,
        }
        assert manifest["dataset"]["requested_revision"] == _SHA
        assert manifest["counts"] == {"train_rows": 16, "eval_rows": 8}
        assert manifest["hyperparams"]["gradient_checkpointing"] is True
        assert manifest["checks"]["step0"]["passed"] is True
        assert manifest["checks"]["memory"]["gradient_checkpointing"] is True
        assert manifest["checks"]["overflow"]["rows_over_max_length"] == 0
        assert manifest["compute"]["gpu_seconds"] == 0.0
        assert set(manifest["compute"]["timing"]) >= {
            "encode_s",
            "load_s",
            "preflight_s",
            "train_s",
        }
        # No row content anywhere in the record.
        text = json.dumps(manifest)
        assert not any(word in text for word in tiny.WORDS[:5])

    def test_an_overflow_refuses_before_the_model_loads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint: Path
    ) -> None:
        spec = _spec(hyperparams={"max_length": 8})
        with pytest.raises(RunError, match="exceed max_length 8"):
            _drive(monkeypatch, tmp_path, checkpoint, spec)
        refused = json.loads((tmp_path / "strata-finetune-output" / "manifest.json").read_text())
        assert refused["outcome"] == "refused"
        assert refused["checks"]["receipt"]["rows_over_max_length"] > 0
        assert refused["model"]["requested_revision"] == _SHA


class TestRoleSplit:
    def test_projects_onto_roles_and_takes_a_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _open(*_a: Any) -> Any:
            return datasets.Dataset.from_list(_source_rows(10))

        monkeypatch.setattr(fr, "_open_split", _open)
        split = fr._load_role_split(_spec(row_limit=4), "train", False)  # pyright: ignore[reportPrivateUsage]
        assert sorted(split.column_names) == sorted(_MAPPING)
        assert len(split) == 4

    def test_an_unmapped_required_role_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _open(*_a: Any) -> Any:
            return datasets.Dataset.from_list(_source_rows(2))

        monkeypatch.setattr(fr, "_open_split", _open)
        mapping = {k: v for k, v in _MAPPING.items() if k != "hypothesis"}
        with pytest.raises(RunError, match="hypothesis"):
            fr._load_role_split(_spec(column_mapping=mapping), "train", False)  # pyright: ignore[reportPrivateUsage]

    def test_a_download_at_another_commit_refuses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        snapshot = tmp_path / "snapshots" / ("c" * 40)
        snapshot.mkdir(parents=True)

        class _Hub:
            def __init__(self, **_k: Any) -> None: ...

            async def download_snapshot(self, *_a: Any, **_k: Any) -> Path:
                return snapshot

        monkeypatch.setattr(fr, "HFHubClient", _Hub)
        with pytest.raises(RunError, match="not the pinned"):
            asyncio.run(fr._fetch_checkpoint(_spec(), False))  # pyright: ignore[reportPrivateUsage]
