"""Actual toy CPU worker export → default pinned hf_classify → actual evaluation."""

from __future__ import annotations

import json
import types

import pytest

from strata_forge.pipelines import _scoring, eval_runner
from strata_forge.pipelines._classifier_checkpoint import reload_classifier
from strata_forge.pipelines.classifier_runner import run_classifier_training
from strata_forge.pipelines.classifier_spec import ClassifierTrainSpec, ClassifierWorkerError
from strata_forge.pipelines.evaluation_spec import EvalSpec
from tests.unit.pipelines.test_classifier_runner import (
    Authority,
    ParquetSource,
    Source,
    save_base,
    spec,
)
from tests.unit.pipelines.test_eval_runner import Policy, Reader, spec_data


def install_hub_transport(monkeypatch, artifact):
    import huggingface_hub as hub

    calls = []

    def download(repo_id, filename, *, revision, token):
        assert repo_id == "example/synthetic-model"
        assert revision == "b" * 40
        assert token is False
        calls.append(filename)
        return str(artifact / filename)

    monkeypatch.setattr(hub, "hf_hub_download", download)
    monkeypatch.setattr(
        hub,
        "HfApi",
        lambda *, token: types.SimpleNamespace(
            list_repo_files=lambda *args, **kwargs: sorted(p.name for p in artifact.iterdir())
        ),
    )
    return calls


@pytest.mark.parametrize("family", ["bert", "gpt2", "modernbert_decoder"])
async def test_default_pinned_export_scores_and_evaluates_without_model_download(
    tmp_path, monkeypatch, family
):
    import torch

    base = tmp_path / "base"
    save_base(base, family)
    training = spec(family)
    source = Source(base)
    if family != "bert":
        values = training.model_dump(mode="json")
        values["dataset"]["format"] = "nli_pair"
        values["dataset"]["columns"] = {
            "item_id": "id",
            "premises": "premises",
            "hypothesis": "hypothesis",
            "label": "label",
        }
        values["training"]["rendering"] = {
            "labels": ["E", "C", "U"],
            "template": "{premises} {hypothesis}",
            "pair_template": None,
            "premise_format": "numbered_lines",
            "max_length": 16,
            "max_premises": 20,
            "label_tokens": {"E": "a", "C": "b", "U": "c"},
        }
        training = ClassifierTrainSpec.model_validate(values)
        source = Source(
            base,
            rows=[
                b'{"id":"a","premises":["a"],"hypothesis":"b [PAD]","label":"E"}',
                b'{"id":"b","premises":["c"],"hypothesis":"d","label":"C"}',
            ],
        )
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet = tmp_path / "synthetic-train.parquet"
        pq.write_table(pa.Table.from_pylist([json.loads(raw) for raw in source.values]), parquet)
        source = ParquetSource(base, parquet)
    result = await run_classifier_training(
        training, authority=Authority(), source=source, attempt_dir=tmp_path / "train"
    )
    calls = install_hub_transport(monkeypatch, result.artifact_dir)
    scored = await _scoring.load_scorer(
        "example/synthetic-model",
        engine="hf_classify",
        token=False,
        revision="b" * 40,
        max_length=16,
    )
    inputs = [_scoring.ScoringInput(text="a [PAD]"), _scoring.ScoringInput(text="b")]
    actual = await scored.score(inputs, batch_size=2)
    loaded = await reload_classifier(result.artifact_dir)
    loaded.model.eval()
    features = loaded.tokenizer([r.text for r in inputs], padding=True, return_tensors="pt")
    with torch.no_grad():
        expected = loaded.model(**features).logits.float().softmax(-1).tolist()
    for row, values in zip(actual.rows, expected, strict=True):
        assert list(row.probabilities) == ["E", "C", "U"]
        assert list(row.probabilities.values()) == pytest.approx(values, abs=1e-6)
    assert calls[0] == "strata_classifier.json"
    p = spec_data("classify")
    p["evaluation_sets"] = p["evaluation_sets"][:1]
    reader = Reader()
    if family == "bert":
        p["rendering"]["max_length"] = 16
    else:
        p["rendering"] = {
            "source": "model_artifact",
            "file": "strata_verifier.json",
            "overrides": {},
        }
        reader.metadata[("example/synthetic-model@" + "b" * 40, "strata_verifier.json")] = (
            json.loads((result.artifact_dir / "strata_verifier.json").read_text())
        )
        reader.metadata[("example/synthetic-model@" + "b" * 40, "strata_classifier.json")] = (
            json.loads((result.artifact_dir / "strata_classifier.json").read_text())
        )
        reader.metadata[("example/synthetic-model@" + "b" * 40, "config.json")] = json.loads(
            (result.artifact_dir / "config.json").read_text()
        )
    # Default scorer_loader is deliberately not replaced: only the fictional Hub transport is.
    evaluated = await eval_runner.run_evaluation(
        EvalSpec.model_validate(p), policy=Policy(), reader=reader, output_dir=tmp_path / "eval"
    )
    report = json.loads(evaluated.metrics_path.read_text())
    assert len(report["evaluations"]) == 1
    entry = report["evaluations"][0]
    assert entry["classification"]["accuracy"]["value"] is not None
    assert entry["metrics"]["brier_raw"]["result"]["coverage"] == 1
    assert evaluated.output_ref is None
    assert list((tmp_path / "eval").rglob("*.parquet"))
    if family != "bert":
        assert (
            entry["source"]["rendering"]["original_sha256"]
            == entry["source"]["rendering"]["effective_sha256"]
        )
        assert entry["source"]["rendering"]["label_tokens_role"] == "initialization_provenance"
        assert entry["source"]["rendering"]["probability_mode"] == "learned_head_full_class"
        calls_before = len(calls)
        p["rendering"]["overrides"] = {"max_length": 15}
        with pytest.raises(eval_runner.EvaluationError, match="rendering_engine_mismatch"):
            await eval_runner.run_evaluation(
                EvalSpec.model_validate(p),
                policy=Policy(),
                reader=reader,
                output_dir=tmp_path / "drift",
            )
        assert len(calls) == calls_before
        p["rendering"]["overrides"] = {}
        metadata = reader.metadata[
            ("example/synthetic-model@" + "b" * 40, "strata_classifier.json")
        ]
        for changed in ("receipt", "tokens"):
            bad = json.loads(json.dumps(metadata))
            if changed == "receipt":
                bad["initialization"]["labels"] = ["C", "E", "U"]
            else:
                bad["config"]["rendering"]["label_tokens"]["E"] = "c"
            reader.metadata[("example/synthetic-model@" + "b" * 40, "strata_classifier.json")] = bad
            with pytest.raises(eval_runner.EvaluationError, match="rendering_engine_mismatch"):
                await eval_runner.run_evaluation(
                    EvalSpec.model_validate(p),
                    policy=Policy(),
                    reader=reader,
                    output_dir=tmp_path / changed,
                )
            assert len(calls) == calls_before
        # A plausible but incorrect resolved-token receipt must fail against the actual saved
        # tokenizer before native model weights are loaded, even when all render digests agree.
        value = json.loads((result.artifact_dir / "strata_classifier.json").read_text())
        value["initialization"]["verbalizer_ids"] = [3, 2, 4]
        (result.artifact_dir / "strata_classifier.json").write_text(json.dumps(value))
        import strata_forge.pipelines._classifier_checkpoint as checkpoints

        def no_weights(*args, **kwargs):
            raise AssertionError("token receipt mismatch must precede model loading")

        monkeypatch.setattr(checkpoints, "_load_complete_classifier", no_weights)
        with pytest.raises(ClassifierWorkerError):
            await _scoring.load_scorer(
                "example/synthetic-model",
                engine="hf_classify",
                token=False,
                revision="b" * 40,
                max_length=16,
            )


async def test_no_eval_authority_never_probes_sidecar(tmp_path, monkeypatch):
    import huggingface_hub as hub

    def forbidden(*args, **kwargs):
        raise AssertionError("no model probe before policy")

    monkeypatch.setattr(hub, "hf_hub_download", forbidden)
    with pytest.raises(eval_runner.EvaluationError, match="custody_policy_unavailable"):
        await eval_runner.run_evaluation(
            EvalSpec.model_validate(spec_data("classify")),
            reader=Reader(),
            output_dir=tmp_path / "denied",
        )


@pytest.mark.parametrize("failure", ["malformed", "missing_head", "labels", "counter"])
async def test_recognized_pinned_damage_never_falls_back_to_native(tmp_path, monkeypatch, failure):
    import transformers
    from safetensors.torch import load_file, save_file

    base = tmp_path / "base"
    save_base(base, "modernbert_decoder")
    result = await run_classifier_training(
        spec("modernbert_decoder"),
        authority=Authority(),
        source=Source(base),
        attempt_dir=tmp_path / "train",
    )
    metadata = result.artifact_dir / "strata_classifier.json"
    if failure == "malformed":
        metadata.write_text('{"private-canary":NaN}')
    elif failure == "missing_head":
        file = result.artifact_dir / "model.safetensors"
        weights = load_file(file)
        del weights["classifier.weight"]
        save_file(weights, file, metadata={"format": "pt"})
    else:
        value = json.loads(metadata.read_text())
        if failure == "labels":
            value["config"]["labels"] = ["C", "E", "U"]
        else:
            value["steps"] = True
        metadata.write_text(json.dumps(value))
    install_hub_transport(monkeypatch, result.artifact_dir)

    def forbidden(*args, **kwargs):
        raise AssertionError("native fallback is forbidden")

    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification, "from_pretrained", forbidden
    )
    with pytest.raises(ClassifierWorkerError) as error:
        await _scoring.load_scorer(
            "example/synthetic-model",
            engine="hf_classify",
            token=False,
            revision="b" * 40,
            max_length=16,
        )
    assert error.value.__context__ is None
    assert "private-canary" not in str(error.value)
