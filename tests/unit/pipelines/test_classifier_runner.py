"""Synthetic streamed rows and tiny randomly initialized CPU models only."""

from __future__ import annotations

import asyncio
import json

import pytest

from strata_forge.pipelines import classifier_runner as worker
from strata_forge.pipelines.classifier_spec import ClassifierTrainSpec, ClassifierWorkerError
from tests.unit.pipelines.test_classifier_spec import payload
from tests.unit.training.test_classifier import source as tiny_source
from tests.unit.training.test_classifier import tokenizer as tiny_tokenizer


class Authority:
    def __init__(self, custody="unsealed", change=None):
        self.calls = []
        self.custody, self.change = custody, change

    async def decide(self, request):
        self.calls.append(request)
        values = {
            **request.model_dump(),
            "authorized": True,
            "custody": self.custody,
            "policy_version": "fictional-v1",
        }
        if self.change:
            values.update(self.change)
        return worker.TrainingReadDecision.model_validate(values)


class Source:
    def __init__(self, path=None, rows=None):
        self.path = path
        self.values = (
            [
                b'{"id":"a","text":"a [PAD]","label":"E"}',
                b'{"id":"b","text":"b","label":"C"}',
            ]
            if rows is None
            else rows
        )
        self.yielded = 0
        self.resolved = []
        self.closed = False

    async def rows(self, source):
        try:
            for raw in self.values:
                self.yielded += 1
                yield raw
        finally:
            self.closed = True

    async def resolve_model(self, ref):
        self.resolved.append(ref)
        assert self.path is not None
        return worker.ResolvedClassifierCheckpoint(ref, self.path)


class ParquetSource(Source):
    def __init__(self, path, file):
        super().__init__(path)
        self.file = file

    async def rows(self, source):
        import pyarrow.parquet as pq

        try:
            for batch in pq.ParquetFile(self.file).iter_batches(
                batch_size=1, columns=list(source.columns.values())
            ):
                for row in batch.to_pylist():
                    self.yielded += 1
                    yield json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8")
        finally:
            self.closed = True


def spec(family="bert"):
    p = payload()
    if family != "bert":
        p["base"]["initialization"] = {
            "kind": "decoder_logit_head",
            "label_tokens": {"E": "a", "C": "b", "U": "c"},
        }
    return ClassifierTrainSpec.model_validate(p)


def save_base(path, family="bert"):
    import torch
    import transformers

    torch.set_num_threads(1)
    native = tiny_source(family, transformers)
    if family == "modernbert_decoder":
        with torch.no_grad():
            native.decoder.bias.copy_(torch.tensor([0.4, -0.6, 0.2, 1.1, -0.2, 0.9, 0.3, -0.1]))
    native.save_pretrained(path, safe_serialization=True)
    tokenizer = tiny_tokenizer()
    tokenizer.eos_token = "[PAD]"
    tokenizer.save_pretrained(path)
    return native


@pytest.mark.parametrize(
    "change",
    [
        {"custody": "sealed"},
        {"custody": "unknown"},
        {"authorized": False},
        {"run_id": "2" * 32},
        {"spec_sha256": "0" * 64},
        {"purpose": "evaluation"},
    ],
)
async def test_wrong_authority_never_reads_source(tmp_path, change):
    source = Source()
    with pytest.raises(ClassifierWorkerError, match="read_denied"):
        await worker.run_classifier_training(
            spec(), authority=Authority(change=change), source=source, attempt_dir=tmp_path / "run"
        )
    assert source.yielded == 0
    assert not source.resolved


async def test_model_resolution_revocation_refuses_before_local_weight_read(tmp_path, monkeypatch):
    policy = Authority()

    class RevokingSource(Source):
        async def resolve_model(self, ref):
            resolved = await super().resolve_model(ref)
            policy.custody = "unknown"
            return resolved

    async def forbidden(path):
        raise AssertionError("revocation must refuse before native model loading")

    monkeypatch.setattr(worker, "load_classifier_base", forbidden)
    source = RevokingSource(tmp_path)
    with pytest.raises(ClassifierWorkerError, match="read_denied"):
        await worker.run_classifier_training(
            spec(), authority=policy, source=source, attempt_dir=tmp_path / "run"
        )
    assert source.resolved == [spec().base.ref]
    assert not (tmp_path / "run").exists()


async def test_byte_limit_and_count_cap_refuse_before_parsing_later_rows(tmp_path, monkeypatch):
    p = payload()
    p["dataset"]["selection"] = {"mode": "all", "row_limit": 1}
    source = Source(rows=[b'{"id":"a","text":"a","label":"E"}', b"not json", b"not fetched"])
    with pytest.raises(ClassifierWorkerError, match="input_limit"):
        await worker.run_classifier_training(
            ClassifierTrainSpec.model_validate(p),
            authority=Authority(),
            source=source,
            attempt_dir=tmp_path / "run",
        )
    assert source.yielded == 2
    assert source.closed
    assert not source.resolved
    source = Source(rows=[b'{"id":"a","text":"a","label":"E"}', b"not json", b"not fetched"])
    monkeypatch.setattr(worker, "MAX_INPUT_BYTES", len(source.values[0]))
    with pytest.raises(ClassifierWorkerError, match="input_limit"):
        await worker.run_classifier_training(
            spec(), authority=Authority(), source=source, attempt_dir=tmp_path / "other"
        )
    assert source.yielded == 2
    assert source.closed
    assert not source.resolved


async def test_skips_count_toward_budget_and_range_is_not_silent_prefix(tmp_path, monkeypatch):
    p = payload()
    p["dataset"]["selection"] = {"mode": "range", "row_start": 1, "row_end": 3, "row_limit": 2}
    source = Source(
        rows=[
            b"not parsed",
            b'{"id":"a","text":"a","label":"E"}',
            b'{"id":"b","text":"b","label":"C"}',
            b"not fetched",
        ]
    )
    prepared, used, _ = await worker._rows(ClassifierTrainSpec.model_validate(p), source)
    assert len(prepared) == 2
    assert used == sum(map(len, source.values[:3]))
    assert source.yielded == 3
    assert source.closed
    source = Source(rows=[b"x" * 20, b'{"id":"a","text":"a","label":"E"}'])
    monkeypatch.setattr(worker, "MAX_INPUT_BYTES", 10)
    with pytest.raises(ClassifierWorkerError, match="input_limit"):
        await worker._rows(ClassifierTrainSpec.model_validate(p), source)
    assert source.yielded == 1


@pytest.mark.parametrize("family", ["bert", "gpt2", "modernbert_decoder"])
async def test_actual_cpu_training_and_safe_receipt(tmp_path, family):
    path = tmp_path / "base"
    original = save_base(path, family)
    import torch

    from strata_forge.pipelines._classifier_checkpoint import reload_classifier
    from strata_forge.training.classifier import initialize_classifier

    if family == "bert":
        initial = await initialize_classifier(
            original, labels=("E", "C", "U"), initializer_id="encoder_random_head", seed=0
        )
        initial_weight = initial.model.classifier.weight.detach().clone()
    else:
        projection = original.lm_head if family == "gpt2" else original.decoder
        initial_weight = projection.weight.detach()[[2, 3, 4]].clone()
    policy, source = Authority(), Source(path)
    result = await worker.run_classifier_training(
        spec(family), authority=policy, source=source, attempt_dir=tmp_path / "run"
    )
    assert result.steps == 1
    assert result.trained_rows == 2
    loaded = await reload_classifier(result.artifact_dir)
    assert not torch.equal(loaded.model.classifier.weight, initial_weight)
    assert (result.artifact_dir / "model.safetensors").is_file()
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["prototype_max_input_bytes"] == 64 * 1024 * 1024
    assert receipt["prototype_max_row_bytes"] == 2 * 1024 * 1024
    assert receipt["stream_bytes"] == sum(map(len, source.values))
    assert receipt["publication"] == "not_supported"
    assert receipt["resume"] == "not_supported"
    assert len(policy.calls) >= 8
    assert not list(result.artifact_dir.rglob("*.bin"))


async def test_cancelled_native_work_cannot_emit_late_success(tmp_path, monkeypatch):
    import threading

    from strata_forge.training.progress import JsonlProgressWriter

    path = tmp_path / "base"
    save_base(path)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def native():
        entered.set()
        release.wait(3)
        finished.set()

    async def held(*args, **kwargs):
        await asyncio.to_thread(native)
        raise AssertionError("cancelled native result must not resume the worker")

    monkeypatch.setattr(worker.ClassifierRunner, "train", held)
    progress = tmp_path / "progress.jsonl"
    writer = JsonlProgressWriter(progress)
    task = asyncio.create_task(
        worker.run_classifier_training(
            spec(),
            authority=Authority(),
            source=Source(path),
            attempt_dir=tmp_path / "run",
            writer=writer,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert await asyncio.to_thread(finished.wait, 3)
        await asyncio.sleep(0)
    finally:
        release.set()
        writer.close()
    assert not (tmp_path / "run/training-receipt.json").exists()
    assert not progress.exists() or '"kind":"end"' not in progress.read_text().replace(" ", "")
