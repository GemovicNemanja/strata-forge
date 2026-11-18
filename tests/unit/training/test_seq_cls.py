"""The classifier runner on tiny synthetic checkpoints (CPU): every check, and whole runs.

Run just the acceptance smoke and the step-0 check with::

    uv sync --extra finetuning
    uv run pytest tests/unit/training/test_seq_cls.py -k "smoke or step0"

Everything is generated (``tiny_classifier``); no real model or dataset is read.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from strata_forge.training.classifier_heads import (
    AUX_HEADS_FILENAME,
    HeadLoadError,
    check_load,
    head_plan,
)
from strata_forge.training.seq_cls import (
    CALIBRATION_FILENAME,
    VERIFIER_CONFIG_FILENAME,
    ArtifactCheckError,
    PreflightError,
    SeqClsConfig,
    SeqClsRunner,
    Step0Error,
)

torch: Any = pytest.importorskip("torch")
transformers: Any = pytest.importorskip("transformers")
datasets: Any = pytest.importorskip("datasets")
safetensors_torch: Any = pytest.importorskip("safetensors.torch")

from tests.unit.training import tiny_classifier as tiny  # noqa: E402 - after the import skips

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def encoder_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny.encoder(tmp_path_factory.mktemp("encoder"))


@pytest.fixture(scope="module")
def decoder_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny.decoder(tmp_path_factory.mktemp("decoder"))


@pytest.fixture(scope="module")
def tied_decoder_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny.decoder(tmp_path_factory.mktemp("tied"), tied=True)


def _dataset(n: int = 24, **kwargs: Any) -> Any:
    return datasets.Dataset.from_list(tiny.rows(n, **kwargs))


def _config(tmp_path: Path, **overrides: Any) -> SeqClsConfig:
    base: dict[str, Any] = {
        "model_id": "synthetic/tiny",
        "output_dir": str(tmp_path / "checkpoints"),
        "labels": tiny.LABELS,
        **tiny.TWO_WAY,
        "max_length": 64,
        "per_device_batch_size": 4,
        "max_steps": 3,
        "learning_rate": 1e-3,
        "logging_steps": 1,
        "precision": "fp32",
        "brier_weight": 0.1,
    }
    return SeqClsConfig(**{**base, **overrides})


def _decoder_config(tmp_path: Path, **overrides: Any) -> SeqClsConfig:
    return _config(
        tmp_path, head_init="decoder_lm_head", verbalizers=dict(tiny.VERBALIZERS), **overrides
    )


class _Probe:
    """A memory probe with fixed readings, standing in for a GPU allocator.

    Each reading after a reset is the next of ``peaks`` (the last repeats), so a test can give the
    training step and the eval pass different peaks.
    """

    def __init__(self, peak: int | list[int], total: int) -> None:
        self._peaks = peak if isinstance(peak, list) else [peak]
        self._total = total
        self._resets = 0

    def reset(self) -> None:
        self._resets += 1

    def peak_bytes(self) -> int | None:
        return self._peaks[min(self._resets, len(self._peaks)) - 1]

    def total_bytes(self) -> int | None:
        return self._total


# ----------------------------------------------------------------------------- encoding


class TestEncode:
    def test_targets_flags_and_support_markers(self, tmp_path: Path, encoder_dir: Path) -> None:
        runner = SeqClsRunner(_config(tmp_path))
        tokenizer = runner.load_tokenizer(encoder_dir)
        rows = tiny.rows(12)
        encoded, receipt = runner.encode(datasets.Dataset.from_list(rows), tokenizer, split="t")
        assert receipt["rows"] == 12
        assert receipt["rows_over_max_length"] == 0
        assert receipt["template_token_overhead"] == 3  # [CLS] [SEP] [SEP] around a pair
        for row, enc in zip(rows, encoded, strict=True):
            if row["label_space"] == "3way":
                assert enc["two_way"] == 0
                assert enc["target"][tiny.LABELS.index(row["label"])] == 1.0
            else:
                assert enc["two_way"] == 1
                assert enc["two_way_target"] == (1.0 if row["label"] == "entailed" else 0.0)
            flag = row["inconsistent_premises"]
            assert enc["flag"] == (-1.0 if flag is None else float(flag))
            if row["support_set"] is None:
                assert enc["support_row"] == 0
            else:
                # Each marker is the token of its premise id, and only the support set is 1.
                tokens = [enc["input_ids"][i] for i in enc["support_positions"]]
                pids = [p["pid"] for p in row["premises"]]
                assert tokens == tokenizer.convert_tokens_to_ids(pids)
                assert enc["support_target"] == [
                    1.0 if pid in row["support_set"] else 0.0 for pid in pids
                ]

    def test_overflow_refuses_the_run_and_counts_the_rows(
        self, tmp_path: Path, encoder_dir: Path
    ) -> None:
        runner = SeqClsRunner(_config(tmp_path, max_length=8))
        tokenizer = runner.load_tokenizer(encoder_dir)
        with pytest.raises(PreflightError, match="exceed max_length 8") as exc:
            runner.encode(_dataset(10), tokenizer, split="train")
        assert exc.value.receipt["rows_over_max_length"] > 0
        assert exc.value.receipt["longest"] > 8
        assert "never truncated" in str(exc.value)

    def test_an_integer_label_needs_an_index_map(self, tmp_path: Path, encoder_dir: Path) -> None:
        rows = [{"premises": [{"pid": "p1", "text": "w1 w2"}], "hypothesis": "w3", "label": 2}]
        runner = SeqClsRunner(_config(tmp_path))
        tokenizer = runner.load_tokenizer(encoder_dir)
        with pytest.raises(PreflightError, match="explicit index map"):
            runner.encode(datasets.Dataset.from_list(rows), tokenizer, split="t")
        mapped = SeqClsRunner(_config(tmp_path, label_index_map={"2": "neutral"}))
        encoded, _ = mapped.encode(
            datasets.Dataset.from_list(rows), mapped.load_tokenizer(encoder_dir), split="t"
        )
        assert encoded[0]["target"] == [0.0, 0.0, 1.0]

    def test_an_unknown_label_is_named_by_row(self, tmp_path: Path, encoder_dir: Path) -> None:
        rows = [{"premises": [{"pid": "p1", "text": "w1"}], "hypothesis": "w3", "label": "maybe"}]
        runner = SeqClsRunner(_config(tmp_path))
        with pytest.raises(PreflightError, match="row 0: label is neither"):
            runner.encode(
                datasets.Dataset.from_list(rows), runner.load_tokenizer(encoder_dir), split="t"
            )

    def test_a_soft_label_distribution(self, tmp_path: Path, encoder_dir: Path) -> None:
        rows = [
            {
                "premises": [{"pid": "p1", "text": "w1"}],
                "hypothesis": "w2",
                "label": "neutral",
                "label_dist": [0.2, 0.3, 0.5],
            }
        ]
        runner = SeqClsRunner(_config(tmp_path))
        tokenizer = runner.load_tokenizer(encoder_dir)
        encoded, _ = runner.encode(datasets.Dataset.from_list(rows), tokenizer, split="t")
        assert encoded[0]["target"] == pytest.approx([0.2, 0.3, 0.5])
        bad = [{**rows[0], "label_dist": [0.5, 0.5, 0.5]}]
        with pytest.raises(PreflightError, match="sum to 1"):
            runner.encode(datasets.Dataset.from_list(bad), tokenizer, split="t")

    @pytest.mark.parametrize(
        "dist", [[float("nan"), 0.5, 0.5], [float("inf"), 0.0, 0.0], "abc", ["S", "x", "y"]]
    )
    def test_a_malformed_label_distribution_refuses_without_echoing_it(
        self, tmp_path: Path, encoder_dir: Path, dist: Any
    ) -> None:
        rows = [
            {
                "premises": [{"pid": "p1", "text": "w1"}],
                "hypothesis": "w2",
                "label": "neutral",
                "label_dist": dist,
            }
        ]
        runner = SeqClsRunner(_config(tmp_path))
        tokenizer = runner.load_tokenizer(encoder_dir)
        with pytest.raises(PreflightError) as exc:
            runner.encode(datasets.Dataset.from_list(rows), tokenizer, split="t")
        assert "label_dist" in str(exc.value) or "could not be encoded" in str(exc.value)
        assert "abc" not in str(exc.value)
        assert "'S'" not in str(exc.value)

    def test_an_integer_label_refusal_does_not_echo_the_value(
        self, tmp_path: Path, encoder_dir: Path
    ) -> None:
        rows = [{"premises": [{"pid": "p1", "text": "w1"}], "hypothesis": "w3", "label": 987654}]
        runner = SeqClsRunner(_config(tmp_path))
        with pytest.raises(PreflightError) as exc:
            runner.encode(
                datasets.Dataset.from_list(rows), runner.load_tokenizer(encoder_dir), split="t"
            )
        assert "987654" not in str(exc.value)

    def test_a_two_way_row_needs_declared_labels(self, tmp_path: Path, encoder_dir: Path) -> None:
        rows = [{**tiny.rows(1)[0]}]  # row 0 is two-way
        assert rows[0]["label_space"] == "2way_entail_vs_not"
        bare = _config(tmp_path).model_copy(
            update={"two_way_positive_label": None, "two_way_labels": None}
        )
        runner = SeqClsRunner(bare)
        with pytest.raises(PreflightError, match="there is no default"):
            runner.encode(
                datasets.Dataset.from_list(rows), runner.load_tokenizer(encoder_dir), split="t"
            )
        declared = SeqClsRunner(_config(tmp_path))
        odd = [{**rows[0], "label": "contradiction"}]
        with pytest.raises(PreflightError, match="declared two_way_labels"):
            declared.encode(
                datasets.Dataset.from_list(odd), declared.load_tokenizer(encoder_dir), split="t"
            )

    def test_the_length_distribution_is_recorded(self, tmp_path: Path, encoder_dir: Path) -> None:
        runner = SeqClsRunner(_config(tmp_path))
        encoded, receipt = runner.encode(
            _dataset(20), runner.load_tokenizer(encoder_dir), split="t"
        )
        lengths = sorted(encoded["length"])
        dist = receipt["length_distribution"]
        assert dist["min"] == lengths[0]
        assert dist["max"] == lengths[-1]
        assert dist["p50"] == lengths[9]  # nearest rank: ceil(0.5 * 20) = 10th
        assert dist["p95"] == lengths[18]  # ceil(0.95 * 20) = 19th
        assert receipt["rows_over_max_length"] == 0

    def test_a_support_set_naming_an_unknown_premise_refuses(
        self, tmp_path: Path, encoder_dir: Path
    ) -> None:
        row = tiny.rows(1)[0]
        row = {**row, "label_space": "3way", "label": "entailment", "support_set": ["p9"]}
        runner = SeqClsRunner(_config(tmp_path))
        with pytest.raises(PreflightError, match="premise id the row does not have"):
            runner.encode(
                datasets.Dataset.from_list([row]), runner.load_tokenizer(encoder_dir), split="t"
            )

    def test_the_decoder_template_overhead_is_recorded(
        self, tmp_path: Path, decoder_dir: Path
    ) -> None:
        runner = SeqClsRunner(_decoder_config(tmp_path))
        tokenizer = runner.load_tokenizer(decoder_dir)
        _, receipt = runner.encode(_dataset(4), tokenizer, split="t")
        # [CLS] and [SEP] around the literal rendering; the newline adds no token here.
        assert receipt["template_token_overhead"] == 2


# ----------------------------------------------------------------------------- load guard


def _load(path: Path, num_labels: int = 3) -> tuple[Any, Any]:
    return transformers.AutoModelForSequenceClassification.from_pretrained(
        str(path), num_labels=num_labels, output_loading_info=True
    )


def _rewrite(
    source: Path, target: Path, *, drop: str | None = None, add: str | None = None
) -> Path:
    """A copy of a checkpoint with one tensor dropped or one stray tensor added."""
    import shutil

    shutil.copytree(source, target)
    state = safetensors_torch.load_file(str(target / "model.safetensors"))
    if drop:
        del state[drop]
    if add:
        state[add] = torch.zeros(2)
    safetensors_torch.save_file(state, str(target / "model.safetensors"), metadata={"format": "pt"})
    return target


class TestLoadGuard:
    def test_the_encoder_may_miss_only_its_classifier(self, encoder_dir: Path) -> None:
        model, info = _load(encoder_dir)
        receipt = check_load(model, info, "new")
        assert receipt.missing == ["classifier.bias", "classifier.weight"]

    def test_a_mean_pooling_encoder_config_is_read_at_its_first_token(self, tmp_path: Path) -> None:
        checkpoint = tiny.encoder(tmp_path / "mean", pooling="mean")
        assert json.loads((checkpoint / "config.json").read_text())["classifier_pooling"] == "mean"
        runner = SeqClsRunner(_config(tmp_path))
        tokenizer = runner.load_tokenizer(checkpoint)
        module, receipt, _ = runner.load_model(checkpoint, tokenizer, roles=frozenset())
        assert module.classifier.config.classifier_pooling == "cls"
        assert receipt.pooling_changed_from == "mean"
        result, _, artifact = _smoke(tmp_path / "run", checkpoint, _config(tmp_path / "run"))
        assert result.receipts["load"]["pooling_changed_from"] == "mean"
        verifier = json.loads((artifact / VERIFIER_CONFIG_FILENAME).read_text())
        assert verifier["classifier_pooling"] == "cls"
        assert verifier["pooling"] == "first_token"
        assert json.loads((artifact / "config.json").read_text())["classifier_pooling"] == "cls"

    def test_the_decoder_may_miss_exactly_the_declared_head_keys(self, decoder_dir: Path) -> None:
        model, info = _load(decoder_dir)
        plan = head_plan(model, "decoder_lm_head")
        assert plan.allowed_missing == {
            "classifier.weight",
            "head.dense.weight",
            "head.norm.weight",
        }
        check_load(model, info, "decoder_lm_head")

    def test_a_missing_body_tensor_refuses(self, tmp_path: Path, encoder_dir: Path) -> None:
        broken = _rewrite(encoder_dir, tmp_path / "broken", drop="model.final_norm.weight")
        model, info = _load(broken)
        with pytest.raises(HeadLoadError, match="missing beyond the declared head"):
            check_load(model, info, "new")

    def test_an_unexpected_tensor_refuses(self, tmp_path: Path, decoder_dir: Path) -> None:
        stray = _rewrite(decoder_dir, tmp_path / "stray", add="stray.weight")
        model, info = _load(stray)
        with pytest.raises(HeadLoadError, match="unexpected keys"):
            check_load(model, info, "decoder_lm_head")

    def test_new_refuses_a_checkpoint_that_already_has_a_head(
        self, tmp_path: Path, encoder_dir: Path
    ) -> None:
        model, _ = _load(encoder_dir)
        model.save_pretrained(str(tmp_path / "headed"))
        again, info = _load(tmp_path / "headed")
        with pytest.raises(HeadLoadError, match="head_init 'existing'"):
            check_load(again, info, "new")
        check_load(again, info, "existing")

    def test_a_label_count_mismatch_refuses(self, tmp_path: Path, encoder_dir: Path) -> None:
        model, _ = _load(encoder_dir)
        model.save_pretrained(str(tmp_path / "three"))
        again, info = transformers.AutoModelForSequenceClassification.from_pretrained(
            str(tmp_path / "three"),
            num_labels=2,
            output_loading_info=True,
            ignore_mismatched_sizes=True,
        )
        with pytest.raises(HeadLoadError, match="shape mismatches"):
            check_load(again, info, "existing")

    def test_an_undeclared_architecture_refuses(self) -> None:
        class _Config:
            model_type = "bert"

        class _Model:
            config = _Config()

        with pytest.raises(HeadLoadError, match="no declared head keys"):
            head_plan(_Model(), "new")
        with pytest.raises(HeadLoadError, match="no declared mapping"):
            head_plan(_Model(), "decoder_lm_head")


# ----------------------------------------------------------------------------- step 0


class TestStep0:
    @pytest.mark.parametrize("fixture", ["decoder_dir", "tied_decoder_dir"])
    def test_step0_matches_the_zero_shot_label_logits(
        self, tmp_path: Path, fixture: str, request: pytest.FixtureRequest
    ) -> None:
        checkpoint: Path = request.getfixturevalue(fixture)
        runner = SeqClsRunner(_decoder_config(tmp_path))
        tokenizer = runner.load_tokenizer(checkpoint)
        train, _ = runner.encode(_dataset(8), tokenizer, split="t")
        module, receipt, frozen = runner.load_model(checkpoint, tokenizer, roles=frozenset())
        result = runner.step0_check(module, checkpoint, train, receipt)
        assert result["passed"]
        assert result["max_abs_logit_diff"] < 1e-5
        assert result["argmax_agree"] == result["rows"] == 8
        assert result["precision"] == "float32, tf32 off"
        assert result["verbalizer_token_ids"] == {
            label: tokenizer.convert_tokens_to_ids(v.strip())
            for label, v in tiny.VERBALIZERS.items()
        }
        assert frozen == ["head.dense.bias"]
        assert "classifier.bias" in receipt.initialised_from

    def test_step0_copies_the_output_projection_bias(
        self, tmp_path: Path, decoder_dir: Path
    ) -> None:

        runner = SeqClsRunner(_decoder_config(tmp_path))
        tokenizer = runner.load_tokenizer(decoder_dir)
        module, receipt, _ = runner.load_model(decoder_dir, tokenizer, roles=frozenset())
        state = safetensors_torch.load_file(str(decoder_dir / "model.safetensors"))
        ids = [receipt.verbalizer_token_ids[label] for label in tiny.LABELS]
        classifier = module.classifier.classifier
        assert torch.equal(classifier.bias.detach(), state["decoder.bias"][ids])
        assert torch.equal(classifier.weight.detach(), state["decoder.weight"][ids])
        assert float(state["decoder.bias"][ids].abs().max()) > 0  # the bias is not trivially 0

    def test_step0_fails_without_the_bias(self, tmp_path: Path, decoder_dir: Path) -> None:
        runner = SeqClsRunner(_decoder_config(tmp_path))
        tokenizer = runner.load_tokenizer(decoder_dir)
        train, _ = runner.encode(_dataset(4), tokenizer, split="t")
        module, receipt, _ = runner.load_model(decoder_dir, tokenizer, roles=frozenset())
        with torch.no_grad():
            module.classifier.classifier.bias.zero_()
        with pytest.raises(Step0Error, match="step-0 check failed") as exc:
            runner.step0_check(module, decoder_dir, train, receipt)
        assert exc.value.receipt["passed"] is False

    def test_a_multi_token_verbalizer_refuses(self, tmp_path: Path, decoder_dir: Path) -> None:
        verbalizers = {**tiny.VERBALIZERS, "neutral": " w1 w2"}
        runner = SeqClsRunner(
            _config(tmp_path, head_init="decoder_lm_head", verbalizers=verbalizers)
        )
        tokenizer = runner.load_tokenizer(decoder_dir)
        with pytest.raises(PreflightError, match="exactly one"):
            runner.load_model(decoder_dir, tokenizer, roles=frozenset())


# ----------------------------------------------------------------------------- memory


class TestMemoryPreflight:
    def _prepared(self, tmp_path: Path, encoder_dir: Path, probe: _Probe) -> tuple[Any, Any, Any]:
        runner = SeqClsRunner(_config(tmp_path), memory_probe=probe)
        tokenizer = runner.load_tokenizer(encoder_dir)
        train, _ = runner.encode(_dataset(12), tokenizer, split="t")
        roles = frozenset(tiny.rows(1)[0])
        module, _, _ = runner.load_model(encoder_dir, tokenizer, roles=roles)
        trainer = runner.build_trainer(module, train, None, tokenizer)
        return runner, trainer, train

    def test_a_peak_over_budget_refuses_with_the_measurement(
        self, tmp_path: Path, encoder_dir: Path
    ) -> None:
        runner, trainer, train = self._prepared(tmp_path, encoder_dir, _Probe(950, 1000))
        with pytest.raises(PreflightError, match="exceeds 92% of the device") as exc:
            runner.memory_preflight(trainer, train)
        receipt = exc.value.receipt
        assert receipt["peak_bytes"] == 950
        assert receipt["budget_bytes"] == 920
        assert receipt["gradient_checkpointing"] is False

    def test_the_in_run_eval_is_measured_and_its_peak_counts(
        self, tmp_path: Path, encoder_dir: Path
    ) -> None:
        runner = SeqClsRunner(_config(tmp_path), memory_probe=_Probe([100, 950], 1000))
        tokenizer = runner.load_tokenizer(encoder_dir)
        train, _ = runner.encode(_dataset(12), tokenizer, split="t")
        evaluation, _ = runner.encode(_dataset(6, seed=3), tokenizer, split="e")
        module, _, _ = runner.load_model(encoder_dir, tokenizer, roles=frozenset(tiny.rows(1)[0]))
        trainer = runner.build_trainer(module, train, evaluation, tokenizer)
        with pytest.raises(PreflightError, match="eval peak") as exc:
            runner.memory_preflight(trainer, train, evaluation)
        receipt = exc.value.receipt
        assert receipt["peak_bytes"] == 100
        assert receipt["eval_peak_bytes"] == 950
        assert receipt["measured_peak_bytes"] == 950
        assert receipt["eval_batch_rows"] == 4

    def test_the_longest_batch_is_measured_and_the_step_is_undone(
        self, tmp_path: Path, encoder_dir: Path
    ) -> None:
        runner, trainer, train = self._prepared(tmp_path, encoder_dir, _Probe(100, 1000))
        before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
        receipt = runner.memory_preflight(trainer, train)
        assert receipt["batch_rows"] == 4
        assert receipt["batch_max_length"] == max(train["length"])
        assert trainer.optimizer is None
        for name, param in trainer.model.named_parameters():
            assert torch.equal(param.detach(), before[name]), name
            assert param.grad is None


# ----------------------------------------------------------------------------- whole runs


def _smoke(tmp_path: Path, checkpoint: Path, config: SeqClsConfig) -> tuple[Any, list[str], Path]:
    seen: list[str] = []
    artifact = tmp_path / "artifact"
    runner = SeqClsRunner(config, memory_probe=_Probe(10, 1000))
    result = runner.run(
        checkpoint=checkpoint,
        artifact_dir=artifact,
        train_dataset=_dataset(24),
        eval_dataset=_dataset(8, seed=1),
        on_receipt=lambda name, _receipt: seen.append(name),
    )
    return result, seen, artifact


class TestAcceptanceSmoke:
    """A zero-cost run through every check, end to end, for each kind of base."""

    def test_smoke_encoder(self, tmp_path: Path, encoder_dir: Path) -> None:
        result, seen, artifact = _smoke(tmp_path, encoder_dir, _config(tmp_path, save_steps=2))
        assert seen == ["overflow", "load", "memory"]
        assert result.steps == 3
        assert {"train_loss"} <= set(result.metrics)
        self._check_artifact(artifact, tmp_path, outputs=["primary", "flag", "support"])
        verifier = json.loads((artifact / VERIFIER_CONFIG_FILENAME).read_text())
        assert verifier["input"]["encoding"] == "pair"
        assert verifier["pooling"] == "first_token"

    def test_smoke_monitors_on_the_dev_split_and_keeps_the_final_checkpoint(
        self, tmp_path: Path, encoder_dir: Path
    ) -> None:
        progress = tmp_path / "progress.jsonl"
        result, _, _ = _smoke(
            tmp_path, encoder_dir, _config(tmp_path, progress_jsonl=str(progress))
        )
        monitoring = result.receipts["monitoring"]
        assert monitoring["label"] == "monitoring (ID dev, final checkpoint is the model)"
        assert monitoring["selects_checkpoint"] is False
        # 24 rows at 4 a batch: 6 steps an epoch, so every 2 steps, plus the final step (3).
        assert monitoring["cadence"]["eval_steps"] == 2
        assert [h["step"] for h in monitoring["history"]] == [2, 3]
        final = monitoring["final"]
        assert final["step"] == 3
        assert {"eval_loss", "monitor/3way/macro_f1", "monitor/3way/accuracy"} <= set(final)
        assert "monitor/2way/f1_entailed" in final
        events = [json.loads(line) for line in progress.read_text().splitlines()]
        evals = [e for e in events if e["kind"] == "eval"]
        assert len(evals) == 2
        assert all(e["message"] == monitoring["label"] for e in evals)
        assert all("monitor/3way/macro_f1" in e["metrics"] for e in evals)
        assert not any(k.startswith("eval_monitor") for e in evals for k in e["metrics"])

    def test_smoke_decoder_with_activation_checkpointing(
        self, tmp_path: Path, decoder_dir: Path
    ) -> None:
        config = _decoder_config(tmp_path, gradient_checkpointing=True)
        result, seen, artifact = _smoke(tmp_path, decoder_dir, config)
        assert seen == ["overflow", "load", "step0", "memory"]
        assert result.receipts["step0"]["passed"]
        parity = result.receipts["reload_parity"]
        assert parity["passed"]
        assert parity["rows_from"] == "eval"
        assert parity["max_abs_logit_diff"] <= 1e-4
        assert parity["zero_parameters"] == ["head.dense.bias"]
        assert result.receipts["load"]["classifier_bias"]["value"] is True
        assert "output projection" in result.receipts["load"]["classifier_bias"]["reason"]
        assert result.receipts["loss"]["brier_weight"] == 0.1
        assert "sum form" in result.receipts["loss"]["form"]
        assert result.receipts["memory"]["gradient_checkpointing"] is True
        verifier = self._check_artifact(artifact, tmp_path, outputs=["primary", "flag", "support"])
        assert verifier["pooling"] == "last_token"
        assert verifier["frozen_parameters"] == ["head.dense.bias"]
        assert verifier["verbalizer_token_ids"] == result.receipts["step0"]["verbalizer_token_ids"]
        # The frozen bias is still exactly zero after training.

        state = safetensors_torch.load_file(str(artifact / "model.safetensors"))
        assert float(state["head.dense.bias"].abs().max()) == 0.0

    def test_smoke_continues_an_existing_classifier(
        self, tmp_path: Path, decoder_dir: Path
    ) -> None:
        _, _, artifact = _smoke(
            tmp_path / "first", decoder_dir, _decoder_config(tmp_path / "first")
        )
        config = _config(tmp_path / "second", head_init="existing")
        result, seen, _ = _smoke(tmp_path / "second", artifact, config)
        assert seen == ["overflow", "load", "memory"]
        assert result.receipts["load"]["missing"] == []
        assert result.receipts["load"]["frozen_parameters"] == ["head.dense.bias"]

    def test_smoke_calibration_is_fitted_on_the_saved_artifact(
        self, tmp_path: Path, encoder_dir: Path
    ) -> None:
        pytest.importorskip("strata_forge.metrics.calibration_fit")
        artifact = tmp_path / "artifact"
        result = SeqClsRunner(_config(tmp_path), memory_probe=_Probe(10, 1000)).run(
            checkpoint=encoder_dir,
            artifact_dir=artifact,
            train_dataset=_dataset(24),
            calibration_dataset=_dataset(12, seed=2),
            calibration_source={"dataset_id": "synthetic/rows", "split": "dev", "commit": "c" * 40},
        )
        record = json.loads((artifact / CALIBRATION_FILENAME).read_text())
        assert record == result.receipts["calibration"]
        assert record["split"]["commit"] == "c" * 40
        # One contract with the evaluator: the temperature and the labels it applies to.
        assert record["schema"] == "strata.calibration/v1"
        assert record["labels"] == list(tiny.LABELS)
        assert record["fit"]["temperature"] > 0
        assert record["fit"]["nll_after"] <= record["fit"]["nll_before"] + 1e-12
        # Two-way rows have no three-way gold and are excluded, by count.
        assert record["rows"] + record["excluded_rows"] == 12
        assert record["excluded_rows"] == 3
        assert len(record["rows_sha256"]) == 64
        assert result.receipts["overflow_calibration"]["rows"] == 12

    def test_a_saved_artifact_that_reloads_differently_refuses(
        self, tmp_path: Path, decoder_dir: Path
    ) -> None:
        runner = SeqClsRunner(_decoder_config(tmp_path), memory_probe=_Probe(10, 1000))
        prepared = runner.prepare(checkpoint=decoder_dir, train_dataset=_dataset(16))
        artifact = tmp_path / "artifact"
        runner.finish(prepared, artifact_dir=artifact)
        path = artifact / "model.safetensors"
        state = safetensors_torch.load_file(str(path))
        state["classifier.weight"] = state["classifier.weight"] + 0.5
        safetensors_torch.save_file(state, str(path), metadata={"format": "pt"})
        with pytest.raises(ArtifactCheckError, match="differ from the trained") as exc:
            runner.artifact_checks(prepared, artifact)
        assert exc.value.receipt["passed"] is False
        state["head.dense.bias"] = state["head.dense.bias"] + 1e-3
        safetensors_torch.save_file(state, str(path), metadata={"format": "pt"})
        with pytest.raises(ArtifactCheckError, match="exactly zero after reload"):
            runner.artifact_checks(prepared, artifact)

    def test_existing_refuses_a_different_label_order(
        self, tmp_path: Path, encoder_dir: Path
    ) -> None:
        _, _, artifact = _smoke(tmp_path / "first", encoder_dir, _config(tmp_path / "first"))
        reordered = ("neutral", "entailment", "contradiction")
        config = _config(tmp_path / "second", head_init="existing", labels=reordered)
        with pytest.raises(PreflightError, match="label order"):
            _smoke(tmp_path / "second", artifact, config)

    def test_existing_decoder_keeps_its_template_and_last_token_pooling(
        self, tmp_path: Path, decoder_dir: Path
    ) -> None:
        _, _, artifact = _smoke(
            tmp_path / "first", decoder_dir, _decoder_config(tmp_path / "first")
        )
        runner = SeqClsRunner(_config(tmp_path / "second", head_init="existing"))
        runner.load_tokenizer(artifact)
        assert runner.decoder is True
        result, _, second = _smoke(
            tmp_path / "second", artifact, _config(tmp_path / "second", head_init="existing")
        )
        verifier = json.loads((second / VERIFIER_CONFIG_FILENAME).read_text())
        assert verifier["pooling"] == "last_token"
        assert verifier["input"]["encoding"] == "template"
        assert result.receipts["overflow"]["template_token_overhead"] == 2

    def test_existing_refuses_a_changed_rendering(self, tmp_path: Path, decoder_dir: Path) -> None:
        _, _, artifact = _smoke(
            tmp_path / "first", decoder_dir, _decoder_config(tmp_path / "first")
        )
        config = _config(
            tmp_path / "second", head_init="existing", decoder_template="{hypothesis}\n{premises}"
        )
        with pytest.raises(PreflightError, match="different decoder_template"):
            SeqClsRunner(config).load_tokenizer(artifact)

    def test_existing_decoder_holds_the_continued_zero_bias_at_zero(
        self, tmp_path: Path, decoder_dir: Path
    ) -> None:
        _, _, artifact = _smoke(
            tmp_path / "first", decoder_dir, _decoder_config(tmp_path / "first")
        )
        verifier = json.loads((artifact / VERIFIER_CONFIG_FILENAME).read_text())
        assert verifier["zero_parameters"] == ["head.dense.bias"]
        result, _, second = _smoke(
            tmp_path / "second", artifact, _config(tmp_path / "second", head_init="existing")
        )
        load = result.receipts["load"]
        assert load["zero_parameters"] == ["head.dense.bias"]
        assert load["classifier_bias"]["value"] is True
        assert "continued from a checkpoint" in load["classifier_bias"]["reason"]
        assert result.receipts["reload_parity"]["zero_parameters"] == ["head.dense.bias"]
        reloaded = safetensors_torch.load_file(str(second / "model.safetensors"))
        assert bool((reloaded["head.dense.bias"] == 0).all())

    @pytest.mark.parametrize("listed", [True, False])
    def test_existing_refuses_a_continued_checkpoint_whose_zero_bias_moved(
        self, tmp_path: Path, decoder_dir: Path, listed: bool
    ) -> None:
        _, _, artifact = _smoke(
            tmp_path / "first", decoder_dir, _decoder_config(tmp_path / "first")
        )
        path = artifact / "model.safetensors"
        state = safetensors_torch.load_file(str(path))
        state["head.dense.bias"] = state["head.dense.bias"] + 1e-3
        safetensors_torch.save_file(state, str(path), metadata={"format": "pt"})
        if not listed:
            # A checkpoint that predates the field: the decoder config's classifier_bias decides.
            verifier_path = artifact / VERIFIER_CONFIG_FILENAME
            verifier = json.loads(verifier_path.read_text())
            del verifier["zero_parameters"]
            verifier_path.write_text(json.dumps(verifier))
        config = _config(tmp_path / "second", head_init="existing")
        with pytest.raises(PreflightError, match=r"head\.dense\.bias must be exactly zero"):
            _smoke(tmp_path / "second", artifact, config)

    def test_a_reload_failure_names_no_path(self, tmp_path: Path, encoder_dir: Path) -> None:
        runner = SeqClsRunner(_config(tmp_path), memory_probe=_Probe(10, 1000))
        prepared = runner.prepare(checkpoint=encoder_dir, train_dataset=_dataset(16))
        artifact = tmp_path / "artifact"
        runner.finish(prepared, artifact_dir=artifact)
        (artifact / "config.json").write_text("{not json")
        with pytest.raises(ArtifactCheckError, match="does not reload cleanly") as exc:
            runner.artifact_checks(prepared, artifact)
        assert str(tmp_path) not in str(exc.value)

    def test_a_refused_temperature_fit_is_an_artifact_refusal(
        self, tmp_path: Path, encoder_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import types

        fake = types.ModuleType("strata_forge.metrics.calibration_fit")

        def _refuse(*_a: Any, **_k: Any) -> Any:
            msg = "a fit input the validator refuses"
            raise ValueError(msg)

        fake.fit_temperature = _refuse  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "strata_forge.metrics.calibration_fit", fake)
        runner = SeqClsRunner(_config(tmp_path), memory_probe=_Probe(10, 1000))
        with pytest.raises(ArtifactCheckError, match="temperature fit refused") as exc:
            runner.run(
                checkpoint=encoder_dir,
                artifact_dir=tmp_path / "artifact",
                train_dataset=_dataset(16),
                calibration_dataset=_dataset(12, seed=2),
            )
        assert exc.value.check == "artifact"

    def test_existing_refuses_missing_aux_heads(self, tmp_path: Path, encoder_dir: Path) -> None:
        _, _, artifact = _smoke(tmp_path / "first", encoder_dir, _config(tmp_path / "first"))
        (artifact / AUX_HEADS_FILENAME).unlink()
        config = _config(tmp_path / "second", head_init="existing")
        with pytest.raises(PreflightError, match=AUX_HEADS_FILENAME):
            _smoke(tmp_path / "second", artifact, config)

    def test_smoke_bf16_autocast_keeps_float32_master_weights(
        self, tmp_path: Path, encoder_dir: Path
    ) -> None:
        # bf16 autocast on CPU stands in for the GPU path; the weights stay float32.
        result, _, artifact = _smoke(tmp_path, encoder_dir, _config(tmp_path, precision="bf16"))
        assert result.steps == 3
        model = transformers.AutoModelForSequenceClassification.from_pretrained(str(artifact))
        assert {p.dtype for p in model.parameters()} == {torch.float32}

    @staticmethod
    def _check_artifact(artifact: Path, tmp_path: Path, *, outputs: list[str]) -> dict[str, Any]:
        # A stock checkpoint: reloads with every key present and nothing re-initialised.
        model, info = _load(artifact)
        assert not info["missing_keys"]
        assert not info["unexpected_keys"]
        assert model.config.id2label == dict(enumerate(tiny.LABELS))
        assert (artifact / AUX_HEADS_FILENAME).is_file()
        assert not list(artifact.glob("checkpoint-*"))
        assert not (artifact / "training_args.bin").exists()
        verifier: dict[str, Any] = json.loads((artifact / VERIFIER_CONFIG_FILENAME).read_text())
        assert verifier["labels"] == list(tiny.LABELS)
        assert verifier["outputs"] == outputs
        assert verifier["input"]["overflow"] == "error"
        del tmp_path
        return verifier
