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
    VERIFIER_CONFIG_FILENAME,
    PreflightError,
    SeqClsConfig,
    SeqClsRunner,
    Step0Error,
)

torch: Any = pytest.importorskip("torch")
transformers: Any = pytest.importorskip("transformers")
datasets: Any = pytest.importorskip("datasets")

from tests.unit.training import tiny_classifier as tiny  # noqa: E402 - after the import skips

if TYPE_CHECKING:
    from pathlib import Path

_ALIASES = {"not_entailment": "not_entailed"}


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
        "label_aliases": _ALIASES,
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
    """A memory probe with a fixed reading, standing in for a GPU allocator."""

    def __init__(self, peak: int, total: int) -> None:
        self._peak, self._total = peak, total

    def reset(self) -> None:
        return None

    def peak_bytes(self) -> int | None:
        return self._peak

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
                assert enc["two_way_target"] == (1.0 if row["label"] == "entailment" else 0.0)
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
        rows = [{"premises": "w1 w2", "hypothesis": "w3", "label": 2}]
        runner = SeqClsRunner(_config(tmp_path))
        tokenizer = runner.load_tokenizer(encoder_dir)
        with pytest.raises(PreflightError, match="explicit index map"):
            runner.encode(datasets.Dataset.from_list(rows), tokenizer, split="t")
        mapped = SeqClsRunner(_config(tmp_path, label_index_map={"2": "neutral"}))
        encoded, _ = mapped.encode(datasets.Dataset.from_list(rows), tokenizer, split="t")
        assert encoded[0]["target"] == [0.0, 0.0, 1.0]

    def test_an_unknown_label_is_named_by_row(self, tmp_path: Path, encoder_dir: Path) -> None:
        rows = [{"premises": "w1", "hypothesis": "w3", "label": "maybe"}]
        runner = SeqClsRunner(_config(tmp_path))
        with pytest.raises(PreflightError, match="row 0: label is neither"):
            runner.encode(
                datasets.Dataset.from_list(rows), runner.load_tokenizer(encoder_dir), split="t"
            )

    def test_a_soft_label_distribution(self, tmp_path: Path, encoder_dir: Path) -> None:
        rows = [
            {
                "premises": "w1",
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

    from safetensors.torch import load_file, save_file

    shutil.copytree(source, target)
    state = load_file(str(target / "model.safetensors"))
    if drop:
        del state[drop]
    if add:
        state[add] = torch.zeros(2)
    save_file(state, str(target / "model.safetensors"), metadata={"format": "pt"})
    return target


class TestLoadGuard:
    def test_the_encoder_may_miss_only_its_classifier(self, encoder_dir: Path) -> None:
        model, info = _load(encoder_dir)
        receipt = check_load(model, info, "new")
        assert receipt.missing == ["classifier.bias", "classifier.weight"]

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
        assert result["verbalizer_token_ids"] == {
            label: tokenizer.convert_tokens_to_ids(v.strip())
            for label, v in tiny.VERBALIZERS.items()
        }
        assert frozen == ["head.dense.bias"]
        assert "classifier.bias" in receipt.initialised_from

    def test_step0_copies_the_output_projection_bias(
        self, tmp_path: Path, decoder_dir: Path
    ) -> None:
        from safetensors.torch import load_file

        runner = SeqClsRunner(_decoder_config(tmp_path))
        tokenizer = runner.load_tokenizer(decoder_dir)
        module, receipt, _ = runner.load_model(decoder_dir, tokenizer, roles=frozenset())
        state = load_file(str(decoder_dir / "model.safetensors"))
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

    def test_smoke_decoder_with_activation_checkpointing(
        self, tmp_path: Path, decoder_dir: Path
    ) -> None:
        config = _decoder_config(tmp_path, gradient_checkpointing=True)
        result, seen, artifact = _smoke(tmp_path, decoder_dir, config)
        assert seen == ["overflow", "load", "step0", "memory"]
        assert result.receipts["step0"]["passed"]
        assert result.receipts["memory"]["gradient_checkpointing"] is True
        verifier = self._check_artifact(artifact, tmp_path, outputs=["primary", "flag", "support"])
        assert verifier["pooling"] == "last_token"
        assert verifier["frozen_parameters"] == ["head.dense.bias"]
        assert verifier["verbalizer_token_ids"] == result.receipts["step0"]["verbalizer_token_ids"]
        # The frozen bias is still exactly zero after training.
        from safetensors.torch import load_file

        state = load_file(str(artifact / "model.safetensors"))
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

    def test_existing_refuses_a_different_label_order(
        self, tmp_path: Path, encoder_dir: Path
    ) -> None:
        _, _, artifact = _smoke(tmp_path / "first", encoder_dir, _config(tmp_path / "first"))
        reordered = ("neutral", "entailment", "contradiction")
        config = _config(tmp_path / "second", head_init="existing", labels=reordered)
        with pytest.raises(PreflightError, match="label order"):
            _smoke(tmp_path / "second", artifact, config)

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
