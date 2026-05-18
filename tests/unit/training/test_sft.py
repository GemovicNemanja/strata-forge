"""Unit tests for `forge.training.sft`."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from forge.training.peft import LoRAConfig, QLoRAConfig
from forge.training.sft import SFTConfig, SFTRunner, SFTRunResult


class TestSFTConfig:
    def test_defaults(self) -> None:
        cfg = SFTConfig(model_id="gpt2", output_dir="./out")
        assert cfg.max_seq_length == 2048
        assert cfg.num_epochs == 1.0
        assert cfg.precision == "bf16"
        assert cfg.dataset_text_field == "text"

    def test_validates_batch_size(self) -> None:
        with pytest.raises(ValueError, match="per_device_batch_size"):
            SFTConfig(model_id="gpt2", output_dir="./out", per_device_batch_size=0)

    def test_extra_args_override_forge_defaults(self) -> None:
        cfg = SFTConfig(
            model_id="gpt2",
            output_dir="./out",
            extra_trainer_args={"learning_rate": 1e-3, "report_to": "wandb"},
        )
        trl = cfg.to_trl_kwargs()
        assert trl["learning_rate"] == 1e-3
        assert trl["report_to"] == "wandb"

    def test_trl_kwargs_precision_bf16(self) -> None:
        kwargs = SFTConfig(model_id="gpt2", output_dir="./out", precision="bf16").to_trl_kwargs()
        assert kwargs["bf16"] is True
        assert "fp16" not in kwargs

    def test_trl_kwargs_precision_fp16(self) -> None:
        kwargs = SFTConfig(model_id="gpt2", output_dir="./out", precision="fp16").to_trl_kwargs()
        assert kwargs["fp16"] is True
        assert "bf16" not in kwargs

    def test_trl_kwargs_precision_fp32(self) -> None:
        kwargs = SFTConfig(model_id="gpt2", output_dir="./out", precision="fp32").to_trl_kwargs()
        assert "bf16" not in kwargs
        assert "fp16" not in kwargs

    def test_save_strategy_no_when_save_steps_zero(self) -> None:
        kwargs = SFTConfig(model_id="gpt2", output_dir="./out", save_steps=0).to_trl_kwargs()
        assert kwargs["save_strategy"] == "no"
        assert "save_steps" not in kwargs

    def test_save_strategy_steps_when_set(self) -> None:
        kwargs = SFTConfig(model_id="gpt2", output_dir="./out", save_steps=100).to_trl_kwargs()
        assert kwargs["save_strategy"] == "steps"
        assert kwargs["save_steps"] == 100

    def test_dataset_text_field_skipped_when_none(self) -> None:
        kwargs = SFTConfig(
            model_id="gpt2", output_dir="./out", dataset_text_field=None
        ).to_trl_kwargs()
        assert "dataset_text_field" not in kwargs

    def test_packing_forwarded(self) -> None:
        kwargs = SFTConfig(model_id="gpt2", output_dir="./out", packing=True).to_trl_kwargs()
        assert kwargs["packing"] is True


# ---------------------------------------------------------------------------
# SFTRunner — uses sys.modules fake transformers + trl
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_ml_stack(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject fake transformers / trl / datasets / peft."""
    state: dict[str, Any] = {
        "model_loaded": None,
        "tokenizer_loaded": None,
        "trl_config_kwargs": None,
        "trainer_kwargs": None,
        "trained": False,
        "saved_to": None,
        "bnb_kwargs": None,
    }

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "BFLOAT16"  # type: ignore[attr-defined]
    fake_torch.float16 = "FLOAT16"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class _FakeBnbConfig:
        def __init__(self, **kwargs: Any) -> None:
            state["bnb_kwargs"] = kwargs

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> Any:
            state["model_loaded"] = {"model_id": model_id, "kwargs": kwargs}
            return MagicMock(name=f"model({model_id})")

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> Any:
            del kwargs
            state["tokenizer_loaded"] = model_id
            return MagicMock(name=f"tokenizer({model_id})")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = _FakeAutoModel  # type: ignore[attr-defined]
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer  # type: ignore[attr-defined]
    fake_transformers.BitsAndBytesConfig = _FakeBnbConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    class _FakeLoraConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = _FakeLoraConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    class _FakeTrlConfig:
        def __init__(self, **kwargs: Any) -> None:
            state["trl_config_kwargs"] = kwargs
            self.kwargs = kwargs

    class _FakeTrainOutput:
        def __init__(self) -> None:
            self.metrics = {
                "train_loss": 0.42,
                "train_runtime": 120.0,
                "train_samples_per_second": 2.5,
            }

    class _FakeSFTTrainer:
        def __init__(self, **kwargs: Any) -> None:
            state["trainer_kwargs"] = kwargs

        def train(self) -> _FakeTrainOutput:
            state["trained"] = True
            return _FakeTrainOutput()

        def save_model(self, output_dir: str) -> None:
            state["saved_to"] = output_dir

    fake_trl = types.ModuleType("trl")
    fake_trl.SFTConfig = _FakeTrlConfig  # type: ignore[attr-defined]
    fake_trl.SFTTrainer = _FakeSFTTrainer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    fake_datasets = types.ModuleType("datasets")
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    return state


class TestSFTRunner:
    def test_construction_holds_config(self) -> None:
        cfg = SFTConfig(model_id="gpt2", output_dir="./out")
        runner = SFTRunner(cfg)
        assert runner.config is cfg
        assert runner.peft_config is None

    def test_construction_with_peft(self) -> None:
        cfg = SFTConfig(model_id="gpt2", output_dir="./out")
        peft = LoRAConfig()
        runner = SFTRunner(cfg, peft_config=peft)
        assert runner.peft_config is peft

    def test_train_missing_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "transformers", None)
        monkeypatch.setitem(sys.modules, "trl", None)
        cfg = SFTConfig(model_id="gpt2", output_dir="./out")
        runner = SFTRunner(cfg)
        with pytest.raises(ImportError, match=r"\[finetuning\] extra"):
            runner.train(train_dataset=[])

    def test_train_happy_path(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = SFTConfig(model_id="gpt2", output_dir="./out", num_epochs=2.0)
        runner = SFTRunner(cfg)
        result = runner.train(train_dataset=["row1", "row2"])

        assert isinstance(result, SFTRunResult)
        assert result.output_dir == "./out"
        assert result.train_loss == 0.42
        assert result.train_runtime_s == 120.0
        assert result.train_samples_per_second == 2.5
        assert fake_ml_stack["trained"] is True
        assert fake_ml_stack["saved_to"] == "./out"
        assert fake_ml_stack["model_loaded"]["model_id"] == "gpt2"
        assert fake_ml_stack["tokenizer_loaded"] == "gpt2"
        assert fake_ml_stack["trl_config_kwargs"]["num_train_epochs"] == 2.0

    def test_train_with_qlora_passes_bnb_config(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = SFTConfig(model_id="gpt2", output_dir="./out")
        runner = SFTRunner(cfg, peft_config=QLoRAConfig())
        runner.train(train_dataset=["row1"])
        # The bnb config got built and was forwarded to from_pretrained.
        assert "quantization_config" in fake_ml_stack["model_loaded"]["kwargs"]
        # peft_config also flowed into the trainer.
        assert "peft_config" in fake_ml_stack["trainer_kwargs"]

    def test_train_with_lora_passes_peft_config(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = SFTConfig(model_id="gpt2", output_dir="./out")
        runner = SFTRunner(cfg, peft_config=LoRAConfig(r=32))
        runner.train(train_dataset=["row1"])
        assert "peft_config" in fake_ml_stack["trainer_kwargs"]
        # plain LoRA: no quantization_config.
        assert "quantization_config" not in fake_ml_stack["model_loaded"]["kwargs"]

    def test_train_uses_provided_model_tokenizer(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = SFTConfig(model_id="gpt2", output_dir="./out")
        runner = SFTRunner(cfg)
        model_sentinel = MagicMock(name="explicit-model")
        tokenizer_sentinel = MagicMock(name="explicit-tokenizer")
        runner.train(
            train_dataset=["row"],
            model=model_sentinel,
            tokenizer=tokenizer_sentinel,
        )
        # Forge skipped from_pretrained when both were provided.
        assert fake_ml_stack["model_loaded"] is None
        assert fake_ml_stack["tokenizer_loaded"] is None
        assert fake_ml_stack["trainer_kwargs"]["model"] is model_sentinel
        assert fake_ml_stack["trainer_kwargs"]["processing_class"] is tokenizer_sentinel

    def test_train_eval_dataset_forwarded(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = SFTConfig(model_id="gpt2", output_dir="./out")
        runner = SFTRunner(cfg)
        runner.train(train_dataset=["row"], eval_dataset=["eval_row"])
        assert fake_ml_stack["trainer_kwargs"]["eval_dataset"] == ["eval_row"]

    def test_train_metrics_non_numeric_skipped(
        self, monkeypatch: pytest.MonkeyPatch, fake_ml_stack: dict[str, Any]
    ) -> None:
        # Swap trainer to one whose train output has a non-numeric metric.
        class _BadOutput:
            def __init__(self) -> None:
                self.metrics = {"train_loss": 0.5, "note": "ignored"}

        class _BadTrainer:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def train(self) -> _BadOutput:
                return _BadOutput()

            def save_model(self, _output_dir: str) -> None:
                pass

        trl_mod: Any = sys.modules["trl"]
        monkeypatch.setattr(trl_mod, "SFTTrainer", _BadTrainer, raising=False)
        cfg = SFTConfig(model_id="gpt2", output_dir="./out")
        runner = SFTRunner(cfg)
        result = runner.train(train_dataset=["row"])
        assert result.train_loss == 0.5
        assert "note" not in result.metrics
