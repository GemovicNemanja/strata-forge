"""Unit tests for `forge.training.preference`."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from forge.training.peft import LoRAConfig
from forge.training.preference import (
    DPOConfig,
    GRPOConfig,
    KTOConfig,
    ORPOConfig,
    PreferenceRunner,
    PreferenceRunResult,
)


class TestConfigs:
    def test_dpo_defaults(self) -> None:
        cfg = DPOConfig(model_id="gpt2", output_dir="./out")
        assert cfg.method == "dpo"
        assert cfg.beta == 0.1
        assert cfg.loss_type == "sigmoid"

    def test_dpo_trl_kwargs(self) -> None:
        cfg = DPOConfig(model_id="gpt2", output_dir="./out", beta=0.2)
        kwargs = cfg.to_trl_kwargs()
        assert kwargs["beta"] == 0.2
        assert kwargs["max_length"] == 2048

    def test_orpo_defaults_and_kwargs(self) -> None:
        cfg = ORPOConfig(model_id="gpt2", output_dir="./out")
        assert cfg.method == "orpo"
        assert "beta" in cfg.to_trl_kwargs()

    def test_kto_extra_weights(self) -> None:
        cfg = KTOConfig(
            model_id="gpt2",
            output_dir="./out",
            desirable_weight=2.0,
            undesirable_weight=0.5,
        )
        kwargs = cfg.to_trl_kwargs()
        assert kwargs["desirable_weight"] == 2.0
        assert kwargs["undesirable_weight"] == 0.5

    def test_grpo_num_generations(self) -> None:
        cfg = GRPOConfig(model_id="gpt2", output_dir="./out", num_generations=16)
        kwargs = cfg.to_trl_kwargs()
        assert kwargs["num_generations"] == 16

    def test_grpo_rejects_low_num_generations(self) -> None:
        with pytest.raises(ValueError, match="num_generations"):
            GRPOConfig(model_id="gpt2", output_dir="./out", num_generations=1)

    def test_extra_args_override(self) -> None:
        cfg = DPOConfig(
            model_id="gpt2",
            output_dir="./out",
            extra_trainer_args={"learning_rate": 1e-3, "report_to": "tensorboard"},
        )
        kwargs = cfg.to_trl_kwargs()
        assert kwargs["learning_rate"] == 1e-3
        assert kwargs["report_to"] == "tensorboard"


@pytest.fixture
def fake_ml_stack(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "model_loaded": None,
        "tokenizer_loaded": None,
        "trl_config_kwargs": None,
        "trainer_kwargs": None,
        "trainer_class": None,
        "trained": False,
        "saved_to": None,
    }

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "BFLOAT16"  # type: ignore[attr-defined]
    fake_torch.float16 = "FLOAT16"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

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
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    class _FakeLoraConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = _FakeLoraConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    class _Recorder:
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, **kwargs: Any) -> Any:
            state["trl_config_kwargs"] = kwargs
            cfg_obj = MagicMock(name=self.name)
            cfg_obj.kwargs = kwargs
            return cfg_obj

    class _FakeTrainOutput:
        def __init__(self) -> None:
            self.metrics = {"train_loss": 0.33, "train_runtime": 90.0}

    def _make_trainer_cls(name: str) -> type:
        class _FakeTrainer:
            def __init__(self, **kwargs: Any) -> None:
                state["trainer_kwargs"] = kwargs
                state["trainer_class"] = name

            def train(self) -> _FakeTrainOutput:
                state["trained"] = True
                return _FakeTrainOutput()

            def save_model(self, output_dir: str) -> None:
                state["saved_to"] = output_dir

        _FakeTrainer.__name__ = name
        return _FakeTrainer

    fake_trl = types.ModuleType("trl")
    fake_trl.DPOConfig = _Recorder("DPOConfig")  # type: ignore[attr-defined]
    fake_trl.ORPOConfig = _Recorder("ORPOConfig")  # type: ignore[attr-defined]
    fake_trl.KTOConfig = _Recorder("KTOConfig")  # type: ignore[attr-defined]
    fake_trl.GRPOConfig = _Recorder("GRPOConfig")  # type: ignore[attr-defined]
    fake_trl.DPOTrainer = _make_trainer_cls("DPOTrainer")  # type: ignore[attr-defined]
    fake_trl.ORPOTrainer = _make_trainer_cls("ORPOTrainer")  # type: ignore[attr-defined]
    fake_trl.KTOTrainer = _make_trainer_cls("KTOTrainer")  # type: ignore[attr-defined]
    fake_trl.GRPOTrainer = _make_trainer_cls("GRPOTrainer")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    return state


class TestPreferenceRunner:
    def test_dpo_dispatch(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = DPOConfig(model_id="gpt2", output_dir="./out")
        runner = PreferenceRunner(cfg)
        result = runner.train(train_dataset=["row"])
        assert isinstance(result, PreferenceRunResult)
        assert result.method == "dpo"
        assert fake_ml_stack["trainer_class"] == "DPOTrainer"
        assert result.train_loss == 0.33

    def test_orpo_dispatch(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = ORPOConfig(model_id="gpt2", output_dir="./out")
        PreferenceRunner(cfg).train(train_dataset=["row"])
        assert fake_ml_stack["trainer_class"] == "ORPOTrainer"

    def test_kto_dispatch(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = KTOConfig(model_id="gpt2", output_dir="./out")
        PreferenceRunner(cfg).train(train_dataset=["row"])
        assert fake_ml_stack["trainer_class"] == "KTOTrainer"

    def test_grpo_requires_reward_funcs(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = GRPOConfig(model_id="gpt2", output_dir="./out")
        runner = PreferenceRunner(cfg)
        with pytest.raises(ValueError, match="reward_funcs"):
            runner.train(train_dataset=["row"])

    def test_grpo_with_reward_funcs(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = GRPOConfig(model_id="gpt2", output_dir="./out")
        runner = PreferenceRunner(cfg)

        def _reward(_completions: Any) -> list[float]:
            return [0.5]

        runner.train(train_dataset=["row"], reward_funcs=_reward)
        assert fake_ml_stack["trainer_class"] == "GRPOTrainer"
        assert fake_ml_stack["trainer_kwargs"]["reward_funcs"] is _reward

    def test_ref_model_forwarded_for_dpo(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = DPOConfig(model_id="gpt2", output_dir="./out")
        runner = PreferenceRunner(cfg)
        ref = MagicMock(name="ref-model")
        runner.train(train_dataset=["row"], ref_model=ref)
        assert fake_ml_stack["trainer_kwargs"]["ref_model"] is ref

    def test_ref_model_not_forwarded_for_orpo(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = ORPOConfig(model_id="gpt2", output_dir="./out")
        runner = PreferenceRunner(cfg)
        ref = MagicMock(name="unused-ref")
        runner.train(train_dataset=["row"], ref_model=ref)
        # ORPO doesn't accept ref_model; runner correctly drops it.
        assert "ref_model" not in fake_ml_stack["trainer_kwargs"]

    def test_peft_config_forwarded(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = DPOConfig(model_id="gpt2", output_dir="./out")
        runner = PreferenceRunner(cfg, peft_config=LoRAConfig())
        runner.train(train_dataset=["row"])
        assert "peft_config" in fake_ml_stack["trainer_kwargs"]

    def test_missing_extra_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "transformers", None)
        monkeypatch.setitem(sys.modules, "trl", None)
        cfg = DPOConfig(model_id="gpt2", output_dir="./out")
        with pytest.raises(ImportError, match=r"\[finetuning\] extra"):
            PreferenceRunner(cfg).train(train_dataset=["row"])

    def test_eval_dataset_forwarded(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = DPOConfig(model_id="gpt2", output_dir="./out")
        runner = PreferenceRunner(cfg)
        runner.train(train_dataset=["row"], eval_dataset=["e"])
        assert fake_ml_stack["trainer_kwargs"]["eval_dataset"] == ["e"]

    def test_runner_properties(self) -> None:
        cfg = DPOConfig(model_id="gpt2", output_dir="./out")
        peft = LoRAConfig()
        runner = PreferenceRunner(cfg, peft_config=peft)
        assert runner.config is cfg
        assert runner.peft_config is peft
