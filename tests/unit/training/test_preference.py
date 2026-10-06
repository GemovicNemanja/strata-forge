"""Unit tests for `strata_forge.training.preference`."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from strata_forge.training.peft import LoRAConfig, MissingBitsAndBytesError, QLoRAConfig
from strata_forge.training.preference import (
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
        "bnb_kwargs": None,
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
            state.setdefault("model_loads", []).append({"model_id": model_id, "kwargs": kwargs})
            return MagicMock(name=f"model({model_id})")

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> Any:
            state["tokenizer_loaded"] = model_id
            state["tokenizer_kwargs"] = kwargs
            return MagicMock(name=f"tokenizer({model_id})")

    class _FakeBnbConfig:
        def __init__(self, **kwargs: Any) -> None:
            state["bnb_kwargs"] = kwargs

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

    @pytest.mark.usefixtures("bitsandbytes_installed")
    def test_qlora_passes_bnb_config(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = DPOConfig(model_id="gpt2", output_dir="./out")
        PreferenceRunner(cfg, peft_config=QLoRAConfig()).train(train_dataset=["row"])
        assert fake_ml_stack["bnb_kwargs"]["load_in_4bit"] is True
        assert "quantization_config" in fake_ml_stack["model_loaded"]["kwargs"]

    @pytest.mark.usefixtures("bitsandbytes_missing")
    def test_qlora_without_bitsandbytes_stops_before_the_model_load(
        self, fake_ml_stack: dict[str, Any]
    ) -> None:
        cfg = DPOConfig(model_id="gpt2", output_dir="./out")
        runner = PreferenceRunner(cfg, peft_config=QLoRAConfig())
        with pytest.raises(MissingBitsAndBytesError):
            runner.train(train_dataset=["row"])
        assert fake_ml_stack["model_loaded"] is None
        assert fake_ml_stack["bnb_kwargs"] is None

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


_TOKEN = "hf_secretreadtoken1234567890"
_LOAD_TERMS = {"token": _TOKEN, "trust_remote_code": False, "use_safetensors": True}


class TestHubLoads:
    """Every load states its terms: the caller's token, no remote code, safetensors only."""

    def test_the_token_reaches_every_load_and_never_the_trl_config(
        self, fake_ml_stack: dict[str, Any]
    ) -> None:
        cfg = DPOConfig(model_id="org/gated", output_dir="./out")
        PreferenceRunner(cfg, peft_config=LoRAConfig()).train(train_dataset=["row"], token=_TOKEN)
        assert fake_ml_stack["model_loads"] == [{"model_id": "org/gated", "kwargs": _LOAD_TERMS}]
        assert fake_ml_stack["tokenizer_kwargs"] == {"token": _TOKEN, "trust_remote_code": False}
        assert _TOKEN not in repr(fake_ml_stack["trl_config_kwargs"])
        assert "token" not in fake_ml_stack["trl_config_kwargs"]

    @pytest.mark.parametrize("config_cls", [DPOConfig, KTOConfig])
    def test_a_full_fine_tune_loads_its_reference_with_the_same_terms(
        self, fake_ml_stack: dict[str, Any], config_cls: type[DPOConfig | KTOConfig]
    ) -> None:
        # Left to TRL, the reference is re-downloaded by name with none of these terms: no token
        # (a gated base fails) and no safetensors requirement.
        cfg = config_cls(model_id="org/gated", output_dir="./out")
        PreferenceRunner(cfg).train(train_dataset=["row"], token=_TOKEN)
        loads = fake_ml_stack["model_loads"]
        assert loads == [{"model_id": "org/gated", "kwargs": _LOAD_TERMS}] * 2
        ref = fake_ml_stack["trainer_kwargs"]["ref_model"]
        assert ref is not fake_ml_stack["trainer_kwargs"]["model"]

    def test_an_adapter_run_needs_no_reference_model(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = DPOConfig(model_id="gpt2", output_dir="./out")
        PreferenceRunner(cfg, peft_config=LoRAConfig()).train(train_dataset=["row"])
        assert len(fake_ml_stack["model_loads"]) == 1
        assert "ref_model" not in fake_ml_stack["trainer_kwargs"]

    def test_orpo_loads_no_reference_model(self, fake_ml_stack: dict[str, Any]) -> None:
        PreferenceRunner(ORPOConfig(model_id="gpt2", output_dir="./out")).train(
            train_dataset=["row"]
        )
        assert len(fake_ml_stack["model_loads"]) == 1

    def test_precomputed_reference_log_probs_need_no_reference_model(
        self, fake_ml_stack: dict[str, Any]
    ) -> None:
        cfg = DPOConfig(
            model_id="gpt2",
            output_dir="./out",
            extra_trainer_args={"precompute_ref_log_probs": True},
        )
        PreferenceRunner(cfg).train(train_dataset=["row"])
        assert len(fake_ml_stack["model_loads"]) == 1
        assert "ref_model" not in fake_ml_stack["trainer_kwargs"]

    def test_a_caller_supplied_model_is_not_reloaded(self, fake_ml_stack: dict[str, Any]) -> None:
        cfg = DPOConfig(model_id="gpt2", output_dir="./out")
        PreferenceRunner(cfg).train(
            train_dataset=["row"], model=MagicMock(name="mine"), tokenizer=MagicMock()
        )
        assert "model_loads" not in fake_ml_stack

    def test_without_a_token_the_library_default_applies_but_remote_code_stays_off(
        self, fake_ml_stack: dict[str, Any]
    ) -> None:
        PreferenceRunner(ORPOConfig(model_id="gpt2", output_dir="./out")).train(
            train_dataset=["row"]
        )
        kwargs = fake_ml_stack["model_loads"][0]["kwargs"]
        assert kwargs == {"token": None, "trust_remote_code": False, "use_safetensors": True}
        assert fake_ml_stack["tokenizer_kwargs"]["trust_remote_code"] is False
