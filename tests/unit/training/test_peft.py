"""Unit tests for `strata_forge.training.peft`."""

from __future__ import annotations

import contextlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from strata_forge.core.errors import ForgeError
from strata_forge.training.peft import (
    LoRAConfig,
    MissingBitsAndBytesError,
    QLoRAConfig,
    require_bitsandbytes,
)


class TestLoRAConfig:
    def test_defaults(self) -> None:
        cfg = LoRAConfig()
        assert cfg.r == 16
        assert cfg.alpha == 32
        assert cfg.dropout == 0.05
        assert cfg.target_modules is None
        assert cfg.bias == "none"
        assert cfg.task_type == "CAUSAL_LM"

    def test_validates_positive_rank(self) -> None:
        with pytest.raises(ValueError, match="r"):
            LoRAConfig(r=0)

    def test_validates_dropout_range(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            LoRAConfig(dropout=2.0)

    def test_frozen(self) -> None:
        cfg = LoRAConfig()
        with pytest.raises(Exception, match="frozen"):
            cfg.r = 32  # type: ignore[misc]

    def test_to_peft_config_missing_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "peft", None)
        with pytest.raises(ImportError, match=r"\[finetuning\] extra"):
            LoRAConfig().to_peft_config()

    def test_to_peft_config_forwards_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _FakeLoraConfig:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        fake = types.ModuleType("peft")
        fake.LoraConfig = _FakeLoraConfig  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "peft", fake)

        cfg = LoRAConfig(
            r=32,
            alpha=64,
            dropout=0.1,
            target_modules=("q_proj", "v_proj"),
            modules_to_save=("lm_head",),
        )
        cfg.to_peft_config()
        assert captured["r"] == 32
        assert captured["lora_alpha"] == 64
        assert captured["lora_dropout"] == 0.1
        assert captured["target_modules"] == ["q_proj", "v_proj"]
        assert captured["modules_to_save"] == ["lm_head"]
        assert captured["task_type"] == "CAUSAL_LM"

    def test_to_peft_config_skips_unset_target_modules(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        class _FakeLoraConfig:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        fake = types.ModuleType("peft")
        fake.LoraConfig = _FakeLoraConfig  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "peft", fake)

        LoRAConfig().to_peft_config()
        assert "target_modules" not in captured
        assert "modules_to_save" not in captured


class TestQLoRAConfig:
    def test_defaults(self) -> None:
        cfg = QLoRAConfig()
        assert cfg.load_in_4bit is True
        assert cfg.bnb_4bit_quant_type == "nf4"
        assert cfg.bnb_4bit_use_double_quant is True
        assert cfg.bnb_4bit_compute_dtype == "bfloat16"
        # Default LoRA has higher rank for QLoRA.
        assert cfg.lora.r == 64

    def test_to_peft_config_delegates_to_lora(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _FakeLoraConfig:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        fake = types.ModuleType("peft")
        fake.LoraConfig = _FakeLoraConfig  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "peft", fake)

        QLoRAConfig().to_peft_config()
        assert captured["r"] == 64

    @pytest.mark.usefixtures("bitsandbytes_installed")
    def test_to_bnb_config_builds_with_dtype(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _FakeBnbConfig:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.BitsAndBytesConfig = _FakeBnbConfig  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = "BFLOAT16_DTYPE"  # type: ignore[attr-defined]
        fake_torch.float16 = "FLOAT16_DTYPE"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        QLoRAConfig().to_bnb_config()
        assert captured["load_in_4bit"] is True
        assert captured["bnb_4bit_quant_type"] == "nf4"
        assert captured["bnb_4bit_use_double_quant"] is True
        assert captured["bnb_4bit_compute_dtype"] == "BFLOAT16_DTYPE"

    @pytest.mark.usefixtures("bitsandbytes_installed")
    def test_to_bnb_config_missing_transformers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.raises(ImportError, match=r"\[finetuning\] extra"):
            QLoRAConfig().to_bnb_config()

    @pytest.mark.usefixtures("bitsandbytes_installed")
    def test_to_bnb_config_missing_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.BitsAndBytesConfig = MagicMock()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ImportError, match=r"\[finetuning\] extra"):
            QLoRAConfig().to_bnb_config()

    @pytest.mark.usefixtures("bitsandbytes_missing")
    def test_to_bnb_config_refuses_without_bitsandbytes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The config itself would construct; the model load is where the absence would surface.
        bnb_config = MagicMock()
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.BitsAndBytesConfig = bnb_config  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        with pytest.raises(MissingBitsAndBytesError, match="bitsandbytes"):
            QLoRAConfig().to_bnb_config()
        bnb_config.assert_not_called()


class TestRequireBitsandbytes:
    @pytest.mark.usefixtures("bitsandbytes_missing")
    def test_names_the_package_the_extra_and_the_way_out(self) -> None:
        with pytest.raises(MissingBitsAndBytesError) as exc_info:
            require_bitsandbytes()
        message = str(exc_info.value)
        assert "bitsandbytes" in message
        assert "strata-forge[finetuning]" in message
        assert "adapter 'lora'" in message

    @pytest.mark.usefixtures("bitsandbytes_missing")
    def test_is_a_forge_error_and_an_import_error(self) -> None:
        # ForgeError for the hierarchy; ImportError so a missing-extra handler already catches it.
        with pytest.raises(ForgeError):
            require_bitsandbytes()
        with pytest.raises(ImportError):
            require_bitsandbytes()

    @pytest.mark.usefixtures("bitsandbytes_installed")
    def test_passes_when_installed(self) -> None:
        require_bitsandbytes()

    def test_looks_the_package_up_without_importing_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Importing bitsandbytes loads its CUDA library; the check must not be what does that.
        import importlib.util

        looked_up: list[str] = []
        real_find_spec = importlib.util.find_spec

        def _find_spec(name: str, package: str | None = None) -> Any:
            looked_up.append(name)
            return real_find_spec(name, package)

        monkeypatch.delitem(sys.modules, "bitsandbytes", raising=False)
        monkeypatch.setattr(importlib.util, "find_spec", _find_spec)
        with contextlib.suppress(MissingBitsAndBytesError):
            require_bitsandbytes()
        assert looked_up == ["bitsandbytes"]
        assert "bitsandbytes" not in sys.modules
