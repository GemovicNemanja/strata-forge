"""Unit tests for `strata_forge.training.methods`."""

from __future__ import annotations

import pytest

from strata_forge.training.dataset_format import FORMATS
from strata_forge.training.methods import (
    METHODS,
    UnsupportedMethodError,
    check_format,
    enabled_methods,
    pick_method,
)
from strata_forge.training.preference import DPOConfig, KTOConfig, ORPOConfig, PreferenceRunner
from strata_forge.training.sft import SFTConfig, SFTRunner


class TestRegistry:
    def test_every_entry_is_keyed_by_its_own_name(self) -> None:
        for key, spec in METHODS.items():
            assert key == spec.name

    def test_enabled_methods_are_the_four_declarable_ones(self) -> None:
        assert [s.name for s in enabled_methods()] == ["sft", "dpo", "orpo", "kto"]

    def test_every_enabled_method_declares_at_least_one_format(self) -> None:
        for spec in enabled_methods():
            assert spec.formats, f"{spec.name} accepts no dataset format"

    def test_every_declared_format_is_a_real_format(self) -> None:
        for spec in METHODS.values():
            assert spec.formats <= set(FORMATS), f"{spec.name} names an unknown format"

    def test_shipped_methods_train_a_causal_head(self) -> None:
        for spec in enabled_methods():
            assert spec.task_type == "CAUSAL_LM"

    def test_config_and_runner_pairings(self) -> None:
        assert pick_method("sft").config_cls is SFTConfig
        assert pick_method("sft").runner_cls is SFTRunner
        assert pick_method("dpo").config_cls is DPOConfig
        assert pick_method("orpo").config_cls is ORPOConfig
        assert pick_method("kto").config_cls is KTOConfig
        for name in ("dpo", "orpo", "kto"):
            assert pick_method(name).runner_cls is PreferenceRunner

    def test_a_disabled_method_carries_a_reason(self) -> None:
        assert METHODS["grpo"].enabled is False
        assert METHODS["grpo"].disabled_reason


class TestPickMethod:
    def test_unknown_method_lists_only_the_enabled_ones(self) -> None:
        with pytest.raises(UnsupportedMethodError) as exc:
            pick_method("ppo")
        assert "sft, dpo, orpo, kto" in str(exc.value)
        assert "grpo" not in str(exc.value)

    def test_disabled_method_explains_itself_rather_than_reading_as_unknown(self) -> None:
        with pytest.raises(UnsupportedMethodError) as exc:
            pick_method("grpo")
        assert "reward function" in str(exc.value)
        assert "unknown" not in str(exc.value)


class TestCheckFormat:
    @pytest.mark.parametrize(
        ("method", "fmt"),
        [
            ("sft", "text"),
            ("sft", "prompt_completion"),
            ("sft", "conversational"),
            ("dpo", "preference"),
            ("orpo", "preference"),
            ("kto", "unpaired_preference"),
        ],
    )
    def test_accepts_a_supported_pairing(self, method: str, fmt: str) -> None:
        check_format(pick_method(method), fmt)

    @pytest.mark.parametrize(
        ("method", "fmt"),
        [
            ("sft", "preference"),
            ("dpo", "text"),
            ("dpo", "unpaired_preference"),
            ("kto", "preference"),
        ],
    )
    def test_rejects_an_unsupported_pairing_and_names_the_alternatives(
        self, method: str, fmt: str
    ) -> None:
        spec = pick_method(method)
        with pytest.raises(UnsupportedMethodError) as exc:
            check_format(spec, fmt)
        assert method in str(exc.value)
        for accepted in spec.formats:
            assert accepted in str(exc.value)


class TestBuildRunner:
    def test_builds_the_registered_runner_without_touching_trl(self) -> None:
        spec = pick_method("sft")
        runner = spec.build_runner(SFTConfig(model_id="gpt2", output_dir="./out"))
        assert isinstance(runner, SFTRunner)
        assert runner.peft_config is None

    def test_forwards_the_adapter_config(self) -> None:
        from strata_forge.training.peft import LoRAConfig

        lora = LoRAConfig(r=8)
        runner = pick_method("dpo").build_runner(
            DPOConfig(model_id="gpt2", output_dir="./out"), peft_config=lora
        )
        assert isinstance(runner, PreferenceRunner)
        assert runner.peft_config is lora
