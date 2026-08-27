"""The names this package hands to TRL / transformers, pinned.

A training config is a bag of kwargs splatted into somebody else's constructor, so a rename upstream
does not fail here -- it fails on the user's GPU box, twenty minutes into a run that already paid to
provision. That is not hypothetical: `transformers` 5 removed `TrainingArguments.warmup_ratio` and
every training path died at config construction, because `pip install` on a fresh VM resolves to
whatever shipped that morning.

So the kwarg NAMES are frozen here rather than left implicit. These sets are a contract against the
majors `pyproject.toml` pins; changing one means checking the new name against those majors and
editing this file deliberately, which is exactly the beat that was missing.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from pydantic import ValidationError

from strata_forge.training.preference import (
    DPOConfig,
    GRPOConfig,
    KTOConfig,
    ORPOConfig,
    _resolve_trl_class,  # pyright: ignore[reportPrivateUsage]
)
from strata_forge.training.sft import SFTConfig

#: Keys every preference config emits via the shared base.
_BASE_KEYS = frozenset(
    {
        "output_dir",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "warmup_steps",
        "weight_decay",
        "logging_steps",
        "gradient_checkpointing",
        "seed",
        "save_strategy",
        "bf16",
    }
)

_EXPECTED_KEYS: dict[str, frozenset[str]] = {
    "dpo": _BASE_KEYS | {"beta", "loss_type", "max_length"},
    "orpo": _BASE_KEYS | {"beta", "max_length"},
    "kto": _BASE_KEYS | {"beta", "desirable_weight", "undesirable_weight", "max_length"},
    "grpo": _BASE_KEYS | {"beta", "num_generations", "max_completion_length"},
}

_CONFIGS: dict[str, Any] = {
    "dpo": DPOConfig,
    "orpo": ORPOConfig,
    "kto": KTOConfig,
    "grpo": GRPOConfig,
}


class TestForwardedKwargNames:
    @pytest.mark.parametrize("method", sorted(_EXPECTED_KEYS))
    def test_emits_exactly_the_pinned_key_set(self, method: str) -> None:
        cfg = _CONFIGS[method](model_id="gpt2", output_dir="./out")
        assert set(cfg.to_trl_kwargs()) == set(_EXPECTED_KEYS[method])

    def test_sft_emits_exactly_the_pinned_key_set(self) -> None:
        cfg = SFTConfig(model_id="gpt2", output_dir="./out")
        assert set(cfg.to_trl_kwargs()) == {
            "output_dir",
            "num_train_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "warmup_steps",
            "weight_decay",
            "logging_steps",
            "gradient_checkpointing",
            "seed",
            "max_length",
            "packing",
            "dataset_text_field",
            "save_strategy",
            "bf16",
        }

    @pytest.mark.parametrize("method", sorted(_EXPECTED_KEYS))
    def test_never_sends_warmup_ratio(self, method: str) -> None:
        # transformers 5 folded it into `warmup_steps`. This is the exact key that took every
        # training path down.
        assert (
            "warmup_ratio"
            not in _CONFIGS[method](model_id="gpt2", output_dir="./out").to_trl_kwargs()
        )

    def test_sft_never_sends_warmup_ratio(self) -> None:
        assert "warmup_ratio" not in SFTConfig(model_id="gpt2", output_dir="./out").to_trl_kwargs()

    @pytest.mark.parametrize("method", sorted(_EXPECTED_KEYS))
    def test_never_sends_max_prompt_length(self, method: str) -> None:
        # TRL removed it across 0.27-0.29 with no replacement -- only `max_length` survives.
        assert (
            "max_prompt_length"
            not in _CONFIGS[method](model_id="gpt2", output_dir="./out").to_trl_kwargs()
        )

    def test_the_knob_keeps_its_meaning_across_the_rename(self) -> None:
        # `warmup_steps` reads a float in [0, 1) as a fraction of total steps, so the value carries
        # over untouched. If it were ever coerced to an int, 0.05 would silently become 0 warmup.
        cfg = SFTConfig(model_id="gpt2", output_dir="./out", warmup_ratio=0.05)
        forwarded = cfg.to_trl_kwargs()["warmup_steps"]
        assert forwarded == 0.05
        assert isinstance(forwarded, float)

    def test_dpo_rejects_a_loss_type_trl_dropped(self) -> None:
        # The static rejection below is half the guard; this asserts the RUNTIME one, which is what
        # a value arriving as JSON from an orchestrator actually meets.
        with pytest.raises(ValidationError):
            DPOConfig(
                model_id="gpt2",
                output_dir="./out",
                loss_type="kto_pair",  # pyright: ignore[reportArgumentType]
            )


class TestResolveTrlClass:
    def test_prefers_the_top_level_namespace(self) -> None:
        trl_mod = types.ModuleType("trl")
        trl_mod.ORPOConfig = "mainline"  # type: ignore[attr-defined]
        assert _resolve_trl_class(trl_mod, "orpo", "ORPOConfig") == "mainline"

    def test_falls_back_to_the_experimental_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # TRL 1.x keeps ORPO only under `trl.experimental.orpo`; a plain getattr raises there.
        experimental = types.ModuleType("trl.experimental.orpo")
        experimental.ORPOConfig = "experimental"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "trl.experimental.orpo", experimental)
        trl_mod = types.ModuleType("trl")  # no ORPOConfig attribute
        assert _resolve_trl_class(trl_mod, "orpo", "ORPOConfig") == "experimental"

    def test_reports_the_method_when_nothing_provides_the_class(self) -> None:
        trl_mod = types.ModuleType("trl")
        with pytest.raises(AttributeError, match="dpo"):
            _resolve_trl_class(trl_mod, "dpo", "DPOConfig")
