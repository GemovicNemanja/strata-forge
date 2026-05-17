"""Unit tests for `forge.evals.sweeps`."""

from __future__ import annotations

import pytest

from forge.evals.experiment import Experiment, SamplingParams
from forge.evals.sweeps import sweep, sweep_sampling


def _base() -> Experiment:
    return Experiment(
        name="base",
        models=("claude-opus-4-7",),
        dataset_name="ds",
        grader_names=("g",),
        sampling=SamplingParams(temperature=0.5, max_tokens=128, top_p=0.9),
    )


# ---------------------------------------------------------------------------
# sweep_sampling
# ---------------------------------------------------------------------------


class TestSweepSampling:
    def test_no_axes_returns_base_unchanged(self) -> None:
        variants = sweep_sampling(_base())
        assert variants == (_base(),)

    def test_single_axis_temperature(self) -> None:
        variants = sweep_sampling(_base(), temperatures=[0.0, 0.7, 1.0])
        assert len(variants) == 3
        temps = [v.sampling.temperature for v in variants]
        assert temps == [0.0, 0.7, 1.0]

    def test_variant_names_include_axis_and_value(self) -> None:
        variants = sweep_sampling(_base(), temperatures=[0.0, 1.0])
        names = [v.name for v in variants]
        assert names == ["base/temperature=0.0", "base/temperature=1.0"]

    def test_cross_product_when_multiple_axes(self) -> None:
        variants = sweep_sampling(_base(), temperatures=[0.0, 1.0], top_p=[0.5, 1.0])
        # 2 x 2 = 4
        assert len(variants) == 4

    def test_max_tokens_axis(self) -> None:
        variants = sweep_sampling(_base(), max_tokens=[64, 128, 256])
        assert len(variants) == 3
        assert [v.sampling.max_tokens for v in variants] == [64, 128, 256]

    def test_other_sampling_fields_preserved(self) -> None:
        variants = sweep_sampling(_base(), temperatures=[0.0])
        # The base had max_tokens=128 and top_p=0.9; varying only
        # temperature must leave the others intact.
        sampling = variants[0].sampling
        assert sampling.temperature == 0.0
        assert sampling.max_tokens == 128
        assert sampling.top_p == 0.9

    def test_variants_share_non_sampling_fields(self) -> None:
        variants = sweep_sampling(_base(), temperatures=[0.0, 1.0])
        for v in variants:
            assert v.models == _base().models
            assert v.dataset_name == _base().dataset_name
            assert v.grader_names == _base().grader_names


# ---------------------------------------------------------------------------
# sweep generic
# ---------------------------------------------------------------------------


class TestSweep:
    def test_models_axis(self) -> None:
        variants = sweep(
            _base(),
            models=[("m1",), ("m2",), ("m3",)],
        )
        assert len(variants) == 3
        assert [v.models for v in variants] == [("m1",), ("m2",), ("m3",)]

    def test_no_axes_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            sweep(_base())

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            sweep(_base(), nonexistent=[1, 2])

    def test_cross_product_multiple_axes(self) -> None:
        variants = sweep(
            _base(),
            models=[("m1",), ("m2",)],
            dataset_version=["v1", "v2", "v3"],
        )
        # 2 x 3 = 6
        assert len(variants) == 6
        seen = {(v.models, v.dataset_version) for v in variants}
        expected = {
            (("m1",), "v1"),
            (("m1",), "v2"),
            (("m1",), "v3"),
            (("m2",), "v1"),
            (("m2",), "v2"),
            (("m2",), "v3"),
        }
        assert seen == expected

    def test_variant_names_include_axis_values(self) -> None:
        variants = sweep(_base(), dataset_version=["v1", "v2"])
        names = [v.name for v in variants]
        assert names == ["base/dataset_version=v1", "base/dataset_version=v2"]

    def test_tuple_value_rendered_in_name(self) -> None:
        variants = sweep(_base(), models=[("m1", "m2")])
        assert variants[0].name == "base/models=m1,m2"

    def test_sampling_params_value_rendered(self) -> None:
        variants = sweep(
            _base(),
            sampling=[
                SamplingParams(temperature=0.0, max_tokens=64),
                SamplingParams(temperature=1.0),
            ],
        )
        assert "temp=0.0" in variants[0].name
        assert "max=64" in variants[0].name
        assert "temp=1.0" in variants[1].name

    def test_sampling_params_with_top_p(self) -> None:
        variants = sweep(_base(), sampling=[SamplingParams(top_p=0.5)])
        assert "top_p=0.5" in variants[0].name

    def test_default_sampling_params_renders_default(self) -> None:
        variants = sweep(_base(), sampling=[SamplingParams()])
        # All fields are None → falls back to "default".
        assert "default" in variants[0].name
