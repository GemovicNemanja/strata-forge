"""The classifier's declaration and premise rendering: pure, no heavy dependency."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from strata_forge.training.dataset_format import (
    DatasetFormatError,
    build_training_rows,
    validate_mapping,
)
from strata_forge.training.seq_cls import (
    SeqClsConfig,
)
from strata_forge.training.verifier_render import fill_template as _decoder_text
from strata_forge.training.verifier_render import render_premises

_LABELS = ("entailment", "contradiction", "neutral")


def _config(**overrides: Any) -> SeqClsConfig:
    return SeqClsConfig(model_id="m", output_dir="o", labels=_LABELS, **overrides)


class TestConfig:
    def test_defaults_are_the_declared_contract(self) -> None:
        config = _config()
        assert config.max_length == 1024
        assert config.per_device_batch_size == 32
        assert config.precision == "bf16"
        assert config.gradient_checkpointing is False
        assert config.brier_weight == 0.0
        assert config.two_way_positive_label is None
        assert config.two_way_labels is None

    @pytest.mark.parametrize("steps", [1, 2, 8])
    def test_gradient_accumulation_is_refused_by_name(self, steps: int) -> None:
        with pytest.raises(ValidationError, match="gradient_accumulation_steps is refused"):
            _config(gradient_accumulation_steps=steps)

    @pytest.mark.parametrize(
        "labels",
        [
            ("only",),
            ("a", "a"),
            ("a", "b c"),
            tuple(f"l{i}" for i in range(17)),
        ],
    )
    def test_labels_are_validated(self, labels: tuple[str, ...]) -> None:
        with pytest.raises(ValidationError):
            SeqClsConfig(model_id="m", output_dir="o", labels=labels)

    def test_an_alias_must_name_a_declared_label(self) -> None:
        with pytest.raises(ValidationError, match="not declared"):
            _config(label_aliases={"yes": "entailed"})
        assert _config(label_aliases={"yes": "entailment"}).label_aliases

    def test_two_way_labels_are_declared_explicitly_and_together(self) -> None:
        with pytest.raises(ValidationError, match="declared together"):
            _config(two_way_positive_label="entailment")
        with pytest.raises(ValidationError, match="declared together"):
            _config(two_way_labels=("entailed", "not_entailed"))
        with pytest.raises(ValidationError, match="one of labels"):
            _config(two_way_positive_label="entailed", two_way_labels=("entailed", "not_entailed"))
        with pytest.raises(ValidationError, match="two distinct"):
            _config(two_way_positive_label="entailment", two_way_labels=("x", "x"))
        config = _config(
            two_way_positive_label="contradiction", two_way_labels=("entailed", "not_entailed")
        )
        assert config.positive_index == 1

    def test_an_index_map_is_integer_keyed_and_declared(self) -> None:
        with pytest.raises(ValidationError, match="integers"):
            _config(label_index_map={"zero": "entailment"})
        with pytest.raises(ValidationError, match="not declared"):
            _config(label_index_map={"0": "other"})

    def test_a_decoder_head_needs_a_verbalizer_per_label(self) -> None:
        with pytest.raises(ValidationError, match="verbalizer for every label"):
            _config(head_init="decoder_lm_head")
        with pytest.raises(ValidationError, match="verbalizer for every label"):
            _config(head_init="decoder_lm_head", verbalizers={"entailment": " True"})
        verbalizers = {"entailment": " True", "contradiction": " False", "neutral": " Unknown"}
        assert _config(head_init="decoder_lm_head", verbalizers=verbalizers).from_lm_head

    def test_verbalizers_belong_to_the_decoder_head_only(self) -> None:
        with pytest.raises(ValidationError, match="only to head_init"):
            _config(verbalizers={"entailment": " a", "contradiction": " b", "neutral": " c"})

    @pytest.mark.parametrize(
        "template", ["{premises}", "{premises} {hypothesis} {premises}", "{premises} {question}"]
    )
    def test_the_decoder_template_names_both_fields_once(self, template: str) -> None:
        with pytest.raises(ValidationError, match="decoder_template"):
            _config(decoder_template=template)

    def test_unknown_knobs_are_refused(self) -> None:
        with pytest.raises(ValidationError):
            _config(label_smoothing=0.1)


class TestRenderPremises:
    def test_ids_lines_and_marker_offsets(self) -> None:
        text, markers = render_premises(
            [{"pid": "p1", "text": "alpha"}, {"pid": "p2", "text": "beta"}], "ids"
        )
        assert text == "p1: alpha\np2: beta"
        assert markers == [("p1", 0), ("p2", 10)]
        assert all(text[offset:].startswith(pid) for pid, offset in markers)

    def test_plain_renders_text_without_markers(self) -> None:
        cell = [{"pid": "p1", "text": "alpha"}, {"pid": "p2", "text": "beta"}]
        assert render_premises(cell, "plain") == ("alpha\nbeta", [])

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            ("alpha. beta.", "list of"),
            (["alpha", "beta"], "pid and text"),
            ([], "non-empty"),
            ([{"pid": "p2", "text": "a"}], "p1..pN"),
            ([{"pid": "p1", "text": "a"}, {"pid": "p1", "text": "b"}], "p1..pN"),
            ([{"pid": "1", "text": "a"}], "p1..pN"),
            ([{"pid": "p1"}], "pid and text"),
            ([{"pid": "p1", "text": 3}], "string"),
            (42, "list of"),
        ],
    )
    def test_premises_must_be_ordered_pid_objects(self, bad: Any, match: str) -> None:
        with pytest.raises(DatasetFormatError, match=match):
            render_premises(bad, "ids")

    def test_the_decoder_template_reports_where_the_premises_start(self) -> None:
        text, start = _decoder_text("Q: {premises}\n{hypothesis}", "p1: a", "h")
        assert text == "Q: p1: a\nh"
        assert text[start:].startswith("p1: a")

    def test_a_field_holding_braces_is_not_substituted_again(self) -> None:
        text, _ = _decoder_text("{premises}\n{hypothesis}", "{hypothesis}", "h")
        assert text == "{hypothesis}\nh"


class TestNliPairFormat:
    def test_roles(self) -> None:
        spec = validate_mapping(
            "nli_pair",
            {"premises": "p", "hypothesis": "h", "label": "y", "support_set": "s"},
            ["p", "h", "y", "s"],
        )
        assert spec.required == ("premises", "hypothesis", "label")
        assert "inconsistent_premises" in spec.optional

    def test_a_missing_required_role_is_named(self) -> None:
        with pytest.raises(DatasetFormatError, match="hypothesis"):
            validate_mapping("nli_pair", {"premises": "p", "label": "y"}, ["p", "y"])

    def test_labels_are_not_coerced_to_booleans(self) -> None:
        # Only KTO reads its label as a boolean; a classification label stays what it is.
        rows = build_training_rows(
            [{"p": "a", "h": "b", "y": "neutral"}],
            "nli_pair",
            {"premises": "p", "hypothesis": "h", "label": "y"},
        )
        assert rows == [{"premises": "a", "hypothesis": "b", "label": "neutral"}]

    def test_kto_still_reads_a_boolean_label(self) -> None:
        rows = build_training_rows(
            [{"p": "q", "c": "a", "y": "yes"}],
            "unpaired_preference",
            {"prompt": "p", "completion": "c", "label": "y"},
        )
        assert rows[0]["label"] is True


@pytest.mark.parametrize("key", ["step0_rows", "step0_tolerance"])
def test_the_check_sample_and_tolerance_cannot_be_set(key: str) -> None:
    from strata_forge.training.seq_cls import CHECK_ROWS, CHECK_TOLERANCE

    with pytest.raises(ValidationError, match=f"{key} cannot be set for seq_cls"):
        _config(**{key: 1.0 if key.endswith("tolerance") else 1})
    assert (CHECK_ROWS, CHECK_TOLERANCE) == (8, 1e-4)


@pytest.mark.parametrize(
    "key",
    [
        "eval_steps",
        "load_best_model_at_end",
        "metric_for_best_model",
        "greater_is_better",
        "early_stopping_patience",
        "early_stopping_threshold",
    ],
)
def test_nothing_may_pick_a_checkpoint_or_move_the_monitoring(key: str) -> None:
    with pytest.raises(
        ValidationError, match=f"{key} cannot be set for seq_cls: the final checkpoint"
    ):
        _config(**{key: 1})
