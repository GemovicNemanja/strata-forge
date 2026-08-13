"""Unit tests for `strata_forge.training.dataset_format`."""

from __future__ import annotations

import pytest

from strata_forge.training.dataset_format import (
    FORMATS,
    DatasetFormatError,
    build_training_rows,
    pick_format,
    sft_text_field,
    validate_mapping,
)


class TestPickFormat:
    def test_every_registered_format_resolves_to_itself(self) -> None:
        for name, spec in FORMATS.items():
            assert pick_format(name) is spec

    def test_unknown_format_lists_the_valid_ones(self) -> None:
        with pytest.raises(DatasetFormatError, match="unknown dataset format 'chatml'"):
            pick_format("chatml")
        with pytest.raises(DatasetFormatError, match="prompt_completion"):
            pick_format("chatml")

    def test_required_roles_come_first_in_roles(self) -> None:
        spec = pick_format("preference")
        assert spec.required == ("chosen", "rejected")
        assert spec.optional == ("prompt",)
        assert spec.roles == ("chosen", "rejected", "prompt")


class TestSftTextField:
    def test_flat_text_names_the_column(self) -> None:
        assert sft_text_field("text") == "text"

    @pytest.mark.parametrize("fmt", ["prompt_completion", "conversational"])
    def test_role_formats_let_trl_infer(self, fmt: str) -> None:
        assert sft_text_field(fmt) is None

    @pytest.mark.parametrize("fmt", ["preference", "unpaired_preference"])
    def test_preference_formats_are_not_sft_formats(self, fmt: str) -> None:
        with pytest.raises(DatasetFormatError, match="cannot be used for supervised fine-tuning"):
            sft_text_field(fmt)


class TestValidateMapping:
    def test_accepts_a_complete_mapping(self) -> None:
        spec = validate_mapping(
            "prompt_completion",
            {"prompt": "question", "completion": "answer"},
            ["question", "answer", "id"],
        )
        assert spec.name == "prompt_completion"

    def test_optional_role_may_be_omitted(self) -> None:
        validate_mapping("preference", {"chosen": "good", "rejected": "bad"}, ["good", "bad"])

    def test_missing_required_role_names_it_and_the_columns(self) -> None:
        with pytest.raises(DatasetFormatError) as exc:
            validate_mapping("preference", {"chosen": "good"}, ["good", "bad"])
        assert "rejected" in str(exc.value)
        # Sorted, so the list a user reads is stable rather than in split order.
        assert "bad, good" in str(exc.value)

    def test_blank_mapping_counts_as_missing(self) -> None:
        with pytest.raises(DatasetFormatError, match="needs a column for: text"):
            validate_mapping("text", {"text": "   "}, ["body"])

    def test_column_absent_from_the_split_is_named(self) -> None:
        with pytest.raises(DatasetFormatError) as exc:
            validate_mapping("text", {"text": "body"}, ["content"])
        assert "'body'" in str(exc.value)
        assert "content" in str(exc.value)

    def test_role_the_format_does_not_have_is_rejected(self) -> None:
        with pytest.raises(DatasetFormatError, match="has no role"):
            validate_mapping("text", {"text": "body", "chosen": "body"}, ["body"])

    def test_empty_split_reports_no_columns_rather_than_an_empty_list(self) -> None:
        with pytest.raises(DatasetFormatError, match=r"\(none\)"):
            validate_mapping("text", {"text": "body"}, [])


class TestBuildTrainingRows:
    def test_renames_roles_and_drops_everything_else(self) -> None:
        rows = [{"question": "q1", "answer": "a1", "id": 7, "source": "web"}]
        out = build_training_rows(
            rows, "prompt_completion", {"prompt": "question", "completion": "answer"}
        )
        assert out == [{"prompt": "q1", "completion": "a1"}]

    def test_omitted_optional_role_is_absent_not_null(self) -> None:
        rows = [{"good": "g", "bad": "b"}]
        out = build_training_rows(rows, "preference", {"chosen": "good", "rejected": "bad"})
        assert out == [{"chosen": "g", "rejected": "b"}]
        assert "prompt" not in out[0]

    def test_included_optional_role_rides_along(self) -> None:
        rows = [{"q": "why", "good": "g", "bad": "b"}]
        out = build_training_rows(
            rows, "preference", {"prompt": "q", "chosen": "good", "rejected": "bad"}
        )
        assert out == [{"chosen": "g", "rejected": "b", "prompt": "why"}]

    def test_row_missing_a_mapped_column_names_the_row(self) -> None:
        rows = [{"body": "one"}, {"other": "two"}]
        with pytest.raises(DatasetFormatError, match="row 1 has no column 'body'"):
            build_training_rows(rows, "text", {"text": "body"})

    def test_empty_input_is_an_error_not_an_empty_run(self) -> None:
        with pytest.raises(DatasetFormatError, match="produced no rows"):
            build_training_rows([], "text", {"text": "body"})


class TestLabelCoercion:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (True, True),
            (False, False),
            (1, True),
            (0, False),
            ("true", True),
            ("TRUE", True),
            ("yes", True),
            (" Y ", True),
            ("1", True),
            ("false", False),
            ("no", False),
            ("0", False),
        ],
    )
    def test_boolean_ish_labels_become_bools(self, raw: object, expected: bool) -> None:
        rows = [{"p": "x", "c": "y", "l": raw}]
        out = build_training_rows(
            rows, "unpaired_preference", {"prompt": "p", "completion": "c", "label": "l"}
        )
        assert out[0]["label"] is expected

    def test_an_arbitrary_string_label_is_refused_rather_than_read_as_true(self) -> None:
        rows = [{"p": "x", "c": "y", "l": "maybe"}]
        with pytest.raises(DatasetFormatError, match="not a yes/no answer"):
            build_training_rows(
                rows, "unpaired_preference", {"prompt": "p", "completion": "c", "label": "l"}
            )

    def test_a_non_scalar_label_is_refused(self) -> None:
        rows = [{"p": "x", "c": "y", "l": {"nested": True}}]
        with pytest.raises(DatasetFormatError, match="not a boolean-ish label"):
            build_training_rows(
                rows, "unpaired_preference", {"prompt": "p", "completion": "c", "label": "l"}
            )

    def test_only_the_label_role_is_coerced(self) -> None:
        rows = [{"body": 42}]
        out = build_training_rows(rows, "text", {"text": "body"})
        assert out[0]["text"] == 42


class TestConversationalShape:
    def test_a_list_of_turns_is_accepted(self) -> None:
        rows = [{"conv": [{"role": "user", "content": "hi"}]}]
        out = build_training_rows(rows, "conversational", {"messages": "conv"})
        assert out[0]["messages"] == [{"role": "user", "content": "hi"}]

    def test_a_plain_string_column_is_refused_with_a_pointer_to_the_right_format(self) -> None:
        rows = [{"conv": "hi there"}]
        with pytest.raises(DatasetFormatError, match="prompt_completion"):
            build_training_rows(rows, "conversational", {"messages": "conv"})

    def test_turns_without_role_and_content_are_refused(self) -> None:
        rows = [{"conv": [{"from": "human", "value": "hi"}]}]
        with pytest.raises(DatasetFormatError, match="'role' and 'content'"):
            build_training_rows(rows, "conversational", {"messages": "conv"})

    def test_an_empty_conversation_is_shape_valid(self) -> None:
        rows = [{"conv": []}]
        out = build_training_rows(rows, "conversational", {"messages": "conv"})
        assert out[0]["messages"] == []
