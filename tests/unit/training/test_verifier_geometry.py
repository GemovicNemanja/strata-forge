"""Synthetic exact rendering/offset cases; no tokenizer/model assets."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from strata_forge.training.verifier_geometry import (
    VerifierExample,
    VerifierGeometryConfig,
    VerifierGeometryError,
    VerifierPremise,
    render_verifier_geometry,
    resolve_pid_markers,
)


def geometry(pool: str = "cls", /, **kwargs: object) -> VerifierGeometryConfig:
    values = {
        "labels": ("entailed", "contradicted", "undetermined"),
        "template": "{premises}"
        if pool == "cls"
        else "Premises:\n{premises}\nHypothesis: {hypothesis}\nAnswer:",
        "pair_template": "{hypothesis}" if pool == "cls" else None,
        "max_length": 64,
        "overflow": "error",
        "pooling": pool,
        "outputs": ("label", "inconsistent", "support_set"),
    }
    return VerifierGeometryConfig.model_validate({**values, **kwargs})


def example() -> VerifierExample:
    return VerifierExample(
        item_id="fixture",
        premises=(
            VerifierPremise(pid="p9", text="é 😀 p2: literal"),
            VerifierPremise(pid="p2", text="{hypothesis}"),
        ),
        hypothesis="q",
        support_set=("p2",),
    )


def test_exact_ids_unicode_and_nonrecursive_render() -> None:
    row = render_verifier_geometry(geometry(), example())
    assert row.text == "p9: é 😀 p2: literal\np2: {hypothesis}"
    assert row.text_pair == "q"
    assert row.premise_ids == ("p9", "p2")
    assert row.pid_starts == (0, len("p9: é 😀 p2: literal\n"))
    decoder = render_verifier_geometry(geometry("answer_position"), example())
    assert decoder.text == "Premises:\n" + row.text + "\nHypothesis: q\nAnswer:"
    assert decoder.pid_starts == tuple(start + len("Premises:\n") for start in row.pid_starts)


def test_marker_filters_sequence_specials_and_padding() -> None:
    row = render_verifier_geometry(geometry(), example())
    # Hypothesis token has the same offset as PID1, but is in sequence1. An
    # earlier special and a padded token also overlap and must not win.
    starts = row.pid_starts
    assert resolve_pid_markers(
        row,
        offsets=[(0, 2), (0, 1), (0, 2), (starts[1], starts[1] + 2), (0, 2)],
        sequence_ids=[None, 1, 0, 0, None],
        special_tokens_mask=[1, 0, 0, 0, 1],
        attention_mask=[1, 1, 1, 1, 0],
    ) == (2, 3)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "shared",
        "outside",
        "negative",
        "reversed",
        "bool",
        "length",
        "none_sequence",
        "special",
        "padding",
    ],
)
def test_bad_markers_refuse_content_free(damage: str) -> None:
    row = render_verifier_geometry(geometry("answer_position"), example())
    first, second = row.pid_starts
    args: dict[str, object] = {
        "offsets": [(first, first + 2), (second, second + 2)],
        "sequence_ids": [0, 0],
        "special_tokens_mask": [0, 0],
        "attention_mask": [1, 1],
    }
    if damage == "missing":
        args["offsets"] = [(first + 1, first + 2), (second, second + 2)]
    elif damage == "shared":
        args["offsets"] = [(first, second + 2), (second, second + 2)]
    elif damage == "outside":
        args["offsets"] = [(first - 1, first + 2), (second, second + 2)]
    elif damage == "negative":
        args["offsets"] = [(-1, first + 2), (second, second + 2)]
    elif damage == "reversed":
        args["offsets"] = [(first + 2, first), (second, second + 2)]
    elif damage == "bool":
        args["offsets"] = [(False, first + 2), (second, second + 2)]
    elif damage == "length":
        args["attention_mask"] = [1]
    elif damage == "none_sequence":
        args["sequence_ids"] = [None, 0]
    elif damage == "special":
        args["special_tokens_mask"] = [1, 0]
    else:
        args["attention_mask"] = [0, 1]
    with pytest.raises(VerifierGeometryError, match=r"^verifier_markers_invalid$") as error:
        resolve_pid_markers(row, **args)
    assert error.value.__context__ is error.value.__cause__ is None


def test_first_piece_wins_and_wrong_support_or_duplicate_pid_refuses() -> None:
    row = render_verifier_geometry(geometry(), example())
    start = row.pid_starts[1]
    assert resolve_pid_markers(
        row,
        offsets=[(0, 1), (0, 2), (start, start + 2)],
        sequence_ids=[0, 0, 0],
        special_tokens_mask=[0, 0, 0],
        attention_mask=[1, 1, 1],
    ) == (0, 2)
    with pytest.raises(ValidationError):
        example().model_validate({**example().model_dump(), "support_set": ("p99",)})
    with pytest.raises(ValidationError):
        example().model_validate(
            {**example().model_dump(), "premises": (example().premises[0], example().premises[0])}
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 2.0},
        {"overflow": "drop"},
        {"outputs": ("inconsistent",)},
        {"outputs": ("label", "label")},
        {"max_length": 8193},
        {"max_premises": 21},
        {"template": "prefix {premises}"},
        {"pooling": "answer_position", "pair_template": None, "template": "{hypothesis}{premises}"},
    ],
)
def test_geometry_choices_are_closed(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        geometry(**changes)


def test_limits_and_tampered_instances_are_not_silently_adopted() -> None:
    with pytest.raises(VerifierGeometryError):
        render_verifier_geometry(geometry(max_premises=1), example())
    with pytest.raises(VerifierGeometryError):
        render_verifier_geometry(geometry(outputs=("label",)), example())
    bad = geometry().model_copy(update={"max_length": True})
    with pytest.raises(VerifierGeometryError):
        render_verifier_geometry(bad, example())
    too_large = example().model_copy(update={"hypothesis": "x" * 200_001})
    with pytest.raises(VerifierGeometryError):
        render_verifier_geometry(geometry(), too_large)
