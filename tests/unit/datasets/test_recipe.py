"""Dataset recipes: the spec's rules, every step's semantics, synthesis, and the golden fixture.

Every row here is synthetic. The golden fixture (``tests/unit/datasets/recipe_golden.json``) is what
the app's browser preview is held to: the same recipe over the same rows must give the same
output, row ids, exclusions and counts in both engines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from strata_forge.datasets.recipe import (
    ConcatStep,
    GenerationResult,
    Recipe,
    RecipeError,
    Table,
    diff_tables,
    execute,
    profile_table,
    recipe_sha256,
    render_prompt,
    split_class,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from strata_forge.datasets.recipe import GenerationRequest, Source, SynthesizeStep

_SHA = "0123456789abcdef0123456789abcdef01234567"
_GOLDEN = Path(__file__).with_name("recipe_golden.json")

_TICKETS = [
    {"ticket": 1, "text": "Refund for a double charge", "lang": "en", "priority": 3},
    {"ticket": 2, "text": "refund for a  double charge", "lang": "en", "priority": 2},
    {"ticket": 3, "text": "Cambio de dirección", "lang": "es", "priority": None},
    {"ticket": 4, "text": "Login loop after reset", "lang": "en", "priority": 5},
    {"ticket": 5, "text": "Warranty outside the window", "lang": "en", "priority": 1},
    {"ticket": 6, "text": "Remove my card", "lang": "de", "priority": 4},
]
_LABELS = [
    {"ticket": 1, "label": "billing"},
    {"ticket": 4, "label": "account"},
    {"ticket": 4, "label": "security"},
    {"ticket": 5, "label": "warranty"},
]
_EXTRA = [{"ticket": 7, "text": "Change my plan", "lang": "en", "priority": 2}]
_HELD = [{"ticket": 9, "text": "Held-out case", "lang": "en", "priority": 1}]
_ROWS: dict[str, list[dict[str, Any]]] = {
    "tickets": _TICKETS,
    "labels": _LABELS,
    "extra": _EXTRA,
    "held": _HELD,
}


def _source(name: str, split: str = "train", **kw: Any) -> dict[str, Any]:
    return {"name": name, "hf_repo_id": f"org/{name}", "revision": _SHA, "split": split, **kw}


def _read(source: Source) -> list[dict[str, Any]]:
    return _ROWS[source.name]


def _recipe(*steps: dict[str, Any], sources: list[dict[str, Any]] | None = None) -> Recipe:
    return Recipe.model_validate(
        {
            "name": "support-edge-cases",
            "sources": sources
            or [
                _source("tickets"),
                _source("labels"),
                _source("extra"),
                _source("held", split="validation"),
            ],
            "steps": [{"kind": "load", "source": "tickets"}, *steps],
        }
    )


# ------------------------------ the spec ------------------------------


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            {"steps": [{"kind": "filter", "conditions": [{"column": "a", "op": "is_null"}]}]},
            "first step loads",
        ),
        (
            {
                "steps": [
                    {"kind": "load", "source": "tickets"},
                    {"kind": "load", "source": "tickets"},
                ]
            },
            "only the first step loads",
        ),
        ({"steps": [{"kind": "load", "source": "nope"}]}, "does not declare"),
        (
            {
                "steps": [
                    {"kind": "load", "source": "tickets"},
                    {"kind": "split", "ratios": {"train": 1.0}, "seed": 1},
                    {"kind": "dedupe", "keys": ["ticket"]},
                ]
            },
            "must be the last step",
        ),
        (
            {
                "steps": [
                    {"kind": "load", "source": "tickets"},
                    {"kind": "split", "ratios": {"train": 0.5, "test": 0.4}, "seed": 1},
                ]
            },
            "sum to 1",
        ),
        (
            {
                "steps": [
                    {"kind": "load", "source": "tickets"},
                    {"kind": "filter", "conditions": [{"column": "a", "op": "in", "value": 1}]},
                ]
            },
            "takes a list",
        ),
        (
            {
                "sources": [_source("tickets", revision="main")],
                "steps": [{"kind": "load", "source": "tickets"}],
            },
            "revision",
        ),
        (
            {
                "sources": [_source("tickets"), _source("tickets")],
                "steps": [{"kind": "load", "source": "tickets"}],
            },
            "unique",
        ),
    ],
)
def test_the_spec_refuses_a_recipe_that_cannot_run(data: dict[str, Any], message: str) -> None:
    base = {"name": "r", "sources": [_source("tickets")]}
    with pytest.raises(ValidationError, match=message):
        Recipe.model_validate(base | data)


def test_the_revision_hash_is_the_content_and_only_the_content() -> None:
    a = _recipe({"kind": "dedupe", "keys": ["ticket"]})
    b = Recipe.model_validate(json.loads(json.dumps(a.canonical())))
    assert recipe_sha256(a) == recipe_sha256(b)
    assert recipe_sha256(a) != recipe_sha256(_recipe({"kind": "dedupe", "keys": ["text"]}))
    assert a.canonical()["schema"] == "strata.dataset_recipe/v1"


# ------------------------------ steps ------------------------------


def test_load_gives_stable_source_row_ids_and_reads_only_declared_columns() -> None:
    recipe = _recipe(sources=[_source("tickets", columns=["ticket", "text"])])
    out = execute(recipe, _read)
    (table,) = out.splits.values()
    assert table.columns == ["ticket", "text"]
    assert [r.id for r in table.rows] == [f"tickets:{i}" for i in range(6)]
    assert list(out.splits) == ["train"]
    record = table.records()[0]
    assert record == {
        "ticket": 1,
        "text": "Refund for a double charge",
        "_row_id": "tickets:0",
        "_synthetic": False,
        "_seed_row_id": None,
    }


def test_filter_is_sql_like_about_nulls_and_records_what_it_dropped() -> None:
    out = execute(
        _recipe(
            {
                "kind": "filter",
                "conditions": [
                    {"column": "lang", "op": "eq", "value": "en"},
                    {"column": "priority", "op": "ge", "value": 2},
                ],
            }
        ),
        _read,
    )
    assert [r.values["ticket"] for r in out.splits["train"].rows] == [1, 2, 4]
    assert {e.row_id for e in out.exclusions} == {"tickets:2", "tickets:4", "tickets:5"}
    assert all(e.reason == "filtered" for e in out.exclusions)
    nulls = execute(
        _recipe({"kind": "filter", "conditions": [{"column": "priority", "op": "ne", "value": 3}]}),
        _read,
    )
    # A null priority is neither equal nor unequal to 3.
    assert "tickets:2" not in {r.id for r in nulls.splits["train"].rows}


def test_map_columns_derives_selects_and_renames() -> None:
    out = execute(
        _recipe(
            {
                "kind": "map_columns",
                "templates": {"prompt": "[{lang}] {text}"},
                "constants": {"origin": "support"},
                "select": ["ticket", "prompt", "origin"],
                "rename": {"ticket": "id"},
            }
        ),
        _read,
    )
    table = out.splits["train"]
    assert table.columns == ["id", "prompt", "origin"]
    assert table.rows[0].values == {
        "id": 1,
        "prompt": "[en] Refund for a double charge",
        "origin": "support",
    }
    with pytest.raises(RecipeError, match="does not have"):
        execute(_recipe({"kind": "map_columns", "select": ["missing"]}), _read)
    with pytest.raises(RecipeError, match="engine-reserved"):
        execute(_recipe({"kind": "map_columns", "rename": {"text": "_row_id"}}), _read)


def test_join_matches_keys_like_sql_and_records_unmatched_rows() -> None:
    inner = execute(_recipe({"kind": "join", "source": "labels", "on": ["ticket"]}), _read)
    rows = inner.splits["train"].rows
    assert [(r.id, r.values["label"]) for r in rows] == [
        ("tickets:0+labels:0", "billing"),
        ("tickets:3+labels:1", "account"),
        ("tickets:3+labels:2", "security"),
        ("tickets:4+labels:3", "warranty"),
    ]
    assert {e.reason for e in inner.exclusions} == {"no_join_match"}
    left = execute(
        _recipe({"kind": "join", "source": "labels", "on": ["ticket"], "how": "left"}), _read
    )
    assert len(left.splits["train"].rows) == 7
    assert left.splits["train"].rows[1].values["label"] is None
    assert not left.exclusions


def test_concat_appends_and_dedupe_keeps_the_first_of_each_key() -> None:
    out = execute(
        _recipe(
            {"kind": "concat", "source": "extra"},
            {"kind": "dedupe", "keys": ["text"], "normalize": "casefold_whitespace"},
        ),
        _read,
    )
    ids = [r.id for r in out.splits["train"].rows]
    assert ids == ["tickets:0", "tickets:2", "tickets:3", "tickets:4", "tickets:5", "extra:0"]
    assert [(e.row_id, e.reason) for e in out.exclusions] == [
        ("tickets:1", "duplicate_of:tickets:0")
    ]
    exact = execute(_recipe({"kind": "dedupe", "keys": ["text"]}), _read)
    assert not exact.exclusions


def test_split_is_seeded_exact_and_stratified() -> None:
    step = {"kind": "split", "ratios": {"train": 0.5, "test": 0.5}, "seed": 7}
    a = execute(_recipe(step), _read)
    b = execute(_recipe(step), _read)
    assert {n: [r.id for r in t.rows] for n, t in a.splits.items()} == {
        n: [r.id for r in t.rows] for n, t in b.splits.items()
    }
    assert [len(t.rows) for t in a.splits.values()] == [3, 3]
    strat = execute(_recipe(step | {"stratify_by": "lang"}), _read)
    en = [r for t in strat.splits.values() for r in t.rows if r.values["lang"] == "en"]
    assert sorted(r.split or "" for r in en).count("train") == 2  # 4 English rows: 2 and 2


def test_row_cap_refuses_rather_than_truncates() -> None:
    with pytest.raises(RecipeError, match="more than 5 rows"):
        execute(_recipe(), _read, row_cap=5)


# ------------------------------ synthesis ------------------------------


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"customer": {"type": "string"}, "response": {"type": "string"}},
        "required": ["customer", "response"],
    }


def _synth(**kw: Any) -> dict[str, Any]:
    return {
        "kind": "synthesize",
        "model_id": "org/generator",
        "model_revision": _SHA,
        "prompt_template": "Write a variation of: {text}",
        "output_schema": _schema(),
        "output_columns": {"customer": "customer", "response": "response"},
        "keep_columns": ["lang"],
        "n": 2,
        "seed": 42,
        "max_seed_rows": 2,
    } | kw


class _Fake:
    def __init__(self, bad: set[int] | None = None) -> None:
        self.calls: list[tuple[SynthesizeStep, list[GenerationRequest]]] = []
        self.bad = bad or set()

    def __call__(
        self, step: SynthesizeStep, requests: Sequence[GenerationRequest]
    ) -> list[GenerationResult]:
        self.calls.append((step, list(requests)))
        out: list[GenerationResult] = []
        for i, r in enumerate(requests):
            if i in self.bad:
                out.append(
                    GenerationResult(text=None, error="timeout")
                    if i == 0
                    else GenerationResult("not json")
                )
            else:
                out.append(
                    GenerationResult(
                        json.dumps({"customer": r.prompt[-12:], "response": f"s{r.seed}"})
                    )
                )
        return out


def test_synthesis_marks_rows_synthetic_with_seed_ids_and_per_request_seeds() -> None:
    fake = _Fake()
    out = execute(_recipe(_synth()), _read, generate=fake)
    (_, requests) = fake.calls[0]
    assert [(r.seed_row_id, r.sample_index, r.seed) for r in requests] == [
        ("tickets:0", 0, 42),
        ("tickets:0", 1, 43),
        ("tickets:1", 0, 44),
        ("tickets:1", 1, 45),
    ]
    assert requests[0].prompt == "Write a variation of: Refund for a double charge"
    table = out.splits["train"]
    assert table.columns == ["lang", "customer", "response"]
    assert [r.id for r in table.rows] == [
        "gen:tickets:0:0",
        "gen:tickets:0:1",
        "gen:tickets:1:0",
        "gen:tickets:1:1",
    ]
    assert all(r.synthetic for r in table.rows)
    record = table.records()[0]
    assert record["_synthetic"] is True
    assert record["_seed_row_id"] == "tickets:0"
    assert out.synthetic_rows == 4
    assert out.counts[-1].generated == 4


def test_failed_and_invalid_generations_are_excluded_with_reasons() -> None:
    out = execute(_recipe(_synth()), _read, generate=_Fake(bad={0, 3}))
    assert [(e.row_id, e.reason) for e in out.exclusions] == [
        ("gen:tickets:0:0", "generation_failed"),
        ("gen:tickets:1:1", "invalid_output"),
    ]
    assert len(out.splits["train"].rows) == 2


def test_a_generator_may_reject_an_output_itself() -> None:
    def _reject(
        step: SynthesizeStep, requests: Sequence[GenerationRequest]
    ) -> list[GenerationResult]:
        return [GenerationResult(text=None, error="invalid_output") for _ in requests]

    out = execute(_recipe(_synth(n=1, max_seed_rows=1)), _read, generate=_reject)
    assert [e.reason for e in out.exclusions] == ["invalid_output"]


def test_append_keeps_the_seed_rows_before_the_generated_ones() -> None:
    out = execute(_recipe(_synth(mode="append", n=1, max_seed_rows=1)), _read, generate=_Fake())
    table = out.splits["train"]
    assert len(table.rows) == 7
    assert table.rows[-1].synthetic
    assert not table.rows[0].synthetic


def test_synthesis_never_reads_held_out_rows_and_needs_a_generator() -> None:
    with pytest.raises(ValidationError, match="training splits only; held"):
        _recipe({"kind": "concat", "source": "held"}, _synth(max_seed_rows=10))
    # The engine re-checks every seed row's origin even for a recipe built without validation.
    valid = _recipe(_synth(max_seed_rows=10))
    forged = valid.model_copy(
        update={"steps": (valid.steps[0], ConcatStep(source="held"), *valid.steps[1:])}
    )
    with pytest.raises(RecipeError, match="held-out data \\(held\\)"):
        execute(forged, _read, generate=_Fake())
    with pytest.raises(RecipeError, match="no generator"):
        execute(_recipe(_synth()), _read)
    with pytest.raises(ValidationError, match="does not declare"):
        _recipe(_synth(output_columns={"x": "nope"}))


def test_render_prompt_refuses_an_unknown_column() -> None:
    assert render_prompt("{a} and {b}", {"a": "x", "b": [1, 2]}) == "x and [1, 2]"
    with pytest.raises(RecipeError, match="'c'"):
        render_prompt("{c}", {"a": 1})


# ------------------------------ review ------------------------------


def test_profile_and_diff_describe_the_output() -> None:
    before = execute(_recipe(), _read).splits["train"]
    after = execute(
        _recipe(
            {"kind": "dedupe", "keys": ["lang"]}, {"kind": "map_columns", "constants": {"k": 1}}
        ),
        _read,
    )
    table = after.splits["train"]
    profile = {p["column"]: p for p in profile_table(before)}
    assert profile["priority"]["nulls"] == 1
    assert profile["priority"]["min"] == 1
    assert profile["priority"]["max"] == 5
    assert profile["lang"]["top"][0] == {"value": "en", "count": 4}
    delta = diff_tables(before, table)
    assert delta["removed"] == 3
    assert delta["added"] == 0
    assert delta["columns_added"] == ["k"]


def test_an_empty_table_profiles_as_empty() -> None:
    assert profile_table(Table([], ["a"]))[0]["rows"] == 0


# ------------------------------ golden ------------------------------


def _golden_case() -> dict[str, Any]:
    recipe = _recipe(
        {"kind": "concat", "source": "extra"},
        {
            "kind": "filter",
            "match": "any",
            "conditions": [{"column": "lang", "op": "in", "value": ["en", "es"]}],
        },
        {"kind": "join", "source": "labels", "on": ["ticket"], "how": "left"},
        {"kind": "map_columns", "templates": {"prompt": "{text} ({label})"}, "constants": {"v": 2}},
        {"kind": "dedupe", "keys": ["text"], "normalize": "casefold_whitespace"},
        {"kind": "split", "ratios": {"train": 0.6, "test": 0.4}, "seed": 11, "stratify_by": "lang"},
    )
    out = execute(recipe, _read)
    return {
        "recipe": recipe.canonical(),
        "rows": {k: v for k, v in _ROWS.items() if k != "held"},
        "expected": {
            "splits": {name: t.records() for name, t in out.splits.items()},
            "columns": {name: t.columns for name, t in out.splits.items()},
            "exclusions": [[e.row_id, e.step, e.reason] for e in out.exclusions],
            "counts": [[c.step, c.kind, c.rows_in, c.rows_out, c.excluded] for c in out.counts],
        },
    }


def test_the_golden_fixture_is_what_the_engine_produces() -> None:
    """Regenerate with ``STRATA_WRITE_GOLDEN=1``; the app's preview tests read the same file."""
    import os

    case = _golden_case()
    if os.environ.get("STRATA_WRITE_GOLDEN") == "1":
        _GOLDEN.write_text(json.dumps(case, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    stored: Mapping[str, Any] = json.loads(_GOLDEN.read_text())
    assert stored == json.loads(json.dumps(case, ensure_ascii=False))


# ------------------------------------------------------------------ held-out data, bounds


@pytest.mark.parametrize(
    "split",
    [
        "train+test",
        "train[:10]+validation",
        "train[:10%]",
        "train[-5:]",
        "train(closest)",
        " train",
        "train[9:3]",
    ],
)
def test_a_source_reads_exactly_one_split(split: str) -> None:
    with pytest.raises(ValidationError, match="one split name"):
        _recipe(sources=[_source("tickets", split=split)])


@pytest.mark.parametrize("split", ["train", "train[0:100]", "train[:5]", "train_v2"])
def test_a_plain_or_row_sliced_split_is_accepted(split: str) -> None:
    assert _recipe(sources=[_source("tickets", split=split)]).sources[0].split == split


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("train", "train"),
        ("training", "train"),
        ("train_v2", "train"),
        ("train2024", "train"),
        ("test", "held_out"),
        ("validation", "held_out"),
        ("val", "held_out"),
        ("dev", "held_out"),
        ("devSet", "held_out"),
        ("eval_v2", "held_out"),
        ("private_test", "held_out"),
        ("public_test", "held_out"),
        ("held_out", "held_out"),
        ("hold-out", "held_out"),
        ("heldOut", "held_out"),
        ("testset", "held_out"),
        ("train_test", "held_out"),
        ("default", "ambiguous"),
        ("full", "ambiguous"),
        ("train_extra", "ambiguous"),
    ],
)
def test_split_names_are_classified_by_their_tokens(name: str, expected: str) -> None:
    assert split_class(name) == expected


@pytest.mark.parametrize("split", ["private_test", "held_out", "default", "test[0:5]"])
def test_a_generating_recipe_refuses_any_split_that_is_not_training(split: str) -> None:
    with pytest.raises(ValidationError, match="training splits only"):
        _recipe(_synth(), sources=[_source("tickets", split=split)])


def test_held_out_rows_keep_a_held_out_name_without_generation() -> None:
    held = [_source("tickets", split="validation")]
    with pytest.raises(ValidationError, match=r"would relabel it"):
        _recipe(sources=held)  # the default output split is `train`
    with pytest.raises(ValidationError, match=r"would relabel it"):
        _recipe({"kind": "split", "ratios": {"train": 0.5, "test": 0.5}, "seed": 1}, sources=held)
    kept = Recipe.model_validate(
        {
            "name": "relabel-check",
            "sources": held,
            "steps": [{"kind": "load", "source": "tickets"}],
            "default_split": "validation",
        }
    )
    assert kept.holdout_sources == ("tickets",)
    assert _recipe({"kind": "split", "ratios": {"test": 0.5, "dev": 0.5}, "seed": 1}, sources=held)


def test_a_declared_but_unread_held_out_source_does_not_mark_the_output() -> None:
    recipe = _recipe(_synth())  # declares `held` (validation) and never reads it
    assert recipe.used_sources == ("tickets",)
    assert recipe.holdout_sources == ()


def test_the_byte_cap_stops_a_source_while_it_loads() -> None:
    with pytest.raises(RecipeError, match="MiB of values"):
        execute(_recipe(), lambda _s: [{**_TICKETS[0], "text": "x" * 4096}] * 600, byte_cap=1024**2)


def test_the_row_cap_stops_a_join_while_it_grows() -> None:
    left = [{"ticket": 1, "text": "a", "lang": "en", "priority": 1}] * 40
    right = [{"ticket": 1, "label": "l"}] * 40
    rows = {"tickets": left, "labels": right}
    recipe = _recipe({"kind": "join", "source": "labels", "on": ["ticket"]})
    with pytest.raises(RecipeError, match="step 2 holds more than 100 rows"):
        execute(recipe, lambda s: rows[s.name], row_cap=100)


def test_text_in_a_recipe_is_bounded() -> None:
    with pytest.raises(ValidationError):
        _recipe({"kind": "map_columns", "templates": {"t": "x" * 20_001}})
    with pytest.raises(ValidationError):
        _recipe({"kind": "map_columns", "constants": {"t": "x" * 10_001}})
    with pytest.raises(ValidationError, match="at most 1,000 values"):
        _recipe(
            {
                "kind": "filter",
                "conditions": [{"column": "lang", "op": "in", "value": ["en"] * 1001}],
            }
        )
    with pytest.raises(ValidationError, match="KiB"):
        _recipe(
            *[
                {"kind": "map_columns", "templates": {f"c{i}": "x" * 19_000 for i in range(16)}}
                for _ in range(2)
            ]
        )


def test_an_output_schema_the_engine_cannot_guide_is_refused_in_the_recipe() -> None:
    with pytest.raises(ValidationError, match="output_schema"):
        _recipe(
            _synth(
                output_schema={
                    "type": "object",
                    "properties": {
                        "customer": {"type": "nonsense"},
                        "response": {"type": "string"},
                    },
                }
            )
        )


def test_a_dataset_version_reference_is_a_record_id() -> None:
    with pytest.raises(ValidationError):
        _recipe(sources=[_source("tickets", dataset_version="../etc")])
    assert _recipe(sources=[_source("tickets", dataset_version="a" * 32)])
