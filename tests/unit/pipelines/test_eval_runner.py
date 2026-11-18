"""The VM-side evaluation runner, end to end over synthetic pinned files.

The Hub client and ``datasets`` are replaced by local stand-ins; the spec loading, join,
metrics, bootstrap, report, items file and manifest are the runner's own code.
"""

from __future__ import annotations

import json
import random
import sys
import types
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from strata_forge.metrics.classification import classification_report
from strata_forge.metrics.cluster_bootstrap import cluster_bootstrap, paired_cluster_bootstrap
from strata_forge.pipelines import eval_runner as er
from strata_forge.pipelines._common import LEGACY_TOKEN_ENV

if TYPE_CHECKING:
    from pathlib import Path

SHA_GOLD = "a" * 40
SHA_CAND = "b" * 40
SHA_BASE = "c" * 40
LABELS = ["entailment", "contradiction", "neutral"]
_TOKEN = "hf_secretwritetoken1234567890"


def _gold(n: int = 60) -> dict[str, list[Any]]:
    rng = random.Random(1)  # noqa: S311 — synthetic fixture.
    return {
        "item_id": [f"it-{i:03d}" for i in range(n)],
        "label": [rng.choice(LABELS) for _ in range(n)],
        "base_problem_id": [f"bp-{i // 3:03d}" for i in range(n)],
        "tier": ["easy" if i % 4 else "hard" for i in range(n)],
        "tags": [["a", "b"] if i % 2 else ["a"] for i in range(n)],
        "refused": [i == 0 for i in range(n)],
        "premises": ["synthetic text that must never reach the report"] * n,
    }


def _predictions(gold: dict[str, list[Any]], *, accuracy: float, seed: int) -> dict[str, list[Any]]:
    rng = random.Random(seed)  # noqa: S311 — synthetic fixture.
    preds = [
        label if rng.random() < accuracy else rng.choice([x for x in LABELS if x != label])
        for label in gold["label"]
    ]
    return {"item_id": list(gold["item_id"]), "pred": preds, "overflow": [False] * len(preds)}


class _Split:
    """The slice of ``datasets.Dataset`` the runner uses."""

    def __init__(self, columns: dict[str, list[Any]]) -> None:
        self._columns = columns
        self.column_names = list(columns)

    def __len__(self) -> int:
        return len(next(iter(self._columns.values())))

    def select_columns(self, names: list[str]) -> _Split:
        return _Split({name: list(self._columns[name]) for name in names})

    def to_dict(self) -> dict[str, list[Any]]:
        return self._columns


class _Hub:
    """Stand-in for HFHubClient: serves local parquet files, records pushes."""

    files: ClassVar[dict[tuple[str, str], Path]] = {}
    named: ClassVar[dict[tuple[str, str, str], Path]] = {}
    calls: ClassVar[list[tuple[str, Any]]] = []
    private: ClassVar[bool] = True

    def __init__(self, token: Any = None) -> None:
        _Hub.calls.append(("init", token))

    async def download_file(self, repo_id: str, filename: str, **kwargs: Any) -> Path:
        _Hub.calls.append(("download_file", (repo_id, filename, kwargs)))
        by_name = _Hub.named.get((repo_id, kwargs["revision"], filename))
        return by_name if by_name is not None else _Hub.files[(repo_id, kwargs["revision"])]

    async def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        _Hub.calls.append(("create_repo", (repo_id, kwargs)))

    async def is_private(self, repo_id: str, **kwargs: Any) -> bool:
        del repo_id, kwargs
        return _Hub.private

    async def upload_files(self, files: Any, repo_id: str, **kwargs: Any) -> str:
        _Hub.calls.append(("upload_files", (dict(files), repo_id, kwargs)))
        return "e" * 40


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    monkeypatch.delenv("FORGE_SECRETS_FILE", raising=False)
    monkeypatch.delenv(LEGACY_TOKEN_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    gold = _gold()
    state: dict[str, Any] = {"gold": gold, "load_calls": [], "tmp": tmp_path}

    def _load_dataset(*args: Any, **kwargs: Any) -> _Split:
        state["load_calls"].append((args, kwargs))
        return _Split(state["gold"])

    fake = types.ModuleType("datasets")
    fake.load_dataset = _load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)
    _Hub.files, _Hub.named, _Hub.calls, _Hub.private = {}, {}, [], True
    monkeypatch.setattr(er, "HFHubClient", _Hub)
    state["cand"] = _predictions(gold, accuracy=0.85, seed=2)
    state["base"] = _predictions(gold, accuracy=0.6, seed=3)
    return state


def _write_predictions(state: dict[str, Any]) -> None:
    for name, repo, sha in (("cand", "u/cand", SHA_CAND), ("base", "u/base", SHA_BASE)):
        path = state["tmp"] / f"{name}.parquet"
        pq.write_table(pa.Table.from_pydict(state[name]), str(path))
        _Hub.files[(repo, sha)] = path


def _settings(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "labels": LABELS,
        "predictions": {"name": "candidate", "repo_id": "u/cand", "commit_sha": SHA_CAND},
        "compare_to": [{"name": "baseline", "repo_id": "u/base", "commit_sha": SHA_BASE}],
        "gold_column": "label",
        "cluster_column": "base_problem_id",
        "slice_columns": ["tier", "tags"],
        "seed": 20261104,
        "resamples": 200,
        "min_slice_rows": 20,
    }
    settings.update(overrides)
    return settings


def _spec(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "model_id": "u/model",
        "dataset_id": "u/gold",
        "dataset_commit_sha": SHA_GOLD,
        "split": "test",
        "evaluation": _settings(),
        "run_id": "run1",
    }
    spec.update(overrides)
    return spec


async def _run(
    monkeypatch: pytest.MonkeyPatch, state: dict[str, Any], spec: dict[str, Any]
) -> tuple[int, list[dict[str, Any]]]:
    _write_predictions(state)
    progress = state["tmp"] / "progress.jsonl"
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(spec))
    monkeypatch.setenv("FORGE_PROGRESS_PATH", str(progress))
    code = await er.main()
    events = [json.loads(line) for line in progress.read_text().splitlines()]
    return code, events


def _outdir(state: dict[str, Any]) -> Path:
    return state["tmp"] / "home" / "strata-eval-results" / "run1"


async def test_report_matches_the_metrics_library(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    code, events = await _run(monkeypatch, env, _spec())
    assert code == 0, events[-1]
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    assert report["schema"] == "strata.eval.report/v1"
    overall = report["slices"][0]
    assert (overall["column"], overall["value"]) == (None, None)
    gold, cand, base = env["gold"], env["cand"]["pred"], env["base"]["pred"]
    labels = gold["label"]
    expected = cluster_bootstrap(
        labels, cand, labels=LABELS, clusters=gold["base_problem_id"], seed=20261104, resamples=200
    )
    stable = {"undefined_share": 0.0, "unstable": False}
    assert overall["systems"]["candidate"]["macro_f1"] == {
        **expected.intervals.macro_f1.model_dump(mode="json"),
        **stable,
    }
    assert overall["systems"]["candidate"]["accuracy"] == {
        **expected.intervals.accuracy.model_dump(mode="json"),
        **stable,
    }
    paired = paired_cluster_bootstrap(
        labels,
        base,
        cand,
        labels=LABELS,
        clusters=gold["base_problem_id"],
        seed=20261104,
        resamples=200,
    )
    (comparison,) = overall["comparisons"]
    assert comparison["baseline"] == "baseline"
    delta = comparison["delta"]["macro_f1"]
    assert delta == {
        **paired.delta.macro_f1.model_dump(mode="json"),
        **stable,
        "excludes_zero": bool(delta["lower"] > 0 or delta["upper"] < 0),
    }
    point = classification_report(labels, cand, labels=LABELS)
    assert overall["systems"]["candidate"]["per_class"][0]["f1"] == point.per_class[0].f1
    assert overall["bootstrap"]["clusters"] == 20
    assert overall["n"] == 60
    end = events[-1]
    assert end["kind"] == "end"
    assert end["message"] == str(_outdir(env))
    assert end["metrics"]["macro_f1"] == expected.intervals.macro_f1.point
    assert end["metrics"]["delta_macro_f1_lower__baseline"] == paired.delta.macro_f1.lower
    assert len(end["metrics"]) <= 64
    _, kwargs = env["load_calls"][0]
    assert kwargs["revision"] == SHA_GOLD
    downloads = [c[1] for c in _Hub.calls if c[0] == "download_file"]
    assert {(d[0], d[2]["revision"]) for d in downloads} == {
        ("u/cand", SHA_CAND),
        ("u/base", SHA_BASE),
    }


async def test_slices_explode_lists_and_flag_small_ones(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    code, _ = await _run(monkeypatch, env, _spec())
    assert code == 0
    slices = json.loads((_outdir(env) / "metrics.json").read_text())["slices"]
    by_key = {(s["column"], s["value"]): s for s in slices}
    assert set(by_key) == {
        (None, None),
        ("tier", "easy"),
        ("tier", "hard"),
        ("tags", "a"),
        ("tags", "b"),
    }
    assert by_key["tags", "a"]["n"] == 60
    assert by_key["tags", "b"]["n"] == 30
    assert by_key["tier", "hard"]["n"] == 15
    assert by_key["tier", "hard"]["diagnostic"] is True
    assert by_key["tier", "easy"]["diagnostic"] is False
    assert "per_class" not in by_key["tier", "easy"]["systems"]["candidate"]


async def test_missing_overflow_and_unknown_predictions_count_as_wrong(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    gold = env["gold"]
    preds: list[str | None] = [*gold["label"][1:], "entailment"]
    overflow = [False] * 60
    preds[0] = "maybe"
    overflow[1] = True
    preds[2] = None
    env["cand"] = {"item_id": [*gold["item_id"][1:], "stray"], "pred": preds, "overflow": overflow}
    spec = _spec(
        evaluation=_settings(
            compare_to=[],
            predictions={
                "name": "candidate",
                "repo_id": "u/cand",
                "commit_sha": SHA_CAND,
                "overflow_column": "overflow",
            },
        )
    )
    code, _ = await _run(monkeypatch, env, spec)
    assert code == 0
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    counts = report["systems"][0]["counts"]
    assert counts == {"scored": 60, "missing": 1, "invalid": 2, "overflow": 1, "unmatched": 1}
    slice0 = report["slices"][0]
    # The overflowed row leaves the SHARED population; missing and failed rows never leave it.
    assert (slice0["n"], slice0["n_tier"], slice0["overflow_excluded"]) == (59, 60, 1)
    assert slice0["overflow_share"] == pytest.approx(1 / 60)
    overall = slice0["systems"]["candidate"]
    assert overall["accuracy"]["point"] == pytest.approx(56 / 59)
    assert overall["invalid_rate"]["point"] == pytest.approx(3 / 59)
    # The full-tier secondary read counts the system's own overflow as invalid.
    assert overall["full_tier"]["accuracy"]["point"] == pytest.approx(56 / 60)
    assert overall["full_tier"]["invalid_rate"]["point"] == pytest.approx(4 / 60)
    items = pq.read_table(str(_outdir(env) / "items.parquet")).to_pylist()
    assert items[0]["pred__candidate"] is None
    assert items[0]["correct__candidate"] is False
    assert [items[i]["status__candidate"] for i in range(4)] == [
        "missing",
        "missing",
        "overflow",
        "missing",
    ]
    assert "premises" not in items[0]


async def test_excluded_rows_leave_every_system(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    spec = _spec(evaluation=_settings(exclude_column="refused"))
    code, _ = await _run(monkeypatch, env, spec)
    assert code == 0
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    assert report["population"]["excluded"] == 1
    assert report["slices"][0]["n"] == 59
    assert all(system["counts"]["scored"] == 59 for system in report["systems"])
    items = pq.read_table(str(_outdir(env) / "items.parquet")).to_pylist()
    assert items[0]["excluded"] is True
    assert items[0]["correct__baseline"] is None


async def test_aliases_map_both_sides(monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]) -> None:
    short = {"entailment": "E", "contradiction": "C", "neutral": "N"}
    env["gold"]["label"] = [short[x] for x in env["gold"]["label"]]
    env["cand"]["pred"] = [x.upper() for x in env["cand"]["pred"]]
    spec = _spec(
        evaluation=_settings(
            compare_to=[],
            gold_aliases={v: k for k, v in short.items()},
            predictions={
                "name": "candidate",
                "repo_id": "u/cand",
                "commit_sha": SHA_CAND,
                "label_aliases": {k.upper(): k for k in LABELS},
            },
        )
    )
    code, _ = await _run(monkeypatch, env, spec)
    assert code == 0
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    assert report["systems"][0]["counts"]["invalid"] == 0


_MALFORMED: dict[str, tuple[str, str, int | None, str | None, str]] = {
    "duplicate gold id": ("gold", "item_id", 1, "it-000", "repeats 1 item id"),
    "unknown gold label": ("gold", "label", 0, "other", "outside the declared labels"),
    "duplicate prediction id": ("cand", "item_id", 1, "it-000", "repeat an item id"),
    "no cluster column": (
        "gold",
        "base_problem_id",
        None,
        None,
        "no column(s): ['base_problem_id']",
    ),
    "no prediction column": (
        "cand",
        "pred",
        None,
        None,
        "prediction file has no column(s): ['pred']",
    ),
}


@pytest.mark.parametrize("case", sorted(_MALFORMED))
async def test_malformed_inputs_refuse(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any], case: str
) -> None:
    source, column, index, value, needle = _MALFORMED[case]
    table: dict[str, list[Any]] = env[source]
    if index is None:
        del table[column]
    else:
        table[column][index] = value
    code, events = await _run(monkeypatch, env, _spec())
    assert code == 1
    assert events[-1]["kind"] == "error"
    assert needle in events[-1]["message"]
    assert not (_outdir(env) / "metrics.json").exists()


@pytest.mark.parametrize(
    ("override", "needle"),
    [
        ({"dataset_commit_sha": "main"}, "dataset_commit_sha"),
        (
            {
                "evaluation": _settings(
                    predictions={"name": "c", "repo_id": "u/c", "commit_sha": "main"}
                )
            },
            "commit_sha",
        ),
        (
            {
                "evaluation": _settings(
                    predictions={
                        "name": "c",
                        "repo_id": "u/c",
                        "commit_sha": SHA_CAND,
                        "path": "../x.parquet",
                    }
                )
            },
            "path",
        ),
        (
            {
                "evaluation": _settings(
                    compare_to=[{"name": "candidate", "repo_id": "u/b", "commit_sha": SHA_BASE}]
                )
            },
            "distinct name",
        ),
        ({"evaluation": _settings(gold_aliases={"x": "maybe"})}, "undeclared labels"),
        ({"evaluation": _settings(labels=["a", "a"])}, "unique"),
        ({"evaluation": _settings(seed=-1)}, "seed"),
        (
            {"evaluation": _settings(tier_rule={"default": {"max_premises": 20}})},
            "needs a premises_column",
        ),
        (
            {"evaluation": _settings(tier_rule={"default": {"max_tokens": 9}})},
            "needs a token_column",
        ),
        (
            {"evaluation": _settings(tier_rule={"token_column": "n_tokens"})},
            "declared together",
        ),
        ({"evaluation": {**_settings(), "sealed": True}}, "sealed"),
    ],
)
async def test_spec_refusals(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any], override: dict[str, Any], needle: str
) -> None:
    code, events = await _run(monkeypatch, env, _spec(**override))
    assert code == 1
    assert needle in events[-1]["message"]
    assert env["load_calls"] == []


async def test_push_is_one_private_commit(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    secrets = env["tmp"] / ".secrets.json"
    secrets.write_text(json.dumps({"HF_TOKEN": _TOKEN}))
    secrets.chmod(0o600)
    monkeypatch.setenv("FORGE_SECRETS_FILE", str(secrets))
    code, events = await _run(monkeypatch, env, _spec(output_repo_id="u/report"))
    assert code == 0
    uploads = [c[1] for c in _Hub.calls if c[0] == "upload_files"]
    assert len(uploads) == 1
    files, repo, kwargs = uploads[0]
    assert repo == "u/report"
    assert set(files) == {"items.parquet", "metrics.json", "manifest.json", "README.md"}
    assert kwargs["repo_type"] == "dataset"
    # The commit this run made, so the report is read back there and never at a moved head.
    assert events[-1]["message"] == "u/report@" + "e" * 40
    assert all(_TOKEN not in json.dumps(event) for event in events)
    manifest = json.loads((_outdir(env) / "manifest.json").read_text())
    assert manifest["kind"] == "evaluation"
    assert manifest["dataset"]["commit"] == SHA_GOLD
    assert {p["commit"] for p in manifest["predictions"]} == {SHA_CAND, SHA_BASE}
    card = (_outdir(env) / "README.md").read_text()
    assert "data_files: items.parquet" in card


async def test_public_output_repo_refuses_before_reading(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    secrets = env["tmp"] / ".secrets.json"
    secrets.write_text(json.dumps({"HF_TOKEN": _TOKEN}))
    secrets.chmod(0o600)
    monkeypatch.setenv("FORGE_SECRETS_FILE", str(secrets))
    _Hub.private = False
    code, events = await _run(monkeypatch, env, _spec(output_repo_id="u/report"))
    assert code == 1
    assert "is public" in events[-1]["message"]
    assert env["load_calls"] == []


async def test_a_tier_rule_refuses_rows_for_every_system(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    env["gold"]["premises"] = [
        [{"pid": f"p{k}", "text": "p"} for k in range(25 if i < 5 else 3)] for i in range(60)
    ]
    env["gold"]["tokens"] = [3000 if i == 10 else 100 for i in range(60)]
    env["gold"]["tokens_method"] = ["tokenizer:org/tok@abc"] * 60
    rule = {
        "tier_column": "tier",
        "limits": {"hard": {"max_premises": 20}},
        "default": {"max_premises": 20, "max_tokens": 2048},
        "token_column": "tokens",
        "token_method_column": "tokens_method",
    }
    spec = _spec(evaluation=_settings(premises_column="premises", tier_rule=rule))
    code, events = await _run(monkeypatch, env, spec)
    assert code == 0, events[-1]
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    assert report["population"]["tier_refused"] == 6
    assert report["slices"][0]["n_tier"] == 54
    assert all(system["counts"]["scored"] == 54 for system in report["systems"])
    items = pq.read_table(str(_outdir(env) / "items.parquet")).to_pylist()
    assert items[0]["status__baseline"] == "tier_refused"
    assert items[0]["correct__candidate"] is None


async def test_rows_without_a_cluster_cluster_on_their_premises(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    env["gold"]["base_problem_id"] = [None] * 60
    env["gold"]["premises"] = [
        [{"pid": "p1", "text": f"fact  {i // 4}"}, {"pid": "p2", "text": "rule"}] for i in range(60)
    ]
    spec = _spec(evaluation=_settings(premises_column="premises"))
    code, _ = await _run(monkeypatch, env, spec)
    assert code == 0
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    assert report["slices"][0]["clusters"] == 15
    items = pq.read_table(str(_outdir(env) / "items.parquet")).to_pylist()
    assert items[0]["cluster"] == er.premise_cluster_key(
        [{"pid": "a", "text": "rule"}, {"pid": "b", "text": "fact 0"}]
    )


def test_the_premise_key_normalises_order_spacing_and_form() -> None:
    import hashlib
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "café")

    def p(*texts: str) -> list[dict[str, str]]:
        return [{"pid": f"p{i}", "text": t} for i, t in enumerate(texts)]

    key = er.premise_cluster_key(p(f"  {decomposed}\n", "a  b"))
    assert key == er.premise_cluster_key(p("a b", "café"))
    expected = hashlib.sha256('["a b","café"]'.encode()).hexdigest()
    assert key == f"premises:sha256:{expected}"
    assert er.premise_cluster_key(p("A")) != er.premise_cluster_key(p("a"))
    # Premises are {pid, text} objects; a plain string refuses.
    with pytest.raises(er.RunError, match="pid, text"):
        er.premise_cluster_key(["a b"])


async def test_comparisons_report_the_paired_discordance_rate(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    code, events = await _run(monkeypatch, env, _spec())
    assert code == 0, events[-1]
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    (comparison,) = report["slices"][0]["comparisons"]
    gold, cand, base = env["gold"]["label"], env["cand"]["pred"], env["base"]["pred"]
    b = sum(1 for g, c, x in zip(gold, cand, base, strict=True) if x == g and c != g)
    c = sum(1 for g, cc, x in zip(gold, cand, base, strict=True) if cc == g and x != g)
    assert comparison["discordance"] == {
        "n": 60,
        "baseline_only_correct": b,
        "candidate_only_correct": c,
        "p_d": (b + c) / 60,
        "n_population": 60,
        "coverage": 1.0,
        "gate": "resolved",
    }
    manifest = json.loads((_outdir(env) / "manifest.json").read_text())
    bootstrap = manifest["bootstrap"]
    assert bootstrap["bootstrap_version"] == "cluster-percentile-bootstrap-v1"
    assert bootstrap["clusters"] == 20
    assert bootstrap["numpy_version"]
    macro = report["slices"][0]["systems"]["candidate"]["macro_f1"]
    assert "undefined_resamples" in macro


@pytest.mark.parametrize("flag", [1, "true", None])
async def test_a_non_boolean_overflow_flag_refuses(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any], flag: object
) -> None:
    # Parquet holds one type per column, so the whole column is the non-boolean type (or null).
    env["cand"]["overflow"] = [flag] * len(env["cand"]["item_id"])
    cand = {
        "name": "candidate",
        "repo_id": "u/cand",
        "commit_sha": SHA_CAND,
        "overflow_column": "overflow",
    }
    code, events = await _run(monkeypatch, env, _spec(evaluation=_settings(predictions=cand)))
    assert code == 1
    assert "must be boolean" in events[-1]["message"]


async def test_a_repeated_list_slice_value_counts_the_row_once(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    env["gold"]["tags"] = [["a", "a"] if i < 10 else ["b"] for i in range(60)]
    spec = _spec(evaluation=_settings(slice_columns=["tags"]))
    code, _ = await _run(monkeypatch, env, spec)
    assert code == 0
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    tagged = next(s for s in report["slices"] if s["value"] == "a")
    assert tagged["n"] == 10


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("item_id", [1, "2"]),
        ("tier", [True, "True"]),
        ("base_problem_id", ["p", 7]),
    ],
)
async def test_an_identifier_column_mixing_types_refuses(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any], column: str, values: list[object]
) -> None:
    env["gold"][column] = [*values, *env["gold"][column][2:]]
    code, events = await _run(monkeypatch, env, _spec())
    assert code == 1
    assert "mixes value types" in events[-1]["message"]


async def test_an_identifier_column_holding_text_refuses(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    env["gold"]["tier"] = ["x" * 257, *env["gold"]["tier"][1:]]
    code, events = await _run(monkeypatch, env, _spec())
    assert code == 1
    assert "short identifiers" in events[-1]["message"]


async def test_an_undefined_macro_f1_leaves_its_bounds_out_of_the_end_event(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    # "neutral" is declared but appears in no gold row and no prediction: macro-F1 has no value,
    # so the end event says it is undefined and carries no point or bounds for it.
    for table, column in ((env["gold"], "label"), (env["cand"], "pred"), (env["base"], "pred")):
        table[column] = ["entailment" if v == "neutral" else v for v in table[column]]
    code, events = await _run(monkeypatch, env, _spec())
    assert code == 0, events[-1]
    metrics = events[-1]["metrics"]
    assert "macro_f1" not in metrics
    assert "macro_f1_lower" not in metrics
    assert metrics["macro_f1_undefined"] == 1.0
    assert metrics["delta_macro_f1_undefined__baseline"] == 1.0
    assert metrics["macro_f1_undefined_share"] == 1.0
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    overall = report["slices"][0]
    assert overall["absent_classes"] == ["neutral"]
    macro = overall["systems"]["candidate"]["macro_f1"]
    assert (macro["point"], macro["reason"], macro["absent_classes"]) == (
        None,
        "absent_gold_class:neutral",
        ["neutral"],
    )
    assert overall["systems"]["baseline"]["macro_f1"]["reason"] == "absent_gold_class:neutral"
    assert overall["systems"]["candidate"]["absent_class_predictions"] == {"neutral": 0}
    assert overall["comparisons"][0]["delta"]["macro_f1"]["excludes_zero"] == "unresolved"
    present = overall["systems"]["candidate"]["macro_f1_present_classes"]
    assert present["classes"] == ["entailment", "contradiction"]
    assert present["gate_read"] is False
    assert present["point"] is not None


async def test_rows_clustered_on_the_fallback_are_counted_per_slice(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    env["gold"]["base_problem_id"] = [
        None if i < 7 else v for i, v in enumerate(env["gold"]["base_problem_id"])
    ]
    env["gold"]["premises"] = [
        [{"pid": "a", "text": "fact"}, {"pid": "b", "text": f"rule {i}"}] for i in range(60)
    ]
    code, events = await _run(
        monkeypatch, env, _spec(evaluation=_settings(premises_column="premises"))
    )
    assert code == 0, events[-1]
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    assert report["population"]["fallback_keyed"] == 7
    assert report["slices"][0]["fallback_keyed"] == 7
    tiers = {s["value"]: s["fallback_keyed"] for s in report["slices"] if s["column"] == "tier"}
    expected = {t: sum(1 for i in range(7) if env["gold"]["tier"][i] == t) for t in tiers}
    assert tiers == expected


async def test_the_token_method_is_one_per_tier_and_echoed(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    method = "tokenizer:answerdotai/ModernBERT-large@45bb4654"
    env["gold"]["n_tokens"] = [100] * 60
    env["gold"]["n_tokens_method"] = [method] * 60
    rule = {
        "tier_column": "tier",
        "default": {"max_tokens": 2048},
        "token_column": "n_tokens",
        "token_method_column": "n_tokens_method",
    }
    code, events = await _run(monkeypatch, env, _spec(evaluation=_settings(tier_rule=rule)))
    assert code == 0, events[-1]
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    assert report["population"]["token_methods"] == dict.fromkeys(set(env["gold"]["tier"]), method)
    env["gold"]["n_tokens_method"][0] = "whitespace"
    code, events = await _run(monkeypatch, env, _spec(evaluation=_settings(tier_rule=rule)))
    assert code == 1
    assert "mixes token-count methods" in events[-1]["message"]


async def test_a_declared_premise_count_must_match_the_platform_s(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    env["gold"]["premises"] = [[{"pid": "1", "text": "a"}, {"pid": "2", "text": "b"}]] * 60
    env["gold"]["n_premises"] = [2] * 60
    settings = _settings(premises_column="premises", premise_count_column="n_premises")
    code, events = await _run(monkeypatch, env, _spec(evaluation=settings))
    assert code == 0, events[-1]
    env["gold"]["n_premises"][3] = 3
    code, events = await _run(monkeypatch, env, _spec(evaluation=settings))
    assert code == 1
    assert "row 3" in events[-1]["message"]


async def test_an_unstable_interval_never_decides_a_gate(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    # "neutral" is in a few gold rows only, so many resamples lack it on both sides.
    gold = env["gold"]["label"]
    for i, label in enumerate(gold):
        if label == "neutral" and i > 3:
            gold[i] = "entailment"
    for table in (env["cand"], env["base"]):
        table["pred"] = ["entailment" if p == "neutral" else p for p in table["pred"]]
    code, events = await _run(monkeypatch, env, _spec())
    assert code == 0, events[-1]
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    assert report["population"]["unstable_threshold"] == 0.01
    macro = report["slices"][0]["systems"]["candidate"]["macro_f1"]
    assert macro["undefined_share"] > 0.01
    assert macro["unstable"] is True
    delta = report["slices"][0]["comparisons"][0]["delta"]["macro_f1"]
    assert delta["unstable"] is True
    assert delta["excludes_zero"] == "unresolved"
    assert "macro_f1_lower" not in events[-1]["metrics"]
    # A higher recorded threshold makes the same interval stable.
    code, _ = await _run(monkeypatch, env, _spec(evaluation=_settings(unstable_threshold=1.0)))
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    assert report["slices"][0]["systems"]["candidate"]["macro_f1"]["unstable"] is False


async def test_a_two_way_tier_declares_its_own_label_space(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    # The "hard" tier is 2-way over entailment / contradiction: its slice declares only those,
    # so it is never "missing" neutral, while the overall population declares all three.
    hard = [t == "hard" for t in env["gold"]["tier"]]
    for table, column in ((env["gold"], "label"), (env["cand"], "pred"), (env["base"], "pred")):
        ids = env["gold"]["item_id"] if table is env["gold"] else table["item_id"]
        by_id = dict(zip(env["gold"]["item_id"], hard, strict=True))
        table[column] = [
            "contradiction" if v == "neutral" and by_id.get(i, False) else v
            for v, i in zip(table[column], ids, strict=True)
        ]
    spaces = dict.fromkeys(set(env["gold"]["tier"]), LABELS) | {
        "hard": ["entailment", "contradiction"]
    }
    settings = _settings(label_space_column="tier", label_spaces=spaces)
    code, events = await _run(monkeypatch, env, _spec(evaluation=settings))
    assert code == 0, events[-1]
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    hard_slice = next(s for s in report["slices"] if s["value"] == "hard")
    assert hard_slice["labels"] == ["entailment", "contradiction"]
    assert hard_slice["absent_classes"] == []
    assert hard_slice["systems"]["candidate"]["macro_f1"]["point"] is not None
    assert report["slices"][0]["labels"] == LABELS
    # A gold label outside its row's space refuses.
    env["gold"]["label"][next(i for i, h in enumerate(hard) if h)] = "neutral"
    code, events = await _run(monkeypatch, env, _spec(evaluation=settings))
    assert code == 1
    assert "outside its label space" in events[-1]["message"]
    # A label-space value the settings do not name refuses, naming the row; no fallback.
    unnamed = _settings(
        label_space_column="tier", label_spaces={"hard": ["entailment", "contradiction"]}
    )
    env["gold"]["label"][next(i for i, h in enumerate(hard) if h)] = "entailment"
    code, events = await _run(monkeypatch, env, _spec(evaluation=unnamed))
    assert code == 1
    assert "names no declared label space" in events[-1]["message"]
    assert "row " in events[-1]["message"]


async def test_each_slice_reports_its_tier_refused_rows(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    env["gold"]["n_tokens"] = [5000 if i % 10 == 0 else 10 for i in range(60)]
    env["gold"]["n_tokens_method"] = ["tokenizer:org/tok@abc"] * 60
    rule = {
        "default": {"max_tokens": 2048},
        "token_column": "n_tokens",
        "token_method_column": "n_tokens_method",
    }
    code, events = await _run(monkeypatch, env, _spec(evaluation=_settings(tier_rule=rule)))
    assert code == 0, events[-1]
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    assert report["slices"][0]["tier_refused"] == 6
    by_tier = {s["value"]: s["tier_refused"] for s in report["slices"] if s["column"] == "tier"}
    expected: dict[str, int] = {}
    for i in range(0, 60, 10):
        expected[env["gold"]["tier"][i]] = expected.get(env["gold"]["tier"][i], 0) + 1
    assert {k: v for k, v in by_tier.items() if v} == expected


async def test_absence_is_the_gold_s_whatever_a_system_predicts(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    # The gold has no neutral row; the candidate still predicts neutral twice. Every system's
    # macro-F1 and the delta are null by rule, the slice and the systems agree on the absent
    # classes, and the stray predictions are counted as a diagnostic.
    env["gold"]["label"] = ["entailment" if v == "neutral" else v for v in env["gold"]["label"]]
    env["base"]["pred"] = ["entailment" if v == "neutral" else v for v in env["base"]["pred"]]
    preds = ["entailment" if v == "neutral" else v for v in env["cand"]["pred"]]
    preds[5], preds[6] = "neutral", "neutral"
    env["cand"]["pred"] = preds
    code, events = await _run(monkeypatch, env, _spec())
    assert code == 0, events[-1]
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    for slice_ in report["slices"]:
        if not slice_["n"]:
            continue
        absent = slice_["absent_classes"]
        for name in ("candidate", "baseline"):
            macro = slice_["systems"][name]["macro_f1"]
            assert macro["absent_classes"] == absent
            assert (macro["point"] is None) == bool(absent)
        for comparison in slice_["comparisons"]:
            assert comparison["delta"]["macro_f1"]["absent_classes"] == absent
    overall = report["slices"][0]
    assert overall["absent_classes"] == ["neutral"]
    assert overall["systems"]["candidate"]["absent_class_predictions"] == {"neutral": 2}
    assert overall["systems"]["baseline"]["absent_class_predictions"] == {"neutral": 0}
    cand_present = overall["systems"]["candidate"]["macro_f1_present_classes"]["classes"]
    base_present = overall["systems"]["baseline"]["macro_f1_present_classes"]["classes"]
    assert cand_present == base_present == ["entailment", "contradiction"]


def _with_logits(state: dict[str, Any]) -> None:
    """Give both systems logit columns consistent with their predictions."""
    rng = random.Random(9)  # noqa: S311 — synthetic fixture.
    for name in ("cand", "base"):
        table = state[name]
        for label in LABELS:
            table[f"logit_{label}"] = [
                (2.0 if pred == label else 0.0) + rng.uniform(-0.5, 0.5) for pred in table["pred"]
            ]


async def test_calibration_reports_raw_cross_fit_and_applied_reads_apart(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    import math
    import statistics

    from strata_forge.metrics.calibration import temperature_scale
    from strata_forge.metrics.classification import brier

    _with_logits(env)
    own = env["tmp"] / "calibration.json"
    own.write_text(
        json.dumps(
            {
                "schema": "strata.calibration/v1",
                "fit": {"temperature": 1.5, "nll_before": 0.7, "nll_after": 0.6},
                "labels": LABELS,
            }
        )
    )
    _Hub.files[("u/model", "d" * 40)] = own
    cand = {
        "name": "candidate",
        "repo_id": "u/cand",
        "commit_sha": SHA_CAND,
        "calibration_ref": {"repo_id": "u/model", "commit_sha": "d" * 40},
    }
    spec = _spec(
        evaluation=_settings(
            predictions=cand, calibration={"fit_column": "tier", "fit_value": "easy"}
        )
    )
    code, events = await _run(monkeypatch, env, spec)
    assert code == 0, events[-1]
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    calibration = report["calibration"]
    assert calibration["fit_on"] == {"split": "test", "column": "tier", "value": "easy"}
    assert len(calibration["systems"]["candidate"]["cross_fit"]["temperatures"]) == 2
    assert calibration["systems"]["candidate"]["own"]["temperature"] == 1.5
    assert calibration["systems"]["baseline"]["own"] is None
    overall = report["slices"][0]
    entry = overall["systems"]["candidate"]
    logits = [[env["cand"][f"logit_{x}"][i] for x in LABELS] for i in range(60)]

    def expected(t: float) -> float | None:
        return brier(
            env["gold"]["label"],
            temperature_scale(logits, temperature=t),
            labels=LABELS,
            probability_kind="class_probabilities",
        ).value

    assert entry["raw"]["brier"]["point"] == expected(1.0)
    assert entry["applied"]["brier"]["point"] == expected(1.5)
    assert 0 <= entry["raw"]["brier"]["point"] <= 2
    assert entry["raw"]["n"] == 60
    assert math.isfinite(entry["raw"]["nll"])
    assert overall["systems"]["baseline"]["applied"] is None
    # The cross-fit read covers only the named fit slice (the 45 "easy" rows).
    assert entry["cross_fit"]["n"] == 45
    # Every read carries the ECE's own bins: they cover its rows and weigh up to its ECE.
    for read in ("raw", "cross_fit", "applied"):
        block = entry[read]
        bins = block["reliability_bins"]
        assert len(bins) == calibration["ece_bins"]
        assert sum(b["n"] for b in bins) == block["n"]
        weighted = (
            sum(b["n"] * abs(b["accuracy"] - b["mean_confidence"]) for b in bins if b["n"])
            / block["n"]
        )
        assert block["ece"] == pytest.approx(weighted, abs=1e-12)
        filled = [b for b in bins if b["n"]]
        assert all(b["lo"] <= b["mean_confidence"] <= b["hi"] for b in filled)
        assert [b["lo"] for b in filled] == sorted(b["lo"] for b in filled)
    hard = next(s for s in report["slices"] if s["value"] == "hard")
    assert hard["systems"]["candidate"]["cross_fit"] is None
    (comparison,) = overall["comparisons"]
    delta = comparison["brier_delta"]
    assert delta["n"] == 60
    base_logits = [[env["base"][f"logit_{x}"][i] for x in LABELS] for i in range(60)]

    def terms(rows: list[list[float]]) -> list[float]:
        probs = temperature_scale(rows, temperature=1.0)
        return [
            sum((p - (lab == g)) ** 2 for p, lab in zip(row, LABELS, strict=True))
            for row, g in zip(probs, env["gold"]["label"], strict=True)
        ]

    diffs = [c - b for b, c in zip(terms(base_logits), terms(logits), strict=True)]
    assert delta["sd"]["point"] == pytest.approx(statistics.stdev(diffs))
    assert delta["sd"]["lower"] <= delta["sd"]["upper"]
    assert events[-1]["metrics"]["brier"] == expected(1.0)


async def test_an_own_calibration_must_name_the_labels_in_order(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    _with_logits(env)
    own = env["tmp"] / "calibration.json"
    own.write_text(
        json.dumps(
            {
                "schema": "strata.calibration/v1",
                "fit": {"temperature": 1.5},
                "labels": list(reversed(LABELS)),
            }
        )
    )
    _Hub.files[("u/model", "d" * 40)] = own
    cand = {
        "name": "candidate",
        "repo_id": "u/cand",
        "commit_sha": SHA_CAND,
        "calibration_ref": {"repo_id": "u/model", "commit_sha": "d" * 40},
    }
    code, events = await _run(
        monkeypatch, env, _spec(evaluation=_settings(predictions=cand, calibration={}))
    )
    assert code == 1
    assert "labels in order" in events[-1]["message"]


async def test_calibration_needs_every_logit_column(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    _with_logits(env)
    del env["base"]["logit_neutral"]
    code, events = await _run(monkeypatch, env, _spec(evaluation=_settings(calibration={})))
    assert code == 1
    assert "logit_neutral" in events[-1]["message"]


async def test_a_fit_slice_is_named_by_column_and_value_together(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    code, events = await _run(
        monkeypatch, env, _spec(evaluation=_settings(calibration={"fit_column": "tier"}))
    )
    assert code == 1
    assert "fit_column and fit_value" in events[-1]["message"]


def _own_calibration(
    env: dict[str, Any], body: dict[str, Any], verifier: Any = None
) -> dict[str, Any]:
    own = env["tmp"] / "calibration.json"
    own.write_text(json.dumps(body))
    _Hub.named[("u/model", "d" * 40, "artifact/calibration.json")] = own
    if verifier is not None:
        sibling = env["tmp"] / "strata_verifier.json"
        sibling.write_text(json.dumps(verifier))
        _Hub.named[("u/model", "d" * 40, "artifact/strata_verifier.json")] = sibling
    return {
        "name": "candidate",
        "repo_id": "u/cand",
        "commit_sha": SHA_CAND,
        "calibration_ref": {
            "repo_id": "u/model",
            "commit_sha": "d" * 40,
            "path": "artifact/calibration.json",
        },
    }


async def test_the_trainer_s_calibration_takes_its_labels_from_the_verifier_beside_it(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    _with_logits(env)
    cand = _own_calibration(
        env,
        {"schema": "strata.calibration/v1", "fit": {"temperature": 2.0}, "rows": 10},
        {"schema": "strata.verifier/v1", "labels": LABELS},
    )
    spec = _spec(evaluation=_settings(predictions=cand, calibration={}))
    code, events = await _run(monkeypatch, env, spec)
    assert code == 0, events[-1]
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    own = report["calibration"]["systems"]["candidate"]["own"]
    assert own["temperature"] == 2.0
    assert own["labels_from"] == "artifact/strata_verifier.json"


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ({"fit": {"temperature": 2.0}, "labels": LABELS}, "strata.calibration/v1"),
        (
            {"schema": "strata.calibration/v1", "temperature": 2.0, "labels": LABELS},
            "fit.temperature",
        ),
    ],
)
async def test_a_calibration_file_must_follow_the_trainer_s_schema(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any], body: dict[str, Any], needle: str
) -> None:
    _with_logits(env)
    cand = _own_calibration(env, body)
    code, events = await _run(
        monkeypatch, env, _spec(evaluation=_settings(predictions=cand, calibration={}))
    )
    assert code == 1
    assert needle in events[-1]["message"]


async def test_a_calibration_ref_needs_calibration_settings(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    cand = _own_calibration(env, {})
    code, events = await _run(monkeypatch, env, _spec(evaluation=_settings(predictions=cand)))
    assert code == 1
    assert "needs calibration settings" in events[-1]["message"]


async def test_probability_reads_short_of_full_coverage_decide_no_gate(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, Any]
) -> None:
    _with_logits(env)
    # The baseline has no logits on three rows: its read covers 57/60, and every paired
    # probability read is on the 57 rows both have, unresolved.
    for label in LABELS:
        env["base"][f"logit_{label}"][:3] = [None, None, None]
    code, events = await _run(monkeypatch, env, _spec(evaluation=_settings(calibration={})))
    assert code == 0, events[-1]
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    overall = report["slices"][0]
    base = overall["systems"]["baseline"]["raw"]
    assert (base["n"], base["n_population"], base["gate"]) == (57, 60, "unresolved")
    assert base["coverage"] == pytest.approx(57 / 60)
    cand = overall["systems"]["candidate"]["raw"]
    assert (cand["coverage"], cand["gate"]) == (1.0, "resolved")
    (comparison,) = overall["comparisons"]
    assert comparison["brier_delta"]["n"] == 57
    assert comparison["brier_delta"]["excludes_zero"] == "unresolved"
    assert comparison["discordance"]["n"] == 57
    assert comparison["discordance"]["gate"] == "unresolved"
    assert events[-1]["metrics"]["brier_coverage"] == 1.0
