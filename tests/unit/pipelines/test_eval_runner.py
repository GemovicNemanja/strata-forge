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
    calls: ClassVar[list[tuple[str, Any]]] = []
    private: ClassVar[bool] = True

    def __init__(self, token: Any = None) -> None:
        _Hub.calls.append(("init", token))

    async def download_file(self, repo_id: str, filename: str, **kwargs: Any) -> Path:
        _Hub.calls.append(("download_file", (repo_id, filename, kwargs)))
        return _Hub.files[(repo_id, kwargs["revision"])]

    async def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        _Hub.calls.append(("create_repo", (repo_id, kwargs)))

    async def is_private(self, repo_id: str, **kwargs: Any) -> bool:
        del repo_id, kwargs
        return _Hub.private

    async def upload_files(self, files: Any, repo_id: str, **kwargs: Any) -> str:
        _Hub.calls.append(("upload_files", (dict(files), repo_id, kwargs)))
        return "commit"


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
    _Hub.files, _Hub.calls, _Hub.private = {}, [], True
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
    assert overall["systems"]["candidate"]["macro_f1"] == expected.intervals.macro_f1.model_dump()
    assert overall["systems"]["candidate"]["accuracy"] == expected.intervals.accuracy.model_dump()
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
    assert comparison["delta"]["macro_f1"] == paired.delta.macro_f1.model_dump()
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
    overall = report["slices"][0]["systems"]["candidate"]
    assert overall["accuracy"]["point"] == pytest.approx(56 / 60)
    assert overall["invalid_rate"]["point"] == pytest.approx(4 / 60)
    items = pq.read_table(str(_outdir(env) / "items.parquet")).to_pylist()
    assert items[0]["pred__candidate"] is None
    assert items[0]["correct__candidate"] is False
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
    assert events[-1]["message"] == "u/report"
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
