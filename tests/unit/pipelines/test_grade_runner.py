"""The LLM grader, end to end over synthetic pinned files and a scripted judge.

The Hub client, ``datasets`` and the judge's chat run are replaced by local stand-ins; the spec,
the joins, the judge rows, the reading of the judge's results, the metrics, the agreement, the
items file, the manifest and the push are the runner's own code. The scripted judge answers
from the output text alone, so every expected score is known.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from strata_forge.metrics.cluster_bootstrap import cluster_mean_bootstrap
from strata_forge.metrics.judge_agreement import judge_agreement
from strata_forge.pipelines import grade_runner as gr
from strata_forge.pipelines import inference_runner
from strata_forge.pipelines._common import LEGACY_TOKEN_ENV, RunError, RunSecrets
from strata_forge.training.progress import JsonlProgressWriter

if TYPE_CHECKING:
    from pathlib import Path

SHA_GOLD = "a" * 40
SHA_CAND = "b" * 40
SHA_BASE = "c" * 40
SHA_JUDGE = "d" * 40
_TOKEN = "hf_secretwritetoken1234567890"
_N = 40
_TEMPLATE = "Question: {question}\nAnswer: {output}\nGrade it on:\n{criteria}"


def _gold() -> dict[str, list[Any]]:
    return {
        "uid": [f"q-{i:03d}" for i in range(_N)],
        "question": [f"synthetic question {i}" for i in range(_N)],
        "group": [f"g-{i // 2:03d}" for i in range(_N)],
        "tier": ["easy" if i % 4 else "hard" for i in range(_N)],
    }


def _answer(i: int, good: bool) -> str:
    # The scripted judge reads the verdict and the score from the text itself.
    return f"answer {i} {'GOOD' if good else 'BAD'} score={3 + (i % 3) if good else 1}"


def _outputs(good_every: int, *, missing: tuple[int, ...] = ()) -> dict[str, list[Any]]:
    return {
        "uid": [f"q-{i:03d}" for i in range(_N)],
        "output": [None if i in missing else _answer(i, i % good_every != 0) for i in range(_N)],
        "error": ["boom" if i in missing else None for i in range(_N)],
        "human_policy": [bool(i % good_every != 0) if i < 20 else None for i in range(_N)],
    }


class _Split:
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
    named: ClassVar[dict[tuple[str, str, str], Path]] = {}
    calls: ClassVar[list[tuple[str, Any]]] = []

    def __init__(self, token: Any = None) -> None:
        _Hub.calls.append(("init", token))

    async def download_file(self, repo_id: str, filename: str, **kwargs: Any) -> Path:
        return _Hub.named[(repo_id, kwargs["revision"], filename)]

    async def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        _Hub.calls.append(("create_repo", (repo_id, kwargs)))

    async def is_private(self, repo_id: str, **kwargs: Any) -> bool:
        del repo_id, kwargs
        return True

    async def upload_files(self, files: Any, repo_id: str, **kwargs: Any) -> str:
        _Hub.calls.append(("upload_files", (sorted(dict(files)), repo_id)))
        return "e" * 40


def _scripted_reply(text: str, mode: str) -> tuple[str | None, str | None, bool | None]:
    """(output, error, json_valid) for one judge row."""
    if mode == "error" and "answer 7 " in text:
        return None, "judge timeout", None
    if mode == "invalid" and "answer 8 " in text:
        return "{not json", None, False
    good = "GOOD" in text
    score = int(text.split("score=")[1].split()[0])
    reply = {
        "rationale": f"the answer is {'fine' if good else 'wrong'}",
        "scores": {"policy": good, "helpfulness": score},
    }
    return json.dumps(reply), None, True


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    monkeypatch.delenv("FORGE_SECRETS_FILE", raising=False)
    monkeypatch.delenv(LEGACY_TOKEN_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    state: dict[str, Any] = {"gold": _gold(), "tmp": tmp_path, "judge_mode": "ok", "judge": []}
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: _Split(state["gold"])  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)
    _Hub.named, _Hub.calls = {}, []
    monkeypatch.setattr(gr, "HFHubClient", _Hub)

    async def _judge(spec: Any, secrets: Any, writer: Any, *, rows: list[dict[str, Any]]) -> str:
        state["judge"].append({"spec": spec, "rows": rows})
        rendered = [
            inference_runner.render_template(spec.template, row, spec.column_mapping)
            for row in rows
        ]
        state["rendered"] = rendered
        out = {"custom_id": [], "output": [], "error": [], "json_valid": [], "json_retried": []}
        for index, text in enumerate(rendered):
            output, error, valid = _scripted_reply(text, state["judge_mode"])
            out["custom_id"].append(f"row-{index}")
            out["output"].append(output)
            out["error"].append(error)
            out["json_valid"].append(valid)
            out["json_retried"].append(valid is False)
        directory = tmp_path / "judge"
        directory.mkdir(exist_ok=True)
        pq.write_table(pa.Table.from_pydict(out), str(directory / "results.parquet"))
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "model": {"commit": SHA_JUDGE},
                    "serving": {"dtype": "bf16"},
                    "counts": {"rows": len(rows)},
                }
            )
        )
        return str(directory / "results.parquet")

    monkeypatch.setattr(inference_runner, "_execute", _judge)
    state["cand"] = _outputs(5)
    state["base"] = _outputs(2)
    return state


def _write(state: dict[str, Any]) -> None:
    for name, repo, sha in (("cand", "u/cand", SHA_CAND), ("base", "u/base", SHA_BASE)):
        path = state["tmp"] / f"{name}.parquet"
        pq.write_table(pa.Table.from_pydict(state[name]), str(path))
        _Hub.named[(repo, sha, "results.parquet")] = path


def _settings(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "rubric": {
            "name": "support quality",
            "template": _TEMPLATE,
            "criteria": [
                {
                    "id": "policy",
                    "description": "Follows policy.",
                    "scale": "pass_fail",
                    "gold_column": "human_policy",
                },
                {
                    "id": "helpfulness",
                    "description": "Helpful.",
                    "scale": "ordinal",
                    "min": 1,
                    "max": 5,
                    "pass_at": 4,
                },
            ],
        },
        "judge_model_id": "org/judge",
        "judge_revision": SHA_JUDGE,
        "column_mapping": {"question": "question"},
        "outputs": [
            {"name": "candidate", "repo_id": "u/cand", "commit_sha": SHA_CAND, "id_column": "uid"},
            {"name": "baseline", "repo_id": "u/base", "commit_sha": SHA_BASE, "id_column": "uid"},
        ],
        "id_column": "uid",
        "cluster_column": "group",
        "slice_columns": ["tier"],
        "seed": 20261104,
        "resamples": 200,
        "min_slice_rows": 20,
    }
    return settings | overrides


def _spec(**overrides: Any) -> gr.GradeSpec:
    return gr.GradeSpec.model_validate(
        {
            "dataset_id": "u/gold",
            "dataset_commit_sha": SHA_GOLD,
            "split": "test",
            "grading": _settings(**overrides),
            "output_repo_id": "u/grades",
            "run_id": "grade1",
        }
    )


def _run(
    spec: gr.GradeSpec, tmp: Path, *, token: str | None = _TOKEN
) -> tuple[str, list[dict[str, Any]]]:
    from pydantic import SecretStr

    progress = tmp / "progress.jsonl"
    writer = JsonlProgressWriter(progress)
    secrets = RunSecrets(hf_token=SecretStr(token) if token else None)
    try:
        destination = asyncio.run(gr._execute(spec, secrets, writer))  # pyright: ignore[reportPrivateUsage]
    finally:
        writer.close()
    events = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    return destination, events


def _outdir(state: dict[str, Any]) -> Path:
    return state["tmp"] / "home" / "strata-grade-results" / "grade1"


# ------------------------------------------------------------------ the run


def test_a_graded_run_scores_aggregates_and_pushes_one_commit(env: dict[str, Any]) -> None:
    _write(env)
    destination, events = _run(_spec(), env["tmp"])
    assert destination == f"u/grades@{'e' * 40}"
    assert [c for c in _Hub.calls if c[0] == "upload_files"] == [
        (
            "upload_files",
            (["README.md", "items.parquet", "manifest.json", "metrics.json"], "u/grades"),
        )
    ]
    # One judge row per (case, system): every case had an output from both.
    (judge,) = env["judge"]
    assert len(judge["rows"]) == 2 * _N
    assert judge["spec"].model_id == "org/judge"
    assert judge["spec"].model_revision == SHA_JUDGE
    assert judge["spec"].hyperparams.temperature == 0.0
    assert judge["spec"].hyperparams.guided_json["required"] == ["rationale", "scores"]
    # The rubric was rendered over the gold question, the output and the criteria list.
    assert env["rendered"][0].startswith("Question: synthetic question 0\nAnswer: answer 0 BAD")
    assert "1. policy (true if it passes" in env["rendered"][0]

    report = json.loads((_outdir(env) / "metrics.json").read_text())
    assert report["schema"] == "strata.eval.grades/v1"
    assert report["judge"]["commit"] == SHA_JUDGE
    assert report["tier_rule"] is None
    overall = report["slices"][0]
    cand = overall["systems"]["candidate"]
    assert (cand["n"], cand["graded"], cand["coverage"], cand["gate"]) == (_N, _N, 1.0, "resolved")
    # Candidate fails every 5th case: pass rate 32/40, matching a direct cluster bootstrap.
    passes = [float(i % 5 != 0) for i in range(_N)]
    expected = cluster_mean_bootstrap(
        passes, clusters=[f"g-{i // 2:03d}" for i in range(_N)], seed=20261104, resamples=200
    ).interval
    assert cand["criteria"]["policy"]["mean"]["point"] == pytest.approx(0.8)
    assert cand["criteria"]["policy"]["mean"]["lower"] == expected.lower
    assert cand["criteria"]["policy"]["pass_rate"] is None
    assert cand["criteria"]["helpfulness"]["pass_rate"] is not None

    (comparison,) = overall["comparisons"]
    assert (comparison["baseline"], comparison["n"], comparison["gate"]) == (
        "baseline",
        _N,
        "resolved",
    )
    moves = comparison["criteria"]["policy"]["transitions"]
    # Baseline fails every 2nd case, candidate every 5th.
    assert moves["fail_to_pass"] == sum(1 for i in range(_N) if i % 2 == 0 and i % 5 != 0)
    assert moves["pass_to_fail"] == sum(1 for i in range(_N) if i % 2 != 0 and i % 5 == 0)
    assert comparison["criteria"]["policy"]["delta"]["point"] == pytest.approx(0.8 - 0.5)

    hard = next(s for s in report["slices"] if s["value"] == "hard")
    assert (hard["n"], hard["diagnostic"]) == (10, True)

    # Agreement with the human gold on the first 20 cases.
    agreement = report["agreement"]["candidate"]["policy"]
    gold = [int(i % 5 != 0) for i in range(20)]
    assert agreement["n"] == 20
    assert agreement["n_gold"] == 20
    assert (
        agreement["percent_agreement"]
        == judge_agreement(gold, gold, values=[0, 1], scale="pass_fail").percent_agreement
    )

    end = events[-1]
    assert end["kind"] == "end"
    assert end["metrics"]["policy"] == pytest.approx(0.8)
    assert end["metrics"]["coverage"] == 1.0
    assert end["metrics"]["agreement_policy"] == 1.0
    assert "delta_policy__baseline" in end["metrics"]
    assert len(end["metrics"]) <= 64
    # The judge's own start/end never reach this run's stream.
    assert [e["kind"] for e in events].count("end") == 1


def test_items_hold_statuses_scores_and_rationales_never_text(env: dict[str, Any]) -> None:
    _write(env)
    _run(_spec(), env["tmp"])
    items = pq.read_table(str(_outdir(env) / "items.parquet")).to_pylist()
    assert len(items) == _N
    first = items[0]
    assert first["item_id"] == "q-000"
    assert first["status__candidate"] == "graded"
    assert first["score__policy__candidate"] == 0
    assert first["score__helpfulness__candidate"] == 1
    assert first["rationale__candidate"] == "the answer is wrong"
    assert first["gold__policy__candidate"] == 0
    flat = json.dumps(items)
    assert "synthetic question" not in flat
    assert "answer 0" not in flat


def test_unscored_cases_stay_visible_and_never_pass(env: dict[str, Any]) -> None:
    env["cand"] = _outputs(5, missing=(3,))
    env["judge_mode"] = "error"
    _write(env)
    _run(_spec(), env["tmp"])
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    cand = report["slices"][0]["systems"]["candidate"]
    assert cand["unscored"] == {"output_missing": 1, "judge_error": 1, "judge_invalid": 0}
    assert cand["graded"] == _N - 2
    assert cand["gate"] == "unresolved"
    graded = [i for i in range(_N) if i not in (3, 7)]
    assert cand["criteria"]["policy"]["mean"]["point"] == pytest.approx(
        sum(1 for i in graded if i % 5 != 0) / len(graded)
    )
    items = pq.read_table(str(_outdir(env) / "items.parquet")).to_pylist()
    assert items[3]["status__candidate"] == "output_missing"
    assert items[7]["status__candidate"] == "judge_error"
    assert items[7]["score__policy__candidate"] is None
    # A run under full coverage carries no bounds in its headline.
    assert (
        "policy_lower"
        not in json.loads((env["tmp"] / "progress.jsonl").read_text().splitlines()[-1])["metrics"]
    )


def test_an_invalid_judge_reply_is_unscored(env: dict[str, Any]) -> None:
    env["judge_mode"] = "invalid"
    _write(env)
    _run(_spec(), env["tmp"])
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    assert report["slices"][0]["systems"]["candidate"]["unscored"]["judge_invalid"] == 1


# ------------------------------------------------------------------ joining by position


def _positional(env: dict[str, Any], manifest: dict[str, Any] | None) -> gr.GradeSpec:
    for name, repo, sha in (("cand", "u/cand", SHA_CAND),):
        data = dict(env[name])
        data["custom_id"] = [f"row-{i}" for i in range(_N)]
        del data["uid"]
        path = env["tmp"] / f"{name}.parquet"
        pq.write_table(pa.Table.from_pydict(data), str(path))
        _Hub.named[(repo, sha, "results.parquet")] = path
        if manifest is not None:
            mpath = env["tmp"] / f"{name}-manifest.json"
            mpath.write_text(json.dumps(manifest))
            _Hub.named[(repo, sha, "manifest.json")] = mpath
    return _spec(
        id_column=None,
        outputs=[{"name": "candidate", "repo_id": "u/cand", "commit_sha": SHA_CAND}],
    )


def test_a_positional_join_needs_the_same_split_at_the_same_commit(env: dict[str, Any]) -> None:
    same = {
        "dataset": {"id": "u/gold", "config": None, "split": "test", "requested_revision": SHA_GOLD}
    }
    _run(_positional(env, same), env["tmp"])
    report = json.loads((_outdir(env) / "metrics.json").read_text())
    assert report["systems"][0]["join"] == "position"
    assert report["slices"][0]["systems"]["candidate"]["graded"] == _N

    other = {
        "dataset": {
            "id": "u/gold",
            "config": None,
            "split": "train",
            "requested_revision": SHA_GOLD,
        }
    }
    with pytest.raises(RunError, match="wrong cases"):
        _run(_positional(env, other), env["tmp"])
    with pytest.raises(RunError, match="no readable manifest"):
        _run(_positional(env, {"model": {}}), env["tmp"])


def test_join_refusals(env: dict[str, Any]) -> None:
    env["cand"]["uid"][1] = "q-000"
    _write(env)
    with pytest.raises(RunError, match="repeats the id"):
        _run(_spec(), env["tmp"])
    env["cand"] = _outputs(5)
    env["cand"]["human_policy"] = [7 if i == 0 else None for i in range(_N)]
    _write(env)
    with pytest.raises(RunError, match="not on its scale"):
        _run(_spec(), env["tmp"])


# ------------------------------------------------------------------ the spec


def test_settings_refusals() -> None:
    with pytest.raises(ValidationError, match="exactly the rubric's column placeholders"):
        gr.GradeSettings.model_validate(_settings(column_mapping={}))
    with pytest.raises(ValidationError, match="filled by the grader"):
        gr.GradeSettings.model_validate(_settings(column_mapping={"question": "q", "output": "o"}))
    with pytest.raises(ValidationError, match="judge_revision"):
        gr.GradeSettings.model_validate(_settings(judge_revision="main"))
    with pytest.raises(ValidationError, match="may not set"):
        gr.GradeSettings.model_validate(_settings(judge_hyperparams={"guided_json": {}}))
    with pytest.raises(ValidationError, match="judge_hyperparams"):
        gr.GradeSettings.model_validate(_settings(judge_hyperparams={"not_a_knob": 1}))
    with pytest.raises(ValidationError, match="need an id_column"):
        gr.GradeSettings.model_validate(
            _settings(outputs=[{"name": "c", "repo_id": "u/c", "commit_sha": SHA_CAND}])
        )
    with pytest.raises(ValidationError, match="none may set id_column"):
        gr.GradeSettings.model_validate(_settings(id_column=None))
    with pytest.raises(ValidationError, match="distinct name"):
        gr.GradeSettings.model_validate(
            _settings(
                outputs=[
                    {"name": "c", "repo_id": "u/c", "commit_sha": SHA_CAND, "id_column": "uid"},
                    {"name": "c", "repo_id": "u/d", "commit_sha": SHA_BASE, "id_column": "uid"},
                ]
            )
        )


def test_the_judge_keeps_tuning_knobs_and_the_grader_owns_the_rest() -> None:
    settings = gr.GradeSettings.model_validate(
        _settings(judge_hyperparams={"max_tokens": 512, "temperature": 0.2})
    )
    hp = gr.judge_hyperparams(settings)
    assert (hp.engine, hp.max_tokens, hp.temperature) == ("vllm_chat", 512, 0.2)
    assert hp.guided_json is not None
