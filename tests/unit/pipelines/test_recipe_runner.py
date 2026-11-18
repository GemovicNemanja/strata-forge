"""The dataset recipe runner through its real ``main``, with the Hub, the split reader, the Parquet
writer and the model server replaced. Every row is synthetic.
"""

from __future__ import annotations

import contextlib
import json
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from strata_forge.datasets.recipe import Recipe, recipe_sha256
from strata_forge.pipelines import SPEC_VERSION
from strata_forge.pipelines import recipe_runner as rr
from strata_forge.pipelines._common import LEGACY_TOKEN_ENV

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

_TOKEN = "hf_secretwritetoken1234567890"
_SHA = "0123456789abcdef0123456789abcdef01234567"
_MODEL_SHA = "89abcdef0123456789abcdef0123456789abcdef"
_SECRET_TEXT = "row-text-that-never-reaches-the-summary"

_ROWS = {
    "tickets": [
        {"ticket": i, "text": f"{_SECRET_TEXT} {i}", "lang": "en" if i % 3 else "es"}
        for i in range(12)
    ],
    "held": [{"ticket": 99, "text": "held", "lang": "en"}],
}


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"customer": {"type": "string"}},
        "required": ["customer"],
        "additionalProperties": False,
    }


def _recipe(*, synth: bool = False, holdout: bool = False) -> dict[str, Any]:
    steps: list[dict[str, Any]] = [
        {"kind": "load", "source": "tickets"},
        {"kind": "dedupe", "keys": ["lang"]}
        if not synth
        else {"kind": "filter", "conditions": [{"column": "lang", "op": "eq", "value": "en"}]},
    ]
    if holdout:
        steps.insert(1, {"kind": "concat", "source": "held"})
    if synth:
        steps.append(
            {
                "kind": "synthesize",
                "model_id": "org/generator",
                "model_revision": _MODEL_SHA,
                "prompt_template": "Vary: {text}",
                "output_schema": _schema(),
                "output_columns": {"customer": "customer"},
                "n": 2,
                "seed": 42,
                "max_seed_rows": 50,
            }
        )
    return {
        "name": "support-edge-cases",
        "sources": [
            {"name": "tickets", "hf_repo_id": "org/tickets", "revision": _SHA, "split": "train"},
            {"name": "held", "hf_repo_id": "org/held", "revision": _SHA, "split": "test"},
        ],
        "steps": steps,
    }


def _spec(recipe: dict[str, Any], **kw: Any) -> dict[str, Any]:
    parsed = Recipe.model_validate(recipe)
    return {
        "engine_version": SPEC_VERSION,
        "recipe": recipe,
        "recipe_revision": 3,
        "recipe_sha256": recipe_sha256(parsed),
        "mode": "materialize",
        "output_repo_id": "me/support-edge-cases",
        "run_id": "run7",
    } | kw


class _Hub:
    calls: ClassVar[list[tuple[str, Any]]] = []
    private: ClassVar[bool] = True

    def __init__(self, *, token: Any) -> None:
        _Hub.calls.append(("token", token))

    async def create_repo(self, repo_id: str, **_kw: Any) -> None:
        _Hub.calls.append(("create_repo", repo_id))

    async def is_private(self, repo_id: str, **_kw: Any) -> bool:
        return _Hub.private

    async def download_snapshot(self, repo_id: str, **kw: Any) -> Path:
        _Hub.calls.append(("download", (repo_id, kw["revision"])))
        snapshot = _Hub.root / "snapshots" / kw["revision"]
        snapshot.mkdir(parents=True, exist_ok=True)
        return snapshot

    async def replace_files(self, files: dict[str, str], repo_id: str, **kw: Any) -> str:
        _Hub.calls.append(("replace_files", (sorted(files), repo_id, kw)))
        _Hub.pushed = {name: Path(path).read_bytes() for name, path in files.items()}  # noqa: ASYNC240
        return "c" * 40

    root: ClassVar[Any] = None
    pushed: ClassVar[dict[str, bytes]] = {}


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"parquet": {}, "requests": []}
    monkeypatch.delenv(LEGACY_TOKEN_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    secrets = tmp_path / ".secrets.json"
    secrets.write_text(json.dumps({"HF_TOKEN": _TOKEN}))
    secrets.chmod(0o600)
    monkeypatch.setenv("FORGE_SECRETS_FILE", str(secrets))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    _Hub.calls = []
    _Hub.private = True
    _Hub.root = tmp_path
    _Hub.pushed = {}
    monkeypatch.setattr(rr, "HFHubClient", _Hub)

    def _read(source: Any, token: Any, row_cap: int) -> list[dict[str, Any]]:
        record.setdefault("read", []).append((source.name, source.revision, token))
        return _ROWS[source.name]

    def _write(records: list[dict[str, Any]], columns: list[str], path: Path) -> Path:
        record["parquet"][path.name] = records
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"parquet")
        return path

    @contextlib.asynccontextmanager
    async def _serving(_backend: Any, task: Any, **_kw: Any) -> AsyncGenerator[Any]:
        record["task"] = task
        yield types.SimpleNamespace(base_url="http://127.0.0.1:8000/v1")

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            record["client"] = kwargs

        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            extras = kwargs["provider_extras"]["openai_compat"]
            record["requests"].append((messages[0].content, extras["seed"], extras["extra_body"]))
            seed = extras["seed"]
            if seed == 43:
                raise RuntimeError("server error")
            text = "not json" if seed == 44 else json.dumps({"customer": f"case {seed}"})
            return types.SimpleNamespace(text=text)

    monkeypatch.setattr(rr, "_read_source", _read)
    monkeypatch.setattr(rr, "_write_parquet", _write)
    monkeypatch.setattr(rr, "serving_endpoint", _serving)
    monkeypatch.setattr(rr, "LLMClient", _Client)
    record["progress"] = tmp_path / "progress.jsonl"
    record["tmp"] = tmp_path
    return record


def _launch(monkeypatch: pytest.MonkeyPatch, record: dict[str, Any], spec: dict[str, Any]) -> None:
    spec = spec | {"progress_path": str(record["progress"])}
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(spec))


def _events(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in record["progress"].read_text().splitlines() if line]


async def test_a_deterministic_recipe_is_materialized_in_one_commit(
    monkeypatch: pytest.MonkeyPatch, run: dict[str, Any]
) -> None:
    _launch(monkeypatch, run, _spec(_recipe()))
    assert await rr.main() == 0
    assert run["read"] == [("tickets", _SHA, _TOKEN), ("held", _SHA, _TOKEN)]
    (_, (files, repo, kw)) = next(c for c in _Hub.calls if c[0] == "replace_files")
    assert repo == "me/support-edge-cases"
    assert files == sorted(
        [
            "README.md",
            "data/train-00000-of-00001.parquet",
            "exclusions.parquet",
            "manifest.json",
            "recipe.json",
        ]
    )
    assert kw["managed"] == rr.MANAGED_PATHS
    assert kw["branch"] is None  # materialize: the default branch
    assert kw["commit_message"] == "materialize recipe support-edge-cases r3"
    # dedupe on lang keeps the first English and the first Spanish row.
    assert [r["_row_id"] for r in run["parquet"]["train-00000-of-00001.parquet"]] == [
        "tickets:0",
        "tickets:1",
    ]
    assert len(run["parquet"]["exclusions.parquet"]) == 10
    manifest = json.loads(_Hub.pushed["manifest.json"])
    assert manifest["kind"] == "dataset_recipe"
    assert manifest["recipe"] == {
        "name": "support-edge-cases",
        "schema": "strata.dataset_recipe/v1",
        "revision": 3,
        "sha256": recipe_sha256(Recipe.model_validate(_recipe())),
    }
    assert [s["revision"] for s in manifest["sources"]] == [_SHA, _SHA]
    recipe = json.loads(_Hub.pushed["recipe.json"])
    assert recipe["revision"] == 3
    card = _Hub.pushed["README.md"].decode()
    assert "path: data/train-*.parquet" in card
    summary = json.loads((run["tmp"] / rr.WORKDIR_SUMMARY_FILENAME).read_text())
    assert summary["commit"] == "c" * 40
    assert summary["splits"]["train"]["rows"] == 2
    assert summary["exclusions_by_reason"] == {"duplicate_of": 10}
    assert summary["profile"]["train"][0] == {
        "column": "ticket",
        "rows": 2,
        "nulls": 0,
        "distinct": 2,
    }
    assert _SECRET_TEXT not in json.dumps(summary)
    end = next(e for e in _events(run) if e["kind"] == "end")
    assert end["message"] == f"me/support-edge-cases@{'c' * 40}"
    assert end["metrics"]["rows"] == 2.0
    assert _TOKEN not in run["progress"].read_text()


async def test_synthesis_serves_the_pinned_model_seeds_each_request_and_marks_rows(
    monkeypatch: pytest.MonkeyPatch, run: dict[str, Any]
) -> None:
    _launch(monkeypatch, run, _spec(_recipe(synth=True)))
    assert await rr.main() == 0
    assert ("download", ("org/generator", _MODEL_SHA)) in _Hub.calls
    assert run["task"].env["HF_HUB_OFFLINE"] == "1"
    assert "HF_TOKEN" not in run["task"].env
    # 8 English seed rows, 2 samples each, seeds 42.. in order.
    seeds = sorted(seed for _, seed, _ in run["requests"])
    assert seeds == list(range(42, 58))
    _, _, body = run["requests"][0]
    assert body == {"structured_outputs": {"json": _schema()}}
    rows = run["parquet"]["train-00000-of-00001.parquet"]
    assert len(rows) == 14  # one request failed, one reply was not JSON
    assert all(r["_synthetic"] for r in rows)
    assert rows[0]["_seed_row_id"] == "tickets:1"
    summary = json.loads((run["tmp"] / rr.WORKDIR_SUMMARY_FILENAME).read_text())
    assert summary["exclusions_by_reason"] == {
        "filtered": 4,
        "generation_failed": 1,
        "invalid_output": 1,
    }
    assert summary["synthetic_rows"] == 14
    assert summary["generation"] == [
        {
            "model_id": "org/generator",
            "model_revision": _MODEL_SHA,
            "requests": 16,
            "valid": 14,
            "invalid": 1,
            "failed": 1,
        }
    ]
    manifest = json.loads(_Hub.pushed["manifest.json"])
    (facts,) = manifest["generation"]
    assert (facts["seed"], facts["n"]) == (42, 2)
    assert "synthetic" in _Hub.pushed["README.md"].decode()


async def test_a_preview_is_bounded_and_goes_to_a_branch(
    monkeypatch: pytest.MonkeyPatch, run: dict[str, Any]
) -> None:
    _launch(monkeypatch, run, _spec(_recipe(synth=True), mode="preview", preview_rows=3))
    assert await rr.main() == 0
    assert len(run["requests"]) == 3 * 2  # 3 seed rows, 2 samples each
    (_, (_, _, kw)) = next(c for c in _Hub.calls if c[0] == "replace_files")
    assert kw["branch"] == "preview-run7"
    assert kw["commit_message"].startswith("preview recipe")
    rows = run["parquet"]["train-00000-of-00001.parquet"]
    assert len(rows) <= 3
    summary = json.loads((run["tmp"] / rr.WORKDIR_SUMMARY_FILENAME).read_text())
    assert summary["mode"] == "preview"
    assert summary["branch"] == "preview-run7"


async def test_generation_never_reads_a_held_out_split(
    monkeypatch: pytest.MonkeyPatch, run: dict[str, Any]
) -> None:
    _launch(monkeypatch, run, _spec(_recipe(synth=True, holdout=True)))
    assert await rr.main() == 1
    error = next(e for e in _events(run) if e["kind"] == "error")
    assert "held-out data (held)" in error["message"]
    assert not [c for c in _Hub.calls if c[0] == "replace_files"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"recipe_sha256": "0" * 64}, "not the declared revision"),
        ({"output_repo_id": "not a repo"}, "output repo"),
    ],
)
async def test_a_spec_that_is_not_its_revision_is_refused(
    monkeypatch: pytest.MonkeyPatch, run: dict[str, Any], change: dict[str, Any], message: str
) -> None:
    _launch(monkeypatch, run, _spec(_recipe()) | change)
    assert await rr.main() == 1
    error = next(e for e in _events(run) if e["kind"] == "error")
    assert message in error["message"]
    assert "read" not in run


async def test_a_public_output_repo_is_refused_before_any_read(
    monkeypatch: pytest.MonkeyPatch, run: dict[str, Any]
) -> None:
    _Hub.private = False
    _launch(monkeypatch, run, _spec(_recipe()))
    assert await rr.main() == 1
    assert "read" not in run


async def test_without_a_write_token_nothing_runs(
    monkeypatch: pytest.MonkeyPatch, run: dict[str, Any], tmp_path: Path
) -> None:
    (tmp_path / ".secrets.json").write_text("{}")
    _launch(monkeypatch, run, _spec(_recipe()))
    assert await rr.main() == 1
    error = next(e for e in _events(run) if e["kind"] == "error")
    assert "write token" in error["message"]


async def test_an_empty_split_has_no_file_and_no_card_entry(
    monkeypatch: pytest.MonkeyPatch, run: dict[str, Any]
) -> None:
    recipe = _recipe()
    recipe["steps"].append({"kind": "split", "ratios": {"train": 0.9, "test": 0.1}, "seed": 1})
    _launch(monkeypatch, run, _spec(recipe))
    assert await rr.main() == 0
    (_, (files, _, _)) = next(c for c in _Hub.calls if c[0] == "replace_files")
    assert "data/train-00000-of-00001.parquet" in files
    assert "data/test-00000-of-00001.parquet" not in files
    assert "split: test" not in _Hub.pushed["README.md"].decode()
    summary = json.loads((run["tmp"] / rr.WORKDIR_SUMMARY_FILENAME).read_text())
    assert summary["splits"]["test"] == {
        "rows": 0,
        "written": 0,
        "synthetic": 0,
        "path": None,
        "columns": ["ticket", "text", "lang"],
    }
