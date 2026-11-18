"""The ``hf_classify`` scoring engine and its path through the inference runner.

The pure parts (labels, rendering, batching, the softmax, the results rows and counts) are tested
with known values. The orchestration runs the real ``main`` with the heavy externals replaced: the
dataset, the model download, the classifier and the Hub client. The real transformers path is
covered by the clean-install smoke check (``check_hf_classify``), against a direct forward pass.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from strata_forge.pipelines import SPEC_VERSION
from strata_forge.pipelines import _scoring as sc
from strata_forge.pipelines import inference_runner as ir
from strata_forge.pipelines._common import LEGACY_TOKEN_ENV, RunError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_TOKEN = "hf_secretwritetoken1234567890"
_SHA = "0123456789abcdef0123456789abcdef01234567"
_DATA_SHA = "89abcdef0123456789abcdef0123456789abcdef"


_ALIAS = {"ENTAILMENT": "entailed", "NOT_ENTAILMENT": "not_entailed"}
_SCORING: dict[str, Any] = {
    "engine": "hf_classify",
    "pair_template": "{h}",
    "dtype": "float32",
    "label_alias": _ALIAS,
}


def _hp(**overrides: Any) -> ir.Hyperparams:
    base: dict[str, Any] = dict(_SCORING)
    return ir.Hyperparams.model_validate(base | overrides)


def _spec_dict(**hyperparams: Any) -> dict[str, Any]:
    return {
        "engine_version": SPEC_VERSION,
        "model_id": "org/nli",
        "model_revision": _SHA,
        "dataset_id": "org/ds",
        "split": "test",
        "column_mapping": {"p": "premises", "h": "hypothesis"},
        "template": "{p}",
        "hyperparams": _SCORING | hyperparams,
        "output_repo_id": "org/scores",
        "progress_path": "progress.jsonl",
        "run_id": "run1",
    }


def _spec(**hyperparams: Any) -> ir.RunSpec:
    return ir.RunSpec.model_validate(_spec_dict(**hyperparams))


# ------------------------------ known values ------------------------------


def test_softmax_matches_its_closed_form() -> None:
    assert sc.softmax([0.0, math.log(3)]) == pytest.approx([0.25, 0.75], abs=1e-15)
    # Stable where a naive exp overflows.
    assert sc.softmax([1000.0, 1000.0]) == [0.5, 0.5]
    assert sum(sc.softmax([2.0, -1.0, 0.5])) == pytest.approx(1.0, abs=1e-15)


_NLI3 = {2: "neutral", 0: "contradiction", 1: "entailment"}
_IDENTITY3 = {"contradiction": "contradiction", "entailment": "entailment", "neutral": "neutral"}


def test_labels_are_read_by_class_id_not_dictionary_order() -> None:
    names, outputs = sc.resolve_labels(_NLI3, _IDENTITY3)
    # The cross-encoder order: a dictionary walk would have put neutral first.
    assert names == outputs == ["contradiction", "entailment", "neutral"]


def test_string_class_ids_are_accepted() -> None:
    assert sc.resolve_labels({"1": "b", "0": "a"}, {"a": "a", "b": "b"}) == (
        ["a", "b"],
        ["a", "b"],
    )


def test_an_alias_renames_every_class_in_class_id_order() -> None:
    names, outputs = sc.resolve_labels(
        {0: "LABEL_0", 1: "LABEL_1"}, {"LABEL_1": "not_entailed", "LABEL_0": "entailed"}
    )
    assert names == ["LABEL_0", "LABEL_1"]
    assert outputs == ["entailed", "not_entailed"]


def test_null_excludes_a_class() -> None:
    alias = {"contradiction": "contradicted", "entailment": "entailed", "neutral": None}
    assert sc.resolve_labels(_NLI3, alias)[1] == ["contradicted", "entailed", None]


@pytest.mark.parametrize(
    ("alias", "match"),
    [
        ({"entailment": "e", "neutral": "n"}, "missing: \\['contradiction'\\]"),
        ({**_IDENTITY3, "other": "o"}, "not in the model: \\['other'\\]"),
        ({"contradiction": "x", "entailment": "x", "neutral": "n"}, "combining classes"),
        ({"contradiction": "c", "entailment": None, "neutral": None}, "at least two"),
        ({"contradiction": "c d", "entailment": "e", "neutral": "n"}, "cannot name a column"),
    ],
)
def test_an_alias_must_name_exactly_the_models_classes(
    alias: dict[str, str | None], match: str
) -> None:
    with pytest.raises(RunError, match=match):
        sc.resolve_labels(_NLI3, alias)


@pytest.mark.parametrize(
    ("id2label", "match"),
    [
        ({0: "a"}, "at least two"),
        ({0: "a", 2: "b"}, "0..n-1"),
        ({0: "a", 1: "a"}, "repeats"),
        ({0: "a", True: "b"}, "integer class id"),
        ({0: "a", 1: 7}, "integer class id"),
    ],
)
def test_an_unusable_id2label_is_refused(id2label: dict[Any, Any], match: str) -> None:
    with pytest.raises(RunError, match=match):
        sc.resolve_labels(id2label, {"a": "a", "b": "b"})


@pytest.mark.parametrize(
    ("element", "kind"),
    [
        ({"text": "All cats purr."}, "dict"),
        (["nested"], "list"),
        (True, "bool"),
        (None, "NoneType"),
    ],
)
def test_a_list_of_non_scalars_is_refused_not_rendered_as_python_text(
    element: object, kind: str
) -> None:
    row = {"premises": ["Tom is a cat.", element]}
    with pytest.raises(
        RunError, match=f"column 'premises' is a list of {kind} values.*list<string>"
    ):
        sc.render_scoring_text("{p}", row, {"p": "premises"}, separator="\n")


def test_a_list_of_numbers_joins_as_text() -> None:
    row = {"premises": ["a", 2, 1.5]}
    assert sc.render_scoring_text("{p}", row, {"p": "premises"}, separator=" ") == "a 2 1.5"


def test_a_list_column_is_joined_and_nothing_is_truncated() -> None:
    row = {"premises": ["All cats purr.", "Tom is a cat."], "hypothesis": "Tom purrs."}
    mapping = {"p": "premises", "h": "hypothesis", "x": "absent"}
    text = sc.render_scoring_text("{p} | {x}", row, mapping, separator="\n")
    assert text == "All cats purr.\nTom is a cat. | {x}"
    long_row = {"premises": "a" * 300_000}
    assert len(sc.render_scoring_text("{p}", long_row, mapping, separator="\n")) == 300_000


def test_batches_drop_overflow_and_run_longest_first() -> None:
    batches, overflow = sc.plan_batches([5, None, 3, 13, 5, 12], max_length=12, batch_size=2)
    assert overflow == [1, 3]  # over the character cap, and one token over the limit
    assert batches == [[5, 0], [4, 2]]  # 12, then the tie 5/5 in row order, then 3


def test_results_rows_carry_raw_logits_probabilities_and_the_argmax() -> None:
    inputs = sc.ScoringInputs(
        texts=["a", "b", "c"], pairs=["x", "y", "z"], item_ids=["i0", "i1", "i2"], slices=None
    )
    logits = {0: [0.0, math.log(3)], 2: [1.0, 1.0]}
    rows = sc.result_rows([4, 99, 4], logits, ["no", "yes"], inputs)

    assert rows[0] == {
        "custom_id": "row-0",
        "item_id": "i0",
        "input_tokens": 4,
        "overflow": False,
        "pred": "yes",
        "logit_no": 0.0,
        "logit_yes": math.log(3),
        "p_no": pytest.approx(0.25, abs=1e-15),
        "p_yes": pytest.approx(0.75, abs=1e-15),
    }
    # An overflow row keeps its id and length but carries no score, never a guessed one.
    assert rows[1]["overflow"] is True
    assert rows[1]["pred"] is None
    assert {rows[1][k] for k in ("logit_no", "logit_yes", "p_no", "p_yes")} == {None}
    # A tie goes to the lowest class id, deterministically.
    assert rows[2]["pred"] == "no"
    assert all(list(row) == list(rows[0]) for row in rows)


def test_an_excluded_class_leaves_the_softmax_and_the_argmax() -> None:
    inputs = sc.ScoringInputs(texts=["a"], pairs=["x"], item_ids=None, slices=None)
    # neutral would win the argmax; excluded, the softmax runs over the other two alone.
    (row,) = sc.result_rows([4], {0: [0.0, 9.0, math.log(3)]}, ["no", None, "yes"], inputs)
    assert "logit_None" not in row
    assert sorted(k for k in row if k.startswith(("logit_", "p_"))) == [
        "logit_no",
        "logit_yes",
        "p_no",
        "p_yes",
    ]
    assert row["pred"] == "yes"
    assert row["p_yes"] == pytest.approx(0.75, abs=1e-15)
    assert row["logit_yes"] == math.log(3)


class _Cfg:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def test_a_padding_offset_model_loses_the_offset_positions() -> None:
    # RoBERTa-style: 514 position slots, ids start after pad_token_id=1, so 512 tokens fit.
    config = _Cfg(model_type="roberta", max_position_embeddings=514, pad_token_id=1)
    no_tokenizer_limit = _Cfg(model_max_length=int(1e30))
    assert sc.length_limit(config, no_tokenizer_limit, None) == (512, "model")  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RunError, match="own limit of 512"):
        sc.length_limit(config, no_tokenizer_limit, 514)  # pyright: ignore[reportPrivateUsage]


def test_other_models_keep_their_position_table() -> None:
    bert = _Cfg(model_type="bert", max_position_embeddings=512, pad_token_id=0)
    assert sc.length_limit(bert, _Cfg(), None) == (512, "model")  # pyright: ignore[reportPrivateUsage]
    # The tokenizer's declared limit wins when it is smaller.
    assert sc.length_limit(bert, _Cfg(model_max_length=256), None) == (256, "model")  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RunError, match=r"set hyperparams\.max_model_len"):
        sc.length_limit(_Cfg(), _Cfg(), None)  # pyright: ignore[reportPrivateUsage]


def test_the_batch_plan_record_is_deterministic() -> None:
    record = sc.batch_plan_record([[3, 1], [2]], 2)
    assert record == {
        "order": "token_length_desc_then_row_index",
        "batch_size": 2,
        "batches": 2,
        "sha256": sc.hashlib.sha256(b"[[3,1],[2]]").hexdigest(),
    }


def test_counts_are_kept_per_slice() -> None:
    rows = [
        {"slice": "deep", "overflow": True},
        {"slice": "deep", "overflow": False},
        {"slice": "shallow", "overflow": False},
    ]
    assert sc.tally(rows) == {
        "rows": 3,
        "scored": 2,
        "overflow": 1,
        "by_slice": {
            "deep": {"rows": 2, "scored": 1, "overflow": 1},
            "shallow": {"rows": 1, "scored": 1, "overflow": 0},
        },
    }


# ------------------------------ inputs ------------------------------


def test_inputs_read_ids_and_slices() -> None:
    spec = _spec(id_column="uid", slice_column="tier")
    rows = [
        {"premises": ["p1", "p2"], "hypothesis": "h", "uid": "a", "tier": 1},
        {"premises": ["q"], "hypothesis": "g", "uid": 7, "tier": None},
    ]
    inputs = sc.build_inputs(spec, rows)
    assert inputs.texts == ["p1\np2", "q"]
    assert inputs.pairs == ["h", "g"]
    assert inputs.item_ids == ["a", "7"]
    assert inputs.slices == ["1", "(null)"]


@pytest.mark.parametrize(
    ("ids", "match"),
    [
        (["a", "a"], "repeats"),
        (["a", 1.0], "string or an integer"),
        (["a", None], "string or an integer"),
        (["a", True], "string or an integer"),
        (["a", ""], "empty or over-long"),
    ],
)
def test_ids_that_could_not_be_joined_on_are_refused(ids: list[Any], match: str) -> None:
    rows = [{"premises": "p", "hypothesis": "h", "uid": value} for value in ids]
    with pytest.raises(RunError, match=match):
        sc.build_inputs(_spec(id_column="uid"), rows)


def test_a_missing_id_or_slice_column_is_named() -> None:
    rows = [{"premises": "p", "hypothesis": "h"}]
    with pytest.raises(RunError, match="id_column 'uid' is not a column"):
        sc.build_inputs(_spec(id_column="uid"), rows)
    with pytest.raises(RunError, match="slice_column 'tier' is not a column"):
        sc.build_inputs(_spec(slice_column="tier"), rows)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([1, "1"], "both int and str"),
        (["(null)"], "the key for a missing value"),
        (["x" * 65], "over 64 characters"),
    ],
)
def test_slice_values_that_would_merge_are_refused(values: list[Any], match: str) -> None:
    rows = [{"premises": "p", "hypothesis": "h", "tier": v} for v in values]
    with pytest.raises(RunError, match=match):
        sc.build_inputs(_spec(slice_column="tier"), rows)


def test_too_many_slices_are_refused() -> None:
    rows = [{"premises": "p", "hypothesis": "h", "tier": i} for i in range(300)]
    with pytest.raises(RunError, match="300 distinct values"):
        sc.build_inputs(_spec(slice_column="tier"), rows)


# ------------------------------ the spec ------------------------------


def test_a_scoring_spec_validates() -> None:
    hp = _hp(
        label_alias={"ENTAILMENT": "entailed", "NOT_ENTAILMENT": "not_entailed"},
        id_column="uid",
        slice_column="tier",
        scoring_batch_size=64,
        max_model_len=512,
        dtype="bfloat16",
        row_limit=10,
    )
    assert hp.engine == "hf_classify"


def test_null_in_the_alias_excludes_a_class() -> None:
    alias = {"contradiction": "contradicted", "entailment": "entailed", "neutral": None}
    assert _hp(label_alias=alias).label_alias == alias


@pytest.mark.parametrize("missing", ["dtype", "label_alias"])
def test_dtype_and_the_alias_are_required(missing: str) -> None:
    body = {k: v for k, v in _SCORING.items() if k != missing}
    with pytest.raises(ValueError, match=f"needs {missing}"):
        ir.Hyperparams.model_validate(body)


def test_the_chat_engine_stays_the_default() -> None:
    assert ir.Hyperparams().engine == "vllm_chat"


@pytest.mark.parametrize(
    "field", [{"pair_template": "{h}"}, {"id_column": "uid"}, {"slice_column": "t"}]
)
def test_scoring_knobs_are_refused_on_the_chat_engine(field: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="apply only to engine 'hf_classify'"):
        ir.Hyperparams.model_validate(field)


@pytest.mark.parametrize(
    "field",
    [
        {"temperature": 0.0},
        {"max_tokens": 5},
        {"seed": 1},
        {"concurrency": 4},
        {"tensor_parallel_size": 2},
    ],
)
def test_generation_knobs_are_refused_on_the_scoring_engine(field: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="generates no text"):
        _hp(**field)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ({"pair_template": None}, "needs pair_template"),
        ({"dtype": None}, "needs dtype, one of float32, bfloat16"),
        ({"dtype": "float16"}, "needs dtype"),
        ({"dtype": "auto"}, "needs dtype"),
        ({"label_alias": None}, "needs label_alias"),
        ({"label_alias": {"a": "x y", "b": "z"}}, "distinct label"),
        ({"label_alias": {"a": "x", "b": "x"}}, "cannot share a label"),
        ({"label_alias": {"a": "x", "b": None}}, "at least two classes included"),
        ({"scoring_batch_size": 0}, "greater than or equal to 1"),
    ],
)
def test_an_unusable_scoring_spec_is_refused(field: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _hp(**field)


def test_the_pair_template_must_be_covered_by_the_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec_dict()
    spec["hyperparams"]["pair_template"] = "{hypothesis}"
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(spec))
    with pytest.raises(RunError, match=r"\['hypothesis'\]"):
        ir.load_spec()


# ------------------------------ the run ------------------------------


class _FakeClassifier:
    """Stands in for a loaded model: token length = word count, logits from the text."""

    loaded: ClassVar[list[dict[str, Any]]] = []

    def __init__(self) -> None:
        self.names = ["ENTAILMENT", "NOT_ENTAILMENT"]
        self.outputs: list[str | None] = ["entailed", "not_entailed"]
        self.labels = ["entailed", "not_entailed"]
        self.max_length = 8
        self.max_length_source = "model"
        self.dtype = "float32"
        self.tf32 = {"matmul": False, "cudnn": False}
        self.device = "cpu"
        self.pad_token = "tokenizer"
        self.unexpected_keys = ["deberta.embeddings.position_ids"]
        self.batches: list[list[str]] = []

    def lengths(self, texts: Sequence[str], pairs: Sequence[str]) -> list[int | None]:
        return [len(t.split()) + len(p.split()) + 3 for t, p in zip(texts, pairs, strict=True)]

    def logits(self, texts: Sequence[str], pairs: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(t)), float(len(p))] for t, p in zip(texts, pairs, strict=True)]


class _Hub:
    """The Hub as the scoring path sees it: visibility, the repo's files and one upload."""

    files: ClassVar[tuple[str, ...]] = (".gitattributes",)
    calls: ClassVar[list[tuple[str, Any]]] = []

    def __init__(self, *, token: Any) -> None:
        del token

    async def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        _Hub.calls.append(("create_repo", repo_id))

    async def is_private(self, repo_id: str, **_kw: Any) -> bool:
        return True

    async def list_repo_files(self, repo_id: str, **_kw: Any) -> tuple[str, ...]:
        _Hub.calls.append(("list_repo_files", repo_id))
        return _Hub.files

    async def upload_files(self, files: Any, repo_id: str, **kwargs: Any) -> None:
        _Hub.calls.append(("upload_files", (sorted(files), repo_id)))


_ROWS = [
    {"premises": ["all", "cats", "purr"], "hypothesis": "tom", "uid": "u0", "tier": "easy"},
    {"premises": ["a", "b", "c", "d"], "hypothesis": "too long", "uid": "u1", "tier": "hard"},
    {"premises": ["x"], "hypothesis": "y", "uid": "u2", "tier": "hard"},
]


@pytest.fixture
def scoring_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Wire `main` for a scoring run; return what the run touched."""
    record: dict[str, Any] = {"classifier": _FakeClassifier(), "rows": list(_ROWS)}
    monkeypatch.delenv(LEGACY_TOKEN_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    secrets = tmp_path / ".secrets.json"
    secrets.write_text(json.dumps({"HF_TOKEN": _TOKEN}))
    secrets.chmod(0o600)
    monkeypatch.setenv("FORGE_SECRETS_FILE", str(secrets))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    _Hub.files = (".gitattributes",)
    _Hub.calls = []
    monkeypatch.setattr(ir, "HFHubClient", _Hub)

    def _rows(spec: Any, token: Any) -> list[dict[str, Any]]:
        record["token"] = token
        record["loaded_at"] = spec.dataset_commit_sha
        return record["rows"]

    async def _commit(spec: Any, token: Any) -> str:
        return spec.dataset_commit_sha or _DATA_SHA

    async def _fetch(spec: Any, token: Any) -> Path:
        snapshot = tmp_path / "models--org--nli" / "snapshots" / _SHA
        snapshot.mkdir(parents=True, exist_ok=True)
        return snapshot

    def _load(snapshot: Path, **kwargs: Any) -> _FakeClassifier:
        record["load"] = kwargs
        return record["classifier"]

    def _write(rows: list[dict[str, Any]], outdir: Path) -> Path:
        record["written"] = rows
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / "results.parquet"
        path.write_bytes(b"")
        return path

    def _no_server(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("a scoring run starts no model server")

    monkeypatch.setattr(ir, "_load_rows", _rows)
    monkeypatch.setattr(ir, "_dataset_commit", _commit)
    monkeypatch.setattr(ir, "_fetch_model", _fetch)
    monkeypatch.setattr(ir, "load_classifier", _load)
    monkeypatch.setattr(ir, "_write_results", _write)
    monkeypatch.setattr(ir, "serving_endpoint", _no_server)
    record["progress"] = tmp_path / "progress.jsonl"
    record["results_dir"] = tmp_path / "strata-inference-results" / "run1"
    return record


def _launch(monkeypatch: pytest.MonkeyPatch, record: dict[str, Any], **hyperparams: Any) -> None:
    spec = _spec_dict(**hyperparams) | {"progress_path": str(record["progress"])}
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(spec))


def _events(record: dict[str, Any]) -> list[dict[str, Any]]:
    lines = record["progress"].read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def test_a_scoring_run_drops_overflow_counts_it_and_pushes_one_commit(
    monkeypatch: pytest.MonkeyPatch, scoring_run: dict[str, Any]
) -> None:
    _launch(
        monkeypatch,
        scoring_run,
        id_column="uid",
        slice_column="tier",
        label_alias={"ENTAILMENT": "entailed", "NOT_ENTAILMENT": "not_entailed"},
        scoring_batch_size=1,
    )
    assert await ir.main() == 0

    written = scoring_run["written"]
    assert [row["item_id"] for row in written] == ["u0", "u1", "u2"]
    assert [row["overflow"] for row in written] == [False, True, False]
    # Row 1 is 4 + 2 + 3 = 9 tokens over a limit of 8: no score, and nothing was truncated to fit.
    assert written[1]["input_tokens"] == 9
    assert written[1]["pred"] is None
    assert written[1]["logit_entailed"] is None
    # The raw logits are exactly what the head returned for the joined premises.
    assert written[0]["logit_entailed"] == float(len("all\ncats\npurr"))
    assert written[0]["logit_not_entailed"] == float(len("tom"))
    assert written[0]["pred"] == "entailed"
    # Longest first, one per batch; the overflow row never reached the model.
    assert scoring_run["classifier"].batches == [["all\ncats\npurr"], ["x"]]
    assert scoring_run["load"] == {
        "label_alias": {"ENTAILMENT": "entailed", "NOT_ENTAILMENT": "not_entailed"},
        "max_length": None,
        "dtype": "float32",
    }
    assert scoring_run["token"] == _TOKEN
    # Unpinned: the head is resolved once, then the split is loaded at exactly that commit.
    assert scoring_run["loaded_at"] == _DATA_SHA

    events = _events(scoring_run)
    end = next(e for e in events if e["kind"] == "end")
    assert end["metrics"] == {"succeeded": 2.0, "failed": 0.0, "overflow": 1.0}
    assert end["message"] == "org/scores"
    assert [e["message"] for e in events if e["kind"] == "phase"] == [
        "Checking the output repo",
        "Loading the dataset",
        "Downloading the model",
        "Loading the classifier",
        "Measuring rows",
        "Scoring rows",
        "Writing results",
        "Uploading results to the Hub",
    ]
    uploads = [call for call in _Hub.calls if call[0] == "upload_files"]
    assert uploads == [("upload_files", (["manifest.json", "results.parquet"], "org/scores"))]

    raw = (scoring_run["results_dir"] / "manifest.json").read_text()
    manifest = json.loads(raw)
    assert manifest["kind"] == "scoring"
    assert manifest["model"] == {"id": "org/nli", "requested_revision": _SHA, "commit": _SHA}
    assert manifest["dataset"]["requested_revision"] is None
    assert manifest["dataset"]["commit"] == _DATA_SHA
    assert manifest["scoring"]["labels"] == ["entailed", "not_entailed"]
    assert manifest["scoring"]["id2label"] == {"0": "ENTAILMENT", "1": "NOT_ENTAILMENT"}
    assert manifest["scoring"]["label_alias"] == _ALIAS
    assert manifest["scoring"]["excluded"] == []
    assert manifest["scoring"]["overflow_policy"] == "drop_and_count"
    assert manifest["scoring"]["calibration"] is None
    assert manifest["scoring"]["dtype"] == "float32"
    assert manifest["scoring"]["tf32"] == {"matmul": False, "cudnn": False}
    plan = manifest["scoring"]["batch_plan"]
    assert plan["order"] == "token_length_desc_then_row_index"
    assert plan["batch_size"] == 1
    assert plan["batches"] == 2
    assert plan["sha256"] == sc.hashlib.sha256(b"[[0],[2]]").hexdigest()
    # The hypothesis template is a digest only, never text in the pushed manifest.
    assert "pair_template" not in manifest["hyperparams"]
    assert manifest["scoring"]["unexpected_checkpoint_keys"] == ["deberta.embeddings.position_ids"]
    assert manifest["counts"] == {
        "rows": 3,
        "scored": 2,
        "overflow": 1,
        "by_slice": {
            "easy": {"rows": 1, "scored": 1, "overflow": 0},
            "hard": {"rows": 2, "scored": 1, "overflow": 1},
        },
    }
    assert manifest["pair_template_sha256"] == sc.hashlib.sha256(b"{h}").hexdigest()
    assert "cats" not in raw
    assert _TOKEN not in raw
    assert _TOKEN not in scoring_run["progress"].read_text()


async def test_a_run_where_every_row_overflows_fails_after_reporting(
    monkeypatch: pytest.MonkeyPatch, scoring_run: dict[str, Any]
) -> None:
    scoring_run["rows"] = [_ROWS[1]]
    _launch(monkeypatch, scoring_run)
    assert await ir.main() == 1
    events = _events(scoring_run)
    assert next(e for e in events if e["kind"] == "end")["metrics"]["overflow"] == 1.0
    assert "none was scored" in events[-1]["message"]
    assert scoring_run["written"][0]["overflow"] is True  # the evidence is still written


async def test_an_output_repo_holding_files_is_refused_before_the_dataset_loads(
    monkeypatch: pytest.MonkeyPatch, scoring_run: dict[str, Any]
) -> None:
    _Hub.files = (".gitattributes", "results.parquet", "manifest.json")
    _launch(monkeypatch, scoring_run)
    assert await ir.main() == 1
    assert "token" not in scoring_run  # never reached the dataset
    assert "repo of its own" in _events(scoring_run)[-1]["message"]


async def test_a_run_without_a_token_keeps_results_on_the_machine(
    monkeypatch: pytest.MonkeyPatch, scoring_run: dict[str, Any]
) -> None:
    monkeypatch.delenv("FORGE_SECRETS_FILE")
    _launch(monkeypatch, scoring_run)
    assert await ir.main() == 0
    assert _Hub.calls == []
    end = next(e for e in _events(scoring_run) if e["kind"] == "end")
    assert end["message"].endswith("results.parquet")
    assert scoring_run["token"] is False


async def test_an_empty_split_fails(
    monkeypatch: pytest.MonkeyPatch, scoring_run: dict[str, Any]
) -> None:
    scoring_run["rows"] = []
    _launch(monkeypatch, scoring_run)
    assert await ir.main() == 1
    assert "split was empty" in _events(scoring_run)[-1]["message"]


class _ExcludingClassifier(_FakeClassifier):
    """Three classes, the FIRST excluded: its logit must never be read as another class's."""

    def __init__(self) -> None:
        super().__init__()
        self.names = ["NEUTRAL", "ENTAILMENT", "NOT_ENTAILMENT"]
        self.outputs = [None, "entailed", "not_entailed"]

    def logits(self, texts: Sequence[str], pairs: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[1000.0, float(len(t)), float(len(p))] for t, p in zip(texts, pairs, strict=True)]


async def test_an_excluded_first_class_does_not_shift_the_columns(
    monkeypatch: pytest.MonkeyPatch, scoring_run: dict[str, Any]
) -> None:
    scoring_run["classifier"] = _ExcludingClassifier()
    _launch(
        monkeypatch,
        scoring_run,
        label_alias={"NEUTRAL": None, "ENTAILMENT": "entailed", "NOT_ENTAILMENT": "not_entailed"},
    )
    assert await ir.main() == 0
    first = scoring_run["written"][0]
    assert first["logit_entailed"] == float(len("all\ncats\npurr"))
    assert first["logit_not_entailed"] == float(len("tom"))
    assert "logit_NEUTRAL" not in first


async def test_a_trained_verifier_scores_through_the_runner(
    monkeypatch: pytest.MonkeyPatch, scoring_run: dict[str, Any], tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    datasets: Any = pytest.importorskip("datasets")
    pytest.importorskip("safetensors.torch")
    from strata_forge.training.seq_cls import SeqClsConfig, SeqClsRunner
    from tests.unit.training import tiny_classifier as tiny

    class _Probe:
        def reset(self) -> None:
            return None

        def peak_bytes(self) -> int:
            return 1

        def total_bytes(self) -> int:
            return 100

    artifact = tmp_path / "verifier"
    SeqClsRunner(
        SeqClsConfig.model_validate(
            {
                "model_id": "synthetic/tiny",
                "output_dir": str(tmp_path / "ckpt"),
                "labels": tiny.LABELS,
                **tiny.TWO_WAY,
                "max_length": 64,
                "per_device_batch_size": 8,
                "max_steps": 1,
            }
        ),
        memory_probe=_Probe(),
    ).run(
        checkpoint=tiny.encoder(tmp_path / "base"),
        artifact_dir=artifact,
        train_dataset=datasets.Dataset.from_list(tiny.rows(8)),
    )
    (artifact / "calibration.json").write_text(
        json.dumps(
            {
                "schema": "strata.calibration/v1",
                "labels": list(tiny.LABELS),
                "fit": {"temperature": 1.5},
            }
        )
    )

    async def _fetch(spec: Any, token: Any) -> Path:
        return artifact

    monkeypatch.setattr(ir, "_fetch_model", _fetch)
    scoring_run["rows"] = tiny.rows(4, seed=9)
    spec = _spec_dict() | {
        "template": "{premises}",
        "column_mapping": {"premises": "premises", "hypothesis": "hypothesis"},
        "hyperparams": {
            "engine": "hf_classify",
            "artifact": "strata_verifier",
            "pair_template": "{hypothesis}",
            "dtype": "float32",
            "apply_calibration": "auto",
        },
        "progress_path": str(scoring_run["progress"]),
    }
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(spec))
    assert await ir.main() == 0

    written = scoring_run["written"]
    assert len(written) == 4
    assert all(row["overflow"] is False for row in written)
    assert {"logit_entailment", "p_entailment", "p_inconsistent", "support_scores"} <= set(
        written[0]
    )
    manifest = json.loads((scoring_run["results_dir"] / "manifest.json").read_text())
    scoring = manifest["scoring"]
    assert scoring["input"] == "strata_verifier"
    assert scoring["artifact"]["heads"] == ["flag", "support"]
    assert scoring["artifact"]["rendering"]["encoding"] == "pair"
    assert scoring["calibration"]["applied"] is True
    assert scoring["calibration"]["temperature"] == 1.5
    assert scoring["calibration"]["raw_logits_kept"] is True
    assert scoring["max_length_source"] == "artifact"
