"""The ``hf_label_logits`` scoring engine, its chat-template kwargs and its path through the runner.

The pure parts (the spec's rules, the kwargs' canonical form and digest, the template's variables,
the mass outside the labels, the prefill's token check) are tested with known values. The
orchestration runs the real ``main`` with the heavy externals replaced. The real transformers
path is :func:`check_hf_label_logits` in the clean-install smoke, run here when torch is installed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from strata_forge.pipelines import SPEC_VERSION
from strata_forge.pipelines import _chat_template as ct
from strata_forge.pipelines import _label_logits as ll
from strata_forge.pipelines import _scoring as sc
from strata_forge.pipelines import inference_runner as ir
from strata_forge.pipelines._common import LEGACY_TOKEN_ENV, RunError

if TYPE_CHECKING:
    from pydantic import JsonValue

_TOKEN = "hf_secretwritetoken1234567890"
_SHA = "0123456789abcdef0123456789abcdef01234567"
_LABELS = [{"label": "true", "token": " True"}, {"label": "false", "token": " False"}]


def _hp(**overrides: Any) -> ir.Hyperparams:
    base: dict[str, Any] = {
        "engine": "hf_label_logits",
        "label_tokens": _LABELS,
        "dtype": "bfloat16",
    }
    return ir.Hyperparams.model_validate(base | overrides)


def _spec_dict(**hyperparams: Any) -> dict[str, Any]:
    return {
        "engine_version": SPEC_VERSION,
        "model_id": "org/judge",
        "model_revision": _SHA,
        "dataset_id": "org/ds",
        "split": "test",
        "column_mapping": {"p": "premises", "h": "hypothesis"},
        "template": "{h}",
        "hyperparams": {"engine": "hf_label_logits", "label_tokens": _LABELS, "dtype": "bfloat16"}
        | hyperparams,
        "output_repo_id": "org/scores",
        "progress_path": "progress.jsonl",
        "run_id": "run1",
    }


# ------------------------------ the spec ------------------------------


def test_a_label_logits_spec_validates() -> None:
    hp = _hp(
        assistant_prefill="<score>",
        chat_template_kwargs={
            "documents": [{"doc_id": "0", "text": "{p}"}],
            "guardian_config": {"criteria_id": "groundedness"},
            "think": False,
        },
        id_column="uid",
        slice_column="tier",
        scoring_batch_size=8,
        max_model_len=4096,
        dtype="bfloat16",
        row_limit=10,
    )
    assert hp.label_tokens is not None
    assert [entry.token for entry in hp.label_tokens] == [" True", " False"]


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ({"label_tokens": None}, "needs label_tokens"),
        ({"label_tokens": _LABELS[:1]}, "at least 2"),
        ({"label_tokens": [_LABELS[0], {"label": "true", "token": " Yes"}]}, "distinct label"),
        ({"label_tokens": [_LABELS[0], {"label": "other", "token": " True"}]}, "distinct token"),
        ({"label_tokens": [_LABELS[0], {"label": "bad label", "token": " x"}]}, "a label is"),
        ({"label_tokens": [_LABELS[0], {"label": "x", "token": ""}]}, "at least 1 character"),
        ({"assistant_prefill": ""}, "at least 1 character"),
        ({"dtype": "int8"}, "needs dtype, one of float32, bfloat16"),
        ({"dtype": None}, "needs dtype"),
        ({"chat_template_kwargs": {}}, "is empty"),
        ({"chat_template_kwargs": {"chat_template": "{{ x }}"}}, "cannot set"),
        ({"chat_template_kwargs": {"add_generation_prompt": False}}, "cannot set"),
        ({"chat_template_kwargs": {"bos_token": "x"}}, "cannot set"),
        ({"chat_template_kwargs": {"bad-key": 1}}, "identifiers"),
    ],
)
def test_an_unusable_label_logits_spec_is_refused(field: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _hp(**field)


@pytest.mark.parametrize(
    ("field", "owner"),
    [
        ({"pair_template": "{h}"}, "apply only to engine 'hf_classify'"),
        ({"label_alias": {"a": "x", "b": "y"}}, "apply only to engine 'hf_classify'"),
    ],
)
def test_classifier_knobs_are_refused_naming_their_engine(
    field: dict[str, Any], owner: str
) -> None:
    with pytest.raises(ValueError, match=f"takes no .*{owner}"):
        _hp(**field)


def test_label_knobs_are_refused_on_the_classifier_engine() -> None:
    with pytest.raises(ValueError, match="apply only to engine 'hf_label_logits'"):
        ir.Hyperparams.model_validate(
            {
                "engine": "hf_classify",
                "pair_template": "{h}",
                "dtype": "float32",
                "label_alias": {"A": "a", "B": "b"},
                "assistant_prefill": "x",
            }
        )


@pytest.mark.parametrize("field", [{"label_tokens": _LABELS}, {"assistant_prefill": "<score>"}])
def test_label_knobs_are_refused_on_the_chat_engine(field: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="apply only to engine 'hf_label_logits'"):
        ir.Hyperparams.model_validate(field)


@pytest.mark.parametrize("field", [{"temperature": 0.0}, {"seed": 1}, {"max_tokens": 1}])
def test_generation_knobs_are_refused(field: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="generates no text"):
        _hp(**field)


def test_kwargs_placeholders_must_be_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec_dict(chat_template_kwargs={"documents": [{"text": "{context}"}]})
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(spec))
    with pytest.raises(RunError, match=r"\['context'\]"):
        ir.load_spec()


def test_template_placeholders_cover_every_rendered_text() -> None:
    hp = _hp(chat_template_kwargs={"documents": [{"text": "{p}"}], "x": ["{q}", 1]})
    assert ir.template_placeholders("{h} and {h2}", hp) == {"h", "h2", "p", "q"}
    classify = ir.Hyperparams.model_validate(
        {
            "engine": "hf_classify",
            "pair_template": "{b}",
            "dtype": "float32",
            "label_alias": {"A": "a", "B": "b"},
        }
    )
    assert ir.template_placeholders("{a}", classify) == {"a", "b"}


# ------------------------------ chat-template kwargs ------------------------------


def test_the_digest_ignores_key_order_at_every_depth() -> None:
    a: dict[str, JsonValue] = {"guardian_config": {"criteria_id": "g", "x": 1}, "think": False}
    b: dict[str, JsonValue] = {"think": False, "guardian_config": {"x": 1, "criteria_id": "g"}}
    assert ct.kwargs_digest(a) == ct.kwargs_digest(b)
    assert ct.kwargs_digest(a) != ct.kwargs_digest(
        {"think": True, "guardian_config": a["guardian_config"]}
    )
    assert ct.kwargs_digest(None) is None


def test_the_digest_is_sha256_of_the_canonical_json() -> None:
    import hashlib

    expected = hashlib.sha256(b'{"a":[{"b":1,"c":"\xc3\xa9"}],"z":null}').hexdigest()
    assert ct.kwargs_digest({"z": None, "a": [{"c": "é", "b": 1}]}) == expected


def test_kwargs_render_per_row_in_canonical_order() -> None:
    kwargs: dict[str, JsonValue] = {
        "think": False,
        "documents": [{"text": "{p}", "doc_id": "0"}],
    }
    rendered = ct.render_kwargs(kwargs, lambda text: text.replace("{p}", "P1"))
    assert rendered == {"documents": [{"doc_id": "0", "text": "P1"}], "think": False}
    assert list(rendered) == ["documents", "think"]
    assert json.dumps(rendered) == '{"documents": [{"doc_id": "0", "text": "P1"}], "think": false}'
    assert ct.kwargs_placeholders(kwargs) == {"p"}


def test_scoring_inputs_render_kwargs_with_list_columns_joined() -> None:
    spec = ir.RunSpec.model_validate(
        _spec_dict(chat_template_kwargs={"documents": [{"doc_id": "0", "text": "{p}"}]})
    )
    rows = [{"premises": ["a", "b"], "hypothesis": "c"}]
    inputs = sc.build_inputs(spec, rows)
    assert inputs.texts == ["c"]
    assert inputs.chat_kwargs == [{"documents": [{"doc_id": "0", "text": "a\nb"}]}]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({f"k{i}": i for i in range(ct.MAX_KWARGS + 1)}, "at most 16 keys"),
        ({"deep": [[[[[[[1]]]]]]]}, "nest at most"),
        ({"big": "x" * ct.MAX_KWARGS_JSON_BYTES}, "bytes of JSON"),
        ({"tools": []}, "cannot set"),
        ({"messages": []}, "cannot set"),
    ],
)
def test_kwargs_problems(kwargs: dict[str, Any], match: str) -> None:
    problem = ct.kwargs_problem(kwargs)
    assert problem is not None
    assert match in problem


def test_kwargs_nested_past_the_interpreters_limit_are_refused_not_crashed() -> None:
    deep: Any = "x"
    for _ in range(5_000):
        deep = [deep]
    problem = ct.kwargs_problem({"documents": deep})
    assert problem is not None
    assert "nest at most" in problem
    assert ct.nests_deeper_than([[1]], 2) is False
    assert ct.nests_deeper_than([[[1]]], 2) is True
    assert ct.nests_deeper_than({"a": [{"b": 1}]}, 3) is False


def test_documents_is_data_not_a_renderer_option() -> None:
    assert ct.kwargs_problem({"documents": [{"text": "x"}]}) is None


def test_template_variables_are_what_the_template_reads() -> None:
    template = (
        "{%- set flag = 'no_think' %}{%- if think %}{%- set flag = 'think' %}{%- endif %}"
        "{%- if guardian_config %}{{ guardian_config['criteria_id'] }}{%- endif %}"
        "{%- for d in documents %}{{ d['text'] }}{%- endfor %}"
        "{%- for m in messages %}{% generation %}{{ m['content'] }}{% endgeneration %}{%- endfor %}"
        "{%- if add_generation_prompt %}<a>{{ flag }}{%- endif %}"
    )
    variables = ct.template_variables(template)
    assert {
        "think",
        "guardian_config",
        "documents",
        "messages",
        "add_generation_prompt",
    } <= variables
    assert "flag" not in variables
    assert ct.unread_kwargs({"think": False, "enable_thinking": False}, template) == [
        "enable_thinking"
    ]
    assert ct.unread_kwargs(None, template) == []


# ------------------------------ the read-out ------------------------------


def test_mass_outside_is_the_full_softmax_off_the_labels() -> None:
    declared = [2.0, 1.0]
    others = [0.0, -1.0, 3.0]
    normalizer = math.log(math.fsum(math.exp(v) for v in declared + others))
    expected = math.fsum(math.exp(v) for v in others) / math.exp(normalizer)
    assert ll.read_out(declared, normalizer) == pytest.approx(expected, rel=1e-12)


def test_mass_outside_keeps_its_precision_near_zero_and_never_goes_negative() -> None:
    declared = [50.0, 10.0]
    normalizer = math.log(math.exp(50.0) + math.exp(10.0) + math.exp(-30.0))
    assert ll.read_out(declared, normalizer) == pytest.approx(math.exp(-80.0), rel=1e-6)
    # A normalizer rounded below the declared share would make the mass negative.
    assert ll.read_out([1.0, 1.0], math.log(2.0) - 1e-12 + 1.0) == 0.0


def test_result_rows_add_mass_outside_beside_the_classifier_columns() -> None:
    reader = _reader(_CharTokenizer())
    inputs = sc.ScoringInputs(texts=["a", "b"], pairs=["", ""], item_ids=None, slices=None)
    normalizer = math.log(math.exp(1.0) + math.exp(0.0) + math.exp(-1.0))
    rows = reader.result_rows([3, None], {0: [1.0, 0.0, normalizer]}, inputs)
    assert rows[0]["pred"] == "true"
    assert rows[0]["logit_true"] == 1.0
    assert rows[0]["p_true"] == pytest.approx(math.exp(1.0) / (math.exp(1.0) + 1.0))
    assert rows[0]["mass_outside"] == pytest.approx(math.exp(-1.0) / math.exp(normalizer))
    assert rows[1]["overflow"] is True
    assert rows[1]["mass_outside"] is None
    assert rows[1]["logit_true"] is None


# ------------------------------ the prefill's tokens ------------------------------


class _CharTokenizer:
    """One token per character, except that ``"ab"`` is a single token, so it can merge."""

    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return f"[{messages[0]['content']}]{kwargs.get('tail', '')}"

    @staticmethod
    def _one(text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text.startswith("ab", i):
                ids.append(1000)
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def __call__(self, texts: Any, **_kw: Any) -> dict[str, Any]:
        if isinstance(texts, str):
            return {"input_ids": self._one(texts)}
        return {"input_ids": [self._one(text) for text in texts]}


def _reader(tokenizer: Any, prefill: str = "") -> ll.LabelReader:
    return ll.LabelReader(
        model=None,
        tokenizer=tokenizer,
        torch=None,
        labels=["true", "false"],
        tokens=["T", "F"],
        token_ids=[ord("T"), ord("F")],
        token_pieces=["T", "F"],
        prefill=prefill,
        chat_template="",
        vocab_size=2000,
        max_length=64,
        max_length_source="model",
        dtype="bfloat16",
        tf32={"matmul": None, "cudnn": None},
        device="cpu",
        unexpected_keys=[],
    )


def _inputs(texts: list[str], tail: str = "") -> sc.ScoringInputs:
    kwargs: dict[str, JsonValue] = {"tail": tail}
    return sc.ScoringInputs(
        texts=texts,
        pairs=[""] * len(texts),
        item_ids=None,
        slices=None,
        chat_kwargs=[kwargs] * len(texts) if tail else None,
    )


def test_the_prefill_is_counted_and_its_tokens_recorded() -> None:
    reader = _reader(_CharTokenizer(), prefill="<s>")
    assert reader.lengths(_inputs(["x", "yz"])) == [3 + 3, 4 + 3]
    assert reader.prefill_ids == [ord("<"), ord("s"), ord(">")]


def test_a_prefill_that_merges_with_the_prompt_is_refused() -> None:
    # The prompt ends in "a" and the prefill starts with "b": tokenized together they are "ab".
    reader = _reader(_CharTokenizer(), prefill="b")
    with pytest.raises(RunError, match="merges with the end of the rendered prompt"):
        reader.lengths(_inputs(["x"], tail="a"))


def test_a_label_token_that_merges_with_the_prompt_is_refused() -> None:
    reader = _reader(_CharTokenizer(), prefill="a")
    reader.tokens = ["b", "F"]
    reader.token_ids = [ord("b"), ord("F")]
    with pytest.raises(RunError, match="not a single-token continuation"):
        reader.lengths(_inputs(["x"]))


def test_a_template_error_stops_the_run() -> None:
    class _Raising(_CharTokenizer):
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            msg = "provided 'criteria_id' is incorrect."
            raise ValueError(msg)

    with pytest.raises(RunError, match="criteria_id"):
        _reader(_Raising()).lengths(_inputs(["x"]))


# ------------------------------ the run ------------------------------


class _FakeReader:
    """Stands in for a loaded causal LM: length = characters, logits from the text."""

    loaded: ClassVar[dict[str, Any]] = {}

    def __init__(self) -> None:
        self.max_length = 6
        self.labels = ["true", "false"]

    def lengths(self, inputs: sc.ScoringInputs) -> list[int | None]:
        return [len(text) for text in inputs.texts]

    def score(self, inputs: sc.ScoringInputs, batch: Any) -> list[list[float]]:
        return [[float(len(inputs.texts[i])), 0.0, 10.0] for i in batch]

    def result_rows(
        self, lengths: Any, values: Any, inputs: sc.ScoringInputs
    ) -> list[dict[str, Any]]:
        return ll.LabelReader.result_rows(self, lengths, values, inputs)  # pyright: ignore[reportArgumentType]

    def manifest(self, spec: Any, rows: Any, **pins: Any) -> dict[str, Any]:
        _FakeReader.loaded["pins"] = pins
        return sc.manifest_frame(
            spec,
            rows,
            model_commit=pins["model_commit"],
            dataset_commit=pins["dataset_commit"],
            scoring={"engine": "hf_label_logits"},
        )


class _Hub:
    calls: ClassVar[list[tuple[str, Any]]] = []

    def __init__(self, *, token: Any) -> None:
        del token

    async def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        _Hub.calls.append(("create_repo", repo_id))

    async def is_private(self, repo_id: str, **_kw: Any) -> bool:
        return True

    async def list_repo_files(self, repo_id: str, **_kw: Any) -> tuple[str, ...]:
        return (".gitattributes",)

    async def upload_files(self, files: Any, repo_id: str, **kwargs: Any) -> None:
        _Hub.calls.append(("upload_files", (sorted(files), repo_id)))


async def test_a_label_logits_run_drops_overflow_and_pushes_one_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record: dict[str, Any] = {}
    monkeypatch.delenv(LEGACY_TOKEN_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    secrets = tmp_path / ".secrets.json"
    secrets.write_text(json.dumps({"HF_TOKEN": _TOKEN}))
    secrets.chmod(0o600)
    monkeypatch.setenv("FORGE_SECRETS_FILE", str(secrets))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    _Hub.calls = []
    monkeypatch.setattr(ir, "HFHubClient", _Hub)
    rows = [
        {"premises": ["p"], "hypothesis": "abc", "uid": "u0"},
        {"premises": ["p"], "hypothesis": "too long", "uid": "u1"},
    ]
    monkeypatch.setattr(ir, "_load_rows", lambda *_a: rows)  # pyright: ignore[reportUnknownLambdaType]

    async def _fetch(spec: Any, token: Any) -> Path:
        snapshot = tmp_path / "snap"
        snapshot.mkdir(exist_ok=True)
        return snapshot

    def _load(snapshot: Path, **kwargs: Any) -> _FakeReader:
        record["load"] = kwargs
        return _FakeReader()

    def _write(out: list[dict[str, Any]], outdir: Path) -> Path:
        record["written"] = out
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / "results.parquet"
        path.write_bytes(b"")
        return path

    def _no_server(*_a: Any, **_kw: Any) -> Any:
        pytest.fail("a scoring run starts no model server")

    async def _commit(spec: Any, token: Any) -> str:
        return _SHA

    monkeypatch.setattr(ir, "_dataset_commit", _commit)
    monkeypatch.setattr(ir, "_fetch_model", _fetch)
    monkeypatch.setattr(ir, "load_label_reader", _load)
    monkeypatch.setattr(ir, "_write_results", _write)
    monkeypatch.setattr(ir, "serving_endpoint", _no_server)
    spec = _spec_dict(
        assistant_prefill="<score>",
        chat_template_kwargs={"documents": [{"text": "{p}"}]},
        id_column="uid",
    ) | {"progress_path": str(tmp_path / "progress.jsonl")}
    monkeypatch.setenv("STRATA_RUN_CONFIG", json.dumps(spec))

    assert await ir.main() == 0
    assert record["load"]["prefill"] == "<score>"
    assert [entry.label for entry in record["load"]["label_tokens"]] == ["true", "false"]
    written = record["written"]
    assert [row["overflow"] for row in written] == [False, True]
    assert written[0]["item_id"] == "u0"
    assert written[0]["pred"] == "true"
    # Declared logits 3 and 0 under a full-vocabulary log-sum-exp of 10.
    assert written[0]["mass_outside"] == pytest.approx(1 - (math.exp(3) + 1) / math.exp(10))
    manifest = json.loads(
        (tmp_path / "strata-inference-results" / "run1" / "manifest.json").read_text()
    )
    assert manifest["kind"] == "scoring"
    assert manifest["counts"] == {"rows": 2, "scored": 1, "overflow": 1}
    assert manifest["dataset"]["commit"] == _SHA
    # Prompt text stays out of the pushed record; the engine's block carries the digests.
    assert "assistant_prefill" not in manifest["hyperparams"]
    assert "chat_template_kwargs" not in manifest["hyperparams"]
    assert manifest["hyperparams"]["label_tokens"][0]["label"] == "true"
    assert _FakeReader.loaded["pins"]["batches"] == [[0]]
    assert [c for c in _Hub.calls if c[0] == "upload_files"] == [
        ("upload_files", (["manifest.json", "results.parquet"], "org/scores"))
    ]


# ------------------------------ the real engine ------------------------------


def test_the_engine_reads_a_saved_causal_lm_exactly() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    script = Path(__file__).resolve().parents[3] / "scripts" / "smoke_clean_install.py"
    spec = importlib.util.spec_from_file_location("smoke_clean_install_label_logits", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.check_hf_label_logits()
