"""Offline forward-pass fakes exercise scoring without torch, downloads or credentials."""

from __future__ import annotations

import contextlib
import math
import sys
import types
from typing import TYPE_CHECKING, cast

import pytest

from strata_forge.core.errors import ValidationError
from strata_forge.pipelines import _scoring as scoring

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from strata_forge.training.progress import ProgressEvent

type _Data = int | float | list[_Data]


class Tensor:
    def __init__(self, data: _Data) -> None:
        self.data = data

    @property
    def shape(self) -> tuple[int, ...]:
        dimensions: list[int] = []
        value = self.data
        while isinstance(value, list):
            dimensions.append(len(value))
            if not value:
                break
            value = value[0]
        return tuple(dimensions)

    def __getitem__(self, key: object) -> Tensor:
        value = self.data
        keys = cast("tuple[object, ...]", key) if isinstance(key, tuple) else (key,)
        for index in keys:
            assert isinstance(value, list)
            assert isinstance(index, int | slice)
            value = value[index]
        return Tensor(value)

    def to(self, device: str) -> Tensor:
        assert device == "cpu"
        return self

    def float(self) -> Tensor:
        return self

    def detach(self) -> Tensor:
        return self

    def cpu(self) -> Tensor:
        return self

    def tolist(self) -> object:
        return self.data


class Tokenizer:
    padding_side = "left"
    pad_token_id: int | None = 0
    eos_token_id: int | None = 98

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.vocab = {"A": 1, "B": 2, "C": 3, "first": 4, "second": 5}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [self.vocab.get(word, 6) for word in text.split()]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 3 if pair else 1

    def __call__(
        self, text: list[str], text_pair: list[str] | None = None, **kwargs: object
    ) -> Mapping[str, Tensor]:
        self.calls.append({"text": text, "text_pair": text_pair, **kwargs})
        assert self.padding_side == "right"
        assert kwargs["return_tensors"] == "pt"
        sequences: list[list[int]] = []
        for index, first in enumerate(text):
            first_ids = self.encode(first, add_special_tokens=False)
            if text_pair is not None:
                assert kwargs["truncation"] == "only_first"
                pair_ids = self.encode(text_pair[index], add_special_tokens=False)
                available = cast("int", kwargs["max_length"]) - len(pair_ids) - 3
                assert available >= 0
                sequences.append([99, *first_ids[:available], 98, *pair_ids, 98])
            else:
                assert kwargs["truncation"] is False
                sequences.append([99, *first_ids])
        width = max(map(len, sequences))
        return {
            "input_ids": Tensor(
                cast("_Data", [row + [0] * (width - len(row)) for row in sequences])
            ),
            "attention_mask": Tensor(
                cast("_Data", [[1] * len(row) + [0] * (width - len(row)) for row in sequences])
            ),
        }


class Model:
    def __init__(self, labels: dict[int | str, str], *, causal: bool) -> None:
        self.config = types.SimpleNamespace(id2label=labels, pad_token_id=None)
        self.causal = causal
        self.evaluating = False
        self.forward_calls = 0
        self.override_logits: Tensor | None = None

    def eval(self) -> Model:
        self.evaluating = True
        return self

    def to(self, device: str) -> Model:
        assert device in ("cpu", "cuda")
        self.device = device
        return self

    def __call__(self, **features: Tensor) -> types.SimpleNamespace:
        assert self.evaluating
        self.forward_calls += 1
        if self.override_logits is not None:
            return types.SimpleNamespace(logits=self.override_logits)
        ids = cast("list[list[int]]", features["input_ids"].tolist())
        if self.causal:
            logits = [
                [
                    ([0.0, math.log(2), math.log(3), math.log(4)] if token else [0.0] * 4)
                    for token in row
                ]
                for row in ids
            ]
        else:
            logits = [[float(row[1]), 0.0, -1.0] for row in ids]
        return types.SimpleNamespace(logits=Tensor(cast("_Data", logits)))


@pytest.fixture
def fake_hf(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    tokenizer = Tokenizer()
    labels: dict[int | str, str] = {2: "neutral", 0: "contradiction", 1: "entailment"}
    calls: list[tuple[str, str, dict[str, object]]] = []
    models: list[Model] = []
    loading_info: dict[str, object] = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }

    def tokenizer_load(model_id: str, **kwargs: object) -> Tokenizer:
        calls.append(("tokenizer", model_id, kwargs))
        return tokenizer

    def model_load(model_id: str, **kwargs: object) -> tuple[Model, dict[str, object]]:
        calls.append(("classifier", model_id, kwargs))
        model = Model(labels, causal=False)
        models.append(model)
        return model, loading_info

    def causal_load(model_id: str, **kwargs: object) -> tuple[Model, dict[str, object]]:
        calls.append(("causal", model_id, kwargs))
        model = Model(labels, causal=True)
        models.append(model)
        return model, loading_info

    transformers = types.ModuleType("transformers")
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=tokenizer_load),
        raising=False,
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForSequenceClassification",
        types.SimpleNamespace(from_pretrained=model_load),
        raising=False,
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForCausalLM",
        types.SimpleNamespace(from_pretrained=causal_load),
        raising=False,
    )
    torch = types.ModuleType("torch")
    monkeypatch.setattr(torch, "inference_mode", contextlib.nullcontext, raising=False)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return types.SimpleNamespace(
        tokenizer=tokenizer, labels=labels, calls=calls, models=models, loading_info=loading_info
    )


async def test_classifier_uses_class_ids_not_dictionary_order(
    fake_hf: types.SimpleNamespace,
) -> None:
    scorer = await scoring.load_scorer(
        "org/model", engine="hf_classify", token=False, max_length=12
    )
    result = await scorer.score([scoring.ScoringInput(text="A", text_pair="B")])
    assert list(result.rows[0].probabilities) == ["contradiction", "entailment", "neutral"]
    assert result.rows[0].pred == "contradiction"
    assert result.rows[0].columns()["p_contradiction"] == pytest.approx(
        math.e / (math.e + 1 + math.exp(-1))
    )
    assert result.rows[0].mass_outside is None
    assert result.overflow_rows == result.failed_rows == 0
    assert fake_hf.models[0].evaluating


@pytest.mark.parametrize("engine", ["hf_classify", "hf_label_logits"])
@pytest.mark.parametrize(
    "missing_key", ["backbone.embeddings.weight", "classifier.weight", "lm_head.weight"]
)
async def test_scoring_never_accepts_an_implicitly_initialized_body_or_head(
    fake_hf: types.SimpleNamespace, engine: scoring.ScoringEngine, missing_key: str
) -> None:
    fake_hf.loading_info["missing_keys"] = {missing_key}
    with pytest.raises(scoring.ModelLoadingError, match="missing_weights"):
        await scoring.load_scorer(
            "org/model",
            engine=engine,
            token=False,
            max_length=12,
            label_tokens={"yes": "A", "no": "B"} if engine == "hf_label_logits" else None,
        )
    assert not fake_hf.models[0].evaluating
    assert fake_hf.models[0].forward_calls == 0


@pytest.mark.parametrize("failure_kind", [OSError, RuntimeError, ValueError])
async def test_factory_structural_failures_do_not_preserve_raw_error_text(
    fake_hf: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure_kind: type[Exception]
) -> None:
    def fail(model_id: str, **kwargs: object) -> object:
        raise failure_kind("private raw warning hf_synthetic_fixture_do_not_persist")

    monkeypatch.setattr(
        sys.modules["transformers"].AutoModelForSequenceClassification, "from_pretrained", fail
    )
    with pytest.raises(scoring.ModelLoadingError) as failure:
        await scoring.load_scorer("org/model", engine="hf_classify", token=False, max_length=12)
    assert failure.value.audit.codes == ("pretrained_load_failed",)
    assert "private" not in str(failure.value)
    assert "synthetic_fixture" not in failure.value.audit.model_dump_json()


@pytest.mark.parametrize(
    "aliases", [{"LABEL_0": ""}, {"": "yes"}, {"LABEL_0": "yes", "LABEL_1": "yes"}]
)
async def test_cheap_invalid_aliases_are_refused_before_tokenizer_or_weight_load(
    fake_hf: types.SimpleNamespace, aliases: dict[str, str]
) -> None:
    with pytest.raises(ValidationError):
        await scoring.load_scorer(
            "org/model", engine="hf_classify", token=False, max_length=12, label_alias=aliases
        )
    assert fake_hf.calls == []


async def test_generic_labels_require_explicit_aliases(fake_hf: types.SimpleNamespace) -> None:
    fake_hf.labels.update({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    with pytest.raises(ValidationError, match="explicit label_alias"):
        await scoring.load_scorer("org/model", engine="hf_classify", token=False, max_length=12)
    scorer = await scoring.load_scorer(
        "org/model",
        engine="hf_classify",
        token=False,
        max_length=12,
        label_alias={"LABEL_0": "contradicted", "LABEL_1": "entailed", "LABEL_2": "undetermined"},
    )
    result = await scorer.score([scoring.ScoringInput(text="A", text_pair="B")])
    assert set(result.rows[0].probabilities) == {"contradicted", "entailed", "undetermined"}


@pytest.mark.parametrize(
    ("labels", "aliases"),
    [
        ({"bad": "yes"}, None),
        ({1: "yes"}, None),
        ({0: "yes", "0": "no"}, None),
        ({0: "yes", 1: "yes"}, None),
        ({0: "yes"}, {"typo": "yes"}),
        ({0: "yes", 1: "no"}, {"yes": "same", "no": "same"}),
        ({0: "yes"}, {"yes": ""}),
    ],
)
def test_bad_label_metadata_is_refused(
    labels: dict[int | str, str],
    aliases: dict[str, str] | None,
) -> None:
    with pytest.raises(ValidationError):
        scoring._label_names(labels, aliases)  # pyright: ignore[reportPrivateUsage]


async def test_first_only_truncation_preserves_hypotheses_and_counts_refusals(
    fake_hf: types.SimpleNamespace,
) -> None:
    scorer = await scoring.load_scorer("org/model", engine="hf_classify", token=False, max_length=7)
    events: list[ProgressEvent] = []
    result = await scorer.score(
        [
            scoring.ScoringInput(text="first first first first first", text_pair="B"),
            scoring.ScoringInput(text="A", text_pair="B B B B B"),
            scoring.ScoringInput(text="A", text_pair="B B B B"),
            scoring.ScoringInput(text="A", text_pair="B"),
        ],
        batch_size=1,
        on_progress=events.append,
    )
    assert result.overflow_rows == 3
    assert result.truncated_rows == 1
    assert result.failed_rows == 2
    assert result.rows[0].truncated
    assert result.rows[1].error is not None
    assert "hypothesis_too_long" in result.rows[1].error
    assert result.rows[2].error is not None
    assert "hypothesis_too_long" in result.rows[2].error
    assert [row.index for row in result.rows] == [0, 1, 2, 3]
    assert [call["text_pair"] for call in fake_hf.tokenizer.calls] == [["B"], ["B"]]
    assert [event.step for event in events] == [2, 3, 4]
    assert events[-1].metrics["overflow_rows"] == 3
    assert events[-1].metrics["failed"] == 2


async def test_length_batches_restore_input_order(fake_hf: types.SimpleNamespace) -> None:
    scorer = await scoring.load_scorer(
        "org/model", engine="hf_classify", token=False, max_length=20
    )
    result = await scorer.score(
        [
            scoring.ScoringInput(text="first first first", text_pair="B"),
            scoring.ScoringInput(text="A", text_pair="B"),
            scoring.ScoringInput(text="second second", text_pair="B"),
        ],
        batch_size=2,
    )
    assert [call["text"] for call in fake_hf.tokenizer.calls] == [
        ["A", "second second"],
        ["first first first"],
    ]
    assert (
        result.rows[0].probabilities["contradiction"]
        > result.rows[1].probabilities["contradiction"]
    )
    assert (
        result.rows[2].probabilities["contradiction"]
        > result.rows[0].probabilities["contradiction"]
    )
    assert fake_hf.models[0].forward_calls == 2


async def test_exact_decoder_scores_use_full_vocabulary_and_last_real_position(
    fake_hf: types.SimpleNamespace,
) -> None:
    scorer = await scoring.load_scorer(
        "org/model",
        engine="hf_label_logits",
        token=False,
        max_length=12,
        label_tokens={"entailed": "A", "contradicted": "B"},
    )
    result = await scorer.score(
        [
            scoring.ScoringInput(text="first"),
            scoring.ScoringInput(text="first second second"),
        ]
    )
    for row in result.rows:
        assert row.probabilities == pytest.approx({"entailed": 0.4, "contradicted": 0.6})
        assert row.mass_outside == pytest.approx(0.5)
        assert row.pred == "contradicted"
        assert row.columns()["mass_outside"] == pytest.approx(0.5)
    assert fake_hf.models[0].forward_calls == 1


async def test_decoder_refuses_overlength_instead_of_truncating(
    fake_hf: types.SimpleNamespace,
) -> None:
    scorer = await scoring.load_scorer(
        "org/model",
        engine="hf_label_logits",
        token=False,
        max_length=2,
        label_tokens={"yes": "A", "no": "B"},
    )
    result = await scorer.score([scoring.ScoringInput(text="first second")])
    assert result.rows[0].probabilities == {}
    assert result.rows[0].mass_outside is None
    assert result.overflow_rows == result.failed_rows == 1
    assert result.truncated_rows == 0
    assert fake_hf.models[0].forward_calls == 0


@pytest.mark.parametrize("verbalizers", [{"yes": "A B"}, {"yes": ""}, {"yes": "A", "no": "A"}])
async def test_bad_verbalizers_are_refused_before_model_loading(
    fake_hf: types.SimpleNamespace,
    verbalizers: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        await scoring.load_scorer(
            "org/model",
            engine="hf_label_logits",
            token=False,
            max_length=12,
            label_tokens=verbalizers,
        )
    assert fake_hf.models == []


@pytest.mark.parametrize("token", [False, "hf_explicit_test_credential"])
@pytest.mark.parametrize("engine", ["hf_classify", "hf_label_logits"])
async def test_loader_boundary_is_safe_and_explicit(
    fake_hf: types.SimpleNamespace,
    token: scoring.HubCredential,
    engine: scoring.ScoringEngine,
) -> None:
    await scoring.load_scorer(
        "org/model",
        engine=engine,
        token=token,
        max_length=12,
        revision="pinned-revision",
        label_tokens={"yes": "A", "no": "B"} if engine == "hf_label_logits" else None,
    )
    for kind, model_id, kwargs in fake_hf.calls:
        assert model_id == "org/model"
        assert kwargs["token"] == token
        assert kwargs["revision"] == "pinned-revision"
        assert kwargs["trust_remote_code"] is False
        if kind != "tokenizer":
            assert kwargs["use_safetensors"] is True
            assert kwargs["output_loading_info"] is True


async def test_decoder_can_pad_with_eos(fake_hf: types.SimpleNamespace) -> None:
    fake_hf.tokenizer.pad_token_id = None
    scorer = await scoring.load_scorer(
        "org/model", engine="hf_label_logits", token=False, max_length=12, label_tokens={"yes": "A"}
    )
    assert fake_hf.tokenizer.pad_token_id == 98
    assert fake_hf.models[0].config.pad_token_id == 98
    assert scorer.loading_audit.status == "accepted"
    assert scorer.padding_policy.source == "eos_fallback"
    assert scorer.padding_policy.original_model_pad_token_id is None


async def test_classifier_model_config_follows_tokenizer_padding(
    fake_hf: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Model(fake_hf.labels, causal=False)
    model.config.pad_token_id = 77

    def load_model(model_id: str, **kwargs: object) -> tuple[Model, dict[str, object]]:
        return model, fake_hf.loading_info

    monkeypatch.setattr(
        sys.modules["transformers"].AutoModelForSequenceClassification,
        "from_pretrained",
        load_model,
    )
    await scoring.load_scorer("org/model", engine="hf_classify", token=False, max_length=12)
    assert model.config.pad_token_id == fake_hf.tokenizer.pad_token_id == 0


@pytest.mark.parametrize("reject_assignment", [False, True])
async def test_model_config_must_accept_and_preserve_the_actual_pad_id(
    fake_hf: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    reject_assignment: bool,
) -> None:
    class Config:
        @property
        def pad_token_id(self) -> None:
            return None

        @pad_token_id.setter
        def pad_token_id(self, value: int) -> None:
            if reject_assignment:
                raise AttributeError("read-only config")

    model = Model(fake_hf.labels, causal=False)
    monkeypatch.setattr(model, "config", Config())

    def load_model(model_id: str, **kwargs: object) -> tuple[Model, dict[str, object]]:
        return model, fake_hf.loading_info

    monkeypatch.setattr(
        sys.modules["transformers"].AutoModelForSequenceClassification,
        "from_pretrained",
        load_model,
    )
    with pytest.raises(ValidationError, match="model config"):
        await scoring.load_scorer("org/model", engine="hf_classify", token=False, max_length=12)
    assert not model.evaluating


@pytest.mark.parametrize("pad_id", [-1, True, "0"])
async def test_invalid_tokenizer_padding_is_refused_before_loading_weights(
    fake_hf: types.SimpleNamespace, pad_id: object
) -> None:
    fake_hf.tokenizer.pad_token_id = pad_id
    with pytest.raises(ValidationError, match="integer pad token"):
        await scoring.load_scorer("org/model", engine="hf_classify", token=False, max_length=12)
    assert fake_hf.models == []


@pytest.mark.parametrize("model_id", ["model", "org/model"])
@pytest.mark.parametrize("local_kind", ["directory", "file"])
async def test_hub_id_cannot_silently_load_a_matching_local_path(
    fake_hf: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_id: str,
    local_kind: str,
) -> None:
    path = tmp_path / model_id
    path.parent.mkdir(parents=True, exist_ok=True)
    if local_kind == "directory":
        path.mkdir()
    else:
        path.touch()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError, match="collides with a local path"):
        await scoring.load_scorer(
            model_id, engine="hf_classify", token=False, revision="a" * 40, max_length=12
        )
    assert fake_hf.calls == []


async def test_no_padding_token_is_refused(fake_hf: types.SimpleNamespace) -> None:
    fake_hf.tokenizer.pad_token_id = fake_hf.tokenizer.eos_token_id = None
    with pytest.raises(ValidationError, match="pad token"):
        await scoring.load_scorer("org/model", engine="hf_classify", token=False, max_length=12)


@pytest.mark.parametrize("cuda_available", [False, True])
async def test_auto_device_and_dtype_are_explicit_at_model_boundary(
    fake_hf: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
) -> None:
    monkeypatch.setattr(
        sys.modules["torch"],
        "cuda",
        types.SimpleNamespace(is_available=lambda: cuda_available),
        raising=False,
    )
    await scoring.load_scorer(
        "org/model",
        engine="hf_classify",
        token=False,
        max_length=12,
        device="auto",
        dtype="float32",
    )
    assert fake_hf.calls[-1][2]["dtype"] == "float32"
    assert fake_hf.models[0].device == ("cuda" if cuda_available else "cpu")


def test_exact_probability_arithmetic_is_stable_and_rejects_bad_logits() -> None:
    probs, outside = scoring._label_probabilities([1000, 1001, 999], [0, 1])  # pyright: ignore[reportPrivateUsage]
    assert math.fsum(probs) == pytest.approx(1)
    assert outside == pytest.approx(math.exp(-1) / (1 + math.e + math.exp(-1)))
    _, tiny_outside = scoring._label_probabilities([0, -40], [0])  # pyright: ignore[reportPrivateUsage]
    assert tiny_outside == pytest.approx(math.exp(-40), rel=1e-12, abs=0)
    with pytest.raises(ValidationError, match="finite"):
        scoring._label_probabilities([math.nan, 1], [0])  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValidationError, match="vocabulary"):
        scoring._label_probabilities([0, 1], [2])  # pyright: ignore[reportPrivateUsage]


async def test_missing_optional_dependencies_have_an_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(scoring, "import_module", missing)
    with pytest.raises(ImportError, match="strata-forge\\[finetuning\\]"):
        await scoring.load_scorer("org/model", engine="hf_classify", token=False, max_length=12)


async def test_import_does_not_load_heavy_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import importlib.util

    calls: list[str] = []

    original = importlib.import_module

    def guarded(name: str, package: str | None = None) -> types.ModuleType:
        if name in {"torch", "transformers"}:
            calls.append(name)
            raise AssertionError("heavy import at module load")
        return original(name, package)

    spec = importlib.util.spec_from_file_location("scoring_lazy_check", scoring.__file__)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "scoring_lazy_check", module)
    monkeypatch.setattr(importlib, "import_module", guarded)
    spec.loader.exec_module(module)
    assert calls == []


async def test_empty_work_and_wrong_input_shapes(fake_hf: types.SimpleNamespace) -> None:
    scorer = await scoring.load_scorer(
        "org/model", engine="hf_classify", token=False, max_length=12
    )
    result = await scorer.score([])
    assert result.rows == []
    assert result.failed_rows == result.overflow_rows == 0
    with pytest.raises(ValidationError, match="text_pair"):
        await scorer.score([scoring.ScoringInput(text="A")])
    with pytest.raises(ValidationError, match="batch_size"):
        await scorer.score([], batch_size=0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_id": "../model"}, "Hub repo"),
        ({"token": True}, "explicit credential"),
        ({"max_length": 0}, "positive"),
        ({"engine": "vllm_chat"}, "unsupported scoring engine"),
        ({"label_tokens": {"yes": "A"}}, "not label_tokens"),
        ({"engine": "hf_label_logits"}, "requires label_tokens"),
    ],
)
async def test_invalid_loader_options_fail_before_imports(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    message: str,
) -> None:
    def forbidden(name: str) -> object:
        raise AssertionError(name)

    monkeypatch.setattr(scoring, "import_module", forbidden)
    options: dict[str, object] = {
        "model_id": "org/model",
        "engine": "hf_classify",
        "token": False,
        "max_length": 12,
    }
    options.update(kwargs)
    with pytest.raises(ValidationError, match=message):
        await scoring.load_scorer(**options)  # pyright: ignore[reportArgumentType] — exercise invalid wire values


@pytest.mark.parametrize(
    ("logits", "message"),
    [
        (Tensor([0.0, 1.0]), "unexpected shape"),
        (Tensor([[0.0, 1.0]]), "match id2label"),
        (Tensor([[0.0, math.inf, 1.0]]), "finite"),
    ],
)
async def test_invalid_model_outputs_are_not_persisted_as_scores(
    fake_hf: types.SimpleNamespace,
    logits: Tensor,
    message: str,
) -> None:
    scorer = await scoring.load_scorer(
        "org/model", engine="hf_classify", token=False, max_length=12
    )
    fake_hf.models[0].override_logits = logits
    with pytest.raises(ValidationError, match=message):
        await scorer.score([scoring.ScoringInput(text="A", text_pair="B")])


async def test_decoder_rejects_pairs_and_shortened_logit_sequences(
    fake_hf: types.SimpleNamespace,
) -> None:
    scorer = await scoring.load_scorer(
        "org/model", engine="hf_label_logits", token=False, max_length=12, label_tokens={"yes": "A"}
    )
    with pytest.raises(ValidationError, match="single rendered prompt"):
        await scorer.score([scoring.ScoringInput(text="first", text_pair="B")])
    fake_hf.models[0].override_logits = Tensor([[[0.0, 1.0, 2.0, 3.0]]])
    with pytest.raises(ValidationError, match="label position"):
        await scorer.score([scoring.ScoringInput(text="first second")])


async def test_empty_decoder_without_special_tokens_has_no_label_position(
    fake_hf: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fake_hf.tokenizer, "num_special_tokens_to_add", lambda pair=False: 0)
    scorer = await scoring.load_scorer(
        "org/model", engine="hf_label_logits", token=False, max_length=12, label_tokens={"yes": "A"}
    )
    result = await scorer.score([scoring.ScoringInput(text="")])
    assert result.rows[0].error == "empty_prompt: decoder scoring needs a label position"
    assert result.failed_rows == 1
    assert fake_hf.models[0].forward_calls == 0


@pytest.mark.parametrize("data", [1.0, [[1.0]]])
def test_model_vector_must_be_flat(data: _Data) -> None:
    with pytest.raises(ValidationError, match="unexpected shape"):
        scoring._vector(Tensor(data))  # pyright: ignore[reportPrivateUsage]
