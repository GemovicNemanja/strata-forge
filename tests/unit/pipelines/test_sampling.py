"""``n`` samples per row, their seeds and keys, and chat-template kwargs on a chat run."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest

from strata_forge.compute.batch import BatchInferenceResult
from strata_forge.core.redact import Redactor
from strata_forge.llm import LLMResponse, Usage
from strata_forge.llm.routing import ModelRoute
from strata_forge.pipelines import SPEC_VERSION
from strata_forge.pipelines import _chat_template as ct
from strata_forge.pipelines import inference_runner as ir
from strata_forge.pipelines._common import RunError

if TYPE_CHECKING:
    from pathlib import Path

    from strata_forge.llm import LLMClient

_THINKING_TEMPLATE = (
    "{%- for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{%- endfor %}"
    "{%- if add_generation_prompt %}<|assistant|>"
    "{%- if enable_thinking is defined and not enable_thinking %}<think></think>{%- endif %}"
    "{%- endif %}"
)


def _hp(**fields: Any) -> ir.Hyperparams:
    return ir.Hyperparams.model_validate(fields)


def _spec(**hyperparams: Any) -> ir.RunSpec:
    return ir.RunSpec.model_validate(
        {
            "engine_version": SPEC_VERSION,
            "model_id": "org/model",
            "dataset_id": "org/ds",
            "split": "train",
            "column_mapping": {"q": "question"},
            "template": "Answer: {q}",
            "hyperparams": hyperparams,
        }
    )


# ------------------------------ the spec ------------------------------


def test_n_is_bounded() -> None:
    assert _hp(n=16, temperature=1.0).n == 16
    for bad in (0, 17):
        with pytest.raises(ValueError, match="n"):
            _hp(n=bad)
    with pytest.raises(ValueError, match="valid integer"):
        _hp(n=True)


@pytest.mark.parametrize("temperature", [0, 1e-6, 9.9e-6])
def test_n_samples_below_the_greedy_threshold_are_refused(temperature: float) -> None:
    # vLLM decodes greedily below 1e-5 (`_SAMPLING_EPS`), so these would be one reply n times.
    with pytest.raises(ValueError, match="greedy decoding"):
        _hp(n=4, temperature=temperature)
    assert _hp(n=1, temperature=temperature).n == 1
    assert _hp(n=4, temperature=1e-5).n == 4


def test_every_sample_seed_fits_int64() -> None:
    assert _hp(n=4, seed=2**63 - 4).n == 4
    with pytest.raises(ValueError, match="seed \\+ n - 1"):
        _hp(n=4, seed=2**63 - 3)


def test_n_is_refused_on_a_scoring_engine() -> None:
    with pytest.raises(ValueError, match="generates no text"):
        _hp(engine="hf_classify", pair_template="{h}", n=2)


def test_static_template_kwargs_ride_a_chat_run() -> None:
    assert _hp(chat_template_kwargs={"enable_thinking": False}).chat_template_kwargs == {
        "enable_thinking": False
    }


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"documents": [{"text": "{q}"}]}, "fixed for the run"),
        ({"enable_thinking": "false"}, "true or false"),
        ({"chat_template": "x"}, "cannot set"),
        ({}, "is empty"),
    ],
)
def test_unusable_chat_kwargs_are_refused(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _hp(chat_template_kwargs=kwargs)


def test_enable_thinking_must_be_a_boolean_on_every_engine() -> None:
    assert ct.kwargs_problem({"enable_thinking": 0}) == (
        "chat_template_kwargs.enable_thinking must be true or false"
    )


# ------------------------------ what each request carries ------------------------------


def test_sample_k_is_seeded_seed_plus_k() -> None:
    hp = _hp(n=3, seed=10, temperature=0.7, chat_template_kwargs={"enable_thinking": False})
    for sample in range(3):
        extras = ir._sampling_extras(hp, sample)  # pyright: ignore[reportPrivateUsage]
        assert extras == {
            "openai_compat": {
                "seed": 10 + sample,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
        }
    assert ir._sampling_extras(_hp(n=2, temperature=1.0), 1) is None  # pyright: ignore[reportPrivateUsage]


def test_template_kwargs_are_sent_in_canonical_order() -> None:
    hp = _hp(chat_template_kwargs={"z": 1, "a": {"y": 2, "b": 3}})
    body = cast("dict[str, Any]", ir._sampling_extras(hp)["openai_compat"]["extra_body"])  # pyright: ignore[reportOptionalSubscript, reportPrivateUsage]
    assert json.dumps(body["chat_template_kwargs"]) == '{"a": {"b": 3, "y": 2}, "z": 1}'


# ------------------------------ the batch ------------------------------


def _response(text: str) -> BatchInferenceResult:
    return BatchInferenceResult(
        response=LLMResponse(
            text=text,
            finish_reason="stop",
            usage=Usage(input_tokens=4, output_tokens=2),
            cost_usd=0.0,
            route=ModelRoute(model="m", provider="openai_compat", provider_model_id="m"),
            latency_ms=10.0,
        )
    )


class _RecordingRunner:
    """Answers ``<prompt>#<seed>`` so every row says which request produced it."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, client: Any, *, concurrency: int, on_error: str) -> None:
        del client, concurrency, on_error

    async def run(self, prompts: Any, **kwargs: Any) -> tuple[BatchInferenceResult, ...]:
        conversations = [list(p) for p in prompts]
        _RecordingRunner.calls.append({"n": len(conversations), **kwargs})
        extras = kwargs.get("provider_extras") or {}
        seed = extras.get("openai_compat", {}).get("seed")
        return tuple(_response(f"{c[0].content}#{seed}") for c in conversations)


async def _batch(spec: ir.RunSpec, questions: list[str]) -> list[dict[str, Any]]:
    prompts, ids = ir._build_requests(spec, [{"question": q} for q in questions])  # pyright: ignore[reportPrivateUsage]
    return await ir._run_batches(  # pyright: ignore[reportPrivateUsage]
        spec,
        client=cast("LLMClient", object()),
        prompts=prompts,
        custom_ids=ids,
        writer=None,
        redactor=Redactor(),
    )


async def test_n_samples_are_keyed_by_row_and_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingRunner.calls = []
    monkeypatch.setattr(ir, "BatchInferenceRunner", _RecordingRunner)
    out = await _batch(_spec(n=3, seed=10, temperature=0.8), ["a", "b"])

    assert [(r["custom_id"], r["sample_index"], r["seed"]) for r in out] == [
        ("row-0", 0, 10),
        ("row-0", 1, 11),
        ("row-0", 2, 12),
        ("row-1", 0, 10),
        ("row-1", 1, 11),
        ("row-1", 2, 12),
    ]
    # The row's output came from the request carrying that sample's seed.
    assert [r["output"] for r in out[:3]] == ["Answer: a#10", "Answer: a#11", "Answer: a#12"]
    # One batch per sample, each over every row of the chunk.
    assert [c["n"] for c in _RecordingRunner.calls] == [2, 2, 2]


async def test_a_chunk_holds_progress_chunk_requests_not_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With n samples per row a chunk holds 1/n as many rows, so the liveness check and the
    # progress event between chunks stay `progress_chunk` requests apart.
    _RecordingRunner.calls = []
    monkeypatch.setattr(ir, "BatchInferenceRunner", _RecordingRunner)
    out = await _batch(_spec(n=4, seed=1, temperature=1.0, progress_chunk=8), list("abcde"))
    assert len(out) == 20
    # 5 rows at 2 rows per chunk: chunks of 2, 2, 1 rows, each sent once per sample.
    assert [c["n"] for c in _RecordingRunner.calls] == [2] * 4 + [2] * 4 + [1] * 4


async def test_an_unseeded_sample_records_no_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ir, "BatchInferenceRunner", _RecordingRunner)
    out = await _batch(_spec(n=2, temperature=1.0), ["a"])
    assert [(r["sample_index"], r["seed"]) for r in out] == [(0, None), (1, None)]


async def test_a_run_that_declares_no_n_keeps_its_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ir, "BatchInferenceRunner", _RecordingRunner)
    out = await _batch(_spec(seed=5), ["a"])
    assert "sample_index" not in out[0]
    assert out[0]["output"] == "Answer: a#5"


async def test_a_declared_single_sample_is_keyed_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ir, "BatchInferenceRunner", _RecordingRunner)
    out = await _batch(_spec(n=1, seed=5), ["a"])
    assert (out[0]["sample_index"], out[0]["seed"]) == (0, 5)


async def test_a_guided_sample_is_repaired_under_its_own_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _InvalidThenValid(_RecordingRunner):
        async def run(self, prompts: Any, **kwargs: Any) -> tuple[BatchInferenceResult, ...]:
            conversations = [list(p) for p in prompts]
            seed = (kwargs.get("provider_extras") or {})["openai_compat"]["seed"]
            _RecordingRunner.calls.append({"seed": seed, "repair": len(conversations[0]) > 1})
            reply = '{"ok": true}' if len(conversations[0]) > 1 or seed == 20 else "nope"
            return tuple(_response(reply) for _ in conversations)

    _RecordingRunner.calls = []
    monkeypatch.setattr(ir, "BatchInferenceRunner", _InvalidThenValid)
    spec = _spec(n=2, seed=20, temperature=0.5, guided_json={"type": "object"})
    out = await _batch(spec, ["a"])
    assert [(r["sample_index"], r["json_retried"], r["json_valid"]) for r in out] == [
        (0, False, True),
        (1, True, True),
    ]
    assert _RecordingRunner.calls == [
        {"seed": 20, "repair": False},
        {"seed": 21, "repair": False},
        {"seed": 21, "repair": True},
    ]


def test_the_manifest_records_sampling_and_template_kwargs() -> None:
    spec = _spec(n=2, seed=3, temperature=0.9, chat_template_kwargs={"enable_thinking": False})
    row = {"output": "x", "reasoning": None, "error": None, "finish_reason": "stop"}
    rows = [
        row | {"custom_id": cid, "input_tokens": 1, "output_tokens": 1}
        for cid in ("row-0", "row-0", "row-1", "row-1")
    ]
    manifest = ir._build_manifest(spec, rows, None, chat_template_sha256="abc")  # pyright: ignore[reportPrivateUsage]
    assert manifest["sampling"] == {"n": 2, "seed": 3, "sample_seed": "seed + sample_index"}
    assert manifest["chat_template"] == {
        "sha256": "abc",
        "kwargs_sha256": ct.kwargs_digest({"enable_thinking": False}),
    }
    assert (manifest["counts"]["rows"], manifest["counts"]["prompts"]) == (4, 2)


def test_a_plain_manifest_records_neither() -> None:
    row = {"output": "x", "reasoning": None, "error": None, "finish_reason": "stop"}
    rows = [row | {"custom_id": "row-0", "input_tokens": 1, "output_tokens": 1}]
    manifest = ir._build_manifest(_spec(), rows, None)  # pyright: ignore[reportPrivateUsage]
    assert (manifest["sampling"], manifest["chat_template"]) == (None, None)
    assert "prompts" not in manifest["counts"]


# ------------------------------ the snapshot's template ------------------------------


def test_the_processor_template_wins_as_it_does_in_vllm(tmp_path: Path) -> None:
    (tmp_path / "chat_template.jinja").write_text("tokenizer's")
    (tmp_path / "chat_template.json").write_text(json.dumps({"chat_template": "processor's"}))
    assert ir._snapshot_chat_template(tmp_path) == "processor's"  # pyright: ignore[reportPrivateUsage]
    (tmp_path / "chat_template.json").write_text("not json")
    assert ir._snapshot_chat_template(tmp_path) == "tokenizer's"  # pyright: ignore[reportPrivateUsage]


def test_the_template_file_wins_over_the_tokenizer_config(tmp_path: Path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "old"}))
    assert ir._snapshot_chat_template(tmp_path) == "old"  # pyright: ignore[reportPrivateUsage]
    (tmp_path / "chat_template.jinja").write_text(_THINKING_TEMPLATE)
    assert ir._snapshot_chat_template(tmp_path) == _THINKING_TEMPLATE  # pyright: ignore[reportPrivateUsage]


def test_a_named_template_list_uses_its_default(tmp_path: Path) -> None:
    templates = [{"name": "tool_use", "template": "t"}, {"name": "default", "template": "d"}]
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": templates}))
    assert ir._snapshot_chat_template(tmp_path) == "d"  # pyright: ignore[reportPrivateUsage]
    assert ir._snapshot_chat_template(tmp_path / "missing") is None  # pyright: ignore[reportPrivateUsage]


def test_read_kwargs_pass_and_return_the_template_digest(tmp_path: Path) -> None:
    (tmp_path / "chat_template.jinja").write_text(_THINKING_TEMPLATE)
    hp = _hp(chat_template_kwargs={"enable_thinking": False})
    digest = ir._check_chat_template_kwargs(tmp_path, hp)  # pyright: ignore[reportPrivateUsage]
    assert digest == ct.template_digest(_THINKING_TEMPLATE)
    assert ir._check_chat_template_kwargs(tmp_path, _hp()) is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("files", "match"),
    [
        ({"chat_template.jinja": "{{ messages }}"}, "never reads \\['enable_thinking'\\]"),
        ({}, "ships no chat template"),
        (
            {"chat_template.jinja": _THINKING_TEMPLATE, "config.json": '{"model_type": "gpt_oss"}'},
            "harmony format",
        ),
    ],
)
def test_kwargs_the_server_would_drop_are_refused(
    tmp_path: Path, files: dict[str, str], match: str
) -> None:
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    hp = _hp(chat_template_kwargs={"enable_thinking": False})
    with pytest.raises(RunError, match=match):
        ir._check_chat_template_kwargs(tmp_path, hp)  # pyright: ignore[reportPrivateUsage]


async def test_seed_and_kwargs_reach_the_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real client and batch runner send each sample's seed and the kwargs to the server."""
    from pydantic import SecretStr

    from strata_forge.compute.batch import BatchInferenceRunner
    from strata_forge.llm import LLMClient as RealClient
    from strata_forge.llm.providers.config import OpenAICompatConfig
    from strata_forge.llm.providers.openai_compat import (
        UNAUTHENTICATED_API_KEY,
        OpenAICompatProvider,
    )

    captured: list[dict[str, Any]] = []

    async def _fake_acompletion(**kwargs: Any) -> Any:
        captured.append(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr("litellm.acompletion", _fake_acompletion)
    monkeypatch.setattr(ir, "BatchInferenceRunner", BatchInferenceRunner)
    provider = OpenAICompatProvider(
        OpenAICompatConfig(
            base_url="http://127.0.0.1:8000/v1", api_key=SecretStr(UNAUTHENTICATED_API_KEY)
        )
    )
    client = RealClient(
        model="org/model", provider="openai_compat", provider_clients={"openai_compat": provider}
    )
    spec = _spec(n=2, seed=7, temperature=0.6, chat_template_kwargs={"enable_thinking": False})
    prompts, ids = ir._build_requests(spec, [{"question": "a"}])  # pyright: ignore[reportPrivateUsage]
    await ir._run_batches(spec, client, prompts, ids, None, redactor=Redactor())  # pyright: ignore[reportPrivateUsage]
    first = [c for c in captured if c.get("seed") == 7]
    second = [c for c in captured if c.get("seed") == 8]
    assert first
    assert second
    assert first[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "n" not in first[0] or first[0]["n"] in (None, 1)
