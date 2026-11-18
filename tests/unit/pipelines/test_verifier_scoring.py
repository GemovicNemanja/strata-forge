"""Scoring a verifier trained by ``seq_cls`` (``hf_classify`` with ``artifact: strata_verifier``).

Tiny synthetic checkpoints, trained for a few steps on CPU, then scored through the scoring path:
the rendering comes from the artifact alone, the trainer and the scorer share one render
function, the auxiliary heads are reported, and calibration is applied only under ``auto``.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import pytest

from strata_forge.pipelines._common import RunError
from strata_forge.pipelines._scoring import (
    ScoringInputs,
    artifact_template_error,
    load_verifier,
    read_verifier,
    verifier_result_rows,
)
from strata_forge.pipelines.inference_runner import Hyperparams, RunSpec

torch: Any = pytest.importorskip("torch")
datasets: Any = pytest.importorskip("datasets")
pytest.importorskip("safetensors.torch")

from strata_forge.training.seq_cls import SeqClsConfig, SeqClsRunner  # noqa: E402
from strata_forge.training.verifier_render import encode_rows  # noqa: E402
from tests.unit.training import tiny_classifier as tiny  # noqa: E402 - after the import skips

if TYPE_CHECKING:
    from pathlib import Path


class _Probe:
    def reset(self) -> None:
        return None

    def peak_bytes(self) -> int | None:
        return 10

    def total_bytes(self) -> int | None:
        return 1000


def _train(tmp_path: Path, checkpoint: Path, **overrides: Any) -> tuple[Any, Path]:
    config = SeqClsConfig.model_validate(
        {
            "model_id": "synthetic/tiny",
            "output_dir": str(tmp_path / "checkpoints"),
            "labels": tiny.LABELS,
            **tiny.TWO_WAY,
            "max_length": 64,
            "per_device_batch_size": 8,
            "max_steps": 2,
            **overrides,
        }
    )
    artifact = tmp_path / "artifact"
    result = SeqClsRunner(config, memory_probe=_Probe()).run(
        checkpoint=checkpoint,
        artifact_dir=artifact,
        train_dataset=datasets.Dataset.from_list(tiny.rows(16)),
    )
    return result, artifact


@pytest.fixture(scope="module")
def encoder_artifact(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Path]:
    root = tmp_path_factory.mktemp("enc")
    return _train(root, tiny.encoder(root / "base"))


@pytest.fixture(scope="module")
def decoder_artifact(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Path]:
    root = tmp_path_factory.mktemp("dec")
    return _train(
        root,
        tiny.decoder(root / "base"),
        head_init="decoder_lm_head",
        verbalizers=dict(tiny.VERBALIZERS),
    )


def _inputs(n: int) -> ScoringInputs:
    return ScoringInputs(texts=[], pairs=[], item_ids=None, slices=None)


def _score(artifact: Path, rows: list[dict[str, Any]], **kwargs: Any) -> tuple[Any, list[Any]]:
    scorer = load_verifier(
        artifact,
        label_alias=kwargs.pop("label_alias", None),
        max_length=None,
        dtype="float32",
        apply_calibration=kwargs.pop("apply_calibration", False),
    )
    rendered = scorer.render([r["premises"] for r in rows], [r["hypothesis"] for r in rows])
    lengths = scorer.measure(rendered)
    scores = dict(enumerate(scorer.score(rendered)))
    return scorer, verifier_result_rows(lengths, scores, scorer, _inputs(len(rows)))


@pytest.mark.parametrize("which", ["encoder_artifact", "decoder_artifact"])
def test_training_itself_checked_the_scoring_path(
    which: str, request: pytest.FixtureRequest
) -> None:
    result, _ = request.getfixturevalue(which)
    parity = result.receipts["scoring_parity"]
    assert parity["passed"] is True
    assert parity["rows"] == 8
    assert parity["argmax_agree"] == 8
    assert parity["max_abs_logit_diff"] <= 1e-4


@pytest.mark.parametrize("which", ["encoder_artifact", "decoder_artifact"])
def test_the_scorer_renders_exactly_as_the_trainer_encoded(
    which: str, request: pytest.FixtureRequest
) -> None:
    _, artifact = request.getfixturevalue(which)
    rows = tiny.rows(6, seed=3)
    scorer = load_verifier(
        artifact, label_alias=None, max_length=None, dtype="float32", apply_calibration=False
    )
    rendered = scorer.render([r["premises"] for r in rows], [r["hypothesis"] for r in rows])
    runner = SeqClsRunner(
        SeqClsConfig.model_validate(
            {
                "model_id": "m",
                "output_dir": "/nonexistent",
                "labels": tiny.LABELS,
                **tiny.TWO_WAY,
                "max_length": 64,
                "head_init": "existing",
            }
        )
    )
    tokenizer = runner.load_tokenizer(artifact)
    trained, _ = runner.encode(datasets.Dataset.from_list(rows), tokenizer, split="t")
    scored = encode_rows(scorer._tokenizer, rendered)["input_ids"]  # pyright: ignore[reportPrivateUsage]
    assert [list(x) for x in scored] == [list(x) for x in trained["input_ids"]]
    assert scorer.rendering.decoder is (which == "decoder_artifact")


def test_aux_heads_are_reported_per_row_and_per_premise(
    encoder_artifact: tuple[Any, Path],
) -> None:
    _, artifact = encoder_artifact
    rows = tiny.rows(5, seed=4)
    scorer, out = _score(artifact, rows)
    assert scorer.heads == ["flag", "support"]
    for row, source in zip(out, rows, strict=True):
        assert row["overflow"] is False
        assert set(tiny.LABELS) == {
            k.removeprefix("logit_")
            for k in row
            if k.startswith("logit_") and k != "logit_inconsistent"
        }
        assert math.isclose(sum(row[f"p_{label}"] for label in tiny.LABELS), 1.0)
        assert 0.0 < row["p_inconsistent"] < 1.0
        assert [s["pid"] for s in row["support_scores"]] == [p["pid"] for p in source["premises"]]
        assert all(0.0 < s["p"] < 1.0 for s in row["support_scores"])


def test_calibration_changes_only_p_and_only_under_auto(
    encoder_artifact: tuple[Any, Path], tmp_path: Path
) -> None:
    _, trained = encoder_artifact
    import shutil

    artifact = tmp_path / "calibrated"
    shutil.copytree(trained, artifact)
    (artifact / "calibration.json").write_text(
        json.dumps(
            {
                "schema": "strata.calibration/v1",
                "labels": list(tiny.LABELS),
                "fit": {"temperature": 2.0},
            }
        )
    )
    rows = tiny.rows(3, seed=5)
    off_scorer, off = _score(artifact, rows)
    auto_scorer, auto = _score(artifact, rows, apply_calibration=True)
    assert off_scorer.temperature is None
    assert auto_scorer.temperature == 2.0
    for raw, cal in zip(off, auto, strict=True):
        logits = [raw[f"logit_{label}"] for label in tiny.LABELS]
        assert logits == [cal[f"logit_{label}"] for label in tiny.LABELS]  # raw logits kept
        expected = [math.exp(v / 2.0) for v in logits]
        total = sum(expected)
        for label, value in zip(tiny.LABELS, expected, strict=True):
            assert math.isclose(cal[f"p_{label}"], value / total, rel_tol=1e-9)


def test_a_checkpoint_without_a_verifier_record_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="not a verifier trained by seq_cls"):
        read_verifier(tmp_path)


def test_a_malformed_premises_cell_is_named_by_row_not_content(
    encoder_artifact: tuple[Any, Path],
) -> None:
    _, artifact = encoder_artifact
    scorer = load_verifier(
        artifact, label_alias=None, max_length=None, dtype="float32", apply_calibration=False
    )
    with pytest.raises(RunError, match=r"row 1: premises must be") as caught:
        scorer.render([[{"pid": "p1", "text": "w1"}], "secret text"], ["w2", "w3"])
    assert "secret" not in str(caught.value)


def _hp(**fields: Any) -> dict[str, Any]:
    return {
        "engine": "hf_classify",
        "artifact": "strata_verifier",
        "pair_template": "{hypothesis}",
        "dtype": "float32",
        **fields,
    }


def test_artifact_mode_needs_no_alias_and_fixes_the_templates() -> None:
    hp = Hyperparams.model_validate(_hp())
    assert hp.label_alias is None
    assert hp.apply_calibration == "off"
    with pytest.raises(ValueError, match="pair_template"):
        Hyperparams.model_validate(_hp(pair_template="Claim: {hypothesis}"))
    with pytest.raises(ValueError, match="list_separator"):
        Hyperparams.model_validate(_hp(list_separator=" "))
    assert artifact_template_error("{premises}", hp) is None
    assert "templates only name the roles" in (artifact_template_error("P: {premises}", hp) or "")
    with pytest.raises(ValueError, match="templates only name the roles"):
        RunSpec.model_validate(
            {
                "model_id": "org/v",
                "dataset_id": "org/d",
                "split": "test",
                "template": "P: {premises}",
                "column_mapping": {"premises": "premises", "hypothesis": "hypothesis"},
                "hyperparams": _hp(),
            }
        )


def test_apply_calibration_is_for_artifact_scoring_only() -> None:
    assert Hyperparams.model_validate(_hp(apply_calibration="auto")).apply_calibration == "auto"
    with pytest.raises(ValueError, match="artifact scoring only"):
        Hyperparams.model_validate(
            {
                "engine": "hf_classify",
                "pair_template": "{h}",
                "dtype": "float32",
                "label_alias": {"a": "x", "b": "y"},
                "apply_calibration": "auto",
            }
        )
    with pytest.raises(ValueError, match="apply only to engine"):
        Hyperparams.model_validate({"artifact": "strata_verifier"})


def test_a_verifier_record_whose_template_was_altered_is_refused(
    decoder_artifact: tuple[Any, Path], tmp_path: Path
) -> None:
    import shutil

    _, trained = decoder_artifact
    artifact = tmp_path / "altered"
    shutil.copytree(trained, artifact)
    path = artifact / "strata_verifier.json"
    record = json.loads(path.read_text())
    record["input"]["decoder_template"] = "{hypothesis}\n{premises}"
    path.write_text(json.dumps(record))
    with pytest.raises(RunError, match="recorded digest"):
        load_verifier(
            artifact, label_alias=None, max_length=None, dtype="float32", apply_calibration=False
        )
    record["input"]["decoder_template"] = "{premises} {premises}\n{hypothesis}"
    path.write_text(json.dumps(record))
    with pytest.raises(RunError, match="exactly once each"):
        load_verifier(
            artifact, label_alias=None, max_length=None, dtype="float32", apply_calibration=False
        )


def test_a_length_cap_above_the_trained_one_is_refused(encoder_artifact: tuple[Any, Path]) -> None:
    _, artifact = encoder_artifact
    with pytest.raises(RunError, match="exceeds the 64 tokens the verifier was trained on"):
        load_verifier(
            artifact, label_alias=None, max_length=65, dtype="float32", apply_calibration=False
        )
    scorer = load_verifier(
        artifact, label_alias=None, max_length=32, dtype="float32", apply_calibration=False
    )
    assert scorer.max_length == 32


def test_a_decoder_whose_pad_is_its_eos_is_refused_not_misread(
    decoder_artifact: tuple[Any, Path], tmp_path: Path
) -> None:
    import shutil

    _, trained = decoder_artifact
    artifact = tmp_path / "pad-is-eos"
    shutil.copytree(trained, artifact)
    config_path = artifact / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["pad_token"] = "[SEP]"
    config_path.write_text(json.dumps(config))
    scorer = load_verifier(
        artifact, label_alias=None, max_length=None, dtype="float32", apply_calibration=False
    )
    assert scorer._tokenizer.pad_token_id == scorer._tokenizer.sep_token_id  # pyright: ignore[reportPrivateUsage]
    rows = tiny.rows(3, seed=6)
    rendered = scorer.render([r["premises"] for r in rows], [r["hypothesis"] for r in rows])
    with pytest.raises(RunError, match="end on the padding token"):
        scorer.measure(rendered)


@pytest.mark.parametrize("which", ["encoder_artifact", "decoder_artifact"])
def test_every_head_reads_the_same_token_alone_or_padded(
    which: str, request: pytest.FixtureRequest
) -> None:
    _, artifact = request.getfixturevalue(which)
    scorer = load_verifier(
        artifact, label_alias=None, max_length=None, dtype="float32", apply_calibration=False
    )
    rows = tiny.rows(6, seed=7)
    rendered = scorer.render([r["premises"] for r in rows], [r["hypothesis"] for r in rows])
    together = scorer.score(rendered)
    for row, joint in zip(rendered, together, strict=True):
        (alone,) = scorer.score([row])
        assert max(abs(a - b) for a, b in zip(alone.logits, joint.logits, strict=True)) < 1e-5
        if alone.flag is not None and joint.flag is not None:
            assert abs(alone.flag - joint.flag) < 1e-5


@pytest.mark.parametrize("which", ["encoder_artifact", "decoder_artifact"])
def test_a_timed_verifier_request_renders_scores_and_syncs_once(
    which: str, request: pytest.FixtureRequest
) -> None:
    import time

    from strata_forge.pipelines._scoring import timed_verifier_request

    _, artifact = request.getfixturevalue(which)
    scorer = load_verifier(
        artifact, label_alias=None, max_length=None, dtype="float32", apply_calibration=False
    )
    syncs = 0
    real = scorer.synchronize

    def _counted() -> None:
        nonlocal syncs
        syncs += 1
        real()

    scorer.synchronize = _counted  # type: ignore[method-assign]
    rows = tiny.rows(3, seed=9)
    cells = [(r["premises"], r["hypothesis"]) for r in rows]
    start, end, forward = timed_verifier_request(scorer, cells, clock=time.perf_counter_ns)
    assert end > start
    assert 0 < forward * 1_000_000 <= end - start
    # The one synchronization ends the request; the forward pass adds none of its own.
    assert syncs == 1
    assert scorer.forward_timer().method == "perf_counter_ns_cpu"


def test_premise_prefix_reuse_scores_exactly_as_the_full_forward(
    decoder_artifact: tuple[Any, Path], encoder_artifact: tuple[Any, Path]
) -> None:
    import time

    from strata_forge.pipelines._scoring import (
        PrefixCache,
        prefix_reuse_error,
        score_with_prefix,
        timed_prefix_request,
    )

    _, artifact = decoder_artifact
    scorer = load_verifier(
        artifact, label_alias=None, max_length=None, dtype="float32", apply_calibration=False
    )
    assert prefix_reuse_error(scorer) is None
    base = tiny.rows(6, seed=11)
    # The same premise set under two hypotheses, then a new set: miss, hit, miss.
    cells = [
        (base[0]["premises"], base[0]["hypothesis"]),
        (base[0]["premises"], base[1]["hypothesis"]),
        (base[2]["premises"], base[2]["hypothesis"]),
    ]
    cache = PrefixCache(max_bytes=1 << 30)
    hits: list[bool] = []
    for premises, hypothesis in cells:
        value, hit = score_with_prefix(scorer, cache, (premises, hypothesis))
        hits.append(hit)
        (full,) = scorer.score(scorer.render([premises], [hypothesis]))
        assert value.logits == pytest.approx(full.logits, abs=1e-5)
        assert value.flag == pytest.approx(full.flag, abs=1e-5)
        assert [p for p, _ in value.support or []] == [p for p, _ in full.support or []]
        assert [v for _, v in value.support or []] == pytest.approx(
            [v for _, v in full.support or []], abs=1e-5
        )
    assert hits == [False, True, False]
    assert len(cache) == 2
    record = cache.record()
    assert record["peak_bytes"] > 0
    # The cache was rolled back to the prefix: the same row again is a hit with the same answer.
    again, hit = score_with_prefix(scorer, cache, cells[0])
    assert hit
    _, end, forward, _ = timed_prefix_request(scorer, cache, cells[1], clock=time.perf_counter_ns)
    assert end > 0
    assert forward > 0
    # A cache too small for one entry keeps none and still answers.
    tiny_cache = PrefixCache(max_bytes=1)
    assert score_with_prefix(scorer, tiny_cache, cells[0])[1] is False
    assert score_with_prefix(scorer, tiny_cache, cells[0])[1] is False
    assert tiny_cache.record()["uncacheable"] == 2
    del again
    _, enc = encoder_artifact
    encoder = load_verifier(
        enc, label_alias=None, max_length=None, dtype="float32", apply_calibration=False
    )
    assert "decoder verifier only" in (prefix_reuse_error(encoder) or "")


def test_the_prefix_cache_evicts_least_recently_used_first() -> None:
    from strata_forge.pipelines import _scoring as scoring

    cache = scoring.PrefixCache(max_bytes=100)
    make: Any = scoring._PrefixEntry  # pyright: ignore[reportPrivateUsage]

    def entry(n: int) -> Any:
        return make(cache=None, markers=None, length=1, nbytes=n)

    cache.put((1,), entry(40))
    cache.put((2,), entry(40))
    assert cache.get((1,)) is not None  # (2,) is now the least recently used
    cache.put((3,), entry(40))
    assert cache.get((2,)) is None
    assert cache.get((1,)) is not None
    assert cache.get((3,)) is not None
    assert cache.record() == {
        "max_bytes": 100,
        "peak_bytes": 80,
        "bytes_at_end": 80,
        "entries_at_end": 2,
        "evictions": 1,
        "uncacheable": 0,
    }
