"""Native per-item drop evidence with offline forward fakes; no pretrained assets."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

import pytest

from strata_forge.core.errors import ValidationError
from strata_forge.pipelines import _scoring as scoring
from tests.unit.pipelines.test_scoring import fake_hf as _fake_hf

fake_hf = _fake_hf

if TYPE_CHECKING:
    import types


async def test_paired_drop_is_exact_and_never_uses_only_first(
    fake_hf: types.SimpleNamespace,
) -> None:
    scorer = await scoring.load_scorer(
        "example/fake", engine="hf_classify", token=False, max_length=6
    )
    rows = [
        scoring.ScoringInput(text="first " * 10, text_pair="second"),
        scoring.ScoringInput(text="first first", text_pair="second"),
        scoring.ScoringInput(text="first", text_pair="second"),
    ]
    result = await scorer.score_with_outcomes(rows, overflow="drop", batch_size=2)
    assert [row.outcome for row in result.rows] == ["overflow", "scored", "scored"]
    assert [row.token_count for row in result.rows] == [14, 6, 5]
    assert [row.index for row in result.rows] == [0, 1, 2]
    assert [row.model_limit for row in result.rows] == [6, 6, 6]
    assert result.overflow_rows == 1
    assert result.scored_rows == 2
    assert result.failed_rows == 0
    assert result.rows[0].result is None
    assert result.rows[0].model_sent is False
    assert fake_hf.models[0].forward_calls == 1
    for call in fake_hf.tokenizer.calls:
        assert call["truncation"] is False
        assert "max_length" not in call
        assert rows[0].text not in call["text"]
    assert all(not row.result.truncated for row in result.rows if row.result is not None)


async def test_all_dropped_does_not_tokenize_a_batch_or_forward(
    fake_hf: types.SimpleNamespace,
) -> None:
    scorer = await scoring.load_scorer(
        "example/fake", engine="hf_classify", token=False, max_length=3
    )
    result = await scorer.score_with_outcomes(
        [scoring.ScoringInput(text="first", text_pair="second")], overflow="drop"
    )
    assert result.overflow_rows == 1
    assert result.rows[0].token_count == 5
    assert fake_hf.tokenizer.calls == []
    assert fake_hf.models[0].forward_calls == 0


async def test_decoder_empty_refusal_is_invalid_not_overflow(
    fake_hf: types.SimpleNamespace,
) -> None:
    fake_hf.tokenizer.num_special_tokens_to_add = lambda pair=False: 0
    scorer = await scoring.load_scorer(
        "example/fake",
        engine="hf_label_logits",
        token=False,
        max_length=5,
        label_tokens={"E": "A", "C": "B", "U": "C"},
    )
    result = await scorer.score_with_outcomes([scoring.ScoringInput(text="")], overflow="drop")
    assert result.failed_rows == 1
    assert result.overflow_rows == 0
    item = result.rows[0]
    assert item.outcome == "invalid"
    assert item.token_count == 0
    assert item.model_sent is False
    assert item.result is not None
    assert item.result.pred is None
    assert fake_hf.models[0].forward_calls == 0


@pytest.mark.parametrize("overflow", [None, "error", "legacy", True, [], {}])
async def test_drop_api_has_no_implicit_policy(
    fake_hf: types.SimpleNamespace, overflow: object
) -> None:
    scorer = await scoring.load_scorer(
        "example/fake", engine="hf_classify", token=False, max_length=6
    )
    with pytest.raises(ValidationError):
        await scorer.score_with_outcomes([], overflow=overflow)  # pyright: ignore[reportArgumentType] — deliberate untyped boundary.
    assert fake_hf.models[0].forward_calls == 0


async def test_late_forward_cannot_emit_callbacks_after_cancel(
    fake_hf: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = await scoring.load_scorer(
        "example/fake", engine="hf_classify", token=False, max_length=6
    )
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = scorer._score  # pyright: ignore[reportPrivateUsage] — controlled owned native-thread schedule.
    events: list[object] = []

    def held(*args: object, **kwargs: object) -> object:
        entered.set()
        assert release.wait(5)
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(scorer, "_score", held)
    task = asyncio.create_task(
        scorer.score_with_outcomes(
            [scoring.ScoringInput(text="first", text_pair="second")],
            overflow="drop",
            on_progress=events.append,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        previous = len(events)
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        assert len(events) == previous
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 5)


async def test_empty_pair_without_special_tokens_is_invalid_without_forward(
    fake_hf: types.SimpleNamespace,
) -> None:
    fake_hf.tokenizer.num_special_tokens_to_add = lambda pair=False: 0
    scorer = await scoring.load_scorer(
        "example/fake", engine="hf_classify", token=False, max_length=4
    )
    result = await scorer.score_with_outcomes(
        [scoring.ScoringInput(text="", text_pair="")], overflow="drop"
    )
    assert result.rows[0].outcome == "invalid"
    assert result.failed_rows == 1
    assert result.overflow_rows == 0
    assert fake_hf.models[0].forward_calls == 0
