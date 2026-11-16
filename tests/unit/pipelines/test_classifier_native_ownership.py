"""Held native boundaries prove cleanup and later optimizer admission wait for physical return."""

from __future__ import annotations

import asyncio
import threading
from contextvars import ContextVar
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from strata_forge.core.errors import ValidationError
from strata_forge.pipelines import _classifier_checkpoint as checkpoint
from strata_forge.pipelines import classifier_runner as worker
from strata_forge.training import _classifier_heads as heads
from strata_forge.training import classifier
from strata_forge.training._classifier_native import owned_native
from tests.unit.pipelines.test_classifier_runner import Authority, Source, spec

if TYPE_CHECKING:
    from pathlib import Path

# Fixed native seams intentionally return synthetic objects, without importing any ML stack.


async def test_owned_native_preserves_the_callers_privacy_and_budget_context() -> None:
    private_context: ContextVar[str | None] = ContextVar("fictional-private-context", default=None)
    token = private_context.set("fictional-private-suppression")
    try:
        assert await owned_native(private_context.get) == "fictional-private-suppression"
    finally:
        private_context.reset(token)


async def entered_thread(entered: threading.Event) -> None:
    for _ in range(1000):
        if entered.is_set():
            return
        await asyncio.sleep(0.001)
    pytest.fail("owned native thread did not enter")


@pytest.mark.parametrize("site", ["base", "reload", "initialize", "load", "trainer"])
async def test_each_primary_native_site_drains_repeated_cancel_before_caller_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    private = tmp_path / "private-input"
    private.write_text("fictional canary")
    values: list[Any] = []

    def held(*args: Any, **_: Any) -> object:
        if site == "trainer":
            values.append(args[-1])
        entered.set()
        release.wait(3)
        assert private.is_file()
        returned.set()
        return object()

    async def operation() -> None:
        try:
            if site == "base":
                monkeypatch.setattr(checkpoint, "_base", held)
                await checkpoint.load_classifier_base(tmp_path)
            elif site == "reload":
                monkeypatch.setattr(checkpoint, "_saved", held)
                await checkpoint.reload_classifier(tmp_path)
            elif site == "initialize":
                monkeypatch.setattr(heads, "_initialize", held)
                await heads.initialize_classifier(
                    object(), labels=("E", "C", "U"), initializer_id="encoder_random_head"
                )
            elif site == "load":
                monkeypatch.setattr(classifier, "_load", held)
                await classifier.load_classifier(tmp_path)
            else:
                monkeypatch.setattr(classifier, "_train", held)
                await classifier.ClassifierRunner(spec().training.sdk_config()).train(
                    initialized=SimpleNamespace(), tokenizer=object(), rows=(), output_dir=tmp_path
                )
            pytest.fail("cancelled native result escaped")
        finally:
            assert returned.is_set()
            private.unlink()

    task = asyncio.create_task(operation())
    try:
        await entered_thread(entered)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert private.is_file()
        if site == "trainer":
            assert values[0].is_set()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not private.exists()


async def test_cancelled_base_load_cannot_admit_head_or_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = threading.Event(), threading.Event()
    later: list[str] = []

    def held(_: Any) -> Any:
        entered.set()
        release.wait(3)
        return SimpleNamespace(model=object(), tokenizer=object())

    async def forbidden(*_: Any, **__: Any) -> Any:
        later.append("head")
        raise AssertionError("head admission after cancelled load")

    monkeypatch.setattr(worker, "check_engine_version", lambda _: None)
    monkeypatch.setattr(checkpoint, "_base", held)
    monkeypatch.setattr(worker, "initialize_classifier", forbidden)
    task = asyncio.create_task(
        worker.run_classifier_training(
            spec(), authority=Authority(), source=Source(tmp_path), attempt_dir=tmp_path / "run"
        )
    )
    try:
        await entered_thread(entered)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not later
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not later
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("site", ["label", "row", "batch"])
async def test_decoder_tokenizer_is_off_loop_and_cancellation_cannot_admit_a_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    later: list[str] = []

    def held() -> None:
        entered.set()
        release.wait(3)
        returned.set()

    class Tokenizer:
        def encode(self, text: str, **_: Any) -> list[int]:
            if site == "label":
                held()
            return [2 + (text != "a")]

        def __call__(self, text: str | list[str], **_: Any) -> dict[str, Any]:
            if (site == "row" and isinstance(text, str)) or (
                site == "batch" and isinstance(text, list)
            ):
                held()
            return {"input_ids": [2], "attention_mask": [1]}

    async def base(_: Any) -> Any:
        return SimpleNamespace(model=object(), tokenizer=Tokenizer())

    async def forbidden(*_: Any, **__: Any) -> Any:
        later.append("head")
        raise AssertionError("head admission after cancelled tokenizer")

    monkeypatch.setattr(worker, "check_engine_version", lambda _: None)
    monkeypatch.setattr(worker, "load_classifier_base", base)
    monkeypatch.setattr(worker, "initialize_classifier", forbidden)
    task = asyncio.create_task(
        worker.run_classifier_training(
            spec("gpt2"),
            authority=Authority(),
            source=Source(tmp_path),
            attempt_dir=tmp_path / "run",
        )
    )
    try:
        await entered_thread(entered)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not returned.is_set()
        assert not later
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert returned.is_set()
    assert not later
    assert not (tmp_path / "run").exists()


async def test_cancelled_artifact_inventory_cannot_write_receipt_or_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = threading.Event(), threading.Event()

    async def base(_: Any) -> Any:
        return SimpleNamespace(model=object(), tokenizer=object())

    async def initialized(*_: Any, **__: Any) -> Any:
        return SimpleNamespace(receipt=SimpleNamespace())

    class Trainer:
        def __init__(self, _: Any) -> None:
            pass

        async def train(self, **_: Any) -> Any:
            artifact = tmp_path / "run/training/artifact"
            artifact.mkdir(parents=True)
            return SimpleNamespace(artifact_dir=artifact, steps=1, trained_rows=2, overflow_ids=())

    def inventory(_: Any) -> Any:
        entered.set()
        release.wait(3)
        return {}

    monkeypatch.setattr(worker, "check_engine_version", lambda _: None)
    monkeypatch.setattr(worker, "load_classifier_base", base)
    monkeypatch.setattr(worker, "initialize_classifier", initialized)
    monkeypatch.setattr(worker, "ClassifierRunner", Trainer)
    monkeypatch.setattr(worker, "reload_classifier", base)
    monkeypatch.setattr(worker, "_inventory", inventory)
    task = asyncio.create_task(
        worker.run_classifier_training(
            spec(), authority=Authority(), source=Source(tmp_path), attempt_dir=tmp_path / "run"
        )
    )
    try:
        await entered_thread(entered)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not (tmp_path / "run/training-receipt.json").exists()


@pytest.mark.parametrize("control", [SystemExit, KeyboardInterrupt, asyncio.CancelledError])
async def test_native_controls_cannot_escape_through_the_event_loop(
    control: type[BaseException],
) -> None:
    def refuse() -> None:
        raise control

    with pytest.raises(ValidationError, match=r"^classifier_native_failed$"):
        await owned_native(refuse)


def test_cooperatively_cancelled_trainer_refuses_before_native_import_or_optimizer(
    tmp_path: Path,
) -> None:
    cancelled = threading.Event()
    cancelled.set()
    dynamic: Any = None
    with pytest.raises(asyncio.CancelledError):
        classifier._train(dynamic, dynamic, dynamic, (), tmp_path, cancelled)
