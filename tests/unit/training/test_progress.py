"""Unit tests for `strata_forge.training.progress` (live training-progress events)."""

from __future__ import annotations

import json
import sys
import types
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from strata_forge.training.progress import (
    JsonlProgressWriter,
    ProgressEvent,
    attach,
    trainer_callback,
)
from strata_forge.training.sft import SFTConfig, SFTRunner

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# ProgressEvent
# ---------------------------------------------------------------------------


class TestProgressEvent:
    def test_defaults(self) -> None:
        ev = ProgressEvent(kind="step")
        assert ev.kind == "step"
        assert ev.step is None
        assert ev.metrics == {}
        assert ev.message == ""
        assert ev.ts.tzinfo is not None  # timezone-aware (UTC)

    def test_json_roundtrip(self) -> None:
        ev = ProgressEvent(kind="step", step=10, loss=0.5, metrics={"grad_norm": 2.0})
        decoded = json.loads(ev.model_dump_json())
        assert decoded["kind"] == "step"
        assert decoded["step"] == 10
        assert decoded["loss"] == 0.5
        assert decoded["metrics"] == {"grad_norm": 2.0}
        assert isinstance(decoded["ts"], str)

    def test_phase_is_a_valid_kind(self) -> None:
        # `phase` reports uncountable work, so it carries a message and no step.
        ev = ProgressEvent(kind="phase", message="Loading the model onto the GPU (90s)")
        decoded = json.loads(ev.model_dump_json())
        assert decoded["kind"] == "phase"
        assert decoded["message"] == "Loading the model onto the GPU (90s)"
        assert decoded["step"] is None
        assert decoded["total_steps"] is None

    def test_unknown_kind_still_rejected(self) -> None:
        # The Literal is the contract: widening it for `phase` must not open it to anything.
        with pytest.raises(ValidationError):
            ProgressEvent(kind="provisioning")  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# JsonlProgressWriter
# ---------------------------------------------------------------------------


class TestJsonlProgressWriter:
    def test_emit_writes_one_line_per_event(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "progress.jsonl"  # parent does not exist yet
        writer = JsonlProgressWriter(path)
        writer.emit(ProgressEvent(kind="start", total_steps=100))
        writer.emit(ProgressEvent(kind="step", step=1, loss=0.9))
        writer.close()

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert first["kind"] == "start"
        assert first["total_steps"] == 100
        assert second["kind"] == "step"
        assert second["step"] == 1
        assert second["loss"] == 0.9

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        with JsonlProgressWriter(path) as writer:
            writer.emit(ProgressEvent(kind="end"))
        assert path.read_text(encoding="utf-8").strip() != ""
        # idempotent close
        writer.close()


# ---------------------------------------------------------------------------
# trainer_callback — fake transformers.TrainerCallback base
# ---------------------------------------------------------------------------


class _FakeState:
    def __init__(self, global_step: int = 0, max_steps: int = 0, epoch: float = 0.0) -> None:
        self.global_step = global_step
        self.max_steps = max_steps
        self.epoch = epoch


@pytest.fixture
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Inject a minimal `transformers` module exposing a TrainerCallback base."""
    mod = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    mod.TrainerCallback = TrainerCallback  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return mod


class TestTrainerCallback:
    def test_maps_hooks_to_jsonl(self, fake_transformers: Any, tmp_path: Path) -> None:
        del fake_transformers
        path = tmp_path / "p.jsonl"
        writer = JsonlProgressWriter(path)
        cb = trainer_callback(writer)
        state = _FakeState(global_step=5, max_steps=100, epoch=0.1)

        cb.on_train_begin(None, state, None)
        cb.on_log(
            None,
            state,
            None,
            logs={"loss": 0.5, "learning_rate": 1e-4, "epoch": 0.1, "grad_norm": 2.0},
        )
        cb.on_log(None, state, None, logs={"eval_loss": 0.4, "eval_runtime": 1.2})
        cb.on_save(None, state, None)
        cb.on_train_end(None, state, None)
        writer.close()

        events = [json.loads(line) for line in path.read_text().splitlines()]
        kinds = [e["kind"] for e in events]
        assert kinds == ["start", "step", "eval", "checkpoint", "end"]

        step = events[1]
        assert step["loss"] == 0.5
        assert step["learning_rate"] == 1e-4
        assert step["epoch"] == 0.1
        # promoted keys are NOT duplicated inside metrics; extras are kept
        assert step["metrics"] == {"grad_norm": 2.0}

        ev = events[2]
        assert ev["kind"] == "eval"
        assert ev["loss"] == 0.4  # promoted from eval_loss
        assert ev["metrics"] == {"eval_runtime": 1.2}

    def test_requires_transformers(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setitem(sys.modules, "transformers", None)
        writer = JsonlProgressWriter(tmp_path / "p.jsonl")
        with pytest.raises(ImportError, match="transformers"):
            trainer_callback(writer)


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------


class _FakeTrainer:
    def __init__(self) -> None:
        self.callbacks: list[Any] = []

    def add_callback(self, cb: Any) -> None:
        self.callbacks.append(cb)


class TestAttach:
    def test_returns_none_without_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FORGE_PROGRESS_PATH", raising=False)
        trainer = _FakeTrainer()
        assert attach(trainer, None) is None
        assert trainer.callbacks == []

    def test_attaches_with_explicit_path(self, fake_transformers: Any, tmp_path: Path) -> None:
        del fake_transformers
        trainer = _FakeTrainer()
        writer = attach(trainer, str(tmp_path / "p.jsonl"))
        assert writer is not None
        assert len(trainer.callbacks) == 1

    def test_env_fallback(
        self, fake_transformers: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        del fake_transformers
        env_path = tmp_path / "from_env.jsonl"
        monkeypatch.setenv("FORGE_PROGRESS_PATH", str(env_path))
        trainer = _FakeTrainer()
        writer = attach(trainer, None)
        assert writer is not None
        assert writer.path == env_path
        assert len(trainer.callbacks) == 1


# ---------------------------------------------------------------------------
# SFTRunner.build_trainer wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_trainer_stack(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Minimal transformers + trl + datasets so build_trainer runs offline."""
    state: dict[str, Any] = {"callbacks": []}

    tmod = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> Any:
            del model_id, kwargs
            return MagicMock(name="model")

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> Any:
            del model_id, kwargs
            return MagicMock(name="tokenizer")

    tmod.TrainerCallback = TrainerCallback  # type: ignore[attr-defined]
    tmod.AutoModelForCausalLM = _AutoModel  # type: ignore[attr-defined]
    tmod.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", tmod)

    trl = types.ModuleType("trl")

    class _SFTConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _SFTTrainer:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def add_callback(self, cb: Any) -> None:
            state["callbacks"].append(cb)

    trl.SFTConfig = _SFTConfig  # type: ignore[attr-defined]
    trl.SFTTrainer = _SFTTrainer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "datasets", types.ModuleType("datasets"))
    return state


class TestSFTRunnerProgressWiring:
    def test_registers_callback_when_path_set(
        self, fake_trainer_stack: dict[str, Any], tmp_path: Path
    ) -> None:
        cfg = SFTConfig(
            model_id="gpt2", output_dir="./out", progress_jsonl=str(tmp_path / "p.jsonl")
        )
        SFTRunner(cfg).build_trainer(train_dataset=["row"])
        assert len(fake_trainer_stack["callbacks"]) == 1

    def test_no_callback_without_path(
        self, fake_trainer_stack: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FORGE_PROGRESS_PATH", raising=False)
        cfg = SFTConfig(model_id="gpt2", output_dir="./out")
        SFTRunner(cfg).build_trainer(train_dataset=["row"])
        assert fake_trainer_stack["callbacks"] == []
