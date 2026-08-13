"""The VM-side fine-tuning runner.

The security-critical surface is the INERT spec (nothing in it can name Python to run) and the
guarantee that the HF write token never reaches a progress event. The orchestration is tested with
the heavy externals faked — `datasets` for the split, the method's runner for the training itself,
the Hub client for the push — because none of them can run here and none of them is what these
assertions are about.

The other thing under test is WHEN a run fails. Every check that can happen before the GPU is
touched is worth more than the same check on the box: on a rented machine the difference between
a validation error and a TRL stack trace is minutes of provisioning and the cost of the hardware.
"""

from __future__ import annotations

import contextlib
import json
import time
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from strata_forge.pipelines import finetune_runner as fr
from strata_forge.pipelines._common import RunError
from strata_forge.training.progress import JsonlProgressWriter

if TYPE_CHECKING:
    from pathlib import Path

_TOKEN = "hf_secretwritetoken1234567890"


@contextlib.asynccontextmanager
async def _fast_ticking_phase(phase: Any, message: str, interval_s: float = 0.0) -> Any:
    """`ticking_phase` at a millisecond cadence, so a short test step still re-stamps."""
    del interval_s
    from strata_forge.pipelines._common import ticking_phase

    async with ticking_phase(phase, message, interval_s=0.005):
        yield


def _spec_json(**overrides: Any) -> str:
    base: dict[str, Any] = {
        "method": "sft",
        "model_id": "org/model",
        "dataset_id": "org/ds",
        "split": "train",
        "dataset_format": "prompt_completion",
        "column_mapping": {"prompt": "question", "completion": "answer"},
        "output_repo_id": "org/out",
        "progress_path": "progress.jsonl",
    }
    base.update(overrides)
    return json.dumps(base)


def _spec(**overrides: Any) -> fr.FinetuneSpec:
    return fr.FinetuneSpec.model_validate_json(_spec_json(**overrides))


# What `_train` returns: the trainer's own numbers, extracted where the phase boundary needs them.
_TRAINED: tuple[dict[str, float], int | None] = ({"train_loss": 0.5}, 12)


# ------------------------------ the inert spec ------------------------------


class TestLoadSpec:
    def test_accepts_a_valid_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json())
        spec = fr.load_spec()
        assert spec.method == "sft"
        assert spec.adapter == "lora"  # the default: a full fine-tune must be asked for

    def test_rejects_an_unknown_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # extra="forbid" is why a spec cannot smuggle an instruction past the runner.
        monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(reward_fn="os.system"))
        with pytest.raises(RunError, match="invalid STRATA_RUN_CONFIG"):
            fr.load_spec()

    @pytest.mark.parametrize(
        "bad", ["../etc/passwd", "org/../x", "org/model\n", "https://hf.co/org/m", "a b/c"]
    )
    def test_rejects_a_malformed_model_id(self, monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
        monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(model_id=bad))
        with pytest.raises(RunError, match="invalid model id"):
            fr.load_spec()

    def test_accepts_a_bare_canonical_model_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # gpt2 / t5-small / distilgpt2 live at the root of the Hub with no owner.
        monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(model_id="gpt2"))
        assert fr.load_spec().model_id == "gpt2"

    def test_rejects_an_unknown_method_before_the_gpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(method="ppo"))
        with pytest.raises(RunError, match="unknown fine-tuning method 'ppo'"):
            fr.load_spec()

    def test_grpo_is_declined_with_its_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Not "unknown": it exists, and cannot be driven from a spec that carries no code.
        monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(method="grpo"))
        with pytest.raises(RunError, match="reward function"):
            fr.load_spec()

    def test_rejects_a_format_the_method_cannot_train_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "STRATA_RUN_CONFIG",
            _spec_json(
                method="sft",
                dataset_format="preference",
                column_mapping={"chosen": "good", "rejected": "bad"},
            ),
        )
        with pytest.raises(RunError, match="sft cannot train on the 'preference'"):
            fr.load_spec()

    def test_accepts_dpo_over_preference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "STRATA_RUN_CONFIG",
            _spec_json(
                method="dpo",
                dataset_format="preference",
                column_mapping={"chosen": "good", "rejected": "bad"},
            ),
        )
        assert fr.load_spec().method == "dpo"

    def test_a_merge_with_no_adapter_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(adapter="none", merge_adapter=True))
        with pytest.raises(RunError, match="nothing to merge"):
            fr.load_spec()

    def test_a_full_finetune_without_a_merge_is_fine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRATA_RUN_CONFIG", _spec_json(adapter="none"))
        assert fr.load_spec().adapter == "none"


# ------------------------------ adapter + trainer config ---------------------


class TestPeftConfig:
    def test_none_means_a_full_finetune(self) -> None:
        from strata_forge.training.methods import pick_method

        assert fr._peft_config(_spec(adapter="none"), pick_method("sft")) is None  # pyright: ignore[reportPrivateUsage]

    def test_lora_carries_the_spec_knobs_and_the_method_s_task_type(self) -> None:
        from strata_forge.training.methods import pick_method
        from strata_forge.training.peft import LoRAConfig

        cfg = fr._peft_config(  # pyright: ignore[reportPrivateUsage]
            _spec(adapter="lora", lora={"r": 8, "alpha": 64, "dropout": 0.1}), pick_method("dpo")
        )
        assert isinstance(cfg, LoRAConfig)
        assert (cfg.r, cfg.alpha, cfg.dropout) == (8, 64, 0.1)
        assert cfg.task_type == "CAUSAL_LM"

    def test_qlora_wraps_the_same_lora(self) -> None:
        from strata_forge.training.methods import pick_method
        from strata_forge.training.peft import QLoRAConfig

        cfg = fr._peft_config(_spec(adapter="qlora", lora={"r": 4}), pick_method("sft"))  # pyright: ignore[reportPrivateUsage]
        assert isinstance(cfg, QLoRAConfig)
        assert cfg.lora.r == 4
        assert cfg.load_in_4bit is True


class TestTrainerConfig:
    def test_sft_over_flat_text_names_the_column(self, tmp_path: Path) -> None:
        from strata_forge.training.methods import pick_method

        cfg = fr._trainer_config(  # pyright: ignore[reportPrivateUsage]
            _spec(dataset_format="text", column_mapping={"text": "body"}),
            pick_method("sft"),
            tmp_path,
        )
        assert cfg.dataset_text_field == "text"

    def test_sft_over_role_columns_lets_trl_infer(self, tmp_path: Path) -> None:
        # Naming a text field here would make TRL ignore the role columns entirely.
        from strata_forge.training.methods import pick_method

        cfg = fr._trainer_config(_spec(), pick_method("sft"), tmp_path)  # pyright: ignore[reportPrivateUsage]
        assert cfg.dataset_text_field is None

    def test_hyperparams_reach_the_method_s_own_config(self, tmp_path: Path) -> None:
        from strata_forge.training.methods import pick_method

        cfg = fr._trainer_config(  # pyright: ignore[reportPrivateUsage]
            _spec(
                method="dpo",
                dataset_format="preference",
                column_mapping={"chosen": "c", "rejected": "r"},
                hyperparams={"beta": 0.25, "learning_rate": 1e-6},
            ),
            pick_method("dpo"),
            tmp_path,
        )
        assert cfg.beta == 0.25
        assert cfg.learning_rate == 1e-6

    def test_a_knob_the_method_does_not_have_is_named_not_ignored(self, tmp_path: Path) -> None:
        # The config's own extra="forbid" is the authority; this is the error the user sees
        # instead of a silently dropped setting.
        from strata_forge.training.methods import pick_method

        with pytest.raises(RunError, match="invalid hyperparams for sft"):
            fr._trainer_config(  # pyright: ignore[reportPrivateUsage]
                _spec(hyperparams={"beta": 0.1}), pick_method("sft"), tmp_path
            )

    def test_the_progress_path_is_threaded_to_the_trainer_callback(self, tmp_path: Path) -> None:
        # This is the whole live-loss wiring: the trainer writes to the same file the runner does.
        from strata_forge.training.methods import pick_method

        cfg = fr._trainer_config(_spec(), pick_method("sft"), tmp_path)  # pyright: ignore[reportPrivateUsage]
        assert cfg.progress_jsonl == "progress.jsonl"


# ------------------------------ loading a split ------------------------------


class _FakeSplit:
    def __init__(self, rows: list[dict[str, Any]], columns: list[str]) -> None:
        self._rows = rows
        self.column_names = columns

    def __iter__(self) -> Any:
        return iter(self._rows)


def _fake_datasets(
    monkeypatch: pytest.MonkeyPatch, split: _FakeSplit, seen: list[dict[str, Any]] | None = None
) -> None:
    import sys
    import types

    mod = types.ModuleType("datasets")

    def _load_dataset(repo: str, **kwargs: Any) -> _FakeSplit:
        if seen is not None:
            seen.append({"repo": repo, **kwargs})
        return split

    class _Dataset:
        @staticmethod
        def from_list(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return rows

    mod.load_dataset = _load_dataset  # pyright: ignore[reportAttributeAccessIssue]
    mod.Dataset = _Dataset  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "datasets", mod)


class TestLoadSplit:
    def test_projects_columns_onto_the_format_roles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        split = _FakeSplit(
            [{"question": "q", "answer": "a", "id": 1}], ["question", "answer", "id"]
        )
        _fake_datasets(monkeypatch, split)
        rows = fr._load_split(_spec(), "train", None)  # pyright: ignore[reportPrivateUsage]
        # Everything the trainer did not ask for is dropped: a stray column changes what TRL infers.
        assert rows == [{"prompt": "q", "completion": "a"}]

    def test_row_limit_bounds_materialisation_during_iteration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A split's size is not under our control, so the cap must apply while iterating rather
        # than after — otherwise a huge split OOMs the box before the limit is ever consulted.
        class _NeverEnding:
            column_names: ClassVar[list[str]] = ["question", "answer"]

            def __iter__(self) -> Any:
                while True:
                    yield {"question": "q", "answer": "a"}

        _fake_datasets(monkeypatch, _NeverEnding())  # pyright: ignore[reportArgumentType]
        rows = fr._load_split(_spec(row_limit=3), "train", None)  # pyright: ignore[reportPrivateUsage]
        assert len(rows) == 3

    def test_a_missing_column_names_the_split_and_the_columns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_datasets(monkeypatch, _FakeSplit([{"q": "x"}], ["q"]))
        with pytest.raises(RunError) as exc:
            fr._load_split(_spec(), "train", None)  # pyright: ignore[reportPrivateUsage]
        assert "train:" in str(exc.value)
        assert "question" in str(exc.value)

    def test_the_pinned_commit_and_config_reach_the_loader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[dict[str, Any]] = []
        _fake_datasets(
            monkeypatch,
            _FakeSplit([{"question": "q", "answer": "a"}], ["question", "answer"]),
            seen,
        )
        fr._load_split(  # pyright: ignore[reportPrivateUsage]
            _spec(dataset_commit_sha="abc123", dataset_config="en"), "validation", _TOKEN
        )
        assert seen[0]["revision"] == "abc123"
        assert seen[0]["name"] == "en"
        assert seen[0]["split"] == "validation"
        # The read token is passed EXPLICITLY, never left to the VM's ambient HF_TOKEN.
        assert seen[0]["token"] == _TOKEN


# ------------------------------ the run ---------------------------------------


def _drive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    spec: fr.FinetuneSpec | None = None,
    token: str | None = _TOKEN,
    push: Any = None,
    train: Any = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run `_execute` with the heavy externals faked; returns (destination, progress events)."""
    import asyncio

    progress = tmp_path / "progress.jsonl"
    writer = JsonlProgressWriter(progress)

    def _rows(*_a: Any, **_k: Any) -> list[dict[str, Any]]:
        return [{"prompt": "q", "completion": "a"}]

    def _identity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return rows

    def _fake_build(*_a: Any, **_k: Any) -> object:
        return object()

    def _fake_train(*_a: Any, **_k: Any) -> tuple[dict[str, float], int | None]:
        return _TRAINED

    def _dir(_run_id: str | None, *, name: str) -> Path:
        return tmp_path / name

    monkeypatch.setattr(fr, "_load_split", _rows)
    monkeypatch.setattr(fr, "_to_dataset", _identity)
    monkeypatch.setattr(fr, "_build_trainer", _fake_build)
    monkeypatch.setattr(fr, "_train", train or _fake_train)
    monkeypatch.setattr(fr, "results_dir", _dir)
    if push is not None:
        monkeypatch.setattr(fr, "_push_artifact", push)

    destination = asyncio.run(fr._execute(spec or _spec(), token, writer))  # pyright: ignore[reportPrivateUsage]
    writer.close()
    events = [json.loads(ln) for ln in progress.read_text().splitlines() if ln.strip()]
    return destination, events


class TestExecute:
    def test_a_pushed_run_reports_the_repo_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def _push(_spec_arg: Any, _dir: Any, _token: str) -> str:
            return "org/out"

        destination, events = _drive(monkeypatch, tmp_path, push=_push)
        assert destination == "org/out"
        assert events[-1]["kind"] == "end"
        assert events[-1]["message"] == "org/out"

    def test_the_terminal_event_carries_the_step_count(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # It is the last event anybody reads; without the count a finished run's progress reads
        # as unknown rather than as complete.
        async def _push(*_a: Any) -> str:
            return "org/out"

        _, events = _drive(monkeypatch, tmp_path, push=_push)
        assert events[-1]["step"] == 12
        assert events[-1]["total_steps"] == 12
        assert events[-1]["metrics"] == {"train_loss": 0.5}

    def test_with_no_output_repo_the_artifact_stays_on_the_vm(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        destination, events = _drive(monkeypatch, tmp_path, spec=_spec(output_repo_id=None))
        assert destination.endswith("artifact")
        assert events[-1]["message"] == destination

    def test_with_no_write_token_the_artifact_stays_on_the_vm(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The push needs BOTH; a repo id alone must not make the runner guess ownership.
        destination, _ = _drive(monkeypatch, tmp_path, token=None)
        assert destination.endswith("artifact")

    def test_the_artifact_dir_lives_outside_the_workdir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The orchestrator deletes the per-run workdir on every terminal state. Weights written
        # there would be destroyed by a failed upload along with the whole run's output.
        destination, _ = _drive(monkeypatch, tmp_path, spec=_spec(output_repo_id=None))
        assert fr._OUTPUT_DIR_NAME in destination  # pyright: ignore[reportPrivateUsage]

    def test_a_failed_push_says_where_the_weights_are(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def _boom(*_a: Any) -> str:
            msg = "429 rate limited"
            raise RuntimeError(msg)

        with pytest.raises(RunError) as exc:
            _drive(monkeypatch, tmp_path, push=_boom)
        assert "on the VM at" in str(exc.value)
        assert fr._OUTPUT_DIR_NAME in str(exc.value)  # pyright: ignore[reportPrivateUsage]

    def test_the_phases_narrate_the_uncountable_stretches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def _push(*_a: Any) -> str:
            return "org/out"

        _, events = _drive(monkeypatch, tmp_path, push=_push)
        phases = [e["message"] for e in events if e["kind"] == "phase"]
        assert "Loading the dataset" in phases
        assert "Loading the model onto the GPU" in phases
        assert "Uploading the model to the Hub" in phases

    def test_the_runner_emits_no_start_of_its_own(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The trainer's callback emits `start` with the total step count only it can know.

        A second one would make the control plane's has-this-actually-begun gate fire on the
        wrong event — while the box is still downloading a model.
        """
        _, events = _drive(monkeypatch, tmp_path, spec=_spec(output_repo_id=None))
        assert [e for e in events if e["kind"] == "start"] == []

    def test_the_write_token_never_reaches_a_progress_event(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def _boom(*_a: Any) -> str:
            msg = f"401 using {_TOKEN}"
            raise RuntimeError(msg)

        with pytest.raises(RunError):
            _drive(monkeypatch, tmp_path, push=_boom)
        assert _TOKEN not in (tmp_path / "progress.jsonl").read_text()

    def test_the_eval_split_is_loaded_when_asked_for(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: list[str] = []

        def _record(_spec_arg: Any, split: str, _token: str | None) -> list[dict[str, Any]]:
            seen.append(split)
            return [{"prompt": "q", "completion": "a"}]

        def _identity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return rows

        def _fake_build(*_a: Any, **_k: Any) -> object:
            return object()

        def _fake_train(*_a: Any, **_k: Any) -> tuple[dict[str, float], int | None]:
            return _TRAINED

        def _dir(_run_id: str | None, *, name: str) -> Path:
            return tmp_path / name

        monkeypatch.setattr(fr, "_load_split", _record)
        monkeypatch.setattr(fr, "_to_dataset", _identity)
        monkeypatch.setattr(fr, "_build_trainer", _fake_build)
        monkeypatch.setattr(fr, "_train", _fake_train)
        monkeypatch.setattr(fr, "results_dir", _dir)
        import asyncio

        asyncio.run(fr._execute(_spec(eval_split="test", output_repo_id=None), None, None))  # pyright: ignore[reportPrivateUsage]
        assert seen == ["train", "test"]

    def test_the_merge_step_runs_only_when_asked_for(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        merged: list[bool] = []

        def _record_merge(*_a: Any) -> None:
            merged.append(True)

        monkeypatch.setattr(fr, "_merge_adapter", _record_merge)
        _, events = _drive(monkeypatch, tmp_path, spec=_spec(output_repo_id=None))
        assert merged == []

        _, events = _drive(
            monkeypatch, tmp_path, spec=_spec(output_repo_id=None, merge_adapter=True)
        )
        assert merged == [True]
        assert any("Merging the adapter" in e.get("message", "") for e in events)


class TestPhaseBoundary:
    """The loop narrates itself; the runner narrates only the silence around it.

    Building the trainer (downloading the model, quantising it, wiring the adapter) is silent and
    can take many minutes, so it gets an elapsed-stamping caption. The loop that follows must NOT:
    the trainer's own callback writes steps, loss and eval to this same file, and a caption
    re-stamped over that is a stale sentence on top of live progress — and on a multi-day run, tens
    of thousands of rows of it, enough to exhaust the orchestrator's per-run event budget and cut
    off the run's own outcome.
    """

    def test_the_caption_covers_building_and_stops_before_the_loop(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import asyncio

        progress = tmp_path / "progress.jsonl"
        writer = JsonlProgressWriter(progress)
        captions_during_build: list[int] = []

        def _slow_build(*_a: Any, **_k: Any) -> object:
            time.sleep(0.05)  # stands in for the model download
            return object()

        def _slow_train(*_a: Any, **_k: Any) -> tuple[dict[str, float], int | None]:
            # Record how many phase rows existed when the loop began, then take just as long.
            captions_during_build.append(len(progress.read_text().splitlines()))
            time.sleep(0.05)
            return _TRAINED

        def _dir(_run_id: str | None, *, name: str) -> Path:
            return tmp_path / name

        monkeypatch.setattr(
            fr, "_load_split", lambda *_a, **_k: [{"prompt": "q", "completion": "a"}]
        )
        monkeypatch.setattr(fr, "_to_dataset", lambda rows: rows)
        monkeypatch.setattr(fr, "_build_trainer", _slow_build)
        monkeypatch.setattr(fr, "_train", _slow_train)
        monkeypatch.setattr(fr, "results_dir", _dir)
        # A fast tick, so an equally-long build and loop are told apart by their row counts.
        monkeypatch.setattr(fr, "ticking_phase", _fast_ticking_phase)

        asyncio.run(fr._execute(_spec(output_repo_id=None), None, writer))  # pyright: ignore[reportPrivateUsage]
        writer.close()

        events = [json.loads(ln) for ln in progress.read_text().splitlines() if ln.strip()]
        loading = [e for e in events if e["message"].startswith("Loading the model onto the GPU")]
        assert len(loading) > 1, "the build step must re-stamp its elapsed time"

        # The row count did not grow across the loop: every event after the build is the
        # terminal one.
        before_loop = captions_during_build[0]
        assert len(events) == before_loop + 1
        assert events[-1]["kind"] == "end"

    def test_build_and_train_are_separate_steps(self) -> None:
        # The seam itself. `SFTRunner.train()` bundles building, running and saving into one call;
        # this module needs the boundary between the first two, which is why it does not use it.
        import inspect

        source = inspect.getsource(fr._execute)  # pyright: ignore[reportPrivateUsage]
        assert "_build_trainer" in source
        build_at = source.index("_build_trainer")
        train_at = source.index("_train, trainer")
        caption_at = source.index('"Loading the model onto the GPU"')
        assert caption_at < build_at < train_at
