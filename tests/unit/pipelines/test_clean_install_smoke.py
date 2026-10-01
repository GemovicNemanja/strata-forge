"""The inputs of the nightly clean-install smoke (``scripts/smoke_clean_install.py``).

The smoke itself runs only in a fresh venv with the heavy extras, once a night. Its INPUTS are pure
and are checked here instead, against the runners' own validation, so a change to a runner spec or
a registry that would break the smoke fails the pull request that makes it rather than the next
night's job. The smoke is registry-driven: these tests also pin that a matrix leg can never
quietly select nothing to check.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

from strata_forge.pipelines import inference_runner
from strata_forge.training.dataset_format import (
    FORMATS,
    build_training_rows,
    pick_format,
    validate_mapping,
)
from strata_forge.training.methods import enabled_methods

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "smoke_clean_install.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("smoke_clean_install", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load_script()


class TestFinetuneCases:
    def test_covers_every_enabled_method_format_and_adapter(self) -> None:
        expected = {
            (method.name, fmt, adapter)
            for method in enabled_methods()
            for fmt in method.formats
            for adapter in ("none", "lora", "qlora")
        }
        assert set(smoke.finetune_cases()) == expected

    def test_every_dataset_format_is_reached(self) -> None:
        reached = {fmt for _, fmt, _ in smoke.finetune_cases()}
        # A format no enabled method trains on would be untested by the smoke; there is none.
        assert reached == set(FORMATS)

    # The runner refuses a qlora spec on a machine without bitsandbytes, and the dev environment
    # does not install [finetuning]; the clean install the smoke runs in does.
    @pytest.mark.usefixtures("bitsandbytes_installed")
    @pytest.mark.parametrize(("method", "fmt", "adapter"), smoke.finetune_cases())
    def test_spec_passes_the_runner_and_builds_its_config(
        self, method: str, fmt: str, adapter: str, tmp_path: Path
    ) -> None:
        built_method, config, peft = smoke.build_finetune_config(
            smoke.finetune_spec(method, fmt, adapter), tmp_path / "out"
        )
        assert built_method.name == method
        assert config.model_id == "smoke-org/smoke-model"
        assert (peft is None) == (adapter == "none")
        assert "output_dir" in config.to_trl_kwargs()

    @pytest.mark.parametrize("fmt", sorted(FORMATS))
    def test_rows_map_every_role_and_drop_the_rest(self, fmt: str) -> None:
        spec = smoke.finetune_spec("sft", fmt, "none")
        rows = smoke.finetune_rows(fmt)
        validate_mapping(fmt, spec["column_mapping"], list(rows[0]))
        built = build_training_rows(rows, fmt, spec["column_mapping"])
        assert [set(row) for row in built] == [set(pick_format(fmt).roles)] * len(rows)


class TestInference:
    def test_spec_passes_the_runner(self) -> None:
        with smoke.run_config(smoke.inference_spec()):
            spec = inference_runner.load_spec()
        assert spec.output_repo_id == "smoke-org/smoke-output"

    def test_serving_command_carries_every_optional_flag(self) -> None:
        argv = smoke.vllm_command()
        assert argv[:2] == [sys.executable, "-m"]
        for flag in ("--model", "--host", "--port", "--tensor-parallel-size"):
            assert flag in argv
        # Both optional flags are switched on, so vLLM's parser sees everything the runner emits.
        assert argv[argv.index("--max-model-len") + 1] == "4096"
        assert argv[argv.index("--dtype") + 1] == "bfloat16"


class TestRunConfig:
    def test_restores_the_previous_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRATA_RUN_CONFIG", "before")
        with smoke.run_config({"a": 1}):
            assert os.environ["STRATA_RUN_CONFIG"] == '{"a": 1}'
        assert os.environ["STRATA_RUN_CONFIG"] == "before"

    def test_removes_a_value_it_introduced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("STRATA_RUN_CONFIG", raising=False)
        with smoke.run_config({"a": 1}):
            pass
        assert "STRATA_RUN_CONFIG" not in os.environ


class TestPlan:
    @staticmethod
    def _names(extras: str) -> list[str]:
        return [name for name, _ in smoke.plan(smoke.selected_extras(extras))]

    def test_finetune_leg_checks_every_case_and_no_serving(self) -> None:
        names = self._names("finetuning,storage")
        assert sum(name.startswith("finetune ") for name in names) == len(smoke.finetune_cases())
        assert not any("vllm" in name or "inference" in name for name in names)

    def test_inference_leg_checks_serving_and_no_training(self) -> None:
        names = self._names("serving,storage,hf")
        assert "inference spec, requests and results" in names
        assert "vllm accepts the serving command line" in names
        assert not any(name.startswith("finetune ") for name in names)

    def test_all_expands_to_every_declared_extra(self) -> None:
        extras = smoke.selected_extras("all")
        assert {"finetuning", "serving", "storage", "hf"} <= extras
        names = self._names("all")
        assert "vllm accepts the serving command line" in names
        assert any(name.startswith("finetune ") for name in names)


def test_install_check_refuses_a_source_tree_import() -> None:
    # The unit suite imports strata_forge from the checkout (an editable install), which is
    # exactly the situation in which the smoke would prove nothing about a clean install.
    with pytest.raises(smoke.SmokeError, match="not from an installed distribution"):
        smoke.check_install()


class TestOverride:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "transformers==4.57.6 huggingface-hub==0.36.0",
            "torch>=2.9,<3",
            "trl~=1.14\nPEFT!=0.21.0",
            "strata_forge[serving,hf]==0.2.0",
            "transformers==5.*",
        ],
    )
    def test_accepts_plain_pins(self, text: str) -> None:
        assert smoke.override_requirements(text) == text.split()

    @pytest.mark.parametrize(
        "token",
        [
            "six@https://example.invalid/six.whl",
            "--index-url=https://example.invalid/simple",
            "--extra-index-url",
            "-e",
            "./six-1.0-py3-none-any.whl",
            "https://example.invalid/six.whl",
            "git+https://example.invalid/six.git",
            "file:///tmp/six.whl",
            "transformers",
            'transformers==4.57.6;python_version>"3"',
            "transformers==4.57.6@https://example.invalid/x.whl",
        ],
    )
    def test_refuses_anything_that_is_not_a_pin(self, token: str) -> None:
        with pytest.raises(smoke.SmokeError, match="override refused"):
            smoke.override_requirements(f"transformers==4.57.6 {token}")

    def test_writes_one_requirement_a_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "override.txt"
        assert smoke.main(["--write-override", str(target), " a==1\n b>=2,<3 "]) == 0
        assert target.read_text(encoding="utf-8") == "a==1\nb>=2,<3\n"
        assert (
            "::warning title=Override drill::resolving with: a==1 b>=2,<3"
            in capsys.readouterr().out
        )

    def test_a_refused_override_writes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "override.txt"
        assert smoke.main(["--write-override", str(target), "six@https://example.invalid/x"]) == 2
        assert not target.exists()
        assert "::error title=Override refused::" in capsys.readouterr().out


class TestRunChecks:
    @staticmethod
    def _fail() -> None:
        raise smoke.SmokeError("broken")

    @staticmethod
    def _exit_green() -> None:
        raise SystemExit(0)

    def test_all_passing_exits_zero(self) -> None:
        assert smoke.run_checks([("a", lambda: None), ("b", lambda: None)]) == 0

    def test_a_failing_check_exits_one_and_the_rest_still_run(self) -> None:
        ran: list[str] = []
        checks = [("a", self._fail), ("b", lambda: ran.append("b"))]
        assert smoke.run_checks(checks) == 1
        assert ran == ["b"]

    def test_a_dependency_exiting_zero_is_a_failure(self) -> None:
        ran: list[str] = []
        checks = [("a", self._exit_green), ("b", lambda: ran.append("b"))]
        assert smoke.run_checks(checks) == 1
        assert ran == ["b"]

    def test_an_empty_plan_is_a_failure(self) -> None:
        assert smoke.run_checks([]) == 1


class TestBindingOnly:
    class _Trainer:
        def __init__(self, model: Any, args: Any, processing_class: Any = None) -> None:
            raise AssertionError("the real constructor must not run")

    class _Inherits(_Trainer):
        pass

    #: The keywords a runner would pass after TRL renamed ``tokenizer`` to ``processing_class``.
    _DROPPED: ClassVar[dict[str, Any]] = {"model": 1, "args": 2, "tokenizer": 3}

    def test_binds_the_runner_keywords_then_stops(self) -> None:
        with smoke.binding_only(self._Trainer), pytest.raises(smoke._Constructed):
            self._Trainer(model=1, args=2, processing_class=3)

    def test_a_keyword_the_class_dropped_fails(self) -> None:
        with (
            smoke.binding_only(self._Trainer),
            pytest.raises(smoke.SmokeError, match="no longer accepts the runner's arguments"),
        ):
            self._Trainer(**self._DROPPED)

    def test_restores_the_constructor(self) -> None:
        original = self._Trainer.__init__
        with smoke.binding_only(self._Trainer):
            pass
        assert self._Trainer.__init__ is original
        with smoke.binding_only(self._Inherits):
            pass
        assert "__init__" not in vars(self._Inherits)


class TestQLoRA:
    @pytest.mark.parametrize(
        ("requires", "declared"),
        [
            (['bitsandbytes>=0.45,<1; extra == "finetuning"'], True),
            (["bitsandbytes>=0.45,<1 ; extra == 'finetuning'"], True),
            (['bitsandbytes>=0.45,<1; extra == "serving"'], False),
            (['bitsandbytes-extras>=1; extra == "finetuning"'], False),
            (['peft>=0.13; extra == "finetuning"'], False),
            ([], False),
        ],
    )
    def test_reads_the_finetuning_extra(
        self, monkeypatch: pytest.MonkeyPatch, requires: list[str], declared: bool
    ) -> None:
        def fake_requires(_name: str) -> list[str]:
            return requires

        monkeypatch.setattr(smoke.importlib.metadata, "requires", fake_requires)
        assert smoke.bitsandbytes_declared() is declared

    def test_a_declared_but_missing_bitsandbytes_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(smoke, "bitsandbytes_declared", lambda: True)

        def not_installed(_name: str) -> None:
            return None

        monkeypatch.setattr(smoke.importlib.util, "find_spec", not_installed)
        with pytest.raises(smoke.SmokeError, match="names bitsandbytes"):
            smoke.check_qlora_loadable()

    def test_an_undeclared_bitsandbytes_is_warned_once(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(smoke, "bitsandbytes_declared", lambda: False)
        no_gaps: set[str] = set()
        monkeypatch.setattr(smoke, "_REPORTED_GAPS", no_gaps)
        smoke.check_qlora_loadable()
        smoke.check_qlora_loadable()
        assert capsys.readouterr().out.count("::warning title=QLoRA cannot load") == 1


def test_probe_carries_every_vllm_variable_the_runner_sets() -> None:
    # vLLM ignores a variable it does not know, so the probe checks the runner's are still read;
    # a variable added to the runner without being added to the smoke would go unchecked.
    source = Path(inference_runner.__file__).read_text(encoding="utf-8")
    runner_env = dict(re.findall(r'"(VLLM_[A-Z0-9_]+)":\s*"([^"]*)"', source))
    smoke_env = {k: v for k, v in smoke._RUNNER_SERVE_ENV.items() if k.startswith("VLLM_")}
    assert runner_env
    assert runner_env == smoke_env
