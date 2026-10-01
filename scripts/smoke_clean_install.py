"""Build what a run builds, inside a fresh install of the engine.

The nightly clean-install job (``.github/workflows/clean-install.yml``) runs this with the
interpreter of an empty virtualenv holding ONLY ``strata-forge[<extras>]``, resolved from the
public index with no lockfile and no constraints. That is the install a run's machine performs,
so an upstream release that renames a keyword, moves a class or drops a CLI flag fails here the
day it ships, rather than on a user's GPU after provisioning has been paid for.

It constructs, offline, everything a run constructs before it loads weights:

- every enabled fine-tuning method x every dataset format it accepts x every adapter: the inert
  spec through the runner's own ``load_spec``, the typed config, the TRL config and trainer class,
  the peft config and the bitsandbytes config, and the projected training rows as a ``Dataset``;
- the batch-inference spec, its requests, its results parquet, the client it talks through, and
  the vLLM command line, parsed by the installed vLLM's own entrypoint;
- every top-level module import and the CLI's ``--help``.

Usage::

    .venv-smoke/bin/python scripts/smoke_clean_install.py finetuning,storage

The argument is the extras string the venv was installed with; it selects which checks apply.
Exits non-zero when any check fails, after running all of them.
"""

from __future__ import annotations

import os

# Before anything imports huggingface_hub, which reads these once at import time: nothing here may
# reach the Hub or phone home, so a check cannot pass or fail on the network's say-so.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

import argparse
import asyncio
import contextlib
import importlib
import importlib.metadata
import json
import pkgutil
import shlex
import subprocess
import sys
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

#: Packages whose resolved version is worth a line in the log. Absent ones are skipped.
_REPORTED = (
    "strata-forge",
    "pydantic",
    "litellm",
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "huggingface_hub",
    "vllm",
)

#: Placeholder repo ids. They only have to pass ``validate_repo_id``; nothing is fetched.
_MODEL = "smoke-org/smoke-model"
_DATASET = "smoke-org/smoke-dataset"
_OUTPUT = "smoke-org/smoke-output"

_ADAPTERS = ("none", "lora", "qlora")


class SmokeError(RuntimeError):
    """A check that ran to completion and found the wrong thing."""


# --------------------------------------------------------------------------------------------
# Inputs. Pure: no extra beyond the base install is needed to build them, so the unit suite
# exercises them too and a spec change breaks a pull request rather than the next night.
# --------------------------------------------------------------------------------------------


def sample_value(role: str) -> Any:
    """A cell of the type a dataset-format role is trained as."""
    if role == "messages":
        return [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ]
    if role == "label":
        return True
    return f"sample {role}"


def finetune_cases() -> list[tuple[str, str, str]]:
    """Every (method, dataset format, adapter) a declaration may name, read from the registries."""
    from strata_forge.training.methods import enabled_methods

    return [
        (method.name, fmt, adapter)
        for method in enabled_methods()
        for fmt in sorted(method.formats)
        for adapter in _ADAPTERS
    ]


def finetune_spec(method: str, fmt: str, adapter: str) -> dict[str, Any]:
    """The inert spec a control plane would deliver for one case, with every role mapped."""
    from strata_forge.training.dataset_format import pick_format

    roles = pick_format(fmt).roles
    return {
        "method": method,
        "model_id": _MODEL,
        "dataset_id": _DATASET,
        "split": "train",
        "eval_split": "test",
        "dataset_format": fmt,
        "column_mapping": {role: f"col_{role}" for role in roles},
        "adapter": adapter,
        "merge_adapter": adapter != "none",
        "row_limit": 8,
        "output_repo_id": _OUTPUT,
        "run_id": "smoke",
    }


def finetune_rows(fmt: str) -> list[dict[str, Any]]:
    """Two source rows carrying a column for every role of ``fmt``, plus one the trainer ignores."""
    from strata_forge.training.dataset_format import pick_format

    row = {f"col_{role}": sample_value(role) for role in pick_format(fmt).roles}
    return [{**row, "unrelated": "dropped"}, dict(row)]


def inference_spec() -> dict[str, Any]:
    """The inert batch-inference spec, with every serving flag the runner can emit switched on."""
    return {
        "model_id": _MODEL,
        "dataset_id": _DATASET,
        "split": "train",
        "column_mapping": {"question": "col_question"},
        "template": "Answer briefly: {question}",
        "hyperparams": {
            "temperature": 0.0,
            "max_tokens": 64,
            "tensor_parallel_size": 1,
            "max_model_len": 4096,
            "dtype": "bfloat16",
            "row_limit": 8,
        },
        "output_repo_id": _OUTPUT,
        "run_id": "smoke",
    }


@contextlib.contextmanager
def run_config(spec: dict[str, Any]) -> Generator[None]:
    """Deliver ``spec`` the way a run receives it: as ``STRATA_RUN_CONFIG``."""
    previous = os.environ.get("STRATA_RUN_CONFIG")
    os.environ["STRATA_RUN_CONFIG"] = json.dumps(spec)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("STRATA_RUN_CONFIG", None)
        else:
            os.environ["STRATA_RUN_CONFIG"] = previous


def build_finetune_config(spec: dict[str, Any], output_dir: Path) -> tuple[Any, Any, Any]:
    """The runner's own path from an inert spec to (method, typed config, adapter config)."""
    from strata_forge.pipelines import finetune_runner
    from strata_forge.training.methods import pick_method

    with run_config(spec):
        parsed = finetune_runner.load_spec()
    method = pick_method(parsed.method)
    config = finetune_runner._trainer_config(parsed, method, output_dir)  # pyright: ignore[reportPrivateUsage]
    peft = finetune_runner._peft_config(parsed, method)  # pyright: ignore[reportPrivateUsage]
    return method, config, peft


# --------------------------------------------------------------------------------------------
# Checks. Each one needs the extras it is registered under, and raises on a failure.
# --------------------------------------------------------------------------------------------


def check_install() -> None:
    import strata_forge

    location = Path(strata_forge.__file__).resolve()
    if "site-packages" not in location.parts:
        msg = f"strata_forge was imported from {location}, not from an installed distribution"
        raise SmokeError(msg)
    packaged = importlib.metadata.version("strata-forge")
    if strata_forge.__version__ != packaged:
        msg = f"__version__ is {strata_forge.__version__} but the distribution is {packaged}"
        raise SmokeError(msg)


def check_module_imports() -> None:
    """Every top-level module imports whatever subset of the extras is installed."""
    import strata_forge

    failed: list[str] = []
    for info in pkgutil.iter_modules(strata_forge.__path__):
        name = f"strata_forge.{info.name}"
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    if failed:
        raise SmokeError("; ".join(failed))


def check_cli() -> None:
    script = Path(sys.executable).parent / "strata-forge"
    result = subprocess.run(  # noqa: S603 - a fixed argv naming this venv's own console script
        [str(script), "--help"], capture_output=True, text=True, check=False, timeout=120
    )
    if result.returncode != 0:
        msg = f"strata-forge --help exited {result.returncode}: {result.stderr.strip()[-2000:]}"
        raise SmokeError(msg)


@contextlib.contextmanager
def bf16_as_on_a_gpu() -> Generator[None]:
    """Let ``TrainingArguments`` accept ``bf16=True`` on a machine with no GPU.

    The configs default to bf16, and transformers refuses it at construction time when no GPU is
    present. The VM has one; this runner does not. Faking the capability probe keeps the kwargs
    byte-identical to a real run, which a ``precision="fp32"`` override would not. Should the
    probe move, the construction fails with the bf16 message and this needs re-pointing.
    """
    torch: Any = importlib.import_module("torch")
    training_args: Any = importlib.import_module("transformers.training_args")
    probe = "is_torch_bf16_gpu_available"
    if torch.cuda.is_available() or not hasattr(training_args, probe):
        yield
        return
    original = getattr(training_args, probe)
    setattr(training_args, probe, lambda: True)
    try:
        yield
    finally:
        setattr(training_args, probe, original)


def check_finetune_case(method_name: str, fmt: str, adapter: str) -> None:
    """One case end to end, up to the point where a real run would download the model."""
    datasets_mod: Any = importlib.import_module("datasets")
    trl_mod: Any = importlib.import_module("trl")
    from strata_forge.training.dataset_format import build_training_rows, validate_mapping
    from strata_forge.training.peft import QLoRAConfig
    from strata_forge.training.preference import (
        _TRAINER_CLASS,  # pyright: ignore[reportPrivateUsage]
        _TRL_CONFIG_CLASS,  # pyright: ignore[reportPrivateUsage]
        _resolve_trl_class,  # pyright: ignore[reportPrivateUsage]
    )

    spec = finetune_spec(method_name, fmt, adapter)
    with tempfile.TemporaryDirectory() as tmp:
        method, config, peft = build_finetune_config(spec, Path(tmp) / "out")
        if method.name == "sft":
            trl_config_cls: Any = trl_mod.SFTConfig
            trainer_cls: Any = trl_mod.SFTTrainer
        else:
            trl_config_cls = _resolve_trl_class(
                trl_mod, method.name, _TRL_CONFIG_CLASS[method.name]
            )
            trainer_cls = _resolve_trl_class(trl_mod, method.name, _TRAINER_CLASS[method.name])
        if not callable(trainer_cls):
            msg = f"{method.name} trainer {trainer_cls!r} is not a class"
            raise SmokeError(msg)
        with bf16_as_on_a_gpu():
            trl_config_cls(**config.to_trl_kwargs())
        if peft is not None:
            peft.to_peft_config()
            if isinstance(peft, QLoRAConfig):
                peft.to_bnb_config()

    rows = finetune_rows(fmt)
    columns = list(rows[0])
    validate_mapping(fmt, spec["column_mapping"], columns)
    built = build_training_rows(rows, fmt, spec["column_mapping"])
    dataset = datasets_mod.Dataset.from_list(built)
    if set(dataset.column_names) != set(spec["column_mapping"]):
        msg = f"training rows carry {dataset.column_names}, expected the roles only"
        raise SmokeError(msg)


def check_inference_spec() -> None:
    """The spec, the requests and the results file, through the runner's own functions."""
    from pydantic import SecretStr

    from strata_forge.llm import LLMClient
    from strata_forge.llm.providers.config import OpenAICompatConfig
    from strata_forge.llm.providers.openai_compat import (
        UNAUTHENTICATED_API_KEY,
        OpenAICompatProvider,
    )
    from strata_forge.pipelines import inference_runner
    from strata_forge.storage import HFHubClient

    with run_config(inference_spec()):
        spec = inference_runner.load_spec()
    rows = [{"col_question": "What is two plus two?"}, {"col_question": "Name a primary colour."}]
    prompts, custom_ids = inference_runner._build_requests(spec, rows)  # pyright: ignore[reportPrivateUsage]
    if len(prompts) != len(rows) or "{question}" in str(prompts[0]):
        msg = f"the template did not render per row: {prompts!r}"
        raise SmokeError(msg)

    provider = OpenAICompatProvider(
        OpenAICompatConfig(
            base_url="http://127.0.0.1:8000/v1", api_key=SecretStr(UNAUTHENTICATED_API_KEY)
        )
    )
    LLMClient(
        model=spec.model_id,
        provider="openai_compat",
        provider_clients={"openai_compat": provider},
    )
    HFHubClient(token=None)

    results = [
        {"custom_id": custom_ids[0], "output": "4", "error": None},
        {"custom_id": custom_ids[1], "output": None, "error": "timed out"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = inference_runner._write_results(results, Path(tmp))  # pyright: ignore[reportPrivateUsage]
        if not path.is_file() or path.stat().st_size == 0:
            msg = f"no results parquet at {path}"
            raise SmokeError(msg)


def vllm_command() -> list[str]:
    """The serving command line exactly as the inference runner builds it on the VM."""
    from strata_forge.compute.serving import build_vllm_task
    from strata_forge.pipelines import inference_runner

    with run_config(inference_spec()):
        spec = inference_runner.load_spec()
    hp = spec.hyperparams
    task = build_vllm_task(
        spec.model_id,
        port=inference_runner._SERVE_PORT,  # pyright: ignore[reportPrivateUsage]
        host=inference_runner._SERVE_HOST,  # pyright: ignore[reportPrivateUsage]
        tensor_parallel_size=hp.tensor_parallel_size,
        max_model_len=hp.max_model_len,
        dtype=hp.dtype,
        setup="",
        python_executable=sys.executable,
    )
    return shlex.split(task.run)


#: Runs :func:`probe_vllm` in a child process: see :func:`check_vllm_command_line`.
_PROBE_FLAG = "--probe-vllm"

#: Generous for a cold import of vLLM and torch; a probe that outlives it started a server.
_PROBE_TIMEOUT_S = 900


class _Parsed(BaseException):
    """Raised in place of starting the server once vLLM has accepted the command line.

    A ``BaseException`` so no ``except Exception`` on vLLM's way to the server can swallow it.
    """


def _intercept(main: Any, *_args: Any, **_kwargs: Any) -> None:
    close = getattr(main, "close", None)
    if callable(close):
        close()  # the never-awaited server coroutine
    raise _Parsed


def check_vllm_command_line() -> None:
    """Run :func:`probe_vllm` in a child process, under a hard timeout.

    A child, because the probe rebinds ``sys.argv`` and the event-loop runners and lets vLLM set
    its own environment, none of which should outlive it; and because if a future vLLM starts its
    server some other way than the stubbed runners, the probe would serve until killed rather
    than return.
    """
    result = subprocess.run(  # noqa: S603 - this interpreter re-running this very script
        [sys.executable, str(Path(__file__).resolve()), _PROBE_FLAG],
        capture_output=True,
        text=True,
        check=False,
        timeout=_PROBE_TIMEOUT_S,
    )
    sys.stdout.write(result.stdout)  # carries the probe's annotations through to the log
    if result.returncode != 0:
        msg = f"the vLLM probe exited {result.returncode}: {result.stderr.strip()[-3000:]}"
        raise SmokeError(msg)


def probe_vllm() -> int:
    """Hand the runner's command line to the installed vLLM's own entrypoint, minus the serving.

    The module is run as ``__main__`` exactly as ``python -m`` would, so its argument parser and
    its validation both see every flag the runner emits. The event-loop runners are swapped for a
    stub that stops at the moment the server would start, which keeps this offline and GPU-free
    without naming a single vLLM internal: a removed entrypoint, a renamed flag or a rejected
    value all fail here. A deprecation the entrypoint announces is reported, not fatal.
    """
    import runpy

    argv = vllm_command()
    if argv[1:2] != ["-m"]:
        msg = f"expected `<python> -m <module> ...`, got {argv[:3]}"
        raise SmokeError(msg)
    module, args = argv[2], argv[3:]

    uvloop: Any = importlib.import_module("uvloop")
    sys.argv = [module, *args]
    uvloop.run = _intercept
    asyncio.run = _intercept  # pyright: ignore[reportAttributeAccessIssue]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        try:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        except _Parsed:
            pass
        except SystemExit as exc:
            print(
                f"vLLM rejected the command line (exit {exc.code}): {shlex.join(args)}",
                file=sys.stderr,
            )
            return 1
        else:
            print(f"{module} returned without reaching the server start", file=sys.stderr)
            return 1

    for warning in caught:
        if issubclass(warning.category, DeprecationWarning) and "vllm" in str(warning.filename):
            annotate("warning", "vLLM deprecation on the serving path", str(warning.message))
    print(f"vLLM accepted: {shlex.join(args)}")
    return 0


# --------------------------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------------------------


def annotate(level: str, title: str, message: str) -> None:
    """A GitHub Actions annotation, which lands on the run summary; plain text elsewhere."""
    flat = " ".join(message.split())
    print(f"::{level} title={title}::{flat}")


def selected_extras(arg: str) -> set[str]:
    """The extras an install string turns on, with ``all`` expanded to every extra declared."""
    extras = {part.strip() for part in arg.split(",") if part.strip()}
    if "all" in extras:
        declared = importlib.metadata.metadata("strata-forge").get_all("Provides-Extra") or []
        extras |= set(declared)
    return extras


def plan(extras: set[str]) -> list[tuple[str, Callable[[], None]]]:
    checks: list[tuple[str, Callable[[], None]]] = [
        ("installed distribution", check_install),
        ("top-level module imports", check_module_imports),
        ("cli --help", check_cli),
    ]
    if "finetuning" in extras:
        for method, fmt, adapter in finetune_cases():

            def case(method: str = method, fmt: str = fmt, adapter: str = adapter) -> None:
                check_finetune_case(method, fmt, adapter)

            checks.append((f"finetune {method} / {fmt} / {adapter}", case))
    if {"serving", "hf"} <= extras:
        checks.append(("inference spec, requests and results", check_inference_spec))
    if "serving" in extras:
        checks.append(("vllm accepts the serving command line", check_vllm_command_line))
    return checks


def report_versions() -> None:
    for name in _REPORTED:
        try:
            print(f"  {name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            continue


def main() -> int:
    if sys.argv[1:] == [_PROBE_FLAG]:
        return probe_vllm()
    parser = argparse.ArgumentParser(description="Build what a run builds, in a clean install.")
    parser.add_argument("extras", help="the extras string the venv was installed with")
    extras = selected_extras(parser.parse_args().extras)

    print(f"python {sys.version.split()[0]} at {sys.executable}")
    print(f"extras: {', '.join(sorted(extras))}")
    report_versions()

    failures = 0
    for name, check in plan(extras):
        try:
            check()
        except Exception as exc:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
            annotate("error", f"clean install: {name}", f"{type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"{failures} failed" if failures else "all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
