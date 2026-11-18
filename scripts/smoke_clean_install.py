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
- the ``hf_classify`` scoring engine on a tiny classifier built from a config and saved to disk,
  against a direct forward pass, and its refusal of a checkpoint saved without its head;
- the ``hf_label_logits`` engine on a tiny causal LM with a chat template, against a direct
  forward pass of each rendered prompt, and its refusals (a multi-token label, a kwarg the
  template never reads);
- the guided-JSON schema check and reply validator on the installed ``jsonschema``, with a remote
  reference refused and never fetched;
- the evaluation spec, a prediction file read back through the runner's own reader, a paired
  cluster bootstrap and the report's items parquet;
- an evaluation run over an ``hf_classify`` scoring run's own output: the scorer runs on a tiny
  classifier, writes its results parquet as a pushed run does, and the evaluation reads that file
  as its predictions with no reshaping (its id, prediction, overflow and logit columns as written);
- an evaluation over a ``seq_cls`` verifier's own files: a tiny verifier trained on CPU with a
  calibration split, scored with ``artifact: strata_verifier``, then evaluated with that results
  parquet as its predictions and the trainer's ``calibration.json`` as its ``calibration_ref``;
- the benchmark's timed forward pass and measurement loop on that classifier, its summary and its
  timings parquet;
- every top-level module import and the CLI's ``--help``.

Usage::

    .venv-smoke/bin/python scripts/smoke_clean_install.py finetuning,storage

The argument is the extras string the venv was installed with; it selects which checks apply.
Exits non-zero when any check fails, after running all of them.

``--write-override PATH TEXT`` validates the drill's override requirements and writes them to
``PATH``, one per line; it needs only the standard library, so it runs before the install.
"""

from __future__ import annotations

import os

# Before anything imports huggingface_hub or litellm, which read these once at import time: nothing
# here may reach the Hub or phone home, so a check cannot pass or fail on the network's say-so.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import argparse
import asyncio
import contextlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import pkgutil
import re
import shlex
import subprocess
import sys
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

#: Packages whose resolved version is worth a line in the log. Absent ones are skipped.
_REPORTED = (
    "strata-forge",
    "pydantic",
    "litellm",
    "numpy",
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
# A method that trains full weights only is checked with no adapter; the runner refuses the others.
_FULL_WEIGHT_TASKS = frozenset({"SEQ_CLS"})
# Placeholder pins: a classification run refuses an unpinned model or dataset.
_PIN = "0" * 40
_SEQ_CLS_HYPERPARAMS = {
    "labels": ["entailment", "contradiction", "neutral"],
    "two_way_positive_label": "entailment",
    "two_way_labels": ["entailed", "not_entailed"],
    "max_length": 64,
    "per_device_batch_size": 2,
}

#: The variables the inference runner sets on the served process beside the command line. vLLM
#: ignores a variable it does not know, so a rename upstream would silently undo the runner's
#: setting; the probe checks each ``VLLM_`` name is still one vLLM reads. A unit test keeps this in
#: step with the runner.
_RUNNER_SERVE_ENV = {"PYTHONUNBUFFERED": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"}

#: One requirement the override drill may force: a project name, optional extras and one or more
#: version comparisons. No URL, path, marker or option can match, so the drill can only pick
#: another release from the public index.
_OVERRIDE_TOKEN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r"(?:\[[A-Za-z0-9._,-]+\])?"
    r"(?:===|==|!=|~=|<=|>=|<|>)[A-Za-z0-9.*+!_-]+"
    r"(?:,(?:===|==|!=|~=|<=|>=|<|>)[A-Za-z0-9.*+!_-]+)*"
)

_OVERRIDE_FLAG = "--write-override"


class SmokeError(RuntimeError):
    """A check that ran to completion and found the wrong thing."""


def override_requirements(text: str) -> list[str]:
    """The drill's whitespace-separated requirements, refused unless each is a plain pin."""
    tokens = text.split()
    refused = [token for token in tokens if not _OVERRIDE_TOKEN.fullmatch(token)]
    if refused:
        msg = (
            f"override refused {[token[:80] for token in refused[:5]]}: each requirement must be "
            "a name, optional extras and a version comparison, such as transformers==4.57.6"
        )
        raise SmokeError(msg)
    return tokens


def write_override(path: str, text: str) -> int:
    """Validate the drill's override and write it as an override file, one requirement a line."""
    try:
        tokens = override_requirements(text)
    except SmokeError as exc:
        annotate("error", "Override refused", str(exc))
        return 2
    Path(path).write_text("".join(f"{token}\n" for token in tokens), encoding="utf-8")
    if tokens:
        annotate("warning", "Override drill", f"resolving with: {' '.join(tokens)}")
    return 0


# --------------------------------------------------------------------------------------------
# Inputs. Pure: no extra beyond the base install is needed to build them, so the unit suite
# exercises them too and a spec change breaks a pull request rather than the next night.
# --------------------------------------------------------------------------------------------


def sample_value(role: str, fmt: str = "") -> Any:
    """A cell of the type a dataset-format role is trained as."""
    if fmt == "nli_pair":
        nli: dict[str, Any] = {
            "premises": [
                {"pid": "p1", "text": "all birds fly"},
                {"pid": "p2", "text": "tweety is a bird"},
            ],
            "hypothesis": "tweety flies",
            "label": "entailment",
            "label_space": "3way",
            "label_dist": [1.0, 0.0, 0.0],
            "inconsistent_premises": False,
            "support_set": ["p1", "p2"],
        }
        return nli[role]
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
        for adapter in (("none",) if method.task_type in _FULL_WEIGHT_TASKS else _ADAPTERS)
    ]


def finetune_spec(method: str, fmt: str, adapter: str) -> dict[str, Any]:
    """The inert spec a control plane would deliver for one case, with every role mapped."""
    from strata_forge.training.dataset_format import pick_format
    from strata_forge.training.methods import pick_method

    roles = pick_format(fmt).roles
    extra: dict[str, Any] = {}
    if pick_method(method).task_type == "SEQ_CLS":
        extra = {
            "model_revision": _PIN,
            "dataset_commit_sha": _PIN,
            "hyperparams": dict(_SEQ_CLS_HYPERPARAMS),
        }
    return {
        **extra,
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

    row = {f"col_{role}": sample_value(role, fmt) for role in pick_format(fmt).roles}
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


_SHA = "0" * 40


def eval_spec() -> dict[str, Any]:
    """The inert evaluation spec, with a baseline, slices, clusters and an output repo."""
    return {
        "model_id": _MODEL,
        "dataset_id": _DATASET,
        "dataset_commit_sha": _SHA,
        "split": "test",
        "evaluation": {
            "labels": ["yes", "no"],
            "predictions": {"name": "candidate", "repo_id": _OUTPUT, "commit_sha": _SHA},
            "compare_to": [{"name": "baseline", "repo_id": _OUTPUT, "commit_sha": _SHA}],
            "gold_column": "label",
            "cluster_column": "base",
            "slice_columns": ["tier"],
            "seed": 1,
            "resamples": 20,
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


def check_scoring() -> None:
    """Exercise the optional numerical API without a model, dataset, or artifact store."""
    import math

    from strata_forge.metrics import brier, classification_report, temperature_scale

    report = classification_report(["yes", "no"], ["yes", None], labels=("yes", "no"))
    if report.accuracy.value != 0.5 or report.confusion.invalid_by_gold != (0, 1):
        raise SmokeError("classification accounting changed")
    score = brier(
        ["yes"], [[0.75, 0.25]], labels=("yes", "no"), probability_kind="class_probabilities"
    )
    if score.value != 0.125:
        raise SmokeError("multiclass Brier sum changed")
    output = temperature_scale([[0.0, math.log(3)]], temperature=1.0)
    if len(output) != 1 or any(
        not math.isclose(actual, expected)
        for actual, expected in zip(output[0], (0.25, 0.75), strict=True)
    ):
        raise SmokeError("temperature application changed")

    from strata_forge.metrics.calibration_fit import fit_temperature

    # Every row says "yes" at 3:1 and 60% are: the NLL optimum is T = log 3 / log 1.5.
    fit = fit_temperature([[0.0, math.log(3)]] * 10, ["yes"] * 6 + ["no"] * 4, labels=("no", "yes"))
    if not math.isclose(fit.temperature, math.log(3) / math.log(1.5), rel_tol=1e-9):
        raise SmokeError("temperature fitting changed")


def check_guided_json() -> None:
    """The guided-JSON checks on the installed ``jsonschema``: schema, replies, no fetching."""
    import urllib.request

    from strata_forge.pipelines import _guided

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"verdict": {"$ref": "#/$defs/verdict"}},
        "required": ["verdict"],
        "$defs": {"verdict": {"enum": ["yes", "no"]}},
    }
    if _guided.schema_problem(schema) is not None:
        msg = f"a valid schema was refused: {_guided.schema_problem(schema)}"
        raise SmokeError(msg)
    if _guided.schema_problem({"$ref": "https://example.com/s.json"}) is None:
        msg = "a schema referencing a remote document was accepted"
        raise SmokeError(msg)
    validator = _guided.OutputValidator(schema)
    if (
        validator.error('{"verdict": "yes"}') is not None
        or validator.error('{"verdict": 1}') is None
    ):
        msg = "the reply validator does not apply the schema"
        raise SmokeError(msg)
    opened: list[object] = []
    real = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: opened.append(a)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownLambdaType]
    try:
        remote = _guided.OutputValidator({"$ref": "https://example.com/s.json"})
        with contextlib.suppress(Exception):
            remote.error("{}")
    finally:
        urllib.request.urlopen = real
    if opened:
        msg = "validating a reply fetched a remote schema reference"
        raise SmokeError(msg)


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


class _Constructed(BaseException):
    """Raised in place of building a trainer once its arguments have bound.

    A ``BaseException`` for the same reason as :class:`_Parsed`.
    """


def trainer_class(method_name: str) -> Any:
    """The TRL trainer class the runner for ``method_name`` constructs."""
    trl_mod: Any = importlib.import_module("trl")
    from strata_forge.training.preference import (
        _TRAINER_CLASS,  # pyright: ignore[reportPrivateUsage]
        _resolve_trl_class,  # pyright: ignore[reportPrivateUsage]
    )

    if method_name == "sft":
        return trl_mod.SFTTrainer
    return _resolve_trl_class(trl_mod, method_name, _TRAINER_CLASS[method_name])


@contextlib.contextmanager
def binding_only(trainer_cls: Any) -> Generator[None]:
    """Make ``trainer_cls(...)`` bind its arguments against the real signature, then stop.

    The runner then builds everything it builds for a trainer (the TRL config included) and calls
    the installed class with its own keywords, so a renamed or removed trainer argument fails
    here, but nothing is loaded, wrapped or trained.
    """
    signature = inspect.signature(trainer_cls.__init__)
    had_own = "__init__" in vars(trainer_cls)
    original: Any = getattr(trainer_cls, "__init__", None) if had_own else None

    def bind(self: Any, *args: Any, **kwargs: Any) -> None:
        try:
            signature.bind(self, *args, **kwargs)
        except TypeError as exc:
            msg = f"{trainer_cls.__name__} no longer accepts the runner's arguments: {exc}"
            raise SmokeError(msg) from exc
        raise _Constructed

    trainer_cls.__init__ = bind
    try:
        yield
    finally:
        if had_own:
            trainer_cls.__init__ = original
        else:
            del trainer_cls.__init__


def bitsandbytes_declared(extra: str = "finetuning") -> bool:
    """Whether the installed distribution's ``extra`` installs bitsandbytes."""
    marker = re.compile(rf"""extra\s*==\s*["']{re.escape(extra)}["']""")
    for requirement in importlib.metadata.requires("strata-forge") or []:
        name = re.split(r"[\s\[<>=!~;(@]", requirement, maxsplit=1)[0]
        if name.lower().replace("_", "-") == "bitsandbytes" and marker.search(requirement):
            return True
    return False


def check_qlora_loadable() -> None:
    """QLoRA quantises the model at load time, which needs bitsandbytes on the run's machine.

    ``BitsAndBytesConfig`` constructs without it, so the config checks alone would pass on an
    install where every QLoRA run fails at ``from_pretrained``. ``[finetuning]`` installs it, so
    an extra that stops naming it, or an install that lacks it, fails the case.
    """
    if not bitsandbytes_declared():
        msg = "[finetuning] does not name bitsandbytes, so every qlora run fails at from_pretrained"
        raise SmokeError(msg)
    if importlib.util.find_spec("bitsandbytes") is None:
        msg = "[finetuning] names bitsandbytes but it is not importable in this install"
        raise SmokeError(msg)


def check_finetune_case(method_name: str, fmt: str, adapter: str) -> None:
    """One case end to end, up to the point where a real run would download the model."""
    if adapter == "qlora":
        # First, so a missing bitsandbytes is reported as the install gap it is, not as the
        # runner's spec refusal or ``to_bnb_config``'s, which both look it up as well.
        check_qlora_loadable()
    datasets_mod: Any = importlib.import_module("datasets")
    from strata_forge.training.dataset_format import build_training_rows, validate_mapping
    from strata_forge.training.peft import QLoRAConfig

    spec = finetune_spec(method_name, fmt, adapter)
    rows = finetune_rows(fmt)
    columns = list(rows[0])
    validate_mapping(fmt, spec["column_mapping"], columns)
    built = build_training_rows(rows, fmt, spec["column_mapping"])
    dataset = datasets_mod.Dataset.from_list(built)
    if set(dataset.column_names) != set(spec["column_mapping"]):
        msg = f"training rows carry {dataset.column_names}, expected the roles only"
        raise SmokeError(msg)

    with tempfile.TemporaryDirectory() as tmp:
        method, config, peft = build_finetune_config(spec, Path(tmp) / "out")
        if method.task_type == "SEQ_CLS":
            check_seq_cls_trainer(method, config, dataset)
            return
        trainer_cls = trainer_class(method.name)
        if not inspect.isclass(trainer_cls):
            msg = f"{method.name} trainer {trainer_cls!r} is not a class"
            raise SmokeError(msg)
        runner = method.build_runner(config, peft_config=peft)
        # A model and a tokenizer are passed in so the runner does not download them; the eval
        # split is passed because the spec names one, so its keyword is bound as well.
        with bf16_as_on_a_gpu(), binding_only(trainer_cls):
            try:
                runner.build_trainer(
                    train_dataset=dataset, eval_dataset=dataset, model=object(), tokenizer=object()
                )
            except _Constructed:
                pass
            else:
                msg = f"{method.name}: the runner built a trainer without calling {trainer_cls}"
                raise SmokeError(msg)
        if isinstance(peft, QLoRAConfig):
            # Passing a model above skipped the quantisation config; a real run builds it.
            peft.to_bnb_config()


def check_seq_cls_trainer(method: Any, config: Any, dataset: Any) -> None:
    """The classifier's ``TrainingArguments`` and ``Trainer`` keywords against the installed stack.

    The runner builds its arguments for real and its trainer subclass up to ``Trainer.__init__``,
    which binds against the installed signature and stops, so a renamed or removed argument fails
    here; nothing is loaded or trained.
    """
    torch: Any = importlib.import_module("torch")
    transformers_mod: Any = importlib.import_module("transformers")

    class _Tokenizer:
        pad_token_id = 0

    runner = method.build_runner(config)
    with bf16_as_on_a_gpu(), binding_only(transformers_mod.Trainer):
        try:
            runner.build_trainer(torch.nn.Linear(1, 1), dataset, dataset, _Tokenizer())
        except _Constructed:
            return
    msg = "seq_cls: the runner built a trainer without calling transformers.Trainer"
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


def _tiny_classifier(directory: Path, *, with_head: bool) -> Any:
    """Save a tiny BERT pair classifier and its tokenizer to ``directory``; return the model.

    Built from a config with a fixed seed, so nothing is downloaded. ``with_head=False`` saves
    the bare encoder, which a classification load must refuse: its head would be random.
    """
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        BertModel,
        PreTrainedTokenizerFast,
    )

    words = ["all", "cats", "are", "animals", "some", "dogs", "bark", "so"]
    specials = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    vocab = specials | {word: index + len(specials) for index, word in enumerate(words)}
    # Tokenizer markers, not credentials: passed by name so no keyword reads as a hardcoded secret.
    markers = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token=markers["unk_token"]))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, model_max_length=12, **markers)
    labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=32,
        num_labels=3,
        id2label=labels,
        label2id={name: index for index, name in labels.items()},
    )
    torch.manual_seed(0)
    model: Any = (BertForSequenceClassification if with_head else BertModel)(config).eval()
    model.save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))
    return model


def check_hf_classify() -> None:
    """The ``hf_classify`` engine on the installed transformers, against a direct forward pass.

    Covers what a fake cannot: the loading report's shape, the pair tokenization's special tokens
    and right padding, and that a checkpoint without its head is refused rather than scored.
    """
    import torch

    from strata_forge.pipelines import _scoring
    from strata_forge.pipelines._common import RunError

    identity = {name: name for name in ("entailment", "neutral", "contradiction")}
    texts = ["all cats are animals", "some dogs bark", "so so so so so so so so so"]
    pairs = ["cats are animals", "all dogs bark", "dogs bark"]
    with tempfile.TemporaryDirectory() as tmp:
        reference = _tiny_classifier(Path(tmp), with_head=True)
        classifier = _scoring.load_classifier(
            Path(tmp), label_alias=identity, max_length=None, dtype="float32"
        )
        if classifier.labels != ["entailment", "neutral", "contradiction"]:
            msg = f"labels were not read by class id: {classifier.labels}"
            raise SmokeError(msg)
        if classifier.tf32 != {"matmul": False, "cudnn": False}:
            msg = f"float32 scoring left TF32 on: {classifier.tf32}"
            raise SmokeError(msg)
        if (classifier.max_length, classifier.max_length_source) != (12, "model"):
            msg = f"the length limit is not the tokenizer's 12: {classifier.max_length}"
            raise SmokeError(msg)
        lengths = classifier.lengths(texts, pairs)
        # [CLS] a [SEP] b [SEP]: three special tokens per pair, nothing truncated.
        if lengths != [4 + 3 + 3, 3 + 3 + 3, 9 + 2 + 3]:
            msg = f"pair lengths are not the untruncated token counts: {lengths}"
            raise SmokeError(msg)
        batches, overflow = _scoring.plan_batches(
            lengths, max_length=classifier.max_length, batch_size=8
        )
        if overflow != [2] or batches != [[0, 1]]:
            msg = f"the over-long pair was not dropped: {batches}, {overflow}"
            raise SmokeError(msg)
        got = classifier.logits(texts[:2], pairs[:2])
        tokenizer: Any = classifier._tokenizer  # pyright: ignore[reportPrivateUsage]
        for index in (0, 1):
            features = tokenizer(texts[index], pairs[index], return_tensors="pt")
            with torch.inference_mode():
                expected = reference(**features).logits[0].tolist()
            if any(abs(a - b) > 1e-5 for a, b in zip(got[index], expected, strict=True)):
                msg = f"batched logits differ from one forward pass: {got[index]} vs {expected}"
                raise SmokeError(msg)
    with tempfile.TemporaryDirectory() as tmp:
        _tiny_classifier(Path(tmp), with_head=False)
        try:
            _scoring.load_classifier(
                Path(tmp), label_alias=identity, max_length=None, dtype="bfloat16"
            )
        except RunError as exc:
            if "classifier.weight" not in str(exc):
                msg = f"a headless checkpoint was refused for the wrong reason: {exc}"
                raise SmokeError(msg) from exc
        else:
            msg = "a checkpoint saved without its classification head was scored"
            raise SmokeError(msg)


#: The tiny causal LM's chat template: a user turn, optional documents, a thinking switch.
_TINY_CHAT_TEMPLATE = (
    "{%- for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}<|end|>"
    "{%- endfor %}"
    "{%- if documents %}{%- for document in documents %}<|doc|>{{ document['text'] }}<|end|>"
    "{%- endfor %}{%- endif %}"
    "{%- if add_generation_prompt %}<|assistant|>"
    "{%- if enable_thinking is defined and not enable_thinking %}<|nothink|>{%- endif %}"
    "{%- endif %}"
)


def _tiny_causal_lm(directory: Path) -> Any:
    """Save a tiny Llama causal LM, a byte-level word tokenizer and a chat template; return the model.

    Built from a config with a fixed seed, so nothing is downloaded. The vocabulary holds each
    word with and without its leading space (``Ġ``), so ``" True"`` is one token, as it is in the
    byte-level vocabularies the engine is for, and ``" Maybe so"`` is two.
    """
    import torch
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    words = ["Answer", ":", "all", "cats", "purr", "dogs", "bark", "True", "False", "Maybe", "so"]
    vocab = {"<unk>": 0}
    for word in words:
        for form in (word, "\u0120" + word):
            vocab.setdefault(form, len(vocab))
    # Tokenizer markers, not credentials: passed by name so no keyword reads as a hardcoded secret.
    markers = {"unk_token": "<unk>", "eos_token": "<|end|>"}
    backend = Tokenizer(models.WordLevel(vocab, unk_token=markers["unk_token"]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()
    specials = ["<|user|>", "<|assistant|>", "<|doc|>", "<|end|>", "<|nothink|>"]
    backend.add_special_tokens([AddedToken(token, special=True) for token in specials])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, model_max_length=64, **markers)
    tokenizer.chat_template = _TINY_CHAT_TEMPLATE
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model: Any = LlamaForCausalLM(config).eval()
    model.save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))
    return model


def check_hf_label_logits() -> None:
    """The ``hf_label_logits`` engine on the installed transformers, against a direct forward pass.

    Covers what a fake cannot: the chat template's rendering with kwargs, the prefill's tokens,
    right padding with only the last positions projected, the full-vocabulary normalizer, a tied
    output layer loading cleanly, and the refusals a real tokenizer decides.
    """
    import torch

    from strata_forge.pipelines import _label_logits, _scoring
    from strata_forge.pipelines._common import RunError
    from strata_forge.pipelines.inference_runner import LabelToken

    # Label tokens are verbalizer text, not credentials: validated from data like a spec's.
    labels = [
        LabelToken.model_validate({"label": "yes", "token": " True"}),
        LabelToken.model_validate({"label": "no", "token": " False"}),
    ]
    multi = LabelToken.model_validate({"label": "maybe", "token": " Maybe so"})
    kwargs: dict[str, Any] = {"enable_thinking": False, "documents": [{"text": "all cats purr"}]}
    texts = ["cats purr", "all dogs bark so so so so", "dogs"]
    inputs = _scoring.ScoringInputs(
        texts=texts,
        pairs=[""] * len(texts),
        item_ids=None,
        slices=None,
        chat_kwargs=[kwargs] * len(texts),
    )
    with tempfile.TemporaryDirectory() as tmp:
        reference = _tiny_causal_lm(Path(tmp))
        reader = _label_logits.load_label_reader(
            Path(tmp),
            label_tokens=labels,
            prefill="Answer:",
            chat_template_kwargs=kwargs,
            max_length=None,
            dtype="float32",
        )
        if reader.max_length != 64 or reader.unexpected_keys:
            msg = f"the load is not the tokenizer's 64 tokens with nothing ignored: {reader.max_length}"
            raise SmokeError(msg)
        lengths = reader.lengths(inputs)
        if reader.prefill_ids is None or len(reader.prefill_ids) != 2:
            msg = f"the prefill 'Answer:' is not its two tokens: {reader.prefill_ids}"
            raise SmokeError(msg)
        got = reader.score(inputs, [1, 0, 2])
        tokenizer: Any = reader._tokenizer  # pyright: ignore[reportPrivateUsage]
        for position, index in enumerate((1, 0, 2)):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": texts[index]}],
                add_generation_prompt=True,
                tokenize=False,
                **kwargs,
            )
            ids = tokenizer(prompt + "Answer:", add_special_tokens=False)["input_ids"]
            if len(ids) != lengths[index] or "<|nothink|>" not in prompt:
                msg = f"row {index} did not render with its kwargs: {prompt!r}"
                raise SmokeError(msg)
            with torch.inference_mode():
                last = reference(torch.tensor([ids])).logits[0, -1].double()
            expected = [float(last[i]) for i in reader.token_ids]
            normalizer = float(torch.logsumexp(last, dim=-1))
            row = got[position]
            if any(abs(a - b) > 1e-5 for a, b in zip(row, [*expected, normalizer], strict=True)):
                msg = f"batched label logits differ from one forward pass: {row} vs {expected}"
                raise SmokeError(msg)
            outside = 1.0 - float(torch.softmax(last, dim=-1)[reader.token_ids].sum())
            if abs(_label_logits.read_out(row[:-1], row[-1]) - outside) > 1e-6:
                msg = "mass_outside is not the full softmax's mass off the label tokens"
                raise SmokeError(msg)
        for bad, needle in (
            ({"label_tokens": [*labels[:1], multi]}, "2 tokens"),
            ({"chat_template_kwargs": {"reasoning": True}}, "never reads"),
        ):
            setup: dict[str, Any] = {
                "label_tokens": labels,
                "prefill": None,
                "chat_template_kwargs": None,
                "max_length": None,
                "dtype": "float32",
            } | bad
            try:
                _label_logits.load_label_reader(Path(tmp), **setup)
            except RunError as exc:
                if needle not in str(exc):
                    msg = f"{bad} was refused for the wrong reason: {exc}"
                    raise SmokeError(msg) from exc
            else:
                msg = f"{bad} was accepted"
                raise SmokeError(msg)


def check_eval_spec() -> None:
    """The evaluation spec, a prediction file, the bootstrap and the items file, offline."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from strata_forge.metrics.cluster_bootstrap import paired_cluster_bootstrap
    from strata_forge.pipelines import eval_runner

    with run_config(eval_spec()):
        spec = eval_runner.load_spec()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.parquet"
        pq.write_table(pa.Table.from_pydict({"item_id": ["a", "b"], "pred": ["yes", None]}), path)
        columns = eval_runner._read_parquet(path, ["item_id", "pred"])  # pyright: ignore[reportPrivateUsage]
        if columns["pred"] != ["yes", None]:
            msg = f"the prediction file read back as {columns!r}"
            raise SmokeError(msg)
        paired = paired_cluster_bootstrap(
            ["yes", "no"],
            ["yes", "yes"],
            columns["pred"],
            labels=spec.evaluation.labels,
            clusters=["c1", "c2"],
            seed=spec.evaluation.seed,
            resamples=spec.evaluation.resamples,
        )
        if paired.settings.resamples != spec.evaluation.resamples:
            raise SmokeError("the bootstrap did not run the requested resamples")
        items = eval_runner._write_items([{"item_id": "a", "slice__tier": ["x"]}], Path(tmp))  # pyright: ignore[reportPrivateUsage]
        if not items.is_file() or items.stat().st_size == 0:
            msg = f"no items parquet at {items}"
            raise SmokeError(msg)


#: The commits the integration check pins its synthetic gold split and scoring output to.
_SCORING_SHA = "1" * 40
_GOLD_SHA = "2" * 40


@contextlib.contextmanager
def _patched(module: Any, **replacements: Any) -> Any:
    """Swap module attributes for the duration of a check, restoring them after."""
    saved = {name: getattr(module, name) for name in replacements}
    for name, value in replacements.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


@contextlib.contextmanager
def _home(path: Path) -> Any:
    """Point ``Path.home()`` (where runners keep their results) at ``path``."""
    saved = os.environ.get("HOME")
    os.environ["HOME"] = str(path)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = saved


def _score_tiny_split(tmp: Path, rows: list[dict[str, Any]]) -> Path:
    """Run the ``hf_classify`` scoring runner end to end on a tiny classifier; return its parquet.

    Only the Hub is stood in for (the split's rows, its commit and the model snapshot); the
    classifier load, the length plan, the forward passes, the result rows and the parquet writer
    are the runner's own.
    """
    from strata_forge.pipelines import inference_runner
    from strata_forge.pipelines._common import RunSecrets

    snapshot = tmp / "model"
    snapshot.mkdir()
    _tiny_classifier(snapshot, with_head=True)
    spec = inference_runner.RunSpec.model_validate(
        {
            "model_id": "org/tiny-nli",
            "model_revision": _SCORING_SHA,
            "dataset_id": "org/tiny-gold",
            "dataset_commit_sha": _GOLD_SHA,
            "split": "test",
            "template": "{p}",
            "column_mapping": {"p": "premise", "h": "hypothesis"},
            "hyperparams": {
                "engine": "hf_classify",
                "pair_template": "{h}",
                "dtype": "float32",
                "label_alias": {name: name for name in ("entailment", "neutral", "contradiction")},
                "id_column": "uid",
                "slice_column": "tier",
            },
            "run_id": "scoring-run",
        }
    )

    async def commit(_spec: Any, _token: Any) -> str:
        return _GOLD_SHA

    async def fetch(_spec: Any, _token: Any) -> Path:
        return snapshot

    def load_rows(_spec: Any, _token: Any) -> list[dict[str, Any]]:
        return rows

    with (
        _home(tmp),
        _patched(
            inference_runner,
            _dataset_commit=commit,
            _load_rows=load_rows,
            _fetch_model=fetch,
        ),
    ):
        asyncio.run(inference_runner._execute_scoring(spec, RunSecrets(), None))  # pyright: ignore[reportPrivateUsage]
    return tmp / "strata-inference-results" / "scoring-run" / "results.parquet"


class _LocalHub:
    """The Hub as the evaluation runner reads it: pinned files served from disk."""

    files: ClassVar[dict[tuple[str, str, str], Path]] = {}

    def __init__(self, token: Any = None) -> None:
        del token

    async def download_file(self, repo_id: str, path: str, **kwargs: Any) -> Path:
        return _LocalHub.files[(repo_id, kwargs["revision"], path)]


def check_eval_consumes_scoring_output() -> None:
    """An evaluation reads an ``hf_classify`` run's results parquet exactly as the scorer wrote it.

    The scorer runs for real on CPU; its file is then the evaluation's ``predictions`` under the
    scorer's own column names (``item_id``, ``pred``, ``overflow``, ``logit_<label>``), with
    calibration reading the raw logits. Any reshaping the evaluation needed would fail here.
    """
    import pyarrow.parquet as pq

    from strata_forge.metrics.classification import classification_report
    from strata_forge.pipelines import eval_runner
    from strata_forge.pipelines._common import RunSecrets

    labels = ["entailment", "neutral", "contradiction"]
    premises = [
        "all cats are animals",
        "some dogs bark",
        "cats are animals",
        "dogs bark",
        "all dogs",
    ]
    hypotheses = ["cats are animals", "all dogs bark", "some cats", "so bark", "dogs are animals"]
    rows: list[dict[str, Any]] = [
        {
            "uid": f"u{i}",
            "premise": premises[i % 5],
            "hypothesis": hypotheses[(i * 2) % 5],
            "label": labels[i % 3],
            "tier": "easy" if i % 2 else "hard",
            "base": f"b{i // 2}",
        }
        for i in range(12)
    ]
    # One pair over the tokenizer's 12-token limit: the scorer drops it as overflow, unscored.
    rows.append(
        {
            "uid": "u-long",
            "premise": "so so so so so so so so so",
            "hypothesis": "dogs bark",
            "label": "neutral",
            "tier": "hard",
            "base": "b-long",
        }
    )
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        results = _score_tiny_split(tmp, rows)
        written = pq.read_table(str(results)).to_pylist()
        expected_columns = {
            "custom_id",
            "item_id",
            "slice",
            "input_tokens",
            "overflow",
            "pred",
            *(f"logit_{label}" for label in labels),
            *(f"p_{label}" for label in labels),
        }
        if set(written[0]) != expected_columns:
            msg = f"the scorer wrote {sorted(written[0])}, not {sorted(expected_columns)}"
            raise SmokeError(msg)
        if [row["overflow"] for row in written].count(True) != 1:
            raise SmokeError("the over-long pair was not written as one overflow row")

        _LocalHub.files = {("org/scores", _SCORING_SHA, "results.parquet"): results}
        gold = {key: [row[key] for row in rows] for key in ("uid", "label", "tier", "base")}

        def load_gold(_spec: Any, _token: Any) -> dict[str, list[Any]]:
            return gold

        spec = eval_runner.EvalSpec.model_validate(
            {
                "model_id": "org/tiny-nli",
                "dataset_id": "org/tiny-gold",
                "dataset_commit_sha": _GOLD_SHA,
                "split": "test",
                "run_id": "eval-run",
                "evaluation": {
                    "labels": labels,
                    # Exactly the scorer's columns: nothing renamed, nothing reshaped.
                    "predictions": {
                        "name": "scorer",
                        "repo_id": "org/scores",
                        "commit_sha": _SCORING_SHA,
                        "path": "results.parquet",
                        "id_column": "item_id",
                        "prediction_column": "pred",
                        "overflow_column": "overflow",
                    },
                    "id_column": "uid",
                    "gold_column": "label",
                    "cluster_column": "base",
                    "slice_columns": ["tier"],
                    "seed": 20261104,
                    "resamples": 200,
                    "min_slice_rows": 1,
                    "calibration": {"logit_prefix": "logit_"},
                },
            }
        )
        with (
            _home(tmp),
            _patched(eval_runner, HFHubClient=_LocalHub, _load_gold=load_gold),
        ):
            asyncio.run(eval_runner._execute(spec, RunSecrets(), None))  # pyright: ignore[reportPrivateUsage]
        report = json.loads((tmp / "strata-eval-results" / "eval-run" / "metrics.json").read_text())

    counts = report["systems"][0]["counts"]
    if counts != {"scored": 13, "missing": 0, "invalid": 0, "overflow": 1, "unmatched": 0}:
        msg = f"the evaluation joined the scorer's file as {counts}"
        raise SmokeError(msg)
    overall = report["slices"][0]
    if (overall["n"], overall["overflow_excluded"]) != (12, 1):
        msg = f"the overflow row did not leave the shared population: {overall['n']}"
        raise SmokeError(msg)
    by_id = {row["item_id"]: row["pred"] for row in written}
    scored = [row for row in rows if row["uid"] != "u-long"]
    point = classification_report(
        [row["label"] for row in scored], [by_id[row["uid"]] for row in scored], labels=labels
    )
    accuracy = overall["systems"]["scorer"]["accuracy"]["point"]
    if abs(accuracy - (point.accuracy.value or 0.0)) > 1e-12:
        msg = f"accuracy {accuracy} is not the scorer's predictions scored directly"
        raise SmokeError(msg)
    raw = overall["systems"]["scorer"]["raw"]
    if raw is None or raw["n"] != 12 or raw["coverage"] != 1.0:
        msg = f"the raw logits did not feed every shared row's probabilities: {raw}"
        raise SmokeError(msg)


#: The commits the verifier check pins its synthetic model, gold split and scores to.
_VERIFIER_SHA = "a" * 40
_VERIFIER_SCORES_SHA = "c" * 40


@contextlib.contextmanager
def _repo_tests() -> Any:
    """Make this checkout's ``tests`` package importable, for its synthetic checkpoint builders.

    Only the repository root is added, which holds no ``strata_forge`` (the package is under
    ``src/``), so the installed distribution is still the one under test.
    """
    root = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, root)
    try:
        yield
    finally:
        sys.path.remove(root)


class _MemoryProbe:
    """A device-memory probe for a CPU run: fixed numbers, so the memory check reports and passes."""

    def reset(self) -> None:
        return None

    def peak_bytes(self) -> int | None:
        return 10

    def total_bytes(self) -> int | None:
        return 1000


def _train_tiny_verifier(root: Path, tiny: Any) -> Path:
    """Train a tiny ``seq_cls`` encoder verifier with a calibration split; return its artifact."""
    import datasets

    from strata_forge.training.seq_cls import SeqClsConfig, SeqClsRunner

    config = SeqClsConfig.model_validate(
        {
            "model_id": "synthetic/tiny",
            "output_dir": str(root / "checkpoints"),
            "labels": tiny.LABELS,
            **tiny.TWO_WAY,
            "max_length": 64,
            "per_device_batch_size": 8,
            "max_steps": 2,
        }
    )
    artifact = root / "artifact"
    SeqClsRunner(config, memory_probe=_MemoryProbe()).run(
        checkpoint=tiny.encoder(root / "base"),
        artifact_dir=artifact,
        train_dataset=datasets.Dataset.from_list(tiny.rows(16)),
        calibration_dataset=datasets.Dataset.from_list(tiny.rows(8, seed=2)),
    )
    return artifact


def _score_with_verifier(tmp: Path, artifact: Path, rows: list[dict[str, Any]]) -> Path:
    """Score ``rows`` with ``artifact: strata_verifier`` through the runner; return its parquet."""
    from strata_forge.pipelines import inference_runner
    from strata_forge.pipelines._common import RunSecrets

    spec = inference_runner.RunSpec.model_validate(
        {
            "model_id": "synthetic/verifier",
            "model_revision": _VERIFIER_SHA,
            "dataset_id": "synthetic/gold",
            "dataset_commit_sha": _GOLD_SHA,
            "split": "test",
            "template": "{premises}",
            "column_mapping": {"premises": "premises", "hypothesis": "hypothesis"},
            "hyperparams": {
                "engine": "hf_classify",
                "artifact": "strata_verifier",
                "pair_template": "{hypothesis}",
                "dtype": "float32",
                "id_column": "uid",
            },
            "run_id": "verifier-scores",
        }
    )

    async def commit(_spec: Any, _token: Any) -> str:
        return _GOLD_SHA

    async def fetch(_spec: Any, _token: Any) -> Path:
        return artifact

    def load_rows(_spec: Any, _token: Any) -> list[dict[str, Any]]:
        return rows

    with (
        _home(tmp),
        _patched(
            inference_runner, _dataset_commit=commit, _load_rows=load_rows, _fetch_model=fetch
        ),
    ):
        asyncio.run(inference_runner._execute_scoring(spec, RunSecrets(), None))  # pyright: ignore[reportPrivateUsage]
    return tmp / "strata-inference-results" / "verifier-scores" / "results.parquet"


def _evaluate_verifier(
    tmp: Path, results: Path, calibration: Path, artifact: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate the verifier's results parquet with its calibration file; return metrics.json."""
    from strata_forge.pipelines import eval_runner
    from strata_forge.pipelines._common import RunSecrets

    labels = ["entailment", "contradiction", "neutral"]
    _LocalHub.files = {
        ("synthetic/scores", _VERIFIER_SCORES_SHA, "results.parquet"): results,
        ("synthetic/verifier", _VERIFIER_SHA, "calibration.json"): calibration,
        ("synthetic/verifier", _VERIFIER_SHA, "strata_verifier.json"): artifact
        / "strata_verifier.json",
    }
    gold = {key: [row[key] for row in rows] for key in ("uid", "label", "base", "premises")}

    def load_gold(_spec: Any, _token: Any) -> dict[str, list[Any]]:
        return gold

    spec = eval_runner.EvalSpec.model_validate(
        {
            "model_id": "synthetic/verifier",
            "dataset_id": "synthetic/gold",
            "dataset_commit_sha": _GOLD_SHA,
            "split": "test",
            "run_id": "verifier-eval",
            "evaluation": {
                "labels": labels,
                "predictions": {
                    "name": "verifier",
                    "repo_id": "synthetic/scores",
                    "commit_sha": _VERIFIER_SCORES_SHA,
                    "id_column": "item_id",
                    "prediction_column": "pred",
                    "overflow_column": "overflow",
                    # The trainer's file as written; without its own labels the evaluation
                    # takes them from the strata_verifier.json beside it at the same commit.
                    "calibration_ref": {
                        "repo_id": "synthetic/verifier",
                        "commit_sha": _VERIFIER_SHA,
                        "path": "calibration.json",
                    },
                },
                "id_column": "uid",
                "gold_column": "label",
                "cluster_column": "base",
                "premises_column": "premises",
                "seed": 20261104,
                "resamples": 200,
                "min_slice_rows": 1,
                "calibration": {"apply_calibration": "auto"},
            },
        }
    )
    with _home(tmp), _patched(eval_runner, HFHubClient=_LocalHub, _load_gold=load_gold):
        asyncio.run(eval_runner._execute(spec, RunSecrets(), None))  # pyright: ignore[reportPrivateUsage]
    return json.loads((tmp / "strata-eval-results" / "verifier-eval" / "metrics.json").read_text())


def check_eval_consumes_verifier() -> None:
    """An evaluation reads a trained verifier's scores and calibration file exactly as written.

    The trainer writes ``calibration.json`` (``strata.calibration/v1``) with its labels; the
    check also evaluates a copy without them, which takes them from ``strata_verifier.json``.
    Either way the applied read must be the trainer's temperature on the scorer's raw logits.
    """
    import pyarrow.parquet as pq

    from strata_forge.metrics.calibration import temperature_scale
    from strata_forge.metrics.classification import brier

    with _repo_tests():
        from tests.unit.training import tiny_classifier as tiny
    labels = list(tiny.LABELS)
    with tempfile.TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        artifact = _train_tiny_verifier(root / "train", tiny)
        trained = json.loads((artifact / "calibration.json").read_text())
        if trained.get("schema") != "strata.calibration/v1" or trained.get("labels") != labels:
            msg = f"the trainer wrote a calibration file the evaluation does not read: {trained}"
            raise SmokeError(msg)
        temperature = trained["fit"]["temperature"]
        # The evaluation's label space is the three-way one; two-way rows are left out.
        three_way = [row for row in tiny.rows(24, seed=7) if row["label_space"] == "3way"]
        rows = [{**row, "uid": f"u{i}", "base": f"b{i // 2}"} for i, row in enumerate(three_way)]
        results = _score_with_verifier(root / "score", artifact, rows)
        written = {row["item_id"]: row for row in pq.read_table(str(results)).to_pylist()}
        logits = [[written[row["uid"]][f"logit_{label}"] for label in labels] for row in rows]
        expected = brier(
            [row["label"] for row in rows],
            temperature_scale(logits, temperature=temperature),
            labels=labels,
            probability_kind="class_probabilities",
        ).value
        unlabelled = root / "calibration-without-labels.json"
        unlabelled.write_text(json.dumps({k: v for k, v in trained.items() if k != "labels"}))
        for calibration, labels_from in (
            (artifact / "calibration.json", "calibration.json"),
            (unlabelled, "strata_verifier.json"),
        ):
            where = root / f"eval-{labels_from}"
            report = _evaluate_verifier(where, results, calibration, artifact, rows)
            counts = report["systems"][0]["counts"]
            if (counts["scored"], counts["missing"]) != (len(rows), 0):
                msg = f"the evaluation joined the verifier's scores as {counts}"
                raise SmokeError(msg)
            own = report["calibration"]["systems"]["verifier"]["own"]
            if (own["temperature"], own["labels_from"]) != (temperature, labels_from):
                msg = f"the calibration file was not read as written: {own}"
                raise SmokeError(msg)
            applied = report["slices"][0]["systems"]["verifier"]["applied"]
            if applied["n"] != len(rows) or abs(applied["brier"]["point"] - expected) > 1e-12:
                msg = f"the applied read is not the trainer's temperature on raw logits: {applied}"
                raise SmokeError(msg)

def check_benchmark() -> None:
    """The benchmark's ``hf_classify`` timing on a tiny CPU classifier, through its real loop.

    Covers what the fakes cannot: the timed request runs on the installed torch and
    transformers with ONE synchronization, its forward_ms sits inside its total_ms, warmups are
    excluded, the summary validates from its JSON text, and the timings parquet is written.
    """
    import asyncio
    import time

    from strata_forge.pipelines import _benchmark as bench
    from strata_forge.pipelines import _scoring, benchmark_runner

    identity = {name: name for name in ("entailment", "neutral", "contradiction")}
    texts = ["all cats are animals", "some dogs bark", "cats are animals", "dogs bark"]
    pairs = ["cats are animals", "all dogs bark", "all cats", "some dogs"]
    with tempfile.TemporaryDirectory() as tmp:
        _tiny_classifier(Path(tmp), with_head=True)
        classifier = _scoring.load_classifier(
            Path(tmp), label_alias=identity, max_length=None, dtype="float32"
        )
        syncs = 0
        real_sync = classifier.synchronize

        def _counted() -> None:
            nonlocal syncs
            syncs += 1
            real_sync()

        classifier.synchronize = _counted  # type: ignore[method-assign]
        start, end, forward = classifier.timed_forward(
            texts[:2], pairs[:2], clock=time.perf_counter_ns
        )
        if not 0 < forward * 1_000_000 <= end - start:
            msg = f"forward_ms is not inside the request: {forward} ms of {end - start} ns"
            raise SmokeError(msg)
        if syncs != 1:
            msg = f"a timed request synchronized {syncs} times; the protocol allows one"
            raise SmokeError(msg)
        step = benchmark_runner._reference_step(  # pyright: ignore[reportPrivateUsage]
            classifier, texts, pairs
        )
        outcome = asyncio.run(
            bench.measure_configuration(
                bench.Configuration(index=0, batch_size=2, concurrency=1),
                sample=[0, 1, 2, 3],
                passes=2,
                warmup=1,
                order_seed=7,
                warmup_seed=1007,
                step=step,
                clock_ns=time.perf_counter_ns,
                deadline_ns=time.perf_counter_ns() + 60 * 10**9,
            )
        )
        if outcome.status != "measured" or len(outcome.iterations) != 4:
            msg = (
                f"the loop did not measure 2 passes of 2 batches: {outcome.status}, {outcome.error}"
            )
            raise SmokeError(msg)
        lengths = classifier.lengths(texts, pairs)
        tokens = [length or 0 for length in lengths]
        result = bench.summarize(
            outcome,
            boundaries=["model_only", "end_to_end"],
            tokens=tokens,
            length_of=lambda _row: None,
            items_of=lambda _row: None,
            bootstrap=(7, 20) if importlib.util.find_spec("numpy") else None,
        )
        summary = bench.BenchmarkSummary(
            engine="hf_classify",
            boundaries=("model_only", "end_to_end"),
            complete=True,
            headline_readable=False,
            model_path="reference_automodel",
            headline_unreadable_reason="a CPU smoke check is not a T13 read",
            gpu_idle_check="unavailable",
            sample={"rows": 4, "eligible": 4, "sampled": 4},
            excluded={},
            cold_start={"load_seconds": None, "first_iteration_ms": outcome.first_iteration_ms},
            gpu={"name": None},
            configurations=(result,),
        )
        parsed = bench.BenchmarkSummary.model_validate_json(summary.model_dump_json())
        model_p50 = parsed.configurations[0].boundaries[0].latency.pooled["p50"]
        e2e_p50 = parsed.configurations[0].boundaries[1].latency.pooled["p50"]
        if model_p50 is None or e2e_p50 is None or model_p50 > e2e_p50:
            msg = f"model-only p50 {model_p50} is not within end-to-end p50 {e2e_p50}"
            raise SmokeError(msg)
        rows = bench.iteration_rows(
            [outcome],
            boundaries=["model_only", "end_to_end"],
            tokens=tokens,
            items=None,
            length_of=lambda _row: None,
            items_of=lambda _row: None,
        )
        path = benchmark_runner._write_timings(rows, Path(tmp) / "out")  # pyright: ignore[reportPrivateUsage]
        if not path.is_file() or path.stat().st_size == 0:
            msg = f"no timings parquet at {path}"
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

    ``VLLM_TARGET_DEVICE=cpu`` because building the parser already resolves a device: vLLM's
    defaults include a device config that refuses to construct when no platform is detected, and
    a CUDA wheel on a machine with no GPU detects none. That variable is vLLM's own switch for
    running an accelerator wheel on a CPU-only host; should it stop being honoured, the probe
    fails with "Failed to infer device type". The child also carries the variables the runner
    sets on the served process, so the entrypoint sees what it sees on a run's machine.
    """
    result = subprocess.run(  # noqa: S603 - this interpreter re-running this very script
        [sys.executable, str(Path(__file__).resolve()), _PROBE_FLAG],
        capture_output=True,
        text=True,
        check=False,
        timeout=_PROBE_TIMEOUT_S,
        env={**os.environ, **_RUNNER_SERVE_ENV, "VLLM_TARGET_DEVICE": "cpu"},
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
    without naming a vLLM internal on the parsing path: a removed entrypoint, a renamed flag or a
    rejected value all fail here. A deprecation the entrypoint announces is reported, not fatal.
    Each ``VLLM_`` variable the runner sets must still be one vLLM reads.
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

    # The one vLLM internal named here: the table its environment lookups are served from.
    envs: Any = importlib.import_module("vllm.envs")
    known = getattr(envs, "environment_variables", None)
    if not isinstance(known, dict):
        print("vllm.envs no longer exposes environment_variables", file=sys.stderr)
        return 1
    unread = sorted(n for n in _RUNNER_SERVE_ENV if n.startswith("VLLM_") and n not in known)
    if unread:
        print(f"vLLM no longer reads {unread}, which the runner sets", file=sys.stderr)
        return 1
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
        ("guided json validates replies offline", check_guided_json),
    ]
    if "scoring" in extras:
        checks.append(("classification point metrics and temperature", check_scoring))
    if "finetuning" in extras:
        for method, fmt, adapter in finetune_cases():

            def case(method: str = method, fmt: str = fmt, adapter: str = adapter) -> None:
                check_finetune_case(method, fmt, adapter)

            checks.append((f"finetune {method} / {fmt} / {adapter}", case))
    if {"scoring", "storage", "hf"} <= extras:
        checks.append(("evaluation spec, predictions, bootstrap and items", check_eval_spec))
    if {"serving", "scoring", "storage", "hf"} <= extras:
        checks.append(
            (
                "evaluation reads an hf_classify scoring output as written",
                check_eval_consumes_scoring_output,
            )
        )
    if {"finetuning", "serving", "scoring", "storage", "hf"} <= extras:
        checks.append(
            (
                "evaluation reads a trained verifier's scores and calibration",
                check_eval_consumes_verifier,
            )
        )
    if {"serving", "hf"} <= extras:
        checks.append(("inference spec, requests and results", check_inference_spec))
        checks.append(("hf_classify scores a saved classifier exactly", check_hf_classify))
        checks.append(("hf_label_logits reads a saved causal LM exactly", check_hf_label_logits))
        checks.append(("benchmark times a saved classifier on its real loop", check_benchmark))
    if "serving" in extras:
        checks.append(("vllm accepts the serving command line", check_vllm_command_line))
    return checks


def report_versions() -> None:
    for name in _REPORTED:
        try:
            print(f"  {name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            continue


def run_checks(checks: list[tuple[str, Callable[[], None]]]) -> int:
    """Run every check, report each, and return the exit status: 0 only if all of them passed.

    A ``SystemExit`` from a dependency inside a check counts as that check failing; otherwise a
    stray ``sys.exit(0)`` would end the run green with the remaining checks never run.
    """
    failures = 0
    executed = 0
    for name, check in checks:
        executed += 1
        try:
            check()
        except (Exception, SystemExit) as exc:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
            annotate("error", f"clean install: {name}", f"{type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    if not checks or executed != len(checks):
        annotate("error", "clean install", f"ran {executed} of {len(checks)} checks")
        return 1
    print(f"{failures} of {executed} failed" if failures else f"all {executed} checks passed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == [_PROBE_FLAG]:
        return probe_vllm()
    if args[:1] == [_OVERRIDE_FLAG]:
        if len(args) != 3:
            print(f"usage: {_OVERRIDE_FLAG} PATH TEXT", file=sys.stderr)
            return 2
        return write_override(args[1], args[2])
    parser = argparse.ArgumentParser(description="Build what a run builds, in a clean install.")
    parser.add_argument("extras", help="the extras string the venv was installed with")
    extras = selected_extras(parser.parse_args(args).extras)

    print(f"python {sys.version.split()[0]} at {sys.executable}")
    print(f"extras: {', '.join(sorted(extras))}")
    report_versions()
    return run_checks(plan(extras))


if __name__ == "__main__":
    sys.exit(main())
