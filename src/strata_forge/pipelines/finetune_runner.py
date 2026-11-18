"""VM-side fine-tuning runner: STRATA_RUN_CONFIG (inert JSON) -> TRL -> an HF model repo.

Launched on the user's own GPU box by the control plane (``python -m
strata_forge.pipelines.finetune_runner``). It reads an INERT spec from the ``STRATA_RUN_CONFIG``
env var, loads a Hugging Face dataset split, projects its columns onto the row shape the chosen
method's trainer expects, trains with :mod:`strata_forge.training` (SFT or a preference method,
optionally through a LoRA / QLoRA adapter), and then EITHER pushes the trained artifact to the
user's HF model repo (when a write token + an output repo are supplied) OR leaves it on the VM
for the user to retrieve over SSH. Live progress is appended to the file named by
``FORGE_PROGRESS_PATH`` (the orchestrator tails it).

The counterpart of :mod:`strata_forge.pipelines.inference_runner`, and deliberately its mirror:
the same inert-spec contract, the same phase/step progress protocol, the same security boundary,
and the same shared plumbing from :mod:`strata_forge.pipelines._common` — so an operator watching
a training run reads the same kind of report as one watching a batch.

Security boundary (the VM is where allow-listed config meets real credentials + the network):

  - The config is parsed as DATA only: ``json`` + Pydantic ``extra="forbid"`` — never
    ``eval``/``pickle``/``yaml.unsafe_load``. ``column_mapping``/``hyperparams``/``lora`` are
    inert values handed to typed configs, never executed. Nothing in the spec can name Python
    to run, which is why GRPO — whose reward is a callable — is not reachable from here.
  - The Hugging Face token arrives in a private secrets file the backend wrote beside the job
    (named by ``FORGE_SECRETS_FILE``, read and deleted before anything else runs — see
    :func:`~strata_forge.pipelines._common.load_secrets`), never in ``STRATA_RUN_CONFIG`` or
    the environment. It is passed EXPLICITLY to every Hub read and write — the split, the base
    model and its tokenizer, the merge, the push — and a run that received none sends no
    credential rather than the VM's ambient ``HF_TOKEN`` or cached login (a repo the VM's Hub
    cache already holds still loads from that cache). It is scrubbed from every surfaced message.
  - Every model load refuses remote code and pickle checkpoints (``trust_remote_code=False``,
    ``use_safetensors=True`` — see :mod:`strata_forge.training.loading`): the base model is
    named by the spec, and its repo is not the operator's. The spec's ``extra_trainer_args``
    cannot reopen any of this: the TRL knobs that load, push or report under other terms are
    refused by name (:data:`_SPEC_REFUSED_TRAINER_ARGS`).
  - Progress events carry step counts, float metrics and a repo id — never a training example.
    A fine-tuning corpus is often the most sensitive thing in a run, and none of it is in the
    channel the control plane relays to a browser.

What a training run reports differs from a batch in one way that matters: the trainer emits its
own ``step``/``eval``/``checkpoint`` events through
:func:`strata_forge.training.progress.attach`, so once training starts there is nothing for this
module to add. Its job is the stretches AROUND the loop — downloading a split, loading a model
onto the GPU, merging and uploading — which have nothing to count and would otherwise be an
indeterminate wait. Those get an elapsed-stamping phase caption; the loop itself deliberately gets
none, which is why building the trainer and running it are two separate steps here rather than the
one call :meth:`SFTRunner.train` offers.
"""

from __future__ import annotations

import asyncio
import itertools
import sys
import time
from collections.abc import Mapping  # runtime: isinstance below
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.pipelines._common import (
    MANIFEST_SCHEMA,
    RunError,
    RunSecrets,
    emit,
    ensure_private_repo,
    load_config,
    manifest_engine,
    phase_sink,
    require_commit_sha,
    results_dir,
    runner_main,
    snapshot_commit,
    ticking_phase,
    validate_repo_id,
    write_manifest,
)
from strata_forge.pipelines.inference_runner import SNAPSHOT_IGNORED, SNAPSHOT_PATTERNS
from strata_forge.storage import HFHubClient
from strata_forge.training.dataset_format import (
    DatasetFormatError,
    build_training_rows,
    sft_text_field,
    validate_mapping,
)
from strata_forge.training.hardware import GpuSampler
from strata_forge.training.loading import model_load_kwargs, tokenizer_load_kwargs
from strata_forge.training.methods import (
    UnsupportedMethodError,
    check_format,
    pick_method,
)
from strata_forge.training.peft import (
    LoRAConfig,
    MissingBitsAndBytesError,
    QLoRAConfig,
    require_bitsandbytes,
)
from strata_forge.training.progress import ProgressEvent, coerce_int, numeric_metrics
from strata_forge.training.seq_cls import PreflightError, SeqClsConfig, SeqClsRunner

if TYPE_CHECKING:
    from strata_forge.training.methods import MethodSpec
    from strata_forge.training.progress import JsonlProgressWriter

__all__ = ["FinetuneSpec", "check_spec", "load_spec", "main"]

# Where the trained artifact is written on the VM. OUTSIDE the per-run workdir the orchestrator
# deletes on every terminal state, so a failed upload does not take the weights with it.
_OUTPUT_DIR_NAME = "strata-finetune-output"
# The subdirectory of that run directory the trainer saves into, so the run dir can later hold
# siblings (a merged copy, a metrics dump) without them being uploaded as part of the adapter.
_ARTIFACT_SUBDIR = "artifact"
# Intermediate trainer checkpoints, a SIBLING of the artifact so they are never part of a push.
_CHECKPOINT_SUBDIR = "checkpoints"
# Never uploaded even when they sit in the artifact directory: TRL writes intermediate checkpoints
# there for the causal-LM methods.
_PUSH_IGNORED = ("checkpoint-*",)
_MANIFEST_KIND = "finetune"
# What the manifest records the installed versions of: everything that shapes a training run.
_RECORDED_DISTRIBUTIONS = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "tokenizers",
    "safetensors",
)


class LoraSpec(BaseModel):
    """Inert adapter knobs. Mirrors :class:`strata_forge.training.peft.LoRAConfig`'s surface."""

    model_config = ConfigDict(extra="forbid")

    r: int = Field(default=16, ge=1, le=512)
    alpha: int = Field(default=32, ge=1, le=1024)
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    # None -> peft's per-architecture defaults, which is the right answer for almost every model.
    target_modules: tuple[str, ...] | None = None


class FinetuneSpec(BaseModel):
    """The inert fine-tuning spec the server delivers in ``STRATA_RUN_CONFIG`` (no secrets)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # The engine the control plane validated this spec against: ``"<version>"`` or
    # ``"<version>+<commit>"``. ``load_config`` refuses any other installed engine (``None`` makes
    # no claim). The control plane sets it, never a user: the same string decides what the setup
    # step installs, so a spec cannot pick an engine the control plane did not validate against.
    engine_version: str | None = None
    method: str
    model_id: str
    # The base model's full Hub commit id. Set, it is the only revision the run reads (model,
    # tokenizer, merge); unset, the run reads the default branch and records the commit it got.
    # ``seq_cls`` requires it, and the dataset pin beside it.
    model_revision: str | None = None
    dataset_id: str
    dataset_commit_sha: str | None = None
    dataset_config: str | None = None
    split: str
    # An optional held-out split. Its rows go through the same format projection, so an eval loss
    # appears in the same progress stream as the training loss.
    eval_split: str | None = None
    dataset_format: str
    column_mapping: dict[str, str] = Field(default_factory=dict)
    adapter: Literal["none", "lora", "qlora"] = "lora"
    lora: LoraSpec = Field(default_factory=LoraSpec)
    # Method + trainer knobs, validated by the method's own Pydantic config (extra="forbid"), so
    # a knob that does not apply to the chosen method is a named error rather than a silent no-op.
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    # Bounds the rows materialised from the split. Not a trainer knob — it is the OOM guard, so it
    # lives beside the spec rather than inside the passthrough hyperparams.
    row_limit: int | None = Field(default=None, ge=1)
    # Fold the adapter into the base weights before pushing. Off by default: an adapter is
    # megabytes and a merged model is tens of gigabytes, and the merge is only useful to someone
    # serving the result without peft.
    merge_adapter: bool = False
    # Where to push. The server populates it ONLY when pushing (a write token is present); when
    # absent the runner keeps the artifact on the VM. The VM never guesses ownership.
    output_repo_id: str | None = None
    progress_path: str | None = None
    # The control plane's run id — names the on-VM output dir.
    run_id: str | None = None


def load_spec(*, writer: JsonlProgressWriter | None = None) -> FinetuneSpec:
    """Parse + validate the spec, including the pairings only this module can check.

    Every check here is one that would otherwise fail on the GPU: an unknown method, a dataset
    shape the method cannot train on, a merge asked for with no adapter to merge, a QLoRA run on
    a machine that cannot quantise. They cost milliseconds at launch and minutes-to-hours
    anywhere later.
    """
    spec = load_config(FinetuneSpec, writer=writer)
    check_spec(spec)
    if spec.adapter == "qlora":
        # Before the dataset download, not at the model load where transformers would find it.
        try:
            require_bitsandbytes()
        except MissingBitsAndBytesError as exc:
            raise RunError(str(exc)) from exc
    return spec


def check_spec(spec: FinetuneSpec) -> None:
    """Every check on a spec that needs neither the machine nor the network.

    A control plane runs this at submit, against the engine it bundles, so a spec the runner would
    refuse is refused before a machine is provisioned; the runner runs it again on the machine.
    The method's own config validates ``hyperparams`` here as well, with placeholder runner-owned
    fields, so a bad knob is named at submit rather than after the dataset download.
    """
    validate_repo_id(spec.model_id, "model")
    validate_repo_id(spec.dataset_id, "dataset")
    if spec.output_repo_id is not None:
        validate_repo_id(spec.output_repo_id, "output repo")
    require_commit_sha(spec.model_revision, "model_revision")
    require_commit_sha(spec.dataset_commit_sha, "dataset_commit_sha")
    try:
        method = pick_method(spec.method)
        check_format(method, spec.dataset_format)
    except UnsupportedMethodError as exc:
        raise RunError(str(exc)) from exc
    if spec.merge_adapter and spec.adapter == "none":
        msg = "merge_adapter has nothing to merge: the run trains full weights (adapter='none')"
        raise RunError(msg)
    if method.task_type == "SEQ_CLS":
        unpinned = [
            name
            for name, value in (
                ("model_revision", spec.model_revision),
                ("dataset_commit_sha", spec.dataset_commit_sha),
            )
            if value is None
        ]
        if unpinned:
            msg = f"seq_cls needs pinned inputs; set {' and '.join(unpinned)} to a full commit id"
            raise RunError(msg)
        if spec.adapter != "none":
            msg = "seq_cls trains full weights; set adapter to 'none'"
            raise RunError(msg)
    _trainer_config(spec, method, Path("/nonexistent/strata-check"))


def _peft_config(spec: FinetuneSpec, method: MethodSpec) -> LoRAConfig | QLoRAConfig | None:
    """Build the adapter config, or ``None`` for a full fine-tune."""
    if spec.adapter == "none":
        return None
    lora = LoRAConfig(
        r=spec.lora.r,
        alpha=spec.lora.alpha,
        dropout=spec.lora.dropout,
        target_modules=spec.lora.target_modules,
        task_type=method.task_type,  # pyright: ignore[reportArgumentType] - registry's own literal
    )
    return lora if spec.adapter == "lora" else QLoRAConfig(lora=lora)


#: Config fields this module DERIVES rather than accepts. ``model_id`` is the value ``load_spec``
#: ran ``validate_repo_id`` over and the value the run is recorded as; ``output_dir`` is the
#: artifact directory the push, merge and cleanup paths all address; ``progress_jsonl`` is the
#: orchestrator's channel. ``extra="forbid"`` cannot defend them because they are DECLARED fields
#: — it rejects unknown keys, not known ones — so naming them has to be refused here.
_RUNNER_OWNED = ("model_id", "output_dir", "progress_jsonl")


def _reject_runner_owned(hyperparams: dict[str, Any]) -> None:
    """Refuse a hyperparam that would retarget a field the runner derives.

    Named rather than silently dropped, matching how every other inapplicable knob is handled: a
    caller who asked for something the run will not do should be told, not quietly overruled.
    """
    clashing = [key for key in _RUNNER_OWNED if key in hyperparams]
    extra = hyperparams.get("extra_trainer_args")
    if isinstance(extra, Mapping):
        clashing += [f"extra_trainer_args.{key}" for key in _RUNNER_OWNED if key in extra]
    if clashing:
        msg = f"hyperparams may not set {', '.join(clashing)}: the runner derives these"
        raise RunError(msg)


#: TRL / ``transformers`` arguments an INERT spec may not set through ``extra_trainer_args``. That
#: passthrough is the libraries' whole argument surface, and these knobs act outside the terms this
#: runner sets on every load and push: they read from or write to the Hub with whatever credential
#: the VM holds rather than the one the run was delivered (``chat_template_path``, ``push_to_hub``,
#: the ``hub_*`` and ``trackio_*`` settings), run or configure a load of a repo's own code
#: (``trust_remote_code``, ``model_init_kwargs``), read a file on the VM, which for a checkpoint
#: means unpickling it (``resume_from_checkpoint``, ``deepspeed``, ``fsdp_config``,
#: ``accelerator_config``), write outside the artifact directory (``logging_dir``), or send the
#: run to a third-party service (``report_to``). Exact names first, then name prefixes.
_SPEC_REFUSED_TRAINER_ARGS = frozenset(
    {
        "accelerator_config",
        "chat_template_path",
        "deepspeed",
        "fsdp_config",
        "logging_dir",
        "model_init_kwargs",
        "ref_model_init_kwargs",
        "report_to",
        "resume_from_checkpoint",
        "trust_remote_code",
    }
)
_SPEC_REFUSED_TRAINER_ARG_PREFIXES = ("hub_", "push_to_hub", "trackio_")


def _reject_unsafe_trainer_args(hyperparams: dict[str, Any]) -> None:
    """Refuse an ``extra_trainer_args`` key that reaches past the run's own credential and files.

    The escape hatch stays whole for a caller driving :mod:`strata_forge.training` from Python,
    who is the operator of the machine. A spec is submitted by someone who is not, so from a spec
    these keys are named and refused, like every other inapplicable knob, rather than dropped.
    """
    extra = hyperparams.get("extra_trainer_args")
    if not isinstance(extra, Mapping):
        return
    keys = [key for key in cast("Mapping[object, object]", extra) if isinstance(key, str)]
    refused = sorted(
        f"extra_trainer_args.{key}"
        for key in keys
        if key in _SPEC_REFUSED_TRAINER_ARGS or key.startswith(_SPEC_REFUSED_TRAINER_ARG_PREFIXES)
    )
    if refused:
        msg = (
            f"hyperparams may not set {', '.join(refused)}: from a run spec these reach a "
            "credential, remote code, a file on the machine or a third-party service that the "
            "run does not govern"
        )
        raise RunError(msg)


def _trainer_config(spec: FinetuneSpec, method: MethodSpec, output_dir: Path) -> Any:
    """Build the method's typed config from the inert spec.

    ``hyperparams`` is splatted in rather than copied field by field: the config's own
    ``extra="forbid"`` is the authority on what each method accepts, and re-stating that list here
    would be a second definition to keep in sync. A knob that does not apply therefore surfaces as
    a validation error naming the field — before the model is downloaded.

    That delegation stops at :data:`_RUNNER_OWNED`. Those three are already-declared fields, so
    ``extra="forbid"`` waves them through; splatting over them would let a submitted hyperparam
    train a different model than the record names and write the checkpoints and the progress log to
    any absolute path — walking straight past the ``validate_repo_id`` re-check this module exists
    to perform. ``extra_trainer_args`` is checked for the same reason: it is applied LAST inside
    ``to_trl_kwargs``, so it reaches TRL's own ``output_dir`` even when this layer is correct. That
    stays a deliberate, documented escape hatch for a caller driving the runners from Python; it is
    only from an INERT SPEC, where the submitter is not the operator, that it must not aim — and
    for the same reason :func:`_reject_unsafe_trainer_args` keeps a spec's ``extra_trainer_args``
    off the knobs that read, push or run anything under terms other than the run's own.
    """
    _reject_runner_owned(spec.hyperparams)
    _reject_unsafe_trainer_args(spec.hyperparams)
    kwargs: dict[str, Any] = {
        **spec.hyperparams,
        "model_id": spec.model_id,
        "output_dir": str(output_dir),
        # The trainer's own callback writes to the same file the bootstrap and this module append
        # to, so live loss/eval/checkpoint events need no wiring beyond this.
        "progress_jsonl": spec.progress_path,
    }
    if method.name == "sft":
        # Only the flat `text` shape names a column; for the others TRL derives the text from the
        # role columns, and naming a field would make it ignore them.
        kwargs.setdefault("dataset_text_field", sft_text_field(spec.dataset_format))
    try:
        return method.config_cls(**kwargs)
    except ValueError as exc:
        msg = f"invalid hyperparams for {method.name}: {exc}"
        raise RunError(msg) from exc


def _open_split(spec: FinetuneSpec, split: str, token: str | Literal[False]) -> Any:
    """The split as a memory-mapped ``datasets.Dataset``, at the pinned commit when there is one."""
    try:
        # Lazy optional-extra import (forge convention: `Any` so pyright skips the unresolved
        # module); the [finetuning] extra ships `datasets`.
        datasets_mod: Any = __import__("datasets")
    except ImportError as exc:
        msg = "the [finetuning] extra is required to load datasets: pip install 'strata-forge[finetuning]'"
        raise RunError(msg) from exc
    return datasets_mod.load_dataset(
        spec.dataset_id,
        name=spec.dataset_config,
        split=split,
        revision=spec.dataset_commit_sha,
        token=token,
        streaming=False,
    )


def _load_split(
    spec: FinetuneSpec, split: str, token: str | Literal[False]
) -> list[dict[str, Any]]:
    """Materialise one split's rows, projected onto the chosen format's roles."""
    dataset = _open_split(spec, split, token)
    try:
        validate_mapping(spec.dataset_format, spec.column_mapping, dataset.column_names or [])
        # Slice DURING iteration (not after): dataset size is not under our control, so row_limit
        # must bound materialisation — a huge split must not OOM the box before the cap applies.
        source = (
            itertools.islice(dataset, spec.row_limit) if spec.row_limit is not None else dataset
        )
        return build_training_rows(
            (dict(r) for r in source), spec.dataset_format, spec.column_mapping
        )
    except DatasetFormatError as exc:
        raise RunError(f"{split}: {exc}") from exc


def _load_role_split(spec: FinetuneSpec, split: str, token: str | Literal[False]) -> Any:
    """One split as a dataset whose columns are the format's roles, never a Python list.

    The classification population runs to hundreds of thousands of rows, so it stays Arrow on
    disk: projection and tokenisation are ``map`` passes, and ``row_limit`` selects a prefix
    without reading the rest.
    """
    dataset = _open_split(spec, split, token)
    try:
        fmt = validate_mapping(spec.dataset_format, spec.column_mapping, dataset.column_names or [])
    except DatasetFormatError as exc:
        raise RunError(f"{split}: {exc}") from exc
    mapped = {
        role: column
        for role in fmt.roles
        if (column := (spec.column_mapping.get(role) or "").strip())
    }
    if spec.row_limit is not None and spec.row_limit < len(dataset):
        dataset = dataset.select(range(spec.row_limit))

    def project(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return {role: batch[column] for role, column in mapped.items()}

    return dataset.map(
        project,
        batched=True,
        remove_columns=list(dataset.column_names),
        desc=f"Reading {split}",
    )


def _to_dataset(rows: list[dict[str, Any]]) -> Any:
    datasets_mod: Any = __import__("datasets")  # already required by _load_split
    return datasets_mod.Dataset.from_list(rows)


def _build_trainer(
    method: MethodSpec,
    config: Any,
    peft_config: LoRAConfig | QLoRAConfig | None,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]] | None,
    token: str | Literal[False],
    *,
    revision: str | None = None,
) -> Any:
    """Construct the trainer WITHOUT starting it. Blocking — call it in a worker thread.

    Split from the training loop because the two report themselves completely differently: this
    step is silent (downloading the model, quantising it, wiring the adapter) and needs a phase
    caption, while the loop that follows narrates its own steps. Running them under one caption
    was the bug — see :func:`_train`.
    """
    runner: Any = method.build_runner(config, peft_config=peft_config)
    return runner.build_trainer(
        train_dataset=_to_dataset(train_rows),
        eval_dataset=_to_dataset(eval_rows) if eval_rows else None,
        token=token,
        revision=revision,
    )


def _train(trainer: Any, output_dir: Path) -> tuple[dict[str, float], int | None]:
    """Run the loop and save the result. Blocking — call it in a worker thread.

    Returns the trainer's own numbers rather than a ``RunResult``: the runner classes' ``train()``
    bundles building, running and saving into one call, and this module needs the phase boundary
    between the first two. The two lines that differ from ``train()`` are these.
    """
    output = trainer.train()
    trainer.save_model(str(output_dir))
    return numeric_metrics(output), coerce_int(getattr(output, "global_step", None))


def _loaded_commit(trainer: Any) -> str | None:
    """The base-model commit ``from_pretrained`` resolved, as transformers records it."""
    config: Any = getattr(getattr(trainer, "model", None), "config", None)
    commit: Any = getattr(config, "_commit_hash", None)
    return commit if isinstance(commit, str) and commit else None


def _merge_adapter(spec: FinetuneSpec, artifact_dir: Path, token: str | Literal[False]) -> Path:
    """Fold the trained adapter into the base weights, in place.

    Loads the base model (at the pinned revision, when there is one) and applies the saved
    adapter, then writes the merged weights over the artifact directory so the push path is
    identical either way. Blocking — call it in a thread.
    """
    try:
        transformers_mod: Any = __import__("transformers")
        peft_mod: Any = __import__("peft")
    except ImportError as exc:
        msg = "the [finetuning] extra is required to merge an adapter"
        raise RunError(msg) from exc
    base = transformers_mod.AutoModelForCausalLM.from_pretrained(
        spec.model_id, **model_load_kwargs(token, spec.model_revision)
    )
    merged = peft_mod.PeftModel.from_pretrained(
        base, str(artifact_dir), token=token
    ).merge_and_unload()
    merged.save_pretrained(str(artifact_dir))
    # The tokenizer rides along: a merged model that cannot be tokenised is not loadable, and the
    # adapter directory does not carry one.
    transformers_mod.AutoTokenizer.from_pretrained(
        spec.model_id, **tokenizer_load_kwargs(token, spec.model_revision)
    ).save_pretrained(str(artifact_dir))
    return artifact_dir


async def _fetch_checkpoint(spec: FinetuneSpec, token: str | Literal[False]) -> Path:
    """Download the base model's loadable files at its pinned commit; return the local snapshot.

    Safetensors, configs and tokenizer files only (never a pickle checkpoint or a repo's own
    code), and the snapshot's commit must be the pin.
    """
    hub = HFHubClient(token=token)  # explicit: never forge settings or the VM's own login
    try:
        snapshot = await hub.download_snapshot(
            spec.model_id,
            revision=spec.model_revision,
            allow_patterns=SNAPSHOT_PATTERNS,
            ignore_patterns=SNAPSHOT_IGNORED,
        )
    except Exception as exc:
        msg = f"could not download the model {spec.model_id}: {exc}"
        if token is False:
            msg += " (the run carries no Hugging Face token, so a gated or private model is unreadable)"
        raise RunError(msg) from exc
    commit = snapshot_commit(snapshot)
    if spec.model_revision is not None and commit != spec.model_revision:
        msg = f"the model download is commit {commit}, not the pinned {spec.model_revision}"
        raise RunError(msg)
    return snapshot


def _receipt_event(name: str, receipt: Mapping[str, Any]) -> ProgressEvent:
    """A passed check as a ``phase`` event: a sentence for the monitor, numbers in ``metrics``."""
    metrics: dict[str, float] = {}
    if name == "overflow":
        message = (
            f"Length check passed: {receipt['rows']} rows, the longest {receipt['longest']} of "
            f"{receipt['max_length']} tokens"
        )
        metrics = {
            "check_rows": float(receipt["rows"]),
            "check_longest_tokens": float(receipt["longest"]),
            "check_max_length": float(receipt["max_length"]),
            "check_template_token_overhead": float(receipt["template_token_overhead"]),
        }
    elif name == "load":
        message = "Load check passed: only the declared head keys were missing"
        metrics = {"check_missing_keys": float(len(receipt.get("missing") or []))}
    elif name == "step0":
        message = (
            f"Step-0 check passed: the label logits match the base model within "
            f"{receipt['tolerance']:g} (largest difference {receipt['max_abs_logit_diff']:.2e})"
        )
        metrics = {
            "step0_max_abs_logit_diff": float(receipt["max_abs_logit_diff"]),
            "step0_tolerance": float(receipt["tolerance"]),
            "step0_rows": float(receipt["rows"]),
        }
    else:
        peak = receipt.get("peak_bytes")
        total = receipt.get("total_bytes")
        flag = "on" if receipt.get("gradient_checkpointing") else "off"
        if isinstance(peak, int) and isinstance(total, int):
            message = (
                f"Memory check passed: peak {peak / 2**30:.1f} of {total / 2**30:.1f} GiB at the "
                f"longest batch (activation checkpointing {flag})"
            )
            metrics = {"preflight_peak_bytes": float(peak), "preflight_total_bytes": float(total)}
        else:
            message = f"Memory check ran without a GPU to measure (activation checkpointing {flag})"
        metrics["preflight_gradient_checkpointing"] = 1.0 if flag == "on" else 0.0
        metrics["preflight_batch_max_length"] = float(receipt["batch_max_length"])
    return ProgressEvent(kind="phase", stage="load_model", message=message, metrics=metrics)


def _gpu_facts() -> dict[str, Any]:
    """The GPUs this process can use, read once: count and device name, or none."""
    try:
        torch_mod: Any = __import__("torch")
    except ImportError:
        return {"gpu_count": 0, "device": "cpu"}
    if not torch_mod.cuda.is_available():
        return {"gpu_count": 0, "device": "cpu"}
    return {
        "gpu_count": int(torch_mod.cuda.device_count()),
        "device": str(torch_mod.cuda.get_device_name(0)),
    }


def _build_manifest(
    spec: FinetuneSpec,
    method: MethodSpec,
    config: Any,
    *,
    outcome: Literal["succeeded", "refused"],
    model_commit: str | None,
    counts: dict[str, int],
    result: dict[str, Any],
    compute: dict[str, Any],
    checks: dict[str, Any] | None,
) -> dict[str, Any]:
    """What a training run needs to be reproduced and audited, with no row content and no secret."""
    hyperparams: dict[str, Any] = config.model_dump(mode="json", exclude=set(_RUNNER_OWNED))
    return {
        "schema": MANIFEST_SCHEMA,
        "kind": _MANIFEST_KIND,
        "outcome": outcome,
        "engine": manifest_engine(spec.engine_version, _RECORDED_DISTRIBUTIONS),
        "method": {"name": method.name, "task_type": method.task_type},
        "model": {
            "id": spec.model_id,
            "requested_revision": spec.model_revision,
            "commit": model_commit,
        },
        "dataset": {
            "id": spec.dataset_id,
            "config": spec.dataset_config,
            "split": spec.split,
            "eval_split": spec.eval_split,
            "requested_revision": spec.dataset_commit_sha,
            "format": spec.dataset_format,
            "column_mapping": spec.column_mapping,
            "row_limit": spec.row_limit,
        },
        "adapter": {
            "kind": spec.adapter,
            "lora": spec.lora.model_dump(mode="json") if spec.adapter != "none" else None,
            "merged": spec.merge_adapter,
        },
        "hyperparams": hyperparams,
        "counts": counts,
        "result": result,
        "compute": compute,
        "checks": checks,
        "output": {"repo_id": spec.output_repo_id},
    }


def _compute_block(started: float, gpu: dict[str, Any], timing: dict[str, Any]) -> dict[str, Any]:
    """Measured wall-clock time on this machine, and the GPU-seconds a ledger charges for it.

    ``gpu_seconds`` is the runner's own wall-clock time (from the spec check to the manifest)
    times the GPUs it used. It excludes provisioning and the engine install, which happen before
    this process exists and which the control plane measures itself.
    """
    wall = round(time.monotonic() - started, 3)
    return {
        **gpu,
        "wall_seconds": wall,
        "gpu_seconds": round(wall * int(gpu["gpu_count"]), 3),
        "timing": timing,
    }


async def _push_artifact(spec: FinetuneSpec, artifact_dir: Path, hf_token: str) -> str:
    """Push the artifact and its manifest in ONE commit to a private repo; never checkpoints."""
    if not spec.output_repo_id:
        msg = "output_repo_id is required to push the trained model"
        raise RunError(msg)
    out_repo = validate_repo_id(spec.output_repo_id, "output repo")
    hub = HFHubClient(token=hf_token)  # EXPLICIT write token, never the ambient HF_TOKEN
    # Re-checked at push time: visibility can change during a long run.
    await ensure_private_repo(hub, out_repo, repo_type="model")
    await hub.upload_folder(
        artifact_dir,
        out_repo,
        repo_type="model",
        commit_message=f"strata fine-tune ({spec.method})",
        ignore_patterns=_PUSH_IGNORED,
    )
    return out_repo


async def _train_causal_lm(
    spec: FinetuneSpec,
    method: MethodSpec,
    config: Any,
    phase: Any,
    artifact_dir: Path,
    hub_token: str | Literal[False],
) -> tuple[dict[str, Any], dict[str, int], str | None]:
    """The TRL methods: rows in memory, a trainer built under a caption, the loop narrating itself."""
    # Downloading a split is unbounded by row_limit (that only slices during iteration) and runs
    # before any countable milestone. to_thread, not a direct call: load_dataset blocks, and a
    # blocked event loop stops the very ticker that says the step is still running.
    async with ticking_phase(phase, "Loading the dataset", stage="load_model"):
        train_rows = await asyncio.to_thread(_load_split, spec, spec.split, hub_token)
        eval_rows = (
            await asyncio.to_thread(_load_split, spec, spec.eval_split, hub_token)
            if spec.eval_split
            else None
        )
    peft_config = _peft_config(spec, method)

    # No `start` event here. The trainer's own callback emits one the moment TRL begins, carrying
    # the total step count that only it can know — and two starts would make the orchestrator's
    # "has this run actually begun" gate fire on the wrong one.
    #
    # This phase covers everything BEFORE that: downloading the model, quantising it for QLoRA, and
    # wiring the adapter. On a cold box with a large model that is the longest silent stretch of
    # the run, and nothing else reports it.
    async with ticking_phase(phase, "Loading the model onto the GPU", stage="load_model"):
        trainer = await asyncio.to_thread(
            _build_trainer,
            method,
            config,
            peft_config,
            train_rows,
            eval_rows,
            hub_token,
            revision=spec.model_revision,
        )
    commit = _loaded_commit(trainer)
    if spec.model_revision is not None and commit not in (None, spec.model_revision):
        msg = f"the base model loaded at commit {commit}, not the pinned {spec.model_revision}"
        raise RunError(msg)

    # And deliberately NO phase around the loop itself. From here the trainer's own callback
    # reports steps, loss and eval on this same stream, so a caption re-stamped over that would be
    # a stale sentence sitting on top of live progress — and on a multi-day run, tens of thousands
    # of rows of it, enough to exhaust the orchestrator's per-run event budget and cut off the
    # run's own outcome.
    metrics, steps = await asyncio.to_thread(_train, trainer, artifact_dir)

    if spec.merge_adapter:
        async with ticking_phase(phase, "Merging the adapter into the base model", stage="push"):
            await asyncio.to_thread(_merge_adapter, spec, artifact_dir, hub_token)
    counts = {"train_rows": len(train_rows), "eval_rows": len(eval_rows or [])}
    return {"steps": steps, "metrics": metrics}, counts, commit or spec.model_revision


async def _train_seq_cls(
    spec: FinetuneSpec,
    config: SeqClsConfig,
    phase: Any,
    writer: JsonlProgressWriter | None,
    run_dir: Path,
    artifact_dir: Path,
    hub_token: str | Literal[False],
    on_refusal: Any,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    """The classifier: Arrow splits, a pinned local checkpoint, every check before step 0."""
    async with ticking_phase(phase, "Loading the dataset", stage="load_model"):
        train = await asyncio.to_thread(_load_role_split, spec, spec.split, hub_token)
        evaluation = (
            await asyncio.to_thread(_load_role_split, spec, spec.eval_split, hub_token)
            if spec.eval_split
            else None
        )
    async with ticking_phase(phase, "Downloading the model", stage="load_model"):
        checkpoint = await _fetch_checkpoint(spec, hub_token)

    loop = asyncio.get_running_loop()

    def on_receipt(name: str, receipt: dict[str, Any]) -> None:
        # Called from the worker thread; the event is written on the loop's thread, the one the
        # ticking caption writes from, so two lines never interleave in the progress file.
        loop.call_soon_threadsafe(emit, writer, _receipt_event(name, receipt))

    runner = SeqClsRunner(config)
    (run_dir / _CHECKPOINT_SUBDIR).mkdir(parents=True, exist_ok=True)
    try:
        async with ticking_phase(phase, "Checking the run before step 0", stage="load_model"):
            prepared = await asyncio.to_thread(
                runner.prepare,
                checkpoint=checkpoint,
                train_dataset=train,
                eval_dataset=evaluation,
                on_receipt=on_receipt,
            )
    except PreflightError as exc:
        on_refusal({"refused": str(exc), "receipt": exc.receipt})
        detail = str(exc)
        if "memory" in detail:
            detail += f"; base {spec.model_id}@{spec.model_revision}"
        raise RunError(detail) from exc
    # No caption over the loop: the trainer narrates its own steps (see `_train_causal_lm`).
    result = await asyncio.to_thread(
        runner.finish, prepared, artifact_dir=artifact_dir, public_model_id=spec.model_id
    )
    counts = {"train_rows": prepared.train_rows, "eval_rows": prepared.eval_rows}
    return {"steps": result.steps, "metrics": result.metrics}, counts, result.receipts


async def _execute(
    spec: FinetuneSpec, secrets: RunSecrets, writer: JsonlProgressWriter | None
) -> str:
    started = time.monotonic()
    hf_token = secrets.hf_token_value()
    hub_token = secrets.hub_credential()
    # One sampler for the run: the phase sink folds its counters into every caption, so the
    # `load_model` and `push` stretches -- where nothing is countable -- still show the box
    # working. The trainer callback covers `run` with its own.
    gpu = GpuSampler()
    phase = phase_sink(writer, hf_token, gpu=gpu)
    method = pick_method(spec.method)
    pushing = bool(hf_token and spec.output_repo_id)

    run_dir = results_dir(spec.run_id, name=_OUTPUT_DIR_NAME)
    artifact_dir = run_dir / _ARTIFACT_SUBDIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    seq_cls = method.task_type == "SEQ_CLS"
    # A classifier's trainer checkpoints go BESIDE the artifact; TRL's go inside it and are
    # excluded from the push by name.
    config = _trainer_config(
        spec, method, run_dir / _CHECKPOINT_SUBDIR if seq_cls else artifact_dir
    )
    if pushing:
        # Before any GPU time: a public repo is refused now, not after the run.
        async with ticking_phase(phase, "Checking the output repo", stage="load_model"):
            await ensure_private_repo(
                HFHubClient(token=hf_token),
                validate_repo_id(cast("str", spec.output_repo_id), "output repo"),
                repo_type="model",
            )

    gpu_facts = await asyncio.to_thread(_gpu_facts)
    checks: dict[str, Any] | None = None
    if seq_cls:

        def on_refusal(refusal: dict[str, Any]) -> None:
            write_manifest(
                _build_manifest(
                    spec,
                    method,
                    config,
                    outcome="refused",
                    model_commit=spec.model_revision,
                    counts={},
                    result={},
                    compute=_compute_block(
                        started, {**gpu_facts, "gpu_count": min(1, gpu_facts["gpu_count"])}, {}
                    ),
                    checks=refusal,
                ),
                run_dir,
            )

        result, counts, checks = await _train_seq_cls(
            spec, config, phase, writer, run_dir, artifact_dir, hub_token, on_refusal
        )
        model_commit = spec.model_revision
        # One device, by the runner's own refusal of more.
        gpu_facts = {**gpu_facts, "gpu_count": min(1, gpu_facts["gpu_count"])}
        timing = dict(checks.get("timing") or {})
    else:
        result, counts, model_commit = await _train_causal_lm(
            spec, method, config, phase, artifact_dir, hub_token
        )
        timing = {}

    compute = _compute_block(started, gpu_facts, timing)
    # Inside the artifact directory, so the push carries the weights and their record in ONE
    # commit, and the copy left on the VM carries it too.
    await asyncio.to_thread(
        write_manifest,
        _build_manifest(
            spec,
            method,
            config,
            outcome="succeeded",
            model_commit=model_commit,
            counts=counts,
            result=result,
            compute=compute,
            checks=checks,
        ),
        artifact_dir,
    )

    if pushing:
        try:
            async with ticking_phase(phase, "Uploading the model to the Hub", stage="push"):
                destination = await _push_artifact(spec, artifact_dir, cast("str", hf_token))
        except Exception as exc:
            # Name WHERE the weights are. The run still fails — the user asked for them on the Hub
            # and they are not there — but at the last step of a run that may have taken hours it
            # matters most that the output was not lost with it.
            msg = f"the trained model is on the VM at {artifact_dir}, but the push failed: {exc}"
            raise RunError(msg) from exc
    else:
        destination = str(artifact_dir)

    steps = cast("int | None", result.get("steps"))
    # The RUN's end, after the TRAINER's own end (its callback emits one when TRL stops). The step
    # count is carried across deliberately: this event is the last one anybody reads, and without
    # it a finished run's progress reads as unknown rather than as complete.
    emit(
        writer,
        ProgressEvent(
            kind="end",
            step=steps,
            total_steps=steps,
            metrics={
                **cast("dict[str, float]", result.get("metrics") or {}),
                "gpu_seconds": float(compute["gpu_seconds"]),
            },
            message=destination,  # WHERE it landed: a repo id (pushed) or a VM path (local)
        ),
    )
    return destination


async def main() -> int:
    """Entry point: returns a process exit code (0 ok, 1 failure). Never leaks the token."""
    return await runner_main(
        lambda writer, secrets: _execute(load_spec(writer=writer), secrets, writer)
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
