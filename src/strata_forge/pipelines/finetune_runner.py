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
  - The HF write token arrives in its OWN env var (``HF_WRITE_TOKEN``), never in
    ``STRATA_RUN_CONFIG``, is passed EXPLICITLY to the Hub/dataset clients (never the VM's
    ambient ``HF_TOKEN``), and is scrubbed from every surfaced message.
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
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.pipelines._common import (
    RunError,
    emit,
    load_config,
    phase_sink,
    results_dir,
    runner_main,
    ticking_phase,
    validate_repo_id,
)
from strata_forge.storage import HFHubClient
from strata_forge.training.dataset_format import (
    DatasetFormatError,
    build_training_rows,
    sft_text_field,
    validate_mapping,
)
from strata_forge.training.methods import (
    UnsupportedMethodError,
    check_format,
    pick_method,
)
from strata_forge.training.peft import LoRAConfig, QLoRAConfig
from strata_forge.training.progress import ProgressEvent, coerce_int, numeric_metrics

if TYPE_CHECKING:
    from pathlib import Path

    from strata_forge.training.methods import MethodSpec
    from strata_forge.training.progress import JsonlProgressWriter

__all__ = ["FinetuneSpec", "load_spec", "main"]

# Where the trained artifact is written on the VM. OUTSIDE the per-run workdir the orchestrator
# deletes on every terminal state, so a failed upload does not take the weights with it.
_OUTPUT_DIR_NAME = "strata-finetune-output"
# The subdirectory of that run directory the trainer saves into, so the run dir can later hold
# siblings (a merged copy, a metrics dump) without them being uploaded as part of the adapter.
_ARTIFACT_SUBDIR = "artifact"


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

    method: str
    model_id: str
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


def load_spec() -> FinetuneSpec:
    """Parse + validate the spec, including the pairings only this module can check.

    Every check here is one that would otherwise fail on the GPU: an unknown method, a dataset
    shape the method cannot train on, a merge asked for with no adapter to merge. They cost
    milliseconds at launch and minutes-to-hours anywhere later.
    """
    spec = load_config(FinetuneSpec)
    validate_repo_id(spec.model_id, "model")
    validate_repo_id(spec.dataset_id, "dataset")
    if spec.output_repo_id is not None:
        validate_repo_id(spec.output_repo_id, "output repo")
    try:
        method = pick_method(spec.method)
        check_format(method, spec.dataset_format)
    except UnsupportedMethodError as exc:
        raise RunError(str(exc)) from exc
    if spec.merge_adapter and spec.adapter == "none":
        msg = "merge_adapter has nothing to merge: the run trains full weights (adapter='none')"
        raise RunError(msg)
    return spec


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


def _trainer_config(spec: FinetuneSpec, method: MethodSpec, output_dir: Path) -> Any:
    """Build the method's typed config from the inert spec.

    ``hyperparams`` is splatted in rather than copied field by field: the config's own
    ``extra="forbid"`` is the authority on what each method accepts, and re-stating that list here
    would be a second definition to keep in sync. A knob that does not apply therefore surfaces as
    a validation error naming the field — before the model is downloaded.
    """
    kwargs: dict[str, Any] = {
        "model_id": spec.model_id,
        "output_dir": str(output_dir),
        # The trainer's own callback writes to the same file the bootstrap and this module append
        # to, so live loss/eval/checkpoint events need no wiring beyond this.
        "progress_jsonl": spec.progress_path,
        **spec.hyperparams,
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


def _load_split(spec: FinetuneSpec, split: str, hf_token: str | None) -> list[dict[str, Any]]:
    """Materialise one split's rows, projected onto the chosen format's roles."""
    try:
        # Lazy optional-extra import (forge convention: `Any` so pyright skips the unresolved
        # module); the [finetuning] extra ships `datasets`.
        datasets_mod: Any = __import__("datasets")
    except ImportError as exc:
        msg = "the [finetuning] extra is required to load datasets: pip install 'strata-forge[finetuning]'"
        raise RunError(msg) from exc
    dataset = datasets_mod.load_dataset(
        spec.dataset_id,
        name=spec.dataset_config,
        split=split,
        revision=spec.dataset_commit_sha,
        token=hf_token or None,
        streaming=False,
    )
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


def _to_dataset(rows: list[dict[str, Any]]) -> Any:
    datasets_mod: Any = __import__("datasets")  # already required by _load_split
    return datasets_mod.Dataset.from_list(rows)


def _build_trainer(
    method: MethodSpec,
    config: Any,
    peft_config: LoRAConfig | QLoRAConfig | None,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]] | None,
) -> Any:
    """Construct the trainer WITHOUT starting it. Blocking — call it in a worker thread.

    Split from the training loop because the two report themselves completely differently: this
    step is silent (downloading the model, quantising it, wiring the adapter) and needs a phase
    caption, while the loop that follows narrates its own steps. Running them under one caption
    was the bug — see :func:`_train`.
    """
    runner = method.build_runner(config, peft_config=peft_config)
    return runner.build_trainer(
        train_dataset=_to_dataset(train_rows),
        eval_dataset=_to_dataset(eval_rows) if eval_rows else None,
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


def _merge_adapter(spec: FinetuneSpec, artifact_dir: Path) -> Path:
    """Fold the trained adapter into the base weights, in place.

    Loads the base model and applies the saved adapter, then writes the merged weights over the
    artifact directory so the push path is identical either way. Blocking — call it in a thread.
    """
    try:
        transformers_mod: Any = __import__("transformers")
        peft_mod: Any = __import__("peft")
    except ImportError as exc:
        msg = "the [finetuning] extra is required to merge an adapter"
        raise RunError(msg) from exc
    base = transformers_mod.AutoModelForCausalLM.from_pretrained(spec.model_id)
    merged = peft_mod.PeftModel.from_pretrained(base, str(artifact_dir)).merge_and_unload()
    merged.save_pretrained(str(artifact_dir))
    # The tokenizer rides along: a merged model that cannot be tokenised is not loadable, and the
    # adapter directory does not carry one.
    transformers_mod.AutoTokenizer.from_pretrained(spec.model_id).save_pretrained(str(artifact_dir))
    return artifact_dir


async def _push_artifact(spec: FinetuneSpec, artifact_dir: Path, hf_token: str) -> str:
    if not spec.output_repo_id:
        msg = "output_repo_id is required to push the trained model"
        raise RunError(msg)
    out_repo = validate_repo_id(spec.output_repo_id, "output repo")
    hub = HFHubClient(token=hf_token)  # EXPLICIT write token, never the ambient HF_TOKEN
    await hub.create_repo(out_repo, repo_type="model", private=True, exist_ok=True)
    await hub.upload_folder(
        artifact_dir,
        out_repo,
        repo_type="model",
        commit_message=f"strata fine-tune ({spec.method})",
    )
    return out_repo


async def _execute(
    spec: FinetuneSpec, hf_token: str | None, writer: JsonlProgressWriter | None
) -> str:
    phase = phase_sink(writer, hf_token)
    method = pick_method(spec.method)

    # Downloading a split is unbounded by row_limit (that only slices during iteration) and runs
    # before any countable milestone. to_thread, not a direct call: load_dataset blocks, and a
    # blocked event loop stops the very ticker that says the step is still running.
    async with ticking_phase(phase, "Loading the dataset"):
        train_rows = await asyncio.to_thread(_load_split, spec, spec.split, hf_token)
        eval_rows = (
            await asyncio.to_thread(_load_split, spec, spec.eval_split, hf_token)
            if spec.eval_split
            else None
        )

    run_dir = results_dir(spec.run_id, name=_OUTPUT_DIR_NAME)
    artifact_dir = run_dir / _ARTIFACT_SUBDIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config = _trainer_config(spec, method, artifact_dir)
    peft_config = _peft_config(spec, method)

    # No `start` event here. The trainer's own callback emits one the moment TRL begins, carrying
    # the total step count that only it can know — and two starts would make the orchestrator's
    # "has this run actually begun" gate fire on the wrong one.
    #
    # This phase covers everything BEFORE that: downloading the model, quantising it for QLoRA, and
    # wiring the adapter. On a cold box with a large model that is the longest silent stretch of
    # the run, and nothing else reports it.
    async with ticking_phase(phase, "Loading the model onto the GPU"):
        trainer = await asyncio.to_thread(
            _build_trainer, method, config, peft_config, train_rows, eval_rows
        )

    # And deliberately NO phase around the loop itself. From here the trainer's own callback
    # reports steps, loss and eval on this same stream, so a caption re-stamped over that would be
    # a stale sentence sitting on top of live progress — and on a multi-day run, tens of thousands
    # of rows of it, enough to exhaust the orchestrator's per-run event budget and cut off the
    # run's own outcome.
    metrics, steps = await asyncio.to_thread(_train, trainer, artifact_dir)

    if spec.merge_adapter:
        async with ticking_phase(phase, "Merging the adapter into the base model"):
            await asyncio.to_thread(_merge_adapter, spec, artifact_dir)

    if hf_token and spec.output_repo_id:
        try:
            async with ticking_phase(phase, "Uploading the model to the Hub"):
                destination = await _push_artifact(spec, artifact_dir, hf_token)
        except Exception as exc:
            # Name WHERE the weights are. The run still fails — the user asked for them on the Hub
            # and they are not there — but at the last step of a run that may have taken hours it
            # matters most that the output was not lost with it.
            msg = f"the trained model is on the VM at {artifact_dir}, but the push failed: {exc}"
            raise RunError(msg) from exc
    else:
        destination = str(artifact_dir)

    # The RUN's end, after the TRAINER's own end (its callback emits one when TRL stops). The step
    # count is carried across deliberately: this event is the last one anybody reads, and without
    # it a finished run's progress reads as unknown rather than as complete.
    emit(
        writer,
        ProgressEvent(
            kind="end",
            step=steps,
            total_steps=steps,
            metrics=metrics,
            message=destination,  # WHERE it landed: a repo id (pushed) or a VM path (local)
        ),
    )
    return destination


async def main() -> int:
    """Entry point: returns a process exit code (0 ok, 1 failure). Never leaks the token."""
    return await runner_main(lambda writer, token: _execute(load_spec(), token, writer))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
