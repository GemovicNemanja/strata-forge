"""``strata-forge train`` — kick off SFT or DPO fine-tuning jobs.

The subcommands are thin wrappers over :class:`SFTRunner` and
:class:`PreferenceRunner` (DPO method). Forge resolves the
dataset from the configured :class:`DatasetStore` by name, or —
with ``--dataset-file`` — from a serialized :class:`Dataset` JSON
file an orchestrator shipped to the worker (so a remote box needs
no store access). Everything else flows from CLI flags. Heavy ML
deps (``torch``, ``transformers``, ``trl``, ``peft``,
``datasets``) lazy-import inside the runners.

For GRPO / ORPO / KTO or full hyperparameter control, drop into
Python and use :mod:`strata_forge.training` directly — the CLI covers
the day-to-day SFT + DPO path, not the entire surface.
"""

from __future__ import annotations

import asyncio
from pathlib import Path  # noqa: TC003 — typer evaluates annotations at runtime
from typing import Any, Literal

import typer
from rich.console import Console

from strata_forge.cli.helpers import (
    dataset_store_from_settings,
    error_exit,
    run_async,
)

__all__ = ["app"]


PrecisionName = Literal["fp32", "fp16", "bf16"]
AdapterName = Literal["none", "lora", "qlora"]


app = typer.Typer(
    name="train",
    help="Run SFT or DPO fine-tuning over the configured dataset store.",
    no_args_is_help=True,
)


@app.command("sft")
def sft_cmd(
    model: str = typer.Option(..., "--model", "-m", help="HuggingFace model id or local path."),
    dataset: str = typer.Option(
        ..., "--dataset", "-d", help="Dataset name in the configured store."
    ),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Where to save checkpoints."),
    dataset_version: str | None = typer.Option(
        None, "--dataset-version", help="Pin a specific dataset version."
    ),
    dataset_file: Path | None = typer.Option(
        None,
        "--dataset-file",
        help="Load the dataset from a serialized Dataset JSON file instead of the store "
        "(used by orchestration to ship a dataset to a remote worker).",
    ),
    epochs: float = typer.Option(1.0, "--epochs", help="Training epochs."),
    batch_size: int = typer.Option(1, "--batch-size", min=1, help="Per-device batch size."),
    grad_accum: int = typer.Option(8, "--grad-accum", min=1, help="Gradient accumulation steps."),
    learning_rate: float = typer.Option(2e-4, "--lr", help="Optimizer LR."),
    precision: PrecisionName = typer.Option("bf16", "--precision", help="Training precision."),
    max_seq_length: int = typer.Option(
        2048, "--max-seq-length", min=8, help="Per-example token budget."
    ),
    adapter: AdapterName = typer.Option("none", "--adapter", help="PEFT adapter shape."),
    adapter_rank: int = typer.Option(16, "--adapter-rank", min=1, help="LoRA/QLoRA rank."),
    seed: int = typer.Option(42, "--seed"),
    progress_jsonl: Path | None = typer.Option(
        None, "--progress-jsonl", help="Write JSONL progress events here (for orchestration)."
    ),
) -> None:
    """Run a supervised fine-tuning job."""
    run_async(
        _run_sft(
            model_id=model,
            dataset_name=dataset,
            output_dir=output_dir,
            dataset_version=dataset_version,
            dataset_file=dataset_file,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            learning_rate=learning_rate,
            precision=precision,
            max_seq_length=max_seq_length,
            adapter=adapter,
            adapter_rank=adapter_rank,
            seed=seed,
            progress_jsonl=progress_jsonl,
        )
    )


@app.command("dpo")
def dpo_cmd(
    model: str = typer.Option(
        ..., "--model", "-m", help="HuggingFace model id (typically your SFT output)."
    ),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Preference-pair dataset name."),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Where to save checkpoints."),
    dataset_version: str | None = typer.Option(
        None, "--dataset-version", help="Pin a specific dataset version."
    ),
    dataset_file: Path | None = typer.Option(
        None,
        "--dataset-file",
        help="Load the dataset from a serialized Dataset JSON file instead of the store "
        "(used by orchestration to ship a dataset to a remote worker).",
    ),
    beta: float = typer.Option(0.1, "--beta", help="DPO beta."),
    epochs: float = typer.Option(1.0, "--epochs", help="Training epochs."),
    batch_size: int = typer.Option(1, "--batch-size", min=1),
    grad_accum: int = typer.Option(8, "--grad-accum", min=1),
    learning_rate: float = typer.Option(5e-6, "--lr"),
    precision: PrecisionName = typer.Option("bf16", "--precision"),
    max_length: int = typer.Option(2048, "--max-length", min=8),
    adapter: AdapterName = typer.Option("none", "--adapter"),
    adapter_rank: int = typer.Option(16, "--adapter-rank", min=1),
    seed: int = typer.Option(42, "--seed"),
    progress_jsonl: Path | None = typer.Option(
        None, "--progress-jsonl", help="Write JSONL progress events here (for orchestration)."
    ),
) -> None:
    """Run a Direct Preference Optimization job."""
    run_async(
        _run_dpo(
            model_id=model,
            dataset_name=dataset,
            output_dir=output_dir,
            dataset_version=dataset_version,
            dataset_file=dataset_file,
            beta=beta,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            learning_rate=learning_rate,
            precision=precision,
            max_length=max_length,
            adapter=adapter,
            adapter_rank=adapter_rank,
            seed=seed,
            progress_jsonl=progress_jsonl,
        )
    )


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _build_peft(adapter: str, rank: int) -> Any:
    if adapter == "none":
        return None
    from strata_forge.training.peft import LoRAConfig, QLoRAConfig

    if adapter == "lora":
        return LoRAConfig(r=rank)
    if adapter == "qlora":
        return QLoRAConfig(lora=LoRAConfig(r=rank))
    error_exit(f"unknown adapter {adapter!r}")


async def _resolve_dataset(name: str, version: str | None, dataset_file: Path | None) -> Any:
    if dataset_file is not None:
        from strata_forge.datasets import Dataset

        try:
            text = await asyncio.to_thread(dataset_file.read_text, encoding="utf-8")
            return Dataset.model_validate_json(text)
        except (OSError, ValueError) as exc:
            error_exit(f"could not load --dataset-file {dataset_file}: {exc}")

    from strata_forge.datasets.store import DatasetNotFoundError

    store = dataset_store_from_settings()
    try:
        return await store.get(name, version=version)
    except DatasetNotFoundError as exc:
        error_exit(str(exc))


async def _run_sft(
    *,
    model_id: str,
    dataset_name: str,
    output_dir: Path,
    dataset_version: str | None,
    dataset_file: Path | None,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    precision: PrecisionName,
    max_seq_length: int,
    adapter: AdapterName,
    adapter_rank: int,
    seed: int,
    progress_jsonl: Path | None,
) -> None:
    from strata_forge.training.sft import SFTConfig, SFTRunner

    dataset = await _resolve_dataset(dataset_name, dataset_version, dataset_file)
    peft_cfg = _build_peft(adapter, adapter_rank)

    config = SFTConfig(
        model_id=model_id,
        output_dir=str(output_dir),
        num_epochs=epochs,
        per_device_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        precision=precision,
        max_seq_length=max_seq_length,
        seed=seed,
        progress_jsonl=str(progress_jsonl) if progress_jsonl else None,
    )

    console = Console()
    console.print(
        f"[bold]strata-forge train sft[/]  model={model_id}  dataset={dataset_name}  adapter={adapter}"
    )

    runner = SFTRunner(config, peft_config=peft_cfg)
    try:
        result = runner.train(train_dataset=dataset)
    except ImportError as exc:
        error_exit(str(exc))
    console.print(
        f"[bold green]done[/]  output_dir={result.output_dir}  train_loss={result.train_loss}"
    )


async def _run_dpo(
    *,
    model_id: str,
    dataset_name: str,
    output_dir: Path,
    dataset_version: str | None,
    dataset_file: Path | None,
    beta: float,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    precision: PrecisionName,
    max_length: int,
    adapter: AdapterName,
    adapter_rank: int,
    seed: int,
    progress_jsonl: Path | None,
) -> None:
    from strata_forge.training.preference import DPOConfig, PreferenceRunner

    dataset = await _resolve_dataset(dataset_name, dataset_version, dataset_file)
    peft_cfg = _build_peft(adapter, adapter_rank)

    config = DPOConfig(
        model_id=model_id,
        output_dir=str(output_dir),
        beta=beta,
        num_epochs=epochs,
        per_device_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        precision=precision,
        max_length=max_length,
        seed=seed,
        progress_jsonl=str(progress_jsonl) if progress_jsonl else None,
    )

    console = Console()
    console.print(
        f"[bold]strata-forge train dpo[/]  model={model_id}  dataset={dataset_name}  beta={beta}"
    )

    runner = PreferenceRunner(config, peft_config=peft_cfg)
    try:
        result = runner.train(train_dataset=dataset)
    except ImportError as exc:
        error_exit(str(exc))
    console.print(
        f"[bold green]done[/]  output_dir={result.output_dir}  train_loss={result.train_loss}"
    )
