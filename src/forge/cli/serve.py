"""``forge serve`` — print or launch self-hosted inference servers.

Three subcommands wrap the corresponding :mod:`forge.compute.serving`
task builder:

- ``vllm`` — vLLM ``serve`` command.
- ``tgi`` — Docker-launched Text Generation Inference.
- ``sglang`` — SGLang ``launch_server`` command.

By default each subcommand only **prints** the YAML task that
would be submitted (so callers can route it to SkyPilot / SSH /
elsewhere). Pass ``--submit local`` to actually run it on the
in-process :class:`LocalBackend`; the submitted job lands in
``~/.forge/jobs`` and ``forge compute status <id>`` takes over.
"""

from __future__ import annotations

from typing import Literal

import typer
from rich.console import Console

from forge.cli.helpers import error_exit, run_async

__all__ = ["app"]


SubmitTarget = Literal["none", "local"]


app = typer.Typer(
    name="serve",
    help="Build / submit self-hosted inference serving tasks.",
    no_args_is_help=True,
)


@app.command("vllm")
def vllm_cmd(
    model: str = typer.Option(..., "--model", "-m", help="Model id."),
    port: int = typer.Option(8000, "--port"),
    tensor_parallel: int = typer.Option(1, "--tp", "--tensor-parallel"),
    max_model_len: int | None = typer.Option(None, "--max-model-len"),
    dtype: str | None = typer.Option(None, "--dtype"),
    submit: SubmitTarget = typer.Option(
        "none",
        "--submit",
        help="`none` prints the YAML; `local` submits to LocalBackend.",
    ),
) -> None:
    """Build (and optionally submit) a vLLM serving task."""
    from forge.compute.serving import build_vllm_task

    task = build_vllm_task(
        model,
        port=port,
        tensor_parallel_size=tensor_parallel,
        max_model_len=max_model_len,
        dtype=dtype,
    )
    base_url = f"http://localhost:{port}/v1"
    _emit_or_submit(task, submit=submit, base_url=base_url)


@app.command("tgi")
def tgi_cmd(
    model: str = typer.Option(..., "--model", "-m", help="Model id."),
    port: int = typer.Option(8080, "--port"),
    num_shard: int = typer.Option(1, "--num-shard"),
    max_input_tokens: int | None = typer.Option(None, "--max-input-tokens"),
    max_total_tokens: int | None = typer.Option(None, "--max-total-tokens"),
    submit: SubmitTarget = typer.Option(
        "none",
        "--submit",
        help="`none` prints the YAML; `local` submits to LocalBackend.",
    ),
) -> None:
    """Build (and optionally submit) a TGI Docker serving task."""
    from forge.compute.serving import build_tgi_task

    task = build_tgi_task(
        model,
        port=port,
        num_shard=num_shard,
        max_input_tokens=max_input_tokens,
        max_total_tokens=max_total_tokens,
    )
    base_url = f"http://localhost:{port}/v1"
    _emit_or_submit(task, submit=submit, base_url=base_url)


@app.command("sglang")
def sglang_cmd(
    model: str = typer.Option(..., "--model", "-m", help="Model id."),
    port: int = typer.Option(30000, "--port"),
    tp_size: int = typer.Option(1, "--tp-size"),
    submit: SubmitTarget = typer.Option(
        "none",
        "--submit",
        help="`none` prints the YAML; `local` submits to LocalBackend.",
    ),
) -> None:
    """Build (and optionally submit) an SGLang serving task."""
    from forge.compute.serving import build_sglang_task

    task = build_sglang_task(model, port=port, tp_size=tp_size)
    base_url = f"http://localhost:{port}/v1"
    _emit_or_submit(task, submit=submit, base_url=base_url)


# ---------------------------------------------------------------------------
# Shared submit / emit helper
# ---------------------------------------------------------------------------


def _emit_or_submit(task: object, *, submit: str, base_url: str) -> None:
    from forge.compute.task import Task

    if not isinstance(task, Task):  # pragma: no cover — defensive
        error_exit("internal error: build_* returned a non-Task")

    console = Console()
    if submit == "none":
        console.print(task.to_yaml())
        return
    if submit == "local":
        run_async(_submit_local(task, base_url))
        return
    error_exit(f"unknown --submit target {submit!r}")


async def _submit_local(task: object, base_url: str) -> None:
    from forge.cli import compute as compute_cli
    from forge.compute.backends.local import LocalBackend
    from forge.compute.task import Task

    if not isinstance(task, Task):  # pragma: no cover
        error_exit("internal error")

    backend = LocalBackend()
    job = await backend.submit(task)
    compute_cli.save_job(job, "local", {})

    console = Console()
    console.print(f"[bold green]submitted[/] {job.id}")
    console.print(f"  task:      {task.name}")
    console.print(f"  base_url:  {base_url}")
    console.print(
        f"\n[dim]use `forge compute status {job.id}` "
        f"and `forge compute logs {job.id}` to monitor.[/]"
    )
