"""``forge compute`` — submit, monitor, cancel, and cleanup compute jobs.

Submitting a task persists the resulting :class:`Job` to
``~/.forge/jobs/<id>.json`` so subsequent commands (``status``,
``logs``, ``cancel``, ``cleanup``) can reconstruct the right
backend without the user passing the same flags every time. The
state file itself doubles as the authoritative record of what
got launched.

Local backend works out of the box. SSH and SkyPilot inherit
their credentials from the standard environment variables
(``SSH_*`` / ``SKYPILOT_*``) the same way the Python API does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import typer
from rich.console import Console
from rich.table import Table

from forge.cli.helpers import error_exit, run_async

if TYPE_CHECKING:
    from forge.compute.backends.base import Backend
    from forge.compute.job import Job

__all__ = ["app"]


BackendName = Literal["local", "ssh", "skypilot"]


app = typer.Typer(
    name="compute",
    help="Submit and manage compute jobs (local / ssh / skypilot).",
    no_args_is_help=True,
)


_STATE_DIR = Path.home() / ".forge" / "jobs"


def _state_path(job_id: str) -> Path:
    return _STATE_DIR / f"{job_id}.json"


def save_job(job: Job, backend_name: str, backend_kwargs: dict[str, Any]) -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _state_path(job.id)
    payload = {
        "job": job.model_dump(mode="json"),
        "backend": backend_name,
        "backend_kwargs": backend_kwargs,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _load_job(job_id: str) -> tuple[Job, str, dict[str, Any]]:
    from forge.compute.job import Job as _Job

    path = _state_path(job_id)
    if not path.exists():
        error_exit(f"unknown job id {job_id!r} (no state file at {path})")
    payload = json.loads(path.read_text())
    job = _Job.model_validate(payload["job"])
    return job, payload["backend"], payload.get("backend_kwargs", {})


def _make_backend(name: str, kwargs: dict[str, Any]) -> Backend:
    if name == "local":
        from forge.compute.backends.local import LocalBackend

        return LocalBackend(**kwargs)
    if name == "ssh":
        from forge.compute.backends.ssh import SSHBackend

        return SSHBackend(**kwargs)
    if name == "skypilot":
        from forge.compute.backends.skypilot import SkyPilotBackend

        return SkyPilotBackend(**kwargs)
    error_exit(f"unknown backend {name!r}")


@app.command("submit")
def submit_cmd(
    task_file: Path = typer.Argument(..., exists=True, dir_okay=False, help="YAML task file."),
    backend: BackendName = typer.Option(
        "local", "--backend", "-b", help="Which backend to run on."
    ),
    ssh_host: str | None = typer.Option(
        None, "--ssh-host", help="Hostname (required when --backend=ssh)."
    ),
    ssh_user: str | None = typer.Option(
        None, "--ssh-user", help="Username (required when --backend=ssh)."
    ),
    ssh_port: int = typer.Option(22, "--ssh-port"),
) -> None:
    """Submit a YAML task to a backend; print the resulting job ID."""
    backend_kwargs: dict[str, Any] = {}
    if backend == "ssh":
        if not ssh_host or not ssh_user:
            error_exit("--backend=ssh requires --ssh-host and --ssh-user")
        backend_kwargs = {
            "host": ssh_host,
            "username": ssh_user,
            "port": ssh_port,
        }
    run_async(_submit(task_file, backend, backend_kwargs))


@app.command("status")
def status_cmd(
    job_id: str = typer.Argument(..., help="Job id printed by `submit`."),
) -> None:
    """Print the current status of a saved job."""
    run_async(_status(job_id))


@app.command("logs")
def logs_cmd(
    job_id: str = typer.Argument(..., help="Job id printed by `submit`."),
    tail: int | None = typer.Option(
        None, "--tail", "-t", min=1, help="Only show the last N lines."
    ),
) -> None:
    """Print captured stdout/stderr for a saved job."""
    run_async(_logs(job_id, tail))


@app.command("cancel")
def cancel_cmd(
    job_id: str = typer.Argument(..., help="Job id printed by `submit`."),
) -> None:
    """Cancel a running job."""
    run_async(_cancel(job_id))


@app.command("cleanup")
def cleanup_cmd(
    job_id: str = typer.Argument(..., help="Job id printed by `submit`."),
) -> None:
    """Run the backend's cleanup and remove the local state file."""
    run_async(_cleanup(job_id))


@app.command("list")
def list_cmd() -> None:
    """List saved jobs from `~/.forge/jobs`."""
    _list()


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


async def _submit(task_file: Path, backend_name: str, backend_kwargs: dict[str, Any]) -> None:
    from forge.compute.task import Task

    try:
        task = Task.from_yaml(task_file)
    except Exception as exc:
        error_exit(f"failed to load task: {exc}")
    backend = _make_backend(backend_name, backend_kwargs)
    try:
        job = await backend.submit(task)
    except Exception as exc:
        error_exit(f"submit failed: {exc}")

    path = save_job(job, backend_name, backend_kwargs)
    console = Console()
    console.print(f"[bold green]submitted[/] {job.id}")
    console.print(f"  task:     {job.task_name}")
    console.print(f"  backend:  {backend_name}")
    console.print(f"  state:    {path}")


async def _status(job_id: str) -> None:
    job, backend_name, backend_kwargs = _load_job(job_id)
    backend = _make_backend(backend_name, backend_kwargs)
    status = await backend.status(job)
    console = Console()
    console.print(f"[bold]{job.id}[/] [dim]({job.task_name})[/]")
    console.print(f"  state:  {status.state}")
    if status.message:
        console.print(f"  msg:    {status.message}")
    if status.exit_code is not None:
        console.print(f"  exit:   {status.exit_code}")


async def _logs(job_id: str, tail: int | None) -> None:
    job, backend_name, backend_kwargs = _load_job(job_id)
    backend = _make_backend(backend_name, backend_kwargs)
    text = await backend.logs(job, tail=tail)
    Console().print(text.rstrip() if text else "[dim](no logs yet)[/]")


async def _cancel(job_id: str) -> None:
    job, backend_name, backend_kwargs = _load_job(job_id)
    backend = _make_backend(backend_name, backend_kwargs)
    await backend.cancel(job)
    Console().print(f"[bold yellow]cancelled[/] {job.id}")


async def _cleanup(job_id: str) -> None:
    job, backend_name, backend_kwargs = _load_job(job_id)
    backend = _make_backend(backend_name, backend_kwargs)
    await backend.cleanup(job)
    _state_path(job_id).unlink(missing_ok=True)
    Console().print(f"[bold green]cleaned up[/] {job.id}")


def _list() -> None:
    from forge.compute.job import Job as _Job

    console = Console()
    if not _STATE_DIR.exists():
        console.print("[dim](no saved jobs)[/]")
        return
    entries = sorted(_STATE_DIR.glob("*.json"))
    if not entries:
        console.print("[dim](no saved jobs)[/]")
        return
    table = Table(title="saved compute jobs")
    table.add_column("id")
    table.add_column("task")
    table.add_column("backend")
    for path in entries:
        try:
            payload = json.loads(path.read_text())
            job = _Job.model_validate(payload["job"])
            table.add_row(job.id, job.task_name, payload["backend"])
        except json.JSONDecodeError, OSError, ValueError:  # pragma: no cover
            continue
    console.print(table)
