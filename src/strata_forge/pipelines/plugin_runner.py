"""Run a trusted plugin's entry point on the compute target.

Launched by an orchestrator as ``python -m strata_forge.pipelines.plugin_runner`` with the inert
spec in ``STRATA_RUN_CONFIG`` and the allowed builds in
:data:`~strata_forge.plugins.pins.ALLOWED_PLUGINS_ENV`. The order is fixed, and every check
happens before any plugin code runs:

1. the spec is parsed as data and the engine version handshake applied (``_common.load_config``);
2. the pinned build must be one the orchestrator allowed for this job (its own configuration,
   never the spec), and the spec's entry point, bindings and configuration are re-checked against
   the interface it carries (the one the orchestrator read at the pinned commit);
3. the installed distribution must be exactly the pinned build
   (:func:`~strata_forge.plugins.pins.installed_mismatch`), and it must register the entry point
   by name;
4. the declared inputs are downloaded HERE, with the run's credential, at their full commits:
   dataset splits to local parquet (each checked for its required columns), model inputs as
   snapshots without pickle-format weights;
5. the plugin runs in a child process (:mod:`strata_forge.plugins.host`) with an allow-listed
   environment that holds no credential, an offline and empty Hub cache, and a work directory of
   its own; a cancellation stops its whole process group;
6. its output is checked (a step scorer must return one score per input step), a manifest is
   written beside it, and the directory is pushed to the run's private output repository.

The plugin never holds the Hub credential, never chooses what it reads, and never pushes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from strata_forge.pipelines._common import (
    COMMIT_SHA_RE,
    MANIFEST_SCHEMA,
    RunError,
    RunSecrets,
    emit,
    load_config,
    manifest_engine,
    results_dir,
    run_redactor,
    runner_main,
    validate_repo_id,
    write_manifest,
)
from strata_forge.plugins.host import SCORES_FILENAME, HostJob
from strata_forge.plugins.manifest import STEP_INPUT_ROLE, EntryPointDecl, ScalarValue
from strata_forge.plugins.pins import (
    ALLOWED_PLUGINS_ENV,
    PluginPin,
    installed_mismatch,
    registered_entry_point,
)
from strata_forge.plugins.preflight import InputBinding, PluginRequest, check_bindings, check_config
from strata_forge.storage.hf_hub import HFHubClient
from strata_forge.training.progress import JsonlProgressWriter, ProgressEvent

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["PLUGIN_RESULTS_DIR", "PluginInput", "PluginRunSpec", "main", "plugin_environ"]

#: Where a plugin run's output survives the job's workdir cleanup, keyed by run id.
PLUGIN_RESULTS_DIR = "strata-plugin-results"
#: Seconds a cancelled plugin gets between SIGTERM and SIGKILL.
TERMINATE_GRACE_SECONDS = 10.0
#: How much of a failed plugin's stderr the run record keeps (scrubbed).
STDERR_TAIL_CHARS = 2000
#: Weight formats a model input is never downloaded in: each can execute code when loaded.
PICKLE_WEIGHT_PATTERNS: tuple[str, ...] = ("*.bin", "*.pt", "*.pth", "*.pkl", "*.ckpt", "*.pickle")

# The environment a plugin process may see. Everything else (the spec, the secrets file's
# path, any credential the job inherited) stays with the runner.
_PLUGIN_ENV_NAMES = frozenset({"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "USER"})
_GPU_ENV_PREFIXES = ("CUDA_", "NVIDIA_")
_CREDENTIAL_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL")


class PluginInput(BaseModel):
    """One declared input, bound to a Hub repository at a full commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Annotated[str, StringConstraints(max_length=48)]
    repo_id: str
    revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    config: str | None = None
    split: str | None = None


class PluginRunSpec(BaseModel):
    """The inert spec for one plugin run. Secret-free by construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine_version: str | None = None
    run_id: str | None = None
    progress_path: str | None = None
    plugin: PluginPin
    #: The entry point as the manifest at the pinned commit declared it (read by the control
    #: plane as data; the installed build is verified to be that commit).
    entry: EntryPointDecl
    config: dict[str, ScalarValue] = Field(default_factory=dict)
    inputs: tuple[PluginInput, ...] = Field(default=(), max_length=16)
    output_repo_id: str
    batch_size: int = Field(default=32, ge=1, le=4096)


def plugin_environ(
    *, gpu: bool, hf_home: Path, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The environment a plugin process gets: an allow-list, no credential, the Hub offline on an
    empty cache (``hf_home``), so neither a variable nor a login cached on the machine reaches it."""
    source = os.environ if environ is None else environ
    env = {
        name: value
        for name, value in source.items()
        if (name in _PLUGIN_ENV_NAMES or (gpu and name.startswith(_GPU_ENV_PREFIXES)))
        and not any(word in name.upper() for word in _CREDENTIAL_WORDS)
    }
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HOME"] = str(hf_home)
    return env


def _allowed_pins() -> tuple[PluginPin, ...]:
    raw = os.environ.get(ALLOWED_PLUGINS_ENV)
    if not raw:
        msg = f"no plugin is allowed on this job ({ALLOWED_PLUGINS_ENV} is not set)"
        raise RunError(msg)
    try:
        data: object = json.loads(raw)
    except ValueError as exc:
        msg = f"{ALLOWED_PLUGINS_ENV} is not valid JSON"
        raise RunError(msg) from exc
    if not isinstance(data, list):
        msg = f"{ALLOWED_PLUGINS_ENV} must be a JSON list"
        raise RunError(msg)
    try:
        return tuple(PluginPin.model_validate(item) for item in cast("list[object]", data))
    except ValueError as exc:
        msg = f"{ALLOWED_PLUGINS_ENV} holds an invalid pin"
        raise RunError(msg) from exc


def admit(spec: PluginRunSpec) -> dict[str, ScalarValue]:
    """Every check that needs no download. Returns the configuration to hand the plugin."""
    if spec.plugin not in _allowed_pins():
        msg = (
            f"{spec.plugin.distribution} {spec.plugin.version} at "
            f"{spec.plugin.source.commit[:12]} is not an allowed plugin on this job"
        )
        raise RunError(msg)
    validate_repo_id(spec.output_repo_id, "output repo")
    request = PluginRequest(
        entry_point=spec.entry.name,
        inputs=tuple(
            InputBinding(
                role=i.role, repo_id=i.repo_id, revision=i.revision, config=i.config, split=i.split
            )
            for i in spec.inputs
        ),
        output_repo_id=spec.output_repo_id,
        config=dict(spec.config),
    )
    problems = check_bindings(spec.entry, request)
    config, config_problems = check_config(spec.entry, dict(spec.config))
    problems.extend(config_problems)
    if problems:
        raise RunError("the plugin request does not match its interface: " + "; ".join(problems))
    mismatch = installed_mismatch(spec.plugin)
    if mismatch is not None:
        raise RunError(mismatch)
    if registered_entry_point(spec.plugin, spec.entry.name) is None:
        msg = (
            f"{spec.plugin.distribution} {spec.plugin.version} registers no plugin entry point "
            f"{spec.entry.name!r}"
        )
        raise RunError(msg)
    return config


def _load_split(
    item: PluginInput, token: str | Literal[False], dest: Path, needed: tuple[str, ...]
) -> int:
    try:
        datasets_mod: Any = __import__("datasets")
    except ImportError as exc:
        msg = "the [hf] extra is required to read plugin inputs: pip install 'strata-forge[hf]'"
        raise RunError(msg) from exc
    validate_repo_id(item.repo_id, "dataset")
    dataset = datasets_mod.load_dataset(
        item.repo_id,
        name=item.config,
        split=item.split,
        revision=item.revision,
        token=token,
        streaming=False,
    )
    columns = set(dataset.column_names or [])
    missing = [c for c in needed if c not in columns]
    if missing:
        msg = f"the {item.role!r} input lacks the required columns {missing}"
        raise RunError(msg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(dest))
    return len(dataset)


async def _download_inputs(
    spec: PluginRunSpec, secrets: RunSecrets, workdir: Path
) -> tuple[dict[str, Path], dict[str, Path], dict[str, int]]:
    token = secrets.hub_credential()
    decls = {d.role: d for d in spec.entry.inputs}
    datasets: dict[str, Path] = {}
    models: dict[str, Path] = {}
    rows: dict[str, int] = {}
    hub = HFHubClient(token=token)
    for item in spec.inputs:
        decl = decls[item.role]
        if decl.kind == "dataset":
            dest = workdir / "inputs" / f"{item.role}.parquet"
            rows[item.role] = await asyncio.to_thread(
                _load_split, item, token, dest, decl.required_columns
            )
            datasets[item.role] = dest
        else:
            validate_repo_id(item.repo_id, "model")
            models[item.role] = await hub.download_snapshot(
                item.repo_id,
                local_dir=workdir / "models" / item.role,
                revision=item.revision,
                ignore_patterns=PICKLE_WEIGHT_PATTERNS,
            )
    return datasets, models, rows


async def _run_child(job_path: Path, *, gpu: bool, cwd: Path, hf_home: Path) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "strata_forge.plugins.host",
        str(job_path),
        cwd=str(cwd),
        env=plugin_environ(gpu=gpu, hf_home=hf_home),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _, stderr = await proc.communicate()
    except asyncio.CancelledError:
        await _stop(proc)
        raise
    return proc.returncode or 0, (stderr or b"").decode("utf-8", "replace")[-STDERR_TAIL_CHARS:]


async def _stop(proc: asyncio.subprocess.Process) -> None:
    """Stop the plugin's whole process group: SIGTERM, then SIGKILL after a grace period."""
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), TERMINATE_GRACE_SECONDS)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        await proc.wait()


def _check_output(spec: PluginRunSpec, outdir: Path, rows: dict[str, int]) -> dict[str, Any]:
    if spec.entry.type == "step_scorer":
        scores = outdir / SCORES_FILENAME
        if not scores.is_file():
            msg = "the step scorer wrote no scores"
            raise RunError(msg)
        pq: Any = __import__("pyarrow.parquet", fromlist=["parquet"])
        written = int(pq.ParquetFile(str(scores)).metadata.num_rows)
        expected = rows.get(STEP_INPUT_ROLE, 0)
        if written != expected:
            msg = f"the step scorer wrote {written} scores for {expected} steps"
            raise RunError(msg)
        return {"path": SCORES_FILENAME, "rows": written}
    files = [p for p in outdir.rglob("*") if p.is_file()]
    if not files:
        msg = "the plugin wrote nothing to its output directory"
        raise RunError(msg)
    return {"files": len(files)}


async def _execute(
    spec: PluginRunSpec, secrets: RunSecrets, writer: JsonlProgressWriter | None
) -> str:
    redactor = run_redactor(secrets.hf_token_value())
    started = datetime.now(UTC)
    config = admit(spec)
    emit(writer, ProgressEvent(kind="phase", stage="load_model", message="reading the inputs"))
    workdir = Path(tempfile.mkdtemp(prefix="strata-plugin-"))
    outdir = results_dir(spec.run_id, name=PLUGIN_RESULTS_DIR)
    if outdir == Path.cwd():
        outdir = workdir / "output"
    try:
        datasets, models, rows = await _download_inputs(spec, secrets, workdir)
        job = HostJob(
            pin=spec.plugin,
            entry_point=spec.entry.name,
            type=spec.entry.type,
            config=config,
            datasets=datasets,
            models=models,
            output_dir=outdir,
            progress_path=writer.path.as_posix() if writer is not None else None,
            batch_size=spec.batch_size,
        )
        job_path = workdir / "job.json"
        job_path.write_text(job.model_dump_json(), encoding="utf-8")
        hf_home = workdir / "hf-home"
        hf_home.mkdir()
        emit(writer, ProgressEvent(kind="start", stage="run", message=f"running {spec.entry.name}"))
        code, stderr = await _run_child(job_path, gpu=spec.entry.gpu, cwd=workdir, hf_home=hf_home)
        if code != 0:
            tail = redactor.redact(stderr.strip().splitlines()[-1] if stderr.strip() else "")
            msg = f"the plugin exited with code {code}" + (f": {tail}" if tail else "")
            raise RunError(msg)
        output = _check_output(spec, outdir, rows)
        write_manifest(
            {
                "schema": MANIFEST_SCHEMA,
                "kind": "plugin",
                "run_id": spec.run_id,
                "engine": manifest_engine(spec.engine_version, ()),
                "plugin": {
                    **spec.plugin.model_dump(),
                    "entry_point": spec.entry.name,
                    "type": spec.entry.type,
                },
                "config": config,
                "inputs": [
                    {
                        **i.model_dump(),
                        "rows": rows.get(i.role),
                    }
                    for i in spec.inputs
                ],
                "output": output,
                "timing": {
                    "started_at": started.isoformat(),
                    "finished_at": datetime.now(UTC).isoformat(),
                },
            },
            outdir,
        )
        emit(writer, ProgressEvent(kind="phase", stage="push", message="pushing the output"))
        repo_type = "model" if spec.entry.outputs[0].kind == "model" else "dataset"
        out_repo = validate_repo_id(spec.output_repo_id, "output repo")
        hub = HFHubClient(token=secrets.hf_token_value() or False)
        await hub.create_repo(out_repo, repo_type=repo_type, private=True, exist_ok=True)
        # An existing public repository is never written into: the output of a plugin run is
        # private until its owner decides otherwise.
        if not await hub.is_private(out_repo, repo_type=repo_type):
            msg = f"the output repository {out_repo} is public; plugin output is pushed only privately"
            raise RunError(msg)
        await hub.upload_folder(
            outdir,
            out_repo,
            repo_type=repo_type,
            commit_message=f"strata plugin run ({spec.entry.name})",
        )
        emit(writer, ProgressEvent(kind="end", message=f"pushed to {out_repo}"))
        return out_repo
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def main() -> int:
    async def execute(writer: JsonlProgressWriter | None, secrets: RunSecrets) -> None:
        spec = load_config(PluginRunSpec, writer=writer)
        if not COMMIT_SHA_RE.fullmatch(spec.plugin.source.commit):
            msg = "the plugin pin needs a full commit id"
            raise RunError(msg)
        await _execute(spec, secrets, writer)

    return await runner_main(execute)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main()))
