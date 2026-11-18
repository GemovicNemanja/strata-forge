"""Run a trusted plugin's entry point on the compute target.

Launched by an orchestrator as ``python -I -m strata_forge.pipelines.plugin_runner`` with the
inert spec in ``STRATA_RUN_CONFIG``, the run's Hub token in the backend's secrets file (never an
environment variable), and, from the orchestrator's OWN configuration, the allowed build in
:data:`~strata_forge.plugins.pins.ALLOWED_PLUGINS_ENV` (the operator's allow-list entry and the
entry points the manifest at the pinned commit declares). The order is fixed, and every check
happens before any plugin code runs:

1. the spec is parsed as data and the engine version handshake applied (``_common.load_config``);
   the secrets file was read into memory and unlinked before that (``runner_main``), and a token
   delivered in an environment variable is refused outright, since ``/proc/<pid>/environ`` keeps
   a process's starting environment for as long as it runs;
2. the pinned build must be the one the orchestrator allowed for this job, the entry point one
   the operator's entry allows, and its interface is the orchestrator's copy of the manifest,
   never the spec's; the bindings and configuration must fit it;
3. a sandbox is chosen (:mod:`~strata_forge.plugins.sandbox`): another uid, else bubblewrap; a
   machine with neither runs no plugin;
4. the plugin's own directory is built from the pin, or reused when it still verifies
   (:mod:`~strata_forge.plugins.environment`). No plugin code runs outside the sandbox while it
   is built: hash-pinned wheels are only unpacked, an ``hf`` wheel is checked against its pinned
   SHA-256 first, a git source is built INSIDE the sandbox, and the result is read from its files,
   never run; every RECORD hash and the pinned files digest are checked, and its distribution
   must register the entry point;
5. the declared inputs are downloaded HERE, with the run's credential, at their full commits:
   dataset splits to local parquet (each checked for its required columns), model inputs as
   snapshots without pickle-capable files;
6. the plugin runs on the engine's interpreter with ``-I`` in the sandbox, the engine bound
   read-only and the plugin's directory appended to its path
   (:mod:`~strata_forge.plugins.child`): an environment with no credential, the Hub offline on
   an empty cache, the operator's time and resource limits, progress relayed through a pipe the
   runner validates, the GPU sampled from THIS process meanwhile. However it ends, its whole
   process group is killed and confirmed gone before anything else happens;
7. its output is copied, without following any link and refusing hard links and special files,
   into a directory the runner owns, and only that copy is read from then on
   (:mod:`~strata_forge.plugins.output`): a step scorer's scores are checked against the input
   by the runner itself; the opt-in diagnostic aggregations are written beside them; a manifest
   records the build, the environment, the sandbox, the inputs and every file's hash; and the
   copy is pushed to the run's private output repository.

The plugin never holds the Hub credential, never chooses what it reads, and never pushes.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from strata_forge.pipelines._common import (
    COMMIT_SHA_RE,
    LEGACY_TOKEN_ENV,
    MANIFEST_SCHEMA,
    RunError,
    RunSecrets,
    emit,
    load_config,
    manifest_engine,
    phase_sink,
    run_redactor,
    runner_main,
    ticking_phase,
    validate_repo_id,
    write_manifest,
)
from strata_forge.plugins.child import (
    STOPPED_LIMIT_SECONDS,
    PluginStillRunningError,
    run_plugin_process,
)
from strata_forge.plugins.environment import BuildJob, PluginEnvironment, ensure_environment
from strata_forge.plugins.host import SCORES_FILENAME, HostJob, score_columns
from strata_forge.plugins.manifest import (
    STEP_INPUT_ROLE,
    EntryPointDecl,
    PluginManifestError,
    ScalarValue,
)
from strata_forge.plugins.output import OutputError, check_scores_file, stage_output
from strata_forge.plugins.pins import (
    ALLOWED_PLUGINS_ENV,
    PluginPin,
    RunnerAllowance,
    registered_entry_point,
)
from strata_forge.plugins.preflight import InputBinding, PluginRequest, check_bindings, check_config
from strata_forge.plugins.sandbox import (
    Sandbox,
    SandboxPaths,
    SandboxUnavailableError,
    chmod_for_uid,
    choose_sandbox,
    engine_paths,
    sandbox_command,
)
from strata_forge.plugins.step_scoring import Aggregation, aggregate
from strata_forge.storage.hf_hub import HFHubClient
from strata_forge.training.hardware import GpuSampler
from strata_forge.training.progress import JsonlProgressWriter, ProgressEvent

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "DIAGNOSTICS_FILENAME",
    "PICKLE_WEIGHT_PATTERNS",
    "PluginInput",
    "PluginRunSpec",
    "admit",
    "main",
    "plugin_environ",
]

#: Model-input files never downloaded: each format can execute code when it is loaded.
PICKLE_WEIGHT_PATTERNS: tuple[str, ...] = (
    "*.bin",
    "*.pt",
    "*.pth",
    "*.pkl",
    "*.pickle",
    "*.ckpt",
    "*.joblib",
    "*.dill",
    "*.npy",
    "*.npz",
    "*.keras",
    "*.h5",
    "*.hdf5",
)
#: Where a step-scoring run's opt-in per-item aggregations land: beside the scores, never in them.
DIAGNOSTICS_FILENAME = "aggregates.diagnostic.parquet"
#: How much of a failed plugin's standard error the run record keeps (scrubbed).
STDERR_TAIL_CHARS = 2000
#: The wall-clock limit on building a git source's wheel inside the sandbox.
BUILD_TIMEOUT_SECONDS = 1800

# The environment a plugin process may see. Everything else (the spec, the secrets file's
# path, any credential the job inherited, the runner's HOME) stays with the runner.
_PLUGIN_ENV_NAMES = frozenset({"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "USER"})
_GPU_ENV_PREFIXES = ("CUDA_", "NVIDIA_")
_CREDENTIAL_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")


class _StartingEnvironment:
    """Whether the token variable was in this process's starting environment (set by ``main``)."""

    token_present = False


class PluginInput(BaseModel):
    """One declared input, bound to a Hub repository at a full commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Annotated[str, StringConstraints(max_length=48)]
    repo_id: str
    revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    config: str | None = None
    split: str | None = None


class PluginRunSpec(BaseModel):
    """The inert spec for one plugin run. Secret-free by construction. It NAMES the pinned build
    and the entry point; what they are (the operator's limits, the declared interface) comes
    from the orchestrator's :data:`ALLOWED_PLUGINS_ENV`, never from here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine_version: str | None = None
    run_id: str | None = None
    progress_path: str | None = None
    plugin: PluginPin
    entry_point: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    config: dict[str, ScalarValue] = Field(default_factory=dict)
    inputs: tuple[PluginInput, ...] = Field(default=(), max_length=16)
    output_repo_id: str
    batch_size: int = Field(default=32, ge=1, le=4096)
    #: A step scorer's per-item reductions to write as a DIAGNOSTIC file of their own. Off unless
    #: asked for: forge picks no aggregation, and the scores never carry one.
    diagnostic_aggregations: tuple[Aggregation, ...] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def _aggregations(self) -> PluginRunSpec:
        if len(set(self.diagnostic_aggregations)) != len(self.diagnostic_aggregations):
            msg = "diagnostic_aggregations repeats an aggregation"
            raise ValueError(msg)
        return self


class Admitted(BaseModel):
    """What admission settled: the allowance, the entry point's interface, the configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowance: RunnerAllowance
    entry: EntryPointDecl
    config: dict[str, ScalarValue]


def plugin_environ(
    *, gpu: bool, home: Path, hf_home: Path, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The environment a plugin process gets: an allow-list, no credential, a private ``HOME``,
    the Hub offline on an empty cache (``hf_home``), so neither a variable nor a login cached on
    the machine reaches it. Inside a bubblewrap sandbox ``HOME`` is the sandbox's own ``/tmp``."""
    source = os.environ if environ is None else environ
    env = {
        name: value
        for name, value in source.items()
        if (name in _PLUGIN_ENV_NAMES or (gpu and name.startswith(_GPU_ENV_PREFIXES)))
        and not any(word in name.upper() for word in _CREDENTIAL_WORDS)
    }
    env["HOME"] = str(home)
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HOME"] = str(hf_home)
    return env


def _allowances() -> tuple[RunnerAllowance, ...]:
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
        return tuple(RunnerAllowance.model_validate(item) for item in cast("list[object]", data))
    except ValueError as exc:
        msg = f"{ALLOWED_PLUGINS_ENV} holds an invalid entry"
        raise RunError(msg) from exc


def admit(spec: PluginRunSpec) -> Admitted:
    """Every check that needs neither an install nor a download."""
    allowance = next((a for a in _allowances() if a.plugin.pin == spec.plugin), None)
    if allowance is None:
        msg = (
            f"{spec.plugin.distribution} {spec.plugin.version} at "
            f"{spec.plugin.source.commit[:12]} is not an allowed plugin on this job"
        )
        raise RunError(msg)
    entry = allowance.entry(spec.entry_point)
    if entry is None:
        msg = f"the operator does not allow the entry point {spec.entry_point!r} of this plugin"
        raise RunError(msg)
    if spec.diagnostic_aggregations and entry.type != "step_scorer":
        msg = "diagnostic_aggregations apply to a step_scorer only"
        raise RunError(msg)
    validate_repo_id(spec.output_repo_id, "output repo")
    request = PluginRequest(
        entry_point=entry.name,
        inputs=tuple(
            InputBinding(
                role=i.role, repo_id=i.repo_id, revision=i.revision, config=i.config, split=i.split
            )
            for i in spec.inputs
        ),
        output_repo_id=spec.output_repo_id,
        config=dict(spec.config),
    )
    problems = check_bindings(entry, request)
    config, config_problems = check_config(entry, dict(spec.config))
    problems.extend(config_problems)
    if problems:
        raise RunError("the plugin request does not match its interface: " + "; ".join(problems))
    return Admitted(allowance=allowance, entry=entry, config=config)


def _sandboxed_builder(sandbox: Sandbox, scratch: Path) -> Any:
    """Run a git source's build (its backend is the plugin's code) inside ``sandbox``: no
    network, the engine and the build requirements read-only, only the source tree and the
    output directory writable, killed and confirmed gone like the plugin's own process."""

    async def build(job: BuildJob) -> None:
        if sandbox.isolation == "uid":
            chmod_for_uid(readable=list(job.readonly), writable=list(job.writable))
        paths = SandboxPaths(readonly=(*engine_paths(), *job.readonly), writable=job.writable)
        command = sandbox_command(sandbox, job.command, paths=paths, env=job.env, gpu=False)
        try:
            result = await run_plugin_process(
                command,
                sandbox=sandbox,
                env={"PATH": job.env.get("PATH", "/usr/bin:/bin")},
                cwd=scratch,
                marker=str(job.source),
                timeout_s=float(BUILD_TIMEOUT_SECONDS),
                sink=lambda _event: None,
            )
        except PluginStillRunningError as exc:
            raise PluginManifestError(str(exc)) from exc
        if result.timed_out or result.stopped or result.code != 0:
            tail = result.stderr_tail.strip()[-STDERR_TAIL_CHARS:]
            why = (
                "ran past its time limit"
                if result.timed_out
                else "was stopped"
                if result.stopped
                else f"failed (exit {result.code})"
            )
            raise PluginManifestError(f"building the plugin {why}" + (f": {tail}" if tail else ""))

    return build


async def _prepare_environment(
    spec: PluginRunSpec,
    entry: EntryPointDecl,
    secrets: RunSecrets,
    workdir: Path,
    phase: Any,
    sandbox: Sandbox,
) -> PluginEnvironment:
    """The plugin's verified environment, and the entry point registered in it."""
    pin = spec.plugin
    scratch = workdir / "environment"
    scratch.mkdir()

    async def fetch_wheel(into: Path) -> Path:
        source = pin.source
        hub = HFHubClient(token=secrets.hub_credential())
        return await hub.download_file(
            source.repo,
            str(source.wheel),
            local_dir=into,
            repo_type="model",
            revision=source.commit,
        )

    try:
        env = await ensure_environment(
            pin,
            scratch=scratch,
            fetch_wheel=fetch_wheel,
            build=_sandboxed_builder(sandbox, scratch),
            phase=phase,
        )
    except PluginManifestError as exc:
        raise RunError(str(exc)) from exc
    if registered_entry_point(pin, entry.name, path=env.site_paths) is None:
        msg = f"{pin.distribution} {pin.version} registers no plugin entry point {entry.name!r}"
        raise RunError(msg)
    return env


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
    spec: PluginRunSpec, entry: EntryPointDecl, secrets: RunSecrets, inputs: Path
) -> tuple[dict[str, Path], dict[str, Path], dict[str, int]]:
    token = secrets.hub_credential()
    decls = {d.role: d for d in entry.inputs}
    datasets: dict[str, Path] = {}
    models: dict[str, Path] = {}
    rows: dict[str, int] = {}
    hub = HFHubClient(token=token)
    for item in spec.inputs:
        decl = decls[item.role]
        if decl.kind == "dataset":
            dest = inputs / "datasets" / f"{item.role}.parquet"
            needed = (*decl.required_columns, *entry.passthrough_columns)
            rows[item.role] = await asyncio.to_thread(_load_split, item, token, dest, needed)
            datasets[item.role] = dest
        else:
            validate_repo_id(item.repo_id, "model")
            models[item.role] = await hub.download_snapshot(
                item.repo_id,
                local_dir=inputs / "models" / item.role,
                revision=item.revision,
                ignore_patterns=PICKLE_WEIGHT_PATTERNS,
            )
    return datasets, models, rows


def _write_diagnostics(
    outdir: Path, rows: list[tuple[str, int, float | None]], how: tuple[Aggregation, ...]
) -> dict[str, Any] | None:
    """Write the opt-in per-item aggregations, labelled diagnostic, beside the scores."""
    if not how:
        return None
    pa: Any = __import__("pyarrow")
    pq: Any = __import__("pyarrow.parquet", fromlist=["parquet"])
    columns: dict[str, list[Any]] = {
        "item_id": [],
        "aggregation": [],
        "value": [],
        "scored_steps": [],
        "unscored_steps": [],
    }
    for name in how:
        for item_id, (value, n_scored, n_unscored) in aggregate(rows, name).items():
            columns["item_id"].append(item_id)
            columns["aggregation"].append(name)
            columns["value"].append(value)
            columns["scored_steps"].append(n_scored)
            columns["unscored_steps"].append(n_unscored)
    table = pa.table(
        {
            "item_id": pa.array(columns["item_id"], type=pa.string()),
            "aggregation": pa.array(columns["aggregation"], type=pa.string()),
            "value": pa.array(columns["value"], type=pa.float64()),
            "scored_steps": pa.array(columns["scored_steps"], type=pa.int64()),
            "unscored_steps": pa.array(columns["unscored_steps"], type=pa.int64()),
        }
    ).replace_schema_metadata({"strata.diagnostic": "true"})
    pq.write_table(table, str(outdir / DIAGNOSTICS_FILENAME))
    return {"path": DIAGNOSTICS_FILENAME, "diagnostic": True, "aggregations": list(how)}


def _check_staged(
    spec: PluginRunSpec, entry: EntryPointDecl, staged: Path, datasets: dict[str, Path]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The output record from the runner's own copy, and the diagnostics record if asked."""
    if entry.type != "step_scorer":
        count = sum(1 for p in staged.rglob("*") if p.is_file())
        if not count:
            msg = "the plugin wrote nothing to its output directory"
            raise RunError(msg)
        return {"count": count}, None
    scores = staged / SCORES_FILENAME
    if not scores.is_file():
        msg = "infrastructure fault: the step scorer wrote no scores"
        raise RunError(msg)
    steps = datasets[STEP_INPUT_ROLE]
    pq: Any = __import__("pyarrow.parquet", fromlist=["parquet"])
    keep = score_columns(entry, list(pq.ParquetFile(str(steps)).schema_arrow.names))
    try:
        record = check_scores_file(entry, scores, steps, keep)
    except OutputError as exc:
        raise RunError(str(exc)) from exc
    rows = record.pop("_rows")
    diagnostics = _write_diagnostics(staged, rows, spec.diagnostic_aggregations)
    return {"path": SCORES_FILENAME, "scoring": record}, diagnostics


def _remove(path: Path, sandbox: Sandbox | None) -> None:
    """Remove the work directory. Files the ``nobody`` uid wrote need ``sudo`` to remove."""
    shutil.rmtree(path, ignore_errors=True)
    if path.exists() and sandbox is not None and sandbox.isolation == "uid":
        sudo = shutil.which("sudo") or "sudo"
        subprocess.run(  # noqa: S603 -- a fixed command on the runner's own temporary directory
            [sudo, "-n", "rm", "-rf", "--", str(path)], check=False, timeout=120
        )


async def _execute(
    spec: PluginRunSpec, secrets: RunSecrets, writer: JsonlProgressWriter | None
) -> str:
    if _StartingEnvironment.token_present or secrets.from_environment:
        msg = (
            f"the plugin runner takes the token only from the secrets file, never from "
            f"{LEGACY_TOKEN_ENV}: a process's starting environment stays readable while it runs"
        )
        raise RunError(msg)
    redactor = run_redactor(secrets.hf_token_value())
    started = datetime.now(UTC)
    admitted = admit(spec)
    entry, allowance = admitted.entry, admitted.allowance
    try:
        sandbox = await choose_sandbox()
    except SandboxUnavailableError as exc:
        raise RunError(str(exc)) from exc
    # Sampled HERE, outside the plugin's process, so the GPU numbers are forge's own reading and
    # never something the plugin reported.
    gpu = GpuSampler()
    phase = phase_sink(writer, secrets.hf_token_value(), gpu=gpu)
    workdir = Path(tempfile.mkdtemp(prefix="strata-plugin-"))
    try:
        async with ticking_phase(phase, "preparing the plugin environment", stage="load_model"):
            env = await _prepare_environment(spec, entry, secrets, workdir, phase, sandbox)
        inputs = workdir / "inputs"
        async with ticking_phase(phase, "reading the inputs", stage="load_model"):
            datasets, models, rows = await _download_inputs(spec, entry, secrets, inputs)
        inputs.mkdir(exist_ok=True)
        output = workdir / "output"
        output.mkdir()
        job = HostJob(
            pin=spec.plugin,
            plugin_paths=(env.site,),
            entry=entry,
            config=admitted.config,
            datasets=datasets,
            models=models,
            output_dir=output,
            batch_size=spec.batch_size,
            limits=allowance.plugin.limits,
        )
        job_path = workdir / "job.json"
        job_path.write_text(job.model_dump_json(), encoding="utf-8")
        # The engine read-only (what runs the host), the plugin's directory, the job and the
        # inputs read-only; only the output is writable.
        paths = SandboxPaths(
            readonly=(*engine_paths(), env.site, job_path, inputs),
            writable=(output,),
        )
        if sandbox.isolation == "uid":
            chmod_for_uid(readable=[job_path, inputs], writable=[output])
        home = workdir / "home"
        home.mkdir()
        hf_home = home / "hf"
        hf_home.mkdir()
        child_env = plugin_environ(gpu=entry.gpu, home=home, hf_home=hf_home)
        inner = [sys.executable, "-I", "-m", "strata_forge.plugins.host", str(job_path)]
        command = sandbox_command(sandbox, inner, paths=paths, env=child_env, gpu=entry.gpu)
        # The sandbox clears the environment and sets the child's inside.
        spawn_env = {"PATH": child_env.get("PATH", "/usr/bin:/bin")}

        def relay(event: ProgressEvent) -> None:
            if event.kind == "phase":
                phase(event.message, stage="run")
            else:
                emit(
                    writer, event.model_copy(update={"metrics": {**event.metrics, **gpu.sample()}})
                )

        emit(writer, ProgressEvent(kind="start", stage="run", message=f"running {entry.name}"))
        async with ticking_phase(phase, f"running {entry.name}", stage="run"):
            try:
                result = await run_plugin_process(
                    command,
                    sandbox=sandbox,
                    env=spawn_env,
                    cwd=workdir,
                    marker=str(job_path),
                    timeout_s=float(allowance.plugin.limits.max_runtime_seconds),
                    sink=relay,
                )
            except PluginStillRunningError as exc:
                raise RunError(str(exc)) from exc
        # From here on the plugin's processes are gone.
        if result.timed_out:
            msg = (
                f"the plugin ran past its limit of {allowance.plugin.limits.max_runtime_seconds} "
                "seconds and was stopped; nothing is pushed"
            )
            raise RunError(msg)
        if result.stopped:
            msg = (
                "a plugin process stayed stopped (traced or suspended) for over "
                f"{STOPPED_LIMIT_SECONDS:.0f} seconds, so the run was ended; nothing is pushed"
            )
            raise RunError(msg)
        if result.code != 0:
            tail = result.stderr_tail.strip()
            last = redactor.redact(tail.splitlines()[-1])[:STDERR_TAIL_CHARS] if tail else ""
            msg = f"the plugin exited with code {result.code}" + (f": {last}" if last else "")
            raise RunError(msg)
        staged = workdir / "staged"
        try:
            hashes = stage_output(output, staged)
        except OutputError as exc:
            raise RunError(str(exc)) from exc
        record, diagnostics = _check_staged(spec, entry, staged, datasets)
        write_manifest(
            {
                "schema": MANIFEST_SCHEMA,
                "kind": "plugin",
                "run_id": spec.run_id,
                "engine": manifest_engine(spec.engine_version, ()),
                "plugin": {
                    **spec.plugin.model_dump(mode="json"),
                    "id": allowance.plugin.id,
                    "repo": spec.plugin.source.repo,
                    "commit": spec.plugin.source.commit,
                    "entry_point": entry.name,
                    "type": entry.type,
                    "score_kind": entry.score_kind,
                },
                "environment": env.record(),
                "sandbox": {**sandbox.record(), "limits": allowance.plugin.limits.model_dump()},
                "config": admitted.config,
                "inputs": [
                    {**i.model_dump(), "commit": i.revision, "rows": rows.get(i.role)}
                    for i in spec.inputs
                ],
                "output": {**record, "files": hashes},
                **({"diagnostics": diagnostics} if diagnostics is not None else {}),
                "timing": {
                    "started_at": started.isoformat(),
                    "finished_at": datetime.now(UTC).isoformat(),
                },
            },
            staged,
        )
        repo_type = "model" if entry.outputs[0].kind == "model" else "dataset"
        out_repo = validate_repo_id(spec.output_repo_id, "output repo")
        async with ticking_phase(phase, "pushing the output", stage="push"):
            hub = HFHubClient(token=secrets.hf_token_value() or False)
            await hub.create_repo(out_repo, repo_type=repo_type, private=True, exist_ok=True)
            # An existing public repository is never written into: the output of a plugin run is
            # private until its owner decides otherwise.
            if not await hub.is_private(out_repo, repo_type=repo_type):
                msg = (
                    f"the output repository {out_repo} is public; plugin output is pushed only "
                    "privately"
                )
                raise RunError(msg)
            # Exactly the staged files, in ONE commit: what the manifest attests is what lands.
            files = {
                str(path.relative_to(staged)): path
                for path in sorted(staged.rglob("*"))
                if path.is_file()
            }
            commit = await hub.upload_files(
                files,
                out_repo,
                repo_type=repo_type,
                commit_message=f"strata plugin run ({entry.name})",
            )
        # ``repo@commit`` is the destination the control plane records, so the run's output is
        # read back at the commit this run made, never at a head a later push has moved.
        destination = f"{out_repo}@{commit}" if COMMIT_SHA_RE.fullmatch(commit) else out_repo
        emit(writer, ProgressEvent(kind="end", stage="push", message=destination))
        return destination
    finally:
        _remove(workdir, sandbox)


async def main() -> int:
    _StartingEnvironment.token_present = LEGACY_TOKEN_ENV in os.environ

    async def execute(writer: JsonlProgressWriter | None, secrets: RunSecrets) -> None:
        spec = load_config(PluginRunSpec, writer=writer)
        if not COMMIT_SHA_RE.fullmatch(spec.plugin.source.commit):
            msg = "the plugin pin needs a full commit id"
            raise RunError(msg)
        await _execute(spec, secrets, writer)

    return await runner_main(execute)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main()))
