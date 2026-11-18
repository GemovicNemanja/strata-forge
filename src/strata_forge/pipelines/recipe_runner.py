"""Dataset recipe runner: execute a recipe revision on the user's machine and push its output.

Spec: ``RecipeRunSpec`` (inert JSON in ``STRATA_RUN_CONFIG``). It carries the recipe itself, its
revision number and content hash (checked here: a spec whose recipe does not hash to the declared
revision is refused), a ``mode`` and the output repo.

- ``preview`` writes NOTHING to the Hub and needs no write token: it reads at most
  ``preview_source_rows`` rows of each source, bounds every ``synthesize`` step to
  ``preview_rows`` seed rows, and leaves only the summary in the workdir, with the counts and a
  capped sample of at most ``SAMPLE_ROWS`` rows per split (each text cut to ``SAMPLE_CELL_CHARS``,
  the whole sample under ``SAMPLE_BYTES``) for the control plane to show for review. Review
  candidates, not a version, and not a branch anyone can read later.
- ``materialize`` runs it in full and makes the result the default branch's content in ONE commit
  (the recipe's files replace the previous version's; nothing else in the repo is touched). That
  commit, normalized to its 40-character id, is in ``receipt.json`` and the summary, and is what
  the control plane records as the new dataset version.

Only the sources a step reads are read, each at its pinned commit with the run's Hub credential,
streamed and bounded: at most ``row_cap + 1`` rows (so the engine refuses a larger split) and
``byte_cap`` bytes of values, a row slice applied while streaming. A ``synthesize`` step's
model is downloaded at its pinned revision and served on loopback with vLLM, exactly as a batch
run serves a chat model; each request carries the step's JSON schema as guided JSON and its own
seed, and every reply is validated against the schema (an invalid one is excluded, not repaired).
Generated rows are marked ``_synthetic``. The engine (:mod:`strata_forge.datasets.recipe`)
refuses generation from any split not classified as training, and requires a recipe reading one to
name its outputs as held out; the summary, manifest and receipt record which read sources were not
training splits (``holdout_sources``) and which output splits are held out.

Outputs, in the one commit: ``data/<split>-00000-of-00001.parquet`` per split, ``exclusions.parquet``
(row id, step, reason), ``recipe.json`` (the revision), ``manifest.json`` (``kind``
``dataset_recipe``: every source pin, every generation pin and seed, the counts) and ``README.md``
(the dataset card's split layout). A copy of the summary (counts, column names, exclusion reasons;
no row content after a materialization) is left in the workdir as ``recipe-summary.json`` for the
control plane, beside ``receipt.json``.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictInt, model_validator

from strata_forge.compute import LocalBackend
from strata_forge.compute.serving import build_vllm_task, serving_endpoint
from strata_forge.datasets.recipe import (
    DEFAULT_BYTE_CAP,
    MAX_BYTE_CAP,
    GenerationRequest,
    GenerationResult,
    Recipe,
    RecipeError,
    RecipeOutput,
    Source,
    SynthesizeStep,
    execute,
    profile_table,
    recipe_sha256,
    split_class,
)
from strata_forge.llm import LLMClient, UserMessage
from strata_forge.llm.providers.config import OpenAICompatConfig
from strata_forge.llm.providers.openai_compat import (
    UNAUTHENTICATED_API_KEY,
    OpenAICompatProvider,
)
from strata_forge.pipelines._common import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    RunError,
    build_receipt,
    commit_oid,
    emit,
    ensure_private_output,
    load_config,
    manifest_engine,
    model_server_environ,
    phase_sink,
    results_dir,
    runner_main,
    snapshot_commit,
    ticking_phase,
    validate_repo_id,
    write_manifest,
    write_receipt,
)
from strata_forge.pipelines._guided import OutputValidator, schema_digest
from strata_forge.pipelines._split_overlap import parse_single_split
from strata_forge.pipelines.inference_runner import SNAPSHOT_IGNORED, SNAPSHOT_PATTERNS
from strata_forge.storage import HFHubClient
from strata_forge.training.progress import JsonlProgressWriter, ProgressEvent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from strata_forge.pipelines._common import PhaseSink, RunSecrets

__all__ = [
    "MANAGED_PATHS",
    "SAMPLE_BYTES",
    "SAMPLE_CELL_CHARS",
    "SAMPLE_ROWS",
    "SUMMARY_FORMAT",
    "WORKDIR_SUMMARY_FILENAME",
    "RecipeRunSpec",
    "Serving",
    "load_spec",
    "main",
]

SUMMARY_FORMAT = "strata.recipe.summary/v1"
WORKDIR_SUMMARY_FILENAME = "recipe-summary.json"
_MANIFEST_KIND = "dataset_recipe"
_RESULTS_DIR_NAME = "strata-recipe-results"
#: The paths a recipe's commit owns in its repo; anything else there is left alone.
MANAGED_PATHS = ("data/", "exclusions.parquet", "recipe.json", MANIFEST_FILENAME, "README.md")
_SERVE_HOST = "127.0.0.1"
_SERVE_PORT = 8000
_MAX_SUMMARY_COLUMNS = 512
#: A preview's sample: rows per split, characters per text value, bytes for the whole sample.
SAMPLE_ROWS = 50
SAMPLE_CELL_CHARS = 1000
SAMPLE_BYTES = 96 * 1024


class Serving(BaseModel):
    """How a ``synthesize`` step's model is served on the machine (shared by every such step)."""

    model_config = ConfigDict(extra="forbid")

    tensor_parallel_size: StrictInt = Field(default=1, ge=1, le=8)
    max_model_len: StrictInt | None = Field(default=None, ge=256, le=1_048_576)
    dtype: Literal["auto", "bfloat16", "float16"] = "auto"
    gpu_memory_utilization: float | None = Field(default=None, gt=0.05, lt=1.0)
    wait_timeout_s: float = Field(default=900.0, ge=30, le=3600)
    # Requests in flight at once against the model server.
    concurrency: StrictInt = Field(default=16, ge=1, le=128)


class RecipeRunSpec(BaseModel):
    """The inert recipe-run spec the control plane delivers in ``STRATA_RUN_CONFIG`` (no secrets)."""

    model_config = ConfigDict(extra="forbid")

    engine_version: str | None = None
    recipe: Recipe
    recipe_revision: StrictInt = Field(ge=1)
    recipe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["preview", "materialize"]
    preview_rows: StrictInt = Field(default=200, ge=1, le=5000)
    # A preview reads at most this many rows of each source (a materialization reads them all).
    preview_source_rows: StrictInt = Field(default=10_000, ge=1, le=100_000)
    row_cap: StrictInt = Field(default=1_000_000, ge=1, le=2_000_000)
    byte_cap: StrictInt = Field(default=DEFAULT_BYTE_CAP, ge=1, le=MAX_BYTE_CAP)
    # Where a materialization commits; a preview writes nowhere and names none.
    output_repo_id: str | None = None
    serving: Serving = Field(default_factory=Serving)
    progress_path: str | None = None
    run_id: str | None = None

    @model_validator(mode="after")
    def _the_recipe_is_the_declared_revision(self) -> RecipeRunSpec:
        if recipe_sha256(self.recipe) != self.recipe_sha256:
            msg = "the recipe does not hash to recipe_sha256: it is not the declared revision"
            raise ValueError(msg)
        if self.mode == "preview" and self.output_repo_id is not None:
            msg = "a preview writes nothing to the Hub, so it takes no output_repo_id"
            raise ValueError(msg)
        if self.mode == "materialize" and not self.output_repo_id:
            msg = "a materialization needs output_repo_id"
            raise ValueError(msg)
        return self


def load_spec(*, writer: JsonlProgressWriter | None = None) -> RecipeRunSpec:
    spec = load_config(RecipeRunSpec, writer=writer)
    if spec.output_repo_id is not None:
        validate_repo_id(spec.output_repo_id, "output repo")
    for source in spec.recipe.sources:
        validate_repo_id(source.hf_repo_id, "source dataset")
    for step in spec.recipe.steps:
        if isinstance(step, SynthesizeStep):
            validate_repo_id(step.model_id, "model")
    return spec


def _bounded(spec: RecipeRunSpec) -> Recipe:
    """The recipe a run executes: a preview bounds every synthesize step to ``preview_rows``."""
    if spec.mode != "preview":
        return spec.recipe
    steps = [
        step.model_copy(update={"max_seed_rows": min(step.max_seed_rows, spec.preview_rows)})
        if isinstance(step, SynthesizeStep)
        else step
        for step in spec.recipe.steps
    ]
    return spec.recipe.model_copy(update={"steps": tuple(steps)})


# ------------------------------ reading sources ------------------------------


def _value_bytes(value: Any) -> int:
    if isinstance(value, str | bytes | bytearray):
        return 8 + len(value)
    if isinstance(value, dict):
        mapping = cast("dict[Any, Any]", value)
        return 8 + sum(_value_bytes(k) + _value_bytes(v) for k, v in mapping.items())
    if isinstance(value, list | tuple):
        return 8 + sum(_value_bytes(v) for v in cast("list[Any]", value))
    return 8


def _read_source(
    source: Source, token: str | Literal[False], limit: int, byte_cap: int
) -> list[dict[str, Any]]:
    """At most ``limit`` rows of the pinned split, streamed (never the whole split in memory).

    The split's row slice is applied while streaming; reading stops at ``limit`` rows, and a read
    past ``byte_cap`` bytes of values is refused rather than held.
    """
    parsed = parse_single_split(source.split)
    if parsed is None:  # the recipe model already refused it
        msg = f"source {source.name!r} has an unreadable split"
        raise RunError(msg)
    try:
        datasets_mod: Any = __import__("datasets")
    except ImportError as exc:
        msg = "the [hf] extra is required to read datasets: pip install 'strata-forge[hf]'"
        raise RunError(msg) from exc
    try:
        dataset: Any = datasets_mod.load_dataset(
            source.hf_repo_id,
            name=source.config,
            split=parsed.name,
            revision=source.revision,
            token=token,
            streaming=True,
        )
    except Exception as exc:
        msg = f"could not read source {source.name!r} ({source.hf_repo_id}@{source.revision[:7]}): {exc}"
        raise RunError(msg) from exc
    start = parsed.start or 0
    stop = start + limit if parsed.end is None else min(parsed.end, start + limit)
    rows: list[dict[str, Any]] = []
    wanted: list[str] | None = list(source.columns) if source.columns is not None else None
    held = 0
    for row in itertools.islice(dataset, start, stop):
        record = cast("dict[str, Any]", row)
        if wanted is None:
            wanted = list(record)
        missing = [c for c in wanted if c not in record]
        if missing:
            msg = f"source {source.name!r} has no columns {missing}"
            raise RunError(msg)
        values = {c: record[c] for c in wanted}
        held += _value_bytes(values)
        if held > byte_cap:
            msg = f"source {source.name!r} holds more than {byte_cap // 1024**2} MiB of values"
            raise RunError(msg)
        rows.append(values)
    return rows


# ------------------------------ generation ------------------------------


class _ServedGenerator:
    """The engine's :class:`~strata_forge.datasets.recipe.Generator`, backed by a served model.

    Called from the worker thread the engine runs in; each step's work is scheduled onto the
    runner's event loop, where the model server is started for the step and stopped after it.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        token: str | Literal[False],
        serving: Serving,
        phase: PhaseSink,
        writer: JsonlProgressWriter | None,
    ) -> None:
        self._loop = loop
        self._token: str | Literal[False] = token
        self._serving = serving
        self._phase = phase
        self._writer = writer
        self.facts: list[dict[str, Any]] = []

    def __call__(
        self, step: SynthesizeStep, requests: Sequence[GenerationRequest]
    ) -> Sequence[GenerationResult]:
        future = asyncio.run_coroutine_threadsafe(self._generate(step, list(requests)), self._loop)
        return future.result()

    async def _download(self, step: SynthesizeStep) -> Path:
        hub = HFHubClient(token=self._token)  # explicit: never the VM's own login
        try:
            snapshot = await hub.download_snapshot(
                step.model_id,
                revision=step.model_revision,
                allow_patterns=SNAPSHOT_PATTERNS,
                ignore_patterns=SNAPSHOT_IGNORED,
            )
        except Exception as exc:
            msg = f"could not download the model {step.model_id}@{step.model_revision[:7]}: {exc}"
            raise RunError(msg) from exc
        commit = snapshot_commit(snapshot)
        if commit is not None and commit != step.model_revision:
            msg = f"the model {step.model_id} loaded at {commit}, not its pin {step.model_revision}"
            raise RunError(msg)
        return snapshot

    async def _generate(
        self, step: SynthesizeStep, requests: list[GenerationRequest]
    ) -> list[GenerationResult]:
        if not requests:
            return []
        serving = self._serving
        async with ticking_phase(self._phase, f"Downloading {step.model_id}", stage="load_model"):
            snapshot = await self._download(step)
        task = build_vllm_task(
            str(snapshot),
            served_model_name=step.model_id,
            port=_SERVE_PORT,
            host=_SERVE_HOST,
            tensor_parallel_size=serving.tensor_parallel_size,
            max_model_len=serving.max_model_len,
            dtype=serving.dtype,
            setup="",
            python_executable=sys.executable,
            extra_args=(
                ("--gpu-memory-utilization", str(serving.gpu_memory_utilization))
                if serving.gpu_memory_utilization is not None
                else ()
            ),
        )
        task = task.model_copy(
            update={
                "env": {
                    **model_server_environ(),
                    **task.env,
                    "PYTHONUNBUFFERED": "1",
                    "VLLM_USE_FLASHINFER_SAMPLER": "0",
                    "HF_HUB_OFFLINE": "1",
                }
            }
        )
        validator = OutputValidator(step.output_schema)
        outcome = {"valid": 0, "invalid": 0, "failed": 0}
        done = 0
        async with AsyncExitStack() as stack:
            endpoint = await stack.enter_async_context(
                serving_endpoint(
                    LocalBackend(log_dir=Path.cwd(), env_inherit=False),
                    task,
                    base_url=f"http://{_SERVE_HOST}:{_SERVE_PORT}/v1",
                    wait_timeout_s=serving.wait_timeout_s,
                    on_phase=lambda message: self._phase(message, stage="load_model"),
                )
            )
            client = LLMClient(
                model=step.model_id,
                provider="openai_compat",
                provider_clients={
                    "openai_compat": OpenAICompatProvider(
                        OpenAICompatConfig(
                            base_url=endpoint.base_url,
                            api_key=SecretStr(UNAUTHENTICATED_API_KEY),
                        )
                    )
                },
            )
            gate = asyncio.Semaphore(serving.concurrency)

            async def _one(request: GenerationRequest) -> GenerationResult:
                nonlocal done
                async with gate:
                    try:
                        response = await client.complete(
                            [UserMessage(content=request.prompt)],
                            temperature=step.temperature,
                            max_tokens=step.max_tokens,
                            provider_extras={
                                "openai_compat": {
                                    "seed": request.seed,
                                    "extra_body": {
                                        "structured_outputs": {"json": step.output_schema}
                                    },
                                }
                            },
                        )
                    except Exception:
                        outcome["failed"] += 1
                        result = GenerationResult(text=None, error="generation_failed")
                    else:
                        if validator.error(response.text) is None:
                            outcome["valid"] += 1
                            result = GenerationResult(text=response.text)
                        else:
                            outcome["invalid"] += 1
                            result = GenerationResult(text=None, error="invalid_output")
                    done += 1
                    if done % 25 == 0 or done == len(requests):
                        emit(
                            self._writer,
                            ProgressEvent(
                                kind="step",
                                stage="run",
                                step=done,
                                total_steps=len(requests),
                                metrics={k: float(v) for k, v in outcome.items()},
                            ),
                        )
                    return result

            async with ticking_phase(self._phase, f"Generating with {step.model_id}", stage="run"):
                results = await asyncio.gather(*(_one(r) for r in requests))
        self.facts.append(
            {
                "model_id": step.model_id,
                "model_revision": step.model_revision,
                "seed": step.seed,
                "n": step.n,
                "temperature": step.temperature,
                "max_tokens": step.max_tokens,
                "max_seed_rows": step.max_seed_rows,
                "prompt_template_sha256": hashlib.sha256(step.prompt_template.encode()).hexdigest(),
                "schema_sha256": schema_digest(step.output_schema),
                "requests": len(requests),
                **outcome,
            }
        )
        return list(results)


# ------------------------------ outputs ------------------------------


def _write_parquet(records: list[dict[str, Any]], columns: list[str], path: Path) -> Path:
    datasets_mod: Any = __import__("datasets")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        datasets_mod.Dataset.from_list(records)
        if records
        else datasets_mod.Dataset.from_dict({c: [] for c in columns})
    )
    data.to_parquet(str(path))
    return path


def _card(output: RecipeOutput, recipe: Recipe, revision: int) -> str:
    lines = ["---", "configs:", "- config_name: default", "  data_files:"]
    for split, table in output.splits.items():
        if table.rows:
            lines += [f"  - split: {split}", f"    path: data/{split}-*.parquet"]
    if output.synthetic_rows:
        lines += ["tags:", "- synthetic"]
    lines += [
        "---",
        "",
        f"# {recipe.name}",
        "",
        f"Made by a Strata dataset recipe (revision {revision}); `recipe.json` is the recipe and "
        "`manifest.json` records every source pin and generation setting.",
        "",
        "Rows with `_synthetic` true were generated by a model and have not been reviewed by "
        "this commit; `_seed_row_id` names the row each was generated from."
        if output.synthetic_rows
        else "No row was generated.",
        "",
    ]
    return "\n".join(lines)


def _summary(
    spec: RecipeRunSpec,
    output: RecipeOutput,
    generation: list[dict[str, Any]],
    *,
    commit: str | None,
    written: dict[str, int],
    truncated: list[str] | None = None,
) -> dict[str, Any]:
    """What the control plane stores: counts, column names and reasons, and for a preview the
    capped sample (:func:`_sample`); a materialization's summary holds no row value."""
    reasons: dict[str, int] = {}
    for exclusion in output.exclusions:
        reason = exclusion.reason.split(":", 1)[0]
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "format": SUMMARY_FORMAT,
        "mode": spec.mode,
        "recipe_revision": spec.recipe_revision,
        "recipe_sha256": spec.recipe_sha256,
        "output_repo_id": spec.output_repo_id,
        "commit": commit_oid(commit),
        "holdout_sources": list(spec.recipe.holdout_sources),
        "truncated_sources": truncated or [],
        "splits": {
            name: {
                "rows": len(table.rows),
                "written": written.get(name, 0),
                "synthetic": sum(1 for r in table.rows if r.synthetic),
                "held_out": split_class(name) == "held_out",
                "path": f"data/{name}-00000-of-00001.parquet" if written.get(name) else None,
                "columns": table.columns[:_MAX_SUMMARY_COLUMNS],
            }
            for name, table in output.splits.items()
        },
        "rows": output.rows,
        "synthetic_rows": output.synthetic_rows,
        "excluded": len(output.exclusions),
        "exclusions_by_reason": reasons,
        "counts": [
            {
                "step": c.step,
                "kind": c.kind,
                "rows_in": c.rows_in,
                "rows_out": c.rows_out,
                "excluded": c.excluded,
                "generated": c.generated,
            }
            for c in output.counts
        ],
        "profile": {
            name: [
                {k: p[k] for k in ("column", "rows", "nulls", "distinct")}
                for p in profile_table(table, top=0)[:_MAX_SUMMARY_COLUMNS]
            ]
            for name, table in output.splits.items()
        },
        "generation": [
            {
                k: g[k]
                for k in ("model_id", "model_revision", "requests", "valid", "invalid", "failed")
            }
            for g in generation
        ],
    }


def _cell(value: Any) -> Any:
    """One sample value: scalars as they are, text cut to ``SAMPLE_CELL_CHARS``, anything else as
    its JSON text, cut the same way."""
    if value is None or isinstance(value, bool | int | float):
        return value
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= SAMPLE_CELL_CHARS else text[: SAMPLE_CELL_CHARS - 1] + "\u2026"


def _sample(output: RecipeOutput, rows: int) -> dict[str, list[dict[str, Any]]]:
    """A preview's sample: the first rows of each split, bounded per value and in total."""
    sample: dict[str, list[dict[str, Any]]] = {name: [] for name in output.splits}
    used = 0
    for name, table in output.splits.items():
        for record in table.records()[: min(rows, SAMPLE_ROWS)]:
            row = {k: _cell(v) for k, v in record.items()}
            size = len(json.dumps(row, ensure_ascii=False, default=str))
            if used + size > SAMPLE_BYTES:
                return sample
            sample[name].append(row)
            used += size
    return sample


def _manifest(
    spec: RecipeRunSpec,
    output: RecipeOutput,
    generation: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    recipe = spec.recipe
    return {
        "schema": MANIFEST_SCHEMA,
        "kind": _MANIFEST_KIND,
        "engine": manifest_engine(spec.engine_version, ()),
        "mode": spec.mode,
        "recipe": {
            "name": recipe.name,
            "schema": recipe.canonical()["schema"],
            "revision": spec.recipe_revision,
            "sha256": spec.recipe_sha256,
        },
        "sources": [
            {
                "name": s.name,
                "hf_repo_id": s.hf_repo_id,
                "revision": s.revision,
                "config": s.config,
                "split": s.split,
                "columns": list(s.columns) if s.columns is not None else None,
                "dataset_version": s.dataset_version,
            }
            for s in recipe.sources
        ],
        "generation": generation,
        "row_cap": spec.row_cap,
        "holdout_sources": list(recipe.holdout_sources),
        "held_out_splits": [name for name in output.splits if split_class(name) == "held_out"],
        "preview_rows": spec.preview_rows if spec.mode == "preview" else None,
        "counts": summary["counts"],
        "splits": {name: s["rows"] for name, s in summary["splits"].items()},
        "synthetic_rows": output.synthetic_rows,
        "exclusions_by_reason": summary["exclusions_by_reason"],
        "files": {
            "data": [s["path"] for s in summary["splits"].values() if s["path"]],
            "exclusions": "exclusions.parquet",
            "recipe": "recipe.json",
        },
    }


def _write_outputs(
    spec: RecipeRunSpec,
    output: RecipeOutput,
    outdir: Path,
) -> tuple[dict[str, Path], dict[str, int]]:
    files: dict[str, Path] = {}
    written: dict[str, int] = {}
    for name, table in output.splits.items():
        records = table.records()
        written[name] = len(records)
        if not records:
            # An empty split has no file: the Hub's loaders refuse a split with no rows, so the
            # card leaves it out and the summary records it with zero rows.
            continue
        relative = f"data/{name}-00000-of-00001.parquet"
        files[relative] = _write_parquet(
            records, [*table.columns, "_row_id", "_synthetic", "_seed_row_id"], outdir / relative
        )
    files["exclusions.parquet"] = _write_parquet(
        [{"row_id": e.row_id, "step": e.step, "reason": e.reason} for e in output.exclusions],
        ["row_id", "step", "reason"],
        outdir / "exclusions.parquet",
    )
    recipe_path = outdir / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "revision": spec.recipe_revision,
                "sha256": spec.recipe_sha256,
                "recipe": spec.recipe.canonical(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    files["recipe.json"] = recipe_path
    return files, written


# ------------------------------ execution ------------------------------


async def _read_sources(
    spec: RecipeRunSpec, recipe: Recipe, token: str | Literal[False]
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Every source a step reads; a preview reads at most ``preview_source_rows`` of each."""
    cache: dict[str, list[dict[str, Any]]] = {}
    truncated: list[str] = []
    preview = spec.mode == "preview"
    for name in recipe.used_sources:
        source = recipe.source(name)
        limit = spec.preview_source_rows if preview else spec.row_cap
        rows = await asyncio.to_thread(_read_source, source, token, limit + 1, spec.byte_cap)
        if preview and len(rows) > limit:
            rows = rows[:limit]
            truncated.append(name)
        cache[name] = rows
    return cache, truncated


async def _execute(
    spec: RecipeRunSpec, secrets: RunSecrets, writer: JsonlProgressWriter | None
) -> str:
    hf_token = secrets.hf_token_value()
    hub_token = secrets.hub_credential()
    preview = spec.mode == "preview"
    if not preview and not hf_token:
        msg = "a recipe run pushes its output, so it needs the account's Hugging Face write token"
        raise RunError(msg)
    phase = phase_sink(writer, hf_token or (hub_token or None))
    hub: HFHubClient | None = None
    out_repo: str | None = None
    if not preview:
        out_repo = validate_repo_id(spec.output_repo_id or "", "output repo")
        hub = HFHubClient(token=hf_token)  # EXPLICIT write token, never the ambient HF_TOKEN
        async with ticking_phase(phase, "Checking the output repo", stage="load_model"):
            await ensure_private_output(hub, out_repo)
    recipe = _bounded(spec)
    async with ticking_phase(phase, "Reading the pinned sources", stage="load_model"):
        cache, truncated = await _read_sources(spec, recipe, hub_token)
    generator = (
        _ServedGenerator(
            asyncio.get_running_loop(),
            token=hub_token,
            serving=spec.serving,
            phase=phase,
            writer=writer,
        )
        if recipe.generates
        else None
    )
    emit(writer, ProgressEvent(kind="start", stage="run"))
    try:
        output = await asyncio.to_thread(
            execute,
            recipe,
            lambda source: cache[source.name],
            generate=generator,
            row_cap=spec.row_cap,
            byte_cap=spec.byte_cap,
        )
    except RecipeError as exc:
        raise RunError(str(exc)) from exc
    generation = generator.facts if generator is not None else []
    metrics = {
        "rows": float(output.rows),
        "synthetic_rows": float(output.synthetic_rows),
        "excluded": float(len(output.exclusions)),
        "splits": float(len(output.splits)),
    }
    workdir = Path.cwd()
    if preview:
        # Nothing leaves the machine but the summary: no file, no branch, no commit.
        summary = _summary(spec, output, generation, commit=None, written={}, truncated=truncated)
        summary["sample"] = _sample(output, spec.preview_rows)
        workdir.joinpath(WORKDIR_SUMMARY_FILENAME).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        emit(writer, ProgressEvent(kind="end", stage="run", metrics=metrics, message="preview"))
        return "preview"
    if hub is None or out_repo is None:  # unreachable: set above for every materialization
        msg = "a materialization has no output repo"
        raise RunError(msg)
    outdir = results_dir(spec.run_id, name=_RESULTS_DIR_NAME)
    async with ticking_phase(phase, "Writing the dataset", stage="push"):
        files, written = await asyncio.to_thread(_write_outputs, spec, output, outdir)
        draft = _summary(spec, output, generation, commit=None, written=written)
        manifest = _manifest(spec, output, generation, draft)
        files[MANIFEST_FILENAME] = await asyncio.to_thread(write_manifest, manifest, outdir)
        card = outdir / "README.md"
        card.write_text(_card(output, spec.recipe, spec.recipe_revision))
        files["README.md"] = card
    try:
        async with ticking_phase(phase, "Pushing the dataset to the Hub", stage="push"):
            await ensure_private_output(hub, out_repo)  # visibility can change during a run
            pushed = await hub.replace_files(
                {k: str(v) for k, v in files.items()},
                out_repo,
                managed=MANAGED_PATHS,
                repo_type="dataset",
                commit_message=f"materialize recipe {spec.recipe.name} r{spec.recipe_revision}",
                branch=None,
            )
    except Exception as exc:
        msg = f"the dataset is on the machine at {outdir}, but the push failed: {exc}"
        raise RunError(msg) from exc
    commit = commit_oid(pushed)
    if commit is None:
        msg = f"the push to {out_repo} reported no commit, so the version cannot be pinned"
        raise RunError(msg)
    summary = _summary(spec, output, generation, commit=commit, written=written)
    receipt = build_receipt(
        _MANIFEST_KIND,
        files[MANIFEST_FILENAME],
        (out_repo, commit),
        rows={name: s["rows"] for name, s in summary["splits"].items()},
    )
    receipt["recipe"] = {
        "revision": spec.recipe_revision,
        "sha256": spec.recipe_sha256,
        "holdout_sources": summary["holdout_sources"],
        "held_out_splits": [n for n, s in summary["splits"].items() if s["held_out"]],
    }
    await asyncio.to_thread(write_receipt, receipt, workdir)
    workdir.joinpath(WORKDIR_SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    emit(
        writer,
        ProgressEvent(kind="end", stage="push", metrics=metrics, message=f"{out_repo}@{commit}"),
    )
    return out_repo


async def main() -> int:
    """Entry point: returns a process exit code (0 ok, 1 failure). Never leaks the token."""
    return await runner_main(
        lambda writer, secrets: _execute(load_spec(writer=writer), secrets, writer)
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
