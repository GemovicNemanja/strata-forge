"""VM-side batch-inference runner: STRATA_RUN_CONFIG (inert JSON) -> vLLM -> HF results.

Launched on the user's own VM by the control plane (``python -m
strata_forge.pipelines.inference_runner``). It reads an INERT run spec from the
``STRATA_RUN_CONFIG`` env var, loads a Hugging Face dataset split, renders one prompt
per row by simple ``{column}`` substitution (NO code execution), serves the model with
vLLM on the same VM, runs a concurrency-bounded batch through an openai-compatible
``LLMClient``, and then EITHER pushes ``{custom_id, output}`` results to the user's HF
dataset repo (when a write token + an output repo are supplied) OR keeps them as a parquet
file on the VM (outside the per-run workdir, for the user to retrieve over SSH). Live
progress is appended to the file named by ``FORGE_PROGRESS_PATH`` (the orchestrator tails it).

Security boundary (the VM is where untrusted-but-allow-listed config meets real
credentials + the network):

  - The config is parsed as DATA only: ``json`` + Pydantic ``extra="forbid"`` — never
    ``eval``/``pickle``/``yaml.unsafe_load``. ``template``/``column_mapping``/
    ``hyperparams`` are never executed.
  - Template rendering is a bounded, NON-executing ``{name}`` substitution (a regex, NOT
    ``str.format`` and NOT a template engine) — only placeholders that map to a real
    column are replaced; anything else stays literal.
  - The HF write token arrives in its OWN env var (``HF_WRITE_TOKEN``), never in
    ``STRATA_RUN_CONFIG``, is passed EXPLICITLY to the Hub/dataset clients (never the
    VM's ambient ``HF_TOKEN``), and is scrubbed from any surfaced error.
  - Progress + result rows carry no secret: events hold step counts + float metrics + a
    repo id; result rows are ``{custom_id, output, error}`` (model text only). Phase
    messages go through the same scrub as errors, because ``serving_endpoint``'s phase hook
    is public API and a caller's phrase is not under this module's control.

The exit code is the run's VERDICT, and the control plane reads it as such. Individual row
failures are collected rather than fatal (a few filtered rows must not discard thousands of good
generations) and reported as succeeded/failed counts, but a run that produced no usable row at
all exits nonzero: there is no reading under which it did its job, and the results file it
leaves behind holds only errors.

Long provisioning steps (downloading the split, installing and starting the model server,
writing and uploading results) have nothing to count, so each reports itself with a
``ProgressEvent(kind="phase")``. Without them a run is a single indeterminate wait between
``start`` and the first ``step`` — which is where a model server that never comes up spends
its entire timeout, invisibly.
"""

from __future__ import annotations

import asyncio
import itertools
import re
import sys
import time
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from strata_forge.compute import LocalBackend
from strata_forge.compute.batch import BatchInferenceRunner
from strata_forge.compute.serving import build_vllm_task, serving_endpoint
from strata_forge.llm import LLMClient, UserMessage
from strata_forge.llm.providers.config import OpenAICompatConfig
from strata_forge.llm.providers.openai_compat import (
    UNAUTHENTICATED_API_KEY,
    OpenAICompatProvider,
)
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
from strata_forge.training.hardware import GpuSampler
from strata_forge.training.progress import JsonlProgressWriter, ProgressEvent

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = ["RunSpec", "main", "render_template"]

# vLLM serves on loopback only — the runner talks to it via localhost, and a multi-tenant
# / network-reachable VM must not expose the model server.
_SERVE_HOST = "127.0.0.1"
_SERVE_PORT = 8000
# A bare ``{name}`` placeholder only — no attribute/index access, no format mini-language.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
# Cap a rendered prompt so a pathological row can't blow up memory / the request.
_MAX_RENDERED_CHARS = 200_000
_RESULTS_FILENAME = "results.parquet"
# Where results land on the VM when they are not pushed — see `_local_results_dir`.
_RESULTS_DIR_NAME = "strata-inference-results"
# How much of one row's error is quoted as the sample when EVERY row failed. Enough to name a
# provider/status/class, short enough that the run's message stays a message.
_MAX_SAMPLE_ERROR_CHARS = 500


class Hyperparams(BaseModel):
    """Inert tuning knobs (sampling + serving + bounds). Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = None
    concurrency: int = Field(default=8, ge=1, le=256)
    progress_chunk: int = Field(default=200, ge=1)
    row_limit: int | None = Field(default=None, ge=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    max_model_len: int | None = Field(default=None, ge=1)
    dtype: str | None = None
    wait_timeout_s: float = Field(default=1800.0, gt=0)


class RunSpec(BaseModel):
    """The inert run spec the server delivers in ``STRATA_RUN_CONFIG`` (no secrets)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    dataset_id: str
    dataset_commit_sha: str | None = None
    dataset_config: str | None = None
    split: str
    column_mapping: dict[str, str] = Field(default_factory=dict)
    template: str
    hyperparams: Hyperparams = Field(default_factory=Hyperparams)
    # Where to push results. The server populates it ONLY when pushing (a write token is present);
    # when absent the runner keeps results on the VM instead of pushing. The VM never guesses ownership.
    output_repo_id: str | None = None
    progress_path: str | None = None
    # The control plane's run id — names the on-VM results dir when NOT pushing to the Hub.
    run_id: str | None = None


def load_spec() -> RunSpec:
    spec = load_config(RunSpec)
    validate_repo_id(spec.model_id, "model")
    validate_repo_id(spec.dataset_id, "dataset")
    if spec.output_repo_id is not None:
        validate_repo_id(spec.output_repo_id, "output repo")
    _validate_template_coverage(spec)
    return spec


def _validate_template_coverage(spec: RunSpec) -> None:
    """Reject a template whose ``{placeholder}`` has no entry in ``column_mapping``.

    An unmapped placeholder renders LITERALLY (see ``render_template``), which is the right
    behaviour for the primitive but a catastrophe for a run: every row gets the byte-identical,
    row-independent prompt, the model answers it N times, every row SUCCEEDS, and the run reports
    `succeeded` with N copies of an answer to the literal text ``{question}``. Nothing downstream
    can notice — the results file records ``{custom_id, output, error}``, so neither the rendered
    prompt nor the source row is in it.

    The one check that existed validated the mapping's VALUES against the split's columns, which
    passes in exactly this case: with ``{"q": "question"}`` against a template of ``{question}``,
    the column ``question`` really does exist. It is the KEYS that fail to cover the template, and
    nobody was looking at them.

    That mistake is easy to make and expensive to discover, so it fails here — before the dataset
    download, before the GPU, within seconds of launch — naming both what is missing and what is
    available.

    The trade-off is deliberate: a template can no longer carry a LITERAL ``{word}`` that is meant
    to survive to the model. That reading is rare, and it is not worth the run this protects.
    """
    placeholders = {m.group(1) for m in _PLACEHOLDER_RE.finditer(spec.template)}
    unmapped = sorted(placeholders - set(spec.column_mapping))
    if not unmapped:
        return
    known = sorted(spec.column_mapping) or ["(none)"]
    msg = (
        f"template placeholders have no column_mapping entry: {unmapped}. "
        f"Mapped placeholders: {known}. "
        "A placeholder is named WITHOUT braces on the left of the mapping "
        '(template "{question}" needs the entry "question" -> the column name); '
        "an unmapped one would be sent to the model literally, identically for every row."
    )
    raise RunError(msg)


def render_template(template: str, row: dict[str, Any], column_mapping: dict[str, str]) -> str:
    """Render ``template`` by replacing each ``{name}`` whose ``column_mapping[name]`` is a
    column present in ``row`` with ``str(row[col])``. Unknown placeholders stay literal.

    Executes nothing: a regex substitution, deliberately NOT ``str.format`` (which reaches
    attributes/indices + the format mini-language) and NOT a template engine. Capped.
    """

    def _sub(match: re.Match[str]) -> str:
        col = column_mapping.get(match.group(1))
        if col is None or col not in row:
            return match.group(0)  # leave an unmapped/missing placeholder literal
        return str(row[col])

    rendered = _PLACEHOLDER_RE.sub(_sub, template)
    return rendered[:_MAX_RENDERED_CHARS]


def _load_rows(spec: RunSpec, hf_token: str | None) -> list[dict[str, Any]]:
    try:
        # Lazy optional-extra import (forge convention: `Any` so pyright skips the unresolved
        # module); the [hf] extra ships `datasets`.
        datasets_mod: Any = __import__("datasets")
    except ImportError as exc:
        msg = "the [hf] extra is required to load datasets: pip install 'strata-forge[hf]'"
        raise RunError(msg) from exc
    dataset = datasets_mod.load_dataset(
        spec.dataset_id,
        name=spec.dataset_config,
        split=spec.split,
        revision=spec.dataset_commit_sha,
        token=hf_token or None,
        streaming=False,
    )
    columns = set(dataset.column_names or [])
    missing = {c for c in spec.column_mapping.values() if c not in columns}
    if missing:
        msg = f"column_mapping references columns not in the split: {sorted(missing)}"
        raise RunError(msg)
    # Slice DURING iteration (not after): dataset size/content is attacker-controlled, so
    # row_limit must bound materialization — a huge split mustn't OOM the VM before the cap.
    limit = spec.hyperparams.row_limit
    source = itertools.islice(dataset, limit) if limit is not None else dataset
    return [dict(r) for r in source]


def _build_requests(
    spec: RunSpec, rows: list[dict[str, Any]]
) -> tuple[list[list[UserMessage]], list[str]]:
    prompts: list[list[UserMessage]] = []
    custom_ids: list[str] = []
    for i, row in enumerate(rows):
        rendered = render_template(spec.template, row, spec.column_mapping)
        prompts.append([UserMessage(content=rendered)])  # one-message conversation
        custom_ids.append(f"row-{i}")  # positional reconciliation key
    return prompts, custom_ids


async def _run_batches(
    spec: RunSpec,
    client: LLMClient,
    prompts: list[list[UserMessage]],
    custom_ids: list[str],
    writer: JsonlProgressWriter | None,
    is_alive: Callable[[], Awaitable[bool]] | None = None,
    gpu: GpuSampler | None = None,
) -> list[dict[str, Any]]:
    """Run the prompts in progress-chunked batches; reconcile results positionally.

    ``is_alive`` is checked between chunks. The model server is verified once before the batch
    starts and then never again, but it can die at any point after that — an OOM on a long prompt,
    a CUDA fault. Every row from then on fails against a socket nobody is listening on, and the
    client RETRIES each one, so a dead server turns into a long expensive silence instead of an
    error: the rest of the run is spent timing out one row at a time, and the failure that
    eventually surfaces describes a connection, not the crash that caused it.
    """
    hp = spec.hyperparams
    runner = BatchInferenceRunner(client, concurrency=hp.concurrency, on_error="collect")
    total = len(prompts)
    out: list[dict[str, Any]] = []
    ok = failed = 0
    # Throughput is measured from the first chunk, not from process start: everything before this
    # point is dataset loading and engine warmup, and folding minutes of that into the denominator
    # would report a rows/s the run never actually ran at.
    started = time.monotonic()
    latencies_ms: list[float] = []
    output_tokens = 0
    for start in range(0, total, hp.progress_chunk):
        # Between chunks, not between rows: the check costs a remote status probe, and a chunk is
        # the granularity the run already reports at. It bounds the waste at one chunk rather than
        # the whole remaining batch.
        if start and is_alive is not None and not await is_alive():
            msg = (
                f"the model server died after {len(out)} of {total} rows "
                "(its log is beside the results on the VM)"
            )
            raise RunError(msg)
        chunk = prompts[start : start + hp.progress_chunk]
        ids = custom_ids[start : start + hp.progress_chunk]
        results = await runner.run(
            chunk, temperature=hp.temperature, max_tokens=hp.max_tokens, top_p=hp.top_p
        )
        for cid, result in zip(ids, results, strict=True):
            if result.succeeded and result.response is not None:
                out.append({"custom_id": cid, "output": result.response.text, "error": None})
                ok += 1
                # Both already measured per request by the client — a true per-row latency and a
                # real token count, not an average reconstructed from the chunk's wall time.
                latencies_ms.append(result.response.latency_ms)
                output_tokens += result.response.usage.output_tokens
            else:
                out.append({"custom_id": cid, "output": None, "error": repr(result.error)})
                failed += 1
        emit(
            writer,
            ProgressEvent(
                kind="step",
                stage="run",
                step=len(out),
                total_steps=total,
                metrics={
                    "succeeded": float(ok),
                    "failed": float(failed),
                    **_throughput(started, len(out), output_tokens, latencies_ms),
                    **(gpu.sample() if gpu is not None else {}),
                },
            ),
        )
    return out


def _throughput(
    started: float, rows_done: int, output_tokens: int, latencies_ms: list[float]
) -> dict[str, float]:
    """Rate and latency counters for the rows finished so far.

    Rates are cumulative rather than per-chunk: a chunk is small enough that its own wall time is
    mostly noise from whichever row happened to be slowest, and a gauge that swings on that reads
    as broken. ``latency_p50_ms`` is a real median over every row completed so far — the tail is
    what makes a mean useless here, and the median is what survives it.
    """
    elapsed = time.monotonic() - started
    metrics: dict[str, float] = {}
    if elapsed > 0:
        metrics["rows_per_s"] = rows_done / elapsed
        metrics["tokens_per_s"] = output_tokens / elapsed
    if latencies_ms:
        metrics["latency_p50_ms"] = median(latencies_ms)
    return metrics


def _check_produced_output(rows: list[dict[str, Any]], ok: int) -> None:
    """Fail a run that reached the end without producing a single usable row.

    Row failures are COLLECTED rather than fatal on purpose: a handful of rows tripping a
    content filter must not throw away thousands of good generations. But that must not decide
    the run's VERDICT, which the control plane reads from this process's exit code. A run whose
    every row failed produced nothing, and calling it succeeded tells the user their results are
    ready when the file holds only errors — a staging run reported `succeeded` with exit 0 over
    2098 failed rows and zero generations, and nothing on the run said otherwise.

    A partial failure is deliberately still a success: it produced usable output, and the
    succeeded/failed counts on the run describe it. Only "nothing at all" is a failure, because
    only that has no reading under which the run did its job.

    The results file is written and pushed BEFORE this runs, so the failure keeps its evidence:
    the per-row `error` column is the record of what went wrong, for every row.
    """
    if ok:
        return
    if not rows:
        msg = "the run produced no rows: the dataset split was empty"
        raise RunError(msg)
    sample = next((str(row["error"]) for row in rows if row["error"]), "")
    msg = f"all {len(rows)} rows failed. First error: {sample[:_MAX_SAMPLE_ERROR_CHARS]}"
    raise RunError(msg)


def _local_results_dir(run_id: str | None) -> Path:
    """Where this run's parquet lands on the VM — see :func:`strata_forge.pipelines._common.results_dir`."""
    return results_dir(run_id, name=_RESULTS_DIR_NAME)


def _write_results(rows: list[dict[str, Any]], outdir: Path) -> Path:
    """Write the results as parquet (renders in the HF Datasets Viewer)."""
    datasets_mod: Any = __import__("datasets")  # already required by _load_rows ([hf] extra)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / _RESULTS_FILENAME
    datasets_mod.Dataset.from_list(rows).to_parquet(str(path))
    return path


async def _push_results(spec: RunSpec, results_path: Path, hf_token: str) -> str:
    if not spec.output_repo_id:
        msg = "output_repo_id is required to push results"
        raise RunError(msg)
    out_repo = validate_repo_id(spec.output_repo_id, "output repo")
    hub = HFHubClient(token=hf_token)  # EXPLICIT write token, never the ambient HF_TOKEN
    await hub.create_repo(out_repo, repo_type="dataset", private=True, exist_ok=True)
    await hub.upload_file(
        results_path,
        _RESULTS_FILENAME,
        out_repo,
        repo_type="dataset",
        commit_message="batch inference results",
    )
    return out_repo


async def _execute(spec: RunSpec, hf_token: str | None, writer: JsonlProgressWriter | None) -> str:
    # One sampler for the run: the phase sink folds its counters into every caption, and the
    # step events below reuse the same cached reading rather than shelling out twice.
    gpu = GpuSampler()
    phase = phase_sink(writer, hf_token, gpu=gpu)

    # The split download is unbounded by row_limit (that only slices during iteration), so it
    # runs BEFORE the first countable milestone and can take minutes on its own.
    # to_thread, not a direct call: `load_dataset` downloads and materialises the split, and a
    # blocked event loop cannot tick the caption that says it is still going.
    async with ticking_phase(phase, "Loading the dataset", stage="load_model"):
        rows = await asyncio.to_thread(_load_rows, spec, hf_token)
    prompts, custom_ids = _build_requests(spec, rows)
    # `start` stays exactly here: it is the documented milestone that says inference is about
    # to happen, and the orchestrator reads it as such. Phases fill the silence around it.
    emit(writer, ProgressEvent(kind="start", stage="run", total_steps=len(prompts)))

    hp = spec.hyperparams
    task = build_vllm_task(
        spec.model_id,
        port=_SERVE_PORT,
        host=_SERVE_HOST,
        tensor_parallel_size=hp.tensor_parallel_size,
        max_model_len=hp.max_model_len,
        dtype=hp.dtype,
        setup="",  # vLLM is already installed by the server's setup step; don't reinstall
        # Serve with the interpreter this runner is executing under. LocalBackend runs the task
        # through a login shell, which re-sources the profile and drops this virtualenv from PATH:
        # a bare `vllm` is then "command not found" even though vLLM is installed right here.
        python_executable=sys.executable,
    )
    # Unbuffered: the served process writes through a pipe, so CPython would otherwise hold
    # its output in an 8 KiB block buffer — and a server that hangs before filling it leaves
    # the log file empty, which is precisely the case the file exists for.
    # A batch run must not depend on a CUDA build toolchain being present on someone else's box.
    # vLLM's default sampler is FlashInfer's, which JIT-COMPILES its kernels during warmup: it
    # shells out to ninja, and a GPU image carrying the driver and runtime but no build tools
    # fails with "No such file or directory: 'ninja'" — after loading the weights, compiling the
    # graph, capturing CUDA graphs and allocating the KV cache, so the run has already paid for
    # everything before it dies. The PyTorch-native sampler is marginally slower per token and
    # needs no compiler, which is the right default for a machine we do not provision.
    task = task.model_copy(
        update={
            "env": {
                **task.env,
                # Unbuffered: the served process writes through a pipe, so CPython would otherwise
                # hold its output in an 8 KiB block buffer — and a server that hangs before filling
                # it leaves the log file empty, which is precisely the case the file exists for.
                "PYTHONUNBUFFERED": "1",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
            }
        }
    )
    # serving_endpoint reports "Starting the model server" itself the moment it submits, so
    # announcing it here as well would show the same phrase twice for one step.
    async with serving_endpoint(
        # Tee the served process's streams into the run workdir. When the runner dies, the
        # buffers die with it; the files are what is left to explain why the server never came up.
        LocalBackend(log_dir=Path.cwd()),
        task,
        base_url=f"http://{_SERVE_HOST}:{_SERVE_PORT}/v1",
        wait_timeout_s=hp.wait_timeout_s,
        # serving.py reports prose through a plain one-arg hook and knows nothing about run
        # stages, which is right: bind the stage here rather than widening its public API.
        on_phase=lambda message: phase(message, stage="load_model"),
    ) as endpoint:
        # The endpoint this runner just launched is on loopback and takes no credential, and
        # saying so EXPLICITLY is what keeps it deterministic: left unset, the client falls back
        # to whatever OPENAI_API_KEY the VM happens to carry, which would send the user's real
        # provider key to a local server that never asked for one.
        provider = OpenAICompatProvider(
            OpenAICompatConfig(
                base_url=endpoint.base_url, api_key=SecretStr(UNAUTHENTICATED_API_KEY)
            )
        )
        client = LLMClient(
            model=spec.model_id,
            provider="openai_compat",
            provider_clients={"openai_compat": provider},  # pins the LOCAL endpoint
        )
        # The first `step` only lands once a whole progress_chunk has completed, and that
        # chunk absorbs the client's cold start on top of its generations.
        async with ticking_phase(phase, "Generating responses", stage="run"):
            out = await _run_batches(
                spec, client, prompts, custom_ids, writer, endpoint.is_alive, gpu
            )

    # Results are ALWAYS written outside the per-run workdir, whether or not they are then pushed.
    # The orchestrator deletes that workdir on every terminal state, so writing there and pushing
    # from it means a failed upload — an expired token, a rate limit, a network blip — takes the
    # entire run's output with it. Hours of generation, gone at the last step, with nothing left to
    # retry from. Outside it, the parquet survives for the user to retrieve over SSH (or push by
    # hand) exactly as in the no-push case.
    #
    # Both branches sit between the last `step` (which reads 100%) and `end`, so a run that dies
    # here would otherwise look like it died complete, with no explanation.
    async with ticking_phase(phase, "Writing results", stage="push"):
        results_path = await asyncio.to_thread(_write_results, out, _local_results_dir(spec.run_id))
    if hf_token and spec.output_repo_id:
        try:
            async with ticking_phase(phase, "Uploading results to the Hub", stage="push"):
                destination = await _push_results(spec, results_path, hf_token)
        except Exception as exc:
            # Name WHERE the results are. The run still fails — the user asked for them on the Hub
            # and they are not there — but a failure at the last step of a long run is the moment
            # it matters most that the output was not lost with it.
            msg = f"results are on the VM at {results_path}, but the push failed: {exc}"
            raise RunError(msg) from exc
    else:
        destination = str(results_path)

    ok = sum(1 for r in out if r["error"] is None)
    emit(
        writer,
        ProgressEvent(
            kind="end",
            stage="push",
            step=len(out),
            total_steps=len(out),
            metrics={"succeeded": float(ok), "failed": float(len(out) - ok)},
            message=destination,  # WHERE results landed: a repo id (pushed) or a VM path (local)
        ),
    )
    # AFTER `end`: the counts and the destination are true whichever way the verdict falls, and
    # for a run that failed this way they are the diagnosis — they say every row failed and where
    # the per-row errors can be read.
    _check_produced_output(out, ok)
    return destination


async def main() -> int:
    """Entry point: returns a process exit code (0 ok, 1 failure). Never leaks the token."""
    return await runner_main(lambda writer, token: _execute(load_spec(), token, writer))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
