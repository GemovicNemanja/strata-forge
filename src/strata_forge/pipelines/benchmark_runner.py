"""VM-side latency benchmark: a pinned model on a pinned workload split, timed at declared boundaries.

Launched like the batch runner (``python -m strata_forge.pipelines.benchmark_runner``) with an
INERT spec in ``STRATA_RUN_CONFIG``. A benchmark is a bounded measurement job, not a serving
deployment: it loads one model at a full commit, reads one dataset split at a full commit, and
times a fixed grid of configurations (batch sizes x concurrent request streams) over a seeded
sample of the split's rows. The engines are the batch runner's own:

- ``hf_classify`` loads the sequence classifier exactly as a scoring run does and times each
  request in this process. **End to end** runs from the text pair in memory through tokenization,
  the host-to-device copy, the forward pass and the softmax to the probabilities back on the host,
  stopping after ``torch.cuda.synchronize()``. **Model only** is the forward pass alone, with a
  synchronization on each side. Both are read from ``time.perf_counter_ns`` in one iteration, so
  the model-only time is a sub-interval of the end-to-end one. One request stream only.
- ``vllm_chat`` serves the model with vLLM on loopback, as a batch run does, and times each chat
  request from this process: **end to end** covers the HTTP request, the server's tokenization,
  queueing, prefill and decode, and the response. The forward pass runs in another process, so
  **model only** is refused for it, as is a batch size above one (batching is the server's).

The protocol, recorded in the manifest: rows the spec excludes are counted, not timed; warmup
iterations drawn from each pass's own arrival order run and are discarded before EVERY timed pass,
so a first-call cost never reaches a percentile, and the cold start (the model load and the very
first iteration) is recorded on its own; each pass serves the sample in a fresh shuffle seeded
with ``seed + pass_index``; a wall-clock cap is a hard stop that marks the results incomplete; an
iteration that fails stops its configuration, which is then reported as not clean.

Outputs, never row text: ``timings.parquet`` (one row per measured iteration), ``summary.json``
(:class:`~strata_forge.pipelines._benchmark.BenchmarkSummary`) and ``manifest.json`` (the pins, the
admitted settings, the protocol, the environment receipt and the counts), pushed to a private
output repo of the run's own in ONE commit, or kept on the VM when the run has no write token. A
copy of the summary is left in the job's working directory for the control plane to read.

The security frame is the batch runner's, reused rather than restated: the spec is data, the
token arrives in the secrets file and is passed explicitly to every Hub call, the model download
admits only servable files, the model server gets an allow-listed environment and no token, and
every surfaced message is redacted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
from importlib import import_module
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from strata_forge.compute import LocalBackend
from strata_forge.compute.serving import build_vllm_task, serving_endpoint
from strata_forge.llm import LLMClient, UserMessage
from strata_forge.llm.providers.config import OpenAICompatConfig
from strata_forge.llm.providers.openai_compat import (
    UNAUTHENTICATED_API_KEY,
    OpenAICompatProvider,
)
from strata_forge.pipelines import _benchmark as bench
from strata_forge.pipelines import inference_runner as _batch
from strata_forge.pipelines._common import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    PhaseSink,
    RunError,
    RunSecrets,
    emit,
    load_config,
    manifest_engine,
    model_server_environ,
    phase_sink,
    require_commit_sha,
    results_dir,
    run_redactor,
    runner_main,
    snapshot_commit,
    ticking_phase,
    validate_repo_id,
    write_manifest,
)
from strata_forge.pipelines._environment import (
    compute_pids,
    gpu_matches,
    gpu_state,
    own_processes,
    receipt,
    short_receipt,
)
from strata_forge.pipelines._scoring import (
    VerifierRunScorer,
    VerifierScorer,
    build_inputs,
    load_classifier,
    load_verifier,
    timed_verifier_request,
)
from strata_forge.pipelines.inference_runner import Hyperparams, RunSpec, render_template
from strata_forge.storage import HFHubClient
from strata_forge.training.hardware import GpuSampler
from strata_forge.training.loading import tokenizer_load_kwargs
from strata_forge.training.progress import JsonlProgressWriter, ProgressEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from strata_forge.pipelines._scoring import Classifier

__all__ = [
    "MAX_CONFIGURATIONS",
    "MAX_MEASURED_ITERATIONS",
    "SUMMARY_FILENAME",
    "TIMINGS_FILENAME",
    "WORKDIR_SUMMARY_FILENAME",
    "BenchmarkHyperparams",
    "BenchmarkSpec",
    "load_spec",
    "main",
]

# The batch runner's download, serving and output-repo steps, reused as they are: a benchmark
# reads, downloads, serves and pushes exactly as a batch run does, so the two cannot drift.
_load_rows = _batch._load_rows  # pyright: ignore[reportPrivateUsage]
_fetch_model = _batch._fetch_model  # pyright: ignore[reportPrivateUsage]
_read_model_config = _batch._read_model_config  # pyright: ignore[reportPrivateUsage]
_effective_hyperparams = _batch._effective_hyperparams  # pyright: ignore[reportPrivateUsage]
_validate_template_coverage = _batch._validate_template_coverage  # pyright: ignore[reportPrivateUsage]
_serve_args = _batch._serve_args  # pyright: ignore[reportPrivateUsage]
_harmony_date = _batch._harmony_date  # pyright: ignore[reportPrivateUsage]
_sampling_extras = _batch._sampling_extras  # pyright: ignore[reportPrivateUsage]
_check_output_repo = _batch._check_output_repo  # pyright: ignore[reportPrivateUsage]
_ensure_private_output = _batch._ensure_private_output  # pyright: ignore[reportPrivateUsage]
_require_fresh_output = _batch._require_fresh_output  # pyright: ignore[reportPrivateUsage]
_served_max_model_len = _batch._served_max_model_len  # pyright: ignore[reportPrivateUsage]
_served_dtype = _batch._served_dtype  # pyright: ignore[reportPrivateUsage]
_checkpoint_dtype = _batch._checkpoint_dtype  # pyright: ignore[reportPrivateUsage]
_serve_flags = _batch._serve_flags  # pyright: ignore[reportPrivateUsage]
_SERVE_HOST = _batch._SERVE_HOST  # pyright: ignore[reportPrivateUsage]
_SERVE_PORT = _batch._SERVE_PORT  # pyright: ignore[reportPrivateUsage]
_HARMONY_DATE_ENV = _batch._HARMONY_DATE_ENV  # pyright: ignore[reportPrivateUsage]

TIMINGS_FILENAME = "timings.parquet"
SUMMARY_FILENAME = "summary.json"
#: The copy of the summary the control plane reads from the job's working directory.
WORKDIR_SUMMARY_FILENAME = "benchmark-summary.json"
_RESULTS_DIR_NAME = "strata-benchmark-results"
_MANIFEST_KIND = "benchmark"
#: The grid's size bound: batch sizes times concurrency levels.
MAX_CONFIGURATIONS = 32
#: The bound on measured iterations summed over every configuration and pass.
MAX_MEASURED_ITERATIONS = 2_000_000
_MAX_SAMPLE = 100_000
# The most rows a split may hold: every row is read, and a split past `item_cap` is refused.
_MAX_ITEM_CAP = 1_000_000
_SEED_MAX = 2**53 - 1
_MAX_PASSES = 20
_MAX_WARMUP = 10_000
_MAX_WALL_CLOCK_S = 6 * 3600.0
_MAX_BUCKET_EDGES = 16
_MAX_BOOTSTRAP_RESAMPLES = 100_000
# The engine settings a benchmark applies, per engine. An allow-list rather than a deny-list: a
# knob the batch engine gains later is refused here until the benchmark actually applies it, so a
# spec never records a setting its measurements did not run with. The grid owns batching,
# concurrency and the row count; one iteration is one request, so `n` samples per row are refused.
_ENGINE_FIELDS: dict[str, frozenset[str]] = {
    "hf_classify": frozenset(
        {
            "pair_template",
            "label_alias",
            "dtype",
            "max_model_len",
            "list_separator",
            "artifact",
            "apply_calibration",
        }
    ),
    "vllm_chat": frozenset(
        {
            "temperature",
            "max_tokens",
            "top_p",
            "seed",
            "top_k",
            "tensor_parallel_size",
            "max_model_len",
            "dtype",
            "gpu_memory_utilization",
            "max_num_batched_tokens",
            "reasoning_parser",
            "reasoning_effort",
            "harmony_date",
            "wait_timeout_s",
            "guided_json",
            "chat_template_kwargs",
        }
    ),
}
_GPU_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}")
# Reported at most this often while measuring, so progress writes stay out of the measurements.
_PROGRESS_INTERVAL_S = 1.0
# A rendered chat prompt longer than this is over any model's limit without tokenizing it.
_MAX_RENDERED_CHARS = 200_000
_FORWARD_DEFINITION = (
    "forward_ms (diagnostic): two CUDA events recorded on the current stream around the forward "
    "pass inside the same request, read after the request's one torch.cuda.synchronize(); no "
    "synchronization is added to take it (on a CPU, time.perf_counter_ns around the forward pass)"
)
_DEFINITIONS = {
    "strata_verifier": {
        "end_to_end": (
            "total_ms (headline): time.perf_counter_ns from the raw (premises, hypothesis) cells in "
            "memory through the verifier's own rendering (strata_verifier.json), tokenization, "
            "the host-to-device copy, the forward pass and the auxiliary heads, the copies to the "
            "host, the 3-way softmax, the flag head's sigmoid (p_inconsistent) and the support "
            "head's per-premise sigmoids (support_scores), stopped after ONE "
            "torch.cuda.synchronize()"
        ),
        "model_only": _FORWARD_DEFINITION + "; the auxiliary heads are inside the interval",
    },
    "reference_automodel": {
        "end_to_end": (
            "total_ms (headline): time.perf_counter_ns from the text pairs in memory through "
            "tokenization, the host-to-device copy, the forward pass and a softmax over every "
            "class to the probabilities copied to the host, stopped after ONE "
            "torch.cuda.synchronize(); a third-party reference model, run as a plain "
            "AutoModelForSequenceClassification"
        ),
        "model_only": _FORWARD_DEFINITION,
    },
    "vllm_chat": {
        "end_to_end": (
            "time.perf_counter_ns around one chat completion request from the runner process to "
            "the model server on loopback: the HTTP request, the server's chat templating, "
            "tokenization, queueing, prefill and decode, and the response"
        ),
    },
}


def _unique(values: Sequence[object], what: str) -> None:
    if len(set(values)) != len(values):
        msg = f"{what} lists a value twice"
        raise ValueError(msg)


def _edges(values: list[int], what: str) -> list[int]:
    if any(b <= a for a, b in pairwise(values)):
        msg = f"{what} must be strictly ascending upper bounds"
        raise ValueError(msg)
    return values


class BenchmarkHyperparams(BaseModel):
    """The measurement: the engine and its settings, the grid, the protocol and its bounds."""

    model_config = ConfigDict(extra="forbid")

    # The engine and how it loads or serves the model: a batch run's own hyperparameters, so a
    # benchmark measures the model exactly as a batch or scoring run would run it.
    engine_settings: Hyperparams
    # `end_to_end` (total_ms) is the headline and always measured; `model_only` adds the
    # forward_ms diagnostic, taken inside the same requests.
    boundaries: list[bench.Boundary] = Field(min_length=1, max_length=2)
    batch_sizes: list[StrictInt] = Field(min_length=1, max_length=16)
    concurrency: list[StrictInt] = Field(default_factory=lambda: [1], min_length=1, max_length=16)
    # Warmup iterations before EVERY timed pass, discarded.
    warmup: StrictInt = Field(default=50, ge=1, le=_MAX_WARMUP)
    passes: StrictInt = Field(default=5, ge=1, le=_MAX_PASSES)
    # Every row of the pinned split is read; a split with more rows than this is refused, never
    # truncated, so the workload is the whole split or nothing.
    item_cap: StrictInt = Field(default=_MAX_SAMPLE, ge=1, le=_MAX_ITEM_CAP)
    # Rows drawn from the eligible rows; unset, every eligible row is measured.
    sample_size: StrictInt | None = Field(default=None, ge=1, le=_MAX_SAMPLE)
    # Three seeds, all recorded: the sample draw, each pass's timed order (order_seed + pass) and
    # each pass's warmup draw (warmup_seed + pass).
    sample_seed: StrictInt = Field(default=0, ge=0, le=_SEED_MAX)
    order_seed: StrictInt = Field(default=1_000, ge=0, le=_SEED_MAX)
    warmup_seed: StrictInt = Field(default=2_000, ge=0, le=_SEED_MAX)
    bootstrap_seed: StrictInt = Field(default=0, ge=0, le=_SEED_MAX)
    wall_clock_cap_s: StrictFloat | StrictInt = Field(default=1800.0, gt=0, le=_MAX_WALL_CLOCK_S)
    # Rows whose rendered input is longer than this many tokens are excluded and counted.
    max_input_tokens: StrictInt | None = Field(default=None, ge=1)
    # A list-valued column whose length is a row's item count (premises, documents, ...).
    item_count_column: str | None = Field(default=None, min_length=1, max_length=255)
    max_items: StrictInt | None = Field(default=None, ge=1)
    # Reporting strata at a batch size of one: inclusive upper bounds, ascending.
    length_buckets: list[StrictInt] = Field(default_factory=list[int], max_length=_MAX_BUCKET_EDGES)
    item_buckets: list[StrictInt] = Field(default_factory=list[int], max_length=_MAX_BUCKET_EDGES)
    # Opt-in percentile-bootstrap intervals for the pooled percentiles; unset, none are estimated.
    bootstrap_resamples: StrictInt | None = Field(default=None, ge=1, le=_MAX_BOOTSTRAP_RESAMPLES)
    # vllm_chat only: the model server's automatic prefix caching. Off by default: every pass
    # replays the same prompts, so a cached prefill would understate the end-to-end time. Recorded
    # in the manifest either way.
    prefix_caching: StrictBool = False
    # Refuse to measure unless every visible GPU's reported name contains this (e.g. "H100").
    require_gpu: str | None = Field(default=None, max_length=64)

    @field_validator("batch_sizes")
    @classmethod
    def _batch_sizes(cls, values: list[int]) -> list[int]:
        if any(not 1 <= v <= 1024 for v in values):
            msg = "batch sizes are whole numbers from 1 to 1024"
            raise ValueError(msg)
        _unique(values, "batch_sizes")
        return values

    @field_validator("concurrency")
    @classmethod
    def _concurrency(cls, values: list[int]) -> list[int]:
        if any(not 1 <= v <= 256 for v in values):
            msg = "concurrency levels are whole numbers from 1 to 256"
            raise ValueError(msg)
        _unique(values, "concurrency")
        return values

    @field_validator("boundaries")
    @classmethod
    def _boundaries(cls, values: list[bench.Boundary]) -> list[bench.Boundary]:
        _unique(values, "boundaries")
        if "end_to_end" not in values:
            msg = "boundaries must include end_to_end: total_ms is the headline"
            raise ValueError(msg)
        return [b for b in bench.BOUNDARIES if b in values]

    @field_validator("length_buckets", "item_buckets")
    @classmethod
    def _buckets(cls, values: list[int]) -> list[int]:
        if any(v < 1 for v in values):
            msg = "bucket edges are positive whole numbers"
            raise ValueError(msg)
        return _edges(values, "bucket edges")

    @field_validator("require_gpu")
    @classmethod
    def _gpu_name(cls, value: str | None) -> str | None:
        if value is not None and not _GPU_NAME_RE.fullmatch(value):
            msg = "require_gpu is a GPU name fragment: letters, digits, spaces, '.', '_' or '-'"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _fits_the_engine(self) -> BenchmarkHyperparams:
        engine = self.engine_settings
        allowed = _ENGINE_FIELDS.get(engine.engine)
        if allowed is None:
            msg = f"engine {engine.engine!r} cannot be benchmarked; use hf_classify or vllm_chat"
            raise ValueError(msg)
        unapplied = sorted(engine.model_fields_set - allowed - {"engine"})
        if unapplied:
            msg = (
                f"engine_settings may not set {', '.join(unapplied)}: a benchmark does not apply "
                "them (the grid owns batching, concurrency and the rows; one iteration is one "
                "request)"
            )
            raise ValueError(msg)
        if engine.engine == "vllm_chat":
            if self.boundaries != ["end_to_end"]:
                msg = (
                    "engine 'vllm_chat' measures end_to_end only: the forward pass runs inside "
                    "the model server, where this runner cannot time it"
                )
                raise ValueError(msg)
            if self.batch_sizes != [1]:
                msg = "engine 'vllm_chat' sends one request per iteration: batch_sizes must be [1]"
                raise ValueError(msg)
            if engine.max_tokens is None:
                msg = (
                    "engine 'vllm_chat' needs engine_settings.max_tokens: a generation's length "
                    "decides its latency, so it is part of the workload"
                )
                raise ValueError(msg)
        elif self.prefix_caching:
            msg = "prefix_caching applies to engine 'vllm_chat' only"
            raise ValueError(msg)
        elif self.concurrency != [1]:
            msg = "engine 'hf_classify' runs one request stream in this process: concurrency must be [1]"
            raise ValueError(msg)
        if abs(self.order_seed - self.warmup_seed) < self.passes:
            msg = (
                "order_seed and warmup_seed are each offset by the pass index, so they must be at "
                "least `passes` apart, or a pass's warmup would replay another pass's order"
            )
            raise ValueError(msg)
        if self.sample_size is not None and self.sample_size > self.item_cap:
            msg = "sample_size cannot exceed item_cap"
            raise ValueError(msg)
        if len(self.batch_sizes) * len(self.concurrency) > MAX_CONFIGURATIONS:
            msg = f"the grid has more than {MAX_CONFIGURATIONS} configurations"
            raise ValueError(msg)
        if self.item_count_column is None and (self.max_items is not None or self.item_buckets):
            msg = "max_items and item_buckets need item_count_column, the column they count"
            raise ValueError(msg)
        if self.sample_size is not None:
            planned = planned_iterations(self, self.sample_size)
            if planned > MAX_MEASURED_ITERATIONS:
                msg = (
                    f"the grid would measure {planned} iterations; the limit is "
                    f"{MAX_MEASURED_ITERATIONS}"
                )
                raise ValueError(msg)
        return self


def planned_iterations(hp: BenchmarkHyperparams, sample: int) -> int:
    """The measured iterations a sample of ``sample`` rows plans over every configuration."""
    return sum(hp.passes * (sample // b) for b in hp.batch_sizes for _ in hp.concurrency)


class BenchmarkSpec(BaseModel):
    """The inert benchmark spec the control plane delivers in ``STRATA_RUN_CONFIG`` (no secrets)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    engine_version: str | None = None
    model_id: str
    # Both pins are required: a benchmark compares like with like, which a moving revision
    # cannot promise.
    model_revision: str
    dataset_id: str
    dataset_commit_sha: str
    dataset_config: str | None = None
    split: str
    column_mapping: dict[str, str] = Field(default_factory=dict[str, str])
    template: str
    hyperparams: BenchmarkHyperparams
    output_repo_id: str | None = None
    progress_path: str | None = None
    run_id: str | None = None

    def batch_spec(self) -> RunSpec:
        """The same rows, model and engine as the batch runner's spec, for its shared steps."""
        return RunSpec(
            engine_version=self.engine_version,
            model_id=self.model_id,
            model_revision=self.model_revision,
            dataset_id=self.dataset_id,
            dataset_commit_sha=self.dataset_commit_sha,
            dataset_config=self.dataset_config,
            split=self.split,
            column_mapping=self.column_mapping,
            template=self.template,
            hyperparams=self.hyperparams.engine_settings,
            output_repo_id=self.output_repo_id,
            progress_path=self.progress_path,
            run_id=self.run_id,
        )


def load_spec(*, writer: JsonlProgressWriter | None = None) -> BenchmarkSpec:
    spec = load_config(BenchmarkSpec, writer=writer)
    validate_repo_id(spec.model_id, "model")
    validate_repo_id(spec.dataset_id, "dataset")
    require_commit_sha(spec.model_revision, "model_revision")
    require_commit_sha(spec.dataset_commit_sha, "dataset_commit_sha")
    if spec.output_repo_id is not None:
        validate_repo_id(spec.output_repo_id, "output repo")
    _validate_template_coverage(spec.batch_spec())
    return spec


# ------------------------------ workload ------------------------------


def _item_counts(rows: Sequence[dict[str, Any]], column: str | None) -> list[int | None] | None:
    """Each row's item count: the length of its list in ``column``, ``None`` when not a list."""
    if column is None:
        return None
    if rows and column not in rows[0]:
        msg = f"item_count_column {column!r} is not a column of the split"
        raise RunError(msg)
    counts: list[int | None] = []
    for row in rows:
        value: object = row.get(column)
        counts.append(len(cast("list[object]", value)) if isinstance(value, list | tuple) else None)
    return counts


def _chat_lengths(
    snapshot: Path, prompts: Sequence[str], kwargs: dict[str, Any] | None = None
) -> list[int | None]:
    """Each prompt's token count as the model server will see it: after the chat template.

    ``kwargs`` are the run's chat-template kwargs (``enable_thinking``), which can change what the
    template renders and so how long the prompt is.
    """
    transformers: Any = import_module("transformers")
    tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, **tokenizer_load_kwargs(False)
    )
    templated = bool(getattr(tokenizer, "chat_template", None))
    out: list[int | None] = []
    for prompt in prompts:
        if len(prompt) > _MAX_RENDERED_CHARS:
            out.append(None)
            continue
        ids: Any
        if templated:
            # transformers returns the ids alone or, by default from v5, a mapping holding them.
            encoded: Any = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                **(kwargs or {}),
            )
            ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        else:
            ids = tokenizer(prompt)["input_ids"]
        out.append(len(cast("list[int]", ids)))
    return out


# ------------------------------ steps ------------------------------


def _reference_step(classifier: Classifier, texts: list[str], pairs: list[str]) -> bench.Step:
    """A third-party reference model's request, through a plain AutoModel classifier."""

    async def _step(rows: list[int]) -> bench.StepTiming:
        start, end, forward = await asyncio.to_thread(
            classifier.timed_forward,
            [texts[i] for i in rows],
            [pairs[i] for i in rows],
            clock=time.perf_counter_ns,
        )
        return bench.StepTiming(start_ns=start, end_ns=end, forward_ms=forward)

    return _step


def _verifier_step(scorer: VerifierScorer, cells: list[tuple[Any, Any]]) -> bench.Step:
    """A trained verifier's request, from the raw cells to every score it ships."""

    async def _step(rows: list[int]) -> bench.StepTiming:
        start, end, forward = await asyncio.to_thread(
            timed_verifier_request,
            scorer,
            [cells[i] for i in rows],
            clock=time.perf_counter_ns,
        )
        return bench.StepTiming(start_ns=start, end_ns=end, forward_ms=forward)

    return _step


def _chat_step(client: LLMClient, prompts: list[str], hp: Hyperparams) -> bench.Step:
    extras = _sampling_extras(hp)

    async def _step(rows: list[int]) -> bench.StepTiming:
        messages = [UserMessage(content=prompts[rows[0]])]
        start = time.perf_counter_ns()
        response = await client.complete(
            messages,
            temperature=hp.temperature,
            max_tokens=hp.max_tokens,
            top_p=hp.top_p,
            provider_extras=extras,
        )
        end = time.perf_counter_ns()
        return bench.StepTiming(
            start_ns=start, end_ns=end, output_tokens=response.usage.output_tokens
        )

    return _step


# ------------------------------ measurement ------------------------------


class _Progress:
    """Throttled step events: a write per second at most, never one per iteration.

    They carry no GPU counters: sampling those shells out to ``nvidia-smi``, and no process is
    started on the machine while iterations are being timed.
    """

    def __init__(self, writer: JsonlProgressWriter | None, total: int) -> None:
        self._writer = writer
        self.total = total
        self._done_before = 0
        self._last = 0.0

    def finished(self, outcome: bench.ConfigurationOutcome) -> None:
        self._done_before += len(outcome.iterations)
        self._report(outcome, force=True)

    def iteration(self, outcome: bench.ConfigurationOutcome) -> None:
        self._report(outcome, force=False)

    def _report(self, outcome: bench.ConfigurationOutcome, *, force: bool) -> None:
        now = time.monotonic()
        if not force and now - self._last < _PROGRESS_INTERVAL_S:
            return
        self._last = now
        done = self._done_before + (0 if force else len(outcome.iterations))
        emit(
            self._writer,
            ProgressEvent(
                kind="step",
                stage="run",
                step=done,
                total_steps=self.total,
                metrics={
                    "configuration": float(outcome.configuration.index),
                    "batch_size": float(outcome.configuration.batch_size),
                    "concurrency": float(outcome.configuration.concurrency),
                    "failed": float(outcome.failed),
                },
            ),
        )


class _GpuGuard:
    """Whether another process holds the GPU, and the GPUs' state around every timed pass.

    Checked once before anything is loaded (a busy GPU refuses the run) and again before every
    pass's warmup (a busy GPU stops the run as incomplete, ``gpu_busy``). The probes run between
    passes, never while an iteration is timed. Clocks are recorded, never locked.
    """

    def __init__(self) -> None:
        self.checked = True
        self.configuration = 0
        self.telemetry: list[bench.PassTelemetry] = []

    @staticmethod
    def _others() -> set[int] | None:
        pids = compute_pids()
        return None if pids is None else pids - own_processes(os.getpid())

    async def require_idle(self) -> None:
        others = await asyncio.to_thread(self._others)
        if others is None:
            self.checked = False
            return
        if others:
            msg = (
                f"{len(others)} other process(es) hold a compute context on this GPU; a latency "
                "benchmark needs the GPU to itself"
            )
            raise RunError(msg)

    async def before_pass(self, pass_index: int) -> str | None:
        others = await asyncio.to_thread(self._others)
        if others is None:
            self.checked = False
        await self._record(pass_index, "start")
        return "gpu_busy" if others else None

    async def after_pass(self, pass_index: int) -> str | None:
        await self._record(pass_index, "end")
        return None

    async def _record(self, pass_index: int, at: Literal["start", "end"]) -> None:
        state = await asyncio.to_thread(gpu_state)
        if state is None:
            return
        self.telemetry.append(
            bench.PassTelemetry(
                configuration=self.configuration,
                pass_index=pass_index,
                at=at,
                gpus=tuple(bench.GpuState.model_validate(g) for g in state[:16]),
            )
        )


async def _measure(
    spec: BenchmarkSpec,
    sample: list[int],
    step: bench.Step,
    *,
    progress: _Progress,
    guard: _GpuGuard,
    is_alive: Callable[[], Any] | None = None,
) -> tuple[list[bench.ConfigurationOutcome], str | None]:
    """Every configuration in grid order, under one wall-clock cap. Returns the stop reason.

    A stop (the cap, a busy GPU, a model server that died) leaves the configurations after it
    ``not_measured`` with that reason, so the measured ones are still summarized and pushed.
    """
    hp = spec.hyperparams
    deadline = time.perf_counter_ns() + int(hp.wall_clock_cap_s * 1e9)
    outcomes: list[bench.ConfigurationOutcome] = []
    stopped: str | None = None
    for configuration in bench.configurations(hp.batch_sizes, hp.concurrency):
        if stopped is not None:
            outcomes.append(
                bench.ConfigurationOutcome(
                    configuration=configuration,
                    status="not_measured",
                    passes_planned=hp.passes,
                    error=stopped,
                )
            )
            continue
        if is_alive is not None and not await is_alive():
            stopped = _SERVER_DIED
            outcomes.append(
                bench.ConfigurationOutcome(
                    configuration=configuration,
                    status="not_measured",
                    passes_planned=hp.passes,
                    error=stopped,
                )
            )
            continue
        guard.configuration = configuration.index
        outcome = await bench.measure_configuration(
            configuration,
            sample=sample,
            passes=hp.passes,
            warmup=hp.warmup,
            order_seed=hp.order_seed,
            warmup_seed=hp.warmup_seed,
            step=step,
            clock_ns=time.perf_counter_ns,
            deadline_ns=deadline,
            on_iteration=progress.iteration,
            before_pass=guard.before_pass,
            after_pass=guard.after_pass,
        )
        progress.finished(outcome)
        outcomes.append(outcome)
        if outcome.status == "wall_clock_cap":
            stopped = "the wall-clock cap was reached before this configuration started"
        elif outcome.status == "gpu_busy":
            stopped = (
                "gpu_busy: another process took the GPU before an earlier configuration's pass"
            )
        elif (
            outcome.status == "stopped_on_failure" and is_alive is not None and not await is_alive()
        ):
            # The failure was the model server dying; the configurations after it cannot run.
            stopped = _SERVER_DIED
    return outcomes, stopped


_SERVER_DIED = "the model server died (its log is in the job directory)"


def _incomplete(
    results: Sequence[bench.ConfigurationResult],
) -> tuple[str | None, tuple[bench.IncompleteCode, ...]]:
    """Why the run is incomplete, in a sentence and as codes; ``(None, ())`` when it is complete."""
    total = len(results)
    measured = sum(1 for r in results if r.status == "measured" and not r.iterations_failed)
    if measured == total:
        return None, ()
    codes: list[bench.IncompleteCode] = []
    parts: list[str] = []
    busy = [r for r in results if r.status == "gpu_busy"]
    if busy:
        codes.append("gpu_busy")
        parts.append(
            f"gpu_busy: another process took the GPU before a pass of configuration "
            f"{busy[0].index}, so the run stopped there"
        )
    capped = [r for r in results if r.status == "wall_clock_cap"]
    if capped:
        codes.append("wall_clock_cap")
        parts.append(f"the wall-clock cap cut configuration {capped[0].index} short")
    failed = [r for r in results if r.status == "stopped_on_failure"]
    if failed:
        codes.append("stopped_on_failure")
        parts.append(f"{len(failed)} configuration(s) stopped on a failed iteration")
    died = [r for r in results if r.status == "not_measured" and r.error == _SERVER_DIED]
    if died:
        codes.append("server_died")
        parts.append("the model server died before the remaining configurations")
    skipped = [r for r in results if r.status == "not_measured"]
    if skipped:
        codes.append("not_measured")
        parts.append(f"{len(skipped)} of {total} configuration(s) were not measured")
    return "; ".join(parts), tuple(codes)


# ------------------------------ outputs ------------------------------


def _write_timings(rows: list[dict[str, Any]], outdir: Path) -> Path:
    datasets_mod: Any = __import__("datasets")  # the [hf] extra, already needed to load the split
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / TIMINGS_FILENAME
    datasets_mod.Dataset.from_list(rows).to_parquet(str(path))
    return path


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _model_path(
    hp: BenchmarkHyperparams,
) -> Literal["strata_verifier", "reference_automodel", "vllm_chat"]:
    if hp.engine_settings.engine == "vllm_chat":
        return "vllm_chat"
    return "strata_verifier" if hp.engine_settings.artifact is not None else "reference_automodel"


def _protocol(spec: BenchmarkSpec, sample: dict[str, int]) -> dict[str, Any]:
    hp = spec.hyperparams
    engine = hp.engine_settings.engine
    path = _model_path(hp)
    definitions = _DEFINITIONS[path]
    return {
        "engine": engine,
        "model_path": path,
        "boundaries": {b: definitions[b] for b in hp.boundaries},
        "headline": (
            "per configuration, the median across completed passes of each pass's end-to-end "
            "(total_ms) p95, with the minimum and maximum of those p95s; read only from a complete "
            "run whose GPU was checked idle at the start and before every pass"
        ),
        "clock": "time.perf_counter_ns",
        "grid": {"batch_sizes": hp.batch_sizes, "concurrency": hp.concurrency},
        "configurations": [
            {"index": c.index, "batch_size": c.batch_size, "concurrency": c.concurrency}
            for c in bench.configurations(hp.batch_sizes, hp.concurrency)
        ],
        "warmup": {
            "iterations": hp.warmup,
            "unit": "requests at batch size 1, full batches above it",
            "scope": "before_every_timed_pass, untimed, no restart and no cache clear",
            "source": "random.Random(warmup_seed + pass_index).shuffle of the sample, discarded",
        },
        "passes": hp.passes,
        "order": "random.Random(order_seed + pass_index).shuffle of the sample; not length-sorted",
        "batching": (
            "consecutive full batches of the arrival order, padded to the longest pair in the "
            "batch; the rows that do not fill a last batch are counted, not timed"
        ),
        "sampling": (
            "every row of the pinned split (refused past item_cap); with sample_size, "
            "random.Random(sample_seed).sample of the eligible rows, in row order"
        ),
        "item_cap": hp.item_cap,
        "seeds": {
            "sample_seed": hp.sample_seed,
            "order_seed": hp.order_seed,
            "warmup_seed": hp.warmup_seed,
            "bootstrap_seed": hp.bootstrap_seed if hp.bootstrap_resamples is not None else None,
        },
        "sample": sample,
        "wall_clock_cap_s": float(hp.wall_clock_cap_s),
        "on_failure": "an iteration that raises stops its configuration",
        "gpu": {
            "idle_check": (
                "nvidia-smi compute processes on the visible GPUs, other than this runner and its "
                "model server: refused at the start, gpu_busy (incomplete) before a pass"
            ),
            "clocks": "not locked; SM/memory clocks, power limit and draw, temperature and "
            "persistence mode recorded at the start and end of every pass",
        },
        "exclusions": {
            "max_input_tokens": hp.max_input_tokens,
            "item_count_column": hp.item_count_column,
            "max_items": hp.max_items,
        },
        "buckets": {"length": hp.length_buckets, "items": hp.item_buckets},
        "statistics": {
            "percentiles": "per completed pass p50/p95/p99 (Hyndman-Fan type 7), their median "
            "across completed passes with min and max, and the pooled percentiles; a truncated "
            "pass is kept in timings.parquet (pass_completed false) and never summarized",
            "bootstrap": (
                {
                    "resamples": hp.bootstrap_resamples,
                    "seed": hp.bootstrap_seed,
                    "role": "diagnostic only: never the headline, never a pass/fail input",
                }
                if hp.bootstrap_resamples is not None
                else None
            ),
            "throughput": "items per second over the union of the measured intervals",
        },
        "require_gpu": hp.require_gpu,
    }


def _build_manifest(
    spec: BenchmarkSpec,
    *,
    model_commit: str | None,
    engine_facts: dict[str, Any],
    environment: dict[str, Any],
    cold_start: dict[str, float | None],
    counts: dict[str, Any],
    summary: bench.BenchmarkSummary,
    compute: dict[str, Any],
) -> dict[str, Any]:
    """The run's manifest (``kind`` ``benchmark``): pins, protocol, receipt and counts, no row text."""
    engine_settings = spec.hyperparams.engine_settings.model_dump(exclude={"pair_template"})
    hyperparams = spec.hyperparams.model_dump(exclude={"engine_settings"})
    return {
        "schema": MANIFEST_SCHEMA,
        "kind": _MANIFEST_KIND,
        "engine": manifest_engine(spec.engine_version, ()),
        "model": {
            "id": spec.model_id,
            "requested_revision": spec.model_revision,
            "commit": model_commit,
        },
        "dataset": {
            "id": spec.dataset_id,
            "config": spec.dataset_config,
            "split": spec.split,
            "requested_revision": spec.dataset_commit_sha,
            "commit": spec.dataset_commit_sha,
        },
        "benchmark": _protocol(spec, summary.sample),
        "engine_facts": engine_facts,
        "environment": environment,
        "cold_start": cold_start,
        "counts": counts,
        "complete": summary.complete,
        "incomplete_reason": summary.incomplete_reason,
        "incomplete_codes": list(summary.incomplete_codes),
        "headline_readable": summary.headline_readable,
        "gpu_idle_check": summary.gpu_idle_check,
        "gpu_telemetry": [t.model_dump(mode="json") for t in summary.gpu_telemetry],
        "files": {"timings": TIMINGS_FILENAME, "summary": SUMMARY_FILENAME},
        "template_sha256": _digest(spec.template),
        "pair_template_sha256": _digest(spec.hyperparams.engine_settings.pair_template or ""),
        "column_mapping": spec.column_mapping,
        "hyperparams": {**hyperparams, "engine_settings": engine_settings},
        "compute": compute,
    }


async def _push(spec: BenchmarkSpec, outdir: Path, hf_token: str) -> str:
    """Push the timings, the summary and the manifest in ONE commit to the run's private repo."""
    out_repo = validate_repo_id(cast("str", spec.output_repo_id), "output repo")
    hub = HFHubClient(token=hf_token)  # EXPLICIT write token, never the ambient HF_TOKEN
    await _ensure_private_output(hub, out_repo)
    await _require_fresh_output(spec.batch_spec(), hf_token)
    await hub.upload_files(
        {
            TIMINGS_FILENAME: outdir / TIMINGS_FILENAME,
            SUMMARY_FILENAME: outdir / SUMMARY_FILENAME,
            MANIFEST_FILENAME: outdir / MANIFEST_FILENAME,
        },
        out_repo,
        repo_type="dataset",
        commit_message="benchmark measurements",
    )
    return out_repo


# ------------------------------ execution ------------------------------


async def _execute(
    spec: BenchmarkSpec, secrets: RunSecrets, writer: JsonlProgressWriter | None
) -> str:
    started = time.monotonic()
    hf_token = secrets.hf_token_value()
    hub_token = secrets.hub_credential()
    hp = spec.hyperparams
    engine = hp.engine_settings.engine
    batch_spec = spec.batch_spec()
    phase = phase_sink(writer, hf_token, gpu=GpuSampler())
    # The sink used while iterations are timed: no GPU counters, so no probe process runs then.
    quiet = phase_sink(writer, hf_token)
    pushing = bool(hf_token and spec.output_repo_id)
    if hp.bootstrap_resamples is not None:
        # Before anything is measured: the intervals are computed after the last iteration.
        try:
            import_module("numpy")
        except ImportError as exc:
            msg = "bootstrap_resamples needs numpy: install 'strata-forge[scoring]'"
            raise RunError(msg) from exc

    if pushing:
        async with ticking_phase(phase, "Checking the output repo", stage="load_model"):
            await _check_output_repo(batch_spec, hf_token)
            await _require_fresh_output(batch_spec, cast("str", hf_token))
    # Before anything is loaded: whether the GPU was idle is a fact about the machine as found.
    environment = await asyncio.to_thread(receipt, use_torch=False)
    guard = _GpuGuard()
    await guard.require_idle()
    if environment.get("gpus"):
        # Refused before any download when nvidia-smi already names the GPUs; a machine it cannot
        # describe is checked against torch's answer once the model is loaded.
        _require_gpu(spec, environment)
    chat_settings: Hyperparams | None = None
    if engine == "vllm_chat":
        # The checkpoint's config alone, before any large download: a reasoning parser that does
        # not fit the architecture is refused in seconds, as in a batch run.
        config, _ = await _read_model_config(batch_spec, hub_token)
        chat_settings = _effective_hyperparams(hp.engine_settings, config)
    # Every row of the pinned split, read up to one past the cap so a larger split is refused
    # without materializing it whole.
    bounded = batch_spec.model_copy(
        update={
            "hyperparams": batch_spec.hyperparams.model_copy(update={"row_limit": hp.item_cap + 1})
        }
    )
    async with ticking_phase(phase, "Loading the workload", stage="load_model"):
        rows = await asyncio.to_thread(_load_rows, bounded, hub_token)
    if not rows:
        msg = "the workload split is empty"
        raise RunError(msg)
    if len(rows) > hp.item_cap:
        msg = (
            f"the split holds more than item_cap ({hp.item_cap}) rows; every row is measured, so "
            "raise item_cap or pin a smaller split"
        )
        raise RunError(msg)
    items = _item_counts(rows, hp.item_count_column)
    async with ticking_phase(phase, "Downloading the model", stage="load_model"):
        snapshot = await _fetch_model(batch_spec, hub_token)
    model_commit = snapshot_commit(snapshot) or spec.model_revision

    if chat_settings is None:
        return await _execute_classifier(
            spec,
            rows,
            items,
            snapshot,
            model_commit,
            environment=environment,
            guard=guard,
            writer=writer,
            quiet=quiet,
            phase=phase,
            hf_token=hf_token,
            started=started,
        )
    return await _execute_chat(
        spec,
        rows,
        items,
        snapshot,
        model_commit,
        environment=environment,
        guard=guard,
        writer=writer,
        quiet=quiet,
        phase=phase,
        hf_token=hf_token,
        hp=chat_settings,
        started=started,
    )


def _admit(
    spec: BenchmarkSpec,
    lengths: list[int | None],
    items: list[int | None] | None,
    model_limit: int | None,
) -> tuple[list[int], dict[str, int], dict[str, int]]:
    hp = spec.hyperparams
    eligible, excluded = bench.admit(
        lengths,
        items,
        model_limit=model_limit,
        max_input_tokens=hp.max_input_tokens,
        max_items=hp.max_items,
    )
    sample = bench.draw_sample(eligible, hp.sample_size, hp.sample_seed)
    if not sample:
        msg = f"no row of the workload is eligible to measure; excluded: {excluded}"
        raise RunError(msg)
    planned = planned_iterations(hp, len(sample))
    if planned > MAX_MEASURED_ITERATIONS:
        msg = f"the grid would measure {planned} iterations; the limit is {MAX_MEASURED_ITERATIONS}"
        raise RunError(msg)
    counts = {"rows": len(lengths), "eligible": len(eligible), "sampled": len(sample)}
    return sample, excluded, counts


def _require_gpu(spec: BenchmarkSpec, environment: dict[str, Any]) -> None:
    required = spec.hyperparams.require_gpu
    if required is not None and not gpu_matches(environment, required):
        gpus: list[dict[str, Any]] = environment.get("gpus") or []
        names = [gpu.get("name") for gpu in gpus] or ["none reported"]
        msg = f"the run requires a GPU named like {required!r}; this machine reports {names}"
        raise RunError(msg)


async def _finish(
    spec: BenchmarkSpec,
    outcomes: list[bench.ConfigurationOutcome],
    *,
    lengths: list[int | None],
    items: list[int | None] | None,
    excluded: dict[str, int],
    counts: dict[str, int],
    environment: dict[str, Any],
    guard: _GpuGuard,
    engine_facts: dict[str, Any],
    cold_start: dict[str, float | None],
    model_commit: str | None,
    writer: JsonlProgressWriter | None,
    phase: PhaseSink,
    hf_token: str | None,
    started: float,
    stopped: str | None = None,
) -> str:
    """Summarize, write the three files, push them in one commit, and report the verdict.

    ``stopped`` is the model server's death when that ended the run: the measured configurations
    are still written and pushed, and the run then fails as incomplete.
    """
    hp = spec.hyperparams
    tokens = [length or 0 for length in lengths]
    # An iteration's exception text reaches summary.json, which is pushed beside the timings.
    redactor = run_redactor(hf_token)
    for outcome in outcomes:
        if outcome.error is not None:
            outcome.error = redactor.redact(outcome.error)[:500]

    def length_of(row: int) -> str | None:
        return bench.bucket_label(tokens[row], hp.length_buckets) if hp.length_buckets else None

    def items_of(row: int) -> str | None:
        if items is None or not hp.item_buckets or items[row] is None:
            return None
        return bench.bucket_label(int(items[row] or 0), hp.item_buckets)

    bootstrap = (
        (hp.bootstrap_seed, hp.bootstrap_resamples) if hp.bootstrap_resamples is not None else None
    )
    results = await asyncio.to_thread(
        lambda: [
            bench.summarize(
                outcome,
                boundaries=hp.boundaries,
                tokens=tokens,
                length_of=length_of,
                items_of=items_of,
                bootstrap=bootstrap,
            )
            for outcome in outcomes
        ]
    )
    reason, codes = _incomplete(results)
    if reason is None:
        unreadable = (
            None
            if guard.checked
            else (
                "the GPU's idleness could not be checked (no nvidia-smi), so this is not a T13 read"
            )
        )
    else:
        unreadable = "an incomplete run has no T13 read"
    gpu_count = int(short_receipt(environment).get("count") or 0)
    wall = round(time.monotonic() - started, 3)
    compute = {
        "wall_seconds": wall,
        "gpu_count": gpu_count,
        "gpu_seconds": round(wall * gpu_count, 3),
    }
    summary = bench.BenchmarkSummary(
        # The spec admits only these two engines (`_ENGINE_FIELDS`).
        engine=cast("Literal['hf_classify', 'vllm_chat']", hp.engine_settings.engine),
        boundaries=tuple(hp.boundaries),
        complete=reason is None,
        incomplete_reason=reason,
        incomplete_codes=codes,
        headline_readable=reason is None and guard.checked,
        model_path=_model_path(hp),
        headline_unreadable_reason=unreadable,
        gpu_idle_check="checked" if guard.checked else "unavailable",
        gpu_telemetry=tuple(guard.telemetry),
        sample=counts,
        excluded=excluded,
        cold_start=cold_start,
        gpu=short_receipt(environment),
        configurations=tuple(results),
    )
    measured = sum(r.iterations_measured for r in results)
    all_counts = {**counts, "excluded": excluded, "iterations_measured": measured}
    outdir = results_dir(spec.run_id, name=_RESULTS_DIR_NAME)
    timing_rows = bench.iteration_rows(
        outcomes,
        boundaries=hp.boundaries,
        tokens=tokens,
        items=items,
        length_of=length_of,
        items_of=items_of,
    )
    async with ticking_phase(phase, "Writing measurements", stage="push"):
        await asyncio.to_thread(_write_timings, timing_rows, outdir)
        summary_json = summary.model_dump(mode="json")
        await asyncio.to_thread(_write_json, summary_json, outdir / SUMMARY_FILENAME)
        await asyncio.to_thread(_write_json, summary_json, Path.cwd() / WORKDIR_SUMMARY_FILENAME)
        manifest = _build_manifest(
            spec,
            model_commit=model_commit,
            engine_facts=engine_facts,
            environment=environment,
            cold_start=cold_start,
            counts=all_counts,
            summary=summary,
            compute=compute,
        )
        await asyncio.to_thread(write_manifest, manifest, outdir)
    if hf_token and spec.output_repo_id:
        try:
            async with ticking_phase(phase, "Uploading measurements to the Hub", stage="push"):
                destination = await _push(spec, outdir, hf_token)
        except Exception as exc:
            msg = f"measurements are on the VM at {outdir}, but the push failed: {exc}"
            raise RunError(msg) from exc
    else:
        destination = str(outdir)
    configured = sum(1 for r in results if r.iterations_measured)
    emit(
        writer,
        ProgressEvent(
            kind="end",
            stage="push",
            step=measured,
            total_steps=sum(r.iterations_planned for r in results),
            metrics={
                "iterations_measured": float(measured),
                "configurations_measured": float(configured),
                "configurations": float(len(results)),
                "complete": 1.0 if summary.complete else 0.0,
                "headline_readable": 1.0 if summary.headline_readable else 0.0,
                "gpu_count": float(gpu_count),
                "gpu_seconds": float(compute["gpu_seconds"]),
            },
            message=destination,
        ),
    )
    if not measured:
        failures = [r.error for r in results if r.error]
        msg = f"no configuration measured an iteration: {failures[0] if failures else reason}"
        raise RunError(msg)
    if stopped == _SERVER_DIED:
        msg = f"incomplete: {_SERVER_DIED}; the measured configurations are at {destination}"
        raise RunError(msg)
    return destination


async def _execute_classifier(
    spec: BenchmarkSpec,
    rows: list[dict[str, Any]],
    items: list[int | None] | None,
    snapshot: Path,
    model_commit: str | None,
    *,
    environment: dict[str, Any],
    guard: _GpuGuard,
    writer: JsonlProgressWriter | None,
    quiet: PhaseSink,
    phase: PhaseSink,
    hf_token: str | None,
    started: float,
) -> str:
    hp = spec.hyperparams
    settings = hp.engine_settings
    inputs = build_inputs(spec.batch_spec(), rows)
    load_started = time.monotonic()
    verifier: VerifierScorer | None = None
    classifier: Classifier
    if settings.artifact is not None:
        async with ticking_phase(phase, "Loading the verifier", stage="load_model"):
            verifier = await asyncio.to_thread(
                load_verifier,
                snapshot,
                label_alias=settings.label_alias,
                max_length=settings.max_model_len,
                dtype=cast("str", settings.dtype),
                apply_calibration=settings.apply_calibration == "auto",
            )
        classifier = verifier
    else:
        async with ticking_phase(phase, "Loading the reference classifier", stage="load_model"):
            classifier = await asyncio.to_thread(
                load_classifier,
                snapshot,
                label_alias=cast("dict[str, str | None]", settings.label_alias),
                max_length=settings.max_model_len,
                dtype=cast("str", settings.dtype),
            )
    load_seconds = round(time.monotonic() - load_started, 3)
    environment = {
        **environment,
        "torch": (await asyncio.to_thread(receipt, use_torch=True))["torch"],
    }
    _require_gpu(spec, environment)
    # Each row's token count with this model's own tokenizer and template (for a verifier, its
    # own rendering), at run time: the exclusions and the length buckets are this model's.
    async with ticking_phase(phase, "Measuring the workload", stage="load_model"):
        if verifier is not None:
            lengths = await asyncio.to_thread(VerifierRunScorer(verifier).lengths, inputs)
        else:
            lengths = await asyncio.to_thread(classifier.lengths, inputs.texts, inputs.pairs)
    sample, excluded, counts = _admit(spec, lengths, items, classifier.max_length)
    progress = _Progress(writer, planned_iterations(hp, len(sample)))
    emit(writer, ProgressEvent(kind="start", stage="run", total_steps=progress.total))
    step = (
        _verifier_step(verifier, inputs.cells or [])
        if verifier is not None
        else _reference_step(classifier, inputs.texts, inputs.pairs)
    )
    async with ticking_phase(quiet, "Measuring", stage="run"):
        outcomes, _ = await _measure(spec, sample, step, progress=progress, guard=guard)
    path_facts: dict[str, Any] = (
        {
            "model_path": "strata_verifier",
            "verifier_sha256": verifier.verifier_sha256,
            "heads": verifier.heads,
            "decoder": verifier.rendering.decoder,
            "calibration_temperature_applied": verifier.temperature,
        }
        if verifier is not None
        else {
            "model_path": "reference_automodel",
            "label": "third-party reference model, run as a plain AutoModelForSequenceClassification",
        }
    )
    engine_facts = {
        **path_facts,
        "engine": "hf_classify",
        "dtype": classifier.dtype,
        "tf32": classifier.tf32,
        "device": classifier.device,
        "attention": classifier.attention,
        "forward_timer": classifier.forward_timer().method,
        "max_length": classifier.max_length,
        "max_length_source": classifier.max_length_source,
        "padding_side": "right",
        "pad_token": classifier.pad_token,
        "labels": classifier.labels,
    }
    first = next((o.first_iteration_ms for o in outcomes if o.first_iteration_ms is not None), None)
    return await _finish(
        spec,
        outcomes,
        lengths=lengths,
        items=items,
        excluded=excluded,
        counts=counts,
        environment=environment,
        guard=guard,
        engine_facts=engine_facts,
        cold_start={"load_seconds": load_seconds, "first_iteration_ms": first},
        model_commit=model_commit,
        writer=writer,
        phase=phase,
        hf_token=hf_token,
        started=started,
    )


async def _execute_chat(
    spec: BenchmarkSpec,
    rows: list[dict[str, Any]],
    items: list[int | None] | None,
    snapshot: Path,
    model_commit: str | None,
    *,
    environment: dict[str, Any],
    guard: _GpuGuard,
    writer: JsonlProgressWriter | None,
    quiet: PhaseSink,
    phase: PhaseSink,
    hf_token: str | None,
    hp: Hyperparams,
    started: float,
) -> str:
    _require_gpu(spec, environment)
    prompts = [render_template(spec.template, row, spec.column_mapping) for row in rows]
    async with ticking_phase(phase, "Measuring the workload", stage="load_model"):
        lengths = await asyncio.to_thread(
            _chat_lengths, snapshot, prompts, cast("dict[str, Any] | None", hp.chat_template_kwargs)
        )
    sample, excluded, counts = _admit(spec, lengths, items, hp.max_model_len)
    harmony_date = _harmony_date(hp)
    task = build_vllm_task(
        str(snapshot),
        served_model_name=spec.model_id,
        port=_SERVE_PORT,
        host=_SERVE_HOST,
        tensor_parallel_size=hp.tensor_parallel_size,
        max_model_len=hp.max_model_len,
        dtype=hp.dtype,
        setup="",
        python_executable=sys.executable,
        extra_args=(
            *_serve_args(hp),
            *(() if spec.hyperparams.prefix_caching else ("--no-enable-prefix-caching",)),
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
                **({_HARMONY_DATE_ENV: harmony_date} if harmony_date else {}),
            }
        }
    )
    load_started = time.monotonic()
    async with serving_endpoint(
        LocalBackend(log_dir=Path.cwd(), env_inherit=False),
        task,
        base_url=f"http://{_SERVE_HOST}:{_SERVE_PORT}/v1",
        wait_timeout_s=hp.wait_timeout_s,
        on_phase=lambda message: phase(message, stage="load_model"),
    ) as endpoint:
        load_seconds = round(time.monotonic() - load_started, 3)
        engine_facts: dict[str, Any] = {
            "engine": "vllm_chat",
            "max_model_len": await _served_max_model_len(endpoint.base_url, spec.model_id),
            "dtype": {
                "requested": hp.dtype,
                "checkpoint": await asyncio.to_thread(_checkpoint_dtype, snapshot),
                "served": await asyncio.to_thread(_served_dtype, Path.cwd()),
            },
            "flags": _serve_flags(task.run),
            "reasoning_parser": hp.reasoning_parser,
            # A declared setting: off unless the spec asks, since passes replay the same prompts.
            "prefix_caching": spec.hyperparams.prefix_caching,
            "harmony_date": harmony_date,
            "client": {"cache": None, "retry_max_attempts": 1},
        }
        provider = OpenAICompatProvider(
            OpenAICompatConfig(
                base_url=endpoint.base_url, api_key=SecretStr(UNAUTHENTICATED_API_KEY)
            )
        )
        # No cache and no retry: a cached answer is not a measurement, and a retried request's
        # time would be two requests reported as one.
        client = LLMClient(
            model=spec.model_id,
            provider="openai_compat",
            provider_clients={"openai_compat": provider},
            retry_max_attempts=1,
        )
        progress = _Progress(writer, planned_iterations(spec.hyperparams, len(sample)))
        emit(writer, ProgressEvent(kind="start", stage="run", total_steps=progress.total))
        async with ticking_phase(quiet, "Measuring", stage="run"):
            outcomes, stopped = await _measure(
                spec,
                sample,
                _chat_step(client, prompts, hp),
                progress=progress,
                guard=guard,
                is_alive=endpoint.is_alive,
            )
    first = next((o.first_iteration_ms for o in outcomes if o.first_iteration_ms is not None), None)
    return await _finish(
        spec,
        outcomes,
        lengths=lengths,
        items=items,
        excluded=excluded,
        counts=counts,
        environment=environment,
        guard=guard,
        engine_facts=engine_facts,
        cold_start={"load_seconds": load_seconds, "first_iteration_ms": first},
        model_commit=model_commit,
        writer=writer,
        phase=phase,
        hf_token=hf_token,
        started=started,
        stopped=stopped,
    )


async def main() -> int:
    """Entry point: returns a process exit code (0 ok, 1 failure). Never leaks the token."""
    return await runner_main(
        lambda writer, secrets: _execute(load_spec(writer=writer), secrets, writer)
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
