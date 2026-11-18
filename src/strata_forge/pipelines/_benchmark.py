"""The engine-independent core of a latency benchmark: plan, measure, summarize.

A benchmark run measures one pinned model on one pinned workload split. This module holds what
does not depend on which engine serves the model:

- **Admission** (:func:`admit`). Rows the protocol excludes are dropped before anything is timed
  and counted by reason: a rendering too long to measure, a pair over the model's own limit, over
  the declared token cap, or over the declared item cap.
- **The plan.** A seeded sample of the eligible rows (:func:`draw_sample`); per timed pass a
  shuffle of it seeded with ``seed + pass_index`` (:func:`pass_order`), served in that arrival
  order; full batches only at a batch size above one, the remainder counted rather than timed as
  a smaller batch (:func:`full_batches`); and warmup iterations drawn from the pass's own order
  and discarded before every timed pass (:func:`warmup_batches`).
- **Measurement** (:func:`measure_configuration`). One configuration (a batch size and a number
  of concurrent request streams) runs its passes through an engine ``step``. The step owns the
  clock and returns the timestamps of the boundaries it measured; this loop never reads a clock
  except to enforce the wall-clock cap, which is checked before each iteration starts. An
  iteration that raises stops its configuration, so a configuration's numbers are either clean or
  marked as not.
- **The summary** (:func:`summarize`). Per configuration and declared boundary, the latency
  summary of :mod:`strata_forge.metrics.latency`, the throughput over the time requests were in
  flight, and, at a batch size of one, the same percentiles per length and item bucket.

Nothing here imports a heavy dependency or reads row text.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from itertools import product
from statistics import median
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strata_forge.metrics.latency import (
    LatencyBootstrap,
    LatencySummary,
    latency_bootstrap,
    latency_summary,
    percentile,
)
from strata_forge.pipelines._common import RunError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

__all__ = [
    "BOUNDARIES",
    "SUMMARY_FORMAT",
    "BenchmarkSummary",
    "Boundary",
    "BoundaryResult",
    "BucketRow",
    "Configuration",
    "ConfigurationOutcome",
    "ConfigurationResult",
    "GpuState",
    "Headline",
    "IncompleteCode",
    "Iteration",
    "NoCacheDiagnostic",
    "PassHook",
    "PassKind",
    "PassTelemetry",
    "PrefixCacheResult",
    "PrefixReuseRecord",
    "Step",
    "StepTiming",
    "Throughput",
    "admit",
    "bucket_label",
    "configurations",
    "draw_sample",
    "full_batches",
    "iteration_rows",
    "measure_configuration",
    "measured_cleanly",
    "pass_order",
    "summarize",
    "union_seconds",
    "warmup_batches",
]

type Boundary = Literal["model_only", "end_to_end"]
#: Every timing boundary, in report order.
BOUNDARIES: tuple[Boundary, ...] = ("model_only", "end_to_end")
SUMMARY_FORMAT = "strata.benchmark.summary/v1"
# The reasons a row is excluded, in the order they are tested.
_EXCLUSIONS = ("unmeasurable", "over_model_limit", "over_max_input_tokens", "over_max_items")
_NS_PER_MS = 1_000_000
_NS_PER_S = 1_000_000_000
# A configuration's error is a message, not a log.
_MAX_ERROR_CHARS = 500


@dataclass(frozen=True, slots=True)
class Configuration:
    """One point of the workload grid: requests of ``batch_size`` rows, ``concurrency`` at once."""

    index: int
    batch_size: int
    concurrency: int


def configurations(batch_sizes: Sequence[int], concurrency: Sequence[int]) -> list[Configuration]:
    """The grid's configurations, batch size major, in the order the spec lists them."""
    return [
        Configuration(index=index, batch_size=batch, concurrency=streams)
        for index, (batch, streams) in enumerate(product(batch_sizes, concurrency))
    ]


def admit(
    lengths: Sequence[int | None],
    item_counts: Sequence[int | None] | None,
    *,
    model_limit: int | None,
    max_input_tokens: int | None,
    max_items: int | None,
) -> tuple[list[int], dict[str, int]]:
    """The eligible row indices, and how many rows each exclusion removed.

    A ``None`` length is a rendering too long to tokenize. A ``None`` item count (the declared
    column held no list) is counted as unmeasurable too, because its item bucket is unknown.
    """
    excluded: dict[str, int] = dict.fromkeys(_EXCLUSIONS, 0)
    eligible: list[int] = []
    for index, length in enumerate(lengths):
        items = item_counts[index] if item_counts is not None else 0
        if length is None or items is None:
            reason = "unmeasurable"
        elif model_limit is not None and length > model_limit:
            reason = "over_model_limit"
        elif max_input_tokens is not None and length > max_input_tokens:
            reason = "over_max_input_tokens"
        elif max_items is not None and items > max_items:
            reason = "over_max_items"
        else:
            eligible.append(index)
            continue
        excluded[reason] += 1
    return eligible, excluded


def draw_sample(eligible: Sequence[int], sample_size: int | None, seed: int) -> list[int]:
    """``sample_size`` of the eligible rows drawn with ``random.Random(sample_seed)``, in row order.

    ``None`` takes every eligible row. A sample larger than the eligible rows is refused: padding
    it out would change the workload the numbers describe.
    """
    if sample_size is None:
        return list(eligible)
    if sample_size > len(eligible):
        msg = (
            f"the workload asks for a sample of {sample_size} rows but the split has only "
            f"{len(eligible)} eligible rows"
        )
        raise RunError(msg)
    rng = random.Random(seed)  # noqa: S311 - a recorded, reproducible draw, not a secret
    return sorted(rng.sample(list(eligible), sample_size))


def pass_order(sample: Sequence[int], seed: int, pass_index: int) -> list[int]:
    """One pass's order: the sample shuffled with ``seed + pass_index``.

    Called with ``order_seed`` for the timed order and with ``warmup_seed`` for the warmup draw.
    """
    order = list(sample)
    random.Random(seed + pass_index).shuffle(order)  # noqa: S311 - recorded, reproducible
    return order


def full_batches(order: Sequence[int], batch_size: int) -> tuple[list[list[int]], int]:
    """Consecutive full batches of ``order``, and how many trailing rows did not fill one."""
    count = len(order) // batch_size
    batches = [list(order[i * batch_size : (i + 1) * batch_size]) for i in range(count)]
    return batches, len(order) - count * batch_size


def warmup_batches(order: Sequence[int], batch_size: int, count: int) -> list[list[int]]:
    """``count`` full batches from the start of ``order`` (the warmup shuffle), wrapping round."""
    if not order:
        return []
    rows = [order[i % len(order)] for i in range(count * batch_size)]
    return [rows[i * batch_size : (i + 1) * batch_size] for i in range(count)]


def bucket_label(value: int, edges: Sequence[int]) -> str:
    """The bucket of ``value`` under ascending upper ``edges``: ``"0-512"``, ``"513-2048"``, ...

    A value over the last edge is ``">8192"``-style. Edges are inclusive upper bounds.
    """
    low = 0
    for edge in edges:
        if value <= edge:
            return f"{low}-{edge}"
        low = edge + 1
    return f">{edges[-1]}" if edges else "all"


@dataclass(frozen=True, slots=True)
class StepTiming:
    """What one engine step measured, on the engine's own nanosecond clock.

    ``start_ns``/``end_ns`` bound the end-to-end request (``total_ms``, the headline); ``forward_ms``
    is the forward pass timed inside the same request (a diagnostic). ``output_tokens`` is a
    generating engine's count.
    """

    start_ns: int
    end_ns: int
    forward_ms: float | None = None
    output_tokens: int | None = None
    # Premise-prefix reuse: whether the request's premise prefix was already cached.
    prefix_cache_hit: bool | None = None


type PassKind = Literal["timed", "no_cache_diagnostic"]
type Step = Callable[[list[int]], Awaitable[StepTiming]]


@dataclass(frozen=True, slots=True)
class Iteration:
    """One measured (never warmup) iteration, without any row text."""

    configuration: int
    pass_index: int
    iteration: int
    rows: tuple[int, ...]
    start_ns: int
    end_ns: int
    forward_ms: float | None
    output_tokens: int | None
    prefix_cache_hit: bool | None = None
    # `no_cache_diagnostic`: the extra pass a prefix-reuse run makes without the cache.
    kind: PassKind = "timed"


type Status = Literal[
    "measured", "stopped_on_failure", "wall_clock_cap", "gpu_busy", "not_measured"
]
type PassHook = Callable[[int], Awaitable[str | None]]


@dataclass(slots=True)
class ConfigurationOutcome:
    """What measuring one configuration produced, before it is summarized."""

    configuration: Configuration
    status: Status
    passes_planned: int
    passes_completed: int = 0
    iterations_planned: int = 0
    warmup_per_pass: int = 0
    remainder_per_pass: int = 0
    failed: int = 0
    error: str | None = None
    first_iteration_ms: float | None = None
    iterations: list[Iteration] = field(default_factory=list[Iteration])


async def measure_configuration(
    configuration: Configuration,
    *,
    sample: Sequence[int],
    passes: int,
    warmup: int,
    order_seed: int,
    warmup_seed: int,
    step: Step,
    clock_ns: Callable[[], int],
    deadline_ns: int,
    on_iteration: Callable[[ConfigurationOutcome], None] | None = None,
    before_pass: PassHook | None = None,
    after_pass: PassHook | None = None,
    kind: PassKind = "timed",
) -> ConfigurationOutcome:
    """Run one configuration's warmups and timed passes through ``step``.

    ``concurrency`` workers take batches from the pass in arrival order, so at most that many
    iterations are in flight. The cap is checked before each iteration starts: in-flight
    iterations finish and count, and the configuration is marked ``wall_clock_cap``. An iteration
    that raises is counted as failed and stops the configuration; a warmup that raises does too,
    since the timed numbers after it would describe a different state.

    ``before_pass`` runs before each pass's warmup and returns a stop reason or ``None``; the only
    one used is ``gpu_busy`` (another process took the GPU), which stops the configuration with
    that status. ``after_pass`` runs after each timed pass. Neither runs while iterations are timed.
    """
    batch = configuration.batch_size
    planned_batches, remainder = full_batches(sample, batch)
    outcome = ConfigurationOutcome(
        configuration=configuration,
        status="measured",
        passes_planned=passes,
        iterations_planned=len(planned_batches) * passes,
        warmup_per_pass=warmup,
        remainder_per_pass=remainder,
    )
    if not planned_batches:
        outcome.status = "not_measured"
        outcome.error = f"the sample of {len(sample)} rows does not fill one batch of {batch}"
        return outcome
    stop: list[str] = []

    async def _run(work: list[tuple[int, list[int]]], *, timed: bool, pass_index: int) -> None:
        queue = iter(work)

        async def _worker() -> None:
            while not stop:
                item = next(queue, None)
                if item is None:
                    return
                # After the queue: a pass that ends exactly at the cap was measured in full.
                if clock_ns() >= deadline_ns:
                    stop.append("wall_clock_cap")
                    return
                index, rows = item
                try:
                    timing = await step(rows)
                except Exception as exc:  # an engine failure is recorded, never swallowed
                    outcome.failed += 1
                    outcome.error = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]
                    stop.append("stopped_on_failure")
                    return
                if outcome.first_iteration_ms is None:
                    outcome.first_iteration_ms = (timing.end_ns - timing.start_ns) / _NS_PER_MS
                if timed:
                    outcome.iterations.append(
                        Iteration(
                            configuration=configuration.index,
                            pass_index=pass_index,
                            iteration=index,
                            rows=tuple(rows),
                            start_ns=timing.start_ns,
                            end_ns=timing.end_ns,
                            forward_ms=timing.forward_ms,
                            output_tokens=timing.output_tokens,
                            prefix_cache_hit=timing.prefix_cache_hit,
                            kind=kind,
                        )
                    )
                    if on_iteration is not None:
                        on_iteration(outcome)

        async with asyncio.TaskGroup() as group:
            for _ in range(configuration.concurrency):
                group.create_task(_worker())

    for pass_index in range(passes):
        if before_pass is not None:
            reason = await before_pass(pass_index)
            if reason is not None:
                stop.append(reason)
                break
        warm = warmup_batches(pass_order(sample, warmup_seed, pass_index), batch, warmup)
        await _run(list(enumerate(warm)), timed=False, pass_index=pass_index)
        if stop:
            break
        batches, _ = full_batches(pass_order(sample, order_seed, pass_index), batch)
        await _run(list(enumerate(batches)), timed=True, pass_index=pass_index)
        if after_pass is not None:
            await after_pass(pass_index)
        if stop:
            break
        outcome.passes_completed += 1
    if stop:
        outcome.status = (
            "stopped_on_failure"
            if "stopped_on_failure" in stop
            else "gpu_busy"
            if "gpu_busy" in stop
            else "wall_clock_cap"
        )
    return outcome


# ------------------------------ summary ------------------------------


class _Frozen(BaseModel):
    # No NaN or infinity anywhere in a summary: the control plane stores it as JSON in Postgres,
    # which refuses both, and a latency that is not a finite number is not a measurement.
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Throughput(_Frozen):
    """Items and tokens per second over the time the boundary was busy.

    ``window_s`` is the measure of the union of the measured intervals: with one stream, the sum
    of the iterations' durations; with several, the time at least one request was in flight. Gaps
    between iterations (progress reporting, scheduling) are not in it.
    """

    items_per_s: float | None
    input_tokens_per_s: float | None
    output_tokens_per_s: float | None
    window_s: float
    items: int = Field(ge=0)


class BucketRow(_Frozen):
    """One bucket's pooled percentiles and its median-of-passes p95 (milliseconds)."""

    bucket: str
    n: int = Field(ge=0)
    p50: float | None
    p95: float | None
    p99: float | None
    p95_median_of_passes: float | None


class BoundaryResult(_Frozen):
    """One declared boundary of one configuration.

    ``end_to_end`` (``total_ms``) is the headline; ``model_only`` (``forward_ms``, the forward pass
    timed inside the same requests) is a diagnostic and never a headline.
    """

    boundary: Boundary
    role: Literal["headline", "diagnostic"]
    latency: LatencySummary
    bootstrap: LatencyBootstrap | None
    throughput: Throughput
    # Only at a batch size of one, where an iteration is a single row with a single bucket.
    by_length: tuple[BucketRow, ...] | None
    by_items: tuple[BucketRow, ...] | None


class Headline(_Frozen):
    """The protocol's headline for one configuration: the median of the per-pass end-to-end p95,
    with the minimum and maximum of the per-pass p95 beside it (prereg §14, ``vm.md`` §5.6)."""

    p95_median_ms: float | None
    p95_min_ms: float | None
    p95_max_ms: float | None
    passes: int = Field(ge=0)
    unit: Literal["ms_per_request", "ms_per_batch"]


class PrefixCacheResult(_Frozen):
    """How a prefix-reuse configuration's summarized requests met the premise-prefix cache.

    A miss filled its prefix inside its own clock; a hit reused it. The latencies are end to end
    (``total_ms``), pooled over the completed passes, split by hit and miss.
    """

    hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    hit_rate: float | None = Field(ge=0, le=1)
    hit_latency: dict[str, float | None]
    miss_latency: dict[str, float | None]


class NoCacheDiagnostic(_Frozen):
    """The configuration's extra pass without the prefix cache: a diagnostic, never a headline."""

    role: Literal["diagnostic"] = "diagnostic"
    status: Status
    passes_completed: int = Field(ge=0)
    error: str | None = Field(default=None, max_length=_MAX_ERROR_CHARS)
    latency: LatencySummary


class ConfigurationResult(_Frozen):
    """One configuration: what was planned, what was measured, and per boundary the numbers."""

    index: int = Field(ge=0)
    batch_size: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    status: Status
    latency_unit: Literal["ms_per_request", "ms_per_batch"]
    passes_planned: int = Field(ge=0)
    passes_completed: int = Field(ge=0)
    iterations_planned: int = Field(ge=0)
    iterations_measured: int = Field(ge=0)
    # The iterations of completed passes, the only ones the statistics read.
    iterations_summarized: int = Field(ge=0)
    iterations_failed: int = Field(ge=0)
    warmup_per_pass: int = Field(ge=0)
    remainder_per_pass: int = Field(ge=0)
    error: str | None = Field(default=None, max_length=_MAX_ERROR_CHARS)
    headline: Headline
    boundaries: tuple[BoundaryResult, ...]
    prefix_cache: PrefixCacheResult | None = None
    no_cache_diagnostic: NoCacheDiagnostic | None = None


class GpuState(_Frozen):
    """One GPU's state at the start or end of a timed pass, as ``nvidia-smi`` reported it."""

    index: str | None = Field(default=None, max_length=8)
    clocks_sm_mhz: float | None = None
    clocks_mem_mhz: float | None = None
    power_limit_w: float | None = None
    power_draw_w: float | None = None
    temperature_c: float | None = None
    persistence_mode: str | None = Field(default=None, max_length=16)


class PassTelemetry(_Frozen):
    """The GPUs' clocks, power limit, temperature and persistence mode around one timed pass."""

    configuration: int = Field(ge=0)
    pass_index: int = Field(ge=0)
    kind: Literal["timed", "no_cache_diagnostic"] = "timed"
    at: Literal["start", "end"]
    gpus: tuple[GpuState, ...] = Field(max_length=16)


class PrefixReuseRecord(_Frozen):
    """The premise-prefix cache of a prefix-reuse run: its limit and what it held."""

    scope: Literal["never_cleared"] = "never_cleared"
    max_bytes: int = Field(ge=1)
    peak_bytes: int = Field(ge=0)
    bytes_at_end: int = Field(ge=0)
    entries_at_end: int = Field(ge=0)
    evictions: int = Field(ge=0)
    uncacheable: int = Field(ge=0)


type IncompleteCode = Literal[
    "gpu_busy", "wall_clock_cap", "stopped_on_failure", "server_died", "not_measured"
]


class BenchmarkSummary(_Frozen):
    """The run's results as the control plane stores them: bounded, no row text, no secret."""

    format: Literal["strata.benchmark.summary/v1"] = SUMMARY_FORMAT
    engine: Literal["hf_classify", "vllm_chat"]
    boundaries: tuple[Boundary, ...]
    complete: bool
    incomplete_reason: str | None = Field(default=None, max_length=_MAX_ERROR_CHARS)
    incomplete_codes: tuple[IncompleteCode, ...] = ()
    # Whether the headline may be read as the protocol's number: only for a complete run, every
    # configuration measured in full on a GPU nobody else used. An incomplete run has no T13 read.
    headline_readable: bool
    headline_boundary: Literal["end_to_end"] = "end_to_end"
    model_path: Literal["strata_verifier", "reference_automodel", "vllm_chat"]
    # Why the headline may not be read, when it may not: incomplete, or the GPU's idleness could
    # not be checked (no ``nvidia-smi``, as on a CPU).
    headline_unreadable_reason: str | None = Field(default=None, max_length=_MAX_ERROR_CHARS)
    # ``checked``: other compute processes were looked for at the start and before every pass.
    gpu_idle_check: Literal["checked", "unavailable"]
    gpu_telemetry: tuple[PassTelemetry, ...] = Field(default=(), max_length=2 * 64 * 21)
    prefix_reuse: PrefixReuseRecord | None = None
    sample: dict[str, int]
    excluded: dict[str, int]
    cold_start: dict[str, float | None]
    gpu: dict[str, str | int | None]
    configurations: tuple[ConfigurationResult, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def _complete_means_every_configuration_measured(self) -> BenchmarkSummary:
        clean = all(measured_cleanly(c) for c in self.configurations)
        if self.complete != clean:
            msg = "complete must say whether every configuration was measured cleanly"
            raise ValueError(msg)
        if self.complete != (self.incomplete_reason is None) or self.complete == bool(
            self.incomplete_codes
        ):
            msg = "an incomplete summary, and only one, carries its reason and codes"
            raise ValueError(msg)
        if self.headline_readable and (not self.complete or self.gpu_idle_check != "checked"):
            msg = "the headline is readable only for a complete run on a GPU checked idle"
            raise ValueError(msg)
        return self


def measured_cleanly(result: ConfigurationResult) -> bool:
    """Every pass measured with no failure, the no-cache diagnostic pass included when made."""
    diagnostic = result.no_cache_diagnostic
    return (
        result.status == "measured"
        and not result.iterations_failed
        and (diagnostic is None or diagnostic.status == "measured")
    )


def union_seconds(intervals: Sequence[tuple[int, int]]) -> float:
    """The measure, in seconds, of the union of ``[start, end]`` nanosecond intervals."""
    total = 0
    reach: int | None = None
    for start, end in sorted(intervals):
        if reach is None or start > reach:
            total += end - start
            reach = end
        elif end > reach:
            total += end - reach
            reach = end
    return total / _NS_PER_S


def _ms(iteration: Iteration, boundary: Boundary) -> float | None:
    if boundary == "end_to_end":
        return (iteration.end_ns - iteration.start_ns) / _NS_PER_MS
    return iteration.forward_ms


def _by_pass(iterations: Sequence[Iteration], boundary: Boundary, passes: int) -> list[list[float]]:
    grouped: list[list[float]] = [[] for _ in range(passes)]
    for item in iterations:
        value = _ms(item, boundary)
        if value is not None and item.pass_index < passes:
            grouped[item.pass_index].append(value)
    return grouped


def _throughput(
    iterations: Sequence[Iteration], boundary: Boundary, tokens: Sequence[int]
) -> Throughput:
    if boundary == "end_to_end":
        window = union_seconds([(i.start_ns, i.end_ns) for i in iterations])
    else:
        window = sum(i.forward_ms or 0.0 for i in iterations) / 1000
    items = sum(len(i.rows) for i in iterations)
    input_tokens = sum(tokens[row] for i in iterations for row in i.rows)
    generated = [i.output_tokens for i in iterations]
    output_tokens = sum(t for t in generated if t is not None)
    has_output = any(t is not None for t in generated)
    rate = window > 0
    return Throughput(
        items_per_s=items / window if rate else None,
        input_tokens_per_s=input_tokens / window if rate else None,
        output_tokens_per_s=output_tokens / window if rate and has_output else None,
        window_s=window,
        items=items,
    )


def _buckets(
    iterations: Sequence[Iteration],
    boundary: Boundary,
    key: Callable[[int], str | None],
    passes: int,
) -> tuple[BucketRow, ...] | None:
    grouped: dict[str, list[list[float]]] = {}
    for item in iterations:
        value = _ms(item, boundary)
        label = key(item.rows[0])
        if value is None or label is None:
            continue
        grouped.setdefault(label, [[] for _ in range(passes)])[item.pass_index].append(value)
    if not grouped:
        return None
    rows: list[BucketRow] = []
    for label, by_pass in sorted(grouped.items(), key=lambda kv: _bucket_sort_key(kv[0])):
        pooled = [v for values in by_pass for v in values]
        p95s = [percentile(values, 95) for values in by_pass if values]
        rows.append(
            BucketRow(
                bucket=label,
                n=len(pooled),
                p50=percentile(pooled, 50) if pooled else None,
                p95=percentile(pooled, 95) if pooled else None,
                p99=percentile(pooled, 99) if pooled else None,
                p95_median_of_passes=float(median(p95s)) if p95s else None,
            )
        )
    return tuple(rows)


def _bucket_sort_key(label: str) -> tuple[int, int]:
    if label.startswith(">"):
        return (1, int(label[1:]))
    head = label.split("-", 1)[0]
    return (0, int(head)) if head.isdigit() else (2, 0)


def summarize(
    outcome: ConfigurationOutcome,
    *,
    boundaries: Sequence[Boundary],
    tokens: Sequence[int],
    length_of: Callable[[int], str | None],
    items_of: Callable[[int], str | None],
    bootstrap: tuple[int, int] | None,
    diagnostic: ConfigurationOutcome | None = None,
) -> ConfigurationResult:
    """One configuration's result: per declared boundary, percentiles, throughput and buckets.

    ``tokens`` is every split row's input token count, by row index. ``bootstrap`` is
    ``(seed, resamples)`` when the run asked for intervals, else ``None``. ``diagnostic`` is a
    prefix-reuse configuration's extra pass without the cache, summarized on its own.
    """
    configuration = outcome.configuration
    # Only completed passes are summarized: a pass cut short by a failure, the wall-clock cap or
    # a busy GPU holds a biased slice of its order, so it never enters a percentile, the
    # across-pass median, the throughput or a bucket. Its iterations stay in timings.parquet.
    passes = outcome.passes_completed
    iterations = [i for i in outcome.iterations if i.pass_index < passes]
    results: list[BoundaryResult] = []
    single = configuration.batch_size == 1
    for boundary in boundaries:
        by_pass = _by_pass(iterations, boundary, passes)
        pooled = [value for values in by_pass for value in values]
        results.append(
            BoundaryResult(
                boundary=boundary,
                role="headline" if boundary == "end_to_end" else "diagnostic",
                latency=latency_summary(by_pass),
                bootstrap=(
                    latency_bootstrap(pooled, seed=bootstrap[0], resamples=bootstrap[1])
                    if bootstrap is not None
                    else None
                ),
                throughput=_throughput(iterations, boundary, tokens),
                by_length=_buckets(iterations, boundary, length_of, passes) if single else None,
                by_items=_buckets(iterations, boundary, items_of, passes) if single else None,
            )
        )
    unit: Literal["ms_per_request", "ms_per_batch"] = "ms_per_request" if single else "ms_per_batch"
    p95s = [
        percentile(values, 95) for values in _by_pass(iterations, "end_to_end", passes) if values
    ]
    headline = Headline(
        p95_median_ms=float(median(p95s)) if p95s else None,
        p95_min_ms=min(p95s) if p95s else None,
        p95_max_ms=max(p95s) if p95s else None,
        passes=len(p95s),
        unit=unit,
    )
    reused = [i for i in iterations if i.prefix_cache_hit is not None]
    prefix_cache: PrefixCacheResult | None = None
    if reused or diagnostic is not None:
        hit = [_ms(i, "end_to_end") or 0.0 for i in reused if i.prefix_cache_hit]
        miss = [_ms(i, "end_to_end") or 0.0 for i in reused if not i.prefix_cache_hit]
        prefix_cache = PrefixCacheResult(
            hits=len(hit),
            misses=len(miss),
            hit_rate=len(hit) / len(reused) if reused else None,
            hit_latency=latency_summary([hit]).pooled,
            miss_latency=latency_summary([miss]).pooled,
        )
    no_cache: NoCacheDiagnostic | None = None
    if diagnostic is not None:
        done = [i for i in diagnostic.iterations if i.pass_index < diagnostic.passes_completed]
        no_cache = NoCacheDiagnostic(
            status=diagnostic.status,
            passes_completed=diagnostic.passes_completed,
            error=diagnostic.error,
            latency=latency_summary([[_ms(i, "end_to_end") or 0.0 for i in done]]),
        )
    return ConfigurationResult(
        index=configuration.index,
        batch_size=configuration.batch_size,
        concurrency=configuration.concurrency,
        status=outcome.status,
        latency_unit=unit,
        passes_planned=outcome.passes_planned,
        passes_completed=outcome.passes_completed,
        iterations_planned=outcome.iterations_planned,
        iterations_measured=len(outcome.iterations),
        iterations_summarized=len(iterations),
        iterations_failed=outcome.failed,
        warmup_per_pass=outcome.warmup_per_pass,
        remainder_per_pass=outcome.remainder_per_pass,
        error=outcome.error,
        headline=headline,
        boundaries=tuple(results),
        prefix_cache=prefix_cache,
        no_cache_diagnostic=no_cache,
    )


def iteration_rows(
    outcomes: Sequence[ConfigurationOutcome],
    *,
    boundaries: Sequence[Boundary],
    tokens: Sequence[int],
    items: Sequence[int | None] | None,
    length_of: Callable[[int], str | None],
    items_of: Callable[[int], str | None],
) -> list[dict[str, Any]]:
    """One row per measured iteration for the timings parquet: indices and numbers, no text."""
    out: list[dict[str, Any]] = []
    for outcome in outcomes:
        configuration = outcome.configuration
        for item in outcome.iterations:
            first = item.rows[0]
            out.append(
                {
                    "configuration": configuration.index,
                    "batch_size": configuration.batch_size,
                    "concurrency": configuration.concurrency,
                    "pass": item.pass_index,
                    # False for a pass cut short: kept for the record, never summarized.
                    "pass_completed": item.pass_index < outcome.passes_completed,
                    "iteration": item.iteration,
                    "row_indices": list(item.rows),
                    "input_tokens": sum(tokens[row] for row in item.rows),
                    "max_input_tokens": max(tokens[row] for row in item.rows),
                    "items": (
                        sum(int(items[row] or 0) for row in item.rows)
                        if items is not None
                        else None
                    ),
                    "length_bucket": length_of(first) if len(item.rows) == 1 else None,
                    "items_bucket": items_of(first) if len(item.rows) == 1 else None,
                    # The headline column, and the forward pass inside the same request.
                    "total_ms": _ms(item, "end_to_end"),
                    "forward_ms": item.forward_ms,
                    # Premise-prefix reuse only (else null); the no-cache pass is its diagnostic.
                    "prefix_cache_hit": item.prefix_cache_hit,
                    "pass_kind": item.kind,
                    "output_tokens": item.output_tokens,
                }
            )
    return out
