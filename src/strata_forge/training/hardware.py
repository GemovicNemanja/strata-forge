"""GPU counters for the progress channel.

A run's step count says how far it has got; it says nothing about whether the box is actually
working. A stalled dataloader, a model that spilled to CPU, a card thermal-throttling at 88C and a
healthy run all look identical from the step count alone. :func:`GpuSampler.sample` reports the
three numbers that tell them apart — utilization, memory, temperature — shaped as
:attr:`~strata_forge.training.progress.ProgressEvent.metrics` keys so they ride the channel that
already exists rather than needing a second one.

**Why ``nvidia-smi`` and not NVML.** ``pynvml``/``nvidia-ml-py`` would be tidier in-process, but
forge's dependencies are installed *fresh on the user's own VM* at the start of every run, so each
new pin is another package that can publish a breaking release between a green CI run and someone's
GPU box. The ``nvidia-smi`` binary is already there on any machine with a driver, needs no pin, and
cannot be yanked from an index. The cost is a subprocess every :attr:`min_interval_s`, which is why
the sampler caches.

**Telemetry must never fail a run.** Every failure path — no binary, no driver, a timeout, a format
that changed, a CPU-only box — returns ``{}``. A missing gauge is a cosmetic gap; an exception
raised out of a metrics call during someone's four-hour fine-tune is not.
"""

from __future__ import annotations

import shutil
import subprocess
import time

__all__ = [
    "GPU_QUERY_FIELDS",
    "GpuSampler",
    "sample_gpu",
]

GPU_QUERY_FIELDS = ("utilization.gpu", "memory.used", "memory.total", "temperature.gpu")
"""The `nvidia-smi --query-gpu` fields read, in the order the CSV returns them."""

_TIMEOUT_S = 2.0
"""Hard cap on the subprocess. A driver wedged in an uninterruptible wait is exactly the moment
telemetry must not join it — the caller is mid-run and holding the event loop."""

_DEFAULT_MIN_INTERVAL_S = 5.0
"""Floor between real samples. Callers emit progress far more often than the numbers move, and a
subprocess per chunk on a fast batch is real overhead for a gauge nobody reads that precisely."""


def _run_nvidia_smi(timeout_s: float) -> str | None:
    """One line of CSV per visible GPU, or ``None`` if the box cannot answer."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, resolved binary, no shell
            [
                binary,
                f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _parse(raw: str) -> dict[str, float]:
    """Fold per-device CSV rows into one set of whole-box counters.

    Aggregated rather than per-device on purpose: a run occupies the box, and the question being
    asked is "is this machine working hard and is it in trouble", not "what is card 3 doing". Mean
    utilization, summed memory, and — deliberately — **max** temperature, because one card cooking
    is the fact worth surfacing and an average would hide it behind its healthy neighbours.
    """
    utils: list[float] = []
    used: list[float] = []
    total: list[float] = []
    temps: list[float] = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(GPU_QUERY_FIELDS):
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue  # "[N/A]" from a card that does not report a field
        utils.append(values[0])
        used.append(values[1])
        total.append(values[2])
        temps.append(values[3])

    if not utils:
        return {}
    return {
        "gpu_count": float(len(utils)),
        "gpu_util_pct": sum(utils) / len(utils),
        "gpu_mem_used_mb": sum(used),
        "gpu_mem_total_mb": sum(total),
        "gpu_temp_c": max(temps),
    }


def sample_gpu(*, timeout_s: float = _TIMEOUT_S) -> dict[str, float]:
    """Read the GPU counters once, uncached. ``{}`` when the box cannot report them."""
    raw = _run_nvidia_smi(timeout_s)
    return _parse(raw) if raw else {}


class GpuSampler:
    """A rate-limited :func:`sample_gpu`, safe to call on every progress event.

    Holds the last reading and returns it unchanged until :attr:`min_interval_s` has passed, so a
    caller can fold ``sampler.sample()`` into every event it emits without thinking about how often
    that is. Repeating the previous numbers is the right answer for a gauge: a stale reading a few
    seconds old is honest, whereas a gap would make the chart look like the GPU stopped.

    Not thread-safe, and does not need to be: each runner emits progress from one place.
    """

    def __init__(
        self,
        *,
        min_interval_s: float = _DEFAULT_MIN_INTERVAL_S,
        timeout_s: float = _TIMEOUT_S,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._last: dict[str, float] = {}
        self._last_at: float | None = None

    def sample(self) -> dict[str, float]:
        """The current counters, re-reading only once per :attr:`min_interval_s`."""
        now = time.monotonic()
        if self._last_at is not None and now - self._last_at < self.min_interval_s:
            return dict(self._last)
        self._last = sample_gpu(timeout_s=self.timeout_s)
        self._last_at = now
        return dict(self._last)
