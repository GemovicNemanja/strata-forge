"""GPU counters for the progress channel.

A run's step count says how far it has got; it says nothing about whether the box is actually
working. A stalled dataloader, a model that spilled to CPU, a card thermal-throttling at 88C and a
healthy run all look identical from the step count alone. :meth:`GpuSampler.sample` reports the
three numbers that tell them apart — utilization, memory, temperature — shaped as
:attr:`~strata_forge.training.progress.ProgressEvent.metrics` keys so they ride the channel that
already exists rather than needing a second one.

**Why ``nvidia-smi`` and not NVML.** ``pynvml``/``nvidia-ml-py`` would be tidier in-process, but
forge's dependencies are installed *fresh on the user's own VM* at the start of every run, so each
new pin is another package that can publish a breaking release between a green CI run and someone's
GPU box. The ``nvidia-smi`` binary is already there on any machine with a driver, needs no pin, and
cannot be yanked from an index.

**Why sampling happens on a background thread.** The obvious implementation — shell out inside
:meth:`GpuSampler.sample` — is wrong twice over, and both failures are the same fact seen from two
sides: ``nvidia-smi`` can block for an unbounded time.

- Its callers are coroutines. A runner emits progress from an event loop that is concurrently
  driving generations and a liveness heartbeat, and a blocking ``fork``/``exec`` on that loop stalls
  all of it.
- ``subprocess.run(timeout=...)`` does **not** bound the call. On POSIX its timeout path kills the
  child and then calls ``process.wait()`` with no timeout of its own. The single most common way
  ``nvidia-smi`` misbehaves — a wedged driver, an Xid, a GPU off the bus — leaves it in
  uninterruptible ``D`` state, where ``SIGKILL`` does not reap it and that ``wait()`` never returns.

So :meth:`GpuSampler.sample` never blocks: it returns the last reading and, if that reading is
stale, hands a refresh to a daemon thread. A wedged probe then costs one abandoned thread, not the
run — and never blocks interpreter exit or a cancellation.

**Telemetry must never fail a run.** Every failure path — no binary, no driver, a timeout, a format
that changed, a CPU-only box — yields no counters. A missing gauge is a cosmetic gap; an exception
raised out of a metrics call during someone's four-hour fine-tune is not.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import threading
import time

__all__ = [
    "GPU_QUERY_FIELDS",
    "GpuSampler",
    "sample_gpu",
]

GPU_QUERY_FIELDS = ("utilization.gpu", "memory.used", "memory.total", "temperature.gpu")
"""The `nvidia-smi --query-gpu` fields read, in the order the CSV returns them."""

_TIMEOUT_S = 2.0
"""How long to wait for the probe before giving up on it. Enforced by
:func:`_run_nvidia_smi`'s own bounded wait rather than by ``subprocess.run``, which cannot."""

_REAP_TIMEOUT_S = 1.0
"""How long to wait for a killed child to actually die before abandoning it. A ``D``-state process
never will, and blocking on it is the bug this whole module is shaped around."""

_DEFAULT_MIN_INTERVAL_S = 5.0
"""Floor between refreshes. Callers emit progress far more often than these numbers move."""


def _run_nvidia_smi(timeout_s: float) -> str | None:
    """One line of CSV per visible GPU, or ``None`` if the box cannot answer in time.

    Uses ``Popen`` rather than ``subprocess.run`` specifically so the wait after the kill is
    bounded: an unkillable child is abandoned (leaking one zombie until the process exits) instead
    of hanging the caller forever. Leaking a zombie is strictly better than stranding a GPU.
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    argv = [
        binary,
        f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, resolved binary, no shell
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError, ValueError:
        return None

    try:
        stdout, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        # Bounded, and suppressed: a D-state child never dies, and blocking on it is the bug
        # this whole module is shaped around. Abandon it instead.
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            proc.communicate(timeout=_REAP_TIMEOUT_S)
        return None
    except OSError, ValueError:
        return None

    if proc.returncode != 0:
        return None
    return stdout


def _parse(raw: str) -> dict[str, float]:
    """Fold per-device CSV rows into one set of whole-box counters.

    Aggregated rather than per-device on purpose: a run occupies the box, and the question being
    asked is "is this machine working hard and is it in trouble", not "what is card 3 doing". Mean
    utilization, summed memory, and — deliberately — **max** temperature, because one card cooking
    is the fact worth surfacing and an average would hide it behind its healthy neighbours.

    Missing fields are handled per FIELD, not per row. A card that reports ``[N/A]`` for temperature
    alone (routine on MIG instances and vGPU passthrough) still contributes its utilization and its
    memory, so the box's totals stay true rather than silently under-reporting a whole GPU.
    """
    columns: list[list[float]] = [[] for _ in GPU_QUERY_FIELDS]
    seen = 0
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(GPU_QUERY_FIELDS):
            continue  # a driver whose CSV shape moved: better silent than mislabelled
        seen += 1
        for index, part in enumerate(parts):
            try:
                columns[index].append(float(part))
            except ValueError:
                continue  # "[N/A]" from a card that does not report this one field

    if not seen:
        return {}

    utils, used, total, temps = columns
    out: dict[str, float] = {"gpu_count": float(seen)}
    if utils:
        out["gpu_util_pct"] = sum(utils) / len(utils)
    if used:
        out["gpu_mem_used_mb"] = sum(used)
    if total:
        out["gpu_mem_total_mb"] = sum(total)
    if temps:
        out["gpu_temp_c"] = max(temps)
    return out


def sample_gpu(*, timeout_s: float = _TIMEOUT_S) -> dict[str, float]:
    """Read the GPU counters once, synchronously. ``{}`` when the box cannot report them.

    Blocks for up to ``timeout_s``. Callers on an event loop want :class:`GpuSampler` instead.
    """
    raw = _run_nvidia_smi(timeout_s)
    return _parse(raw) if raw else {}


class GpuSampler:
    """A non-blocking view of the GPU counters, safe to call on every progress event.

    :meth:`sample` returns immediately with the most recent reading and refreshes in the
    background when that reading has gone stale. It is safe to call from a coroutine, from a
    trainer callback, and at any frequency.

    Repeating the previous numbers is the right answer for a gauge: a reading a few seconds old is
    honest, whereas a gap would make the chart look like the GPU stopped. For the same reason a
    *failed* refresh leaves the last good reading in place rather than blanking it — one timed-out
    probe should not erase a gauge that was fine a moment ago.
    """

    def __init__(
        self,
        *,
        min_interval_s: float = _DEFAULT_MIN_INTERVAL_S,
        timeout_s: float = _TIMEOUT_S,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._last: dict[str, float] = {}
        self._last_at: float | None = None
        self._refreshing = False

    def sample(self) -> dict[str, float]:
        """The most recent counters. Never blocks; may be empty until the first refresh lands."""
        now = time.monotonic()
        with self._lock:
            stale = self._last_at is None or now - self._last_at >= self.min_interval_s
            start = stale and not self._refreshing
            if start:
                self._refreshing = True
                # Stamped BEFORE the probe runs, so a slow or hung refresh cannot make the sampler
                # consider itself stale again and pile up threads behind it.
                self._last_at = now
            snapshot = dict(self._last)

        if start:
            # Daemon: a probe wedged on a dead driver must never hold up interpreter exit.
            threading.Thread(target=self._refresh, name="gpu-sampler", daemon=True).start()
        return snapshot

    def refresh_now(self) -> dict[str, float]:
        """Refresh synchronously and return the result. For tests and non-async callers."""
        self._refresh()
        with self._lock:
            return dict(self._last)

    def _refresh(self) -> None:
        reading = sample_gpu(timeout_s=self.timeout_s)
        with self._lock:
            # An empty reading is a failed probe, not a box that lost its GPUs: keep the last
            # good numbers rather than punching a hole in the chart.
            if reading:
                self._last = reading
            self._last_at = time.monotonic()
            self._refreshing = False
