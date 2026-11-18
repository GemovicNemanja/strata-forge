"""The hardware and software a measurement actually ran on, as the machine reports it.

A latency number is a claim about one GPU, driver, CUDA runtime and library stack, so a benchmark
records all of them from the machine it ran on rather than from what the run asked for. Every
field is what was observed or ``None``: a field the machine could not report stays unknown and is
never filled with an expectation.

Two sources, each optional. ``nvidia-smi`` answers for the devices, the driver, clocks, power
limit and whether other processes hold the GPU; it is queried with a bounded wait, the way
:mod:`strata_forge.training.hardware` does, because a wedged driver can leave it unkillable. torch,
when importable, answers for its own version, the CUDA runtime and cuDNN it was built with, and
the device it would compute on. Distribution versions come from package metadata.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from importlib import import_module
from pathlib import Path
from typing import Any

from strata_forge.pipelines._common import installed_versions

__all__ = [
    "GPU_FIELDS",
    "RECORDED_DISTRIBUTIONS",
    "STATE_FIELDS",
    "compute_pids",
    "gpu_matches",
    "gpu_state",
    "own_processes",
    "parse_gpus",
    "receipt",
    "short_receipt",
    "visible_indices",
    "visible_tokens",
]

#: The ``nvidia-smi --query-gpu`` fields recorded per device, in CSV order.
GPU_FIELDS = (
    "index",
    "name",
    "memory.total",
    "driver_version",
    "persistence_mode",
    "clocks.sm",
    "clocks.max.sm",
    "power.limit",
    "pci.bus_id",
)
#: The installed distributions a benchmark's numbers depend on.
RECORDED_DISTRIBUTIONS = (
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "vllm",
    "numpy",
    "datasets",
    "huggingface-hub",
)
_TIMEOUT_S = 5.0
_REAP_TIMEOUT_S = 1.0
_MAX_FIELD_CHARS = 128
_MISSING = frozenset({"", "[N/A]", "N/A", "[Not Supported]"})


def _nvidia_smi(args: list[str]) -> str | None:
    """``nvidia-smi`` output for ``args``, or ``None`` when there is none to be had in time."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, resolved binary, no shell
            [binary, *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
    except OSError, ValueError:
        return None
    try:
        stdout, _ = proc.communicate(timeout=_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            proc.communicate(timeout=_REAP_TIMEOUT_S)
        return None
    except OSError, ValueError:
        return None
    return stdout if proc.returncode == 0 else None


def _field(raw: str) -> str | None:
    value = raw.strip()
    return None if value in _MISSING else value[:_MAX_FIELD_CHARS]


def _number(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def visible_tokens(value: str | None) -> list[str] | None:
    """``CUDA_VISIBLE_DEVICES`` as its device tokens, or ``None`` when unset (every device).

    As CUDA reads it: comma-separated indices or ``GPU-`` UUID prefixes, ending at the first token
    that is neither (an empty value hides every device).
    """
    if value is None:
        return None
    tokens: list[str] = []
    for raw in value.split(","):
        token = raw.strip()
        if not (token.isdigit() or token.startswith("GPU-")):
            break
        tokens.append(token)
    return tokens


def _visible(index: str | None, uuid: str | None, tokens: list[str] | None) -> bool:
    if tokens is None:
        return True
    for token in tokens:
        if token.isdigit() and index is not None and index.isdigit() and int(token) == int(index):
            return True
        if token.startswith("GPU-") and uuid is not None and uuid.startswith(token):
            return True
    return False


def _device_map() -> dict[str, str | None] | None:
    """Each physical GPU's index mapped to its UUID, or ``None`` without ``nvidia-smi``."""
    raw = _nvidia_smi(["--query-gpu=index,uuid", "--format=csv,noheader"])
    if raw is None:
        return None
    out: dict[str, str | None] = {}
    for line in raw.splitlines():
        cells = [_field(c) for c in line.split(",")]
        if len(cells) == 2 and cells[0] is not None:
            out[cells[0]] = cells[1]
    return out


def visible_indices() -> set[str] | None:
    """The physical indices of the GPUs this process may use (``CUDA_VISIBLE_DEVICES``)."""
    devices = _device_map()
    if devices is None:
        return None
    tokens = visible_tokens(os.environ.get("CUDA_VISIBLE_DEVICES"))
    return {index for index, uuid in devices.items() if _visible(index, uuid, tokens)}


def parse_gpus(raw: str) -> list[dict[str, Any]]:
    """One record per CSV row of :data:`GPU_FIELDS`; a row of the wrong width is skipped."""
    gpus: list[dict[str, Any]] = []
    for line in raw.splitlines():
        cells = line.split(",")
        if len(cells) != len(GPU_FIELDS):
            continue
        values = dict(zip(GPU_FIELDS, (_field(cell) for cell in cells), strict=True))
        memory = _number(values["memory.total"])
        index = values["index"]
        gpus.append(
            {
                "index": int(index) if index is not None and index.isdigit() else None,
                "name": values["name"],
                "memory_total_mib": int(memory) if memory is not None else None,
                "driver_version": values["driver_version"],
                "persistence_mode": values["persistence_mode"],
                "clocks_sm_mhz": _number(values["clocks.sm"]),
                "clocks_max_sm_mhz": _number(values["clocks.max.sm"]),
                "power_limit_w": _number(values["power.limit"]),
                "pci_bus_id": values["pci.bus_id"],
            }
        )
    return gpus


#: The per-GPU state recorded at the start and end of every timed pass. Clocks are recorded, not
#: locked: the measurement is of the machine as it runs.
STATE_FIELDS = (
    "index",
    "clocks.sm",
    "clocks.mem",
    "power.limit",
    "power.draw",
    "temperature.gpu",
    "persistence_mode",
)


def compute_pids() -> set[int] | None:
    """The processes holding a compute context on a GPU this process may use (``None``: unknown).

    A process on a GPU that ``CUDA_VISIBLE_DEVICES`` hides is another tenant's, not a competitor.
    """
    devices = _device_map()
    raw = _nvidia_smi(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
    if raw is None or devices is None:
        return None
    tokens = visible_tokens(os.environ.get("CUDA_VISIBLE_DEVICES"))
    visible = {uuid for index, uuid in devices.items() if _visible(index, uuid, tokens)}
    pids: set[int] = set()
    for line in raw.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) == 2 and cells[1].isdigit() and cells[0] in visible:
            pids.add(int(cells[1]))
    return pids


def own_processes(root: int) -> set[int]:
    """``root`` and every process descended from it (a model server the runner started).

    Read from ``/proc``; where there is none, ``root`` alone.
    """
    parents: dict[int, int] = {}
    proc = Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
            except OSError:
                continue
            # `pid (comm) state ppid ...`; comm may hold spaces and parentheses.
            fields = stat[stat.rfind(")") + 2 :].split()
            if len(fields) > 1 and fields[1].isdigit():
                parents[int(entry.name)] = int(fields[1])
    tree = {root}
    grew = True
    while grew:
        grew = False
        for pid, parent in parents.items():
            if parent in tree and pid not in tree:
                tree.add(pid)
                grew = True
    return tree


def _compute_processes() -> int | None:
    pids = compute_pids()
    return None if pids is None else len(pids)


def gpu_state() -> list[dict[str, Any]] | None:
    """Every visible GPU's clocks, power, temperature and persistence mode right now."""
    raw = _nvidia_smi([f"--query-gpu={','.join(STATE_FIELDS)}", "--format=csv,noheader,nounits"])
    if raw is None:
        return None
    visible = visible_indices()
    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        cells = [_field(c) for c in line.split(",")]
        if len(cells) != len(STATE_FIELDS):
            continue
        record = dict(zip(STATE_FIELDS, cells, strict=True))
        if visible is not None and record["index"] not in visible:
            continue
        out.append(
            {
                "index": record["index"],
                "clocks_sm_mhz": _number(record["clocks.sm"]),
                "clocks_mem_mhz": _number(record["clocks.mem"]),
                "power_limit_w": _number(record["power.limit"]),
                "power_draw_w": _number(record["power.draw"]),
                "temperature_c": _number(record["temperature.gpu"]),
                "persistence_mode": record["persistence_mode"],
            }
        )
    return out


def _torch_facts() -> dict[str, Any]:
    try:
        torch: Any = import_module("torch")
    except ImportError:
        return {"available": False}
    facts: dict[str, Any] = {
        "available": True,
        "version": str(torch.__version__),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cudnn": None,
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "device_memory_bytes": None,
        "device_capability": None,
    }
    with contextlib.suppress(Exception):  # a broken CUDA install reports nothing, not a failure
        cudnn = torch.backends.cudnn.version()
        facts["cudnn"] = int(cudnn) if cudnn is not None else None
    with contextlib.suppress(Exception):
        if bool(torch.cuda.is_available()):
            facts["cuda_available"] = True
            facts["device_count"] = int(torch.cuda.device_count())
            facts["device_name"] = str(torch.cuda.get_device_name(0))
            properties = torch.cuda.get_device_properties(0)
            facts["device_memory_bytes"] = int(properties.total_memory)
            facts["device_capability"] = f"{properties.major}.{properties.minor}"
    return facts


def receipt(*, use_torch: bool) -> dict[str, Any]:
    """Everything this machine reports about its GPUs and stack.

    ``use_torch`` asks torch about its CUDA device too. An engine that serves the model in
    another process passes ``False``, so this process does not open a CUDA context of its own on
    the GPU being measured.
    """
    raw = _nvidia_smi([f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"])
    gpus = parse_gpus(raw) if raw is not None else None
    visible = visible_indices() if gpus is not None else None
    if gpus is not None and visible is not None:
        # Only the devices this process may use: the count is the GPU-seconds multiplier.
        gpus = [gpu for gpu in gpus if str(gpu.get("index")) in visible]
    return {
        "nvidia_smi": raw is not None,
        "gpus": gpus,
        "compute_processes_before": _compute_processes(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": _torch_facts() if use_torch else None,
        "packages": installed_versions(RECORDED_DISTRIBUTIONS),
    }


def short_receipt(full: dict[str, Any]) -> dict[str, str | int | None]:
    """The few fields a results summary shows: device name, count, memory, driver, CUDA."""
    gpus: list[dict[str, Any]] = full.get("gpus") or []
    torch: dict[str, Any] = full.get("torch") or {}
    first: dict[str, Any] = gpus[0] if gpus else {}
    name = first.get("name") or torch.get("device_name")
    memory = first.get("memory_total_mib")
    if memory is None and torch.get("device_memory_bytes") is not None:
        memory = int(torch["device_memory_bytes"]) // (1024 * 1024)
    count = len(gpus) if gpus else (torch.get("device_count") if torch.get("available") else None)
    return {
        "name": name,
        "count": count,
        "memory_total_mib": memory,
        "driver_version": first.get("driver_version"),
        "cuda_runtime": torch.get("cuda_runtime"),
    }


def gpu_matches(full: dict[str, Any], required: str) -> bool:
    """Whether every visible GPU's reported name contains ``required`` (case-insensitive).

    No reported GPU is no match: a requirement on the hardware is never met by a machine that
    could not say what it has.
    """
    gpus: list[dict[str, Any]] = full.get("gpus") or []
    names = [str(gpu.get("name") or "") for gpu in gpus]
    if not names:
        torch: dict[str, Any] = full.get("torch") or {}
        names = [str(torch.get("device_name") or "")] if torch.get("cuda_available") else []
    needle = required.lower()
    return bool(names) and all(needle in name.lower() for name in names)
