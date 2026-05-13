"""Reproducibility helpers: seed control, stable content hashing, env snapshots.

These primitives back the project's reproducibility story: any long-running
operation (eval run, training job, experiment sweep) should call
``set_seed`` at startup and persist ``env_snapshot()`` alongside its outputs.
``content_hash`` provides a stable, JSON-canonical fingerprint suitable for
caching, dataset versioning, and deduplication.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

__all__ = ["content_hash", "env_snapshot", "set_seed"]

# Packages worth recording in run metadata. Lazy lookups: if a package isn't
# installed the entry maps to None rather than raising.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "pydantic",
    "pydantic-settings",
    "litellm",
    "structlog",
    "tenacity",
    "typer",
    "anyio",
    "pyyaml",
    "python-dotenv",
    "rich",
    # heavy / optional deps — recorded when present
    "torch",
    "numpy",
    "transformers",
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "openai",
    "anthropic",
    "google-cloud-aiplatform",
    "boto3",
    "qdrant-client",
    "vllm",
    "skypilot",
)


def set_seed(seed: int) -> None:
    """Seed Python's ``random`` and, if installed, ``numpy`` and ``torch``.

    Heavy ML deps are imported lazily so calling this on a base install
    (without the ``finetuning`` extra) doesn't trigger an ImportError.
    """
    random.seed(seed)
    try:
        import numpy as np  # pyright: ignore[reportMissingImports]
    except ImportError:
        pass
    else:
        np.random.seed(seed)  # pyright: ignore[reportUnknownMemberType]
    try:
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
        if torch.cuda.is_available():  # pyright: ignore[reportUnknownMemberType]
            torch.cuda.manual_seed_all(seed)  # pyright: ignore[reportUnknownMemberType]


def content_hash(obj: object) -> str:
    """Return a stable SHA-256 hex digest for a JSON-serializable object.

    Keys are sorted and whitespace is collapsed before hashing, so two
    logically-equivalent inputs (e.g. dicts whose keys were inserted in
    different orders) produce the same digest. Non-JSON values are coerced
    via ``str``, so callers can pass dataclass-y structures without
    preprocessing — but the resulting hash is only stable if those objects
    have a stable ``__str__``.
    """
    canonical = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def env_snapshot() -> dict[str, str | None]:
    """Capture a snapshot of the runtime environment for experiment provenance.

    Returns a flat dict suitable for serializing into run metadata. Includes
    Python version, platform info, CWD, and the installed version of every
    package in ``_TRACKED_PACKAGES`` (``None`` if not installed).
    """
    snapshot: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cwd": str(Path.cwd()),
    }
    for pkg in _TRACKED_PACKAGES:
        try:
            snapshot[f"pkg.{pkg}"] = version(pkg)
        except PackageNotFoundError:
            snapshot[f"pkg.{pkg}"] = None
    return snapshot
