"""Lazy Langfuse client construction with a cached singleton accessor.

:func:`get_client` returns the same Langfuse client across calls,
built lazily on first use from
:class:`strata_forge.config.LangfuseConfig`. The function returns ``None`` in
three cases — Langfuse credentials aren't configured, the
``[langfuse]`` extra isn't installed, or the constructor itself
failed — so every public function in :mod:`strata_forge.tracing` can
short-circuit cleanly without crashing a production call path. The
"why is this None?" diagnosis surfaces through ``strata-forge doctor``, not
through exceptions on the hot path.

:func:`reset_client` clears the cache; tests use it between cases via
the autouse fixture in ``tests/unit/tracing/conftest.py``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from strata_forge.config import get_settings

__all__ = [
    "get_client",
    "reset_client",
]


@lru_cache(maxsize=1)
def _build_client() -> Any:
    """Construct a Langfuse client from the active :class:`LangfuseConfig`.

    Returns ``None`` when Langfuse isn't configured (either credential
    missing) or when the ``[langfuse]`` extra isn't installed. The
    cached value sticks for the lifetime of the process —
    :func:`reset_client` clears it.
    """
    config = get_settings().langfuse
    if not config.enabled:
        return None
    try:
        from langfuse import (  # pyright: ignore[reportMissingImports]
            Langfuse,  # pyright: ignore[reportUnknownVariableType]
        )
    except ImportError:
        return None
    if config.public_key is None or config.secret_key is None:  # pragma: no cover
        # Defensive: enabled implies both keys are set, but narrow for
        # the type checker. Unreachable in practice.
        return None
    return Langfuse(  # pyright: ignore[reportUnknownVariableType]
        host=config.host,
        public_key=config.public_key.get_secret_value(),
        secret_key=config.secret_key.get_secret_value(),
        # Tags every trace with the deployment environment (prod vs dev). `None`
        # lets the SDK fall back to its native `LANGFUSE_TRACING_ENVIRONMENT` read
        # (→ "default"); a configured value wins.
        environment=config.tracing_environment,
    )


def get_client() -> Any:
    """Return the cached Langfuse client, or ``None`` when Langfuse isn't usable.

    Lazy-imports the ``langfuse`` package on first call so
    :mod:`strata_forge.tracing` is importable without the ``[langfuse]``
    extra installed. Subsequent calls return the cached value (which
    may be ``None``).
    """
    return _build_client()


def reset_client() -> None:
    """Clear the cached client.

    Tests call this between cases (via the autouse fixture in
    ``tests/unit/tracing/conftest.py``) when they mutate
    ``LANGFUSE_*`` env vars and want a fresh client on the next
    :func:`get_client` call.
    """
    _build_client.cache_clear()
