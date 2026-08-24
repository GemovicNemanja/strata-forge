"""Backend implementations for :mod:`strata_forge.compute`.

The :class:`Backend` Protocol is the contract every implementation
satisfies. :class:`LocalBackend` ships in core; :class:`SSHBackend`
and :class:`SkyPilotBackend` lazy-import their SDKs behind the
``[compute]`` extra.
"""

from strata_forge.compute.backends.base import (
    MAX_CONSOLE_CHUNK_BYTES,
    Backend,
    safe_workdir_relpath,
)
from strata_forge.compute.backends.local import LocalBackend
from strata_forge.compute.backends.skypilot import SkyPilotBackend
from strata_forge.compute.backends.ssh import SSHBackend

__all__ = [
    "MAX_CONSOLE_CHUNK_BYTES",
    "Backend",
    "LocalBackend",
    "SSHBackend",
    "SkyPilotBackend",
    "safe_workdir_relpath",
]
