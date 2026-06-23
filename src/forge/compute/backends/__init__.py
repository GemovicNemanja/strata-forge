"""Backend implementations for :mod:`forge.compute`.

The :class:`Backend` Protocol is the contract every implementation
satisfies. :class:`LocalBackend` ships in core; :class:`SSHBackend`
and :class:`SkyPilotBackend` lazy-import their SDKs behind the
``[compute]`` extra.
"""

from forge.compute.backends.base import Backend, safe_workdir_relpath
from forge.compute.backends.local import LocalBackend
from forge.compute.backends.skypilot import SkyPilotBackend
from forge.compute.backends.ssh import SSHBackend

__all__ = [
    "Backend",
    "LocalBackend",
    "SSHBackend",
    "SkyPilotBackend",
    "safe_workdir_relpath",
]
