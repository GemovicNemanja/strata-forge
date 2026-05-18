"""Backend implementations for :mod:`forge.compute`.

The :class:`Backend` Protocol is the contract every implementation
satisfies. Phase 5.1 ships :class:`LocalBackend`; SSH and SkyPilot
backends land in Phase 5.2.
"""

from forge.compute.backends.base import Backend
from forge.compute.backends.local import LocalBackend

__all__ = [
    "Backend",
    "LocalBackend",
]
