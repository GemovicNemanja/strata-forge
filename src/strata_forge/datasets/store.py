"""Abstract :class:`DatasetStore` and the not-found error.

Backends (in-memory, Langfuse) subclass :class:`DatasetStore`. The
store is store-agnostic by design — anything that satisfies the
async ``get`` / ``put`` / ``versions`` / ``list_names`` / ``delete``
shape is interchangeable. The :mod:`strata_forge.evals` runner depends on
the abstract surface, not on any specific backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from strata_forge.core.errors import ForgeError

if TYPE_CHECKING:
    from strata_forge.datasets.schema import Dataset

__all__ = [
    "DatasetNotFoundError",
    "DatasetStore",
]


class DatasetNotFoundError(ForgeError):
    """A requested dataset name or version isn't in the store.

    ``name`` and optional ``version`` are attached for programmatic
    inspection; ``version`` is ``None`` when the lookup was for the
    latest version of an unknown name.
    """

    def __init__(
        self,
        message: str,
        *,
        name: str,
        version: str | None = None,
    ) -> None:
        super().__init__(message)
        self.name = name
        self.version = version


class DatasetStore(ABC):
    """Abstract backend for dataset persistence.

    Implementations own the version scheme (the in-memory store uses
    content-hash versions; the Langfuse store uses whatever Langfuse
    returns). The registry layer treats versions as opaque strings.

    A ``get`` with ``version=None`` MUST return the latest stored
    version. A ``delete`` with ``version=None`` MUST remove every
    version of the name. A delete against an unknown name is a no-op
    (it's already absent); a delete against an unknown version of a
    known name raises :class:`DatasetNotFoundError`.
    """

    @abstractmethod
    async def get(self, name: str, version: str | None = None) -> Dataset:
        """Return a stored dataset; ``version=None`` means latest.

        Raises:
            DatasetNotFoundError: When ``name`` is unknown, or
                ``version`` is specified but doesn't exist for
                ``name``.
        """

    @abstractmethod
    async def put(self, dataset: Dataset) -> str:
        """Store ``dataset`` and return the assigned version string.

        Implementations may dedupe by content (the in-memory store
        does: putting the same content twice returns the same
        version). The returned string is opaque — callers shouldn't
        parse it.
        """

    @abstractmethod
    async def versions(self, name: str) -> list[str]:
        """Return every version of ``name``, newest-first.

        Raises:
            DatasetNotFoundError: When ``name`` is unknown.
        """

    @abstractmethod
    async def list_names(self) -> list[str]:
        """Return every distinct dataset name in the store, sorted."""

    @abstractmethod
    async def delete(self, name: str, version: str | None = None) -> None:
        """Delete a specific version, or every version when ``version`` is None.

        Unknown name → no-op. Known name + unknown version → raise.
        """
