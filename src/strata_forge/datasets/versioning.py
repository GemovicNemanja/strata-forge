"""Content-hash versioning + dataset diff helpers.

A :class:`Dataset`'s version is the SHA-256 of its canonical content —
name + sorted item IDs + metadata. Two puts with the same content
produce the same version, so deduplication is automatic and the
:mod:`strata_forge.evals` CI gate can detect "the dataset changed between
runs" without separate bookkeeping.

The :func:`diff` helper computes the added / removed / unchanged
partition between two dataset versions by item ID, returning a
:class:`DatasetDelta` consumers can render in reports or use to
decide whether to invalidate cached eval results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from strata_forge.core.repro import content_hash

if TYPE_CHECKING:
    from strata_forge.datasets.schema import Dataset, DatasetItem

__all__ = [
    "DatasetDelta",
    "dataset_version",
    "diff",
]


def dataset_version(dataset: Dataset) -> str:
    """Return the canonical content-hash version for ``dataset``.

    The hash covers the dataset's ``name``, the sorted list of item
    IDs, and ``metadata`` — but **not** ``description`` (free-form
    prose shouldn't bump the version) or the items' ``metadata``
    fields (re-annotating an item without changing its content
    shouldn't change the dataset version).
    """
    return content_hash(
        {
            "name": dataset.name,
            "item_ids": sorted(item.id for item in dataset.items),
            "metadata": dataset.metadata,
        }
    )


@dataclass(frozen=True, slots=True)
class DatasetDelta:
    """The partition of two datasets into added / removed / unchanged items.

    Identity is by ``DatasetItem.id``. Two items with the same ID are
    considered the "same item" even if their ``metadata`` differs —
    so re-annotating items between versions doesn't show up as
    add+remove. The :attr:`unchanged` field carries the items from
    the new dataset (so any metadata updates are visible to
    consumers), not the old.
    """

    added: tuple[DatasetItem, ...]
    removed: tuple[DatasetItem, ...]
    unchanged: tuple[DatasetItem, ...]

    @property
    def is_empty(self) -> bool:
        """True when no items were added or removed (unchanged ignored)."""
        return not self.added and not self.removed


def diff(old: Dataset, new: Dataset) -> DatasetDelta:
    """Compute the :class:`DatasetDelta` from ``old`` to ``new``.

    Items are matched by ``id``. The returned tuples are sorted by ID
    so callers get a deterministic order in reports and diffs.
    """
    old_by_id = {item.id: item for item in old.items}
    new_by_id = {item.id: item for item in new.items}

    added_ids = set(new_by_id) - set(old_by_id)
    removed_ids = set(old_by_id) - set(new_by_id)
    unchanged_ids = set(old_by_id) & set(new_by_id)

    return DatasetDelta(
        added=tuple(new_by_id[i] for i in sorted(added_ids)),
        removed=tuple(old_by_id[i] for i in sorted(removed_ids)),
        unchanged=tuple(new_by_id[i] for i in sorted(unchanged_ids)),
    )
