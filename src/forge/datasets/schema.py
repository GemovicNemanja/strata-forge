"""Typed dataset shapes — :class:`DatasetItem` and :class:`Dataset`.

These are the canonical in-memory types every other piece of
:mod:`forge.datasets` produces or consumes. The Langfuse and HF
backends are converters into and out of them; the eval runner
(Phase 2.4) consumes them directly.

Both models are frozen Pydantic v2 with ``extra="forbid"`` so wire
shapes are stable. Items default to content-hash IDs derived from
their ``input`` + ``expected_output`` so two callers constructing the
"same item" end up with the same ID regardless of order of operations
— useful for dedup, cache-key composition, and the diff helpers in
:mod:`forge.datasets.versioning`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge.core.errors import ValidationError
from forge.core.repro import content_hash

__all__ = [
    "Dataset",
    "DatasetItem",
]


class DatasetItem(BaseModel):
    """One example in a dataset.

    Attributes:
        id: Stable identifier. Defaults to a SHA-256 content hash over
            ``input`` + ``expected_output`` (see :meth:`from_input`)
            so identical content produces identical IDs across
            constructions.
        input: The example's inputs as a JSON-shaped dict. Free-form;
            different datasets carry different field names (``query``,
            ``messages``, ``prompt``, …) — :class:`DatasetItem` doesn't
            constrain the keys.
        expected_output: Reference output, typically what a grader
            compares the model's response against. ``None`` for
            datasets that capture inputs only.
        metadata: Arbitrary JSON-serializable annotations (source,
            difficulty tier, language tag, …).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    input: dict[str, Any]
    expected_output: Any | None = None
    metadata: dict[str, Any] = Field(default={})

    @classmethod
    def from_input(
        cls,
        input: dict[str, Any],  # noqa: A002 — symmetric with the field name
        *,
        expected_output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetItem:
        """Build a :class:`DatasetItem` whose ``id`` is a content hash.

        The hash covers ``input`` + ``expected_output`` (but **not**
        ``metadata``) so re-tagging an item with new metadata doesn't
        invalidate its ID. If you need a different ID scheme, construct
        :class:`DatasetItem` directly with an explicit ``id``.
        """
        derived_id = content_hash({"input": input, "expected_output": expected_output})
        return cls(
            id=derived_id,
            input=input,
            expected_output=expected_output,
            metadata=metadata or {},
        )


class Dataset(BaseModel):
    """A named collection of :class:`DatasetItem` instances.

    Attributes:
        name: Human-readable identifier; also the key the stores use.
        items: Frozen tuple of items. Immutability is structural
            (Pydantic's ``frozen=True`` plus a tuple field) so callers
            can rely on ``len(dataset.items)`` not changing under them.
        description: Free-form prose describing the dataset.
        metadata: Arbitrary JSON-serializable annotations.

    A :class:`ValidationError` fires at construction when two items
    share an ``id`` — duplicate IDs make diffing ambiguous and break
    the content-hash version invariant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    items: tuple[DatasetItem, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = Field(default={})

    @model_validator(mode="after")
    def _unique_ids(self) -> Dataset:
        seen: set[str] = set()
        for index, item in enumerate(self.items):
            if item.id in seen:
                msg = (
                    f"Dataset {self.name!r}: duplicate item id {item.id!r} "
                    f"at position {index}"
                )
                raise ValidationError(msg)
            seen.add(item.id)
        return self

    def __len__(self) -> int:
        return len(self.items)

    def by_id(self, item_id: str) -> DatasetItem | None:
        """Return the item with ``id == item_id``, or ``None`` if absent."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None
