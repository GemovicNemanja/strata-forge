"""Bidirectional bridge between Forge and Hugging Face ``Dataset`` objects.

:func:`to_hf_dataset` flattens a Forge :class:`Dataset` into an HF
``Dataset`` with one row per item and four columns (``id``, ``input``,
``expected_output``, ``metadata``). Forge-level name, description, and
metadata round-trip through ``Dataset.info``.

:func:`from_hf_dataset` is the inverse: rows become items. Datasets
that didn't originate in Forge can be ingested by passing the column
names of the input / expected-output / metadata fields. Items without
a stored ``id`` get content-hash IDs via
:meth:`DatasetItem.from_input`.

The ``datasets`` package is imported lazily inside each conversion
function, so importing :mod:`forge.datasets.hf_bridge` works without
the ``[hf]`` extra installed.
"""

from __future__ import annotations

from typing import Any

from forge.datasets.schema import Dataset, DatasetItem

__all__ = [
    "from_hf_dataset",
    "to_hf_dataset",
]


def _import_datasets() -> Any:
    try:
        import datasets as _datasets  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        msg = (
            "The [hf] extra is required for hf_bridge. Install it with: pip install 'strata-forge[hf]'."
        )
        raise ImportError(msg) from exc
    return _datasets


def to_hf_dataset(dataset: Dataset) -> Any:
    """Convert a Forge :class:`Dataset` to a Hugging Face ``Dataset``.

    Each Forge item becomes one HF row with columns ``id``, ``input``,
    ``expected_output``, ``metadata``. Forge-level ``name`` /
    ``description`` are stored on ``Dataset.info`` for downstream
    consumers; Forge-level ``metadata`` is JSON-encoded into
    ``info.description`` only when it can't fit on a typed field.
    """
    datasets_mod = _import_datasets()

    rows: dict[str, list[Any]] = {
        "id": [item.id for item in dataset.items],
        "input": [dict(item.input) for item in dataset.items],
        "expected_output": [item.expected_output for item in dataset.items],
        "metadata": [dict(item.metadata) for item in dataset.items],
    }
    hf_ds = datasets_mod.Dataset.from_dict(rows)
    info = hf_ds.info
    info.dataset_name = dataset.name
    if dataset.description:
        info.description = dataset.description
    return hf_ds


def from_hf_dataset(
    hf_dataset: Any,
    *,
    name: str,
    description: str = "",
    metadata: dict[str, Any] | None = None,
    id_column: str = "id",
    input_column: str = "input",
    expected_output_column: str | None = "expected_output",
    metadata_column: str | None = "metadata",
) -> Dataset:
    """Convert a Hugging Face ``Dataset`` to a Forge :class:`Dataset`.

    Columns are addressed by name (``input_column`` defaults to
    ``"input"``, etc.) so HF datasets that use different conventions
    can be ingested without pre-processing. Items without an
    ``id_column`` value (missing column, or per-row ``None``) get
    content-hash IDs via :meth:`DatasetItem.from_input`.

    Setting ``expected_output_column=None`` or
    ``metadata_column=None`` ingests rows without those fields. The
    target column is then ignored even if it exists.
    """
    columns: list[str] = list(getattr(hf_dataset, "column_names", []) or [])
    has_id = id_column in columns
    has_input = input_column in columns
    if not has_input:
        msg = (
            f"Input column {input_column!r} missing from HF dataset; available columns: {columns!r}"
        )
        raise KeyError(msg)
    has_expected = expected_output_column is not None and expected_output_column in columns
    has_metadata = metadata_column is not None and metadata_column in columns

    items: list[DatasetItem] = []
    for row in hf_dataset:
        row_input: Any = row[input_column]
        if not isinstance(row_input, dict):
            msg = f"Column {input_column!r} must hold dicts; got {type(row_input).__name__}"
            raise TypeError(msg)
        input_dict: dict[str, Any] = dict(row_input)  # type: ignore[arg-type]
        row_expected: Any = row[expected_output_column] if has_expected else None
        raw_meta: Any = row[metadata_column] if has_metadata else None
        row_metadata: dict[str, Any] = dict(raw_meta) if raw_meta else {}  # type: ignore[arg-type]
        row_id: Any = row[id_column] if has_id else None
        if row_id:
            items.append(
                DatasetItem(
                    id=str(row_id),
                    input=input_dict,
                    expected_output=row_expected,
                    metadata=row_metadata,
                )
            )
        else:
            items.append(
                DatasetItem.from_input(
                    input_dict,
                    expected_output=row_expected,
                    metadata=row_metadata,
                )
            )

    return Dataset(
        name=name,
        items=tuple(items),
        description=description,
        metadata=metadata or {},
    )
