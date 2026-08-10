"""Build a `Dataset`, store + retrieve versions, compute a diff.

No provider keys or optional extras needed — the in-memory store
ships in the core install. The example demonstrates:

- :class:`DatasetItem.from_input` for content-hash IDs.
- :class:`InMemoryDatasetStore.put` / ``.get`` / ``.versions`` with
  natural deduplication on identical content.
- :func:`dataset_version` and :func:`diff` for tracking what
  changed between two versions of the same dataset.

Usage::

    uv run python examples/18_dataset_basics.py
"""

from __future__ import annotations

import asyncio

from strata_forge.datasets import (
    Dataset,
    DatasetItem,
    InMemoryDatasetStore,
    dataset_version,
    diff,
)


def _build_v1() -> Dataset:
    items = (
        DatasetItem.from_input(
            {"question": "What is the capital of France?"},
            expected_output="Paris",
        ),
        DatasetItem.from_input(
            {"question": "What is 2 + 2?"},
            expected_output="4",
        ),
    )
    return Dataset(
        name="trivia",
        items=items,
        description="A tiny trivia eval set.",
    )


def _build_v2(v1: Dataset) -> Dataset:
    """Drop one item, keep one, add two new ones."""
    keep = v1.items[0]  # capital of France stays
    additions = (
        DatasetItem.from_input(
            {"question": "What language do they speak in Brazil?"},
            expected_output="Portuguese",
        ),
        DatasetItem.from_input(
            {"question": "How many continents are there?"},
            expected_output="7",
        ),
    )
    return Dataset(
        name="trivia",
        items=(keep, *additions),
        description="A tiny trivia eval set, revised.",
    )


async def _main() -> None:
    store = InMemoryDatasetStore()

    v1 = _build_v1()
    v1_version = await store.put(v1)
    print(f"--- put v1: version={v1_version[:16]}... ({len(v1)} items)")

    # Putting the same content twice is idempotent (content-hash versioning).
    repeat_version = await store.put(v1)
    print(
        f"--- repeat put: version={repeat_version[:16]}... (same as above: {repeat_version == v1_version})"
    )

    v2 = _build_v2(v1)
    v2_version = await store.put(v2)
    print(f"--- put v2: version={v2_version[:16]}... ({len(v2)} items)")

    # Verify dataset_version() agrees with the store-assigned version.
    assert dataset_version(v1) == v1_version
    assert dataset_version(v2) == v2_version

    versions = await store.versions("trivia")
    print("\n--- versions of 'trivia' (newest first):")
    for v in versions:
        print(f"  {v[:16]}...")

    delta = diff(v1, v2)
    print("\n--- diff(v1, v2):")
    print(f"  added:     {len(delta.added)}")
    print(f"  removed:   {len(delta.removed)}")
    print(f"  unchanged: {len(delta.unchanged)}")
    for item in delta.added:
        print(f"    + {item.input['question']}")
    for item in delta.removed:
        print(f"    - {item.input['question']}")

    # Fetch a specific historical version.
    recovered = await store.get("trivia", version=v1_version)
    assert recovered == v1
    print(f"\n--- recovered v1: {len(recovered)} items, name={recovered.name!r}")


if __name__ == "__main__":
    asyncio.run(_main())
