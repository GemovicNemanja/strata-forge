"""Round-trip a `Dataset` through Hugging Face Datasets.

Requires the ``[hf]`` extra (``pip install 'ai-forge[hf]'``). The
example skips cleanly when the extra isn't installed.

Usage::

    uv run python examples/18_dataset_hf_bridge.py
    uv pip install datasets  # if you only need the bridge locally

The HF bridge is bidirectional: any Hugging Face `Dataset` can be
ingested by pointing :func:`from_hf_dataset` at the right column
names; any Forge :class:`Dataset` can be converted to an HF object
via :func:`to_hf_dataset` for upload to the Hub, parquet
serialization, or hand-off to the training pipeline.
"""

from __future__ import annotations

import sys

from strata_forge.datasets import (
    Dataset,
    DatasetItem,
    from_hf_dataset,
    to_hf_dataset,
)


def _main() -> None:
    try:
        import datasets as _  # noqa: F401  # check the extra is installed
    except ImportError:
        print("[skip] [hf] extra not installed; pip install 'ai-forge[hf]'", file=sys.stderr)
        sys.exit(0)

    original = Dataset(
        name="qa-mini",
        description="Two-item Q&A sample.",
        items=(
            DatasetItem.from_input(
                {"question": "What's the boiling point of water in Celsius?"},
                expected_output="100",
                metadata={"tier": "easy"},
            ),
            DatasetItem.from_input(
                {"question": "Name the largest ocean."},
                expected_output="Pacific",
                metadata={"tier": "easy"},
            ),
        ),
    )

    # Forge -> HF
    hf_ds = to_hf_dataset(original)
    print("--- HF dataset shape ---")
    print(f"  rows:     {len(hf_ds)}")
    print(f"  columns:  {hf_ds.column_names}")
    print(f"  info.name:        {hf_ds.info.dataset_name}")
    print(f"  info.description: {hf_ds.info.description}")

    # HF -> Forge (round trip — proves bidirectionality)
    recovered = from_hf_dataset(
        hf_ds,
        name=original.name,
        description=original.description,
        metadata=dict(original.metadata),
    )
    print("\n--- round-trip recovered Forge dataset ---")
    print(f"  name:        {recovered.name}")
    print(f"  description: {recovered.description}")
    print(f"  items:       {len(recovered)}")
    for item in recovered.items:
        print(f"    [{item.id[:12]}...] {item.input['question']} -> {item.expected_output!r}")

    assert recovered == original
    print("\n--- equality check: recovered == original")


if __name__ == "__main__":
    _main()
