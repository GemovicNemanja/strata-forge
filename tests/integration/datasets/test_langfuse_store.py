"""Integration test for :class:`LangfuseDatasetStore` against a live stack.

Marked ``@pytest.mark.integration`` so it doesn't run during the
default ``make test`` pass. To exercise:

1. ``make stack-up`` to start the local Langfuse stack.
2. Set ``LANGFUSE_HOST``, ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``
   to point at the local instance.
3. ``make integration``.

The test exercises a full put/versions/get/delete cycle on a uniquely
named dataset so reruns don't conflict and so a single stack
instance can host multiple test runs in parallel without
collisions.
"""

from __future__ import annotations

import os
import uuid

import pytest

from strata_forge.datasets import (
    Dataset,
    DatasetItem,
    LangfuseDatasetStore,
    dataset_version,
)

pytestmark = pytest.mark.integration


def _langfuse_configured() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


@pytest.fixture(autouse=True)
def _skip_unless_configured() -> None:  # pyright: ignore[reportUnusedFunction]
    if not _langfuse_configured():
        pytest.skip("Langfuse env vars not set; run `make stack-up` and configure")


@pytest.fixture
def unique_name() -> str:
    """Each test gets a unique dataset name so reruns / parallel runs don't collide."""
    return f"forge-test-{uuid.uuid4().hex[:12]}"


class TestLangfuseStoreLive:
    async def test_put_get_versions_delete_cycle(self, unique_name: str) -> None:
        store = LangfuseDatasetStore()
        try:
            ds = Dataset(
                name=unique_name,
                description="integration test",
                items=(
                    DatasetItem.from_input({"question": "What is 2 + 2?"}, expected_output="4"),
                    DatasetItem.from_input(
                        {"question": "Capital of Japan?"}, expected_output="Tokyo"
                    ),
                ),
            )
            version = await store.put(ds)
            assert version == dataset_version(ds)

            # versions() reports our newly-created version.
            versions = await store.versions(unique_name)
            assert version in versions

            # get() round-trips the items.
            recovered = await store.get(unique_name, version=version)
            assert {item.id for item in recovered.items} == {item.id for item in ds.items}

            # Putting identical content is a no-op (still one version).
            again = await store.put(ds)
            assert again == version
            assert len(await store.versions(unique_name)) == 1
        finally:
            await store.delete(unique_name)
