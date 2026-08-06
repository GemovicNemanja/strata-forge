"""Unit tests for `strata_forge.datasets.store` — the abstract interface + error type."""

from __future__ import annotations

import pytest

from strata_forge.core.errors import ForgeError
from strata_forge.datasets.store import DatasetNotFoundError, DatasetStore


class TestDatasetNotFoundError:
    def test_is_forge_error(self) -> None:
        assert issubclass(DatasetNotFoundError, ForgeError)

    def test_carries_name_only(self) -> None:
        err = DatasetNotFoundError("missing", name="train")
        assert err.name == "train"
        assert err.version is None

    def test_carries_name_and_version(self) -> None:
        err = DatasetNotFoundError("missing v3", name="train", version="3")
        assert err.name == "train"
        assert err.version == "3"


class TestDatasetStoreAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            DatasetStore()  # pyright: ignore[reportAbstractUsage]
