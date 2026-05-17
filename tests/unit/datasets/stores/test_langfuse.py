"""Unit tests for `forge.datasets.stores.langfuse`."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from forge.datasets.schema import Dataset, DatasetItem
from forge.datasets.store import DatasetNotFoundError
from forge.datasets.stores.langfuse import (
    VERSION_SEPARATOR,
    LangfuseDatasetStore,
    compose_langfuse_name,
    decompose_langfuse_name,
)
from forge.datasets.versioning import dataset_version


@dataclass
class FakeLangfuseDataset:
    """Shape that the real Langfuse `get_dataset` / list endpoint returns."""

    name: str
    items: list[Any] = field(default_factory=list)
    description: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: str = "2026-01-01T00:00:00Z"


@dataclass
class FakeLangfuseDatasetItem:
    id: str
    input: dict[str, Any]
    expected_output: Any = None
    metadata: dict[str, Any] | None = None


@dataclass
class FakeListMeta:
    total_pages: int = 1


@dataclass
class FakeListResponse:
    data: list[FakeLangfuseDataset]
    meta: FakeListMeta = field(default_factory=FakeListMeta)


class FakeDatasetsAPI:
    """Sub-API at `langfuse_client.api.datasets`."""

    def __init__(self, parent: FakeLangfuseClient) -> None:
        self._parent = parent
        self.delete_calls: list[str] = []

    def list(self, page: int = 1, limit: int = 100) -> FakeListResponse:
        page_size = max(1, limit)
        start = (page - 1) * page_size
        end = start + page_size
        items = self._parent.dataset_list[start:end]
        total = (len(self._parent.dataset_list) + page_size - 1) // page_size or 1
        return FakeListResponse(data=items, meta=FakeListMeta(total_pages=total))

    def delete(self, dataset_name: str) -> None:
        self.delete_calls.append(dataset_name)
        self._parent.datasets.pop(dataset_name, None)
        self._parent.dataset_list = [d for d in self._parent.dataset_list if d.name != dataset_name]


class FakeAPI:
    def __init__(self, parent: FakeLangfuseClient) -> None:
        self.datasets = FakeDatasetsAPI(parent)


class FakeLangfuseClient:
    """A minimal Langfuse client surface covering the methods we call."""

    def __init__(self) -> None:
        self.datasets: dict[str, FakeLangfuseDataset] = {}
        self.dataset_list: list[FakeLangfuseDataset] = []
        self.api = FakeAPI(self)
        self._counter = 0

    def _next_timestamp(self) -> str:
        self._counter += 1
        return f"2026-01-01T00:00:{self._counter:02d}Z"

    def create_dataset(
        self,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FakeLangfuseDataset:
        ds = FakeLangfuseDataset(
            name=name,
            description=description,
            metadata=metadata,
            created_at=self._next_timestamp(),
        )
        self.datasets[name] = ds
        self.dataset_list.append(ds)
        return ds

    def create_dataset_item(
        self,
        dataset_name: str,
        id: str,  # noqa: A002 — matches Langfuse SDK kwarg
        input: dict[str, Any],  # noqa: A002 — matches Langfuse SDK kwarg
        expected_output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> FakeLangfuseDatasetItem:
        item = FakeLangfuseDatasetItem(
            id=id,
            input=input,
            expected_output=expected_output,
            metadata=metadata,
        )
        self.datasets[dataset_name].items.append(item)
        return item

    def get_dataset(self, name: str) -> FakeLangfuseDataset:
        if name not in self.datasets:
            raise RuntimeError(f"Langfuse: dataset {name} not found")
        return self.datasets[name]


@pytest.fixture
def fake_client() -> FakeLangfuseClient:
    return FakeLangfuseClient()


@pytest.fixture
def store(fake_client: FakeLangfuseClient) -> LangfuseDatasetStore:
    return LangfuseDatasetStore(client=fake_client)


def _item(query: str, *, expected: Any = None) -> DatasetItem:
    return DatasetItem.from_input({"q": query}, expected_output=expected)


# ---------------------------------------------------------------------------
# Lazy import contract
# ---------------------------------------------------------------------------


class TestLazyImport:
    def test_importing_module_does_not_import_langfuse(self) -> None:
        # The whole point of the lazy import: importing the module
        # must work without the [langfuse] extra installed.
        import importlib

        from forge.datasets.stores import langfuse as lf_mod

        # Reimport with langfuse blocked to make sure the module body
        # doesn't eagerly import the SDK.
        sys.modules.pop("langfuse", None)
        importlib.reload(lf_mod)
        assert hasattr(lf_mod, "LangfuseDatasetStore")

    def test_constructor_without_langfuse_or_explicit_client_raises_at_use(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without an explicit client and without the extra, the lazy
        # build fails — but only at first use, not at construction.
        import asyncio as _asyncio

        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        from forge.config import reset_settings

        reset_settings()
        store = LangfuseDatasetStore()  # construction is cheap, succeeds
        with pytest.raises(RuntimeError, match="not configured"):
            _asyncio.run(store.list_names())

    def test_extra_missing_raises_import_error_at_use(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGFUSE_HOST", "http://x")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setitem(sys.modules, "langfuse", None)
        from forge.config import reset_settings

        reset_settings()
        store = LangfuseDatasetStore()
        import asyncio as _asyncio

        with pytest.raises(ImportError, match=r"\[langfuse\] extra"):
            _asyncio.run(store.list_names())


# ---------------------------------------------------------------------------
# Construction with explicit client
# ---------------------------------------------------------------------------


class TestExplicitClient:
    def test_uses_explicit_client(self, fake_client: FakeLangfuseClient) -> None:
        store = LangfuseDatasetStore(client=fake_client)
        # _get_client must return the injected client.
        assert store._get_client() is fake_client  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# put — content-hash version, dedup on identical content
# ---------------------------------------------------------------------------


class TestPut:
    async def test_put_creates_versioned_langfuse_dataset(
        self,
        store: LangfuseDatasetStore,
        fake_client: FakeLangfuseClient,
    ) -> None:
        ds = Dataset(name="myset", items=(_item("a"), _item("b")))
        version = await store.put(ds)
        composed = f"myset{VERSION_SEPARATOR}{version}"
        assert composed in fake_client.datasets
        stored = fake_client.datasets[composed]
        assert len(stored.items) == 2

    async def test_put_returns_content_hash_version(self, store: LangfuseDatasetStore) -> None:
        ds = Dataset(name="myset", items=(_item("a"),))
        version = await store.put(ds)
        assert version == dataset_version(ds)

    async def test_put_is_idempotent_on_identical_content(
        self,
        store: LangfuseDatasetStore,
        fake_client: FakeLangfuseClient,
    ) -> None:
        ds = Dataset(name="myset", items=(_item("a"),))
        v1 = await store.put(ds)
        v2 = await store.put(ds)
        assert v1 == v2
        # Only one Langfuse dataset was created — the second put short-circuited.
        composed = f"myset{VERSION_SEPARATOR}{v1}"
        assert len(fake_client.datasets) == 1
        assert composed in fake_client.datasets

    async def test_put_distinct_content_distinct_versions(
        self,
        store: LangfuseDatasetStore,
        fake_client: FakeLangfuseClient,
    ) -> None:
        ds1 = Dataset(name="myset", items=(_item("a"),))
        ds2 = Dataset(name="myset", items=(_item("b"),))
        v1 = await store.put(ds1)
        v2 = await store.put(ds2)
        assert v1 != v2
        assert len(fake_client.datasets) == 2

    async def test_put_includes_description_and_metadata(
        self,
        store: LangfuseDatasetStore,
        fake_client: FakeLangfuseClient,
    ) -> None:
        ds = Dataset(
            name="myset",
            items=(_item("a"),),
            description="hello",
            metadata={"source": "synthetic"},
        )
        version = await store.put(ds)
        composed = f"myset{VERSION_SEPARATOR}{version}"
        stored = fake_client.datasets[composed]
        assert stored.description == "hello"
        assert stored.metadata == {"source": "synthetic"}


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    async def test_get_latest_returns_most_recently_created(
        self,
        store: LangfuseDatasetStore,
    ) -> None:
        v1 = await store.put(Dataset(name="myset", items=(_item("a"),)))
        v2 = await store.put(Dataset(name="myset", items=(_item("b"),)))
        assert v1 != v2
        latest = await store.get("myset")
        # Whichever was put most recently is "latest"; we check by version equality.
        # Order in fake list = creation order, so v2 is latest.
        recovered_version = dataset_version(latest)
        assert recovered_version == v2

    async def test_get_specific_version(self, store: LangfuseDatasetStore) -> None:
        v1 = await store.put(Dataset(name="myset", items=(_item("a"),)))
        await store.put(Dataset(name="myset", items=(_item("b"),)))
        recovered = await store.get("myset", version=v1)
        assert dataset_version(recovered) == v1

    async def test_get_unknown_name_raises(self, store: LangfuseDatasetStore) -> None:
        with pytest.raises(DatasetNotFoundError) as info:
            await store.get("nonexistent")
        assert info.value.name == "nonexistent"
        assert info.value.version is None

    async def test_get_unknown_version_raises(self, store: LangfuseDatasetStore) -> None:
        await store.put(Dataset(name="myset", items=(_item("a"),)))
        with pytest.raises(DatasetNotFoundError) as info:
            await store.get("myset", version="bogus")
        assert info.value.name == "myset"
        assert info.value.version == "bogus"

    async def test_get_round_trips_items(self, store: LangfuseDatasetStore) -> None:
        original = Dataset(
            name="myset",
            items=(
                _item("a", expected="A"),
                _item("b", expected="B"),
            ),
        )
        await store.put(original)
        recovered = await store.get("myset")
        assert {it.id for it in recovered.items} == {it.id for it in original.items}
        for orig_item, rec_item in zip(
            sorted(original.items, key=lambda x: x.id),
            sorted(recovered.items, key=lambda x: x.id),
            strict=True,
        ):
            assert rec_item.input == orig_item.input
            assert rec_item.expected_output == orig_item.expected_output


# ---------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------


class TestVersions:
    async def test_versions_returns_newest_first(self, store: LangfuseDatasetStore) -> None:
        v1 = await store.put(Dataset(name="myset", items=(_item("a"),)))
        v2 = await store.put(Dataset(name="myset", items=(_item("b"),)))
        v3 = await store.put(Dataset(name="myset", items=(_item("c"),)))
        versions = await store.versions("myset")
        assert versions == [v3, v2, v1]

    async def test_versions_unknown_name_raises(self, store: LangfuseDatasetStore) -> None:
        with pytest.raises(DatasetNotFoundError):
            await store.versions("nonexistent")

    async def test_versions_only_for_matching_name(self, store: LangfuseDatasetStore) -> None:
        await store.put(Dataset(name="a", items=(_item("x"),)))
        await store.put(Dataset(name="b", items=(_item("y"),)))
        versions = await store.versions("a")
        assert len(versions) == 1


# ---------------------------------------------------------------------------
# list_names
# ---------------------------------------------------------------------------


class TestListNames:
    async def test_list_names_returns_sorted_distinct(self, store: LangfuseDatasetStore) -> None:
        await store.put(Dataset(name="b-set", items=(_item("a"),)))
        await store.put(Dataset(name="a-set", items=(_item("a"),)))
        await store.put(Dataset(name="a-set", items=(_item("b"),)))
        names = await store.list_names()
        assert names == ["a-set", "b-set"]

    async def test_list_names_ignores_non_forge_datasets(
        self,
        store: LangfuseDatasetStore,
        fake_client: FakeLangfuseClient,
    ) -> None:
        # A Langfuse dataset whose name doesn't follow the
        # `{forge_name}__v{version}` convention is invisible to Forge.
        fake_client.create_dataset(name="legacy-unversioned")
        await store.put(Dataset(name="real", items=(_item("a"),)))
        names = await store.list_names()
        assert names == ["real"]

    async def test_list_names_when_empty(self, store: LangfuseDatasetStore) -> None:
        assert await store.list_names() == []


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_delete_specific_version(
        self,
        store: LangfuseDatasetStore,
        fake_client: FakeLangfuseClient,
    ) -> None:
        v1 = await store.put(Dataset(name="myset", items=(_item("a"),)))
        v2 = await store.put(Dataset(name="myset", items=(_item("b"),)))
        await store.delete("myset", version=v1)
        remaining = await store.versions("myset")
        assert remaining == [v2]
        assert fake_client.api.datasets.delete_calls == [f"myset{VERSION_SEPARATOR}{v1}"]

    async def test_delete_all_versions(
        self,
        store: LangfuseDatasetStore,
        fake_client: FakeLangfuseClient,
    ) -> None:
        v1 = await store.put(Dataset(name="myset", items=(_item("a"),)))
        v2 = await store.put(Dataset(name="myset", items=(_item("b"),)))
        await store.delete("myset")
        with pytest.raises(DatasetNotFoundError):
            await store.versions("myset")
        composed1 = f"myset{VERSION_SEPARATOR}{v1}"
        composed2 = f"myset{VERSION_SEPARATOR}{v2}"
        assert set(fake_client.api.datasets.delete_calls) == {composed1, composed2}

    async def test_delete_unknown_name_is_noop(
        self,
        store: LangfuseDatasetStore,
        fake_client: FakeLangfuseClient,
    ) -> None:
        await store.delete("nonexistent")  # must not raise
        assert fake_client.api.datasets.delete_calls == []

    async def test_delete_unknown_version_raises(self, store: LangfuseDatasetStore) -> None:
        await store.put(Dataset(name="myset", items=(_item("a"),)))
        with pytest.raises(DatasetNotFoundError) as info:
            await store.delete("myset", version="bogus")
        assert info.value.name == "myset"
        assert info.value.version == "bogus"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


class TestNameComposition:
    def test_compose_and_decompose_roundtrip(self) -> None:
        composed = compose_langfuse_name("myset", "abc123")
        assert composed == f"myset{VERSION_SEPARATOR}abc123"
        result = decompose_langfuse_name(composed)
        assert result == ("myset", "abc123")

    def test_decompose_without_separator_returns_none(self) -> None:
        assert decompose_langfuse_name("legacy-name") is None

    def test_decompose_uses_rightmost_separator(self) -> None:
        # A user-chosen name that happens to contain the separator
        # should still decompose to (full_name_minus_last_chunk, version).
        composed = f"weird{VERSION_SEPARATOR}name{VERSION_SEPARATOR}abc"
        result = decompose_langfuse_name(composed)
        assert result == (f"weird{VERSION_SEPARATOR}name", "abc")


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestPagination:
    async def test_list_walks_all_pages(self) -> None:
        client = FakeLangfuseClient()
        for i in range(250):
            client.create_dataset(name=f"ds{i}{VERSION_SEPARATOR}v{i}")
        store = LangfuseDatasetStore(client=client)
        names = await store.list_names()
        # Each fake-created dataset has a distinct forge name, so we
        # expect 250 unique names back — page=100 page size means
        # this only succeeds if pagination is followed.
        assert len(names) == 250


# ---------------------------------------------------------------------------
# Settings-based construction (no explicit client)
# ---------------------------------------------------------------------------


def _install_fake_langfuse(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake_module = types.ModuleType("langfuse")
    constructor = MagicMock(return_value=FakeLangfuseClient())
    fake_module.Langfuse = constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    return constructor


class TestSettingsBasedClient:
    async def test_builds_client_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_HOST", "http://test-langfuse")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-x")
        from forge.config import reset_settings

        reset_settings()
        constructor = _install_fake_langfuse(monkeypatch)
        store = LangfuseDatasetStore()
        await store.list_names()
        constructor.assert_called_once_with(
            host="http://test-langfuse",
            public_key="pk-x",
            secret_key="sk-x",
        )

    async def test_caches_built_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_HOST", "http://test-langfuse")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-x")
        from forge.config import reset_settings

        reset_settings()
        constructor = _install_fake_langfuse(monkeypatch)
        store = LangfuseDatasetStore()
        await store.list_names()
        await store.list_names()
        # Cached: built once across multiple async calls.
        assert constructor.call_count == 1
