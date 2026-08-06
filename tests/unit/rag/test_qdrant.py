"""Unit tests for `strata_forge.rag.qdrant`."""

from __future__ import annotations

import sys
import types
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from strata_forge.rag.qdrant import QdrantVectorStore
from strata_forge.rag.vector_store import VectorItem, VectorStore

# ---------------------------------------------------------------------------
# Fake qdrant_client module + AsyncQdrantClient
# ---------------------------------------------------------------------------


@dataclass
class FakeVectorParams:
    size: int
    distance: Any


@dataclass
class FakePointStruct:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class FakePointIdsList:
    points: list[str]


@dataclass
class FakeSearchHit:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None


@dataclass
class FakeQueryResponse:
    """Mirrors qdrant-client's QueryResponse (returned by query_points)."""

    points: list[FakeSearchHit] = field(default_factory=list)


class FakeAsyncQdrantClient:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.collections: dict[str, FakeVectorParams] = {}
        self.points: dict[str, dict[str, FakePointStruct]] = {}
        # call tracking for assertions
        self.upsert_calls: list[tuple[str, list[FakePointStruct]]] = []
        self.delete_calls: list[tuple[str, list[str]]] = []
        self.create_collection_calls: list[tuple[str, FakeVectorParams]] = []
        self.delete_collection_calls: list[str] = []

    async def collection_exists(self, name: str) -> bool:
        return name in self.collections

    async def create_collection(
        self, *, collection_name: str, vectors_config: FakeVectorParams
    ) -> None:
        self.collections[collection_name] = vectors_config
        self.points.setdefault(collection_name, {})
        self.create_collection_calls.append((collection_name, vectors_config))

    async def upsert(self, *, collection_name: str, points: list[FakePointStruct]) -> None:
        bucket = self.points.setdefault(collection_name, {})
        for p in points:
            bucket[p.id] = p
        self.upsert_calls.append((collection_name, list(points)))

    async def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = True,
    ) -> FakeQueryResponse:
        # Trivial similarity: rank by dot-product with the query vector.
        # Mirrors qdrant-client v1.16+ where `search()` was replaced by
        # `query_points()` returning a QueryResponse with `.points`.
        del with_payload, with_vectors
        bucket = self.points.get(collection_name, {})
        scored: list[FakeSearchHit] = []
        for point in bucket.values():
            score = sum(a * b for a, b in zip(point.vector, query, strict=False))
            scored.append(
                FakeSearchHit(
                    id=point.id,
                    score=score,
                    payload=dict(point.payload),
                    vector=list(point.vector),
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return FakeQueryResponse(points=scored[:limit])

    async def delete(self, *, collection_name: str, points_selector: FakePointIdsList) -> None:
        bucket = self.points.setdefault(collection_name, {})
        for point_id in points_selector.points:
            bucket.pop(point_id, None)
        self.delete_calls.append((collection_name, list(points_selector.points)))

    async def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)
        self.points.pop(name, None)
        self.delete_collection_calls.append(name)


class _DistanceEnum:
    COSINE = "cosine"
    DOT = "dot"
    EUCLID = "euclid"


@pytest.fixture
def install_fake_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake `qdrant_client` module in sys.modules."""
    main_mod = types.ModuleType("qdrant_client")
    main_mod.AsyncQdrantClient = FakeAsyncQdrantClient  # type: ignore[attr-defined]
    models_mod = types.ModuleType("qdrant_client.models")
    models_mod.VectorParams = FakeVectorParams  # type: ignore[attr-defined]
    models_mod.PointStruct = FakePointStruct  # type: ignore[attr-defined]
    models_mod.PointIdsList = FakePointIdsList  # type: ignore[attr-defined]
    models_mod.Distance = _DistanceEnum  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", main_mod)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models_mod)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_collection_rejected(self) -> None:
        with pytest.raises(ValueError, match="collection_name"):
            QdrantVectorStore(collection_name="")

    def test_invalid_dimensions(self) -> None:
        with pytest.raises(ValueError, match="dimensions"):
            QdrantVectorStore(collection_name="c", embedding_dimensions=0)

    def test_invalid_distance(self) -> None:
        with pytest.raises(ValueError, match="distance"):
            QdrantVectorStore(collection_name="c", distance="manhattan")

    def test_properties_exposed(self, install_fake_qdrant: None) -> None:
        del install_fake_qdrant
        store = QdrantVectorStore(collection_name="my-col", embedding_dimensions=128)
        assert store.collection_name == "my-col"
        assert store.embedding_dimensions == 128

    def test_extra_missing_raises_at_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio

        # No qdrant_client in sys.modules — block it.
        monkeypatch.setitem(sys.modules, "qdrant_client", None)
        store = QdrantVectorStore(collection_name="c")
        with pytest.raises(ImportError, match=r"\[rag\] extra"):
            asyncio.run(store.add([VectorItem(id="x", embedding=(1.0,), text="t")]))

    def test_explicit_client_short_circuits_import(self) -> None:
        # An explicit client means we never try to import qdrant_client.
        fake = FakeAsyncQdrantClient()
        store = QdrantVectorStore(collection_name="c", client=fake)
        assert store._get_client() is fake  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Add + search + delete + clear
# ---------------------------------------------------------------------------


@pytest.fixture
def store_with_client(install_fake_qdrant: None) -> tuple[QdrantVectorStore, FakeAsyncQdrantClient]:
    del install_fake_qdrant
    client = FakeAsyncQdrantClient()
    store = QdrantVectorStore(
        collection_name="forge-test",
        embedding_dimensions=2,
        client=client,
    )
    return store, client


class TestAddAndSearch:
    async def test_add_creates_collection_on_first_use(
        self, store_with_client: tuple[QdrantVectorStore, FakeAsyncQdrantClient]
    ) -> None:
        store, client = store_with_client
        await store.add([VectorItem(id="a", embedding=(1.0, 0.0), text="x")])
        assert "forge-test" in client.collections

    async def test_add_uses_deterministic_uuid_for_id(
        self, store_with_client: tuple[QdrantVectorStore, FakeAsyncQdrantClient]
    ) -> None:
        store, client = store_with_client
        await store.add([VectorItem(id="my-string-id", embedding=(1.0, 0.0), text="x")])
        # The point's id is a deterministic UUID derived from "my-string-id".
        point = next(iter(client.points["forge-test"].values()))
        # Verify it's a valid UUID string
        uuid.UUID(point.id)

    async def test_add_stashes_original_id_and_text_in_payload(
        self, store_with_client: tuple[QdrantVectorStore, FakeAsyncQdrantClient]
    ) -> None:
        store, client = store_with_client
        await store.add([VectorItem(id="orig-id", embedding=(1.0, 0.0), text="some text")])
        payload = next(iter(client.points["forge-test"].values())).payload
        assert payload["__forge_id"] == "orig-id"
        assert payload["__forge_text"] == "some text"

    async def test_add_empty_is_noop(
        self, store_with_client: tuple[QdrantVectorStore, FakeAsyncQdrantClient]
    ) -> None:
        store, client = store_with_client
        await store.add([])
        assert client.upsert_calls == []

    async def test_search_returns_vector_search_results(
        self, store_with_client: tuple[QdrantVectorStore, FakeAsyncQdrantClient]
    ) -> None:
        store, _ = store_with_client
        await store.add(
            [
                VectorItem(id="a", embedding=(1.0, 0.0), text="x-aligned"),
                VectorItem(id="b", embedding=(0.0, 1.0), text="y-aligned"),
            ]
        )
        results = await store.search([1.0, 0.0])
        assert len(results) == 2
        # Fake hit-scorer prefers x-aligned for [1, 0] query.
        assert results[0].item.id == "a"
        assert results[0].item.text == "x-aligned"

    async def test_search_top_k_limits(
        self, store_with_client: tuple[QdrantVectorStore, FakeAsyncQdrantClient]
    ) -> None:
        store, _ = store_with_client
        for i in range(5):
            await store.add([VectorItem(id=str(i), embedding=(float(i), 0.0), text=str(i))])
        results = await store.search([1.0, 0.0], top_k=2)
        assert len(results) == 2

    async def test_search_invalid_top_k(
        self, store_with_client: tuple[QdrantVectorStore, FakeAsyncQdrantClient]
    ) -> None:
        store, _ = store_with_client
        with pytest.raises(ValueError, match="top_k"):
            await store.search([1.0, 0.0], top_k=0)


class TestDelete:
    async def test_delete_by_string_id(
        self, store_with_client: tuple[QdrantVectorStore, FakeAsyncQdrantClient]
    ) -> None:
        store, _client = store_with_client
        await store.add(
            [
                VectorItem(id="a", embedding=(1.0, 0.0), text="x"),
                VectorItem(id="b", embedding=(0.0, 1.0), text="y"),
            ]
        )
        await store.delete(["a"])
        # The original "a" mapping was deleted; we can search and not see it.
        results = await store.search([1.0, 0.0])
        assert all(r.item.id != "a" for r in results)

    async def test_delete_empty_list_noop(
        self, store_with_client: tuple[QdrantVectorStore, FakeAsyncQdrantClient]
    ) -> None:
        store, client = store_with_client
        await store.delete([])
        assert client.delete_calls == []


class TestClear:
    async def test_clear_drops_and_recreates_collection(
        self, store_with_client: tuple[QdrantVectorStore, FakeAsyncQdrantClient]
    ) -> None:
        store, client = store_with_client
        await store.add([VectorItem(id="x", embedding=(1.0, 0.0), text="t")])
        await store.clear()
        # Collection was deleted and recreated; points are empty.
        assert client.delete_collection_calls == ["forge-test"]
        assert client.points.get("forge-test", {}) == {}
        # New collection exists again.
        assert "forge-test" in client.collections


# ---------------------------------------------------------------------------
# Connection kwargs
# ---------------------------------------------------------------------------


class TestConnectionKwargs:
    def test_url_takes_precedence(self, install_fake_qdrant: None) -> None:
        del install_fake_qdrant
        store = QdrantVectorStore(
            collection_name="c",
            url="https://my-cluster.example.com",
            api_key="secret",
        )
        client: FakeAsyncQdrantClient = store._get_client()  # type: ignore[assignment]
        assert client.init_kwargs.get("url") == "https://my-cluster.example.com"
        assert client.init_kwargs.get("api_key") == "secret"
        assert "host" not in client.init_kwargs

    def test_host_port_default(self, install_fake_qdrant: None) -> None:
        del install_fake_qdrant
        store = QdrantVectorStore(collection_name="c")
        client: FakeAsyncQdrantClient = store._get_client()  # type: ignore[assignment]
        assert client.init_kwargs["host"] == "localhost"
        assert client.init_kwargs["port"] == 6333


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_satisfies_vector_store_protocol(self, install_fake_qdrant: None) -> None:
        del install_fake_qdrant
        store = QdrantVectorStore(collection_name="c")
        assert isinstance(store, VectorStore)
