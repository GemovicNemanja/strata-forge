"""Qdrant-backed :class:`VectorStore`.

Requires the ``[rag]`` extra. The ``qdrant-client`` SDK is imported
lazily inside the constructor so ``import strata_forge.rag`` works without
the extra installed; the :class:`ImportError` surfaces only when a
caller actually constructs the store.

Qdrant assigns point IDs that are either ints or UUIDs;
:class:`VectorItem.id` is a free-form string. The store derives a
deterministic UUID via :func:`uuid.uuid5` so the same string id
maps to the same point across runs, and stashes the original string
on the Qdrant point's payload under ``__forge_id``.

Search results decode the payload back into :class:`VectorItem`
plus the cosine score Qdrant returns.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

from strata_forge.rag.vector_store import VectorItem, VectorSearchResult

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "QdrantVectorStore",
]

_FORGE_ID_KEY = "__forge_id"
_FORGE_TEXT_KEY = "__forge_text"
_FORGE_METADATA_KEY = "__forge_metadata"
_NAMESPACE = uuid.UUID("9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d")


def _point_id_for(item_id: str) -> str:
    """Derive a deterministic Qdrant point id (UUID string) from ``item_id``."""
    return str(uuid.uuid5(_NAMESPACE, item_id))


class QdrantVectorStore:
    """Qdrant-backed :class:`strata_forge.rag.VectorStore`.

    Args:
        collection_name: Qdrant collection holding the embeddings.
            Created on first use if it doesn't exist.
        embedding_dimensions: Vector size for the collection. Must
            match the embedder's output dimensionality. Default
            1536 (OpenAI ``text-embedding-3-small``).
        url: Qdrant cluster URL (cloud-hosted setups). When set,
            takes precedence over ``host``/``port``.
        api_key: API key for cloud-hosted Qdrant.
        host: Qdrant host (local docker setups). Default
            ``"localhost"``.
        port: Qdrant gRPC/HTTP port. Default ``6333``.
        distance: Distance metric — one of ``"cosine"``, ``"dot"``,
            ``"euclid"``. Default ``"cosine"``.
        client: Optional pre-built ``AsyncQdrantClient`` (handy for
            tests and for callers who want to share a client). When
            omitted, the store builds one from the host / port / url
            / api_key arguments.
    """

    def __init__(
        self,
        *,
        collection_name: str,
        embedding_dimensions: int = 1536,
        url: str | None = None,
        api_key: str | None = None,
        host: str = "localhost",
        port: int = 6333,
        distance: str = "cosine",
        client: Any | None = None,
    ) -> None:
        if not collection_name:
            err = "QdrantVectorStore: collection_name must be non-empty"
            raise ValueError(err)
        if embedding_dimensions < 1:
            err = (
                f"QdrantVectorStore: embedding_dimensions must be >= 1; got {embedding_dimensions}"
            )
            raise ValueError(err)
        if distance not in ("cosine", "dot", "euclid"):
            err = f"QdrantVectorStore: unsupported distance {distance!r}"
            raise ValueError(err)
        self._collection_name = collection_name
        self._embedding_dimensions = embedding_dimensions
        self._distance = distance
        self._explicit_client = client
        self._connect_kwargs: dict[str, Any] = {}
        if url is not None:
            self._connect_kwargs["url"] = url
        else:
            self._connect_kwargs["host"] = host
            self._connect_kwargs["port"] = port
        if api_key is not None:
            self._connect_kwargs["api_key"] = api_key
        self._client: Any | None = None
        self._collection_ready = False
        self._collection_lock = asyncio.Lock()

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def embedding_dimensions(self) -> int:
        return self._embedding_dimensions

    def _build_client(self) -> Any:
        if self._explicit_client is not None:
            return self._explicit_client
        try:
            qdrant_mod: Any = __import__("qdrant_client", fromlist=["AsyncQdrantClient"])
        except ImportError as exc:
            msg = (
                "The [rag] extra is required for QdrantVectorStore. "
                "Install it with: pip install 'strata-forge[rag]'."
            )
            raise ImportError(msg) from exc
        return qdrant_mod.AsyncQdrantClient(**self._connect_kwargs)

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _distance_enum(self) -> Any:
        models: Any = __import__("qdrant_client.models", fromlist=["Distance"])
        distance = models.Distance
        return {
            "cosine": distance.COSINE,
            "dot": distance.DOT,
            "euclid": distance.EUCLID,
        }[self._distance]

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        async with self._collection_lock:
            if self._collection_ready:
                return
            client = self._get_client()
            existing = await client.collection_exists(self._collection_name)
            if not existing:
                models: Any = __import__("qdrant_client.models", fromlist=["VectorParams"])
                await client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=self._embedding_dimensions,
                        distance=self._distance_enum(),
                    ),
                )
            self._collection_ready = True

    @staticmethod
    def _build_payload(item: VectorItem) -> dict[str, Any]:
        return {
            _FORGE_ID_KEY: item.id,
            _FORGE_TEXT_KEY: item.text,
            _FORGE_METADATA_KEY: dict(item.metadata),
        }

    @staticmethod
    def _decode_payload(payload: dict[str, Any], embedding: Sequence[float]) -> VectorItem:
        return VectorItem(
            id=str(payload[_FORGE_ID_KEY]),
            embedding=tuple(embedding),
            text=str(payload[_FORGE_TEXT_KEY]),
            metadata=dict(payload.get(_FORGE_METADATA_KEY) or {}),
        )

    async def add(self, items: Sequence[VectorItem]) -> None:
        if not items:
            return
        await self._ensure_collection()
        models: Any = __import__("qdrant_client.models", fromlist=["PointStruct"])
        points = [
            models.PointStruct(
                id=_point_id_for(item.id),
                vector=list(item.embedding),
                payload=self._build_payload(item),
            )
            for item in items
        ]
        await self._get_client().upsert(
            collection_name=self._collection_name,
            points=points,
        )

    async def search(
        self, embedding: Sequence[float], *, top_k: int = 5
    ) -> tuple[VectorSearchResult, ...]:
        if top_k <= 0:
            err = f"top_k must be >= 1; got {top_k}"
            raise ValueError(err)
        await self._ensure_collection()
        # qdrant-client v1.16+ removed `.search()`; `.query_points()` is
        # the replacement. The query vector moves from `query_vector=` to
        # `query=`, and the response is a QueryResponse whose `.points`
        # field holds the same ScoredPoint list the old API returned.
        response = await self._get_client().query_points(
            collection_name=self._collection_name,
            query=list(embedding),
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )
        results: list[VectorSearchResult] = []
        for hit in response.points:
            payload = dict(hit.payload or {})
            vector = list(hit.vector) if hit.vector is not None else []
            results.append(
                VectorSearchResult(
                    item=self._decode_payload(payload, vector),
                    score=float(hit.score),
                )
            )
        return tuple(results)

    async def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        await self._ensure_collection()
        models: Any = __import__("qdrant_client.models", fromlist=["PointIdsList"])
        point_ids = [_point_id_for(item_id) for item_id in ids]
        await self._get_client().delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=point_ids),
        )

    async def clear(self) -> None:
        """Drop and recreate the collection.

        Qdrant doesn't expose a "truncate" primitive; the standard
        approach is to delete the collection and recreate it with
        the same schema.
        """
        client = self._get_client()
        if await client.collection_exists(self._collection_name):
            await client.delete_collection(self._collection_name)
        self._collection_ready = False
        await self._ensure_collection()
