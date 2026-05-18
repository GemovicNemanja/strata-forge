"""HuggingFace Hub helpers for models, datasets, and individual files.

:class:`HFHubClient` wraps ``huggingface_hub.HfApi`` with an async
public surface and a small set of high-level operations Forge
needs day-to-day: push/pull whole repo snapshots, push/pull
individual files, create/delete repos, and list repo contents.
Every call off-loads the sync HfApi method via
``asyncio.to_thread``.

The ``huggingface_hub`` module is imported lazily inside
:meth:`HFHubClient._api`, so importing
:mod:`forge.storage.hf_hub` works without the ``[storage]``
extra installed — the :class:`ImportError` surfaces only on
first network call.

This client is intentionally thin. Anything fancy (revisions,
LFS, allow/ignore patterns) is exposed via the ``extras={}``
passthroughs on each method — Forge never blocks you from the
full HfApi surface.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["HFHubClient", "RepoType"]


type RepoType = Literal["model", "dataset", "space"]
"""HuggingFace Hub repo categories. ``model`` is the default."""


class HFHubClient:
    """Async wrapper around ``huggingface_hub.HfApi``.

    Args:
        token: Hugging Face access token. When ``None``, reads
            from the ``HF_TOKEN`` env var (or the cached login).
        endpoint: Custom Hub endpoint URL. Default is the public
            ``https://huggingface.co``.
        api: Optional pre-built ``HfApi`` instance — useful for
            tests and for sharing one API client across many
            calls. When set, ``token`` / ``endpoint`` are ignored.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        endpoint: str | None = None,
        api: Any | None = None,
    ) -> None:
        self._token = token
        self._endpoint = endpoint
        self._explicit_api = api
        self._api_cached: Any | None = None

    def _import_hf(self) -> Any:
        try:
            return __import__("huggingface_hub")
        except ImportError as exc:
            msg = (
                "The [storage] extra is required for HFHubClient. "
                "Install it with: pip install 'ai-forge[storage]'."
            )
            raise ImportError(msg) from exc

    def _api(self) -> Any:
        if self._explicit_api is not None:
            return self._explicit_api
        if self._api_cached is not None:
            return self._api_cached
        hf_mod = self._import_hf()
        kwargs: dict[str, Any] = {}
        if self._token is not None:
            kwargs["token"] = self._token
        if self._endpoint is not None:
            kwargs["endpoint"] = self._endpoint
        self._api_cached = hf_mod.HfApi(**kwargs)
        return self._api_cached

    # -----------------------------------------------------------------
    # Repo lifecycle
    # -----------------------------------------------------------------

    async def create_repo(
        self,
        repo_id: str,
        *,
        repo_type: RepoType = "model",
        private: bool = False,
        exist_ok: bool = True,
        extras: Mapping[str, Any] | None = None,
    ) -> str:
        """Create a repo on the Hub. Returns the canonical repo URL.

        Args:
            repo_id: ``"owner/name"``.
            repo_type: ``"model"`` / ``"dataset"`` / ``"space"``.
            private: Visibility.
            exist_ok: Don't raise when the repo already exists.
            extras: Verbatim passthrough to ``HfApi.create_repo``
                (e.g. ``space_sdk=`` for spaces).
        """
        api = self._api()
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "private": private,
            "exist_ok": exist_ok,
        }
        if extras:
            kwargs.update(extras)
        url: Any = await asyncio.to_thread(api.create_repo, **kwargs)
        return str(url)

    async def delete_repo(
        self,
        repo_id: str,
        *,
        repo_type: RepoType = "model",
    ) -> None:
        """Delete a repo from the Hub."""
        api = self._api()
        await asyncio.to_thread(api.delete_repo, repo_id=repo_id, repo_type=repo_type)

    async def list_repo_files(
        self,
        repo_id: str,
        *,
        repo_type: RepoType = "model",
        revision: str | None = None,
    ) -> tuple[str, ...]:
        """List file paths in a repo at the given revision."""
        api = self._api()
        kwargs: dict[str, Any] = {"repo_id": repo_id, "repo_type": repo_type}
        if revision is not None:
            kwargs["revision"] = revision
        raw: Any = await asyncio.to_thread(api.list_repo_files, **kwargs)
        return tuple(str(p) for p in raw)

    # -----------------------------------------------------------------
    # Single-file operations
    # -----------------------------------------------------------------

    async def download_file(
        self,
        repo_id: str,
        filename: str,
        *,
        local_dir: str | Path | None = None,
        repo_type: RepoType = "model",
        revision: str | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> Path:
        """Download one file from a repo. Returns the local path."""
        hf_mod = self._import_hf()
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "filename": filename,
            "repo_type": repo_type,
        }
        if local_dir is not None:
            kwargs["local_dir"] = str(local_dir)
        if revision is not None:
            kwargs["revision"] = revision
        if self._token is not None:
            kwargs["token"] = self._token
        if self._endpoint is not None:
            kwargs["endpoint"] = self._endpoint
        if extras:
            kwargs.update(extras)
        path: Any = await asyncio.to_thread(hf_mod.hf_hub_download, **kwargs)
        return Path(str(path))

    async def upload_file(
        self,
        local_path: str | Path,
        path_in_repo: str,
        repo_id: str,
        *,
        repo_type: RepoType = "model",
        commit_message: str | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> str:
        """Upload one file. Returns the resulting commit URL / hash."""
        api = self._api()
        kwargs: dict[str, Any] = {
            "path_or_fileobj": str(local_path),
            "path_in_repo": path_in_repo,
            "repo_id": repo_id,
            "repo_type": repo_type,
        }
        if commit_message is not None:
            kwargs["commit_message"] = commit_message
        if extras:
            kwargs.update(extras)
        result: Any = await asyncio.to_thread(api.upload_file, **kwargs)
        return str(result)

    # -----------------------------------------------------------------
    # Snapshot operations (whole models, whole datasets)
    # -----------------------------------------------------------------

    async def download_snapshot(
        self,
        repo_id: str,
        *,
        local_dir: str | Path | None = None,
        repo_type: RepoType = "model",
        revision: str | None = None,
        allow_patterns: tuple[str, ...] | None = None,
        ignore_patterns: tuple[str, ...] | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> Path:
        """Download an entire repo snapshot. Returns the local dir."""
        hf_mod = self._import_hf()
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "repo_type": repo_type,
        }
        if local_dir is not None:
            kwargs["local_dir"] = str(local_dir)
        if revision is not None:
            kwargs["revision"] = revision
        if allow_patterns is not None:
            kwargs["allow_patterns"] = list(allow_patterns)
        if ignore_patterns is not None:
            kwargs["ignore_patterns"] = list(ignore_patterns)
        if self._token is not None:
            kwargs["token"] = self._token
        if self._endpoint is not None:
            kwargs["endpoint"] = self._endpoint
        if extras:
            kwargs.update(extras)
        path: Any = await asyncio.to_thread(hf_mod.snapshot_download, **kwargs)
        return Path(str(path))

    async def upload_folder(
        self,
        local_dir: str | Path,
        repo_id: str,
        *,
        repo_type: RepoType = "model",
        path_in_repo: str | None = None,
        commit_message: str | None = None,
        allow_patterns: tuple[str, ...] | None = None,
        ignore_patterns: tuple[str, ...] | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> str:
        """Upload an entire directory. Returns the commit URL / hash."""
        api = self._api()
        kwargs: dict[str, Any] = {
            "folder_path": str(local_dir),
            "repo_id": repo_id,
            "repo_type": repo_type,
        }
        if path_in_repo is not None:
            kwargs["path_in_repo"] = path_in_repo
        if commit_message is not None:
            kwargs["commit_message"] = commit_message
        if allow_patterns is not None:
            kwargs["allow_patterns"] = list(allow_patterns)
        if ignore_patterns is not None:
            kwargs["ignore_patterns"] = list(ignore_patterns)
        if extras:
            kwargs.update(extras)
        result: Any = await asyncio.to_thread(api.upload_folder, **kwargs)
        return str(result)

    # -----------------------------------------------------------------
    # Convenience wrappers
    # -----------------------------------------------------------------

    async def push_model(
        self,
        local_dir: str | Path,
        repo_id: str,
        *,
        commit_message: str | None = None,
        private: bool = False,
        allow_patterns: tuple[str, ...] | None = None,
        ignore_patterns: tuple[str, ...] | None = None,
    ) -> str:
        """Create-if-missing + upload a model directory to ``repo_id``.

        Convenience over :meth:`create_repo` + :meth:`upload_folder`.
        """
        await self.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        return await self.upload_folder(
            local_dir,
            repo_id,
            repo_type="model",
            commit_message=commit_message,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    async def pull_model(
        self,
        repo_id: str,
        local_dir: str | Path,
        *,
        revision: str | None = None,
        allow_patterns: tuple[str, ...] | None = None,
    ) -> Path:
        """Download a model snapshot to ``local_dir``."""
        return await self.download_snapshot(
            repo_id,
            local_dir=local_dir,
            repo_type="model",
            revision=revision,
            allow_patterns=allow_patterns,
        )

    async def push_dataset(
        self,
        local_dir: str | Path,
        repo_id: str,
        *,
        commit_message: str | None = None,
        private: bool = False,
    ) -> str:
        """Create-if-missing + upload a dataset directory to ``repo_id``."""
        await self.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        return await self.upload_folder(
            local_dir,
            repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )

    async def pull_dataset(
        self,
        repo_id: str,
        local_dir: str | Path,
        *,
        revision: str | None = None,
    ) -> Path:
        """Download a dataset snapshot to ``local_dir``."""
        return await self.download_snapshot(
            repo_id,
            local_dir=local_dir,
            repo_type="dataset",
            revision=revision,
        )
