"""Push and pull a model snapshot via :class:`HFHubClient`.

This example does NOT hit the Hub — it would need a write-scope
token and a real repo. Instead, it prints what the client would
do: the HfApi calls it would issue, the kwargs it would forward,
and the local paths it would resolve.

To actually run against the Hub, set ``HF_TOKEN`` and either
import this code or change the ``DRY_RUN`` flag.

Usage::

    uv run python examples/34_storage_hf_hub.py
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

from strata_forge.storage import HFHubClient

DRY_RUN = True


class _DryHfApi:
    """A tiny in-memory HfApi stand-in that just logs every call."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.log: list[tuple[str, dict[str, Any]]] = []

    def create_repo(self, **kwargs: Any) -> str:
        self.log.append(("create_repo", kwargs))
        return f"https://huggingface.co/{kwargs['repo_id']}"

    def upload_folder(self, **kwargs: Any) -> str:
        self.log.append(("upload_folder", kwargs))
        return f"https://huggingface.co/{kwargs['repo_id']}/commit/abcdef0"


def _install_dry_module() -> _DryHfApi:
    """Replace huggingface_hub with a dry version that records calls."""
    api_instance = _DryHfApi()

    def _hf_api(**_kwargs: Any) -> _DryHfApi:
        return api_instance

    def _snapshot_download(**kwargs: Any) -> str:
        api_instance.log.append(("snapshot_download", kwargs))
        return str(kwargs.get("local_dir", "/cache/" + kwargs["repo_id"]))

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = _hf_api  # type: ignore[attr-defined]
    fake.snapshot_download = _snapshot_download  # type: ignore[attr-defined]
    fake.hf_hub_download = lambda **kw: "/cache/" + kw["filename"]  # type: ignore[attr-defined]
    sys.modules["huggingface_hub"] = fake
    return api_instance


async def _main() -> None:
    api = _install_dry_module() if DRY_RUN else None

    hub = HFHubClient(token="hf_dummy", endpoint="https://huggingface.co")  # noqa: S106

    print("--- push_model('./ckpt/llama-sft', 'me/llama-sft', private=True)")
    commit = await hub.push_model(
        "./ckpt/llama-sft",
        "me/llama-sft",
        commit_message="snapshot v0.1",
        private=True,
        allow_patterns=("*.safetensors", "*.json"),
    )
    print(f"  -> {commit}")

    print("\n--- pull_model('me/llama-sft', './weights', revision='v0.1.0')")
    path = await hub.pull_model(
        "me/llama-sft",
        "./weights",
        revision="v0.1.0",
        allow_patterns=("*.safetensors", "*.json"),
    )
    print(f"  -> {path}")

    print("\n--- push_dataset('./data/prefs', 'me/preference-pairs')")
    await hub.push_dataset(
        "./data/prefs",
        "me/preference-pairs",
        commit_message="first batch",
    )

    if api is not None:
        print("\n--- HfApi calls captured (dry run):")
        for name, kwargs in api.log:
            print(f"  {name}:")
            for k, v in kwargs.items():
                print(f"    {k} = {v!r}")


if __name__ == "__main__":
    asyncio.run(_main())
