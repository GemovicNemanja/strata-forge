"""In-process :class:`PromptStore` — dict-backed, monotonic integer versions.

Use this for tests, notebooks, and ad-hoc scripts. Versions are
assigned monotonically per template name: the first ``put`` of
``"greet"`` returns ``"1"``, the second ``"2"``, and so on. The "latest"
version is whichever was stored most recently. The store doesn't
deduplicate by content — two consecutive ``put`` calls with identical
template bodies produce two distinct versions.

For multi-process or persistent storage, use the Langfuse-backed store
in :mod:`strata_forge.prompts.stores.langfuse`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strata_forge.prompts.registry import PromptNotFoundError, PromptStore

if TYPE_CHECKING:
    from strata_forge.prompts.template import PromptTemplate

__all__ = ["InMemoryPromptStore"]


class InMemoryPromptStore(PromptStore):
    """Dict-backed prompt store.

    Operations are atomic per asyncio scheduling quantum (no internal
    awaits during state mutation), so concurrent ``put``/``get`` from
    multiple tasks won't tear state — but if you need cross-process
    safety, reach for the Langfuse store.
    """

    def __init__(self) -> None:
        # name → list of (version, template), newest last so latest is `[-1]`.
        self._versions: dict[str, list[tuple[str, PromptTemplate]]] = {}
        # name → next integer version counter.
        self._next_version: dict[str, int] = {}

    async def get(self, name: str, version: str | None = None) -> PromptTemplate:
        entries = self._versions.get(name)
        if entries is None:
            msg = f"Template {name!r} not in store"
            raise PromptNotFoundError(msg, name=name, version=version)
        if version is None:
            return entries[-1][1]
        for stored_version, template in entries:
            if stored_version == version:
                return template
        msg = f"Version {version!r} of template {name!r} not in store"
        raise PromptNotFoundError(msg, name=name, version=version)

    async def put(self, template: PromptTemplate) -> str:
        name = template.name
        next_n = self._next_version.get(name, 1)
        version = str(next_n)
        self._next_version[name] = next_n + 1
        self._versions.setdefault(name, []).append((version, template))
        return version

    async def versions(self, name: str) -> list[str]:
        entries = self._versions.get(name)
        if entries is None:
            msg = f"Template {name!r} not in store"
            raise PromptNotFoundError(msg, name=name)
        return [v for v, _ in reversed(entries)]

    async def list_names(self) -> list[str]:
        return sorted(self._versions)

    async def delete(self, name: str, version: str | None = None) -> None:
        entries = self._versions.get(name)
        if entries is None:
            # Unknown name → no-op per PromptStore.delete contract.
            return
        if version is None:
            del self._versions[name]
            self._next_version.pop(name, None)
            return
        for index, (stored_version, _) in enumerate(entries):
            if stored_version == version:
                entries.pop(index)
                if not entries:
                    del self._versions[name]
                    self._next_version.pop(name, None)
                return
        # Name exists but the version doesn't — raise so typos surface.
        msg = f"Version {version!r} of template {name!r} not in store"
        raise PromptNotFoundError(msg, name=name, version=version)
