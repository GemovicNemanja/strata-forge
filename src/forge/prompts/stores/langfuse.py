"""Langfuse-backed :class:`PromptStore` — requires the ``[langfuse]`` extra.

Maps Forge's :class:`PromptTemplate` onto Langfuse's prompt management
API: the stable + dynamic sections serialize into the ``prompt`` field
as a JSON blob, and the typed fields (declared variables, description,
metadata) live in the ``config`` dict. Versions are whatever Langfuse
returns — we stringify them for the abstract interface.

The ``langfuse`` package is imported lazily inside the constructor so
``import forge.prompts`` works without the extra installed; only
constructing :class:`LangfusePromptStore` triggers the import.

.. note::

   Langfuse's Python SDK doesn't expose prompt deletion as a stable
   API, so :meth:`LangfusePromptStore.delete` raises
   :exc:`NotImplementedError` with a pointer to the Langfuse UI. The
   ``versions`` and ``list_names`` methods are best-effort — see the
   docstrings for the assumptions they make.
"""

from __future__ import annotations

import json
from typing import Any

from forge.core.errors import ForgeError
from forge.prompts.registry import PromptNotFoundError, PromptStore
from forge.prompts.template import PromptTemplate

__all__ = [
    "LangfusePromptStore",
]


_TEMPLATE_FLAG = "forge_template_v1"


def _serialize(template: PromptTemplate) -> tuple[str, dict[str, Any]]:
    """Pack a :class:`PromptTemplate` into ``(prompt_body, config_dict)``.

    The prompt body is a JSON blob with the two sections; the config
    dict carries the typed fields Langfuse's prompt model can't
    represent natively.
    """
    body = json.dumps(
        {
            "stable_section": template.stable_section,
            "dynamic_section": template.dynamic_section,
        }
    )
    config: dict[str, Any] = {
        _TEMPLATE_FLAG: True,
        "stable_variables": list(template.stable_variables),
        "dynamic_variables": list(template.dynamic_variables),
        "description": template.description,
        "metadata": template.metadata,
    }
    return body, config


def _deserialize(name: str, body: str, config: dict[str, Any]) -> PromptTemplate:
    """Unpack a Langfuse-stored prompt back into a :class:`PromptTemplate`.

    Raises :exc:`ForgeError` when the prompt body isn't the JSON shape
    this store writes — most likely because the prompt was created
    through Langfuse directly rather than via Forge.
    """
    try:
        data: dict[str, Any] = json.loads(body)
    except json.JSONDecodeError as exc:
        msg = (
            f"Langfuse prompt {name!r}: prompt body is not the JSON format "
            "forge.prompts expects. Was this prompt created outside Forge?"
        )
        raise ForgeError(msg) from exc
    return PromptTemplate(
        name=name,
        stable_section=data.get("stable_section", ""),
        dynamic_section=data.get("dynamic_section", ""),
        stable_variables=tuple(config.get("stable_variables", [])),
        dynamic_variables=tuple(config.get("dynamic_variables", [])),
        description=config.get("description", ""),
        metadata=dict(config.get("metadata") or {}),
    )


class LangfusePromptStore(PromptStore):
    """:class:`PromptStore` backed by Langfuse's prompt management API.

    Pass an existing Langfuse client via ``client=`` (tests inject a
    mock here) or let the constructor build one from
    ``host``/``public_key``/``secret_key`` (which fall back to the
    standard ``LANGFUSE_*`` env vars when omitted).
    """

    def __init__(
        self,
        *,
        client: Any = None,
        host: str | None = None,
        public_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        if client is not None:
            self._client: Any = client
            return
        try:
            from langfuse import (  # pyright: ignore[reportMissingImports]
                Langfuse,  # pyright: ignore[reportUnknownVariableType]
            )
        except ImportError as exc:
            msg = (
                "LangfusePromptStore requires the [langfuse] extra. "
                "Install with `pip install ai-forge[langfuse]`."
            )
            raise ImportError(msg) from exc
        kwargs: dict[str, Any] = {}
        if host is not None:
            kwargs["host"] = host
        if public_key is not None:
            kwargs["public_key"] = public_key
        if secret_key is not None:
            kwargs["secret_key"] = secret_key
        self._client = Langfuse(**kwargs)  # pyright: ignore[reportUnknownVariableType]

    async def get(self, name: str, version: str | None = None) -> PromptTemplate:
        """Fetch a prompt, returning the latest version when ``version`` is None."""
        try:
            if version is None:
                prompt = self._client.get_prompt(name)
            else:
                prompt = self._client.get_prompt(name, version=int(version))
        except Exception as exc:
            label = f" version {version}" if version is not None else ""
            msg = f"Template {name!r}{label} not in Langfuse"
            raise PromptNotFoundError(msg, name=name, version=version) from exc
        body: str = prompt.prompt
        raw_config: Any = prompt.config or {}
        return _deserialize(name, body, dict(raw_config))

    async def put(self, template: PromptTemplate) -> str:
        """Create a new prompt version in Langfuse; return its version string."""
        body, config = _serialize(template)
        result = self._client.create_prompt(
            name=template.name,
            prompt=body,
            config=config,
        )
        return str(result.version)

    async def versions(self, name: str) -> list[str]:
        """Return every version of ``name``, newest-first.

        Langfuse's Python SDK doesn't expose a "list versions" endpoint
        directly, so we fetch the latest prompt and assume versions run
        consecutively from 1 to the latest — which holds for prompts
        created via this store. Prompts edited via the Langfuse UI may
        have gaps; for those the list reports the range but a ``get`` on
        a missing version still raises :class:`PromptNotFoundError`.
        """
        try:
            latest = self._client.get_prompt(name)
        except Exception as exc:
            msg = f"Template {name!r} not in Langfuse"
            raise PromptNotFoundError(msg, name=name) from exc
        latest_version = int(latest.version)
        return [str(v) for v in range(latest_version, 0, -1)]

    async def list_names(self) -> list[str]:
        """Best-effort listing of every distinct prompt name in the project.

        Walks Langfuse's prompt-list pagination. The Langfuse SDK
        surface for this varies between major versions; if the call
        fails we surface it as :class:`ForgeError` rather than
        swallowing.
        """
        try:
            page = self._client.api.prompts.list()
        except Exception as exc:
            msg = f"Langfuse list_names failed: {exc}"
            raise ForgeError(msg) from exc
        items: Any = getattr(page, "data", None) or []
        names: set[str] = {item.name for item in items}
        return sorted(names)

    async def delete(self, name: str, version: str | None = None) -> None:
        """Not supported via the Python SDK.

        Raises :exc:`NotImplementedError` with a pointer to the
        Langfuse UI. Soft-delete via labels is possible via the SDK; if
        you need it, drop down to ``self._client`` directly.
        """
        del name, version
        msg = (
            "LangfusePromptStore does not implement delete; use the "
            "Langfuse UI to remove prompts, or update labels via the "
            "underlying client to soft-delete."
        )
        raise NotImplementedError(msg)
