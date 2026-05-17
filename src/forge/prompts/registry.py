"""Version-aware prompt registry over a pluggable :class:`PromptStore`.

The store is store-agnostic: an in-process backend ships in
:mod:`forge.prompts.stores.memory`, a Langfuse-backed one ships in
:mod:`forge.prompts.stores.langfuse`. The :class:`PromptRegistry` layer
on top adds two things stores shouldn't have to repeat:

- **Declaration validation.** Every :meth:`PromptRegistry.put` runs
  :func:`forge.prompts.variables.validate_template_variables` before
  the template reaches the store; invalid templates can't be
  registered.
- **Lint signal.** When the template declares a stable section but
  its length falls below
  :data:`forge.prompts.cache_aware.DEFAULT_MIN_CACHEABLE_TOKENS`, the
  registry emits a warning via the structlog logger. The author paid
  the stable/dynamic boilerplate cost for no caching benefit; the
  message is informational, not blocking.

Version scheme is up to the store. The in-memory store assigns
monotonic ``"1"``, ``"2"``, ``"3"``... strings; the Langfuse store
uses whatever Langfuse returns. The registry doesn't care about the
format — it treats versions as opaque strings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from forge.core.errors import ForgeError
from forge.core.logging import get_logger
from forge.prompts.cache_aware import StableDynamicSplit, is_stable_too_short
from forge.prompts.variables import validate_template_variables

if TYPE_CHECKING:
    from forge.prompts.template import PromptTemplate

__all__ = [
    "PromptNotFoundError",
    "PromptRegistry",
    "PromptStore",
]


class PromptNotFoundError(ForgeError):
    """A requested prompt name or version isn't present in the store.

    ``name`` and ``version`` are attached for programmatic inspection;
    the version is ``None`` when the lookup was for the latest version
    and the name itself was missing.
    """

    def __init__(
        self,
        message: str,
        *,
        name: str,
        version: str | None = None,
    ) -> None:
        super().__init__(message)
        self.name = name
        self.version = version


class PromptStore(ABC):
    """Abstract backend for prompt persistence.

    Implementations own the version scheme — the registry treats versions
    as opaque strings. A ``get`` with ``version=None`` must return the
    latest stored version; a ``delete`` with ``version=None`` must remove
    every version for ``name``.
    """

    @abstractmethod
    async def get(self, name: str, version: str | None = None) -> PromptTemplate:
        """Return a stored template by name and version.

        Raises:
            PromptNotFoundError: When ``name`` is unknown, or
                ``version`` is specified but doesn't exist for ``name``.
        """

    @abstractmethod
    async def put(self, template: PromptTemplate) -> str:
        """Store ``template`` as a new version of ``template.name``.

        Returns:
            The version string the store assigned. The format is
            backend-specific.
        """

    @abstractmethod
    async def versions(self, name: str) -> list[str]:
        """Return every version known for ``name``, newest-first.

        Raises:
            PromptNotFoundError: When ``name`` is not in the store.
        """

    @abstractmethod
    async def list_names(self) -> list[str]:
        """Return every distinct template name in the store, sorted."""

    @abstractmethod
    async def delete(self, name: str, version: str | None = None) -> None:
        """Delete a specific version, or every version when ``version`` is None.

        A delete against an unknown name is a no-op (it's already absent);
        a delete against an unknown version raises
        :class:`PromptNotFoundError` so callers don't silently swallow a
        typo'd version.
        """


class PromptRegistry:
    """User-facing wrapper over a :class:`PromptStore`.

    Validates every template before it reaches the store, emits a
    "stable section too short" lint warning when applicable, and
    otherwise delegates straight through. Construct with the store
    implementation of your choice::

        from forge.prompts import PromptRegistry
        from forge.prompts.stores.memory import InMemoryPromptStore

        registry = PromptRegistry(InMemoryPromptStore())
    """

    def __init__(self, store: PromptStore) -> None:
        self._store = store
        self._logger = get_logger("forge.prompts.registry")

    async def get(self, name: str, version: str | None = None) -> PromptTemplate:
        """Fetch a template; see :meth:`PromptStore.get`."""
        return await self._store.get(name, version)

    async def put(self, template: PromptTemplate) -> str:
        """Validate and store ``template``; return the new version string.

        Raises:
            PromptValidationError: When the template's declared variables
                don't match its body references.
        """
        validate_template_variables(template)
        self._lint_stable_section(template)
        return await self._store.put(template)

    async def versions(self, name: str) -> list[str]:
        """List every version of ``name``, newest-first."""
        return await self._store.versions(name)

    async def list_names(self) -> list[str]:
        """List every template name in the store."""
        return await self._store.list_names()

    async def delete(self, name: str, version: str | None = None) -> None:
        """Delete a version (or all versions) of ``name``."""
        await self._store.delete(name, version)

    # --- internal helpers --------------------------------------------------

    def _lint_stable_section(self, template: PromptTemplate) -> None:
        """Warn when a declared stable section is too short to benefit from cache.

        Skipped entirely for templates with an empty ``stable_section``
        (the :meth:`PromptTemplate.simple` shorthand sets it empty by
        design — no point complaining that "no stable section" is short).

        The check uses the unrendered Jinja source as a length proxy.
        It overstates slightly when the source has heavy templating
        syntax that collapses on render, but for a lint signal that's
        the right trade-off — cheap to compute, no rendering required.
        """
        if not template.stable_section:
            return
        split = StableDynamicSplit.build(
            stable_text=template.stable_section,
            dynamic_text=template.dynamic_section,
        )
        if is_stable_too_short(split):
            self._logger.warning(
                "prompt.stable_section_too_short",
                template_name=template.name,
                stable_chars=len(template.stable_section),
                note=(
                    "stable section is below the cacheable threshold; either "
                    "expand it or collapse to PromptTemplate.simple"
                ),
            )
