"""Sandboxed Jinja2 environment and the :class:`PromptTemplate` data shape.

Templates are authored by declaring two sections — ``stable_section``
and ``dynamic_section``. The split is required by
:doc:`ADR 0007 <../../../docs/architecture/adr/0007-stable-prefix-dynamic-suffix-prompts>`
because every provider's prompt cache works only on a byte-identical
prefix, and a template that mingles variables throughout produces a
fresh prefix per call.

This module owns the data shape and the locked-down Jinja2 environment;
adjacent modules in :mod:`forge.prompts` handle the orthogonal concerns:

- Per-provider cache markers — :mod:`forge.prompts.cache_aware`
- Variable extraction and validation — :mod:`forge.prompts.variables`
- Rendering to :class:`forge.llm.AnyMessage` — :mod:`forge.prompts.rendering`

The sandbox itself blocks attribute exploits (``__class__`` games, etc.)
via :class:`jinja2.sandbox.SandboxedEnvironment`, refuses to load
templates from disk (``loader=None``), keeps autoescape off (prompts
are not HTML), strips every filter not on :data:`SAFE_FILTERS`, and
disables Jinja's test set entirely (tests can invoke methods on
objects, which we don't want).
"""

from __future__ import annotations

from typing import Any

import jinja2
from jinja2.exceptions import TemplateSyntaxError as JinjaTemplateSyntaxError
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge.core.errors import ForgeError

__all__ = [
    "SAFE_FILTERS",
    "PromptError",
    "PromptTemplate",
    "PromptValidationError",
    "create_sandboxed_environment",
]


class PromptError(ForgeError):
    """Base class for errors raised from :mod:`forge.prompts`."""


class PromptValidationError(PromptError):
    """A prompt failed validation.

    Raised for Jinja syntax errors, references to disallowed filters,
    undeclared variables, and stable/dynamic boundary violations.
    """


# Curated allowlist of Jinja2 filters. Every filter that reaches a
# ``PromptTemplate`` must be in this set. The starting list is Jinja2's
# stdlib filter set minus anything that touches I/O, executes code, or
# leaks module attributes.
SAFE_FILTERS: frozenset[str] = frozenset(
    {
        # String manipulation
        "upper",
        "lower",
        "title",
        "capitalize",
        "trim",
        "replace",
        "truncate",
        "wordwrap",
        "indent",
        "center",
        "string",
        # Numeric / format
        "abs",
        "int",
        "float",
        "round",
        "format",
        # Collection helpers
        "first",
        "last",
        "length",
        "count",
        "join",
        "list",
        "reverse",
        "sort",
        "unique",
        "min",
        "max",
        "sum",
        "items",
        "dictsort",
        "groupby",
        # Filtering / mapping
        "select",
        "reject",
        "selectattr",
        "rejectattr",
        "map",
        # Conditional
        "default",
        "d",
        # JSON-ish
        "tojson",
    }
)


def create_sandboxed_environment() -> SandboxedEnvironment:
    """Build the Forge-flavored Jinja2 environment.

    Restrictions:

    - :class:`SandboxedEnvironment` blocks attribute exploits (no
      ``__class__`` / ``__bases__`` / module access on rendered values).
    - ``loader=None`` — no template loading from disk.
    - ``autoescape=False`` — prompts are not HTML.
    - ``undefined=StrictUndefined`` — referencing an undeclared variable
      raises at render time rather than silently substituting empty
      strings (a known prompt-quality footgun).
    - Filters are pruned to :data:`SAFE_FILTERS`.
    - The Jinja test set (``is something``) is disabled entirely.
    """
    env = SandboxedEnvironment(
        autoescape=False,
        loader=None,
        undefined=jinja2.StrictUndefined,
    )
    env.filters = {name: fn for name, fn in env.filters.items() if name in SAFE_FILTERS}
    env.tests = {}
    return env


# Module-level shared environment for compile-time syntax checks.
# Renders happen against fresh environments in ``rendering.py`` so this
# instance is never mutated after construction.
_ENV: SandboxedEnvironment = create_sandboxed_environment()


def _compile(source: str, *, location: str) -> None:
    """Compile a Jinja source string under the sandbox; raise on failure.

    Only the syntax check matters here; the compiled template is
    discarded. ``rendering.py`` re-compiles per render so each
    invocation gets a fresh, isolated AST.
    """
    try:
        _ENV.from_string(source)
    except JinjaTemplateSyntaxError as exc:
        msg = (
            f"Invalid Jinja syntax in {location}: {exc.message} "
            f"(line {exc.lineno})"
        )
        raise PromptValidationError(msg) from exc


class PromptTemplate(BaseModel):
    """A typed prompt template with a structural stable/dynamic split.

    The **stable section** is content identical across many requests —
    the system prompt, role/persona, instructions, few-shot exemplars,
    long-lived retrieval context. The **dynamic section** is per-call
    input: the user's query, conversation tail, anything that varies.

    Variables permitted in each section are declared up-front via
    ``stable_variables`` and ``dynamic_variables``; the registry and the
    renderer enforce the contract. A variable referenced from the
    "wrong" section, or referenced without being declared, raises
    :class:`PromptValidationError`.

    Both Jinja sections are compiled at construction so syntax errors
    surface immediately rather than at render time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    stable_section: str = ""
    dynamic_section: str = ""
    stable_variables: tuple[str, ...] = ()
    dynamic_variables: tuple[str, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = Field(default={})

    @model_validator(mode="after")
    def _compile_sections(self) -> PromptTemplate:
        _compile(self.stable_section, location=f"{self.name}.stable_section")
        _compile(self.dynamic_section, location=f"{self.name}.dynamic_section")
        return self

    @classmethod
    def simple(
        cls,
        name: str,
        body: str,
        *,
        variables: tuple[str, ...] = (),
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PromptTemplate:
        """Shorthand for genuinely stable-less prompts (forfeits caching).

        Treats the entire body as the dynamic section. Use only for
        truly per-request prompts (one-shot classification, raw Q&A);
        for anything with a stable system prompt or exemplars, declare
        ``stable_section`` and ``dynamic_section`` explicitly so
        provider prompt caching can kick in.
        """
        return cls(
            name=name,
            stable_section="",
            dynamic_section=body,
            stable_variables=(),
            dynamic_variables=variables,
            description=description,
            metadata=metadata if metadata is not None else {},
        )
