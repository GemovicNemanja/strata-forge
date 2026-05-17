"""Variable extraction and declaration validation for prompt templates.

A :class:`forge.prompts.PromptTemplate` declares its variables up front
via ``stable_variables`` and ``dynamic_variables``. This module enforces
the contract:

- Every variable referenced in ``stable_section`` must appear in
  ``stable_variables``.
- Every variable referenced in ``dynamic_section`` must appear in
  ``dynamic_variables``.
- The two declaration lists must not overlap — a variable is *either*
  stable across calls *or* per-call dynamic; doing both makes the
  stable-prefix cache fingerprint ambiguous.

A variable referenced only in the dynamic section but mistakenly added
to ``stable_variables`` (or vice versa) shows up as an "undeclared in
this section" error — which is the right diagnostic because the author
declared it in the wrong place.

Validation is NOT called from ``PromptTemplate.__init__`` to avoid a
circular import between ``template.py`` and this module. The registry
calls :func:`validate_template_variables` at ``put()`` time so every
registered template is verified; callers who construct templates by
hand should call the validator themselves before relying on the
template.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jinja2 import meta

from forge.prompts.template import (
    PromptValidationError,
    create_sandboxed_environment,
)

if TYPE_CHECKING:
    from forge.prompts.template import PromptTemplate

__all__ = [
    "extract_variables",
    "validate_template_variables",
]


def extract_variables(source: str) -> frozenset[str]:
    """Return the set of variable names referenced in a Jinja source string.

    Uses :func:`jinja2.meta.find_undeclared_variables` on the AST parsed
    by the locked-down sandbox environment. "Undeclared" here is Jinja's
    own term for variables that don't have a corresponding ``{% set %}``
    or loop binding inside the template — i.e. those the caller must
    provide at render time.

    Empty templates and templates with no variable references return an
    empty frozenset; parsing failures propagate as
    :class:`jinja2.TemplateSyntaxError` (callers that want a Forge-typed
    error should pre-validate via the :class:`PromptTemplate` model
    validator, which raises :class:`PromptValidationError`).
    """
    env = create_sandboxed_environment()
    ast = env.parse(source)
    return frozenset(meta.find_undeclared_variables(ast))


def validate_template_variables(template: PromptTemplate) -> None:
    """Verify a template's declared variables match its body references.

    Raises :class:`PromptValidationError` on the first violation found.
    The order of checks is fixed (overlap → stable refs → dynamic refs)
    so error messages are predictable.

    Args:
        template: The template to validate.
    """
    overlap = set(template.stable_variables) & set(template.dynamic_variables)
    if overlap:
        msg = (
            f"Template {template.name!r}: variables {sorted(overlap)!r} appear in "
            "both stable_variables and dynamic_variables; choose one"
        )
        raise PromptValidationError(msg)

    stable_refs = extract_variables(template.stable_section)
    declared_stable = set(template.stable_variables)
    undeclared_stable = stable_refs - declared_stable
    if undeclared_stable:
        msg = (
            f"Template {template.name!r}: stable_section references "
            f"{sorted(undeclared_stable)!r} but those names are not in "
            f"stable_variables ({list(template.stable_variables)!r})"
        )
        raise PromptValidationError(msg)

    dynamic_refs = extract_variables(template.dynamic_section)
    declared_dynamic = set(template.dynamic_variables)
    undeclared_dynamic = dynamic_refs - declared_dynamic
    if undeclared_dynamic:
        msg = (
            f"Template {template.name!r}: dynamic_section references "
            f"{sorted(undeclared_dynamic)!r} but those names are not in "
            f"dynamic_variables ({list(template.dynamic_variables)!r})"
        )
        raise PromptValidationError(msg)
