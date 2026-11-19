"""Prompt authoring: sandboxed Jinja2 templates with a stable/dynamic split, and compiled builds.

:mod:`strata_forge.prompts.compiled` is the authoritative compile + render for versioned prompts
(an ordered composition of parts with typed variables, compiled to a hashed build).
"""

from strata_forge.prompts.cache_aware import (
    DEFAULT_MIN_CACHEABLE_TOKENS,
    CacheHints,
    StableDynamicSplit,
    emit_cache_hints,
    is_stable_too_short,
)
from strata_forge.prompts.compiled import (
    BUILD_FORMAT,
    COMPILER_VERSION,
    Binding,
    CompileIssue,
    CompileResult,
    PromptBuild,
    PromptBuildError,
    PromptPart,
    PromptRenderError,
    VariableSpec,
    bind_row,
    compile_prompt,
    parse_part,
    render_braces,
    render_build,
    value_text,
)
from strata_forge.prompts.registry import (
    PromptNotFoundError,
    PromptRegistry,
    PromptStore,
)
from strata_forge.prompts.rendering import RenderedPrompt, render
from strata_forge.prompts.stores.langfuse import LangfusePromptStore
from strata_forge.prompts.stores.memory import InMemoryPromptStore
from strata_forge.prompts.template import (
    SAFE_FILTERS,
    PromptError,
    PromptTemplate,
    PromptValidationError,
    create_sandboxed_environment,
)
from strata_forge.prompts.variables import (
    extract_variables,
    validate_template_variables,
)

__all__ = [
    "BUILD_FORMAT",
    "COMPILER_VERSION",
    "DEFAULT_MIN_CACHEABLE_TOKENS",
    "SAFE_FILTERS",
    "Binding",
    "CacheHints",
    "CompileIssue",
    "CompileResult",
    "InMemoryPromptStore",
    "LangfusePromptStore",
    "PromptBuild",
    "PromptBuildError",
    "PromptError",
    "PromptNotFoundError",
    "PromptPart",
    "PromptRegistry",
    "PromptRenderError",
    "PromptStore",
    "PromptTemplate",
    "PromptValidationError",
    "RenderedPrompt",
    "StableDynamicSplit",
    "VariableSpec",
    "bind_row",
    "compile_prompt",
    "create_sandboxed_environment",
    "emit_cache_hints",
    "extract_variables",
    "is_stable_too_short",
    "parse_part",
    "render",
    "render_braces",
    "render_build",
    "validate_template_variables",
    "value_text",
]
