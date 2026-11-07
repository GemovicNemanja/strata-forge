"""Sandboxed Jinja2 templates and immutable portable modular prompt composition."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata_forge.prompts.consumption import (
        ConsumedBuild as ConsumedBuild,
    )
    from strata_forge.prompts.consumption import (
        LocalBuildReference as LocalBuildReference,
    )
    from strata_forge.prompts.consumption import (
        LocalBuildSource as LocalBuildSource,
    )
    from strata_forge.prompts.consumption import (
        consume_build as consume_build,
    )
    from strata_forge.prompts.consumption import (
        render_consumed as render_consumed,
    )

from strata_forge.prompts.cache_aware import (
    DEFAULT_MIN_CACHEABLE_TOKENS,
    CacheHints,
    StableDynamicSplit,
    emit_cache_hints,
    is_stable_too_short,
)
from strata_forge.prompts.composition import (
    CompositionCapabilities,
    CompositionRegistry,
    DependencyLock,
    PromptBuild,
    PromptRender,
    SourceSpan,
    compile_surface,
    composition_capabilities,
    definition_identity,
    render_build,
)
from strata_forge.prompts.composition_errors import (
    CompositionDiagnostic,
    PromptCompositionError,
    PromptOrigin,
)
from strata_forge.prompts.composition_types import (
    ArtifactTarget,
    ChatMessageParts,
    ChatTarget,
    ModulePart,
    ModuleRevision,
    PortableDocument,
    PromptConstraint,
    SourceProvenance,
    SurfaceRevision,
    TextPart,
    VariableDecl,
    VariablePart,
    definition_from_wire,
)
from strata_forge.prompts.portable import (
    canonical_bytes,
    parse_portable_json,
    portable_sha256,
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
    "DEFAULT_MIN_CACHEABLE_TOKENS",
    "SAFE_FILTERS",
    "ArtifactTarget",
    "CacheHints",
    "ChatMessageParts",
    "ChatTarget",
    "CompositionCapabilities",
    "CompositionDiagnostic",
    "CompositionRegistry",
    "ConsumedBuild",
    "DependencyLock",
    "InMemoryPromptStore",
    "LangfusePromptStore",
    "LocalBuildReference",
    "LocalBuildSource",
    "ModulePart",
    "ModuleRevision",
    "PortableDocument",
    "PromptBuild",
    "PromptCompositionError",
    "PromptConstraint",
    "PromptError",
    "PromptNotFoundError",
    "PromptOrigin",
    "PromptRegistry",
    "PromptRender",
    "PromptStore",
    "PromptTemplate",
    "PromptValidationError",
    "RenderedPrompt",
    "SourceProvenance",
    "SourceSpan",
    "StableDynamicSplit",
    "SurfaceRevision",
    "TextPart",
    "VariableDecl",
    "VariablePart",
    "canonical_bytes",
    "compile_surface",
    "composition_capabilities",
    "consume_build",
    "create_sandboxed_environment",
    "definition_from_wire",
    "definition_identity",
    "emit_cache_hints",
    "extract_variables",
    "is_stable_too_short",
    "parse_portable_json",
    "portable_sha256",
    "render",
    "render_build",
    "render_consumed",
    "validate_template_variables",
]


def __getattr__(name: str) -> object:
    if name in {
        "ConsumedBuild",
        "LocalBuildReference",
        "LocalBuildSource",
        "consume_build",
        "render_consumed",
    }:
        import strata_forge.prompts.consumption as consumption

        return getattr(consumption, name)
    raise AttributeError(name)
