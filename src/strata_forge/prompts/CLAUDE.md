# Agent rules — strata_forge.prompts

`strata_forge.prompts` is the typed prompt authoring layer. Templates are
Jinja2-driven with a curated safe filter set; prompts are
**structurally split** into a stable prefix and a dynamic suffix so
provider prompt caching works by default; a store-agnostic registry
backs in-process and Langfuse-hosted catalogs.

## Purpose

The Jinja2 authoring/storage API renders to `list[strata_forge.llm.AnyMessage]`.
The independent modular API returns immutable chat/artifact snapshots with byte attribution.
The module owns:

- Jinja2 templating with no I/O extensions and an allowlist of safe
  filters.
- The stable-prefix / dynamic-suffix split required by
  [ADR 0007](../../../docs/architecture/adr/0007-stable-prefix-dynamic-suffix-prompts.md),
  including per-provider cache-marker emission.
- A `PromptRegistry` with pluggable backends (in-memory, Langfuse).
- Validation: every variable referenced in a template body must be
  declared on the template; stable-section variables must not appear
  in dynamic positions or vice versa.
- Rendering to `list[strata_forge.llm.AnyMessage]` with the cache hints the
  LLM client uses to populate provider-specific caching mechanics.

## Boundaries

- **Owns:** `template.py`, `cache_aware.py`, `variables.py`,
  `registry.py`, `stores/memory.py`, `stores/langfuse.py`,
  `rendering.py`.
- **Imports from inside `forge`:** `strata_forge.core` (errors, logging,
  reproducibility helpers) and `strata_forge.llm` (message types and the
  `AnyMessage` union — used as the rendering target).
- **Does NOT:** call LLMs, store datasets, run evals, build agents.
  Rendering a prompt is a pure transformation from
  `(PromptTemplate, variables)` to `list[AnyMessage]`.

## Modular composition boundary

`composition.py`, `composition_types.py`, `composition_errors.py` and `portable.py` own the
independent explicit modular dialect under [ADR 0023](../../../docs/architecture/adr/0023-portable-modular-prompt-composition.md).
The Jinja2 API retains its behavior and signatures. `CompositionRegistry`, `compile_surface`
and `render_build` are async. Canonicalization, schema validation and immutable snapshot
inspection are synchronous pure helpers; no synchronous I/O or runtime wrapper is admitted.

- Text is literal; only explicit variable nodes insert scalar data once. Never reinterpret
  braces or run expressions, filters, imports or callable resolution in this dialect.
- Pin exact coordinates. Registry snapshots never replace a coordinate with different content,
  and builds never consult moving channels or mutate when registry entries are added.
- Preserve ordered empty/repeated leaves, declared chat roles and artifact purpose/descriptor
  boundaries. Constraints are explicit facts; duplicate prose is advisory and is never removed.
- Hash only the portable profile and versioned domain-bearing values. Reject duplicate JSON
  keys, nonportable numbers/objects, surrogates and exceeded bounds; no normalization or truncation.
- Source offsets are half-open UTF-8 bytes. Validate exact target shape, full ordered byte
  coverage, scalar boundaries and compiler-owned separators when accepting render snapshots.
- Imported hashes and validated schemas are not compiler attestations, access grants or proof
  of provider use. Server consumers must verify expected immutable IDs and permissions.
- Keep Python/Node fixtures synchronized, including adverse metadata and Unicode. Operational
  import fixtures require explicit source authorization; research/verifier prompts are excluded.

Reference: [`docs/modules/prompt-composition.md`](../../../docs/modules/prompt-composition.md).

## Public API

The module's `__init__.py` re-exports the supported surface.
Treat anything not in `__init__.py` as internal.

- `PromptTemplate` — the authored unit. Carries a stable section,
  a dynamic section, declared variables, and metadata.
- `PromptRegistry` — store-agnostic accessor; `get(name, version=...)`,
  `put(template)`, `versions(name)`, `list()`.
- `PromptStore` — abstract base; implementations live in `stores/`.
- `InMemoryStore` and `LangfuseStore` — the two shipped backends.
- `render(template, variables)` — produces `list[AnyMessage]` plus the
  cache hints `strata_forge.llm` consumes.
- `PromptError`, `PromptValidationError` — errors raised from this
  module, both `ForgeError` subclasses.
- `ModuleRevision`, `SurfaceRevision`, `TextPart`, `VariablePart`, `ModulePart`, `VariableDecl`,
  `PromptConstraint`, `SourceProvenance` and `PortableDocument` — immutable modular authoring data.
- `ChatTarget`, `ChatMessageParts`, `ArtifactTarget` — distinct declared output boundaries.
- `CompositionRegistry`, `compile_surface`, `render_build` — exact snapshot/compile/render APIs.
- `PromptBuild`, `PromptRender`, `DependencyLock`, `SourceSpan`, `PromptOrigin`,
  `CompositionDiagnostic`, `PromptCompositionError` — immutable content, attribution and diagnostics.
- `CompositionCapabilities`, `composition_capabilities` — fixed versioned bounds.
- `definition_from_wire`, `definition_identity`, `canonical_bytes`, `parse_portable_json`,
  `portable_sha256` — pure closed-schema and portable-byte helpers.

## Internal patterns

- **Jinja2 environment is locked down.** No
  `loader=FileSystemLoader(...)`, no `include`/`import` extension, no
  `do`/`with` extension. Autoescape is off (prompts are not HTML).
  Filters are an explicit allowlist starting with the Jinja2 safe set
  minus anything that touches files or modules.
- **Stable-section variables are declared separately from dynamic
  ones.** The template constructor takes
  `stable_variables: list[str]` and `dynamic_variables: list[str]`;
  validation refuses overlap and refuses references in the wrong
  section.
- **Render output is a Pydantic-validated message list.** Callers
  never get back a raw string — they get
  `list[AnyMessage]` ready to hand to `LLMClient.complete`.
- **Cache hints are per-message metadata, not separate plumbing.** The
  render function returns the messages with an associated
  `CacheHints` value the LLM client consumes when constructing the
  provider payload. Providers without explicit cache markers (OpenAI)
  receive the messages as-is; the hint is informational.
- **Langfuse access is lazy.** `LangfuseStore` lazy-imports
  `langfuse` inside its constructor. The `[langfuse]` extra carries
  the dep.

## Test expectations

- Unit tests under `tests/unit/prompts/`, one file per source module.
- Coverage target: ≥ 90 % line.
- Jinja2 sandbox tests: forbidden filters / extensions raise
  `PromptError` at compile time.
- Cache-marker tests assert byte-exact stable prefixes across renders
  with different dynamic inputs.
- Registry tests cover both backends; the Langfuse one uses a mocked
  client.
- Hypothesis property tests on the variable validator (any subset of
  declared variables passed at render time, fail if a referenced
  variable is missing).

## Gotchas

- **Jinja2 sandboxing is opt-in.** Using the default `Environment`
  silently allows file includes. We use
  `jinja2.sandbox.SandboxedEnvironment` plus the filter allowlist —
  removing either is a regression that won't fail any test by itself
  unless you specifically exercise the unsafe filter, so a code review
  must catch it.
- **`PromptTemplate.simple(...)` exists for genuinely
  stable-less prompts** (one-shot classification, raw Q&A). Don't reach
  for it as the default — it forfeits prompt caching, which the
  validator will warn about when the stable section is short.
- **Cache hints depend on variable values, not just the template.**
  Two renders of the same template can produce different cache
  configurations if a stable-section variable changes. The validator
  catches "this variable is stable in name but varies per call" by
  refusing variables marked stable from appearing in `variables` passed
  to `render` unless they match the registered baseline (a per-template
  cache key).
- **Do not log full rendered prompts at `INFO`.** They contain user
  data. Use `DEBUG`; the NDJSON diagnostic dump in `strata_forge.llm` is the
  appropriate channel.

## When to update this file

- Adding a new public class / function to `__init__.py`.
- Adding a new store backend.
- Changing the Jinja2 sandbox configuration (filter allowlist,
  extension set).
- Changing the cache-hint emission rules per provider.
- Changing the validation rules for stable vs dynamic variables.
