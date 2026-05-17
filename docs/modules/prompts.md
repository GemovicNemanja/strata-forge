# `forge.prompts` — typed prompt authoring with structural cache-awareness

`forge.prompts` is where prompts live: a Jinja2-driven template layer
backed by a sandboxed environment, a store-agnostic registry with
in-memory and Langfuse backends, and a renderer that compiles
`(template, variables)` into `list[forge.llm.AnyMessage]` plus the
cache hints `forge.llm.LLMClient` uses to populate provider-specific
prompt caching.

The module's organizing principle is the **structural split between a
stable prefix and a dynamic suffix** — see
[ADR 0007](../architecture/adr/0007-stable-prefix-dynamic-suffix-prompts.md).
Provider prompt caching (Anthropic `cache_control`, OpenAI's automatic
prefix cache, Gemini `CachedContent`) only kicks in when the cached
portion is byte-identical across requests. By forcing template authors
to declare what's stable and what's per-call, the module makes cache
hits the default behavior rather than something you remember to opt
into.

Module rules: [`src/forge/prompts/CLAUDE.md`](../../src/forge/prompts/CLAUDE.md).
Source: [`src/forge/prompts/`](../../src/forge/prompts/).

---

## Contents

- [Quickstart](#quickstart)
- [The stable/dynamic split](#the-stabledynamic-split)
- [Public API](#public-api)
  - [`PromptTemplate`](#prompttemplate)
  - [`render` + `RenderedPrompt`](#render--renderedprompt)
  - [`PromptRegistry` + `PromptStore`](#promptregistry--promptstore)
  - [`StableDynamicSplit` + `CacheHints`](#stabledynamicsplit--cachehints)
- [Variable validation](#variable-validation)
- [The Jinja2 sandbox](#the-jinja2-sandbox)
- [Cache integration with `forge.llm`](#cache-integration-with-forgellm)
- [Stores](#stores)
- [Errors](#errors)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```python
from forge.prompts import PromptTemplate, render

template = PromptTemplate(
    name="explainer",
    stable_section="You are {{ persona }}. Be concise.",
    dynamic_section="Explain: {{ topic }}",
    stable_variables=("persona",),
    dynamic_variables=("topic",),
)

rendered = render(template, {"persona": "a physicist", "topic": "entropy"})
# rendered.messages    → [SystemMessage(...), UserMessage(...)]
# rendered.cache_hints → CacheHints(cache_stable_prefix=..., ...)
# rendered.split       → StableDynamicSplit(stable_text=..., ...)
```

For end-to-end usage including the LLM client, see
[`examples/13_prompt_with_llm.py`](../../examples/13_prompt_with_llm.py).
For local-only rendering see
[`examples/11_prompt_template.py`](../../examples/11_prompt_template.py).
For version-tracked storage see
[`examples/12_prompt_registry.py`](../../examples/12_prompt_registry.py).

---

## The stable/dynamic split

Every `PromptTemplate` declares two sections:

- **`stable_section`** — the system prompt, role/persona, instructions,
  few-shot exemplars, retrieval-augmented context that's stable across
  a session. Variables are allowed (e.g. a persona drawn from config)
  but every variable must be declared in `stable_variables`. The
  rendered output of this section is byte-identical across requests
  that share the same stable-variable values, which is what provider
  prompt caching requires.
- **`dynamic_section`** — the per-call payload: the user's question,
  conversation tail, anything that varies. Variables must be declared
  in `dynamic_variables`.

The two declaration lists must be disjoint. A variable that's both
stable and dynamic makes the stable-prefix fingerprint ambiguous, so
the validator refuses templates that overlap. A variable that's
referenced in one section but declared in the other surfaces as an
"undeclared in this section" error — which is the right diagnosis,
because the author put it in the wrong list.

If a prompt genuinely has no stable portion (raw classification,
one-shot Q&A), use the shorthand:

```python
PromptTemplate.simple("classify", "Is '{{ x }}' positive or negative?", variables=("x",))
```

`simple` produces an empty stable section and forfeits prompt caching
by design — use it when there's nothing worth caching.

---

## Public API

### `PromptTemplate`

A frozen Pydantic model. Both Jinja sections are compiled at
construction, so syntax errors and disallowed filters surface
immediately rather than at render time.

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | Non-empty; the registry key. |
| `stable_section` | `str` | Jinja source. Empty by default. |
| `dynamic_section` | `str` | Jinja source. Empty by default. |
| `stable_variables` | `tuple[str, ...]` | Declared variables allowed in `stable_section`. |
| `dynamic_variables` | `tuple[str, ...]` | Declared variables allowed in `dynamic_section`. |
| `description` | `str` | Free-form, surfaced in Langfuse. |
| `metadata` | `dict[str, Any]` | Arbitrary JSON-shaped extras. |

`PromptTemplate.simple(name, body, *, variables=(), description="", metadata=None)`
is a classmethod shorthand that empties the stable section. The
template is otherwise a normal `PromptTemplate`.

### `render` + `RenderedPrompt`

```python
def render(
    template: PromptTemplate,
    variables: Mapping[str, Any] | None = None,
    *,
    model: str | None = None,
    min_cacheable_tokens: int = DEFAULT_MIN_CACHEABLE_TOKENS,
) -> RenderedPrompt
```

The pipeline:

1. Validate the template's declared variables match the body's
   references (`validate_template_variables`).
2. Confirm `variables` covers exactly the declared set — missing
   variables and extras both raise `ValidationError`.
3. Partition the call-site variables by their declared section and
   render the stable + dynamic sources through a fresh sandbox env.
   Partitioning is structural: the stable section physically cannot
   see dynamic variables (or vice versa).
4. Build a `StableDynamicSplit` (rendered stable + dynamic text plus
   a content-hash digest of the stable text).
5. Emit `CacheHints` for the target model.
6. Compose `[SystemMessage, UserMessage]`, skipping either side when
   the corresponding section rendered empty.

`RenderedPrompt` is a frozen dataclass with three fields:

- `messages: list[AnyMessage]` — ready to hand to `LLMClient`.
- `cache_hints: CacheHints` — provider-agnostic cache decision.
- `split: StableDynamicSplit` — surfaced so the eval runner can
  re-fingerprint without re-rendering when it sweeps the same template
  across models.

### `PromptRegistry` + `PromptStore`

A registry wraps a store and adds validation + a "stable section too
short" lint signal on every `put()`:

```python
from forge.prompts import PromptRegistry, InMemoryPromptStore

registry = PromptRegistry(InMemoryPromptStore())

version = await registry.put(template)   # validates first
latest  = await registry.get("name")     # latest version
specific = await registry.get("name", version)
all_versions = await registry.versions("name")
names = await registry.list_names()
await registry.delete("name", version)   # or `delete("name")` for all versions
```

`PromptStore` is the abstract base. Two implementations ship — see
[Stores](#stores). Versions are opaque strings; the store owns the
format.

### `StableDynamicSplit` + `CacheHints`

```python
@dataclass(frozen=True, slots=True)
class StableDynamicSplit:
    stable_text: str   # rendered stable section
    dynamic_text: str  # rendered dynamic section
    stable_digest: str # SHA-256 of stable_text — cache fingerprint

@dataclass(frozen=True, slots=True)
class CacheHints:
    cache_stable_prefix: bool   # should the prefix be flagged for cache?
    stable_digest: str          # for analytics / cross-call comparison
    stable_token_estimate: int  # for the linter signal
```

`emit_cache_hints(split, *, model=..., min_cacheable_tokens=...)` makes
the cache decision; `is_stable_too_short(split, ...)` exposes the same
threshold as a lint signal for the registry.

---

## Variable validation

Two checks run before any render happens:

1. `validate_template_variables(template)` — the template's *declared*
   variables match its body's *referenced* variables. Runs in fixed
   order so error messages are predictable: overlap check first, then
   stable-section references, then dynamic-section references.
2. The renderer compares the *call-site* `variables` mapping against
   the declared set: missing variables raise (the model would otherwise
   render with `StrictUndefined`-suppressed empty strings), and extras
   raise (a typo'd variable name is a programming error worth catching).

`extract_variables(source)` returns the set of *undeclared* references
in a Jinja source — Jinja's term for the variables the caller must
supply. `{% set %}` bindings and `{% for x in xs %}` loop variables
don't count.

---

## The Jinja2 sandbox

`create_sandboxed_environment()` returns a `SandboxedEnvironment` with:

- **No filesystem access.** `loader=None`; `include`/`import`
  extensions disabled.
- **No autoescape.** Prompts are not HTML.
- **`StrictUndefined`.** Referencing a variable that wasn't provided at
  render time raises immediately (instead of silently substituting
  empty strings).
- **A curated filter allowlist** (`SAFE_FILTERS`): string manipulation
  (`upper`, `lower`, `trim`, `replace`, `wordwrap`, `indent`, …),
  numeric/format (`abs`, `int`, `float`, `round`, `format`), collection
  helpers (`first`, `last`, `length`, `join`, `sort`, …), filtering
  (`select`, `reject`, `selectattr`, `rejectattr`, `map`), conditional
  (`default`, `d`), and `tojson`. Filters like `attr` that traverse
  arbitrary attributes are stripped.
- **No tests.** Jinja's test set (`x is something`) is disabled — tests
  can invoke methods on objects.

The sandbox blocks `__class__`, `__bases__`, and the rest of the
attribute-traversal exploit family via `SandboxedEnvironment`'s own
machinery. Module-level shared environment is used for syntax-check
compiles at template construction; renders happen in a fresh
environment each call so per-call state doesn't leak.

---

## Cache integration with `forge.llm`

The hints `render` returns are provider-agnostic. When `LLMClient`
consumes them (Phase 3+ wiring), it'll dispatch per provider:

- **Anthropic / Bedrock (Claude with explicit `cache_control`):** when
  `cache_stable_prefix=True`, insert
  `cache_control: {"type": "ephemeral"}` on the last content block of
  the stable portion.
- **OpenAI / Azure / openai_compat:** the flag is informational —
  caching is automatic on prefixes ≥ ~1024 tokens. The message-
  composition layer keeps the stable content contiguous at the top so
  the auto-cache can match the longest possible prefix.
- **Vertex Gemini:** `cache_stable_prefix=True` is a hint that the
  stable portion is a good candidate for a `CachedContent` resource;
  resource lifecycle is out of scope for this module.

The `stable_digest` field is exposed so the eval runner and the
diagnostic dump can correlate cache hits across calls.

---

## Stores

### `InMemoryPromptStore`

Dict-backed; monotonic per-name integer versions (`"1"`, `"2"`, …); the
latest version is whatever was stored most recently. Suitable for
tests, notebooks, and ad-hoc scripts. Not safe across processes — use
the Langfuse store for that.

### `LangfusePromptStore`

Backed by Langfuse's prompt management API. Requires the `[langfuse]`
extra (`pip install ai-forge[langfuse]`). The constructor lazy-imports
`langfuse`; importing `forge.prompts` without the extra is safe.

`PromptTemplate` serializes into Langfuse's prompt model by packing
both Jinja sections into a JSON blob in the `prompt` field and the
typed metadata (declared variables, description, `metadata`, plus a
`forge_template_v1` flag) into the `config` dict. The flag lets
external readers identify prompts written by Forge.

`delete` raises `NotImplementedError` — the Langfuse Python SDK
doesn't ship a deletion endpoint. Use the Langfuse UI to remove
prompts, or drop down to the underlying client for soft-delete via
labels. `versions` walks `1..latest` (an assumption that holds for
prompts created through this store; UI edits may produce gaps).

---

## Errors

Every exception raised from `forge.prompts` is a `ForgeError` subclass.

| Exception | When it fires |
|---|---|
| `PromptError` | Base class — catch-all for the module. |
| `PromptValidationError` | Invalid Jinja syntax, disallowed filter, undeclared variable reference, stable/dynamic overlap. |
| `PromptNotFoundError` | `PromptStore.get` against an unknown name or version. Carries `name` and `version` attributes. |
| `ValidationError` (from `forge.core.errors`) | Render-time variable mismatch — missing required or extra unexpected. |
| `ForgeError` | Misc failures from the Langfuse store (e.g., malformed prompt body, list call failed). |

---

## Troubleshooting

**`PromptValidationError: ... dynamic_section references ['x'] but those names are not in dynamic_variables`.**
Either declare `x` in `dynamic_variables` (or `stable_variables` if it
belongs to the stable section), or remove the reference. A common
cause is renaming a variable in the body without updating the
declaration tuple.

**`PromptValidationError: ... variables ['x'] appear in both stable_variables and dynamic_variables`.**
A variable must live in exactly one section. The stable-prefix cache
fingerprint can't depend on something that varies per call.

**`ValidationError: Template 'x' missing variables at render time: ['y']`.**
The call-site `variables` mapping doesn't cover every declared
variable. Inspect the template's `stable_variables` and
`dynamic_variables`.

**`ValidationError: ... variables ['typo'] passed but not declared on the template`.**
An extra variable in the render call. Most often a typo in the
variable name — the renderer surfaces this loudly because silently
ignoring it would mean the actual variable's render is broken (and
likely renders an empty string because of `StrictUndefined`).

**Warning: `prompt.stable_section_too_short`.**
The registry's lint signal: the template declared a stable section
that won't trigger provider prompt caching. Either expand it (a
multi-sentence persona, exemplars) or collapse to
`PromptTemplate.simple` to opt out of caching explicitly.

**`ImportError: LangfusePromptStore requires the [langfuse] extra`.**
Install with `pip install ai-forge[langfuse]` (or `uv sync --extra
langfuse`) before instantiating `LangfusePromptStore`.

**Cache hits aren't happening even with a long stable section.**
Verify the stable section is byte-identical across calls — render two
calls and compare `rendered.split.stable_text`. If they differ, a
stable variable is changing per call (it shouldn't be — that's why
`stable_variables` is structurally separate). The `stable_digest` is
the canonical comparison.
