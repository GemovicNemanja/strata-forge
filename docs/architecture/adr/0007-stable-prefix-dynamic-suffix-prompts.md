# ADR 0007 — Prompts are structurally split into stable prefix and dynamic suffix

**Status:** Accepted
**Date:** Initial scaffolding for `strata_forge.prompts`
**Supersedes:** —
**Superseded by:** —

## Context

Every supported provider exposes some form of prompt caching:

- **Anthropic** — explicit `cache_control: {"type": "ephemeral"}` markers
  on content blocks. The provider hashes the prefix up to the marker
  and stores its KV state for ~5 minutes; subsequent requests with the
  same prefix pay cache-read rates (≈10× cheaper) and skip prefill
  cost.
- **OpenAI** — automatic on prompts above ~1024 tokens. Caches the
  longest matching prefix that the client has sent recently; no
  markers required. Cache-read tokens bill at 50% of input rate.
- **Gemini** — explicit `CachedContent` resource: you create a cache
  ahead of time and reference its name on each call.

All three rely on the same invariant: **the cached portion must be
byte-identical across requests, and the per-request data must come
after it.** A prompt written as a monolithic Jinja template with
variables interpolated throughout produces different rendered text for
every request — the prefix shifts a few characters and the cache
misses every time.

In Phase 1 the registry's `capabilities.prompt_caching` flag tells the
client whether the model supports it, but Phase 1 has no opinion about
*how* to author prompts so they're cacheable. That's a Phase-2
decision: either we let prompt authors solve cache-friendliness
themselves (and watch most of them not), or we make the split
structural.

## Decision

`strata_forge.prompts` requires every `PromptTemplate` to declare two
sections:

- `stable` — the system prompt, role/persona, instructions, few-shot
  examples, retrieval-augmented context that's stable across a session.
  Variables are allowed (e.g. an A/B-flag pulled from config), but any
  variable referenced here MUST be marked as a stable variable, and the
  registry rejects prompts where a stable-section variable name
  collides with a dynamic-section variable name.

- `dynamic` — the user's per-request input, conversation history, and
  anything else that varies per call.

The split is a structural property of the template, not a convention.
Rendering produces a `list[AnyMessage]` plus per-message cache markers
the LLM client uses to populate provider-specific cache hints:

- For Anthropic / Bedrock / Claude-on-Vertex: insert
  `cache_control` on the last stable content block.
- For OpenAI / Azure / openai_compat: no marker needed — caching is
  automatic — but we order content so the stable portion is
  contiguous at the top, maximizing the prefix the provider's
  auto-cache can match.
- For Gemini: emit a hint that `strata_forge.llm` can use to construct a
  `CachedContent` resource (the actual resource lifecycle is a Phase-3
  concern when the request volume justifies it).

The `strata_forge.prompts.cache_aware` module owns the split, the
per-provider marker emission, and the validation that stable
variables don't leak into dynamic positions (or vice versa).

## Consequences

**Positive**

- Cache hits happen by default. Prompt authors who don't even know
  about provider prompt caching still get its cost benefit, because
  the template structure enforces a cacheable shape.
- The "what's cacheable here?" question has a single, machine-checkable
  answer per template. No silent drift where someone edits the system
  prompt and the cache silently invalidates.
- Provider-specific cache mechanics live in one place. Adding a new
  provider with a new caching API is one function in `cache_aware.py`,
  not a change to every prompt.
- The `strata_forge.evals` Phase-2.4 reports can attribute cost differences
  to cache-hit rate, because each call's cache markers are deterministic.

**Negative**

- Authoring overhead: a one-line prompt is now a template with two
  named sections, not a string. Cost: ~3 lines of boilerplate per
  template.
- Migration friction for prompts that conceptually don't have a stable
  portion (raw classification with no system prompt, single-turn
  Q&A). For those, `stable=""` is acceptable but feels redundant.

**Mitigations**

- A shorthand `PromptTemplate.simple(body, variables=...)` constructor
  treats the entire body as dynamic with an empty stable section,
  trading cache hits for ergonomics. Use it for the genuinely
  stable-less cases.
- `strata_forge.prompts.validation` includes a "prompt smells" lint pass that
  warns when a template's stable portion is short enough (< 100 tokens
  by default) that prompt caching can't kick in on most providers —
  authors get told they're paying the boilerplate tax for nothing.

## Alternatives considered

1. **Manual cache markers in templates.** Let authors place
   `{% cache %}…{% endcache %}` blocks. Rejected: easy to forget, easy
   to get wrong (insert in the dynamic section by accident), and
   silently degrades to no caching when authors don't know about it.

2. **Heuristic auto-split.** Parse the template, classify each segment
   as "looks stable" vs "looks dynamic" based on whether it contains
   variable interpolation. Rejected: heuristics are fragile, the
   resulting boundary is opaque to the author, and the cache invalidates
   any time the heuristic changes.

3. **No structural split; let `strata_forge.llm` do best-effort caching.** The
   LLM client could try to identify a stable prefix across recent
   requests and insert markers post-hoc. Rejected: doing this safely
   requires knowing which content is "stable" by intent, not just by
   coincidence — heuristics on rendered output are unreliable. Better
   to make intent explicit at the template level.

4. **A separate `strata_forge.prompts.caching` module instead of a built-in
   split.** Authors opt into caching by wrapping their templates.
   Rejected for the same reason as ADR 0006 rejected
   "tools-as-opt-in": the right default is "caching works", not
   "caching works if you remember to opt in."
