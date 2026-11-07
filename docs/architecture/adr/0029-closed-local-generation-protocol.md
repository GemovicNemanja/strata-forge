# ADR0029: Closed local generation protocol

Status: Accepted

## Context

The inert batch generation path exposes three sampling controls and returns one normalized
completion. Its first-choice response cannot preserve multiple samples or explicit local
engine reasoning channels. A generic provider adapter may reinterpret literal tag-shaped text.
Generation needs explicit identities, bounded schema/channel semantics and truthful capability
refusals, while existing completion callers retain their behavior.

## Decision

Define a separate versioned generation contract with a finite engine-profile table, strict
controls, stable item/sample identities and explicit seed semantics. One logical sample maps
to one engine choice. Native support requires verified installed/profile/process evidence;
a requested model name, profile or endpoint does not grant capability or read authorization.
The initial fixed observation seam is unavailable until owned native-process integration and
installed render/parser/backend conformance exist. Synthetic tests may privately substitute
that observation; production has no test flag or asserted capability constructor.

An additive LLMClient operation is the sole high-level entry to a narrow raw OpenAI SDK method
inside the existing OpenAI-compatible provider module. This fixed local-generation method is
an explicit exception to the LiteLLM transport rule: it preserves actual protocol fields
without regex channel extraction, silent choice loss or schema rewriting. It uses a bounded
raw response stream, explicit observed endpoint, no ambient credentials/redirects/retries/cache
or fallback. Legacy complete/stream/normalizers/results remain unchanged. The exception does
not authorize provider SDK imports in workers or a second general provider/plugin framework.

The finite inert JSON dialect rejects unsupported keywords rather than altering their meaning.
Final-channel JSON/schema validity and completion state are distinct. Seed derivation controls
requested inputs, not output equivalence across hardware or scheduling. Generic engine defaults
are explicit unknown request seeds, never a claim of randomness. Private artifacts may preserve
channels; public progress/agent projections do not acquire raw reasoning by this decision.

## Consequences

Native vLLM and named model profiles remain unavailable without their exact implementation
checks. A schema or synthetic protocol fixture is insufficient to claim native model support.
Separate worker/process, policy/custody, release/dispatch and app/tool integration gates remain.
Existing inference/scoring/evaluation serialized defaults, digests and result models do not
inherit these controls or sampling semantics.

Owned response streams and HTTP pools are closed before cancellation returns to the caller.
Repeated cancellation while closing is retained until that owned closure finishes; it cannot
produce a later successful result. The request deadline does not promise a bound on resource
closure itself.
