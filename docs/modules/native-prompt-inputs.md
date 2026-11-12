# Private native human prompt inputs

`pipelines._native_prompt_input` renders a retained prompt build through the installed
`HumanRenderInputReader`. It is a private in-process consumer: no package export, worker,
CLI, registry, provider call or transport is installed. The existing
[immutable reader](immutable-inputs.md) owns byte/FD resolution and cleanup; the existing
[modular prompt consumer](prompt-composition.md) owns build validation and rendering.

`native_prompt_intent_digest` accepts explicit principal, project, operation and attempt
IDs, complete surface and build version references, the complete manifest payload entry
and scalar values. The fixed canonical object contains `domain` equal to
`strata-human-prompt-render-intent-v1`, every `HumanRenderInputRequest` field except
`spec_sha256`, plus `surface_ref` and `values`. Thus `version`, `admission="human_render"`,
`purpose="prompt_render"` and `kind="prompt_build"` are explicit digest inputs. The
helper uses the existing portable JSON profile; its SHA-256 excludes only the digest
field itself. The server computes this value before constructing the request.

`consume_native_prompt_input(reader, request, *, surface_ref, values, deadline)` requires
the explicit human request variant. Human operation IDs are never mapped to legacy run
IDs and do not assert durable Run records. Installed server code owns live admission;
the digest and IDs cannot grant source access. The complete manifest entry must describe
`build.json`, `application/json`, `strata.prompt.build.v1` and a positive payload no larger
than 8 MiB.

The first stage returns a `NativePromptConsumedCandidate` containing the request,
surface reference, existing `ConsumedBuild` and canonical `values_json` bytes. The server
validates `candidate.consumed.canonical` with its existing retained-build loader against
the actual native expected build ID, size, surface and complete dependency pins before
calling `render_native_prompt_consumed(candidate, *, deadline)`. This stages the existing
algorithms around server validation without a second source read or a validator callback.
The second stage revalidates the complete intent, descriptive source and consumed content,
and parses a fresh value snapshot. `render_native_prompt_input` composes both stages for
callers that keep their own required native validation boundary.

Values are a plain dictionary of at most 128 keys and 64 KiB of portable canonical JSON.
Only strings, portable integers and booleans are admitted. A fresh scalar snapshot is
hashed before source resolution and used across every await. Required variable names
and exact declared types remain the existing renderer's responsibility. Values are
inserted once and never interpreted as template syntax.

One original deadline, capped at twelve seconds from first-stage entry, covers intent
validation, byte reading, intermediate validation and rendering. The consumed candidate's
read-only `original_deadline` property is backed by private frozen state retaining the
actual originating task and original value snapshot. That local consistency state grants
no server or native authority. A later second-stage deadline can only shorten the retained
deadline, and a different caller task refuses. The caller task's cancellation and the original
deadline are checked before and after rendering. Bounded synchronous rendering can
exceed a soft deadline; its result is then refused. The complete canonical render,
including descriptor and source-map metadata, is limited to 8 MiB. Overflow refuses
without truncation; this is narrower than the modular renderer's text-only profile.

The returned `NativePromptRenderCandidate` has a content-free representation and is an
internal candidate, not a disclosure authorization, compiler attestation or provider-use
receipt. The server must verify the actual retained build's complete dependency pins
against its retained producer state, and must perform its final current browser/session,
project, source/custody and unchanged-provenance fence after rendering and before exposing
the candidate. The local descriptive source reference carries the surface resource ID;
the outer intent binds the entire surface reference. Neither establishes native linkage.

Errors contain fixed private codes rather than rejected values, paths or raw diagnostics.
Focused tests use synthetic builds and installed authority/source doubles with actual
reader ownership and temporary regular files. They establish no server deployment,
remote grant, real research artifacts, provider use or model quality.
