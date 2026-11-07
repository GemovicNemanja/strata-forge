# Local modular prompt consumer for Node 24

`strata-prompt-sdk-local` is a private source package, with no runtime dependencies or registry
publication. It consumes Python modular builds and uses the existing portable reference kernel.
It cannot compile authored modules/surfaces, publish versions, fetch resources, authenticate,
call a model, render Jinja or export traces. Node crypto/util/Buffer are required; this is not a
browser package. The name claims no npm scope or package availability.

From a Node 24 source-checkout application:

```ts
import { consumePinnedBuild, renderConsumed } from "./packages/prompt-sdk/src/index.mts";

const build = consumePinnedBuild(callerBytes, {
  expectedBuildId: expectedSha256,
  expectedSizeBytes: expectedByteLength,
  source: { kind: "caller_bytes", partition: "my-application:local", reference: null },
});
const rendered = renderConsumed(build, { question: "Explain {{literal braces}}." });
```

The application supplies bytes, both exact expectations and the explicit descriptive source.
Maximum build size is 8 MiB. The wrapper verifies size/hash, existing closed schema and canonical
bytes; it snapshots mutable byte input and metadata. Byte carriers require a genuine native
Uint8Array/Buffer brand; a matching prototype alone refuses through the fixed consumption
error before any native byte accessor. A caller-declared native reference is not an authorized
server read or source attestation. Partitions are caller assertions, not tenant
grants. There is no URL/token/default/latest/offline fallback or content cache.

`renderConsumed` rechecks the snapshot and delegates to the existing `renderBuild`. Missing/extra
or wrongly typed string/integer/boolean bindings refuse. Text is literal, inserted once and
never evaluated. The returned immutable render has the existing `render_id`, build/input
fingerprints, exact chat/artifact output and UTF-8 source map. Its offsets are byte offsets,
not JavaScript UTF-16 indexes. Artifact purpose/descriptor does not imply a system message/tool.

The lower-level `canonicalBytes`, `parsePortableJSON`, `hashDomain`, `consumeBuild`, `consumeRender`
and `renderBuild` exports retain the existing restricted profile and reference behavior.
`consumeBuild` validates schema/canonical content only; use `consumePinnedBuild` when a mandatory
expected pin is required. No content hash attests a compiler, current permission or provider use.
Fractional request settings are outside this portable prompt profile.

All `.mts` exports run directly on Node 24: no hidden `dist` build, filesystem access or CLI work
at import. Shared conformance from the repository root remains:

```sh
node scripts/prompt_reference.mts tests/fixtures/prompts/portable-bytes-v1.json tests/fixtures/prompts/composition-renders-v1.json
node --test scripts/prompt_reference.test.mts
node --test packages/prompt-sdk/test/*.test.mts
```

`npm test` inside this directory runs the package tests with Node's built-in runner. TypeScript
development checking can use the existing repository conformance toolchain; no compiler/type
dependency is installed into the Python environment or added to its wheel/sdist. See
[`docs/modules/prompt-composition.md`](../../docs/modules/prompt-composition.md) for the exact
types/check command. The fixture copy/provenance is in [`test/fixtures/`](test/fixtures/README.md).

Request/response attempt capture, provider integration, remote service credentials, retention,
server ingestion and MCP remain unavailable. This local source wrapper does not implement those
capabilities or complete their product acceptance.

## Explicit local attempt metadata

`prepareInvocation(pinnedBuild, values)` revalidates caller-provided bytes and uses the
existing modular renderer. `captureOnce(invocation, requestBytes, transport,
{parents, declaredBaseline, signal?})` invokes only the explicitly supplied
`transport.exchange(bytes, {signal})` behavior. It does not install a provider/HTTP client,
read credentials, fetch a build or export a trace. Retrieval is qualified `caller_bytes`,
not remote retrieval or native access. Provider/model/usage/cost and the relationship
between the render and opaque request bytes remain unknown. Request JSON (including
fractional settings) is never parsed or reserialized by the recorder.

The sole retention mode is `metadata_only`. `snapshot()` returns a copied/frozen closed
metadata DTO; `validateAttemptMetadata`/`validateInvocationMetadata` only parse reported
metadata and cannot issue a live handle or authority. `rendered()` and `responseBody()`
explicitly return working content. Weak-map working state, safe `util.inspect` and a
refusing `toJSON` avoid accidental diagnostic serialization; `structuredClone` produces
an unusable empty object with no content, rather than another live handle. This is not
hostile-process protection, zeroization or a guarantee about intentionally logged accessors.

The exchange callable is resolved once before ticket allocation. Only a genuine native
Promise return is supported (no subclass, custom constructor or arbitrary thenable).
There is one active promise/controller per invocation. Caller abort retires the generation,
forwards one fixed reason and waits until the same promise settles; no `Promise.race` or
background drain permits premature slot reuse. A late return is not inspected/hashed or
accepted as success. Cancellation throws SDK-owned `PromptCaptureError` with fixed
`name='AbortError'`, `code='local_cancelled'`; caller signal.reason and raw errors are
never propagated. Already-aborted preflight allocates no attempt. Close and settled
snapshots refuse while active; closing is otherwise idempotent.

Limits are 64 attempts, 16 MiB request/response, 4 KiB canonical attempt metadata and
256 KiB invocation metadata. Preflight reserves metadata budget. Requests use native
byte-brand validation and copies; the delegate receives a separate copy. Under-limit
returned bodies are copied. Oversize response working objects retain only the caller's
existing byte reference/size without copying or hashing it during capture; metadata
records explicit omitted coverage and a fixed capture error. Bodies never enter the
ledger. A declared baseline is an assertion: `not_observed` applies if it exists but no
handoff occurred, `not_supplied` if absent, and same/changed only to captured hash+size.
Retries/fallback/continuation are explicit calls linked only to prior terminal attempts
in this invocation. No model success, spend permission or process-death claim follows.
