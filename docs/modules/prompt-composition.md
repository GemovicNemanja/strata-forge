# Portable modular prompts

`strata_forge.prompts` provides an independent explicit composition dialect alongside its
existing Jinja2 templates. Python compiles authored modules/surfaces. A bounded TypeScript
reference consumer renders Python builds against the same fixtures; it cannot compile
definitions or publish an alternative compiler identity.

See [ADR 0023](../architecture/adr/0023-portable-modular-prompt-composition.md), the
[portable contract](../contracts/modular-prompts-v1.md) and
[module rules](../../src/strata_forge/prompts/CLAUDE.md).

## Author, compile and render

```python
from strata_forge.prompts import (
    ChatMessageParts, ChatTarget, CompositionRegistry, ModulePart, ModuleRevision,
    SurfaceRevision, TextPart, VariableDecl, VariablePart, compile_surface, render_build,
)

registry = CompositionRegistry()
await registry.put(ModuleRevision(
    name="shared/style", revision=1, parts=(TextPart(text="Be clear."),),
))
await registry.put(SurfaceRevision(
    name="agent/system", revision=1,
    variables=(VariableDecl(name="topic", type="string"),),
    target=ChatTarget(messages=(ChatMessageParts(
        role="system", separator="\n",
        parts=(ModulePart(revision="shared/style@1"), VariablePart(name="topic")),
    ),)),
))
build = await compile_surface(registry, "agent/system@1")
rendered = await render_build(build, {"topic": "Explain {{literal braces}}."})
output = rendered.wire_payload()["output"]
# {"kind": "chat", "messages": [{"role": "system",
#   "content": "Be clear.\nExplain {{literal braces}}."}]}
```

`ModuleRevision` contains ordered `TextPart`, `VariablePart` and `ModulePart` nodes. Text
including empty strings, braces, Markdown and template-looking fragments remains literal.
Variables are inserted once without reparsing. A separator belongs to the final message or
artifact and is inserted between every expanded leaf; nested modules add no hidden whitespace.
Repeated inclusions have independent occurrence paths. A module with no parts contributes no
leaves; an explicit empty text part contributes one zero-length leaf.

`SurfaceRevision.target` is either `ChatTarget` with ordered system/user/assistant messages or
`ArtifactTarget` with purpose `system_instruction`, `tool_description`, `skill` or `text`.
An artifact is never converted into a system message. Its optional `PortableDocument` descriptor
preserves schema/metadata separately from the composed text. JSON authoring accepts a portable
object directly; Python callers can use `PortableDocument.from_value({...})`.

Names are exact letter-leading ASCII namespaces, at most 128 characters, using letters,
digits, `_`, `.`, `/` and `-`. Empty slash segments and `..` refuse. Revision coordinates use
`name@positive_decimal`, with no leading zeros and no aliases. Revisions are safe integers.
Variables use `[A-Za-z_][A-Za-z0-9_]{0,63}` and explicit string/integer/boolean types. Every
reference must be declared by its owning definition; a surface declares the compatible union
of all dependency inputs. Extra surface declarations remain required at render time. Missing,
extra, float, boolean-as-integer and incompatible values refuse.

`definition_from_wire(value)` validates a closed authored JSON value with fixed diagnostics.
Registry `put` also validates/snapshots input. `get` returns an independent frozen definition.
An identical coordinate/content repeat is idempotent; replacing its content refuses. There
are no mutable channels, database stores or automatic dependent-build updates in this API.

## Identity, constraints and attribution

`PromptBuild.canonical` and `PromptRender.canonical` are immutable UTF-8 bytes. Their respective
`build_id` and `render_id` properties are SHA-256 hashes. `wire_payload()` returns fresh JSON
objects; mutating an inspection result cannot change a stored registry value or identity.

A build includes schema/dialect/compiler versions, surface identity, complete ASCII-sorted
dependency lock, ordered leaves, input declarations, resolved constraints, provenance and target
metadata. Registry insertion order does not affect it. Literal bytes, dependency revisions,
target, declaration/leaf order, provenance and compiler versions participate in identity.

A render freezes the build ID, domain-separated input fingerprint, exact output and source
map. It does not retain a separate input-value copy, although inserted values appear in output.
Execution IDs belong to separate invocation evidence. Repeated renders can have identical IDs
while different attempts remain distinct. No model/provider request or response is produced.

`PromptConstraint(key=..., value=...)` records an explicit named fact. Equal repetitions are
allowed. Distinct values for the same key refuse with both definition origins. This does not
detect contradictory natural-language instructions. Exact nonempty duplicate literal leaves
produce advisory diagnostics with all occurrence references and remain in output. More than
the diagnostic bound refuses; findings are never silently truncated or text rewritten.

Every source-map span has output index, start/end, kind, origin coordinate and occurrence path;
variable spans also name the input. Separators have compiler-owned `$compiler` origins.
Offsets are half-open **UTF-8 byte offsets**, never Python character or JavaScript UTF-16
indexes. For `A😀é\r\n`, leaf byte ranges are `0:1`, `1:5`, `5:8` and `8:10`. Slice encoded
bytes and decode UTF-8 to recover a leaf. Empty leaves/separators retain zero-length spans.
JavaScript span decoding uses `TextDecoder("utf-8", {fatal: true, ignoreBOM: true})` so a U+FEFF
inside prompt text stays data. A leading BOM on the JSON document itself is refused.

Accepting a render snapshot validates closed target/message shape, leaf/separator ordering,
full contiguous coverage, actual content bounds and Unicode scalar boundaries. Accepting a
build validates its closed schema, lock/target/leaf consistency and declared inputs. A
content hash plus schema validation is **not a compiler attestation**, provenance proof,
authorization grant or evidence that a provider used the prompt. Trusted server consumers
must verify expected immutable IDs, authorized source reads and publication permissions.
Origins cannot be independently proved from a build hash without the trusted build/source.

## Portable bytes and bounds

`canonical_bytes`, `parse_portable_json`, `portable_sha256`, `definition_identity` and immutable
inspection methods are synchronous pure helpers. Registry/compile/render operations are async.
There is no synchronous I/O exception or runtime wrapper. Existing Jinja2 APIs keep their
signatures, expression dialect, stores and cache-hint behavior.

The `strata-prompt-json-v1` profile permits null, booleans, Unicode scalar strings, safe integers,
dense arrays and printable-ASCII-keyed objects. It rejects duplicate keys before overwriting,
numeric exponents/fractions/negative zero, nonfinite values, surrogates and host objects. It
sorts keys, preserves array order and all Unicode/whitespace bytes, and uses compact UTF-8 JSON
without BOM or normalization. This is a named restricted profile, not general RFC 8785 support.
SHA-256 helpers require a nonempty domain inside the hashed object.

`await composition_capabilities()` describes `strata-modular-compiler/1`:

| Capability | Bound |
| --- | ---: |
| Registry definitions / aggregate canonical definition bytes | 4096 / 8 MiB |
| Parts per definition / messages per surface | 1024 / 128 |
| Nested module depth / expanded leaves | 32 / 16384 |
| Distinct declared variables / returned diagnostics | 256 / 64 |
| Rendered UTF-8 bytes across all outputs | 8 MiB |
| Portable JSON envelope / nesting depth | 64 MiB / 64 |
| Expansion visits | 1048576 |

The larger JSON envelope allows escaped text, provenance and source maps around bounded output.
The graph pass counts leaves/depth before expansion; empty dependency DAGs are not expanded
exponentially. Exceeded bounds refuse with `PromptCompositionError.diagnostic`, containing
a stable code and bounded origins, without prompt/input bodies. Compiler bounds are platform
capabilities and do not declare research acceptance or task-specific thresholds.

## Conformance and authorized imports

Python tests recompile `tests/fixtures/prompts/composition-renders-v1.json` to frozen identities.
`portable-bytes-v1.json` carries independent canonical byte/hash and UTF-8 slice vectors.
Node 24 runs the same corpus plus malformed schema/metadata/host-value tests:

```bash
uv run pytest tests/unit/prompts
node --test scripts/prompt_reference.test.mts
node scripts/prompt_reference.mts tests/fixtures/prompts/portable-bytes-v1.json tests/fixtures/prompts/composition-renders-v1.json
uv run python scripts/import_operational_prompts.py --output /tmp/strata-operational-prompts.json
node scripts/prompt_reference.mts /tmp/strata-operational-prompts.json
```

The conformance commands require an explicit Node 24 runtime and check the frozen corpus and
operational import bridge independently of the Python test jobs. Base clean-install smoke
constructs and renders the compiler without extras or tokenizer assets. With TypeScript and Node types installed as development tools, the
strict reference check is:

```bash
tsc --strict --noEmit --target es2022 --module nodenext --moduleResolution nodenext --types node --typeRoots ./node_modules/@types --skipLibCheck --allowImportingTsExtensions scripts/prompt_reference.mts scripts/prompt_reference.test.mts
```

These development types are not Python package/runtime dependencies. The portable corpus also
preserves `__proto__`, `constructor` and `toString` as ordinary descriptor/variable keys;
null-prototype parsing and own-property/Map lookups prevent inherited values or prototype
pollution from changing output. Sparse arrays, custom prototypes, proxies, getters and hidden
properties are not portable JSON.

The operational importer accepts only the hash-pinned archived fixture from Strata server commit
`962d4da84ae4235c736dc2f9f7f72d0ee71530df`, source file SHA-256
`232816d1cd753cac1cf6533119b2565b862ae498dae2e92f1a74d1ce144667ea`. It preserves six modules and
seven surfaces against their recorded literal/f-string baselines, including inserted braces.
It imports/executes no server code, reads no research/verifier prompts, and refuses fixture drift.
This is archived operational equality, not migration of the currently deployed server prompts.

The compiler itself needs no tokenizer/model assets, network or provider calls. Existing legacy
Jinja2 cache-hint tests use their normal public tiktoken rank fixtures. Database persistence,
authorization, promotion, UI, MCP and SDK transport capture remain separate integration work.
