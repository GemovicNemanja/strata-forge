# Modular prompt composition contract v1

Status: coordinator-frozen implementation contract. It is not a shipped API, persisted resource schema or agent wire. [ADR 0023](../architecture/adr/0023-portable-modular-prompt-composition.md) chooses the Python-authoritative compiler. Existing Jinja2 APIs keep their dialect and behavior.

## Portable bytes

The profile is `strata-prompt-json-v1`, not a claim of general RFC 8785 support. Values are null, booleans, Unicode strings, integers from −9007199254740991 through 9007199254740991, dense arrays and string-keyed objects. Object keys contain printable ASCII only; duplicate keys, lone UTF-16 surrogates, floats, nonfinite numbers, negative zero and host-language objects refuse. A JSON parser must reject duplicate keys and invalid numeric spellings before converting to host types. Numbers are integer decimal without exponent or leading zeros. Booleans are never treated as integers.

Canonical encoding sorts object keys by ASCII, preserves array order, writes compact JSON without extra whitespace, and encodes UTF-8 without a BOM. Strings preserve Unicode scalar values and all whitespace exactly: no NFC normalization, trimming or CRLF conversion. Quote, backslash and U+0000–001F use JSON escapes (short escapes for backspace, tab, newline, formfeed and carriage return; other controls lowercase `\\u00xx`); slash, non-ASCII, U+2028 and U+2029 remain literal. Hashes are lowercase SHA-256 of those exact bytes, with a domain string included in the hashed object. A hash never claims secrecy or authorization.

## Definitions and resolution

Modules and surfaces have a logical name, positive safe-integer revision, declared variable types, optional structured constraints and ordered content. Names and variable identifiers are bounded ASCII. Name+revision is an exact coordinate within the supplied registry; no `latest`, moving channel, URL or filesystem path is a dependency. An immutable registry rejects replacing a coordinate with different content. Server integration must resolve permitted immutable versions and verify stored content identity before invoking the compiler.

Each part is exactly one of:

- `{kind:"text", text:string}`: exact literal content, including empty text and braces.
- `{kind:"variable", name:string}`: one declared input, inserted as data once. Inserted text is never reparsed.
- `{kind:"module", revision:string}`: exact pinned module coordinate.

A module contains ordered parts. A surface contains exactly one output target: ordered chat messages (`system`, `user`, `assistant`), or one textual artifact with explicit purpose (`system_instruction`, `tool_description`, `skill`, `text`). Textual artifacts retain their purpose and descriptor metadata; they do not pretend to be chat messages. A tool descriptor's schema is preserved separately from its composed description. No tool-result or multimodal message emulation is implied by v1.

Messages and textual artifacts declare an exact separator, including the empty string. Separators are inserted between expanded leaf parts in their declared order and recorded as compiler-owned source spans. Nested modules never add hidden newlines. Empty leaf parts and repeated module inclusions are preserved, not deduplicated. A repeated inclusion receives its own occurrence path.

Variables have `string`, `integer` or `boolean` type. Every variable part must be declared by its owning definition and compatible with every dependency and surface declaration. Missing values, extra values and incompatible types refuse. Integer and boolean rendering uses the canonical decimal/true/false spelling; string rendering is exact. No ambient variables, attribute access, filters or expression evaluation exists in this dialect. Defaults and partial application are unavailable in v1.

Structured constraints use explicit named string values. Two definitions asserting distinct values for the same key refuse, with bounded references to both origins. A repeated equal value is allowed. These are declared constraints, not an inference about prose meaning. No automatic last-instruction-wins rule exists.

## Build and render identity

A build freezes schema/dialect/compiler versions, surface coordinate+content hash, ordered compiled leaf structure, input declarations, constraints, target/descriptor metadata, source provenance and the complete transitive dependency lock. A dependency lock has one record per coordinate in deterministic ASCII coordinate order; occurrence paths retain expansion order separately. Timestamps, mutable channel names, permissions and UI state do not enter content identity. Different literal bytes, label/order, target, dependency version, provenance or compiler version change identity. Registry insertion order does not.

A render freezes the build identity, input fingerprint, exact output and source map. A caller execution/attempt ID belongs to invocation evidence, separate from deterministic render identity: repeated renders may be identical while two executions remain distinct. Sensitive input values need not be retained for a fingerprint, and a fingerprint is not proof of provider use.

Source spans use half-open UTF-8 byte offsets within each final message/text output. Each span records output index, start/end, leaf origin coordinate and exact occurrence path; variable spans also name the declared variable. Separator spans are explicitly compiler-owned. Astral characters, combining sequences and CRLF are measured as their actual UTF-8 bytes. Consumers must not apply byte offsets as JavaScript code-unit indexes or Python character indexes. Zero-length leaves may have zero-length spans. Slicing output bytes over a span recovers exactly that emitted leaf, and no span splits a Unicode scalar.

Provenance can retain repository, full commit, path, source-file hash, selector and declared import dialect. It is content, not a URL to fetch. Importing requires explicit authorized source reads; protected research prompts are not fixture material. A source pointer never grants access.

## Diagnostics, limits and acceptance

Missing pins, cycles, replaced versions, unsupported targets/fields, invalid variables, contradictory structured constraints and exceeded bounds refuse with a stable diagnostic code and bounded origin paths. Diagnostics never echo input values or whole prompt bodies. Duplicate literal passages are advisory with exact passage references and remain in output. A semantic analyzer is a separate optional service whose findings retain analyzer/model/prompt versions and uncertainty.

The first implementation must bound registry size (4096 definitions), per-definition parts/messages (1024/128), dependency depth (32), expanded leaves (16384), distinct variables (256), total definition bytes (8 MiB), rendered bytes (8 MiB) and returned diagnostics (64). Exceeding a limit refuses before publication; no partial successful build or silent truncation. Limits are versioned compiler capabilities, not claims about universal prompt lengths.

Golden fixtures cover canonical bytes/hashes, key order, empty/repeated leaves, nested ordering, pins/cycles, mismatched variables/constraints, astral and combining characters, LF/CRLF, controls/escaping, target preservation, source-map round trips and render/attempt distinction. Python and TypeScript must pass the same corpus. A separately authorized importer must prove unchanged Strata development prompt semantics; research prompts stay with Claude unless explicitly approved. Persistence, review/promotion, transport capture and product UI remain downstream acceptance.
