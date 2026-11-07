# ADR 0032 — isolated local dataset SQL

Status: local value/codec prerequisite; executable worker remains unavailable.

## Context

Browser DuckDB-WASM is not a headless SQL executor. Native SQL has process privileges;
SELECT-kind parsing, locked settings and an outer result limit are insufficient isolation.
The existing local compute backend is a shell runner and does not enforce its ResourceSpec.
A source hash or a caller's metadata cannot stand in for project/custody/session authority.

## Decision

Closed Pydantic-backed values retain immutable canonical UTF-8 payload bytes. Public parsing
and execution revalidation use a shared stdlib-only schema/codec, exact classes and fixed
errors. Metadata defaults enter canonical identities; file/type/column order is retained.
The synchronous helpers are a narrow pure-value exception to the async-first rule; they
perform no I/O, SQL parsing, imports selected by inputs, runtime setup or authority checks.
Internal worker wire errors are stdlib exceptions with fixed codes and never leave that
boundary; public failures extend ForgeError without raw inputs, cause or context.

The declared first profile is one source relation, pinned local Parquet manifest, Linux
ARM64, Python 3.14.5 and DuckDB 2.0.0 in a fresh isolated container. Trusted operator engine
socket/image metadata is separate from query input. Missing engine/image refuses without
pulling, resolving a tag or falling back to native SQL in the parent. A complete executor
must own source staging, verified resource/profile admission, the actual worker, bounded
result framing and stopped/waited/removed cleanup. No executor is implemented by the value
prerequisite, and this decision does not register a worker or a remote capability.

The executable boundary must disable and verify Python replacement/frame scans before
user SQL, use the actual native one-SELECT-kind parser, and test dynamic query/query_table/
pragma paths. SELECT-kind is not a purity proof. All provided source parts are validated
and materialized; no source prefix is substituted. Query result collection uses fixed small
batches with an aggregate byte cap, not result-sized fetchall/arrays. Result-row truncation
is explicit and distinct from source completeness and unmeasured query scan coverage.

The first codec preserves bounded integer/decimal/float/null/text/bytes and supported finite
microsecond temporal/list/array/struct values. NS timestamps, intervals, maps, unions and
unknown types refuse instead of losing precision or structure. Input casefold collisions
are conservatively refused; this is a declared source-profile restriction, not an assertion
that every such Unicode spelling collides in DuckDB. Actual native conversion, Parquet,
UTC/ICU and no-renaming behavior require installed-image tests before execution availability.

## Consequences

No parent DuckDB dependency/import, Docker operation, source read, remote authority, project
sharing, durable crash recovery, API/MCP wire or tool registration follows from these types.
A trusted local caller already possesses its file descriptors; their origin labels and
revisions are assertions, not resolved source provenance or permission. Descriptors are
never admitted by the query wire and must be duplicated/checked/staged by the executor.

A future local executor has one active operation per owner; it is not an organization-wide
quota or a hosted service. Cancellation and deadlines must withhold results until actual
container wait and owned cleanup; OS/engine failure can delay cleanup, with no hard reap
bound or new admission on uncertainty. No model, provider, research or remote job is part
of this contract.
