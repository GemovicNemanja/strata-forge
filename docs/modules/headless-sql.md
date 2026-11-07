# Isolated local dataset SQL

`strata_forge.datasets.sql_types` provides closed immutable values; it does not execute SQL,
read source files, inspect/start Docker or confer access. Importing it requires only the
existing Forge base dependencies. DuckDB is neither installed nor imported in the parent.
The execution architecture is recorded in [ADR0032](../architecture/adr/0032-isolated-local-dataset-sql.md).

`SQLQuerySpec.from_wire` accepts version 1, scalar SQL UTF-8 (1..32768 bytes), an exact input
manifest SHA256, result rows 1..10000 (default 1000) and timeout 100..10000 ms (default 5000).
No engine, image, source path/URL, callback, grant, parameter or export field is accepted.
The SQL remains unchanged and is not parsed here. Defaults are materialized before hashing.

`SQLInputManifest.from_wire` binds a complete provided local population: descriptive origin
with explicit full git_sha1/sha256 revision, config/split, contiguous ordered parts with
byte/row counts and SHA256, exact ordered schema/schema digest, population label and totals.
The ceilings are 32 files/64 MiB/1M rows, 128 columns, logical depth 8/1024 type nodes and
128 KiB canonical manifest. Empty schema-bearing files may describe zero rows; zero files
refuse. This does not verify upstream inventory or current Hub/native access.

`LocalSQLInstallation.from_wire` describes the fixed operator profile, absolute Docker
executable/Unix socket, full immutable image ID and worker-bundle SHA256. It does no filesystem
or engine inspection. `LocalParquetFiles.from_open_fds` describes a matching tuple of caller
file descriptors without opening, duplicating, reading or authorizing them. These operator/
file objects are local Python inputs, not query-wire fields or remote credential types.

Values retain canonical bytes and return fresh payload copies via `wire_payload()`. `digest()`
is SHA256 of those exact versioned bytes. Direct construction, malformed copied/constructed
instances and subclasses do not bypass inspection. Fixed `SQLExecutionError` codes/messages
exclude SQL, rejected values, native error text and causes. A content digest grants no access.

The shared private stdlib codec encodes one bounded length-prefixed frame, rejects duplicate
keys, unsafe/alternate numeric spellings, invalid Unicode, cycles, depth/byte limits and
trailing/incomplete frames. Any feed refusal permanently ends that decoder, so a valid
earlier prefix cannot become a successful EOF after discarded invalid bytes. Decimal bounds
are checked before formatting and closed object cardinality before traversing keys. It
contains no SQL parser. Supported cell values use explicit tags:
integers are decimal strings, decimals preserve scale, finite floats use exact hex, and NaN/
infinities/null are distinct. Bytes use canonical base64; lists/arrays/structs preserve bounded
ordered values. Supported finite dates/times/timestamps retain units/offsets; sentinel bounds,
NS timestamps, intervals, maps, unions, UUID/enum and unknown types refuse. No arbitrary object
stringification occurs. Input casefold collisions refuse; duplicate result column names can
remain separate ordinal descriptors.

`native_cell` operates on resolved builtin Python values only. The generated `typed.parquet`
fixture and owned-container tests verify the fixed native conversion path separately; the
JSON manifest is descriptive metadata and confers no access.


## Fixed image worker

`containers/headless-sql/setup.py` is an explicit operator setup operation, separate from
query execution. It anonymously obtains only the hash-pinned public Python/DuckDB/pytz
artifacts in `profile.json`, builds the fixed Linux ARM64 image with networking disabled,
and writes its immutable image ID and exact two-source worker bundle hash. It does not
install a native engine into Forge. Query execution must not invoke setup or pull an image.
The retained notices cover the image-only distributions. Setup requires an explicit trusted
Docker executable and Unix socket; no ambient Docker context or credential configuration is
used. Image cache availability is not guaranteed by a prior successful build.

The standalone worker requires its fixed UID/cgroups/limits/no-new-privileges/seccomp profile
before READY. It disables Python replacement/frame scans, extension installation/loading,
persistent secrets and logging. DuckDB reports disabled logging as `0`; the worker verifies
that exact installed-version value. Every supplied Parquet part is hashed, inspected for
physical casefold collisions and exact logical schema, materialized and counted before
external access is disabled and configuration locked. Native parsing admits one SELECT-kind
statement, including WITH/SUMMARIZE/DESCRIBE, rather than a second lexical SQL validator.
SELECT-kind admission alone is not a purity or OS isolation guarantee.

Results are fetched in batches at most 64, with one-row lookahead for prefix truncation.
Cell/type/aggregate byte failures refuse the whole result. Image-only pytz is pinned because
DuckDB needs it to fetch TIMESTAMPTZ: only exact builtin datetimes whose tzinfo **is** the
installed pytz.UTC singleton are adapted to stdlib UTC. No arbitrary timezone callbacks or
fuzzy conversions are accepted. The shared table validator checks closed type/cell/coverage
metadata, not execution authority.

`containers/headless-sql/worker-proof.json` retains the synthetic three-row worker proof,
installed source/artifact/profile hashes, actual engine identity and cleanup receipt, plus
its initial logging-readback refusal. Parent lifecycle sources used for that probe were
uncommitted and are explicitly outside this worker checkpoint's acceptance. A complete SDK
operation still requires independently checked parent admission, cancellation and cleanup.


## Local executor

`LocalSQLExecutor` and `SQLResult` are lazy dataset exports. A trusted local Unix operator
provides `LocalSQLInstallation` and already-held ordinary read-only file descriptors:

```python
async with LocalSQLExecutor(installation=installation) as executor:
    result = await executor.query(spec=query_spec, source=provided_files)
    payload = result.wire_payload()
```

Each executor captures its first async operation's event loop under a short thread lock.
Cross-loop calls refuse before source/engine work; it is not a cross-thread executor.
This is one active query with no queue **per executor instance**. Multiple caller-created
instances are not constrained by a process/organization-wide quota. A future service must
provide its own shared owner/admission and reviewed authorization; none is installed here.
`aclose` refuses busy work rather than silently cancelling another caller's query. Use the
query task's cancellation, await it, then close. Repeated close after confirmed cleanup is
safe. An uncertain owner stays unavailable; an explicit close can retry only its cleanup,
never SQL or adoption of another epoch's resources.

The original monotonic budget starts before staging. The SDK duplicates caller descriptors,
checks read-only regular files and copies/hashes every part through bounded reads without
moving the caller's file offsets. A private 0700 job directory contains a read-only `input`
child; only that child is mounted into the worker. Caller descriptors remain open. Source
hash/count/schema mismatches, changed bytes or unknown cleanup never produce a result.
Blocking copy/hash/removal work is owned and drained after repeated cancellation.

The selected Docker Unix socket and immutable image/bundle are checked before create;
pre-start container settings and worker READY independently check the supported profile.
No shell, parent credentials, caller environment, source URLs or Docker context is inherited.
The minimal CLI environment has only a fixed PATH; HOME is omitted, never repurposed. Raw
SQL/data travel only through the private framed input/output; stderr is discarded and the
container log driver is `none`. Parent imports no DuckDB/pytz to perform SQL.

The captured public engine ID is rechecked before cleanup and after the exact absence check.
It is an operational identity, not hostile-operator or cloned-daemon attestation. An observed
identity mismatch retires future admission even if create was never issued; that pre-create
case is an availability policy, not evidence that any worker ran. Cleanup/close cannot silently
clear that retirement through the no-create shortcut. A possible
start requires successful actual `docker wait` and a fresh nonrunning/PID0 observation before
removal and admission release. A same-engine 404/empty list is not stop evidence. A lost create
acknowledgement can be investigated only by its exact opaque name and owner/operation/image
bindings; an empty lookup retains uncertainty because a daemon create may still be pending.
If no start was issued, an observed owned unstarted container may be removed as cleanup-only.
No arbitrary name collision or foreign container is killed. There is no automatic query retry.

The parent reads READY, writes one request and incrementally reads one capped response through
EOF. It verifies exact operation/request/source/bundle identities, observed population and
closed typed table fields. A successful result returns only after actual wait/stoppedPID0,
container removal, helper/attach process reap and staging removal. A result frame requires
actual exit zero; a closed worker error is projected only with the fixed failure exit one.
A plausible result frame after a failed/nonzero/OOM exit cannot mint a zero-exit receipt.
Its immutable canonical
receipt records those observed facts, asserted manifest, source/SQL/spec/result hashes,
selected image/engine/bundle and elapsed stage/container/total milliseconds. Content and
receipt digests do not certify source authority or a hostile host. `SQLResult` reconstruction
revalidates its complete closed data and returns independent copies.

Deadlines are logical result-admission limits, not hard OS cleanup/reap guarantees. A native
call is retired by killing its actual owned container; cancellation is propagated only after
owned cleanup settles. Unavailable engine/unknown wait/removal keeps ownership and refuses
new admission. Cleanup can exceed the original budget, and no late success is returned. Host
or interpreter death has no durable recovery owner and may leave an unknown outcome. The
local Unix/Docker Desktop and Linux ARM64 image evidence does not establish other platforms,
hostile-kernel isolation or network/filesystem isolation from an operator with the Docker
socket or same UID.

`tests/integration/datasets/test_headless_sql.py` is an explicit actual-container suite. First
run the separate setup script, then supply its installation-file path as
`STRATA_TEST_SQL_INSTALLATION`; selection without it fails as a missing prerequisite instead
of counting a fake protocol pass. Query never invokes setup. Tests use generated fictional
Parquet only: SELECT/WITH/DESCRIBE/SUMMARIZE, dynamic mutation/external-read refusals, UTC and
nested values, physical case collisions, prefix versus aggregate/cell refusal, native
recursive-query deadline, repeated cancellation held through actual wait, and missing image.
Oversized text currently uses the shared scalar validator's fixed `invalid_spec`; it returns
no result. SELECT-kind admission plus these tests is a bounded supported profile, not a proof
of every DuckDB function's purity. Engine settings and the OS/container boundary are required.
No server route, MCP tool, worker registry, hosted authorization or remote source is enabled.
