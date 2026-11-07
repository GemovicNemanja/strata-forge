# Local SQL value contracts

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

`native_cell` operates on resolved builtin Python values only; its fixtures are not evidence
of DuckDB result conversion. Actual native Parquet/UTC/ICU/schema behavior and full SQL
execution remain unavailable until the isolated worker and lifecycle pass their own tests.
The committed JSON fixture is metadata-only, not an actual Parquet source or authority.


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
