# Local native dataset execution

The fixed dataset transformation profile is separate from the supplied-Parquet read-SQL
profile. Closed local values describe raw import bytes, scalar conversion schema and exact
ordered aliases/Parquet manifests. They do not confer remote source or platform authority.
The first column ceiling is 128, matching the existing lossless typed table codec.

`DatasetInputManifest`, `DatasetConvertSpec`, `DatasetQuerySpec`, `DatasetInstallation`,
`ProvidedDatasetFiles` and `NamedParquetFiles` validate exact closed shapes. No missing option
selects a default engine, SQL timeout or result budget. Parquet conversion derives actual
schema; CSV/JSONL requires explicit scalar schema and nullability. CSV uses fixed UTF-8,
header/comma/quote-doubling and reserved literal `\N` for null even when quoted; empty text is
an empty string. Decimal inputs have the exact declared scale; negative decimal zero refuses
because native storage would discard its sign. No nested CSV/JSONL conversion is inferred.

The shared container owner has only two internal enum profiles. The old SQL default retains
its exact flags, image, worker, wire, READY and cleanup rules. The dataset profile adds only
its fixed memory/file/tmpfs limits, entrypoint and exact owned output mount. It accepts no
caller configuration map or command. Its worker permits only the single fixed output file
before locking external access. Native SELECT admission and that file allowlist are separate
from OS isolation and must be verified in the actual image.

The dataset worker composes unchanged fixed native type/schema/settings helpers from the
read-SQL worker. It never invokes or modifies that worker's profile/preload/entrypoint.
This internal reuse requires no parent native engine import. Source/profile consistency
hashes remain descriptive evidence, not source permission or hostile-host attestation.

CSV lexical validation rejects quotes inside unquoted fields and lone CR separators before
stdlib CSV parsing; it preserves quoted doubled quotes and embedded LF/CRLF without rewriting
the source. The dataset worker sets engine memory to 192 MB and temp storage to 64 MB and checks
DuckDB 2.0's reported `183.1 MiB`/`61.0 MiB` both before READY and after configuration lock.
These are engine settings inside the independent 512 MiB OS memory/128 MiB scratch profile.

## Executor and byte ownership

`LocalDatasetExecutor(installation=DatasetInstallation)` is an explicit async context with one
active operation and no queue **per instance**. The first operation captures its loop. There
is no process/project quota, platform source resolver, durable job or query-time image setup.
A server adapter must supply its own shared admission and current native source policy.

- `convert(spec=DatasetConvertSpec, source=ProvidedDatasetFiles)` validates supplied Parquet
  or explicitly typed CSV/JSONL and returns normalized Parquet.
- `query(spec=DatasetQuerySpec, sources=tuple[NamedParquetFiles,...])` returns the existing
  bounded lossless typed table. Its prefix does not imply a complete scan or persisted data.
- `materialize(spec=DatasetQuerySpec, sources=...)` writes the full SELECT result, refusing
  more than one million output rows before the cell scan. Result-row preview limits do not
  silently truncate materialized data.

All options are explicit canonical v1 values; supplied descriptors are read-only regular
files, duplicated and copied through bounded `pread` without changing caller offsets. Raw
imports bind exact part hashes/sizes. Named inputs bind complete existing SQLInputManifest
values and unique ASCII aliases; their ordered binding digest must match the query spec.
Aggregate limits apply across all tables: four tables, 32 parts, 64 MiB and one million rows.
Text imports additionally cap at 16 MiB. Source assertions remain unverified upstream claims.

`DatasetResult.wire_payload()` returns independent metadata/typed table copies. Its canonical
receipt binds exact spec/input/alias/query bytes, operation, selected engine/image/bundle,
output schema/count/hash/size and observed cleanup. `output_bytes()` returns immutable Parquet
bytes for conversion/materialization only; query results have no output bytes. Both fields
are excluded from automatic model dumps/repr. Receipts can be reconstructed/validated but
are descriptive, not platform permission or hostile-host execution attestation.

Native output is opened no-follow as a regular file, checked for exact sole filename,
size/hash and read with a 64 MiB cap. Success waits for actual container exit/wait/PID0,
removal/attach reap and staging removal. Cancellation retires admission and drains owned
copy/read/container/cleanup before propagating; unknown cleanup retains an unavailable owner.
Close refuses active work; cancel/await that operation first. No automatic query retry or
terminal inference from elapsed time exists. The original deadline limits result adoption,
not the time required to prove process death.

## Observing a settled failure

`LocalDatasetExecutor.failure_observation(error)` inspects only the exact actual
`DatasetExecutionError` issued by that executor's failed call. It returns an opaque
immutable `DatasetFailureObservation`, or `None` when the original task/loop/call,
error identity, current owner or completed settlement cannot be established. A
caller-created exception with the same code is not an observation. Capture it in
the actual SDK call's exception handler before crossing a task boundary.

The observation's `wire_payload()` returns a fresh closed descriptive mapping, or
`None` after retirement. It contains only version, fixed code/kind, operation and
request digests, measured engine/image/bundle/profile, wait/PID-zero/removal/attach/
staging/exit/OOM facts and `reusable_at_settlement`. SQL, schema, rows, paths,
descriptors, raw exceptions, credentials and permission proofs are absent. The
handle has a safe representation, rejects copy/pickle/mutation, and retains only a
weak executor. The executor retains weak error/task references and bounded immutable
metadata; an observation does not extend their lifetime.

A complete correlated worker refusal plus actual container and staging settlement
can describe the same executor as reusable at that instant. This does not promise
that another query succeeds, certify input origin or authorize a platform read or
write. Known profile/deadline/resource/unavailable refusals remain conservatively
nonreusable in this projection. Missing or uncertain create/start/cleanup evidence
and cancellation issue no observation. The unchanged direct-Python API may permit
reuse after confirmed cancellation cleanup; absence of a public witness is not an
assertion that the legacy executor is poisoned. A conservative downstream adapter
must not infer continued health from an error code, elapsed time, task completion
or an `available` flag.

The next admitted operation or `aclose()` retires inspection of the previous
observation. Mutating `error.code`, retaining a copied payload, closing an owner or
using another task cannot create eligibility. Returned metadata can be retained as
historical description; neither it nor its booleans are a source or execution
capability. Actual same-instance failure→success acceptance is separate from the
controlled parent tests. Worker, wire, image, profile and resource caps are unchanged;
the legacy read-SQL result serializer ignores the optional internal settlement data.

The original deadline remains in the parent monotonic clock domain. Loop cancellation is
scheduled from its remaining duration, including CLI setup and cleanup; arbitrary event-loop
clock origins are not interchangeable absolute timestamps. The read-SQL executor shares this
rule without changing its worker, wire, image or resource limits.

Immutable output bytes may require transient copies; callers must account for them in their
own global quota/admission and stream chunks respecting their actual storage adapter limits.
The package does not silently enlarge a store's chunk or object cap. Parent DuckDB/Arrow,
provider/model execution, remote credentials and browser authority are absent from this API.

## Explicit operator setup and evidence

`containers/native-datasets/setup.py --docker <absolute-executable> --engine-socket <absolute-UDS>
--artifact-cache <owned-public-cache>` builds the separate fixed Linux ARM64 image from pinned
public Python/DuckDB/pytz artifacts. It uses empty private Docker config, fixed PATH and no
ambient credentials/context; it inspects only its exact pinned base/tag. An absent base can
be pulled only by this explicit setup, never by an executor query. Public artifact hashes
and licenses are retained alongside the profile. No native dependency is added to the parent.

The image applies explicit `0444` permissions to its public worker scripts so
the fixed nonroot worker can read them regardless of the operator's host umask.
The [recipe](../../containers/native-datasets/Dockerfile) sets only those file modes
inside the image before selecting the worker user and supports the legacy builder.
Private host context and credential modes remain restricted.

`tests/integration/datasets/test_native_dataset_execution.py` requires the explicit
STRATA_TEST_DATASET_INSTALLATION file; selection without it fails instead of silently using
fake execution. Its fixtures are generated fictional bytes, not upstream datasets. Native
profile and parent-platform evidence must be reported separately; macOS parent plus Linux
ARM64 worker is not a Linux parent or hostile-kernel isolation claim. Unknown create/start,
engine identity or cleanup is never converted into success. This local executor does not
complete the native dataset service, durable background SQL jobs or DATA-001.
