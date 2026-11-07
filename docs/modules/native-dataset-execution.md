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
