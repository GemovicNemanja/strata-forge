# Independent portable-path successor — actual installed SDK

Exact reviewed runtime `41b9ae61735e68fd618f9689c874533b4e2132b2` changes only the parent
staging leaf from fixed `/private/tmp` to fixed `/tmp` followed by strict canonical resolve.
No worker/profile/compiler/source format or dependency is changed. Review uses an immutable
git archive; no owner source, profile or operator configuration was edited.

The NEW isolated wheel SHA-256 is
`f720e5a9186c2b8cf80e486e600f1d7876587be6d4aa370e19b625ee59ee134e`;
installed target `/private/tmp/strata-headless-sql-installed-portable-20261107`.
All six SQL runtime/export leaves match both frozen source and wheel, before and after the
case. Only datasets/sql.py differs from the predecessor. Setup, Dockerfile, profile,
DuckDB/pytz artifact hashes, fixture and other five leaves are unchanged. Detailed hashes
are in preflight.json. The old wheel/target and all prior evidence files are unchanged,
verified against predecessor-evidence-hashes.json after this operation.

Exactly ONE separately authorized successor operation ran the fixed anonymous pinned
setup and immediately the SAME eight-column known installed-SDK query. New empty setup
configuration and credential-cleared process environments were used with the explicit
Docker executable/socket. Operator setup used only the pinned public Python base and two
hash-verified cached public wheels; native build networking was disabled. Query used no
source overlay, runtime pull, retry or extra selector, and inspected no foreign Docker
resources. The setup configuration remains empty afterward.

The committed fictional Parquet fixture is 1,461 bytes with exact source/manifest hashes.
Actual three-row output is exhausted and equals the predecessor's typed table: full int64
values including 9007199254740993, Unicode/null text, scale-preserved decimals, date,
naive/UTC microsecond timestamps, nested list/null and ordered struct values including
literal special field names. Exact expected tags/values, spec/manifest/source hashes and
SQLResult canonical reconstruction pass. actual-result.json retains the result and actual
receipt; result.canonical.json preserves exact SDK canonical bytes.

Caller FD remained open at offset 17 with unchanged access flags, file identity/mode/size/
mtime/ctime and content hash. Reads are not claimed to preserve access time. Parent SQL
imports remain lazy at package import; native DuckDB import attempts and loaded modules
are zero. No general no-LLM-dependency claim is made about the legacy dataset package.

Actual receipt reports wait, nonrunning PID0, container removal, helper/attach reap,
staging removal, exit zero and no OOM. Executor close completed. Selected image was
`sha256:05af70db10d7935a425621709a97b11607bfcb84370ec53b30444771770deb41`;
worker bundle `53bbc58ff31d5592666b97511fc010efb36e0f042a3a4b70ef75f48f9a632aac`.
Engine/image IDs are descriptive operational observations, not hostile-host attestation.
No running service remains from this query; operator setup artifacts/images are retained
as local setup evidence without an application/profile activation.

This is **actual macOS parent evidence for the portable-path successor**, including its
canonical Docker mount/readback. It is not a Linux end-to-end SDK claim. The owner's
separate tiny Linux stdlib tempdir probe is outside this review and was not duplicated.
No broad tests/native sweep were run. Predecessor macOS 697 evidence remains separate:
there is one actual query in each independently authorized operation, not one combined
run or a hidden automatic retry.

Input inventory remains caller-asserted; upstream_inventory_verified is false and scan
coverage is not measured. No source access grant, hosted tenancy/global quota, general
SQL purity, provider/model behavior, research/scientific gate, browser/MCP API or production
readiness is established. No real accounts, credentials, data, models or shared profile
were used or changed. No additional finding arose in this bounded successor case.
