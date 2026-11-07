# Isolated local SQL SDK handoff

Runtime is frozen at `41b9ae61735e68fd618f9689c874533b4e2132b2`, based on
`6c1dca30786ecbf7a33f4d2204af53bead7b65f4`. The fixed worker is byte-identical to
checkpoint `8c5c21de55bb70f9e73e5facecf52e35fb854084`. All commits use the personal author.
No remote write, release, current preview, root dependency, schema, server/MCP mount or job
registry was changed. The exact public surface and limits are in
[the module reference](../../../docs/modules/headless-sql.md).

`LocalSQLExecutor.query` owns caller-FD duplication, private staging, the fixed-profile
container, typed result verification and actual wait/PID0/removal before its immutable
`SQLResult` receipt. One active/no queue is **per instance on one captured loop**, not a
process/organization quota. Native SQL never executes in the parent. Origin assertions
and hashes are descriptive; no source authorization, custody attestation or upstream
inventory resolution is implemented. Host/same-UID/Docker-operator isolation, hard OS reap
bounds, cross-platform parity and durable crash recovery are not claimed.

The source-qualified evidence is in [manifest.json](manifest.json): all 285 dataset tests
pass on 4e; 25 actual installed-image SDK controls pass in 20.45 seconds on 4e; the final
697 successor passes 31 parent/fault controls. The successor only latches retirement when
a pre-create engine identity mismatch was observed. It does not change the worker or
normal started cleanup. The portable successor uses fixed Unix `/tmp` followed by canonical resolution, preserving
macOS Docker mount/readback. It passes 32 parent controls. An independent actual
eight-column installed-SDK query also passes on that exact source and new wheel; see
[the source-qualified peer handoff](portable-peer-handoff.md). This is actual macOS parent
evidence, not a Linux end-to-end SDK claim.
10 owned files pass strict typing with the existing larger read-only SDK environment;
the final narrow guard passes its owned strict checks. Ruff/format/diff checks are clean.
No broad Forge test or optional-package installation is presented as SQL acceptance.

The installed local wheel is version 0.3.0 solely for packaging verification, not a release.
Its hash, size and six SQL runtime file hashes are recorded; every installed file equals
41b source. The new portable wheel SHA256 is
`f720e5a9186c2b8cf80e486e600f1d7876587be6d4aa370e19b625ee59ee134e`;
the old wheel and target remain separate historical evidence. A base-only Python 3.14.4 cold/lazy import and pure-constructor proof has
DuckDB/Torch/NumPy absent, no parent native imports and zero socket attempts. Build uses
an isolated public Hatchling cache; no existing environment is mutated. Image Python is
separately pinned 3.14.5 with hash-pinned DuckDB2.0/pytz2026.6 wheels and retained notices.

Run actual tests only after explicit operator setup, in the same owned tool operation:

```sh
python containers/headless-sql/setup.py --docker /usr/local/bin/docker \
  --engine-socket /Users/nemanjagemovic/.docker/run/docker.sock \
  --artifact-cache /private/tmp/strata-headless-sql-owned-profile-20261107
# Use the returned installation-file path; query never calls setup/build/pull.
STRATA_TEST_SQL_INSTALLATION=/path/from/setup LITELLM_LOCAL_MODEL_COST_MAP=True \
  PYTHONPATH=src python -B -m pytest --confcutdir=tests/integration/datasets \
  tests/integration/datasets/test_headless_sql.py -q
```

A missing installation is an explicit prerequisite failure when this integration suite is
selected, never a silent pass. The local image cache unexpectedly disappeared between
several calls; no prune/context/account action was taken and the cause remains uninvestigated.
Each `--pull=never` miss refused; public setup is separate, never a query fallback. Use only
the selected explicit socket, fixed hashes and exact owned containers; do not inspect or
clean unrelated Docker resources.

Retained negative evidence includes the original frame-latch and failed-exit witnesses,
pre-create retirement RED, initial native missing-pytz/logging refusals and initial actual
matrix. Its two failures were harness expectations (list `items` versus accepted `values`,
and oversized text's existing `invalid_spec` code); assertions were corrected to the
accepted wire. A valid result followed by failed exit was a real defect, fixed in 4e with
an unchanged root witness plus actual separately labelled stdlib fault-child control.
A pre-create engine mismatch restoring admission was a real policy defect, fixed in 697;
no worker had executed in that RED. Both are distinct from actual normal worker evidence.
The initial root frame probe had one blocked LiteLLM cache attempt; its corrected guarded
RED and successor proof had zero. No zero-attempt claim covers that initial probe.


The optional Linux stdlib-only layout probe could not start because its explicit image/
engine create attempt returned no ID. No start or SQL was issued, and no Linux execution
is claimed. The test-only helper did not persist its generated name before that refusal,
so absence of a possible unstarted/no-mount residue is not independently proved. No cleanup
claim or broad Docker-state inspection was substituted. This limitation belongs to that
optional test attempt, not to the SDK, which retains exact names and ownership on uncertain
create. See [its qualified record](linux-stdlib-unavailable.json). No further Docker work
was performed after the independent portable SDK case.
