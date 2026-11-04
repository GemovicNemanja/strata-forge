# Installed evaluation policy adapter handoff

`execute_evaluation` installs the fixed `EvaluationPolicyRPC` from the private authenticated
Task secret `STRATA_EVAL_AUTHORITY`. No spec, CLI option, environment variable, callback import
or unsigned caller JSON installs a grant. Missing authority remains closed. The adapter sends
exact Run/spec/typed-ref requests to the fixed internal namespace; server credential expiry,
active Run binding, fresh monotone custody and output authority remain mandatory server work.

Implementation: `src/strata_forge/pipelines/_evaluation_authority.py` and `eval_runner.py`.
Maintained synthetic exact wire fixture: `tests/unit/pipelines/fixtures/evaluation_policy_rpc.json`.
Reference: `docs/modules/pipelines.md#private-run-authority-and-publication`, ADR0021 and ADR0019.

The version-1 private frame has exactly `version`, `run_id`, `spec_sha256`, `policy_origin`,
`worker_credential`. Run IDs are 32 lowercase hex, spec digests 64 lowercase hex. Credentials
are `strata_worker_` plus 43 base64url characters. Operator origin is HTTPS or literal loopback
HTTP for local review, without userinfo/path/query/fragment. The server stores only hash/expiry
and derives actor/owner itself; credentials do not enter persisted/public specs or diagnostics.
The actual full stamped `wire_payload()` digest, including configs/splits/selection, is checked
before RPC. Every response echoes exact version/run/spec/domain/ref fields.

Fixed POST routes under `/api/internal/evaluation-policy` are `/read`, `/output/plan`,
`/output/ready`, `/output/receipt`. Requests use run Bearer only, no cookies/redirects or ambient
proxy credentials; bounded JSON and deadlines prevent arbitrary body/diagnostic growth.

Read permission never authorizes publication. Existing output requires fresh canonical,
private/unprotected metadata and expected parent. Prospective output is create-only with
`exist_ok=False`, no conflict fallback; server fresh readback supplies the actual post-create
parent before upload. Worker fresh privacy/parent checks and `parent_commit` enforce the ready
contract, then the actual `CommitRef` must be accepted by the separate receipt endpoint before
success. Output canonical/alias identity must not overlap input datasets. Unknown metadata,
positive custody on prospective names, cancellation/terminal/expiry, missing authority or changed
parents deny. Failed prospective creation/readback may leave an empty private repo; no cleanup
mutation is attempted. Local attempt status records known commits or unknown outcomes.

The server must install and independently exercise its resolver, active Run/output fences,
credential service and private Task transport before enabling evaluation launches. Synthetic
injected policy tests alone are insufficient. No real Hub writes, pretrained checkpoints or
research rows are needed for integration acceptance. Current M1 lineage/alias coverage remains
explicitly bounded; this adapter does not establish general sealed execution or external lineage.

Training checkpoint selection/final defaults, marked gate/test exclusion, the research owner
25 GPUh hard/20h hold budget and exact Ettin initialization remain TRAIN/operator scopes.
Evaluation never relaxes complete trained-head loading or runs the real research checkpoint.
