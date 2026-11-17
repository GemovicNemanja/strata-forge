# Classifier training authority: source contract for review

Status: proposal only, 2026-11-05. No source lease, route, issuer, schema revision,
credential, launch capability or research acceptance is activated by this packet.
The coordinator reviews this contract before assigning implementation. Claude Code
exclusively owns custom-model data, conversion, experiments and scientific decisions.

## Recommendation and verified baseline

Implement a purpose-separated **read-only classifier execution authority** next.
Keep the frozen `ClassifierTrainSpec` v1 bytes/digest and `TrainingReadAuthority`
interface intact. Do not widen EvalPolicyRPC, remove its `run.kind == eval` check,
or send an Eval credential to a classifier. Model publication is a later, separately
authorized operation with a versioned execution/output binding and receipt contract.

| Evidence | Exact source and finding |
| --- | --- |
| This docs branch | Canonical Forge `dev` `6871611fd283966fcd7c713ff4a69aa3ee11f7e7`, verified from GitHub on 2026-11-05; PR65 is merged documentation, not the worker runtime |
| Prior plan | [PR65 contract](classifier-worker-contract.md) and [review](classifier-worker-review.md), source head `0e4b32011cebd50b032d354deb07f8a30cfbd6f4`; its proposed output/adapter/merge fields must not be mistaken for accepted v1 fields |
| Frozen worker | Forge `4ae68684a94c412b8b45ca206e53e805dca0498c`; runtime also present in combined `71719b68248df3888b36b55076a25c40554160b5` |
| Inner spec at combined717 | `pipelines/classifier_spec.py:162-206`: strict v1 has run, engine, method/task, base, dataset, training and progress fields; **no output, adapter, merge or resume field**, even when null |
| Worker authority at combined717 | `pipelines/classifier_runner.py:56-127`: request/decision purpose `classifier_training`, exact run/ref/domain/**inner** spec digest; `_authorize` requires valid authorization and refuses missing/mismatched decisions, unknown custody and positive protection/restriction facts |
| Worker execution at combined717 | `classifier_runner.py:240-354`: repeated authority checks, safe local export/reload, bounded file inventory and receipt; `publication` and `resume` are `not_supported` |
| Installed source handshake | Forge `_common.py:485-535` supports exact `<version>+<full40sha>` against executing version, installed distribution metadata and PEP 610 commit; server `run_tasks.py:455-462` already requires an immutable ref for Eval, not classifier dispatch |
| Eval transport at combined717 | `pipelines/_evaluation_authority.py`: private Eval frame, fixed Eval routes, dataset publication DTOs and bounded retry; it is not a training adapter |
| Server audit pin | `strata-server` `429ad0c8bd39d76be5e8884b2ba85e820a0967ad`; the authority/custody files below are unchanged by its prompt ingress correction |
| Eval issuer/service | `worker_authority.py:22-28,106-175` uses Eval-specific HKDF/frame/hash metadata; `evaluation_policy.py:150-165` requires `kind=eval`; `:243-280` rechecks retained lineage and permits dataset publication only |
| Shared custody | `custody_run_policy.py:138-158,365-480` binds actual typed reads/coverage to the pending Run. `require_distinct_dataset_output:191-210` **does not protect model destinations** |
| Existing launch/ownership | `run_launch.py:104` prepares this custody path for Eval only. `models.py:415-445` has user-owned Runs and a nullable project pointer; that pointer is not collaboration, vault or compute authority |

The existing worker is a trusted Python runner with injected authority/source and
attempt directory. It does not supply the installed remote bootstrap, server launch
branch or concrete training policy adapter proposed here. These source observations
are not a trained-model result, clean VM acceptance or protected-data release gate.

## Smallest useful implementation unit

Build the following together behind an unavailable/default-off integration boundary:

1. A closed server-owned classifier execution binding, exact read admission, separate
   credential derivation/hash record and deterministic recovery for redaction.
2. A fixed private training-read service and RPC, using existing custody predicates
   without changing Eval behavior or granting publication.
3. A fixed installed Forge bootstrap/read adapter that validates the actual inner and
   outer digests and satisfies the existing `TrainingReadAuthority` protocol.
4. Synthetic issuer→actual ASGI RPC→actual adapter→worker boundary conformance, including
   the refusal/cancellation/cleanup schedules below. No model download is required.

Credential issuance must remain impossible until **trusted execution binding, live
policy checks and redaction integration all exist**. A pure derivation helper alone
does not justify an enabled call site. No caller `trusted`, `confirmed`, `sealed=false`,
`redaction_ready` or method/hyperparameter flag can satisfy these prerequisites.

This unit may be tested through trusted internal builders with fictional Runs. Public
classifier submission, fixed worker CLI dispatch, target spending and SSH activation
still need their own reviewed source leases/wire. Adding all `finetune` requests to
`PREPARED_READ_KINDS` is not an implementation: it would also change causal-LM clients.
The internal bootstrap/read-adapter unit registers no CLI dispatch entrypoint; its
conformance calls use only the explicitly trusted internal construction boundary.

## Two digests and a trusted execution binding

Use a separately versioned `ClassifierExecution` envelope. Proposed read-only v1:

| Field | Proposed meaning |
| --- | --- |
| `schema_version` | Exact integer 1; strict/closed/frozen |
| `purpose` | Literal `classifier_training` |
| `run_id` | Actual server-assigned 32-hex Run ID, equal to the inner spec |
| `attempt_id` | Server-owned 32-hex execution identity; durable before delivery, never a caller path or restart hint |
| `worker_spec` | The complete existing `ClassifierTrainSpec` v1, stamped/validated by a fixed classifier builder |
| `output` | Required literal null in this envelope version; **not a field added to the inner spec** |

The inner digest is exactly current `ClassifierTrainSpec.digest()`: SHA-256 of its
existing sorted-key compact UTF-8 canonical JSON, including its current defaults.
Do not apply EvalSpec's serialization choices or omit new defaults silently. The outer
digest is SHA-256 of the complete typed execution envelope using an explicitly tested
sorted-key compact UTF-8 JSON encoding, no NaN and no secret frame/hash record. The
outer envelope embeds the complete inner object, not a caller's asserted inner hash.

The server stores/revalidates the canonical execution and both digests under a new
reserved classifier binding. The bootstrap recomputes **both from the actual installed
execution/spec**, checks run/engine/attempt/output, then constructs the read adapter.
An outer hash can never be passed as `TrainingReadRequest.spec_sha256`: the existing
worker compares that field with the inner digest. Changing dataset config/split/map/
selection, rendering, labels, initialization or any executed setting changes the
appropriate digests; changing the outer attempt changes the outer digest too.

`kind=finetune` and `method=seq_cls` are necessary labels in a future launch branch,
not proof of a trusted execution. The binding is created only from a fixed validated
classifier command by the server builder, under the admission transaction. Ordinary
causal RunParams, arbitrary hyperparameters, a nested lookalike JSON object or a public
project selector cannot mint it. A Run with missing/malformed/mismatched binding denies
training authority. Existing causal runs and all Eval digests/wire remain unchanged.

The future remote builder must stamp the **existing inner `engine_version` field** as
exact `<version>+<full40sha>` from an immutable server-selected Forge source ref. The
fixed bootstrap must enforce `check_engine_version` against the executing version,
installed distribution metadata and PEP 610 commit before authority/source access. A
version-only claim, editable install or source-overlay fixture cannot establish that
remote source identity. Reuse this handshake rather than add a caller-controlled worker
source hash or change inner v1's canonicalization. The declared engine value is itself
included in the inner and outer digests; record the actual resolved dependency versions
as execution evidence rather than claiming the source commit locks all dependencies.

The initial boundary is user-owned. Derive the owner from the actual Run, never a
worker/caller owner ID. If a later launch binds a project, verify the exact stored
project and current applicable grants through the existing lock policy; do not infer
membership from names/output repos or adopt historical null-project Runs. Project
membership does not borrow another user's compute credential or vault. A nullable
project column alone does not implement project-shared execution.

## Proposed private capability

The names below are candidates for coordinator wire review, not installed fields:

| Element | Proposed closed value |
| --- | --- |
| Task secret key | `STRATA_CLASSIFIER_AUTHORITY`, separate from `STRATA_EVAL_AUTHORITY` |
| Credential spelling | `strata_classifier_` followed by exactly 43 base64url characters (256-bit PRF output) |
| HKDF info | `strata.classifier-training.worker-policy.key.v1`; separate from Eval/encryption/signing domains |
| Reserved binding | `_classifier_execution_binding`: server-owned canonical execution, inner/outer digests and exact read approval references |
| Hash-only record | `_classifier_policy_credential`: exact `{version:1,purpose:classifier_training,sha256,expires_at}` |
| Original origin | `_classifier_policy_origin`: immutable trusted operator-origin metadata, not a request-selected URL |

The masked version-1 private frame contains exactly `version`, `purpose`, `run_id`,
`spec_sha256` (inner), `execution_sha256` (outer), `policy_origin`, `worker_credential`.
No owner, grant, secret repository token or publisher permission enters the frame.

Derive a 32-byte key with HKDF-SHA256 from the existing validated master ring and the
classifier-only info label. HMAC-SHA256 binds a canonical versioned context containing
purpose, actual owner/run/attempt, both digests, UTC expiry and original canonical
policy origin; encode its output without base64 padding after the classifier prefix.
This follows the existing derivation approach while retaining a separate domain and
binding the new context explicitly. Do not change Eval derivation or its metadata.

Proposed expiry is the existing 24-hour ceiling, fixed at issuance rather than sliding
on requests/recovery. Longer execution needs a separately reviewed renewal protocol;
this packet supplies none. Only hash/expiry/purpose metadata persists, never the raw
credential/frame. Recovery tries retained ring keys and constant-time checks the stored
hash using the exact frozen context, including after expiry for redaction; it does not
renew authority. Dropping the matching key fails closed rather than emitting unsafe logs.

Before issuance/delivery, add the new shape to server/Forge redactors and the app's
defence-in-depth scrub, and a distinct injected-secret kind such as
`worker:classifier-policy`. Poller recovery must select only a validated classifier
binding and scrub raw/encoded/frame/fragment variants before caps. Do not infer this
secret kind from arbitrary fine-tune hyperparameters. Test rotation, restart and missing
ring keys. Secret delivery uses the private Task file channel, not spec/env/argv, a
public RunView, HTTP issuer, cookie exchange or generic agent/service capability.

Operator-origin configuration is explicit and unavailable when empty; choose its
exact configuration key during source review. Canonical HTTPS only, with explicitly
allowed literal loopback HTTP in development; no userinfo/path/query/fragment, redirects
or environment-proxy inheritance. Recovery uses the original stored origin, not a newly
changed operator value. No data/card/Hub redirect may select a policy host.

The proposed JSON records use reserved Run configuration, **not an approved migration**.
Review persistence/recovery bounds and legacy-null behavior before code. If a dedicated
execution/attempt/grant table or project adoption is needed, obtain an additive migration
lease in the current single Alembic chain; this proposal reserves no revision or schema.

## Fixed training-read RPC and adapter

Proposed sole route: `POST /api/internal/classifier-policy/read`, bearer-authenticated
only by the classifier credential. No cookie/HF/provider/Eval-token fallback. There are
no output methods in this unit. Reuse bounded HTTP mechanics, not EvalPolicyRPC's
purpose, DTOs, issuer or dataset publication service.

| Proposed request field | Required check |
| --- | --- |
| `version`, `purpose` | Exact integer 1 and literal `classifier_training` |
| `run_id`, `execution_sha256`, `spec_sha256` | Actual Run and the two independently verified stored digests |
| `role` | `base_model` or `training_dataset`, derived by the adapter from the frozen spec and checked by the server |
| `repo_type`, `ref` | Exact domain and full immutable `namespace/repo@40hex`, matching that role's requested read |

The successful response echoes those exact fields plus `authorized:true`,
`custody:unsealed`, `policy_version:custody-v1`. Strict types/extra-forbid, duplicate-key
refusal, fixed safe errors, bounded input/output and an aggregate deadline are mandatory.
Proposed bounds follow current private policy transport: 4 KiB uncompressed JSON request,
2-second body receipt, 16 KiB response and 30-second aggregate worker call. Verify ingress
before framework/SDK extraction with active in-memory Sentry, and preserve ordinary
route instrumentation. No raw refs/specs/credentials/source in error telemetry.

The worker's unchanged request has no role or outer hash. The installed adapter maps
`model + exact spec.base.ref` to `base_model`, and `dataset + exact spec.dataset.ref`
to `training_dataset`; everything else denies. It adds the outer binding from the
validated private frame. After validating every response echo, it removes only this
transport wrapper and constructs the existing `TrainingReadDecision` with the original
**inner** digest/purpose/ref/domain. No broad repo allowance, same-name cross-domain
substitution, arbitrary resolver/import, cached negative result or output grant exists.

The service must, before a decision:

1. Validate the actual Run's classifier binding, credential hash/purpose/expiry, exact
   attempt/engine/digests and pending/running state. Cancelled/terminal/superseded denies.
   Do not select an owner or project from the request.
2. Lock the owner/account and a freshly owner-filtered Run in the established order;
   when real project policy is involved, preserve sorted Users→Projects→Run ordering.
   Recheck binding, state, expiry and applicable grants after lock/metadata waits.
3. Apply fresh fixed metadata/current+pinned identity, canonical aliases, local registry/
   portable markers and known Strata producer lineage through the shared custody seam.
   Metadata credentials are actor/Run-owner-bound in this user-owned slice; no project
   role borrows another profile's secret. Unknown/inaccessible/positive/mismatched facts deny.
4. Recheck **every frozen lineage ref**, even after producer deletion, and deny newly
   discovered parents outside the frozen closure. Lineage refs fence/provenance only;
   they never expand the two direct read grants. Preserve provenance retention/delete guards.
5. Recheck live authority after awaited work before committing/replying. Admission must
   authorize and register the exact typed reads while holding the same custody account
   mutex through Run persistence, so sealing cannot race the authorization/registration gap.

The first implementation needs a classifier-specific prepared-read selector. It must not
enable prepared reads for every causal fine-tune or loosen the existing Eval selector.
Preserve current coverage qualifications: known current Hub/registry/Strata facts, incomplete
historical aliases and unproven external lineage, no cross-system HF/SQL transaction and no
retroactive erasure. This ordinary unsealed policy is not protected-research/annotation authority.

## Cancellation, retries and local evidence

Current `_authorize` checks both direct reads before rows, after rows, after awaited
checkpoint resolution before loading, before native training, after training and after
export inventory before the local receipt. Preserve those strict boundaries and deny an
absent adapter. Read approval is a fresh point-in-time decision plus the registered live
fence, not permission to ignore later cancellation or portable metadata changes.

Proposed RPC retries are bounded identical read requests for transient 503 and transport
timeouts before a complete response, with at most two backoffs inside one aggregate
deadline. Never retry 401/403/409/422, malformed/oversized/compressed responses, binding
mismatch or cancellation. Each attempt obtains a fresh decision; no sliding credential
expiry. HTTP retry must not restart training, choose a new Run/attempt or consume different rows.

Coroutine cancellation requests stop and never emits a later success; native threads can
outlive it. Keep the registered read fence through terminal state **until process death
and cleanup are positively confirmed**. A missing/ambiguous launch result, unreconciled
PendingCleanup or timeout is not proof that work stopped. Do not release custody because
a future cancelled Run appears terminal, because the worker coroutine returned, or because
its credential expired. A rerun uses a newly authorized Run/attempt; no resume is invented.

Read-only execution accepts no output request. Keep the actual verified local artifact and
training-receipt hashes under a fixed trusted attempt layout; no caller path/remote arbitrary
file read or inferred report quality. Successful process exit alone is not a model publication
or platform artifact. The bootstrap/attempt/poller contract still needs review before launch.

## Model output: separate contract gate

Publication needs a separately versioned outer execution/output envelope; do not add an
optional field to frozen inner v1 or reinterpret its digest. An explicit model destination
changes the outer binding. A separate publication-scoped authority/grant is required: the
read credential defined here remains read-only even when later publication code exists.

Before assigning that unit, freeze all of:

- Typed model destination, existing-private/create-only modes, current canonical/known-alias
  protection checks, writer/output live fence and actual expected parent. Ambiguous 404 is
  not absence; creation is atomic `exist_ok=False`, no conflict fallback/overwrite, followed
  by fresh private/protection/identity readback. Positive/unknown destinations refuse.
- Model-domain collision checks against the base model and every retained model/evidence
  parent through requested/canonical/known-alias identities, case-insensitively. The existing
  dataset-only helper is insufficient. Read grants and lineage refs confer no write authority.
- A strict local export receipt/inventory from the actual safe reloaded classifier, including
  inner/outer/run/attempt/engine, training-receipt SHA, manifest/file SHA/size bindings and
  renderer/label-token provenance. No optimizer, unsafe checkpoint or arbitrary extra file.
- A closed publication receipt with `repo_type:model`, requested/canonical destination,
  existing/create-only mode, ready parent (40 hex), actual resulting commit (40 hex), run/attempt,
  both digests, local training-receipt and artifact-manifest SHA256, and policy revision.
  The authenticated worker's actual Hub CommitRef is required; config/stdout/progress/private
  flags or repository existence are not substitutes. Define immutable replay/conflict behavior.
- Exact bounded Hub observations and their limits: metadata can verify identity/protection
  and pinned commits but does not independently attest training quality, every uploaded byte
  or upload parentage. Specify what worker/Hub evidence establishes each receipt assertion;
  do not claim independent server weight verification or scientific attestation.
- Poller/result projection consumes only an accepted model receipt, preserves it after later
  failure/cancellation, and refuses requested-publication success without it. Retain producer
  lineage so Run deletion cannot erase origin. Native artifact adoption also needs the actual
  project/resource binding; no inferred sharing from equal owner/output name.

Until that contract and both consumers pass, no model push method/endpoint is enabled and
local-only training is not advertised as a complete Hub-ref-only train→Eval workflow.
Protected destination/derived-data release requires its separate custody/research protocol;
an ordinary private model repo is not a declassification mechanism.

## Proposed leases and acceptance before activation

| Unit / owner to assign | Narrow source scope after review |
| --- | --- |
| Server authority | New classifier-only binding/credential/service/schema modules; explicit reserved config and redaction/secret-recovery consumers; no Eval-kind widening |
| Forge adapter/bootstrap | New fixed private classifier authority adapter/bootstrap helper plus matching tests, without CLI dispatch registration; existing inner spec/digest and core training/initialization semantics unchanged |
| Server admission/dispatch | Classifier-specific typed builder/selector and private Task delivery, then explicit Run/result projection; preserve causal/Eval paths and user-owned compute |
| Model publisher | Separate reviewed output envelope, model fence/collision/receipt and worker+poller consumers; not included in the read-only authority unit |

No shared source file, schema revision or public DTO is reserved by this proposal. A later
implementation lease must enumerate config/main/routes/launch/poller/redaction/migrations
before edits and keep the capability unavailable until prerequisites are integrated.

Required synthetic evidence for the first unit:

- Golden frozen inner digests unchanged; allowed attempt/inner changes alter the outer
  digest, while any non-null output refuses in this envelope version. Future versioned
  output tests own output mutations. Actual parsed inner/outer mismatch denies before
  source/model/row access; version-only/mismatched installed source refuses remote launch.
- Eval/causal/browser/HF/provider credentials rejected; ordinary `finetune` plus lookalike
  flags/hyperparameters cannot mint/recover classifier authority. Wrong owner/run/attempt/
  purpose/engine/role/domain/ref, expiry, terminal/cancelled state and missing binding deny.
- Fixed issuer→real ASGI service→actual adapter echoes the inner digest into the existing
  worker protocol while validating the outer digest separately. No actual Hub/VM/model read.
- Fresh grant/session launch checks where applicable; real disposable-PG lock-wait expiry/
  cancellation, seal-vs-registration serialization, retained-parent/new-parent/alias tests,
  unknown cleanup and process-lifetime fences. No configured account database.
- Transient retries preserve exact identity/body/deadline; permanent/malformed/cancelled
  refusals never retry. Native cancellation has no late success or early fence release.
- Hash-only restart/rotation recovery, missing-key fail-closed, no raw frame in projections
  or logs, full/encoded/fragment scrub before truncation, default-off refusal before probes/
  persistence/SSH, and current causal/Eval suites unchanged.

Remaining decisions for coordinator review: exact execution/record/frame/RPC names above,
durable attempt storage/recovery and any migration need, origin configuration, first launcher
source leases and bounded local result projection. Weighting, smoothing, grouped binary
objectives, support/inconsistency heads, calibration/selection, row conversions and research
population choices are **not answered here**. Unsupported knobs remain refused; ordinary CE
and the already named preset/final-only worker do not resolve those scientific questions.
