# Agent rules — strata_forge.pipelines

Owns inert compute-target worker specs and orchestration. Imports may compose storage,
compute, training, metrics and LLM primitives; pure metrics never import workers. Read
`docs/modules/pipelines.md` and ADR0021 before changing the evaluation boundary.

- Generation uses a separate exact-pin schema and finite controls, with stable item/sample
  identity and explicit seed semantics. Sample planning grants no source/native capability.
  Keep lexical JSON validation, full-selection caps and legacy serialized contracts separate;
  no CLI/registry/process activation follows from the data types.
- Specs reject extra fields. Fixed installed commands consume data, never executable imports,
  callbacks, shell strings or repo-supplied code. Preserve the engine-version handshake.
- Primary classifier specs are separate from registry activation. Require an installed
  training-purpose authority and streamed row bytes; enforce count/byte/range bounds BEFORE
  parsing later rows. No Eval credential or environment policy fallback. Keep prototype bounds
  distinct from scientific populations and preserve final-only/no-adapter/no-publication scope.
- `classifier_worker` uses the unchanged primary spec and its own private
  `FORGE_CLASSIFIER_AUTHORITY` frame, never evaluation credentials. Preflight the installed
  engine before the canonical secret loader; rely on the backend outer shell for unread-file
  cleanup. Fixed preinstalled commands use `-I -B` and no job-time setup/install. The HF pilot's
  bounds and worker-local safe outputs are documented in `docs/modules/classifier-worker.md`;
  they do not establish native publication, resume or large-population readiness.
- Recognized pinned Forge classifiers reload complete SDK state. Stored single-sequence render
  metadata must match original/effective/SDK digests, labels, family and length; stored tokens
  are initialization provenance, never label-logit evaluation parameters. Do not loosen native
  paired behavior or accept malformed sidecars through fallback. Cancellation cannot let a
  later native thread emit final worker success; transport termination/fences remain separate.
- Preflight exact-pin metadata sizes before download and check actual regular-file sizes before
  bounded local copies. Unsupported/unknown preflight refuses without unbounded fallback. Count
  every lazy-tree file/folder and stop at cap+1 before later pages/assets; do not eagerly exhaust
  a repository listing. Distinguish advertised metadata limits from hard HTTP wire-byte limits.
- `evaluation_spec.EvalSpec` is the sole classification-evaluation task schema, including semantic
  validators beyond JSON Schema. Population-v2 operational envelopes wrap it rather than mirror
  it; they preserve nested presence and bind the complete envelope digest. Legacy v1 serialization
  and entrypoint defaults stay separate. Consumers import schemas rather than guess a mirror.
- Evaluation authorizes every exact ref through a trusted `ReadPolicy` before preparing models
  or materializing rows. Missing, unknown, denied or positive custody refuses; caller flags,
  Hub readability and cached bytes are not authorization. Standalone execution denies without
  installed trusted policy. No spec-controlled policy import or permissive fallback.
- Native model scoring reuses `_scoring.load_scorer` and its complete-checkpoint guard. No
  implicit fresh head, adapter-base discovery, remote code, pickle fallback or hypothesis cut.
- The loaded scorer's explicit per-call error overflow policy refuses overlength calls before
  model forward, disables pair/single truncation and verifies complete right-padded masks.
  Legacy default/spec/worker policy remains separate; no dropped-item semantics are inferred.
- Population-v2 execution uses native score_with_outcomes: full token evidence, no cut input,
  suite exclusions before render/tokenization and exact N=E+D+S+F/M=S+F counts. Ordinary invalid
  outputs remain wrong; valid independent probability vectors survive invalid hard labels.
  Imported flags/manifests never self-grant a verified drop. Model/tokenizer/rendering conditions
  may differ in comparisons; exact gold/source/M identities and compatible targets must agree.
- Population wrappers revalidate a serialized copy before every operation. Read policy binds
  the OUTER digest before source/model IO; raw outcome/count DTOs are not authorization or
  native execution evidence. Keep CLI/registry/server/issuer/publication activation separate.
  Local population attempts require an empty current-UID 0700 leaf; never chmod/take over an
  existing caller directory. Owned new directories/files are 0700/0600 independent of umask,
  including temporary metric Parquet. Existing parents remain caller-managed POSIX paths.
  Pending writes remain incomplete after cancellation; final manifests move only after awaited
  writes and an active owner-task check. Native thread termination is not claimed.
- The verifier renderer is a declared bounded premise/hypothesis shape, not a universal task
  constraint. Its config lives in `training.verifier_config`; it performs no I/O.
- Stable item IDs are exact unique strings. Inference keeps legacy `custom_id` and optionally
  emits `item_id` plus explicit source ID semantics. Ordinals are not an automatic stable join.
- Per-config populations, ordered labels/projections, invalid-as-wrong counts and probability
  family/coverage survive reports. No implicit pooling, one-hot repair or malformed-vector
  normalization. Restricted softmax is not calibrated full-class probability.
- Bound reads, JSON, row populations, slices and artifacts. Refuse an exceeded bound; never
  silently drop rows/slices. Parquet fields are explicitly typed even when the first row refuses.
- Publication is unavailable without separate trusted destination authority/fence. Read approval
  or private=True never grants writes. Preserve the fixed private authority RPC, exact canonical
  identity, fresh private/protection state, atomic create-only refusal and expected-parent commit.
  Accept the actual CommitRef receipt before success; retain unknown outcomes on interruption.
  Never invent a SHA or overwrite local attempt evidence. No spec-controlled policy import,
  environment/argv authority, redirect/cookie authentication or credential diagnostics.
  Policy RPC retries are limited to identical requests after HTTP 503 or transport timeouts,
  at most three attempts inside one 30-second budget; never retry denials or Hub mutations.
- Coroutine cancellation does not terminate a model thread. Supervised process-group teardown
  remains required. Local attempt completion is not an immutable publication receipt.
- Tests use synthetic rows/fake policy/model/Hub plus real local Parquet; no pretrained downloads.
  Keep source/module docs, clean-install constructors and meaningful regressions in the same change.

## Local classifier benchmark SDK

`benchmark_spec.py` and `classifier_benchmark.py` own bounded CPU/FP32/concurrency-one local
measurements over resolved objects. Lazy package exports are not worker/CLI registration.
Preserve the exact boundary name, call/row counts, warmup exclusion, nearest-rank method and
both rate denominators. No successful headline after overflow/drop/truncation/missing output,
changed workload, invalid clock, deadline or failure. Current native family/layout checks are
benchmark admission only and never change scorer support or defaults. Native calls may overrun
a soft budget; shield/drain cancellation and hold known alias claims until cleanup finishes.
Do not add a serialized callable/import/factory, loader, publication or authority fallback.
Private clock/operation fakes are test seams only. ADR0028 documents the pure-value exception.

## Registered local temperature fitting

The separate temperature-fit schema/SDK admits only raw exact logits, identity projection
and complete registered named sources. Missing/unknown policy refuses; ordinary read access
is not fit eligibility. Never fit ho_dev or aliases; preserve canonical typed refs, roles,
registration revision, exact count/digest and current checks before rows/through final return.
Adapters are trusted Python code, never serialized grants/imports. No default issuer/worker,
model call, file output, bootstrap placeholder or verified-application claim. Owned reads/math
are shielded/drained under repeated cancellation; cleanup faults cannot replace cancellation
or emit late success. Bound all inputs/work/receipts and keep raw diagnostics out of errors.

## Registered local paired cluster uncertainty

Keep the separate cluster-analysis schema/SDK independent of fit and evaluation-worker schemas.
Both explicit trusted source objects need current typed unsealed resolution and complete paired
population registration before NumPy/rows. Exact M IDs/gold/groups, complete probability-family
compatibility and source counts/digests survive every check. No caller JSON grants, intersections,
implicit projection, raw-row receipts, issuer, remote-purpose or artifact/worker activation.
Shield/drain repeated cancellation, close streams and recheck current policy through final return;
no partial10000-run, late success, hard thread timeout, independence or gate claim. ADR0031 owns
the fixed numerical profile and source-codec pure-value exception.

## Private native immutable input reader

`_immutable_input.py` and `immutable_input_types.py` provide a private injected byte/held-FD
port, not a worker, authority producer or native consumer. Keep complete request/manifest
binding, exact supported purposes, actual caller-task current rights/custody checks and
the original deadline through final disclosure. Operational children carry no policy
proof and remain owned through actual resolver/thread/close completion. Never replace
current policy with a digest, URL, caller flag or prior result; never retry an uncertain
FD number after close. Source leases must own delayed resolution before it begins and
provide idempotent cleanup. Preserve the bounded profile and qualification documented in
[`docs/modules/immutable-inputs.md`](../../../docs/modules/immutable-inputs.md).

The private human prompt consumer uses the separate `human_render` admission variant,
never operation IDs in durable-run fields. It stages verified bytes through the existing
prompt consumer around the Server's actual retained dependency-pin validation. The
original caller, exact frozen intent and original deadline survive both stages; the
Server must recheck current authority after rendering and physical cleanup. Descriptive
source metadata, local candidates and live-operation receipts grant no native access or
durable execution. See [`docs/modules/native-prompt-inputs.md`](../../../docs/modules/native-prompt-inputs.md).
