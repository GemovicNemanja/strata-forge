# Agent rules — strata_forge.pipelines

Owns inert compute-target worker specs and orchestration. Imports may compose storage,
compute, training, metrics and LLM primitives; pure metrics never import workers. Read
`docs/modules/pipelines.md` and ADR0021 before changing the evaluation boundary.

- Specs reject extra fields. Fixed installed commands consume data, never executable imports,
  callbacks, shell strings or repo-supplied code. Preserve the engine-version handshake.
- `evaluation_spec.EvalSpec` is the sole classification-evaluation schema, including semantic
  validators beyond JSON Schema. Consumers import it; they do not independently guess a mirror.
- Evaluation authorizes every exact ref through a trusted `ReadPolicy` before preparing models
  or materializing rows. Missing, unknown, denied or positive custody refuses; caller flags,
  Hub readability and cached bytes are not authorization. Standalone execution denies without
  installed trusted policy. No spec-controlled policy import or permissive fallback.
- Native model scoring reuses `_scoring.load_scorer` and its complete-checkpoint guard. No
  implicit fresh head, adapter-base discovery, remote code, pickle fallback or hypothesis cut.
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
