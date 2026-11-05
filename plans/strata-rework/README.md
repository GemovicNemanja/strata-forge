# Forge execution plan for the Strata rework

Internal execution handoff, prepared 2026-11-03 for agents starting 2026-11-04. This is a
plan and evidence ledger, not a list of shipped features, a public roadmap or a release promise.

Execution scope confirmed 2026-11-04: platform infrastructure and redesign are assigned to Codex;
all MODEL/custom-model data and experiments are reserved for Claude Code. Synthetic offline
executor tests are platform work. The next product review milestone is a working local stack,
before production promotion.

## Authority, ownership and starting state

Read the repo's `AGENTS.md` / `CLAUDE.md` and the rules in every touched module first. Then
read the [central plan](../../../strata-app/plans/strata-rework/README.md),
[coordination rules](../../../strata-app/plans/strata-rework/coordination.md),
[task registry](../../../strata-app/plans/strata-rework/tasks.json), and
[shared contracts](../../../strata-app/plans/strata-rework/contracts.md). The registry is the
single cross-repo task/ownership authority; do not create a competing copy here.

The audit source for implementation work is Forge `origin/dev`
`28a5bcda9a3e013509ca8763f7699d55192a0db7`, checked out in
`/private/tmp/strata-scoring-foundation-20261103`. The user's original checkout remains on
`docs/public-oss-readiness` at `1386bd0aea009024e47664d8990ef14a81b212c2`; its locally stored
`origin/dev` is `d7a7c017a4881d0aca71f9b566dcdceef3949f68`. These different refs are evidence
of stale local state, not permission to overwrite another agent's branch. `PLAT-001` must
refresh refs and identify the actual integration base before agents branch tomorrow. No
implementation in this handoff has been committed, pushed, merged, released or deployed.

One isolated checkout and one owner per cohesive unit. Claim task IDs and shared files before
editing, especially `pipelines/inference_runner.py`, `pipelines/finetune_runner.py`,
`pyproject.toml`, `training/methods.py`, `training/dataset_format.py`, and
`training/progress.py`. Prepare a reviewable patch, tests, documentation and evidence before
requesting merge sign-off. Follow the repo's branch/release rules and the coordinator's
branch naming decision; no direct commits to `dev` or `main`.

### Research specification precedence

The verifier's signed work is not replaced by this platform plan. On this machine, read:

1. `~/.claude/plans/logic-verifier/decisions.md`, including all later addenda.
2. `final.md`, then `insert-jev-clm.md`, only where not overridden by those decisions.
3. `impl/outline.md`, `impl/cross.md`, and the relevant `impl/phase-P*.md` reconciliation.
4. `impl/spec-forge.md` and the matching server/app specifications.

The phase documents resolve several real gaps in the older Forge specification: phase-P4
puts shared metric math in a leaf `metrics` module rather than creating a training→evals
dependency; phase-P5 adds the complete support/inconsistency/abstention outputs and reconciles
their knob names. Preserve those decisions. Copying an older isolated paragraph is unsafe.
Record any remaining contradiction in the central decision ledger and resolve it before
dependent implementation; do not silently invent a third schema.

The latest user preference is downloadable verifier weights with commercial licensing.
That changes distribution planning; it does not clear a source for training or remove the
signed strict-deduction semantics, contamination checks, sealed-data rules, research gates
or spending limits. Tasksource checkpoints are comparison baselines, never released-model
initializations. No agent working on this platform plan may inspect sealed/fresh rows,
regenerate research data, resume Claude experiments or launch paid work by inference.

Older research documents say “staging.” The deployment checkpoint retired hosted staging.
Use the coordinator-approved locally run server plus isolated execution target until a
replacement is explicitly provisioned. A successful unit test is not a staging/VM result.

## Work queue and dependency order

| ID | State at handoff | Outcome | Prerequisites |
| --- | --- | --- | --- |
| `INFER-001` | Draft patch; offline checks passed; unmerged | Reusable exact HF scoring foundation | `PLAT-001` review/base reconciliation |
| `INFER-002` | Planned | Scoring engines and reproducible inference are launchable and persisted | `INFER-001`; approved wire design; security/sealed prerequisites |
| `EVAL-001` | Planned; can design in parallel | Shared, well-defined classification/statistical metrics | `PLAT-001`; freeze numeric and probability contract |
| `EVAL-002` | Planned | Evaluation runner, artifact outputs and registered grader composition | `INFER-002`, `EVAL-001`; approved additive current-run/HF artifact contract; custom-code worker extension uses `PLAT-012` |
| `TRAIN-001` | Planned | Full verifier training and calibrated v1 output contract | `INFER-002`, `EVAL-001/002`; research-owner handoff and accepted data schema |
| `PROMPT-001` | Planned | Authoritative modular-prompt compiler and cross-language conformance | `PLAT-001` |
| `PROMPT-004` | Planned | SDK build consumption, rendering and execution capture | `PROMPT-001`, `PLAT-011` |
| `PLAT-012` | Planned contract work | Named, pinned custom workers on authorized infrastructure | project/artifact/job contracts; operator trust decision |
| `TRAIN-002` | Planned; split into small capability additions | Broader training methods, optimizations and multi-GPU execution | `TRAIN-001`, `PLAT-012`; research gates where applicable |
| `INFER-003` | Planned | Serve and batch custom model artifacts with tested quantization paths | `INFER-002`, `PLAT-012`, `COMPUTE-001`; serving lifecycle contract |
| `DATA-002` | Planned | Reproducible dataset recipes and annotation/adjudication engine | `DATA-001`, `EVAL-002`; research posture rules |

This is ordering, not a calendar estimate. Pure metrics can proceed while inference integration
is reviewed. `TRAIN-001` can split the format, eval-default fix, classifier, calibration and
artifact changes into sequential reviews; four touch the same runner, so they are not four
independent agents. G1 requires accepted INFER-002, EVAL-001 and EVAL-002: every baseline/scoring/evaluation
run uses Strata. The former E0/E0+ execution alternative is withdrawn. A negative research gate changes experiments and claims; it does not
cancel useful pre-signal platform work already accepted by the user.

## INFER-001 — review and retain the exact scoring foundation

**Evidence and files.** Working checkout `/private/tmp/strata-scoring-foundation-20261103`,
branch `codex/scoring-foundation`, base `28a5bcda9a3e013509ca8763f7699d55192a0db7`.
Portable draft patch `/private/tmp/strata-scoring-foundation-20261103.patch`:

- `src/strata_forge/pipelines/_scoring.py` — internal typed loader/scorer, 477 lines.
- `tests/unit/pipelines/test_scoring.py` — fake tokenizer/tensor/model checks, 540 lines.
- `docs/modules/compute.md` — private scope, behavior and integration gaps, 59 added lines.

Three files, 1,076 insertions. The original Forge code/rules were not modified. The patch is
not a live feature: no `Hyperparams.engine`, runner route, server/app control or dependency
change exists. Review before transplanting onto tomorrow's verified `origin/dev`.

**Implemented contract.** `await load_scorer(..., engine, token, max_length, revision)` and
`await scorer.score(rendered_rows, batch_size, on_progress)` provide numeric class order,
explicit generic-name aliases, exact conditional verbalizer softmax plus full-vocabulary
`mass_outside`, first-sequence-only classifier truncation, explicit length refusals,
length batching, original-order results, worker-thread forwards and existing step events.
Both Hub loads use explicit `token` and `trust_remote_code=False`; model loads additionally
require `use_safetensors=True`. No secret is read from environment or put in progress.
See [the exact scoring contract](execution-contracts.md#scoring-foundation-contract).

**Offline evidence.** 39 new tests pass with Python 3.14.5 using the original checkout's venv
as a read-only tool source and `PYTHONPATH=src`. Ruff passes. Source-only strict Pyright:
zero errors, zero warnings. Source+tests: zero errors, eight warnings from pytest's incomplete
`approx` typing and the fake tensor index. The reused environment has neither torch nor
transformers. No real model weights, paid APIs or GPU runs were used.

The broader pipeline sweep at the earlier 25-test checkpoint was 317 passed, 75 failed,
one skipped. The untouched base independently produced the same 75 failures, 292 passed,
one skipped. The existing venv records distribution version 0.0.1 while the audited source
is 0.3.0; most failures are version-handshake failures before runner logic, and an installed
metadata check misses the current bitsandbytes declaration. The skip requires datasets.
Do not call that broader suite green. Rebuild an isolated correctly installed environment
and rerun it during integration; do not weaken version-handshake tests to reuse this venv.

**Review checklist.** Inspect semantic decisions and fakes, verify a hypothesis is never
altered, check alias direction (source `id2label` name→output name), ensure no hidden pickle
or remote-code fallback, and review full-vocabulary numeric accuracy. The conservative
equal-limit refusal prevents `only_first` tokenizers from trying to remove an entire
nonempty premise. Check cancellation behavior: cancelling an `asyncio.to_thread` await
does not stop the forward already running; job-level process termination remains required.

**Acceptance.** The patch is suitable as private groundwork after review and a correctly
installed offline sweep. It is not complete signed `f:SCORE` until `INFER-002` passes real
VM/reference acceptance. Keep both statuses distinct in the registry.

The portable handoff copies are [drafts/scoring-foundation.patch](drafts/scoring-foundation.patch)
and [review-evidence.md](review-evidence.md). Use these if `/private/tmp` has been cleared;
the temporary checkout is a convenience, not the handoff's only copy. Apply only in the
assigned isolated checkout after base reconciliation and review, never over Claude's work.

## INFER-002 — integrated scoring and reproducible inference

**Signed scope.** Forge `SCORE`, P2a and P2b; phase-P3's direct dependency and knob review.
Work in cohesive reviews: scoring routing and dependencies; multi-choice/logprob/provenance;
then guided format/system prompt. Do not expose knobs whose worker branch is unimplemented.

**Files.** `pipelines/inference_runner.py`, `_scoring.py`, `_common.py` only if required by
shared output changes; `compute/batch.py`; `llm/client.py`, `llm/responses.py`;
`storage/hf_hub.py`; `training/progress.py` only under a coordinated event-contract change;
`pyproject.toml`, lockfile, clean-install workflow/smoke script; matching unit tests and
compute/LLM/storage docs. Verify paths against the refreshed base rather than old line numbers.

**Implementation sequence.**

1. Freeze reviewed `Hyperparams.engine = vllm_chat | hf_classify | hf_label_logits` (default
   `vllm_chat`), `pair_template`, `label_alias`, `label_tokens` and incompatible-field errors.
   Reuse inert bounded substitution for both rendered strings. A new pair renderer must
   reject oversized hypotheses rather than inherit the existing renderer's character cut.
2. Route directly to the private scorer on HF engines, without starting vLLM. Keep current
   vLLM behavior compatible. Explicitly choose token-length/model limits and output schema;
   persist failed rows and overflow counters rather than quietly dropping them.
3. Declare bounded direct torch/transformers dependencies in `[serving]` compatible with
   the actual vLLM release, plus `sentencepiece>=0.2,<0.3` and `protobuf>=5,<7` as specified.
   Prove fresh Linux Python 3.14 resolves for the shipped extras; do not guess bounds from
   the transitive lockfile. Add tiny classify/label-logit construction/load checks to the
   isolated clean-install smoke, with model downloads confined to authorized CI/VM checks.
4. Add seed, bounded `n` (1–64), long-format `sample_index`, censored top-k label logprobs,
   allow-listed chat-template kwargs/reasoning parser, joinable IDs and ≤32 echo columns.
   Normalize all choices while retaining `LLMResponse.text == choices[0].text` compatibility.
5. Add `CommitRef{sha,url}`-returning commit methods without changing old upload methods.
   Capture model and dataset SHA provenance and the actual output commit. Coordinate the
   typed progress/artifact extension with server readers; do not stuff object refs into
   the event's float-metric mapping.
6. Add bounded system prompts, guided choice/JSON/regex where the pinned vLLM supports
   them, memory utilization and `response_format` propagation. Freeze upstream key names
   in a contract test. Count invalid outputs. No arbitrary serving-extra passthrough that
   can load files or `lora_modules`.

**Offline checks.** Existing inference outputs stay compatible; all engines use identical
row IDs and failure accounting; `n=4` gives exactly four rows per successful input; per-sample
choices/reasoning/logprobs survive normalization; top-k missing labels are marked censored,
never assigned exact zero. Verify explicit tokens, artifact commit equality, sealed echo and
redacted errors/reasoning. Use fake loads and transports, and upstream-shape contract tests.

**VM acceptance.** In the approved integration environment, score a pinned approved 1k
development slice with ModernBERT NLI and Qwen3-1.7B label logits. Both Parquet outputs must
open and match independent local reference probabilities within 1e-4. Run `n=4` generation
and verify 4×N rows plus commit refs. Measure actual latency, memory, batch/length mix and
overflow counts; the decoder's full-sequence logits can consume substantial memory, so
optimize only with a demonstrated equivalent last-position result. Store manifests/logs,
not protected rows, in the review evidence. Agent knob design requires per-PR explicit
sign-off before merging, because the existing hyperparams passthrough makes it reachable.

## EVAL-001 — shared classification, calibration and statistics

**Files and boundary.** Follow phase-P4's correction: pure functions live in
`src/strata_forge/metrics/classification.py`, importing only core and bounded numpy.
`evals/classification.py` re-exports the eval-facing API. Add module README/rules and the
leaf row in `docs/architecture/module-boundaries.md`; permit training and evals to import
metrics, never training→evals. Update eval docs and repair the unsupported README F1 claim.
Reserve the module/ADR edits through the coordinator.

**Functions.** Accuracy, macro-F1, confusion matrix; multiclass sum-form Brier, NLL;
equal-mass/equal-width ECE (15 bins), classwise ECE, bounded log-temperature fitting;
one-vs-rest AUROC with tied ranks, risk–coverage, AURC, threshold fitting and selective
risk/coverage; binary PRF, support-set micro/macro PRF; invalid rate and masked 2-way
metrics; deterministic clustered bootstrap, paired delta bootstrap, exact McNemar,
Wilson intervals, Holm–Bonferroni; Cohen/Fleiss κ. Preserve the signed function names.

**Contract first.** Freeze label ordering, absent-class behavior, normalization tolerance,
invalid/missing prediction denominators, masked 2-way projection, empty-set conventions,
confidence tie handling, ECE boundaries, random seed/cluster resampling, NLL clipping and
undefined metric representation before parallel consumers start. Do not silently normalize
malformed inputs or represent undefined statistics as zero. See
[numeric contract](execution-contracts.md#numeric-contract-to-freeze-in-eval-001).

**Checks.** Hand-computed fixtures for every metric, tied/empty/all-one-class cases,
probability validation, identical seed→identical interval, cluster resampling integrity,
paired item alignment and a synthetic known-temperature recovery. Verify unreachable
abstention target returns `None`; support PRF handles both sets empty. Static import-boundary
checks prove the leaf stays independent. This task needs no weights, network or VM spend.
Acceptance is mathematical/reference agreement and type/lint/import checks, not a dashboard.

## EVAL-002 — evaluation runner and custom graders

**Delivery boundary.** The signed classify/score-predictions runner and server eval builder
can ship additively through the existing authorized run lifecycle and pinned HF artifacts.
They do not wait for workspace/native-storage/custom-job migration. PLAT-012 adopts this
executor into the broader job envelope and adds isolated arbitrary-code extensions later.
Named registered graders and the protocol can be implemented without accepting executable
code in a control-plane spec. Preserve all existing security/sealed and wire-design gates.

**Core signed contract.** `pipelines/eval_runner.py` with inert `EvalSpec(extra="forbid")`,
`mode=classify|score_predictions`, reading shared scoring and metrics. Follow phase-P4's
naming reconciliation, not guessed server fields. `classify` uses pinned model/dataset refs,
`hf_classify|hf_label_logits`, exact rendering and `apply_calibration=auto|none`.
`score_predictions` joins pinned predictions to gold by stable ID and uses named parsers:
`label_logprobs|json_field|choice_regex|verbalized_conf|vote_fraction`. Freeze duplicate ID,
missing ID, extra prediction and invalid-output behavior before implementation.

**Files.** New runner and tests, typed rendering-config schema/reader at a reviewed common
path, shared artifact writers, scoring/metrics modules only through their owners; `[scoring]`
extra and clean-install matrix; eval/compute docs, examples and an available ADR number.
The signed spec suggests ADR 0019, but the current base already uses 0019 for secrets.
Choose the next unused ADR number; do not overwrite an immutable ADR to match an old plan.

**Outputs.** `items.parquet` with IDs, gold/prediction, raw/calibrated class probabilities,
correctness and slices; optional inconsistency/support/abstention values when produced.
`metrics.json` contains overall and per-slice statistics/CIs, diagnostic flags for slices
under 500, count/exclusion reasons and comparison alignment. A dataset card points only
Parquet at the Viewer. Model heads not available→null/“not produced,” never an artificial
zero or success. Emit step-bearing eval events and ≤64 headline floats, with actual output
artifact refs in the reviewed typed event field.

**Rendering/calibration.** Read `strata_verifier.json` at the pinned model revision; explicit
spec overrides win. If neither supplies required rendering, refuse instead of guessing.
Validate bounded JSON, no executable templates, label order and ≤20 premises/≤8192 tokens
per the verifier protocol (a user knob may only lower that premise cap). `abstain=auto`
reads the fitted threshold; absent threshold means threshold-free curve/AURC only, with
no fabricated selective operating point. Calibration uses development/crisp labels only.

**Custom grading extension.** Reuse `evals.graders.base.Grader` and named deterministic/
LLM judges; no provider SDK in a grader. A fixed named grader registry may compose useful
agent/tool-state checks. Arbitrary user Python is a `PLAT-012` worker/plugin deployment,
not JSON containing `eval`, pickles, dynamic import strings or a shell command. A verifier
score is one grader, not sole authorization for auto-promotion. Freeze a typed artifact
schema for general grader results with the central contract owner before adding a third
eval mode or changing the signed classification modes.

**Checks and acceptance.** Fake complete/missing/duplicate joins; invalid counts-as-wrong;
list-valued slice explosion; diagnostics; sealed marker before first upload; no row text in
phase/errors; card configs; all output caps; missing optional heads; calibration/schema
override round-trip. Then use an approved pinned 3k development set in both modes, with
independent recomputation matching macro-F1 and Brier within 1e-9 and deterministic CI
endpoints at the same seed. Verify reload and artifacts in the app. Forge release precedes
server eval-builder promotion and app activation; exact MCP/UI launch parity is separate
cross-repo acceptance, not satisfied by a local runner.

## TRAIN-001 — sequence classification and the complete verifier output

**Bounded authority review.** The [classifier authority packet](drafts/classifier-authority-review.md)
proposes purpose-separated read-only integration against the frozen primary-classifier worker.
It neither implements nor resolves the broader scientific/output target below.

**Scope authority.** Use signed phase-P5, which extends the older spec's incomplete output
heads. The deliverable is 3-way strict deduction plus separate inconsistency flag,
per-premise support scores and fitted abstention, for encoder and decoder-head variants.
It is platform capability; a benchmark win is a separate Models task.

**Cohesive sequence and files.**

1. `training/dataset_format.py` and format tests: `nli_pair`, required premises/hypothesis/
   label; optional distribution, label-space, inconsistency, support, weight and primary
   head roles. Accept string/list/`{pid,text}` premises; explicit aliases and integer index
   map; KTO boolean coercion remains independent. Reject unknown support IDs, duplicate
   premise IDs, malformed distributions and non-primary heads before weights load.
2. Separate eval-default fix in `training/sft.py`, `preference.py`, `pipelines/finetune_runner.py`:
   an eval split gives `eval_strategy=epoch` unless validly overridden; protect runner-owned
   fields from `extra_trainer_args`. Confirm current Trainer keyword names in contract tests.
3. `training/classifier.py`, configs/registry, collator and fine-tune runner: `seq_cls` /
   `task_type=SEQ_CLS`; soft CE + default 0.1 Brier, masked 2-way loss, optional class weights,
   smoothing and sample weights. Use the metrics leaf, not evals. `overflow=error|drop`
   reports counts and never silently cuts a hypothesis. `premise_format=ids|plain` is explicit.
4. Auxiliary heads follow role mappings. Inconsistency is separate BCE; refuse an unsat
   split with that signal silently unmapped. Support requires ID-shaped premises and
   `premise_format=ids`; score exactly each row's premise IDs, conditioned on the hypothesis.
   Phase-P5 proposes a bilinear answer/CLS–marker head and default weights 1.0: confirm the
   design in the implementation PR, including causal-decoder conditioning, rather than
   treating a proposed architecture as already measured.
5. `arch=decoder_logit_head`: initialize score rows from the named single-token LM head/
   tied embeddings; tiny-model step-zero restricted probabilities must match SCORE. Built-in
   HF class or Forge-installed wrapper only; saved safetensors/config must reload without
   repo-provided remote code. Resolve leading-space verbalizer normalization once and record
   actual token IDs in the artifact; do not silently give training and scoring different IDs.
6. Allow-list `lora.modules_to_save` (≤8 names, `^[A-Za-z0-9_.]{1,64}$`) and auto-retain all
   trained heads. Initial classifier adapter capability is LoRA/full, as signed; QLoRA is
   not enabled just because causal-LM QLoRA exists. Merge chooses the task-type auto-class.
7. Calibration/writer review: fit positive bounded T on the approved calibration split,
   defaulting to eval split; optional group T with bounded group keys. Fit abstention on
   calibrated `max(p)` for the preregistered target risk; unreachable→null+reason, not run
   failure. Fit flag/support thresholds by the signed F1 rule. Store T, thresholds, target,
   coverage, input `dataset@sha:split`, n and before/after NLL/ECE in `calibration.json`;
   write exact rendering and supported outputs in `strata_verifier.json`.
8. Exclude checkpoints and unsafe Trainer artifacts from the pushed model artifact, retaining
   resumable state in a separate approved location. Copy rendering/calibration JSON on merges.
   Extend HF classifier scoring to optional `p_inconsistent` / `support_scores`; models with
   no such heads still emit nulls, not zeros.

**Offline verification.** Hand-computed losses; zero auxiliary weights reduce to CE; masked
2-way gradients; no silent integer-label reordering; step-zero decoder equivalence; support
IDs and cross-row isolation; changing only a decoder hypothesis changes its support scores;
unsat role guard; module-retention/merge; import boundaries; known T/threshold recovery;
calibration reader/writer reproduces the training collator's exact rendering; no checkpoint
in model uploads; safe load flags on every init, reference, merge and fallback path.

**VM/model acceptance.** First tiny encoder/decoder one-step smoke with both auxiliary heads;
then approved ModernBERT-large full and Qwen3-0.6B LoRA runs on the signed 50k `nli_pair`
development workflow, with explicit spend authorization and manifests. Reload saved models
with remote code disabled and evaluate without manual rendering overrides, reproducing dev
calibration and all optional outputs. Run the matched-size Ettin probe only under the
research owner's exact protocol. No trained result or <50ms claim exists in this handoff.
App/server method mirrors, per-PR agent sign-off, registry version and release order must be
accepted before declaring end-to-end completion.

## PROMPT-001 — authoritative composition and conformance

**Goal and boundary.** Make module/surface/version/build semantics deterministic and shared
by Python training/evaluation code, TypeScript agent apps, API and MCP. The server owns
workspace permission checks and persisted resources; Forge owns generic composition/rendering
primitives and conformance fixtures. The browser editor is not a second authority for builds.
An isolated TypeScript prompt spike already exists in the earlier work, but its UTF-16 source
offsets and any assumed template behavior require review before becoming a portable contract.

**Before code.** `PLAT-001` chooses the authoritative compiler implementation and publishes
fixtures for canonical bytes/fingerprints, newline/Unicode policy, dependency order, roles,
template dialect and source-map indexing. A Python-authoritative implementation fits the
training path, but do not silently override the central decision. If compilation is server-
authoritative, TypeScript can consume its builds; any independent renderer/compiler must
pass the same fixtures. Do not let two “equivalent” implementations produce unrelated hashes.

**Files and sequence.** Inspect `prompts/template.py`, `variables.py`, `rendering.py`, registry/
store protocols and tests first. Reuse existing useful pieces; preserve Langfuse as an optional
adapter rather than making its SDK the new compiler's domain model. Add reviewed Pydantic
module/surface/build/source-map shapes at a small documented location in `prompts/`, a pure
compiler and a shared fixture corpus. Update `prompts/README.md`, module rules/reference and
exports only when behavior is actually supported. The server resource schema is coordinated
through `PROMPT-002` and central contracts, not copied into an independent Forge database.

1. Import actual Strata and verifier development prompts with their exact original template
   dialect and source provenance. Preserve f-string/Jinja/format braces and whitespace; do
   not mechanically reinterpret every `{...}` as a Strata variable. Protected/sealed prompt
   inputs and research rows are not imported into fixtures or provider-bound analyzers.
2. Represent immutable module versions with ordered content/dependencies, variable declarations,
   explicit constraints and provenance; surfaces contain ordered pinned module refs plus
   surface-specific text and message/output targets. Editing creates a candidate version.
   Dependency resolution records every transitive version; no hidden “latest” in a released build.
3. Compile a typed immutable build with compiler/schema version, declared inputs, exact output
   representation, source map and canonical fingerprint. Preserve system/user/tool/skill
   boundaries; do not flatten them into text and claim exact prompt equivalence.
4. Refuse missing dependency, cycles, incompatible variable declarations and contradictory
   structured constraints with bounded structured diagnostics referencing source locations.
   Keep duplication, possible natural-language conflicts and ambiguous precedence advisory.
   No automatic text deletion/rewrite or assumption that later instructions always win.
5. Impact analysis computes candidate affected surfaces/builds and exact diffs. Selected
   consumers update explicitly; others keep their pins. Publication/deployment is server-
   authorized and tied to an evaluated build, not a mutation of existing releases.

**Offline checks.** Golden nested-dependency/order/variable-binding fixtures; cycle/missing
dependency refusal; namespace/type mismatch; Unicode astral characters, combining characters,
CRLF/LF and escaping; empty/duplicate module inclusion policy; message/tool/skill target
preservation; source-map passage round-trip; deterministic hashes; original prompt import
before/after equality. Advisory findings never block compilation unless the reviewed contract
requires an explicit structured constraint. Any model-assisted analyzer carries model/prompt/
analyzer versions and passage references and is tested as advice, not proof of contradiction.

**Acceptance.** Python and TypeScript pass the same byte/hash/source-map fixtures. An actual
development prompt collection can be imported, modularized without accidental semantic edits,
compiled twice identically, changed as a candidate, and compared while the released build
remains immutable. No model call or paid conflict-analysis run is needed for compiler acceptance.
Persisted editor/publish/evaluation integration is separate cross-repo work; a compiler demo
alone does not complete prompt management.

## PROMPT-004 — SDK consumption and execution capture

**Goal.** Let Python and TypeScript apps use pinned builds and recover what was actually
observed for each model attempt. Depend on `PROMPT-001` plus `PLAT-011`'s immutable artifact
contract. Coordinate a maintained TypeScript package owner/location with the central registry;
do not embed an unowned second SDK inside the Python wheel or invent a parallel prompt API.

**Consumption sequence.** Resolve explicit build IDs or environment labels through the server's
authorized API, recording the immutable build actually selected. Fetch/cache with hash and
schema validation; labels may advance, pinned builds do not. Render against typed declared
bindings with explicit missing-variable and incompatible-version failures. A cache policy
states stale/offline behavior; it never silently replaces a requested pin with a newer build.
Use the shared conformance fixtures for Python/TypeScript message content, tools, output
contracts, exact token-verbalizer literals and generation settings where applicable.

**Capture boundary.** Extend the existing `llm` request/tracing seams or a clearly documented
SDK adapter; do not require users to move every provider call into Strata. Distinguish
`retrieved`, `rendered`, `request_captured` and `response_observed`. MCP retrieval proves only
that content was returned. The SDK records its actual observation boundary (for example
normalized Forge request before provider transformation), whether inputs/settings were
observed or caller-reported, and any modification after rendering. It must not claim provider
wire-byte equivalence or acceptance it did not observe.

**Attempt lineage.** Capture immutable build plus binding/input refs, actual provider/model/
revision where known, messages/tool schemas/response contract under retention policy,
generation parameters, parent trace/span, retries/fallbacks/continuation legs, response/error,
usage/cost when observed and missing fields explicitly. A fallback is a separate linked
attempt, not the first request relabeled as success. Tool messages and tool-call schemas are
part of prompt behavior; do not retain only system text and claim the request is reproducible.

**Privacy and operational behavior.** Reuse the shared redactor and retention/tenant controls.
Never capture provider/vault credentials; store original-content and redacted-content hashes
as distinct evidence. Sampling, truncation, omitted binary/large content and transport losses
are explicit. Background export has bounded buffering/retry and a clear failure policy; an
observability outage must not silently rewrite the app's model request or fabricate an outcome.
SDK compatibility/version negotiation and migration use known schema versions.

**Files/checks.** `prompts/` build-consumption/rendering APIs and tests; `llm/client.py` /
provider normalization hooks and `tracing/` only under coordinated ownership; SDK adapters
and shared fixtures in their declared package; prompt/LLM/tracing docs and examples. Fake
transports cover changed prompts after rendering, tools/settings preservation, failed export,
partial capture, multiple attempts, fallback and resumed tool loop, revision mismatch, no raw
key, stale cache and cross-workspace cache keys. Verify artifact/render fingerprint agreement.

**Acceptance.** One Python fixture app and one TypeScript fixture app consume the same pinned
build, render identically, execute against a fake deterministic provider and produce matching
evidence-level/attempt records with explicit boundaries. A bounded approved live request can
later verify a specific provider adapter; offline tests do not establish that provider's actual
wire payload or tokenization. Cross-repo trace inspection and prompt-outcome comparisons
must reload persisted records before this is called the complete product slice.

## TRAIN-002 — broader training registry and optimizations

**Deliver small verified capabilities.** Build a capability matrix of method × task type ×
dataset shape × adapter × precision × device/distribution strategy. Current SFT/DPO/ORPO/KTO
and disabled GRPO are the baseline, not proof of arbitrary methods. Add each method through
`training/methods.py`, typed configs, the right role coercers, runner dispatch and the same
server/app/MCP capability projection. Each cell is supported, refused with reason, or
experimental; do not show “all methods” as an untested enum.

**Candidate units.** Signed named reward registry and prompt-only format before enabling
GRPO; newer alignment/distillation methods only after checking the pinned TRL primary APIs;
adapter/optimizer variants and quantized training; packing/bucketing; gradient checkpointing;
mixed precision; resumable checkpoints; multi-GPU DDP/FSDP. Keep custom task/architecture
implementations behind `PLAT-012`, rather than bloating one causal-LM method switch.

**Signed constraints.** Rewards are names plus validated params, never code. Private verifier
reward stays in a private pinned plugin until its release decision. GRPO verifier/reward
experiments and multi-GPU research follow the signed G3 gate; same-base merging follows G5.
Use actual latest decision addenda rather than assuming the user's broad platform ambition
waived those research budgets/gates. QLoRA/FSDP incompatibility is refused where specified;
rank zero alone writes progress/uploads. Resume state must match base/data/format/optimizer
semantics and artifact lineage, not merely find a filename.

**Checks/VM acceptance.** Fake config/registry/role dispatch and incompatible combinations;
contract tests against bounded upstream APIs; no remote/pickle escape; deterministic reward
fixtures and cancellation/resume behavior. One tiny authorized reference run per supported
cell, loss/gradient agreement at matched effective batch where relevant, and measured speed/
memory tradeoffs before a performance claim. Update clean-install combinations, docs,
examples and capability metadata in the same change. Every advanced feature is incomplete
until launch, failure, monitoring, artifact reload and evaluation work together.

## INFER-003 — custom-model inference, serving and quantization

**Problem.** Existing inference is a finite SSH batch launcher. A durable endpoint has distinct
deployment, readiness, authentication, scaling and termination semantics; do not label a
long-running batch run “deployed.” Freeze the server-owned deployment contract and its link
to immutable model/rendering/calibration artifacts first.

**Work.** Extend existing `compute/serving.py` and backend protocols where appropriate, with
typed supported task families (generation, classification/verifier, embeddings/reranking
as separately validated adapters). Models from `TRAIN-001` reload through installed Forge
code. Fully custom Python architecture enters only through a named `PLAT-012` runtime or
user-controlled environment, not `trust_remote_code=True` in a multitenant worker. Keep
metadata visualization weight-free; weight movement occurs on the user's/managed target.

Define reviewed dtype/quantization presets with compatibility metadata; preserve exact base,
conversion parameters and output hashes. A quantized verifier needs accuracy/calibration/
support/flag drift evaluation, not only a successful load. Do not assume generation-serving
vLLM supports every classifier, multimodal head, adapter or quantization backend. Missing
support is a useful capability refusal. Keep provider-specific launch options behind explicit
bounded adapters, and cloud credentials outside specs/logs/environment propagation.

**Checks.** Backend contract fakes for start/readiness/auth/health/drain/restart/cancel;
artifact task-type validation; finite batch versus service lifecycle; endpoint access scope;
no leaked secret or unsafe artifact load. VM acceptance covers reload after restart, mixed
length batching, failed readiness cleanup, explicit shutdown, p50/p95/throughput/VRAM and
quality/calibration before/after quantization on approved data. Project isolation, metering
and endpoint authorization are server acceptance, not facts established by Forge tests.

## DATA-002 — dataset recipes and annotation/adjudication engine

**Signed recipe core.** `datasets/recipes/` owns named transforms with `extra="forbid"`
params: pinned load/column-map, union, label-map, restricted predicate filtering,
exact/MinHash dedup, 13-gram decontamination, distractor injection, grouped splits, license/
redistribution/generator filters, jury merge and adjudication. `pipelines/recipe_runner.py`
materializes Parquet plus a hash/lineage manifest. Do not substitute arbitrary SQL/Python
expressions for the signed predicate DSL. Full dataset SQL is a separate reviewed data
execution capability coordinated with `DATA-001`; relational operations may compile to
an internal query plan without exposing file/network/extension side effects to agents.

**Native artifacts.** Define load/write adapters against the central artifact contract so
HF can become optional for platform users; keep the verifier research's pinned HF workflow
until its owner accepts an equivalent migration. Resolve and hash each input before execution;
retry writes a deterministic/idempotent result, not duplicate dataset versions. Preserve
source IDs, license/attribution, source revision, transform params and output rows/counts.

**Annotation.** Server/app own tasks, human sessions, saves and review UI. Forge supplies
κ/agreement math and deterministic `adjudicate` output: final label, source and agreement,
without falsely treating model labels as human labels. Use explicit source/version checks;
stale annotation commits produce a conflict, not a silent last-write-wins merge. Preserve
the signed blind/sealed/locked workflow: provider/agent execution never reads sealed rows,
while permitted humans use chat-less annotation surfaces. No annotation text in logs.

**Tests/acceptance.** One reference fixture per transform, source schema errors, preserved
label aliases, grouped split isolation, near-duplicate groups, deterministic recipe hashes,
no restricted or test-only/frontier-generated row entering a training output, attribution,
and sealed-taint propagation before data writes. Distractor mutation of formal rows triggers
the approved solver re-check, never an assumed unchanged label. Re-materialize E1 byte-
identically under its pinned supported environment; for broader distributed engines record
canonical content equivalence plus file hashes rather than falsely promising byte identity
across arbitrary tool versions. A small manual→adjudicated→versioned→train/eval workflow
must survive reload and retry. Signed recipe research adoption is G2-gated.

## PLAT-012 — named custom task/worker contract

**Reuse the signed plugin design.** Entry-point groups `strata_forge.runners` and
`strata_forge.rewards`; inert `PluginSpec{plugin,entry,dist:"name==x.y.z",params}`;
exact installed distribution/version match; plugin-owned strict Pydantic params; operator
allow-list independent of the run spec. Public Forge owns the generic mechanism, private
project packages own research generators/rewards. See
[worker contract](execution-contracts.md#custom-worker-and-trust-contract).

**Work before exposure.** Coordinate input/output artifact refs and capability declarations
with the server job/artifact owners. Distinguish a trusted operator-installed plugin from
untrusted team-supplied arbitrary code; allow-listing a package name does not sandbox code.
Choose execution isolation/resource/network/secret scope for the hosted case in a reviewed
worker design. Keep arbitrary import paths, install commands and executable payloads out of
the run/MCP spec. The earlier signed path installs user-owned packages on their own VM;
automatic managed installation is a separate implementation and trust decision.

**Files.** `pipelines/plugin_runner.py` and a small plugin protocol/registry; compute task/
backend adapters only if the reviewed contract needs them; shared progress/secrets/artifact
readers; fake entry-point/metadata tests; module docs and an unused ADR number. No private
generator source is copied into public `src/` or wheel/sdist.

**Checks/acceptance.** Unknown name, operator disallow, distribution/version mismatch,
duplicate/confusable entry point, unknown params, bad outputs, resource exhaustion,
cancellation, retry and cleanup all have explicit behavior. A pinned toy plugin on an
approved target reads one immutable fixture, emits valid step/end events, writes a bounded
artifact and can be cancelled/retried without duplicate commits. Record exactly which tests
cover trusted self-hosted execution and which prove hosted isolation. No remote MCP tool is
enabled until the server permission and wire-design review is complete.

## Evidence, checks and handoff format

Use a correctly installed isolated Python 3.14 environment. Record commands, versions,
extras and outcomes; importing from source with an old installed distribution is insufficient
for version-handshake/clean-install acceptance. Initial draft check commands are:

```sh
PYTHONPATH=src /Users/nemanjagemovic/Documents/Personal/Code/strata-forge/.venv/bin/python -m pytest tests/unit/pipelines/test_scoring.py -q
/Users/nemanjagemovic/Documents/Personal/Code/strata-forge/.venv/bin/ruff check src/strata_forge/pipelines/_scoring.py tests/unit/pipelines/test_scoring.py
/Users/nemanjagemovic/Documents/Personal/Code/strata-forge/.venv/bin/pyright --pythonpath /Users/nemanjagemovic/Documents/Personal/Code/strata-forge/.venv/bin/python src/strata_forge/pipelines/_scoring.py
```

Those absolute paths are machine-local review conveniences, not CI commands or package
dependencies. For subsequent work use the repo Makefile/tool commands in the claimed
checkout, target module suites plus required CI, and a clean-install smoke for changed extras.

Every handoff records task ID, owner, actual base SHA, branch/worktree/patch path, changed
files, contract decisions, offline command results, remaining warnings/failures, real VM
manifests and spend (if authorized), integration state, docs, and next owner. Mark each
acceptance item as passed, failed or not run. A fake model passing math tests, a job returning
exit zero, an uploaded file and a usable cross-repo feature are different pieces of evidence.

The [primary classifier worker review](drafts/classifier-worker-review.md) and [inert task contract](drafts/classifier-worker-contract.md) narrow the SDK-to-worker increment, including mandatory safe decoder export→installed scoring→evaluation interoperability. The [example](drafts/classifier-worker-example.json) is fictional design metadata, not an available worker. Training-purpose read authority, guarded model publication and exact app/agent wire are separate reviewed delivery units; no Eval credential or research exception may be reused.
