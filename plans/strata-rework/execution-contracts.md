# Forge execution contracts and review decisions

Internal companion to [the Forge work queue](README.md). These contracts distinguish the
implemented private draft from signed integration scope and decisions still to freeze. The
[central contracts](../../../strata-app/plans/strata-rework/contracts.md) own cross-repo
resource/permission/artifact/job schemas. Do not duplicate or rename them here.

## Scoring foundation contract

Status: implemented only in the unmerged `/private/tmp/strata-scoring-foundation-20261103`
draft, based on `28a5bcda9a3e013509ca8763f7699d55192a0db7`.

| Element | Draft behavior |
| --- | --- |
| `load_scorer` | Async internal API; requires Hub model ID, scoring engine, explicit nonempty token string or `False`, positive token-length limit; optional revision, aliases/verbalizers and device (CPU default) |
| `ScoringInput` | Frozen strict shape `{text, text_pair?}` of already-rendered text; no template interpretation |
| `hf_classify` | Every input requires a pair; full classifier-logit softmax, classes ordered by numeric `id2label` IDs |
| `label_alias` | Map original `id2label` name→output name. Each generic `LABEL_n` requires an entry. Unknown source keys, duplicate outputs, empty/oversized names and malformed IDs refuse |
| `hf_label_logits` | Single prompt only. `label_tokens` map output label→literal verbalizer string. Exactly one distinct token each with `add_special_tokens=False`; no top-k censorship |
| Probabilities | Classifier: full class softmax. Decoder: softmax conditional on the named token IDs; probabilities sum to one across those labels |
| `mass_outside` | Decoder full-vocabulary softmax mass outside all named IDs. Sum the complement directly to preserve tiny nonzero mass; this is neither logical “unknown” nor a calibrated error probability |
| Padding/position | Right-pad batches; read each decoder prompt's last real token index, not the longest row's padding position. No chat-template application or auto-generated answer prefix |
| Batch execution | Stable length sorting, one model forward per batch, `.eval()` and inference mode, restored input ordering; forwards run in a worker thread |
| Result | `ScoringResult{rows,overflow_rows,truncated_rows,failed_rows}`; each `ScoredRow` has original index, probabilities, predicted label, optional outside mass and length/error fields |
| Flat columns | `p_<label>`, `pred`, `error`; decoder additionally `mass_outside`. No output dataset upload or item-ID join is owned by this module |
| Progress | Existing `ProgressEvent(kind="step",stage="run")`, cumulative completed rows and succeeded/failed/overflow/truncated counts. Callback runs in the worker; no start/end lifecycle |
| Loads | Explicit credential and revision to tokenizer/model; remote code disabled on both; safetensors required for weights; no local path, pickled checkpoint fallback or arbitrary model kwargs |

### Length and error behavior

Classifier raw token lengths include pair special tokens. If over the explicit limit, use
`truncation="only_first"`; count the overflow and truncated premise. The hypothesis is never
truncated. A hypothesis plus pair special tokens that cannot fit is a row refusal; if it
exactly exhausts the budget while the premise is nonempty, also refuse, because HF tokenizers
do not universally support deleting the whole first sequence under `only_first`.

Decoder prompts are never truncated. Overlimit→`prompt_too_long`; zero tokens after the
tokenizer's special-token accounting→`empty_prompt`. These refused rows have no probabilities,
prediction or outside mass. All are preserved in original order and counted as completed/
failed; classifier truncated rows may still succeed and are counted separately.

Configuration/mapping/shape/nonfinite-logit failures raise Forge `ValidationError`.
Missing optional packages raise `ImportError` with an install hint. Unexpected upstream
exceptions propagate: the integrating runner must redact before events, stderr or artifact
error columns. No caller should infer these exceptions are safe merely because they came
through this module. A batch-model failure is not converted to invented per-item scores.

`Scorer.score` cancellation stops waiting on its worker thread, not an already-running
forward. Production run cancellation must terminate the supervised process and prove resource
cleanup. The initial internal API materializes rows for length sorting; it is not a claimed
streaming/constant-memory dataset engine. Full decoder sequence logits may need a tested
memory optimization before larger context/batch use.

### Decisions owned by INFER-002, not silently settled by the draft

- Exact serialized model/dataset/output reference fields in the existing event/spec contract.
- Per-row result schema/IDs, Parquet preservation and all-failed/partial-failed run verdicts.
- Inert pair-template rendering and a nontruncating hypothesis character-limit policy.
- Engine-specific incompatible generation knobs and their user-facing refusal text.
- `[serving]` direct version bounds proven against actual vLLM resolves on shipped platforms.
- An approved real-model reference smoke and the correct model token/context limits.
- Whether/where to expose batching knobs and literal versus leading-space verbalizer forms.
  The current draft uses the literal string. The later signed decoder-training plan proposes
  normalization; if adopted, record one rule and the final token IDs for both train/eval.

## Numeric contract to freeze in EVAL-001

The signed function list is in phase-P4 and [the task brief](README.md#eval-001--shared-classification-calibration-and-statistics).
Freeze these details in type signatures, docs and fixture tests before consumers ship:

1. **Labels and shapes.** Ordered labels are explicit; probability column order is never
   inferred lexicographically. Gold/predictions/probabilities have matching item dimensions.
   Duplicate labels, nonfinite values and out-of-range probabilities refuse. State the sum
   tolerance and whether a near-rounding correction is permitted; arbitrary renormalization
   of invalid arrays is not an implicit repair.
2. **Denominators.** Invalid/missing predictions count as wrong for accuracy/F1 as signed.
   Decide and document how missing probability rows enter Brier/NLL/calibration (reported
   coverage and invalid counts, rather than silently dropping failures or inventing a p-vector).
   Confusion reporting retains invalid counts even if its core matrix has only valid classes.
3. **Brier/NLL.** Multiclass Brier uses the sum over classes, not a class-average whose scale
   changes by K. NLL clipping epsilon and zero-probability behavior are explicit. Calibrated
   and raw probabilities retain separate names and metrics.
4. **Two-way rows.** Define the mask/projection of entailed versus not-entailed exactly from
   the signed `label_space`; do not relabel ambiguous not-entailed as contradicted or apply
   three-way calibration to soft/partially observed gold without protocol approval.
5. **Undefined/empty metrics.** All-one-class AUROC, no items, no predicted positives, absent
   gold class, empty support sets and κ degeneracy each have documented results. Optional
   metric not produced is null/reason in reports, never a false zero. Reject malformed data
   separately from mathematically undefined statistics.
6. **ECE and selective curves.** Freeze equal-mass tie/edge handling, equal-width bin edges,
   classwise aggregation and risk–coverage tie groups. An abstain threshold selects on
   calibrated max probability, not on the undetermined class. Threshold search follows the
   signed smallest-cut target-risk rule; unreachable returns `None`, with reason downstream.
7. **Statistical comparisons.** Paired resampling uses the same joined IDs and draws; cluster
   bootstrap samples whole named groups with a fixed generator seed and bounded chunks.
   Empty groups, duplicate IDs, missing pairs and shared calibration data are disclosed.
   Do not call overlapping per-item intervals a paired significance test. Freeze exact
   McNemar tail, Wilson confidence level and Holm multiplicity family.
8. **Support/inconsistency.** Support PRF compares premise-ID sets belonging to the same item;
   IDs from another row are invalid. Predicted support thresholds are fitted only on the
   permitted split. Binary inconsistency metrics are separate from the 3-way logical label.
9. **Performance/provenance.** Record the metric implementation/version, seed, bins, labels,
   fitted temperature/split and threshold policy with results. Keep full per-slice reports
   outside the ≤64-float progress channel. Large bootstrap jobs need bounded memory and
   useful progress/cancellation at runner level.

These are review requirements, not permission to change preregistered metric definitions
after seeing model outcomes. The research owner's frozen protocol wins for research reports.

## Model artifact and training/evaluation contract

`TRAIN-001` writes and `EVAL-002` reads `strata_verifier.json`: exact label order, resolved
verbalizers, premise format, template, max length and declared output capabilities. Explicit
eval overrides win and are recorded; no file and no adequate override means refusal.
Bounded JSON is data, never code; avoid Jinja execution or `str.format` attribute lookup.

`calibration.json` records positive finite T (or allowed per-group T), fitted split ref/n,
before/after metrics, abstain target/threshold/coverage and flag/support thresholds when
those heads exist. Thresholds are finite [0,1] or null with a reason. Group keys are bounded
column values. Missing optional heads produce nulls and “not produced” metrics. Calibration
is an estimated operating policy on a specified distribution, not a safety or truth guarantee.

The label “undetermined” means the hypothesis is not established or refuted under the signed
premise-relative semantics. “Abstained” means the model confidence fell below a fitted cut.
“Inconsistent premises” is a separate output. Preserve the signed unsat-label policy from
the research specification; do not accidentally substitute a different logic's behavior
when generating rows or computing the masked loss. No world-knowledge fallback is added.

IDs support per-premise scores and selected evidence, but a predicted support set is not a
formal proof or automatically minimal. First-invalid-step and missing-premise predictions
are outside the signed v1 output; do not advertise them because a JSON field could hold them.

Weights use safetensors; base/source/model/config refs and task type survive training, adapter
merge, quantization and evaluation. A Forge-specific head must load through installed Forge
code, not repository `auto_map` execution. Checkpoint/optimizer/training_args artifacts stay
outside the distributed model bundle, with retention/resume handled separately. Public
artifacts require the Models release/rights review, not merely a successful Hub upload.

## Custom worker and trust contract

The signed plugin API is the starting point: named entry-point groups, exact installed
distribution pin, strict plugin-owned params, and an operator allow-list independent of the
spec. The central job/task contract owns workspace permissions, authorized compute, artifact
references, metering and durable run identity. The worker does not independently infer access
from a Hub URL or a user-supplied namespace.

Before `PLAT-012` is exposed, record these reviewed answers:

| Concern | Required contract |
| --- | --- |
| Resolution | No spec-controlled Python import path or installation shell string; fail on missing/ambiguous entry point or distribution/version mismatch |
| Trust | Distinguish trusted operator-installed self-host plugins from untrusted uploaded code. A name/version allow-list does not provide process/container isolation |
| Inputs | Immutable authorized artifact refs, task-type/schema version, bounded strict params; sealed and locked policies checked again at execution |
| Secrets | Delivered separately through the shared private-file mechanism; minimum task scope, unlinked after read; no secret in params, argv, progress, model artifact or child environment |
| Invocation | Fixed installed worker entry point and typed spec; no shell interpolation of task fields |
| Resources | Enforced CPU/GPU/RAM/disk/runtime/network bounds according to the selected execution profile, with understandable failure reasons |
| Progress | Valid shared events, bounded metrics/messages, no protected row text; monotonic step/stage accounting where meaningful |
| Outputs | Declared typed artifacts and bounded summaries, hash/size/schema checks, actual committed refs; safe path handling and no artifact path traversal |
| Retry | Attempt identity plus immutable inputs; idempotent commit/publication; no duplicate billable execution by an automatic retry without policy |
| Cancellation | Supervised termination and verifiable cleanup; plugin coroutine cancellation alone is not proof GPU subprocesses stopped |
| Failure | Structured bounded code/reason and redacted diagnostic location; invalid output is not marked succeeded; partial results follow the task's explicit policy |
| Release | Generic protocols public, project-specific plugin source private until approved; wheel/sdist review proves no private sibling code or credentials entered |

For user-controlled VMs, the signed design lets the user install the private package into the
known Forge interpreter; it does not transfer a repository deploy secret to the VM. Managed
package building, images, registries and hostile-code isolation require separate server/
compute implementation. Use approved operator-owned fixtures to test the worker before
running research generators, arbitrary architectures or optimization jobs.

## Integration and acceptance records

Each task keeps offline checks and authorized VM acceptance separate. Add concrete evidence
to the central ledger: input/output refs, actual engine/model/dependency versions, exact run
configuration, counts including invalid/overflow/refused rows, independent comparison,
machine and batch/length mix, terminal state/cleanup, spend and reviewed limitations.
Never replace an unavailable real-model check with a passing fake or a claimed benchmark.

Release/spec/dependency changes follow the repository's minor-version rule, exact engine
handshake and Forge→server→app sequence. The older signed version numbers are planned release
train labels; reconcile them with actual intervening releases before shipping. Agent wire
changes retain the exact per-PR sign-off requirement even if an earlier design packet exists.

## Safe model initialization and reload

Every classification or causal scoring load requests `output_loading_info=True`. Refuse missing non-head weights before processing rows; inference and reload cannot silently initialize a head. Training init may permit only exact keys declared by a trusted new-head initializer, with decoder heads requiring the explicit decoder-logit-head path. Merge/reload must re-apply the guard. Bounded loading metadata records key names/codes only, without weight values or arbitrary warning/error text. Verification uses synthetic checkpoints missing body/head keys and complete checkpoints.

Record the explicit tokenizer/model pad policy. Full immutable commit SHAs resolved from Hub pull-request refs are valid pins; mutable refs are not frozen research inputs. Apertus generation support does not imply classification support: a classifier requires an implemented Forge-owned head, never remote code or pickle fallback. These are accepted platform requirements, not evidence that every training/merge consumer is implemented.
