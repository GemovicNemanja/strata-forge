# Proposed primary-classifier task contract

This is inert design metadata, not a shipped schema. Unknown fields refuse. Pydantic models
are frozen/strict/extra-forbid/hide-input-errors; public validation wraps failures in fixed
codes outside exception handlers. No rejected row, prompt, config body or adapter exception
is logged. See [the review packet](classifier-worker-review.md) for source pins and acceptance ownership.

## Task and server boundary

`ClassifierTrainSpec` has `schema_version=1`, server-stamped `engine_version`, opaque32hex
`run_id`, `method=seq_cls`, `task_type=SEQ_CLS`, `base`, `dataset`, `training`,
`adapter=none`, `merge_adapter=false`, `output=null|{repo_id}`, and fixed
`progress_path=progress.jsonl`. No executable/callback/import/checkpoint path is expressible.
The spec's sorted compact JSON digest includes every executed field. The server derives
reserved fields; user hyperparameters cannot replace authority, pins, run ID or filesystem terms.
The included example uses the audited SDK SHA as a placeholder; an implementation must stamp
the future exact worker revision, never pretend SDK9006 already contains the worker.

Existing `POST /api/runs` remains `kind=finetune`, with `method=seq_cls`. Its existing model ID,
model revision, dataset ID/commit/config/split/mapping fields populate the strict worker spec.
Add one typed classifier-settings DTO; reject causal hyperparams, eval_split, nonempty LoRA,
merge, extra Trainer args, output paths and other ignored fields on this branch. Permit a
fine-tune model revision only on the validated classifier branch; preserve legacy causal wire.
Echo all effective classifier params on reads so rerun is reproducible. A requested output is
never a produced result. Default-off capability refuses before Hub probes, records or launches.

Use current owner-scoped SSH targets/credentials. Never enable `backend=local` on the server:
the synthetic CPU test uses an isolated trusted harness/fake SSH transport, not a production
request flag. Stamp the owner's actual personal project from trusted context; no inferred
collaborator entitlement to another account's vault, compute or legacy Runs.

## Base model and initialization

`base.ref` is `namespace/repo@<40 lowercase hex>` for both weights and tokenizer. No mutable
main, separate unpinned tokenizer, adapter-base discovery, remote code, pickle fallback,
auto-map execution or file path is accepted. Resolve the exact ref with actor credentials
after policy approval; local-cache presence does not authorize access.

`base.initialization` is one of:

- `{kind: encoder_random_head}`: complete built-in BertModel with its actual pooler. Safe
  source loading must prove every body key and shape before the SDK creates the named head.
  An MLM-only checkpoint missing a pooler is unsupported, rather than silently randomized.
- `{kind: decoder_logit_head, label_tokens: {label: exact_string, ...}}`: complete installed
  GPT2LMHeadModel or ModernBertDecoderForCausalLM. Tokenize each declared string without special
  tokens using that exact tokenizer, require one distinct token ID per ordered label, then
  call the existing SDK initializer. No guess about a leading space. Record strings and actual
  IDs. Use bounded prepared training examples to build the mandatory step-zero input batch;
  record its deterministic selection/recipe without copying source text into telemetry.

Family/class comes from a fixed Forge registry matched against safe config, never a Python
import supplied in the spec. A future family needs a separate trusted initializer/test, not
an enlarged wildcard-key exemption. Initialize only after complete checkpoint validation;
at inference/reload every body/head key must exist and match. Receipts never exempt a head.

## Training population and rows

`dataset` has one exact dataset ref, explicit config/split, declared format, role→column map,
and bounded selection. Use ordinary current-actor read authority and closed native Parquet
files/card metadata. Reuse the installed bounded reader via a trusted adapter; no dataset
script, arbitrary local builder or hidden external file dependency. Do not fall back to the
legacy worker's unpinned `load_dataset` behavior.

Proposed formats:

| Format | Required roles | Optional roles |
| --- | --- | --- |
| `classification_text` | item_id, text, label | text_pair, label_dist |
| `nli_pair` | item_id, premises, hypothesis, label | label_dist |

Item IDs are exact distinct nonempty strings, not generated ordinals. Required labels remain
required with soft targets. ClassifierConfig's explicit aliases/index map decide label values;
no dataset-order inference. Distributions follow ordered labels, are finite/nonnegative and
sum to one without repair. Boolean labels are refused. First implementation exposes no class,
sample, task-group weight, smoothing, support or inconsistency role.

For `nli_pair`, explicit `VerifierConfig` is required and must agree on label order/max length/
classification-only outputs. Use the existing deterministic renderer and SDK rendering recheck.
The currently supported renderer input is a list of strings; unsupported premise shapes refuse.
Do not guess a conversion from PID objects or silently erase premise IDs. A broader signed
research row shape needs a reviewed pure mapping from Claude's metadata handoff before its use;
it does not require us to inspect research rows or prompt bodies.

Use explicit half-open selection with `row_start`, nullable `row_end`, `row_limit`, and
`mode=all|range`. At most100,000 selected rows, matching SDK scope. In all-mode, read one beyond
the cap to refuse oversize input; range-mode records its boundaries and actual selected count.
No silent prefix truncation, skipped malformed rows or invented total count. Per rendered row
use the SDK's existing field limits and max8192 tokens. For verifier rendering, preserve its
200,000 combined rendered-character and20-premise caps; these are that renderer's supported
scope, not universal platform gates. Generic text/text-pair fields retain their declared SDK
200,000-character-per-field bounds unless a separately explicit worker limit is reviewed.
`overflow=error|drop` is explicit; drop retains complete private item audit plus bounded counts
in telemetry, never truncates the hypothesis, and refuses an empty retained population.

## Objective and optimization

`training.labels` preserves2–16 ordered distinct nonempty labels≤64characters. Expose exactly
`objective_preset=cross_entropy|research_ce_brier_0_1` on this worker. The first maps to ordinary
CE/weight0; the second calls the existing named research preset/weight0.1 sum-form Brier.
An arbitrary Brier coefficient remains a Python SDK option, not an extra inert-worker feature.
Generic two-class CE is not the unresolved grouped entail-vs-not research objective.

The remaining fields map one-to-one to current ClassifierConfig: label aliases, integer index
map, max_length, overflow, epochs, max_steps, learning_rate, batch_size, seed, precision,
gradient_checkpointing, use_cpu, scheduler, warmup_steps and rendering. Preserve SDK bounds and
strict number types. Require `checkpoint_selection=final`, selection_split absent, no eval/gate/
test rows or metric-based selection. No auxiliary outputs/calibration/abstention are advertised.
Unsupported objective settings refuse before inputs or model load, not after provisioning work.

Use process isolation because seeds are process-wide. Actual SDK training runs off-loop; a
trusted progress sink reports actual optimizer start/steps with finite scalar metrics and a
single run-final end after export/verification/publication. No row IDs/text/logits are progress
payloads. No fake count or final checkpoint event is emitted before an artifact exists.

## Read authority and guarded writes

Keep policy and credential outside the spec. Reuse the exact ref/type/spec-bound decision shape,
but install a **training-purpose** authority adapter. Its credential is hash-only at rest,
run/spec/method/purpose/expiry bound and delivered in Task.secrets/private one-use file. No
session cookie, environment token, caller import, broad account capability or Eval token fallback.
Any supplied training secret is registered for redaction before failures are surfaced.

Server admission freezes exact model+dataset reads under User/account mutex and current actor
metadata. The worker approves the full closure before any tokenizer/model/row read, repeats
checks at strict boundaries, and keeps live reads fenced until confirmed process cleanup.
Fresh checks preserve every frozen ancestor even if a producer disappears. Metadata unknown,
positive custody, identity retargeting, new unresolved producer or changed required lineage
refuses. Use sorted User→Project→Run locks where those resources are involved; recheck expiry
and exact run state after waits/metadata before granting. No new research exception is implied.

Do not widen the existing Eval `_credential` kind check, route, public schemas or retry behavior.
New private training-policy routes/method payloads need coordinator wire review before code.
Adding all finetunes to PREPARED_READ_KINDS would break causal clients: use a classifier-specific
prepared-read selector, retaining existing Eval behavior.

For `output=null`, keep verified local artifacts and emit a fixed retained path after valid end.
For an output request, unit3 must obtain model-domain private create-only/existing-parent plan,
check input/output collisions and positive historical aliases, then acquire a writer fence.
If the accepted custody policy requires a protection marker, write its marker-only first
application-controlled commit before data. An ordinary private unsealed output must not gain
an invented `sealed=false` authority or weaken monotone source protection.
Publish only the verified safe model/tokenizer/sidecar inventory and accept the actual full
CommitRef receipt through training-purpose server authority. Never infer success from private=True,
read permission, uploader stdout, terminal message or a requested repo. Unknown mutation outcome
stays uncertain and fenced until trusted reconciliation. Reuse custody semantics, not blind push.

## Artifacts, checkpoints, resume and cleanup

Retain a private per-run attempt outside the disposable compute workdir:
`~/strata-classifier-output/<server_run_id>/attempt-<server_attempt_id>/`.
The installed worker generates paths and exclusively reserves a new attempt; a rerun uses a new
Run ID. The public spec accepts neither output directory nor arbitrary attempt ID. Do not
overwrite previous failures or silently restart/resume within them.

Separate evidence:

1. Existing `HeadInitializationReceipt`: exact copied/created/removed keys, labels, pooling,
   source-object transformation, resolved verbalizer IDs and step-zero check. No Hub/rights proof.
2. New bounded `training-receipt.json`: run+spec+engine, source model/tokenizer ref, dataset ref/
   config/split/selection and population digest/counts, objective/config/render digest, dependency
   versions, actual steps/trained/overflow counts and safe artifact inventory hashes. It reports
   what the trusted worker did; it is neither an access grant nor benchmark/calibration evidence.
3. Optional accepted `publication-receipt.json`: model repo+actual commit+expected parent+spec,
   separately bound to the server's trusted writer record. Persist even after later failure.

The SDK exports final safetensors/config/tokenizer plus classifier/render metadata. Before final
success, load it completely and score a bounded private prepared batch through the installed
same-label scorer. Guard source inventory, symlink/path/size limits and forbidden files. No
`*.bin`, pickle, optimizer state, Python/repo code or Trainer arguments enter the published model.

Periodic optimizer checkpoints and exact resume are unavailable in the first worker. Refuse
resume fields/paths. A failed/cancelled run can be restarted as a new record; never label it a
resume. Multi-day research requiring recovery needs a separate owner decision or resume increment
before spend. Future resume must bind base, data population, rendering, objective, optimizer,
scheduler, RNG, step and engine/dependency fingerprint; keep its state separate from publication
and refuse untrusted pickle. Final-only short useful jobs need not wait for that future feature.

Cancellation requests Trainer stop, then supervising group TERM/grace/KILL owns hard termination.
Releasing read/writer authority or deleting attempts requires confirmed terminal state; timeout
is not proof. Retain successful artifacts after upload failure, cancel and transient API loss.
Existing generic cleanup may remove only its transient workdir/secret file, not retained results.
Artifact purge/recovery and hosted multi-tenant isolation are separate reviewed platform work.

Classifier result projection is method-specific. `output_repo_id`, `output_commit_sha`,
`output_ref=repo@sha` come only from an accepted model receipt; `training_spec_sha256` binds the
run. With no output, expose only the computed retained artifact path after actual valid end;
do not imply a browser download or verified report quality. Requested output cannot create a
result link. Causal fine-tune destination handling and Eval result keys stay unchanged.

## Installed evaluation interop is first-slice acceptance

Recognize a bounded `strata_classifier.json` at the exact authorized model pin. Snapshot only
the required safe config/tokenizer/safetensors/index/sidecars under trusted file/byte bounds;
then call the fixed Forge `load_classifier` implementation. Metadata/config/family/label order
must agree, including the original receipt; no remote factory or fresh-head exemption. A malformed
recognized Forge artifact refuses rather than falling back to AutoModel. Ordinary native HF
classifiers retain their existing strict loading path. Decoder label-token scoring mode is unchanged.

Adapt the reloaded native object to the existing Scorer. It retains full-class probabilities,
padding/last-attended behavior, class ordering, length refusal and null unsupported outputs.
Run actual `run_evaluation` with the existing EvalSpec/modes/metrics, trusted fake policy and
fictional exact model@SHA whose installed-reader transport maps to the just-exported synthetic
artifact. Compare probabilities before/after save and compute actual report outputs. This proves
installed worker/eval interoperability without downloading or claiming provenance for a real model.
Missing/unexpected weights, labels, metadata, unauthorized/unknown pins and wrong purpose tokens
must fail. This acceptance belongs to TRAIN-001 and cannot be left as an unnamed later gap.
