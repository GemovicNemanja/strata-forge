# Compute-target pipeline workers

`strata_forge.pipelines` contains fixed installed entrypoints for inert JSON run specs:
`inference_runner`, `finetune_runner` and `eval_runner`. They compose Forge primitives rather
than executing code supplied in a spec. Common execution loads/deletes the private secrets
file, checks the exact engine version, emits JSONL progress, redacts surfaced failures and
handles cancellation. Models and datasets use the explicitly delivered credential or `False`,
which prevents ambient-token fallback.

## Primary classifier execution

`classifier_spec.ClassifierTrainSpec` is a strict, frozen inert primary-training specification.
`classifier_runner.run_classifier_training(spec, *, authority, source, attempt_dir, writer=None)`
requires installed trusted Python objects; it has no registry/CLI/server/Task-secret activation.
It does not accept an Eval authority frame, discover a policy from the environment, publish a
model, or authorize reads from a caller flag. The method name inside this schema does not make
`seq_cls` a registered causal fine-tuning method.

`TrainingReadAuthority.decide(TrainingReadRequest)` returns a typed decision bound to
`purpose=classifier_training`, run ID, complete spec digest, exact model/dataset ref and domain.
Only matching authorized/unsealed decisions permit reads. Unknown, positive, expired/revoked
adapter decisions and mismatches refuse; the eventual installed adapter owns actual expiry,
registry/marker/lineage freshness and live fences. The worker repeats decisions at strict
boundaries; synthetic test adapters are not production authentication infrastructure.

`ClassifierInputSource.rows(dataset)` yields UTF-8 JSON bytes as an AsyncIterator, not a tuple of
preloaded rows. `resolve_model(ref)` returns a trusted exact-ref local checkpoint directory,
never a JSON path or import. The worker selects the declared half-open range, enforces stable
unique string item IDs, checks mapped roles/labels/rendering and uses fixed native safetensor
loaders with no remote code, adapter discovery or fresh body/head loading exemption.

The prototype limits are exported from `classifier_spec`: 32 KiB spec, 100,000 selected rows,
2 MiB per raw JSON row and 64 MiB cumulative row UTF-8 bytes, including skipped range rows.
All-mode examines cap+1 before parsing that later row; range start is at most 100,000 and its
finite span cannot exceed its declared row cap. Exceeded bounds refuse with fixed codes; no
silent prefix, omitted malformed selected rows or inferred total count is reported. These are
execution bounds, separate from scientific dataset scope and the verifier renderer's limits.

`classification_text` maps item_id/text/label and optional text_pair/label_dist. `nli_pair` maps
item_id/premises/hypothesis/label and optional label_dist, requiring explicit VerifierConfig
and list-of-string premises. No PID conversion or rendering is guessed. Ordered labels,
explicit aliases/index map and distributions use the SDK's validation. The worker exposes only
generic CE or the named CE+0.1 Brier preset; no free coefficient, weights, smoothing, grouped NLI
binary objective, auxiliary head, calibration, adapter, publication or resume field exists.
Selection is final-only. Unsupported knobs refuse even when supplied as null.

The trusted caller provides an owned private empty attempt directory. It is exclusively
reserved; SDK work lives under `training/artifact`. `training-receipt.json` records actual
steps, counts, input population digest, complete config/render/initialization and file hashes,
including prototype byte limits. SDK metadata retains dependency/environment details. This
is private local evidence, not a downloadable browser result, rights proof or publication
receipt. Attempts/artifacts remain after failure; no worker purge or recovery command exists.
Only the coroutine writes final receipt/progress after complete reload and final policy check;
a cancelled native thread cannot later emit a success event. Supervised termination and lease
release remain required transport responsibilities. No periodic checkpoint or exact resume is
supported, and a restart is not mislabeled a continuation.

The loaded scorer supports an explicit per-call `overflow_policy="error"` for trusted Python
consumers. It refuses overlength calls before model forward and uses no truncation on accepted
pairs/single inputs, checking complete right-padded attention masks. Its legacy default and all
inert run specs/workers remain unchanged; they do not forward this option. This is not a
research-run eligibility flag or a drop policy. See the [scoring length contract](compute.md#exact-hugging-face-scoring).

Full-pin `hf_classify` probes a bounded `strata_classifier.json`. A definite file 404 permits the
unchanged native factory; other failures refuse. Recognized Forge artifacts load their safe
fixed file inventory and complete SDK classes. Metadata/config/receipt, exact class order,
token IDs when stored, and rendered metadata agree; missing or extra learned weights refuse.
Unpinned legacy native paths do not perform this probe and make no recognition guarantee.

Before materialization, the installed Hub `dry_run` metadata must bind the exact requested
filename/full commit and a known integer size: at most 64 KiB per Strata sidecar and 32 MiB per
other non-weight asset. Unknown sizes, unsupported preflight and mismatched pins refuse without
an unbounded download fallback. All selected metadata is preflighted before remaining assets
or weights. Downloaded metadata must then be a regular file with the exact advertised size;
its local copy/read is bounded. These are trusted Hub-advertised size checks plus postchecks,
not a hard wire-byte guarantee against a dishonest HTTP response. Weight sizes have no added
prototype threshold. Repository discovery uses lazy expanded tree pages (50 entries per page
in the supported Hub API), counts files and folders, and refuses at entry 129 before requesting
later pages or assets. One SDK page is decoded before its entries are yielded; this is not a
custom bounded HTTP/JSON transport. Fixed permitted asset names and safe local loading still
apply. The integration requires the installed Hub's typed dry-run and lazy-tree APIs.

Recognized pinned Forge classifiers accept a single rendered sequence with rank-two logits and
full-class softmax. Combined inputs exceeding the declared length refuse rather than cutting a
hypothesis. Forge decoder classifiers refuse tokenizer-pair input. Native HF classifiers retain
their paired-input/first-sequence truncation contract and label-logit scoring is unchanged.
Each batch has one input shape. Evaluation's existing model_artifact source supports the single
sequence only when original, effective and typed SDK rendering digests are identical and typed
native config matches family/labels/length. No override drift is accepted on that path. Stored
label_tokens remain initialization provenance; report rendering audit distinguishes them from
the actual `learned_head_full_class` inference mode. EvalSpec, its wire digests and retry behavior
remain unchanged. See [ADR0024](../architecture/adr/0024-primary-classifier-worker.md).

## Classification evaluation

`EvalSpec` lives in `strata_forge.pipelines.evaluation_spec`; its `model_json_schema()` exports
the installed structural contract. Cross-field validators are also authoritative. Import the
model to validate a submission; JSON Schema alone does not implement all semantic checks.
The spec is frozen/strict, rejects unknown fields, and has `schema_version=1`.
Use `spec.wire_payload()` for JSON-native serialization and `EvalSpec.model_validate(payload)`
for the supported roundtrip. It preserves field presence: prediction mode forbids model fields
including explicit null placeholders, so a generic model_dump that inserts defaults is not its
wire format. `spec.digest()` hashes that canonical payload and binds the full stamped job.

The root fields are `engine_version`, `run_id`, `progress_path="progress.jsonl"`, pinned
`dataset_id`/`dataset_commit_sha`, optional `output_repo_id`, `mode`, `evaluation_sets`, and
explicit capability fields: `sealed=false`, empty `echo_columns`, `apply_calibration="none"`,
`bootstrap=null`, and declared `holm_families`. These capability values authorize nothing.

Each set declares `evaluation_id`, config, split, stable string ID/gold columns, integer-gold
index map, ordered target labels, `label_space`, projection, row selection, slice columns,
cluster column, sampling unit, diagnostic flag, truncation policy and comparisons. No global
accuracy or macro-F1 pools configs, splits or label spaces. `2way_entail_vs_not` describes a
caller-declared binary task; a not-entailed label is not silently renamed contradiction.
Integer labels require an explicit index map; booleans/floats are not guessed class indices.

`classify` additionally requires a full pinned model revision, `hf_classify` or
`hf_label_logits`, source `model_labels`, aliases/verbalizers, rendering, batch size and dtype.
Its sets map `premises` and `hypothesis` onto real columns. `score_predictions` forbids the
model-side fields entirely, including null placeholders; each set instead declares source
labels, pinned predictions ref/config/split, prediction-ID column, optional known producer
manifest and a string `parse` discriminator with its strict `parse_config`.

Supported parsers:

- `probability_columns`: explicit source label-to-column map, full-class or restricted-softmax
  provenance, optional outside mass, and either target argmax or an explicit hard-label column.
  Complete probabilities must be finite [0,1] and sum to one within `1e-6`; malformed or partially
  missing vectors fail the job. All-null vectors are absent. Projection sums probabilities
  before target argmax. A separate invalid hard label stays wrong even when its forecast exists.
- `json_field`: native scalar/object or bounded JSON text, explicit object-key path, aliases
  and integer mapping. Malformed/oversized text or absent/invalid labels stay wrong with fixed
  codes. No probability vectors are fabricated. JSON text is capped at 65,536 UTF-8 bytes and
  paths at eight keys.

Stable joins preserve gold order and all selected gold items. Missing predictions remain
wrong; duplicate/extra prediction IDs, invalid gold/IDs and mismatched producer identity fail.
`scoring.json` verifies stable ID declaration, gold pin/config/split, selection, source labels
and probability family. `metrics.json` verifies the same population against its declared
evaluation entry and effective spec. An omitted manifest is explicitly external-declared
provenance, not proof of Strata production.

## Read policy and native files

`run_evaluation(spec, *, token=False, policy=None, publication_policy=None, reader=None, writer=None, output_dir=None)`
is async. `ReadPolicy`, `ReadDecision`, `EvaluationReader` and `EvaluationError` are exported
from `strata_forge.pipelines.eval_runner`. Trusted Python callers can provide an installed policy/reader; no spec field or CLI
option can choose a policy implementation. `ReadPolicy.decide(ref, repo_type=..., spec_sha256=...)` returns a strict
`ReadDecision{ref,repo_type,spec_sha256,authorized,custody,policy_version}`. Only an authorized `unsealed` decision matching the exact ref, repository type and full
`EvalSpec.digest()` permits access. Model/dataset domains never share a grant by name alone. Unknown metadata, sealed/locked state, resolver errors and ref
mismatches fail before model/row access. Trusted resolvers must combine canonical and historical
aliases, portable markers, registry facts and lineage monotonically; a positive fact cannot
be erased by an alias or caller declaration. The default standalone worker denies access.

An application installs a fixed code adapter that verifies its authenticated issuer's read
approval and rechecks fresh custody/authorization facts. It passes that adapter to the worker
through code. An injected synthetic test resolver is not a production approval mechanism.
Neither Hub public readability nor successful token/cache access grants permission.

`HubEvaluationReader` confirms actual immutable revisions, uses explicit native Parquet files
from the pinned README `configs/data_files`, and follows no dataset scripts or external URLs.
Model preparation refuses adapter-only repos; implicit base reads are outside this reader's
contract. Safetensors/native classes and the complete loading guard precede row materialization.
Loads retain bounded key/count/code audit and actual tokenizer/model pad policy. Loading refusal
saves a bounded local `loading-refusal.json`, without arbitrary upstream diagnostics.

Execution bounds: at most 64 sets and 64 distinct primary refs, an explicitly finite selection
of at most 1,000,000 row memberships per job, at most 32 native Parquet files per read, and a
256 MiB local file cap. The reader batches row iteration and applies `[start,end)` then row_limit
before Python row materialization. It downloads native files to local disk; these limits do
not imply zero-copy, constant-memory or streaming Hub transport. Each prediction read is capped
at 1,000,000 rows. Metadata/report JSON is bounded, and verifier metadata has a 64 KiB cap.

## Rendering and outcomes

`training.verifier_config.VerifierConfig` is pure versioned rendering metadata. Explicit inert
overrides or `strata_verifier.json` at the model SHA supply exact templates, premise format,
ordered labels and declared limits; absent rendering refuses. Overrides win and both original
and effective config digests are recorded. Only `{premises}` and `{hypothesis}` substitute;
row braces are not reinterpreted. Numbered lines use LF with no trailing LF; JSON is compact
UTF-8. There is no Jinja execution or Python attribute/format lookup.

This renderer supports list-of-string premises plus a string hypothesis, declared
`max_premises≤20`, `max_length≤8192`, and 200,000 combined rendered characters. Lower limits
are explicit caller choices. These are the supported renderer scope, not universal platform
or research acceptance rules. Other task shapes use their own workers. Over-limit premises,
bad shapes and character overflows are retained as refusals; no hypothesis is silently sliced.

HF scoring can report premise truncation. Each population explicitly chooses `retain_reported`
or `count_as_invalid`. The latter retains the item as wrong and excludes its vector, with
coverage/exclusion reason visible. Overflow and truncation counts are preserved in both cases.
Missing/invalid/refused outputs count wrong for accuracy/F1. Missing vectors affect only
probability coverage, with no renormalization or one-hot repair. A completed all-wrong evaluation
is a measured outcome, never a model acceptance signal; no performance threshold is implicit.

## Artifacts and statistics

Each fresh local attempt contains:

```text
README.md
metrics.json
evaluations/<evaluation_id>/items.parquet
```

The card points the Viewer only at each declared Parquet config. `metrics.json` uses
`strata.eval.metrics.v1`, records effective spec/digest, exact role-tagged input refs,
implementation versions, source/renderer/load/pad/probability context, counts, per-set reports,
independent single-column slices and explicit comparison families. Raw row content, rendered
prompts, responses and reasoning are excluded. Optional head/calibrated outputs are typed
null/not-produced, rather than false zeros. Brier uses the multiclass sum; NLL records its
clipping/coverage through the shared exact `MetricValue` model.

Tables declare Arrow types before writing: IDs/gold/status strings, nullable canonical pred,
fixed error code, correctness/overflow/truncation bools, nullable float64 raw/calibrated class
columns, probability context, source IDs/refs, cluster ID, outside mass, nullable inconsistency/
support/abstention fields, and ordinal slice columns. Slice columns are typed lists of tagged
structs so null, bool, integer and string values remain distinct. A map retains source column
names. Generated probability-column collisions refuse; labels are never silently sanitized.

Slices deduplicate list membership per item without forming Cartesian products. Bounds are 16
columns, 16 members per cell, 256 distinct slice memberships per population and 32 MiB metrics
JSON. Reports exceeding a bound fail, not truncate. The declared diagnostic flag and n<500
diagnostic rule are recorded; they do not select favorable slices.

Paired comparisons require exact IDs, gold identity, pin/config/split/selection, label order
and projection. Invalid outputs remain wrong, and accuracy delta is candidate minus baseline
over all pairs. Exact McNemar and explicit-family Holm use the shared comparison helpers only
under the declared independent-item contract; repeated/missing base groups withhold inference
as `requires_cluster_method`. No hidden alpha, rejection or model promotion is added. Intervals
are not requested by this schema. Calibration, clustered bootstrap, ECE, support/inconsistency
and selective-prediction methods are represented as not-produced; their unimplemented requests
are rejected rather than simulated.

At most 64 finite headline count values enter progress. Full reports stay in artifacts.

## Private run authority and publication

`execute_evaluation` is the installed CLI/worker entrypoint. Its fixed `EvaluationPolicyRPC`
adapter consumes a `STRATA_EVAL_AUTHORITY` string only from the authenticated Task secrets file,
read and unlinked by shared execution. The masked version-1 JSON frame has exactly `version`,
`run_id` (32 lowercase hex), `spec_sha256` (64 lowercase hex), `policy_origin`, and
`worker_credential` (`strata_worker_` followed by 43 base64url characters). The origin comes
from trusted operator configuration: HTTPS, or literal loopback HTTP for local review, with
no userinfo, path, query or fragment. Environment copies are discarded, never used as authority.
The actual `wire_payload()` digest and Run ID must match before a request. No serialized spec,
CLI option or import path chooses an adapter. Missing authority leaves execution closed.

The adapter POSTs only `/api/internal/evaluation-policy/read` and
`/api/internal/evaluation-policy/output/{plan,ready,receipt}`. Every request and response carries
version, run and spec bindings. Read replies also match the exact typed pin. Requests use only
the run-scoped Bearer credential, no cookies, ambient proxy credentials or redirects; JSON
requests are capped at 4096 bytes, replies at 16384 bytes, and each call at 30 seconds overall.
Within that same overall budget, HTTP 503 and HTTP transport timeouts permit at most three
identical attempts, separated by 0.25 and 0.5 seconds. Every attempt clears cookies and reuses
the exact encoded body, origin/path and run-scoped credential. Permission denials, other status
codes, redirects, other transport errors and invalid replies are never retried; cancellation
propagates. A lost receipt reply can replay the same immutable receipt, without repeating the
Hub creation or content commit. These retries never renew a grant or bypass fresh server checks.
Malformed, unknown, denied or mismatched replies refuse with fixed errors. Server expiry,
terminal/cancellation checks, fresh custody resolution and active Run fences remain required
server responsibilities; the private frame itself is not a portable signed authorization.
The maintained synthetic wire fixture is `tests/unit/pipelines/fixtures/evaluation_policy_rpc.json`.

A non-null `output_repo_id` needs a separate `PublicationPolicy` and explicit Hub credential,
otherwise it refuses before input reads. Read approval, lineage and private visibility never
provide write permission. A preflight output plan runs before rows are materialized. At upload,
existing destinations require exact approved canonical identity, fresh private/protection checks
and a full expected parent. Create-only plans perform atomic `create_repo(private=True,
exist_ok=False)`, refuse conflicts, then obtain fresh server readback and the actual parent;
ambiguous absence never becomes permission to upload. Failed readback can leave an empty
private repository; the worker does not delete it. Known input-dataset/output collisions refuse.
The server must compare canonical identities and verified alias closures, including protected
prospective names; worker metadata checks add privacy/parent verification and do not replace
server custody checks.

Under the ready Run fence, only the card, versioned metrics and exact declared item files are
uploaded in one `commit_folder` with `parent_commit=expected_parent`. Local failure/audit files
are excluded. The actual returned `CommitRef` is sent to the receipt endpoint and must be
accepted before the successful end event contains `canonical_repo@actual_sha`. The artifact
cannot contain its own future commit SHA; immutable publication identity is carried separately
by the progress/receipt contract. No output SHA is inferred from a moving branch.

Local output directories are reserved for one attempt and are not overwritten. A complete
local-only result has `output_ref=null`. Publication attempts retain bounded local status,
parent and actual commit when known. Cancellation emits no successful end/receipt; an upload
already underway is shielded and observed for up to ten seconds, recording a known commit or
`commit_outcome_unknown`. Network/process death can leave a successful mutation with no
accepted receipt; retries must reconcile rather than claim rollback or overwrite evidence.
Supervised process-group teardown remains necessary because coroutine cancellation alone does
not prove that model threads stopped. This is an ordinary unsealed worker on trusted compute,
not a general sealed execution guarantee. Server installation and end-to-end consumer review
are required before enabling launches.

## CLI and execution profile

`strata-forge eval schema` exports the installed schema. `eval validate-spec FILE` validates
without resolving or authorizing inputs. `eval classify FILE` and `eval score-predictions FILE`
delegate to the same worker and refuse without a trusted installed policy. Existing
`eval run/list/show` behavior is unchanged. The fixed `finetuning,storage` profile constructs
both new modes in the clean-install smoke; it is deliberately a heavy HF execution profile.
The point/comparison arithmetic itself is pure Python and requires no NumPy extra.

Batch scoring `RunSpec.id_column` optionally adds stable `item_id` output alongside unchanged
ordinal `custom_id`. IDs are exact unique nonempty strings, bounded to 1024 characters, and
the scoring manifest records source/artifact column semantics. It is available only for HF
scoring engines; existing ordinal artifacts are not reinterpreted as stable-ID evidence.

See [metrics](metrics.md), [training](training.md), [storage](storage.md) and
[the worker boundary decision](../architecture/adr/0021-pinned-evaluation-workers.md).

## Population-v2 specification boundary

`scoring_population_spec.InferencePopulationSpecV2` and
`evaluation_population_spec.EvaluationPopulationSpecV2` are strict operational envelopes
around the existing inference and evaluation task models. They require version2, an explicit
protocol and `overflow="drop"`; a dataset suite-filter column is explicit, with null meaning
no supplied suite filter. Tier metadata has a64-distinct-value/128-byte-per-value budget.
Those are prototype metadata bounds, not scientific tier definitions.

The nested legacy job remains authoritative for its task-specific fields. Wrapper serialization
retains nested field presence: absent generation defaults never become present-null keys that
could change the profile or digest. `wire_payload`, ordinary model/JSON serialization and
revalidation agree. `digest()` binds the complete wrapper, including population columns,
selection and every condition's pinned inputs. The inference envelope requires exact model/
dataset SHAs, finite selection, stable IDs and HF scoring knobs; generation/serving fields and
remote output destinations refuse. Evaluation envelopes cover each declared set exactly and
retain the existing strict EvalSpec task validators. Legacy RunSpec, EvalSpec v1 and their
serialization/digests remain separate.

The pure `_population` metadata contracts retain successful, invalid, native-overflow and
suite-excluded outcomes separately. Integer counts satisfy `N=E+D+S+F` and `M=S+F`; failed
outputs are never called successfully scored. These types and counters do not grant access or verify an imported overflow assertion.
Installed CLI/registered workers do not accept these envelopes.


### Trusted local population execution

`inference_runner.execute_population_scoring(spec, *, token=False, policy=None, reader=None,
writer=None, output_dir=None, scorer_loader=load_scorer)` and
`eval_runner.run_population_evaluation` with the same keyword shape revalidate a serialized
wrapper copy, check the engine handshake, then authorize every exact role/ref against the
**outer** digest before source/model access. Missing/unknown/positive custody or inner-only
approvals refuse. Policy and test-only resolution hooks are installed Python code, never
serialized imports or callbacks. Inference uses the same bounded native-Parquet reader as
evaluation; the legacy dataset backend is unchanged.

Native execution reuses safe complete-checkpoint loading, supported paired classifier or
single label-logit shapes, existing VerifierConfig/SDK rendering identity checks, and
`Scorer.score_with_outcomes`. Suite-excluded rows never render or tokenize. Over-limit inputs
receive `outcome="overflow"`, exact token_count/model_limit and no prediction/forward. Equal
limits remain eligible. Render/input refusals remain invalid and wrong; unexpected loader,
model, tokenization or batch errors fail the attempt. No implicit seq2seq/chat-template or
additional architecture support is enabled.

The local scoring result has `directory`, `report_path`, full `spec_sha256` and typed counts.
`scoring.json` uses `strata.scoring.population.v2` with all selected rows in results.parquet.
Evaluation writes `strata.eval.metrics.v2` and all selected memberships in each
`evaluations/<id>/items.parquet`. Nullable gold/correctness/vectors on excluded rows never
create a false zero. Counts distinguish suite exclusions E, native overflow D, valid S and
ordinary invalid F; metrics use M=S+F. Failed hard labels can retain independently valid
vectors. All-excluded/overflow attempts retain complete counts and explicit
no_metric_population status; all-invalid metric populations remain all wrong.

Reports retain per-tier counts, selected/metric/target identity hashes, ordered label and
probability semantics and each model/render condition. The declared effective spec includes
static templates and is private audit data; item tables omit dataset text, rendered requests,
responses and reasoning. Public result projection is a separate integration. Attempts reserve
an empty current-UID directory with exact POSIX 0700 permissions, refusing existing public,
non-owner or symlink leaves without changing their permissions. Newly created ancestor/pending/
audit directories are 0700 and owned files (including temporary metric tables) are 0600,
independent of umask. Existing parents remain caller-managed; a public parent does not grant
read access through the private leaf. This is a trusted local POSIX path contract, not a
general hostile-filesystem capability. Writes run in a pending subdirectory off-loop; the complete manifest moves
last only after the awaited write and an active owner-task check. Canceled native/write threads
cannot publish a later final manifest/end event. This is not proof of process/GPU teardown.

Imported predictions keep honest caller-reported selection semantics. Raw overflow/token-count
columns do not exclude a row. A request to import a population-v2 manifest as verified drop
refuses until separately reviewed receipt evidence or exact recomputation exists. Legacy and
external ordinary invalid/missing forecasts remain usable and wrong; they are never promoted
to native untruncated execution evidence.

V2 paired comparisons require the same declared gold/source population and exact M IDs/gold,
with compatible target labels/metric semantics. Model/checkpoint/tokenizer/source projection/
condition-specific rendering may differ and remain provenance. Different drop sets refuse;
no silent common subset or altered Holm family is formed. Stored baseline report consistency
is not producer execution attestation. Existing independent/clustered inference eligibility
and metric helpers remain authoritative.

Remote output destinations refuse in this profile. No installed CLI, registry, private secret,
server/app/agent route, research admission, engine release or source pin is activated by these
trusted Python functions. Their synthetic CPU/Parquet acceptance does not execute research.

## Local classifier latency and throughput

`ClassifierBenchmarkPlanV1`, `ResolvedBenchmarkTarget`, `benchmark_scorer` and
`benchmark_verifier` provide a separate bounded local SDK. Package exports load lazily and do
not register a worker or CLI. The native objects are already resolved; neither function loads,
writes or publishes artifacts. Declared full model/tokenizer refs are metadata rather than
origin/custody attestations. Inputs and native aliases remain caller-owned and serialized.

The first profile admits only CPU/FP32/concurrency-one BERT, GPT-2 and ModernBERT decoder
models and fixed Forge wrappers with matching actual context/input embedding layouts. Unknown
layouts, other devices/precision, compiled or quantized models refuse without fallback. This
does not narrow the independent scorer's family support. Plans explicitly declare batch size,
warmup calls, measured calls and elapsed milliseconds; no warmup/probe/autotuning is hidden.
Inputs are bounded to 128 rows, 256 KiB per source row, 8 MiB total source, 32 MiB encoded
inputs, supported native context and 100,000 row applications including warmups. These are
admission limits, not activation-memory or VRAM guarantees.

`hf_scoring_sdk` measures the whole accepted scorer call with `overflow_policy="error"` and
no progress callback. It includes local executor queueing, length scan/grouping, tokenization,
all native batches, host conversion and result construction. A call may contain several native
batches and is not a per-row latency. `verifier_preencoded_sdk` measures accepted scoring of one
fixed encoded batch, including mode handling/postprocessing; it is not forward-only.
`verifier_encode_score_sdk` includes actual rendering/tokenization/geometry and scoring for
one fixed batch. Neither verifier mode silently re-batches. No original source is inferred
from preencoded tensors or their geometry hash.

Warmup calls are counted separately and excluded from measured quantiles/rates. Named
nearest-rank p50/p95/p99 preserve their sample ranks and empirical CDF step; one sample yields
identical percentiles without a confidence claim. `rows_per_second` divides all successful
measured row applications by the whole measurement window through output validation.
`sdk_active_rows_per_second` uses summed SDK durations and excludes benchmark validation/gaps.
Amortized time per row is labelled amortized, not an observed row percentile. Zero elapsed
rate denominators yield null plus `zero_elapsed`, never infinity. No output is dropped,
truncated, silently repaired or normalized to make a failed call appear faster. Restricted
label-token softmax remains distinct from full-class forecasts; support scores remain raw.

The fixed fingerprint binds input order/shapes/markers and tokenizer semantic backend/special
IDs. It excludes only fast-tokenizer backend `padding`/`truncation` runtime state which the
SDK explicitly overrides. This permits zero declared warmups without manufacturing an untimed
probe. Semantic tokenizer/input changes refuse the result. Fingerprinting/rechecking occurs
outside measured call/loop windows but inside the total elapsed budget. No weights are hashed
or loaded inside a timer. Model training/device/optimizer operations through other aliases
must be serialized; the SDK cannot detect every external alias or mutation restored in place.

A complete `ClassifierBenchmarkResult` requires every declared warmup/measured call and every
row to succeed. Execution failure or soft deadline returns an incomplete status with fixed
reason, partial bounded timing/count audit and null headline statistics. Invalid admission
raises fixed `BenchmarkError`; missing extras raise a useful `ImportError`. Cancellation
propagates after draining the outstanding SDK operation and returns no result. Cancel during
verifier encoding prevents the subsequent score leg. An already admitted scorer call may
finish its internal batches while draining; no later SDK call starts. Native calls are not
forcibly killed, and an overrun never becomes a complete result.

Cold start, isolated forward, separate tokenization timing, accelerator/serving latency and
remote queue/artifact time are explicitly unavailable. Results contain no raw input/IDs,
predictions, credentials, model repr, hostname or environment variables, and are not automatic
tracing or agent grounding. Comparison consumers must compare matching workload, boundary,
settings, heads, dtype/device and runtime conditions rather than pooling incompatible results.
See [the benchmark decision](../architecture/adr/0028-bounded-local-classifier-benchmarks.md).

## Separate generation contract

`generation_spec.GenerationRunSpecV2` uses protocol `strata.generation.v2`; it is independent
of legacy `RunSpec` and the classification-evaluation/scoring population schemas. It carries
exact model/dataset pins, one finite Selection, stable ID column, inert column mapping,
explicit serving/generation controls and local-only output. `parse_generation_spec` performs
bounded strict lexical JSON validation before Pydantic. `revalidate_generation_spec` copies
mutable nested caller data before any future operation.

`_generation_samples.plan_samples` validates all supplied IDs and selection/sample caps before
returning a lazy ordered item/sample iterator. Each item has1–16 samples with its original ID
and0-based sample_index. The domain-separated SHA256 seed policy includes source pin/config/
split, base seed, item ID and sample index, independent of row order, run ID or model condition.
Effective request seeds are decimal strings to preserve int64 precision in future JSON clients.
The alternative engine_default policy leaves request seed unknown; it does not mean random.
No deterministic model-output or independent-draw guarantee is made.

These source primitives perform no source/model IO, engine preparation or generation, and do
not register a CLI/worker/server capability. A logical sample or requested profile is not
native evidence or read authority. See
[ADR0029](../architecture/adr/0029-closed-local-generation-protocol.md) for the constrained
provider path and separate installed-native, policy and operational gates.

## Registered local temperature fitting

`TemperatureFitPlanV1` and `fit_registered_temperature(plan, *, source, policy)` provide an
independent local raw-logit fit SDK. The closed plan has protocol
`strata.eval.analysis.local.v1` and only `mode="temperature_fit"`. There is no advertised
bootstrap mode, worker/CLI registration, model loader/forward, artifact writer, output repo,
credential or policy discovery. Existing EvalSpec/population/default/RPC wire is unchanged.

`TemperatureFitSource` exposes metadata and a closeable async stream; `TemperatureFitPolicy`
issues current `RegisteredFitDecision` values. Both are trusted installed Python seams,
never imports or grants reconstructed from caller JSON. No production registry/issuer is
installed. Missing policy refuses. Ordinary read permission alone does not grant fit
eligibility, and source refs/content hashes do not attest origin or resolve custody.
The caller must supply independent owned source streams and cannot treat an arbitrary Python
object's matching fields as authentication.

`TemperatureFitSourceMetadata` binds full typed dataset/model/tokenizer refs, optional actual
prediction ref, config/split, column roles, explicit ordered labels/integer map, expected
count, normalized-row digest, source-receipt/head/rendering identities and input kind.
Only identity projection and a complete named split are admitted. Full-class versus
restricted-label logits map to distinct component probability families; temperature never
reconstructs outside mass, full vocabulary or a grouped binary objective. Current scoring
SDKs expose probabilities, so a live exact-logit capture adapter remains a separate task.
No inverse-log/clamping or extra hidden model call is performed.

Fresh policy must cover every exact ref/domain with unsealed canonical resolution and a
matching registered calibration-fit role/revision/count/digest. Unknown/positive protection,
missing refs, held-out/test/gate roles, ho_dev or its known aliases refuse before iteration.
Current registration/canonical binding is frozen through the operation. Explicit monitoring
or selection reuse remains recorded, not inferred independent. No scientific data or
checkpoint choice is authorized by the interface itself.

`temperature_fit_row_bytes` defines `strata.analysis.fit.rows.v1`: sorted compact UTF-8 JSON
of item_id/gold/logits, numeric logit values normalized to binary64, followed by LF. Scalar
integer gold requires the explicit map. The full stream must have the expected count and
unique IDs and match the declared digest. No prefix, row exclusion, null target, probability
input or weighting is inferred. Inputs are copied into owned numerical data.

Admission limits are 100,000 rows, 16 KiB per normalized record, 64 MiB cumulative record bytes,
256 MiB planned numerical working data and an explicit soft budget up to one hour. Known
codec dimensions are checked before storage/math; admission arithmetic is not an RSS or
caller-memory guarantee. Point helpers run in chunks up to 1024 rows; derivative work is
stepped in chunks up to 8 passes. Policy/elapsed checks occur at bounded chunks, before rows,
and after all final awaited metadata/version work. A single awaited operation may overrun;
there is no hard thread/process timeout.

The frozen `TemperatureFitReceipt` records actual count/bytes/input digest, exact plan/source,
solver status/T/unclipped objective, component content digest, before/after point NLL/Brier
with explicit clipping/coverage, full initial/final registry-and-canonical-read snapshots,
decision-chain hashes/check counts, elapsed time through final policy and software versions.
No raw IDs/gold/logits, hostname/env capture, traces or native errors are included. Its
component/receipt digest is content consistency, not persisted artifact identity, scientific
quality, model resolution or authorization. No verified application consumer is supplied.
Before/after clipped NLL can differ from the unclipped fit objective by design.

Cancellation drains the owned read/policy/math step and closes the stream; repeated cancel
and cleanup errors cannot emit a late receipt. Admission, revocation, missing extras,
malformed source and soft deadlines return no partial T. Errors are fixed and content-free;
missing NumPy has a scoring-extra installation hint. Optional numerical availability is
checked after policy but before row iteration. Public execution stays async; the pure
metric exception is documented in [ADR0030](../architecture/adr/0030-registered-local-temperature-fitting.md).

## Registered local paired cluster uncertainty

`paired_registered_cluster_uncertainty(plan, baseline=..., candidate=..., policy=...)` uses
closed `PairedClusterPlanV1`/source/row values in `cluster_analysis_spec` and explicit trusted
Python `PairedClusterSource` and `PairedClusterPolicy` protocols. No registry, issuer, loader,
worker/CLI, artifact writer or policy-RPC activation is installed. Source objects cannot be
selected by serialized imports/URLs/callbacks, and metadata must not materialize rows.

Metadata and a current exact typed read/population-registration decision precede NumPy and
both row iterators. Unknown/protected/denied/missing facts refuse. Decisions bind the complete
plan/source metadata, exact canonical refs and counts/digests; same-name model/dataset pins
need separate typed coverage. The canonical registration/reads stay fixed, with checks during
1024-row read chunks, before/after numerical preparation, at most every 100 replicates and
following final fresh metadata. Readable data or a caller's SHA/registration flag is no grant.

Both streams are consumed through EOF with exact row counts/digests and unique stable IDs.
The join requires identical M IDs, normalized explicit gold and exact base_problem_id under
the same target/config/split/label/projection/slice context. Models/renderers may differ.
There is no positional join, intersection, guessed integer map or source-name authorization.
Each source retains N/E/D/S/F/M; S/F and invalid-origin subdivisions are recomputed from M,
while outside-M N/E/D remain declared source evidence. Invalid hard labels stay wrong but may
retain valid independent vectors. Complete compatible probabilities are mandatory if Brier
or NLL is requested; hard-only analysis may retain partial-vector coverage without emitting
probability intervals. Raw versus declared-applied calibration remains provenance, not a
verified calibration-application claim. This operation performs no fitting.

`paired_cluster_row_bytes` emits the strict normalized UTF-8 source codec, with explicit gold
representation, float64 probabilities, sorted compact keys and LF. It grants no access.
Combined source bytes are limited to 64 MiB, rows 16 KiB, metadata/receipt 64 KiB, M 100000 and
planned numerical arrays 256 MiB. The fixed10000/seed 20261104 row-pooled profile follows
[ADR0031](../architecture/adr/0031-paired-local-cluster-percentiles.md). Larger populations
refuse rather than silently taking a prefix or changing existing worker caps.

Every read/policy/numerical chunk is owned and shielded/drained on repeated cancellation;
streams close before release. A soft budget can be exceeded by one native call; no partial
result or late complete receipt follows cancellation, denial or deadline. Fixed errors carry
no raw diagnostics/context. Receipts contain metadata/checksums/points/intervals/versions,
never raw row/group labels, vectors, draws or replicate arrays. Their hashes are consistency,
not source/custody/independence attestation. Existing fit/EvalSpec/population/scorer behavior
and worker purpose/wire remain separate and unchanged.
