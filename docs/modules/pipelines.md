# Compute-target pipeline workers

`strata_forge.pipelines` contains fixed installed entrypoints for inert JSON run specs:
`inference_runner`, `finetune_runner` and `eval_runner`. They compose Forge primitives rather
than executing code supplied in a spec. Common execution loads/deletes the private secrets
file, checks the exact engine version, emits JSONL progress, redacts surfaced failures and
handles cancellation. Models and datasets use the explicitly delivered credential or `False`,
which prevents ambient-token fallback.

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
