# Installed primary classifier HF worker

`strata_forge.pipelines.classifier_worker` composes the existing primary classifier
SDK, exact-pin Hub inputs and the distinct training-purpose policy channel. Its
inert input remains `ClassifierTrainSpec`; no public field chooses a Python
executable, import, policy URL, credential or local source path. This module is
separate from the causal-LM fine-tuning registry and the population SDK.

## Installed execution

The operator provisions a pinned environment with Forge's training, HF and
compute dependencies before admitting work. The control plane invokes its
absolute interpreter with `-I -B -m strata_forge.pipelines.classifier_worker` and
an empty Task setup. A run does not fetch code, install packages or create an
environment. `STRATA_RUN_CONFIG` carries the canonical resolved specification,
including the actual Run ID and the engine version selected by the control plane.
Spec parsing and the installed engine handshake precede secret-file reads and
model/dataset I/O. The accepted path uses the common runner's secret
read-and-unlink, redaction, progress and termination handling.

The backend's outer shell trap owns the unread private secrets file when import
or preflight fails. The task command runs in a subshell, so its traps cannot
replace that owner. SIGKILL bypasses shell traps; backend cleanup is a separate
obligation. This worker introduces no second secret-file reader.

## Private authority and policy checks

`FORGE_CLASSIFIER_AUTHORITY` is a separate private Task secret. Its closed frame
contains `version=1`, `purpose=classifier_training`, actual `run_id`, complete
`spec_sha256`, operator-fixed `policy_origin` and `worker_credential`. The frame
is at most 8 KiB UTF-8. The credential matches the classifier domain; an evaluation
frame cannot start classifier training. It is not accepted as an environment
fallback and is never a public spec field.

The fixed endpoint is
`POST /api/worker-runs/{run_id}/classifier-training/reads`. Its body and result
reuse the exact closed `TrainingReadRequest` and `TrainingReadDecision`, without
an added RPC version field. Each request binds purpose, run/spec digest, exact HF
ref and model/dataset domain. A result must echo every binding, authorize the
read and report unsealed custody. The Server owns current source rights, purpose
eligibility, credential expiry/revocation and project/account scope; Hub access
or local cached bytes grant no authority.

The client accepts a fixed HTTPS origin or explicit literal-loopback HTTP for an
owned test deployment. It disables environment proxies, ambient cookies/default
auth/query parameters and redirects. Responses are uncompressed JSON of at most
16 KiB, reject duplicate keys/nonfinite values, and cannot contain rows or secret
fields. Only a 503 or transport timeout retries, at most three identical requests
inside the original thirty-second decision budget. Denials do not retry. Each
source boundary obtains a fresh decision; a previous positive grant is not cached.

## Input and population limits

The Hub source uses only the explicitly delivered HF credential or `False` and
the fixed public HF origin. It requires exact full commit refs, verifies the
returned revision, counts a lazy inventory through cap+1 and validates canonical
paths, metadata sizes and content hashes before local use. Model staging copies
only fixed inert configuration/tokenizer files and safetensors. Adapter discovery,
pickle weights, dataset scripts and remote code are refused.

The model inventory has at most 128 entries and 64 GiB selected bytes; individual
non-weight metadata is at most 32 MiB. Dataset inventory has at most 4,096 entries,
the required inert README configuration is at most 64 KiB and the selected
Parquet files total at most 256 MiB of advertised compressed bytes. Configuration
and split selection reuse the evaluation reader's inert selector. Only mapped
columns enter 32-row Parquet batches, capped at 8 MiB serialized JSON per batch.
Downloaded files are size/hash checked and staged privately. These advertised
preflight and post-download checks are not hard wire-byte, native decoder memory
or OS process limits; supervised process/resource ownership remains necessary.

The unchanged v1 runner additionally enforces 100,000 selected rows, 2 MiB per
raw row and 64 MiB cumulative source-row bytes, including skipped range rows.
Over-limit populations refuse instead of silently taking a prefix. This profile
does not claim complete large research-population support. The separate
complete-population Python SDK retains its own supported contract and is not
automatically selected by this installed entrypoint.

Only the primary SDK's fixed supported families, initialization/rendering rules,
CE or named CE+Brier objective and final-only selection apply. Auxiliary verifier
training, calibration, adapters, periodic checkpoints and exact resume are outside
this worker profile.

## Local outputs and completion

The exclusively reserved output is `$HOME/strata-classifiers/<run_id>`. The SDK
retains the safe reloaded model under `training/artifact` and a
`training-receipt.json` with actual input digest, configuration, counts and file
hashes. `worker-result.json` adds the run/spec/engine identity, relative artifact
and receipt names, receipt hash, steps and row/overflow counts. It reports
`publication=not_supported` and `resume=not_supported`; it contains no worker
absolute path. The identical canonical bytes also appear exclusively as
`classifier-result.json` in the admitted Job workdir, so the existing bounded
backend file reader needs no arbitrary remote path. The lightweight
`classifier_result.ClassifierLocalResult` codec accepts at most 8 KiB, rejects
extra fields, duplicate keys and nonfinite JSON, and fixes the relative filenames.
The poller must additionally match actual run/spec/engine and admitted row bounds;
the record itself grants no authority. Only accepted counts enter the existing
public metrics projection. Artifact publication remains unavailable until an
actual native output commit channel accepts it.

The installed worker emits its final `end` event only after source cleanup, fresh
model/dataset policy checks, HTTP-client closure and both exclusive result writes.
Primary load/decoder tokenization/initialization/training/reload/inventory, Hub source I/O and local
result writes execute in owned threads and drain before cancellation propagates.
Cancellation after a held load or initializer cannot admit a later optimizer;
training receives its cooperative stop event immediately and does not export
after observing it. Repeated cancellation cannot release staged files early, and
a cancelled call cannot later emit success. A native operation may still need
transport-supervised termination, and failure retains the exclusively reserved
output for inspection. Reusing a Run ID refuses overwrite: an explicit fresh
admission must allocate another real Run. An uncertain launch must not be
automatically replaced or represented as a resumed run.

See [pipeline workers](pipelines.md) for the underlying SDK/spec and
[training](training.md) for scientific configuration and local population APIs.
