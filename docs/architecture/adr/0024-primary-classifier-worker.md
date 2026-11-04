# ADR 0024: primary classifier workers require streamed inputs and training authority

## Decision

Provide a strict standalone `ClassifierTrainSpec` and an async trusted-Python execution API,
without activating the method registry, CLI launch, server dispatch or a credential transport.
Authority and source adapters are required installed objects outside the spec. Decisions bind
training purpose, run, full spec digest and exact typed input refs. Neither an Eval credential,
public/cached model nor serialized flag is authority.

Sources yield UTF-8 JSON row bytes incrementally. The worker checks selected-count cap+1,
individual byte and cumulative byte budgets before parsing the next row. Range skips are
explicitly bounded and count toward the byte budget. Exceeded budgets refuse rather than
return a prefix. These prototype execution limits do not redefine scientific populations.

Use the existing primary-classifier SDK's explicit CE or named CE+0.1 Brier preset and final-only
safe export. No adapter, publication, resume, weighting/smoothing, auxiliary or calibration
field is accepted, even null. A private local receipt records the actual selected population,
pins/config/digests/counts and artifact hashes; it is not a Hub or quality receipt.

Recognized full-pin Forge artifacts reload through fixed installed SDK classes and the complete
state guard. Only a definite missing sidecar chooses native HF loading; unavailable or malformed
recognized artifacts refuse. Legacy unpinned native loading remains compatible and does not
claim Forge-sidecar recognition. No head is exempted at inference.

Preflight typed Hub metadata binds exact full pins/filenames and advertised non-weight sizes
before materialization; unsupported or unknown metadata refuses without fallback. Verify exact
regular-file size and bounded local copy afterward. Enumerate repository files and folders
incrementally through expanded SDK pages and refuse at entry 129, before later pages or assets.
The supported Hub API returns 50 expanded entries per page. This does not claim a hard HTTP
wire-byte cap against a dishonest response, or introduce a weight-size threshold.

Stored single-sequence rendering can be consumed by the existing `hf_classify` evaluation mode
only after typed classifier metadata/native config agree, original/effective/SDK render digests
are identical and labels/family/length match. Stored label tokens describe initialization, not
evaluation verbalizers. Full learned-head class probabilities are used. Native paired and
label-logit behavior and EvalSpec's public shape remain unchanged.

## Consequences

Tiny randomly initialized CPU fixtures prove actual optimization, safe export, default installed
scoring and actual evaluation without downloading models or using research data. They prove no
scientific performance or remote production readiness. Missing training authority refuses before
reads. A cancelled native thread may still finish its own work; the worker coroutine alone owns
final receipt/progress and cannot emit later success after cancellation. Process supervision,
live read fences, private model publication and actual server/agent activation require separately
reviewed consumers. Retained attempts are never silently overwritten or treated as resumes.
