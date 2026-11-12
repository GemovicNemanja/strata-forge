# Private immutable input reader

`pipelines._immutable_input.ImmutableInputReader` is an injected asynchronous byte reader
for an exact native model, dataset or prompt-build manifest payload. It is private: no
package export, worker registration, CLI, legacy Hub-reference change or native pipeline
consumer is installed. Existing [pipeline policies](pipelines.md) remain independent.

The installed caller supplies both an `ImmutableInputAuthority` and an
`ImmutableInputSource`. Missing or malformed authority refuses before source resolution.
An inert specification, readable URL, local path, digest, caller Boolean or previous
successful read cannot select an authority or grant access. The implementation supplies
no server authority producer, policy RPC, credential transport or authorization token.

## Exact request and policy observations

The closed `ImmutableInputRequest` binds the principal, project, run, attempt, explicit
purpose, outer admitted-spec SHA-256, resource kind and complete `NativeVersionRef`
(`resource_id`, `version_id`, `manifest_sha256`). Its `NativeManifestPayload` binds every
entry field: `artifact_id`, `logical_name`, `sha256`, `size_bytes`, `media_type` and
`schema_id`. These shapes preserve server manifest identity without serializing a
delivery location. Values are copied and revalidated rather than trusting constructed or
mutated Pydantic instances. The reader does not independently prove that a declared
entry belongs to a declared manifest; the installed authority must verify that relation.

The supported purposes are exactly `evaluation`, `inference`, `classifier_training` and
`prompt_render`. They name the operation the installed authority must authorize; they
are not interchangeable grants. Generic `read` or `training`, temperature-calibration
fitting and paired cluster uncertainty are unsupported. This reader neither supplies a
checkpoint/population registration nor authorizes fitting, scoring or execution.

The mandatory trusted authority performs actual current rights and custody checks for
the complete request in the original caller task. Its descriptive decision must return
that exact request, `allow`, `unsealed` custody, a stable policy identity and revision,
and a current expiry. Denial, unknown/protected custody, mismatched scope, changed policy
identity/revision or expiry returns no payload. Current checks precede resolution and
each bounded read, surround actual FD reads and follow completed cleanup. The result's
bytes and digest never become reusable permission. Already disclosed bytes cannot be
retracted when rights subsequently change.

## Bounded operational ownership

This profile admits at most two live reads, an 8 MiB payload, 64 KiB chunks and one
original 12-second budget (or an earlier caller deadline). These are this reader's
profile limits, not a universal maximum dataset or model size. Observed policy expiry
may shorten the original deadline and cannot renew it. Exceeded bounds refuse without
truncation or a partial result.

`source.lease(request)` must perform zero I/O and return a fresh operational lease before
any delayed resolution. `lease.resolve()` returns exactly bound immutable bytes or a
held regular-file descriptor. The lease owns all original resolver resources, including
resources acquired after a delayed resolver resumes, until its idempotent `aclose()`
actually finishes. The reader does not open a path or fetch a URL. Its duplicated FD is
owned separately; `pread` preserves the original descriptor's offset. Exact size,
regular-file type, device/inode/size/timestamps, EOF and full SHA-256 are verified. A
mutable file that changes during the read refuses; no immutable-filesystem claim is made.

Policy stays in the caller task. Resolver, FD operations and cleanup use private children
with empty inherited context and no authority object or policy proof. The source is
trusted installed Python code, not a sandbox for malicious extensions. Queued FD work
checks the original deadline and stop state before starting I/O. Cancellation and expiry
discard the result but retain ownership until the actual resolver/thread and cleanup
settle, including repeated cancellation. A running kernel operation can exceed a soft
deadline; coroutine cancellation does not terminate it.

The duplicate closes before the source lease, and the final current-policy check happens
after both complete. Shutdown stops admission, requests cancellation of active callers
and waits for their actual cleanup. Failed source cleanup quarantines its operational
lease for idempotent shutdown retry. An uncertain duplicate-FD close poisons the reader
and is never blindly retried after its descriptor number could have been reused. Cleanup
failure, cancellation and sanitized control exceptions cannot produce a late success.
Fixed errors contain no path, raw diagnostic, source contents or rejected identity.

The focused synthetic tests use installed policy/source doubles, real temporary regular
files, actual descriptor duplication and held real threads. They establish the private
primitive's behavior; they do not establish a deployed authority, installed consumer,
provider access, model quality or successful training/evaluation.
