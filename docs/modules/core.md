# Shared core utilities

`strata_forge.core` contains the upstream exception hierarchy, retry, structured logging,
budgets, reproducibility helpers, IDs, redaction and common types. It imports no other Forge
module and performs no network I/O. Public exports live in `strata_forge.core.__init__`.

## Runtime observations

The internal `runtime_environment` leaf captures the actual Python implementation,
version, platform and pointer width, plus version metadata for nineteen fixed
distributions. It reads distribution metadata without importing the packages. A
record is `present`, `absent` or `unavailable`: absence means metadata was not found;
presence proves neither importability nor usable GPU/runtime support. Invalid
version metadata and lookup failures have separate fixed reasons, with no raw
exception detail. Control exceptions remain control exceptions.

The immutable `distribution_metadata_v1` profile explicitly leaves source revision,
image, lock, hardware and importability unobserved and execution unassessed. Capture
time is UTC with six fractional digits. Canonical bytes/digest revalidate copied
nested values, enforce the complete ordered distribution profile and cap the output
at 16 KiB. The digest includes observation time and identifies those bytes; it is
not a worker image, source attestation or eligibility verdict. The retained schema
describes structure; ordering, state/version consistency and fixed profile membership
also require the runtime semantic validators.

No package/path selector, working directory, environment/credential value, source URL,
GPU probe or process execution enters this profile. Capture uses trusted installed
metadata finders synchronously; it does not impose a hard time or filesystem-read
budget on an operator's metadata implementation. A future worker adapter must own
that execution boundary. There is no initializer export, runner or transport consumer.
The legacy `env_snapshot()` result and existing capability declarations are unchanged.

## Secret redaction

`Redactor.redact(text)` and `Redactor.stream()` share bounded pattern/value matching. Register
delivered secrets for encoded-value coverage; do not maintain another scrubber in a consumer.
Default credential patterns include Hugging Face, provider/GitHub tokens, private-key blocks
and `strata_worker_` followed by exactly 43 base64url characters. Worker tokens remain scrubbed
when split across stream chunks. Patterns have a declared maximum span, no end-of-input test
and no lookbehind longer than one character, preserving streamed/one-shot equivalence.

`RedactingStream.feed`, `flush` and `gap` retain enough undecided input to cover split secrets
without retaining an unbounded stream. Diagnostic rendering must scrub before display caps can
split a credential. Neither a redactor repr nor its errors may include registered secret values.
See [pipeline authority](pipelines.md#private-run-authority-and-publication) for private Task
credential delivery and the distinction between authentication and custody authorization.
