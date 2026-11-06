# Shared core utilities

`strata_forge.core` contains the upstream exception hierarchy, retry, structured logging,
budgets, reproducibility helpers, IDs, redaction and common types. It imports no other Forge
module and performs no network I/O. Public exports live in `strata_forge.core.__init__`.

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
