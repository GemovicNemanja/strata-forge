# Registered local scalar-temperature fitting

## Context

A positive temperature transformation is not a fit, and a readable named split is not fit
eligibility. Calibration must not infer a held-out split's role from its name or silently lose
rows. Floating-point zero derivatives can arise through rounding or underflow, so they cannot
establish statistical identification. A local numerical primitive also cannot authenticate an
external dataset, model or registry.

## Decision

Expose a separate pure bounded scalar-logit fitter and an async local registered-source fit
SDK. The numerical leaf imports only core/metrics and ordinary dependencies. It uses owned
binary64 buffers, stable logsumexp and an explicit positive bounded inverse-temperature
search. Its synchronous stepping is within ADR0020's numerical exception. Execution, trusted
source/policy calls and expensive numerical passes stay async/off-loop.

The local schema admits only implemented temperature fitting with raw exact full-class or
restricted-label logits and identity projection. It accepts no paths, executable imports,
callbacks, model factories, credentials, output destinations or uncertainty placeholder mode.
Sources and policies are explicit trusted Python adapters, not serialized authority. Missing
policy refuses. Every exact typed ref and the registered complete named split must be current,
known and unsealed. Fitting ho_dev, known aliases or test/held-out/gate roles refuses before
iteration. Monitoring/selection reuse is retained rather than inferred independent.

The SDK consumes exactly the expected unique-ID population under a fixed normalized row
codec and digest. It copies inputs before math, enforces byte/row/work bounds and performs no
model forward, artifact read/write, hidden prefix or probability inversion. Positive current
policy is checked before iteration, at bounded read/compute/point chunks and after final
metadata/version work before complete return. No default registry, credential or transport is
installed by these interfaces.

Minimize unweighted mean raw-logit NLL in beta=1/T with explicit bounds containing T=1,
three endpoint/unit evaluations and at most 64 bisection steps. Record chosen candidate,
derivative residual, final search bracket, underflow count and numerical search status.
Only actually equal class logits on every row receive flat_unidentified. A rounded zero
receives numerical_stationary, with identification explicitly not claimed. Candidate ties
prefer T closest to 1 then smaller T. Numerical limits and bounds are implementation choices,
not scientific thresholds or accuracy/quality guarantees.

Reuse temperature_scale and point NLL/Brier definitions. Receipts distinguish the unclipped
fitting objective from clipped point NLL with its epsilon/count. The existing temperature
component stays a draft component; its content digest is not a published artifact ref or a
verified application grant. Declared model/tokenizer/head/rendering identities and input kind
remain bound in the enclosing immutable receipt.

Shield and drain owned reads, policy and numerical work under repeated cancellation. Close
the stream; cleanup faults cannot turn cancellation into a returned fit. A single operation
can overrun the soft elapsed budget; no hard thread termination is promised. Failed,
revoked, invalid or late operations return no partial temperature/receipt. Errors are fixed,
content-free and raised outside native exception handlers.

## Consequences

Pure math is usable without a registry and makes no named-source eligibility claim. Complete
registered fitting lazily requires the existing scoring extra. Imports/schema work without
NumPy/ML libraries. Software versions are allowlisted; broad environment/hostname capture
and raw data/diagnostic logging are absent. The caller owns source adapters and existing data
materialization; this SDK cannot authenticate an arbitrary Python object's issuer.

Raw-logit capture, installed fit eligibility/protection resolution, verified component
application, bootstrap, thresholds, ECE, group temperatures, artifact/worker/server/app/tool
integration and scientific execution remain separate reviewed capabilities. Existing
calibration, scoring, evaluation/population schemas, worker/RPC purposes and defaults are
unchanged.
