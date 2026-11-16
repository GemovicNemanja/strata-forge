# ADR 0025: verifier loss consumes explicit supplied logits and masks

## Context

The accepted primary classifier SDK supports ordinary CE and explicit CE+Brier. Grouped
entailment labels and optional flag/support supervision require a different, precisely
defined objective; changing a generic default or guessing auxiliary-head geometry would
conflate numerical availability with model/worker support.

Formula provenance is the content-free coordinator inbox request CC-20261106-01 at
`strata-app/plans/strata-rework/inbox/claude-code/20261106-0900-loss-annotation-and-budget-definitions.md`.
It fixes grouped 2-way labels, null masks and separate reduction denominators. Its fabricated
examples contain no research rows. The stable public preset ID avoids research milestone names.

## Decision

Provide `training.verifier_loss.research_verifier_loss` with preset ID
`verifier_grouped_aux_v1`, strict frozen resolved targets and caller-supplied primary/flag/
ragged premise logits. The numerical leaf is synchronous and differentiable, with Torch
imported only when called; it performs no model, data, storage or network operation.

Require explicit E/C/U class order. Three-way rows use soft CE and sum-form Brier; two-way
rows group C+U using stable log-sum-exp and two-component sum-form Brier. The complete-batch
primary mean has coefficient 0.1 for Brier. Flag BCE averages non-null flag rows. Support BCE
first averages premises within each eligible row and then eligible rows. Both auxiliary
coefficients are 1.0. Null does not become false or an empty set. A non-null empty support
set is valid over nonempty premises; an eligible empty premise list refuses. IDs are unique
within each row and never interpreted as token positions.

All-null auxiliaries yield graph-compatible zero without division by zero or summing large
logits before multiplying by zero. Return differentiable terms and actual row denominators.
No weighting, smoothing, calibration, free coefficient, shape repair or silent target
normalization is accepted. The fixed-error function boundary revalidates supplied target
instances and does not expose nested native/Pydantic messages.

## Consequences

Fabricated CPU tensors and numerical gradient checks establish the formulas and masks
without creating models, tokenizers, datasets or training jobs. Generic classifier source,
its older explicit CE+Brier objective and worker/EvalSpec behavior remain unchanged.

This does not implement auxiliary-head geometry, marker/PID-token alignment, hypothesis
conditioning, model initialization/retention/loading, worker role schema, training-purpose
authority, process supervision, larger scientific populations or distributed accumulation.
The caller must supply one complete intended batch. Research/model acceptance remains
separate from this reusable numerical capability.
