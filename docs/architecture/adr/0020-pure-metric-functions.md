# ADR 0020: Pure metric functions are synchronous

**Status:** Accepted

## Context

Training and evaluation need identical definitions for classification metrics. A pure numerical
leaf must be usable from synchronous training callbacks and asynchronous evaluation workers
without either importing the other. Making a deterministic calculation a coroutine provides
no scheduling benefit and encourages event-loop adapters inside training code.

## Decision

`strata_forge.metrics` contains synchronous deterministic calculations and frozen value models.
It may import only `strata_forge.core`, its own modules, and ordinary dependencies. Numerical
extras are imported lazily. `strata_forge.evals.classification` re-exports these functions.

This is a narrow exception to ADR 0002 for this numerical leaf and its direct re-exports.
Execution, model calls, artifact I/O, and orchestration remain asynchronous. Callers run
expensive numerical jobs on a worker or off the event loop; the synchronous interface does
not imply that large workloads are safe to execute in a request handler.

Definitions, eligible populations, omissions, and probability provenance are explicit. Metric
undefinedness is represented by a null value with a reason, rather than NaN or a fabricated zero.

## Consequences

Training can reuse the same definitions without importing evaluation. The leaf has no hidden
network, file, random, or model state. This does not approve other synchronous execution APIs,
automatic calibration fitting, or statistical definitions that have not been specified.
