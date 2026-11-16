# Compute-target workers

Workers compose Forge primitives from inert JSON specifications. Inference serves batch chat
or computes exact Hugging Face classifier/label-token scores. Fine-tuning dispatches declared
methods and dataset shapes. Evaluation scores pinned prediction tables or native classifiers,
reports each declared dataset population separately, and writes Parquet plus versioned JSON.

Primary classifier execution is a separate trusted-Python API in `classifier_runner`, with a
strict `ClassifierTrainSpec`, required training-purpose authority and streamed byte-bounded
rows. It reuses the resolved-object SDK and safe installed artifact reload. No method registry,
CLI, Task-secret transport, server dispatch, publication or resume is enabled. Prototype row
budgets are 2 MiB per item/64 MiB cumulative and refuse instead of returning a prefix.

`strata_forge.pipelines.evaluation_spec.EvalSpec` is the authoritative evaluation schema.
`strata_forge.pipelines.eval_runner.run_evaluation` requires a trusted, pin-bound read policy;
the standalone command refuses access without one. A caller's `sealed=false` is never a grant.
Policy implementations belong to installed trusted code, outside the serialized spec. The fixed
entrypoint consumes a run/spec-bound `STRATA_EVAL_AUTHORITY` frame only from the private Task
secrets file. It contacts the fixed server policy endpoints without cookies or redirects.
Transient HTTP 503 and transport timeouts permit at most three identical requests within one
30-second budget. Denials and invalid replies refuse immediately; cancellation propagates.

Publication requires separate destination authority: fresh private/custody/canonical readback,
atomic create-only handling or an existing expected parent, an expected-parent commit, and an
accepted actual-commit receipt. Without it a non-null destination refuses. Interrupted attempts
retain local known/unknown mutation evidence; cancellation is never reported as success.
Server policy installation and consumer review are required before enabling launches.

All model loads disable remote code and require safetensors. Classification requires complete
trained weights, explicit label semantics and rendering. Evaluation preserves invalid outputs
as wrong, exposes missing-probability coverage, and never pools different label spaces. Raw
inputs, rendered prompts, reasoning and response text are excluded from evaluation artifacts.

The evaluation execution profile is `strata-forge[finetuning,storage]`. It deliberately includes
the existing full HF/forward stack; the metric arithmetic itself requires no NumPy. The installed
CLI exposes `eval schema`, `eval validate-spec`, `eval classify` and `eval score-predictions`.
The launch commands use the same worker and deny without a trusted policy adapter.

Reference: https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/pipelines.md

The population-v2 specification models wrap the existing task schemas with explicit drop
and suite/tier accounting. Presence-preserving serialization keeps their full-envelope
identity distinct from v1. Trusted Python execution is available through `inference_runner.execute_population_scoring`
and `eval_runner.run_population_evaluation`, with required outer-digest-bound read policy and
private local receipts. Installed CLI/registered entrypoints do not accept these envelopes;
see the population specification boundary in the module reference.

`generation_spec` and `_generation_samples` define a separate exact-pin generation contract,
strict source membership and lazy stable1–16 sample/seed planning. They perform no native
execution or registration. Existing worker/spec defaults remain separate; see
[the generation contract](../../../docs/modules/pipelines.md#separate-generation-contract).

The local `paired_registered_cluster_uncertainty` SDK requires explicit trusted source/policy
objects, exact paired metric populations and current typed unsealed decisions. It installs no
issuer, loader, worker or remote wire. Fixed cluster percentiles retain declared outside-M
population counts and refuse partial probabilities, unknown custody and incomplete/canceled work.
