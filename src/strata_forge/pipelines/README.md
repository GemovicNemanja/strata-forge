# Compute-target workers

Workers compose Forge primitives from inert JSON specifications. Inference serves batch chat
or computes exact Hugging Face classifier/label-token scores. Fine-tuning dispatches declared
methods and dataset shapes. Evaluation scores pinned prediction tables or native classifiers,
reports each declared dataset population separately, and writes Parquet plus versioned JSON.

`strata_forge.pipelines.evaluation_spec.EvalSpec` is the authoritative evaluation schema.
`strata_forge.pipelines.eval_runner.run_evaluation` requires a trusted, pin-bound read policy;
the standalone command refuses access without one. A caller's `sealed=false` is never a grant.
Policy implementations belong to installed trusted code, outside the serialized spec.

All model loads disable remote code and require safetensors. Classification requires complete
trained weights, explicit label semantics and rendering. Evaluation preserves invalid outputs
as wrong, exposes missing-probability coverage, and never pools different label spaces. Raw
inputs, rendered prompts, reasoning and response text are excluded from evaluation artifacts.

The evaluation execution profile is `strata-forge[finetuning,storage]`. It deliberately includes
the existing full HF/forward stack; the metric arithmetic itself requires no NumPy. The installed
CLI exposes `eval schema`, `eval validate-spec`, `eval classify` and `eval score-predictions`.
The launch commands use the same worker and deny without a trusted policy adapter.

Reference: https://github.com/GemovicNemanja/strata-forge/blob/main/docs/modules/pipelines.md
