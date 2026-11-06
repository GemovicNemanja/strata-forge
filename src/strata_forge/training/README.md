# strata_forge.training

Fine-tuning primitives: SFT via TRL `SFTTrainer`; preference tuning (DPO, ORPO, KTO, GRPO); PEFT
(LoRA, QLoRA); chat-template formatting; sequence packing; and progress telemetry.

`loading` supplies explicit Hub credentials, safetensors-only model loads, and no remote code.
The private `_load_guard` checks bounded checkpoint-completeness reports for trained-artifact
consumers. HF scoring consumes that guard; training initialization and merge integration have
their own contracts. An exact head-key decision never authorizes fresh inference weights.

`VerifierConfig` and `render_verifier_input` share versioned inert premise/hypothesis metadata
with evaluation workers. The renderer has explicit task-shape limits and never slices a
hypothesis; it performs no model or dataset I/O.


Primary classification also has an async SDK over resolved objects: `ClassifierConfig`,
`ClassifierExample`, `ClassifierRunner`, `initialize_classifier`, `load_classifier`. Its default
objective is ordinary CE; Brier is explicit. Fixed native encoder/decoder head initialization
and complete safetensors reload are separate operations. Decoder artifacts use Forge's installed
wrapper for exact output bias and answer-position pooling. The SDK performs no Hub or remote
operations and does not enable a worker method. See the primary-classifier section in the
training reference for supported families, refusals and the final-only export policy.

`training.verifier_loss` supplies the independent `verifier_grouped_aux_v1` numerical objective
over caller-supplied logits and resolved targets. It combines grouped entailment-versus-not
CE/Brier with null-masked flag and per-row support BCE. It does not create heads, parse dataset
roles or change `ClassifierConfig` defaults. See [the supplied-logit contract](../../../docs/modules/training.md#supplied-logit-verifier-loss).
